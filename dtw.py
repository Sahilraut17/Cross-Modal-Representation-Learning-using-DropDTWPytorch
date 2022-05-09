import os
import torch
import numpy as np
from torch.nn import functional as F

opj = lambda x, y: os.path.join(x, y)


class VarTable():
    def __init__(self, dims, device):
        self.dims = dims
        d1, d2, d_rest = dims[0], dims[1], dims[2:]
        
        self.vars = []
        
        # creating the dtw table
        for i in range(d1):
            self.vars.append([])
            for j in range(d2):
                var = torch.zeros(d_rest).to(torch.float).to(device)
                self.vars[i].append(var)
    
    def __getitem__(self, pos):
        i, j = pos
        return self.vars[i][j]

    def __setitem__(self, pos, new_val):
        i, j = pos
        if self.vars[i][j].sum() != 0:
            assert False, 'already assigned'
        else:
            self.vars[i][j] = self.vars[i][j] + new_val
            
    def show(self):
        pass # TODO: needs to be added for visualization

def minProb(inputs, gamma = 1, keepdim = True):
    if inputs[0].shape[0] == 1:
        inputs = torch.cat(inputs)
    else:
        inputs = torch.stack(inputs, dim = 0)
    probs = F.softmax(- inputs / gamma, dim = 0)
    minP = (probs * inputs).sum(dim = 0, keepdim = keepdim)
    return minP


def batch_dropDTW(zx_costs_list, drop_costs_list, gamma_min=1, exclusive=True, contiguous=True, device='cpu'):
    
    inf = 99999999
    min_fn = minProb
    
    # to find max padding need to run drop-dtw in batches 
    B = len(zx_costs_list)
    Ns, Ks = [], []
    
    for i in range(B):
        Ki, Ni = zx_costs_list[i].shape
        Ns.append(Ni)
        Ks.append(Ki)
    
    N, K = max(Ns), max(Ks)
    
    
    padded_cum_drop_costs, padded_drop_costs, padded_zx_costs = [], [], []
    
    for i in range(B):
        zx_costs = zx_costs_list[i]
        drop_costs = drop_costs_list[i]
        cum_drop_costs = torch.cumsum(drop_costs, dim=0)
        
        row_pad = torch.zeros([N - Ns[i]]).to(device)
#         print(row_pad.shape)
#         print(cum_drop_costs.shape)
#         print(torch.cat([cum_drop_costs, row_pad]).shape)
        padded_cum_drop_costs.append(torch.cat([cum_drop_costs, row_pad]))
#         print(len(padded_cum_drop_costs))
        padded_drop_costs.append(torch.cat([drop_costs, row_pad]))
        
        multirow_pad = torch.stack([row_pad + inf] * Ks[i], dim=0) # to add padding to each row
#         print(multirow_pad.shape)

#         print('padded_table', zx_costs.shape)

        padded_table = torch.cat([zx_costs, multirow_pad], dim=1)
#         print('padded_table', padded_table.shape)
        
        rest_pad = torch.zeros([K - Ks[i], N]).to(device) + inf
        padded_table = torch.cat([padded_table, rest_pad], dim=0)

#         print(padded_table.shape)

        padded_zx_costs.append(padded_table)

#         print("####")

    all_zx_costs = torch.stack(padded_zx_costs, dim=-1)
    all_cum_drop_costs = torch.stack(padded_cum_drop_costs, dim=-1)
    all_drop_costs = torch.stack(padded_drop_costs, dim=-1)
    
    
    D = VarTable((K + 1, N + 1, 3, B), device)
    for zi in range(1, K + 1): 
        D[zi, 0] = torch.zeros_like(D[zi, 0]) + inf # init all rows '0th' row with inf 
    for xi in range(1, N + 1):
        D[0, xi] = torch.zeros_like(D[0, xi]) + all_cum_drop_costs[(xi - 1):xi] # init all columns '0th' col with cumulative drops
        
        
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            z_cost_ind, x_cost_ind = zi - 1, xi - 1    # indexind in costs is shifted by 1

            d_diag, d_left = D[zi - 1, xi - 1][0:1], D[zi, xi - 1][0:1]
            dp_left, dp_up = D[zi, xi - 1][2:3], D[zi - 1, xi][2:3]
            
            # positive transition, i.e. matching x_i to z_j
            if contiguous:
                pos_neighbors = [d_diag, dp_left]
            else:
                pos_neighbors = [d_diag, d_left]
            
            if not exclusive:
                pos_neighbors.append(dp_up)

            Dp = min_fn(pos_neighbors, gamma=gamma_min) + all_zx_costs[z_cost_ind, x_cost_ind]

            # negative transition, i.e. dropping xi
            Dm = d_left + all_drop_costs[x_cost_ind]

            # update final solution matrix
            D_final = min_fn([Dm, Dp], gamma=gamma_min)
            
            D[zi, xi] = torch.cat([D_final, Dm, Dp], dim=0)
    
    min_costs = []
    for i in range(B):
        Ni, Ki = Ns[i], Ks[i]
        min_cost_i = D[Ki, Ni][0, i]
        min_costs.append(min_cost_i / Ni)
        
    return min_costs, D


def drop_dtw(zx_costs, drop_costs, exclusive=True, contiguous=True, return_labels=False):
    
    K, N = zx_costs.shape
    
    D = np.zeros([K+1, N+1, 2])

    D[1:, 0, :] = np.inf
    D[0, 1:, :] = np.inf
    D[0, 1:, 1] = np.cumsum(drop_costs)

    P = np.zeros([K+1, N+1, 2, 3], dtype=int)
    for xi in range(1, N+1):
        P[0, xi, 1] = 0, xi - 1, 1

    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # define frequently met neighbors here
            diag_neigh_states = [0, 1] 
            diag_neigh_coords = [(zi - 1, xi - 1) for _ in diag_neigh_states]
            diag_neigh_costs = [D[zi - 1, xi - 1, s] for s in diag_neigh_states]

            left_neigh_states = [0, 1]
            left_neigh_coords = [(zi, xi - 1) for _ in left_neigh_states]
            left_neigh_costs = [D[zi, xi - 1, s] for s in left_neigh_states]

            left_pos_neigh_states = [0] if contiguous else left_neigh_states
            left_pos_neigh_coords = [(zi, xi - 1) for _ in left_pos_neigh_states]
            left_pos_neigh_costs = [D[zi, xi - 1, s] for s in left_pos_neigh_states]

            top_pos_neigh_states = [0]
            top_pos_neigh_coords = [(zi - 1, xi) for _ in left_pos_neigh_states]
            top_pos_neigh_costs = [D[zi - 1, xi, s] for s in left_pos_neigh_states]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            # state 0: matching x to z
            if exclusive:
                neigh_states_pos = diag_neigh_states + left_pos_neigh_states
                neigh_coords_pos = diag_neigh_coords + left_pos_neigh_coords
                neigh_costs_pos = diag_neigh_costs + left_pos_neigh_costs
            else:
                neigh_states_pos = diag_neigh_states + left_pos_neigh_states + top_pos_neigh_states
                neigh_coords_pos = diag_neigh_coords + left_pos_neigh_coords + top_pos_neigh_coords
                neigh_costs_pos = diag_neigh_costs + left_pos_neigh_costs + top_pos_neigh_costs
            costs_pos = np.array(neigh_costs_pos) + zx_costs[z_cost_ind, x_cost_ind] 
            opt_ind_pos = np.argmin(costs_pos)
            P[zi, xi, 0] = *neigh_coords_pos[opt_ind_pos], neigh_states_pos[opt_ind_pos]
            D[zi, xi, 0] = costs_pos[opt_ind_pos]

            # state 1: x is dropped
            costs_neg = np.array(left_neigh_costs) + drop_costs[x_cost_ind] 
            opt_ind_neg = np.argmin(costs_neg)
            P[zi, xi, 1] = *left_neigh_coords[opt_ind_neg], left_neigh_states[opt_ind_neg]
            D[zi, xi, 1] = costs_neg[opt_ind_neg]

    cur_state = D[K, N, :].argmin()
    min_cost = D[K, N, cur_state]
            
    # backtracking the solution
    zi, xi = K, N
    path, labels = [], np.zeros(N)
    x_dropped = [] if cur_state == 1 else [N]
    while not (zi == 0 and xi == 0):
        path.append((zi, xi))
        zi_prev, xi_prev, prev_state = P[zi, xi, cur_state]
        if xi > 0:
            labels[xi - 1] = zi * (cur_state == 0)  # either zi or 0
        if prev_state == 1:
            x_dropped.append(xi_prev)
        zi, xi, cur_state = zi_prev, xi_prev, prev_state
    
    if not return_labels:
        return min_cost, path, x_dropped
    else:
        return labels

