import torch
import numpy as np

def traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)

def dropdtw(match_costs, drop_costs):
    K, N = match_costs.shape

    # grids initialization
    d_plus = torch.zeros(K+1, N+1)
    d_plus[:, 0] = float('inf')
    d_plus[0, :] = float('inf')

    d_minus = torch.zeros(K+1, N+1)
    d_minus[:, 0] = float('inf')
    d_minus[0, :] = [torch.cumsum(drop_costs[:i]) for i in range(len(drop_costs))]

    d = torch.zeros(K+1, N+1)
    d[:, 0] = d_minus[:, 0]
    d[0, :] = d_minus[0, :]

    for i in range(1, K+1):
        for j in range(1, N+1):
            d_plus = match_costs[i-1, j-1] + min(d[i-1, j-1], d[i,j-1], d_plus[i-1, j])
            d_minus = drop_costs[j-1] + d[i, j-1]
            d[i, j] = min(d_plus[i, j], d_minus[i, j])
    
    
    



    