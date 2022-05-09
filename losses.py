import os

from zmq import device
import torch
import numpy as np
import seaborn as sns

from dtw import batch_dropDTW
from torch import log, exp
from torch.nn import functional as F
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

opj = lambda x, y: os.path.join(x, y)

def mil_nce(features_1, features_2, correspondance_mat, eps=1e-8, gamma=1, hard_ratio=1):
    corresp = correspondance_mat.to(torch.float32)
    prod = features_1 @ features_2.T / gamma
    
    prod_exp = exp(prod - prod.max(dim=1, keepdim=True).values)
    nominator = (prod_exp * corresp).sum(dim=1) # this sum needs to maximized?
    denominator = prod_exp.sum(dim=1)
    
    nll = -log(nominator / (denominator + eps)) # minimize this ratio will give spread to the data?

    if hard_ratio < 1:
        n_hard_examples = int(nll.shape[0] * hard_ratio)
        hard_indices = nll.sort().indices[-n_hard_examples:]
        nll = nll[hard_indices]
    
    return nll.mean()

def compute_clust_loss(samples, xz_gamma=30, frame_gamma=10, xz_hard_ratio=0.3, l2_normalize=False, device='cuda'):
    sf, step_len, ff, video_len, dif = samples
    all_pooled_frames = []
    all_step_features = []
    pooled_frames_labels = []
    store_s_l = []
    all_step_labels = []
    frame_labels = [0]
    global_step_id_count = 0
    for idx, sample in enumerate(zip(sf, step_len, ff, video_len, dif)):
        st, s_l, fr, v_l, dis = sample
        
        step_ids = torch.arange(global_step_id_count,
                                    global_step_id_count + s_l)
        global_step_id_count += s_l

        if dis is not None:
            bg_step_id = torch.tensor([99999]).to(step_ids.dtype).to(step_ids.device)
            step_ids = torch.cat([step_ids, bg_step_id])
            # print(dis[None, :].shape)
            st, fr = torch.cat([st[:s_l], dis[None, :]]), fr[:v_l] # appending distractor to step as a frame can also be correlated to be dropped
            # st = torch.cat([st, dis[None, :]])
        else:
            # print(st)
            # return
            st, fr = st[:s_l], fr[:v_l]
            # st, fr = st, fr   ## bugs - can not just ignore after passing through model
            # print('here', fr.shape, fr)
            # return
        
        if l2_normalize:
            st = F.normalize(st, p=2, dim=1)
            fr = F.normalize(fr, p=2, dim=1)
            if dis is not None:
                dis = F.normalize(dis, p=2, dim=1)
            frame_gamma = 0.1
            xz_gamma = 0.3
        
        unique_step_labels, unique_idxs = [
            torch.from_numpy(t) for t in np.unique(step_ids.detach().cpu().numpy(), return_index=True)]
        pooled_frames_labels.append(unique_step_labels)

        sim = st @ fr.T # similarity comparison between steps and frames
        # TODO: check to see if some kind of attention can be learned here
        weights = F.softmax(sim / frame_gamma, dim=1) # this gamma allows expanded attention of steps to all frames -> temperature
        # print(weights.shape)
        attended_st = weights @ fr
        all_pooled_frames.append(attended_st)
        frame_labels.append(s_l)
        store_s_l.append(s_l)
        all_step_features.append(st)
        all_step_labels.append(step_ids)
    all_pooled_frames = torch.cat(all_pooled_frames, dim=0) ## how much the steps are being paid attention to by frames
    all_step_features = torch.cat(all_step_features, dim=0)
    pooled_frames_labels = torch.cat(pooled_frames_labels, dim=0)
    all_step_labels = torch.cat(all_step_labels, dim=0)

    assert pooled_frames_labels.shape[0] == all_pooled_frames.shape[0], "Shape mismatch occured"

    unique_labels, unique_idxs = [
        torch.from_numpy(t) for t in np.unique(all_step_labels.detach().cpu().numpy(), return_index=True)]
    unique_step_features = all_step_features[unique_idxs]

    # print(global_step_id_count, pooled_frames_labels.size(), pooled_frames_labels)
    N_steps = all_pooled_frames.shape[0]
    # frame_labels = np.cumsum(frame_labels)
    xz_label_mat = torch.zeros([N_steps, unique_labels.shape[0]]).to(all_pooled_frames.device)

    for i in range(all_pooled_frames.shape[0]):
        for j in range(unique_labels.shape[0]):
            xz_label_mat[i, j] = pooled_frames_labels[i] == unique_labels[j]
    # print(xz_label_mat)

    # xz_label_mat = torch.from_numpy(np.diag([1.] * N_steps)).float() ##########

    # for i in range(1, len(frame_labels)):
    #     xz_label_mat[frame_labels[i-1]:frame_labels[i], frame_labels[i-1]:frame_labels[i]] = 1.
    #     assert store_s_l[i-1] == xz_label_mat[frame_labels[i-1]:frame_labels[i], frame_labels[i-1]:frame_labels[i]].shape[0], 'sizes are not matching'
    xz_label_mat = xz_label_mat.to(device)
    
    xz_loss = mil_nce(all_pooled_frames, unique_step_features, xz_label_mat, gamma=xz_gamma, hard_ratio=.3)
    return xz_loss

def compute_all_costs(sample, l2_normalize, gamma_xz, drop_cost_type, keep_percentile):
    
    
    sf, step_len, ff, video_len, dis = sample
    sf, ff = sf[:step_len], ff[:video_len]
    
    labels = torch.arange(0, step_len)

    if l2_normalize:
        sf = F.normalize(sf, p=2, dim=1)
        ff = F.normalize(ff, p=2, dim=1)
        gamma_xz = 0.1
    sim = sf @ ff.T # getting similarity costs


    unique_labels, unique_index, unique_inverse_index = np.unique(
        labels.detach().cpu().numpy(), return_index=True, return_inverse=True)
    unique_sim = sim[unique_index]
    if not torch.is_tensor(unique_sim): 
        unique_sim = torch.from_numpy(unique_sim)
    
    if drop_cost_type == 'logits':
        
        k = max([1, int(torch.numel(unique_sim) * keep_percentile)])
        baseline_logit = torch.topk(unique_sim.reshape(-1), k).values[-1].detach()
        baseline_logits = baseline_logit.repeat([1, unique_sim.shape[1]])
        sims_ext = torch.cat([unique_sim, baseline_logits], dim=0)
    else:
        if l2_normalize:
            dis = F.normalize(dis, p=2, dim=1)
        distractor_sim = ff @ dis
        # print(type(unique_sim), type(distractor_sim))
        if not torch.is_tensor(distractor_sim):
            distractor_sim = torch.from_numpy(distractor_sim)
        sims_ext = torch.cat([unique_sim, distractor_sim[None, :]], dim=0)

    unique_softmax_sims = F.softmax(sims_ext / gamma_xz, dim=0)
    unique_softmax_sim, drop_probs = unique_softmax_sims[:-1], unique_softmax_sims[-1]
    matching_probs = unique_softmax_sim[unique_inverse_index]
    zx_costs = -torch.log(matching_probs + 1e-5)
    drop_costs = -torch.log(drop_probs + 1e-5)
    return zx_costs, drop_costs

def compute_alignment_loss(samples, drop_cost_type, gamma_xz=10, gamma_min=1, keep_percentile=1, l2_normalize=False, device='cuda'):
    
    gamma_xz = 0.1 if l2_normalize else gamma_xz
    gamma_min = 0.1 if l2_normalize else gamma_min
    sf, step_len, ff, video_len, dif = samples

    zx_costs_list = []
    drop_costs_list = []
    
    for idx, sample in enumerate(zip(sf, step_len, ff, video_len, dif)):
        zx_costs, drop_costs = compute_all_costs(sample, l2_normalize=l2_normalize, gamma_xz=gamma_xz, drop_cost_type=drop_cost_type, keep_percentile=keep_percentile)
        zx_costs_list.append(zx_costs)
        drop_costs_list.append(drop_costs)

    min_costs, _ = batch_dropDTW(zx_costs_list, drop_costs_list, gamma_min=gamma_min, device=device)    

    dtw_losses = [c / len(samples) for c in min_costs]
    return sum(dtw_losses)
    