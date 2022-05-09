import torch
import numpy as np

def compute_normalization_parameters(dataset, feat):
    mean_x, mean_z = torch.zeros(feat), torch.zeros(feat)
    mean_x2, mean_z2 = torch.zeros(feat), torch.zeros(feat)
    x_count, z_count = 0, 0
    
    for idx, s in dataset.iterrows():
        if s['video_feat'].endswith('.npy'):
            vid_feat = torch.from_numpy(np.load(s['video_feat']))
        else:
            vid_feat = torch.load(s['video_feat'])
        mean_x += vid_feat.sum(0)
        mean_x2 += (vid_feat ** 2).sum(0)
        x_count += vid_feat.size(0)
        
        step_feat = torch.load(s['text_feat'])
        mean_z += step_feat.sum(0)
        mean_z2 += (step_feat**2).sum(0)
        z_count += step_feat.size(0)
        
    mean_x = mean_x / x_count
    mean_z = mean_z / z_count
    
    sigma_x = (mean_x2/x_count - mean_x ** 2).sqrt()
    sigma_z = (mean_z2/z_count - mean_z ** 2).sqrt()
    
    return mean_x, sigma_x, mean_z, sigma_z