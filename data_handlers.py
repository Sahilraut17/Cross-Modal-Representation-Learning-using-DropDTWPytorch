from dataclasses import replace
import os
import torch
import random
import numpy as np
from utils import compute_normalization_parameters
from torch.utils.data import Dataset

opj = lambda x, y: os.path.join(x, y)


class YCDataset(Dataset):
    
    def __init__(self, data_df, text_len=16, video_len=500):
        self.df = data_df
        self.text_len = text_len
        self.video_len = video_len
        self.normalizer()

    def normalizer(self):
        self.mean_x, self.sigma_x, self.mean_z, self.sigma_z = compute_normalization_parameters(self.df, 512)

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        ## read step features
        step_feature = torch.load(self.df.iloc[idx]['text_feat']) ## K x 512
        step_feature = (step_feature - self.mean_z)/ self.sigma_z
        true_step_len = step_feature.size(0) # K
        assert step_feature.size(0) <= self.text_len, 'Increase text_len'
        padding = self.text_len - step_feature.size(0)
        step_feature = torch.vstack([step_feature, torch.zeros(padding, step_feature.size(1))])
            
        ## read video features
        if self.df.iloc[idx]['video_feat'].endswith('.npy'):
            video_feature = torch.from_numpy(np.load(self.df.iloc[idx]['video_feat']))
        else:
            video_feature = torch.load(self.df.iloc[idx]['video_feat']).cpu()
        video_feature = (video_feature - self.mean_x)/ self.sigma_x
        true_vid_len = video_feature.size(0)
        assert video_feature.size(0) <= self.video_len, 'Increase video_len'
        padding = self.video_len - video_feature.size(0)
        video_feature = torch.vstack([video_feature, torch.zeros(padding, video_feature.size(1))])
        
        
        return {'id':self.df.iloc[idx]['key'], 'step_len': true_step_len, 'video_len':true_vid_len, 'step_feature': step_feature.float(), 'video_feature': video_feature.float()}

class SampleBatchIdx():
    def __init__(self, dataset, cls_per_batch, batch_size):

        assert batch_size % cls_per_batch == 0, f'The batch size {batch_size} and classes per batch {cls_per_batch} are not compatible'
        self.dataset = dataset
        self.cls_per_batch = cls_per_batch
        self.batch_size = batch_size
        self.samples_per_cls = self.batch_size // self.cls_per_batch 

        
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def get_random_classes(self):
        return random.sample(list(self.dataset.df['class'].unique()), self.cls_per_batch)
    
    def __iter__(self):
        for _ in range(len(self)):
            sampled_classes = self.get_random_classes()
            batch_idxs = self.dataset.df[self.dataset.df['class'].isin(sampled_classes)].groupby('class').apply(lambda x: x.sample(n=self.samples_per_cls, replace=True)).reset_index(drop = True).sample(frac=1.0, replace=True).index
            yield batch_idxs.tolist()