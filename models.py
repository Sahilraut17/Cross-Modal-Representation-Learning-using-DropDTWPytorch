import os
import torch

from torch import nn
from utils import compute_normalization_parameters

opj = lambda x, y: os.path.join(x, y)

class NonLinearBlock(nn.Module):
    def __init__(self, in_feat, out_feat, batch_norm):
        super(NonLinearBlock, self).__init__()
        self.fc = nn.Linear(in_feat, out_feat)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.do_batchnorm = batch_norm
        if self.do_batchnorm:
            self.norm_func = nn.BatchNorm1d(out_feat)
            
        
    def forward(self, x):
        x = self.fc(x)
        # TODO: we can switch positions of relu and batch norm to see what happens
        if self.do_batchnorm:
            x = self.norm_fn(x)
        x =  self.relu(x)
        # x = self.dropout(x)
        return x

class NonLinearMapping(nn.Module):
    def __init__(self, feat, num_layers, normalization_params=None, batch_norm=False):
        super(NonLinearMapping, self).__init__()
        self.nonlin_mapping = nn.Sequential(*[NonLinearBlock(feat, feat, batch_norm) for _ in range(num_layers - 1)])
        
        if num_layers > 0:
            self.lin_mapping = nn.Linear(feat, feat)
        else:
            self.lin_mapping = lambda x : torch.zeros_like(x) ## for no layers, do not do anything
        
        self.register_buffer('norm_mean', torch.zeros(feat))
        self.register_buffer('norm_sigma', torch.ones(feat))
    
    def initialize_normalization(self, normalization_params):
        if normalization_params is not None:
            if len(normalization_params) > 0:
                self.norm_mean.data.copy_(normalization_params[0])
            if len(normalization_params) > 1:
                self.norm_sigma.data.copy_(normalization_params[1]) ## bugs - changed from norm_mean to norm_sigma
    
    def forward(self, x):
        # x = (x - self.norm_mean)/ self.norm_sigma ## bugs - MAYBE OFF HERE
        res = self.nonlin_mapping(x)
        # TODO: maybe add a dropout here
        res = self.lin_mapping(res)
        return x + res

class EmbeddingsMapping(nn.Module):
    def __init__(self, feat, video_layers=2, text_layers=2, drop_layers=1, learnable_drop=False, normalization_dataset=None, batch_norm=False):
        super(EmbeddingsMapping, self).__init__()
        self.video_mapping = NonLinearMapping(feat, video_layers, batch_norm)
        self.text_mapping = NonLinearMapping(feat, text_layers, batch_norm)
        
        if learnable_drop:
            self.drop_mapping = NonLinearMapping(feat, drop_layers, batch_norm)
        
        if normalization_dataset is not None:
            # # norm_params = compute_normalization_parameters(normalization_dataset, feat)
            # self.video_mapping.initialize_normalization(norm_params[:2])
            # self.text_mapping.initialize_normalization(norm_params[2:])
            # print(norm_params)
            pass
            # print(self.video_mapping.norm_mean, self.video_mapping.norm)
            #  TODO: check this
            
    def map_video(self, x):
        return self.video_mapping(x)

    def map_text(self, z):
        return self.text_mapping(z)
    
    def compute_distractors(self, v):
        return self.drop_mapping(v)
