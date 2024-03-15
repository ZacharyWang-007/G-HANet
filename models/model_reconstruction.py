from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *
from models.attention_modules import *


 
###########################
### Reconstruction Network Implementation ###
###########################
class Reconstruction_Net(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600], num_tokens=6, dropout=0.25):
        super(Reconstruction_Net, self).__init__()
        
        self.omic_sizes = omic_sizes
        self.num_tokens = num_tokens

        ### FC Layer over WSI bag
        size = [1024, 512]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(dropout))
        self.wsi_net = nn.Sequential(*fc)

        ### Constructing the recover modules
        sig_reconstruct_net = []

        # add dropout before the prediction
        for input_dim in omic_sizes: 
            sig_reconstruct_net.append(nn.Sequential(nn.Linear(512, 512),
                                                     nn.LayerNorm(512),
                                                     nn.ELU(),
                                                     nn.Dropout(dropout),
                                                     nn.Linear(512, input_dim),
                                                    #  nn.Sigmoid()
                                                     ))    
        self.sig_reconstruct_net = nn.ModuleList(sig_reconstruct_net)

        self.tokens = nn.Parameter(torch.randn((1, self.num_tokens, 512), requires_grad=True))

        self.cross_attention0 = Cross_Attention(query_dim=512, context_dim=512, heads=4, dim_head=128, dropout=dropout)
        self.ffn0 = FeedForward(dim=512, mult=4, dropout=dropout)

        self.cross_attention1 = Cross_Attention(query_dim=512, context_dim=512, heads=4, dim_head=128, dropout=dropout)
        self.ffn1 = FeedForward(dim=512, mult=4, dropout=dropout)

        self.ln_1 = nn.LayerNorm(512)
        self.ln_2 = nn.LayerNorm(512)
        self.ln_3 = nn.LayerNorm(512)
        self.ln_4 = nn.LayerNorm(512)
        self.ln_5 = nn.LayerNorm(512)
        self.ln_6 = nn.LayerNorm(512)


    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        
        h_path_bag = self.wsi_net(x_path).unsqueeze(0) 

        reconstruct, _ = self.cross_attention0(self.ln_1(self.tokens), self.ln_2(h_path_bag)) 
        reconstruct = self.ffn0(self.ln_3(reconstruct)) + reconstruct

        reconstruct, sim = self.cross_attention0(self.ln_4(self.tokens + reconstruct), self.ln_2(h_path_bag)) 
        reconstruct = self.ffn1(self.ln_5(reconstruct)) + reconstruct
    
        reconstruct = self.ln_6(reconstruct)

        reconstruct_ = torch.chunk(reconstruct, dim=1, chunks=self.num_tokens)
        omic = [self.sig_reconstruct_net[i](reconstruct_[i]).squeeze() for i in range(self.num_tokens)]        
        
        return omic, sim, reconstruct

    