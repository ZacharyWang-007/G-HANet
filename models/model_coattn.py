from collections import OrderedDict
from os.path import join

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *
from models.attention_modules import *
from models.hyper_attention_module import *
from models.model_reconstruction import Reconstruction_Net


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 256, D = 256, dropout = False, n_classes = 1):
        
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A

 

class G_HANet_Surv(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, wsi_size=512, dropout=0.25, num_tokens=6, ratio=0.2):
        super(G_HANet_Surv, self).__init__()
    
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        self.num_tokens = num_tokens
        self.ratio = ratio

        ### Gated Layer
        self.gate = Attn_Net_Gated(L=512)
        
        ### FC Layer over WSI bag
        fc = [nn.Linear(1024, wsi_size), nn.LayerNorm(wsi_size), nn.ReLU()]
        fc.append(nn.Dropout(dropout))
        self.wsi_net = nn.Sequential(*fc)

        ### For Reconstruction 
        self.recon_net = Reconstruction_Net(omic_sizes=omic_sizes, num_tokens=num_tokens)

        ### Classifier
        self.classifier = nn.Linear(num_tokens * 256, n_classes)

        ### Hypergraph layer
        self.hyperconv1 = Hyper_attention(in_ft=wsi_size, out_ft=wsi_size)

        ### MHSA
        self.attention = Self_Attention(query_dim=512, context_dim=512, heads=4, dim_head=128, dropout=dropout)
        self.ffn = FeedForward(dim=512, dropout=dropout)
        self.ln1 = nn.LayerNorm(512)

        self.butter = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        re_omics, matrix, fea_reconst = self.recon_net(x_path=x_path)

        ### for bag transformation
        h_path_bag = self.wsi_net(x_path)

        ### generate node-level weights
        weights = self.gate(h_path_bag).T
        
        ### generate binary matrix
        matrix = matrix.mean(dim=0)
        length = h_path_bag.size(0)

        edge = (matrix >= torch.topk(matrix, dim=1, k=int(length * self.ratio))[0][:, -1].unsqueeze(dim=-1)).float() 

        edge1 = (weights) 
        edge1 = F.softmax(edge1, dim=1)

        edge2 = (matrix) + (1-edge) * (-100000)                #  (matrix_scaled + weights)/2 * edge
        edge2 = F.softmax(edge2, dim=1).detach()

        fea_hypergraph = self.hyperconv1(h_path_bag, (edge1 + edge2) /2)
        
        # fea = torch.cat([fea_hypergraph[None, :, :], fea_reconst], dim=2)

        fea = (fea_hypergraph[None, :, :] + fea_reconst) / 2

        fea = self.attention(fea) + fea
        fea = self.ffn(self.ln1(fea)) + fea
        
        ### feature compression and flatten
        h = self.butter(fea).view(1, -1)

        ### Survival Layer
        logits = self.classifier(h) #.unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return hazards, S, Y_hat, None, re_omics, matrix

