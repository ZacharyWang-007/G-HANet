import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Hyper_attention(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(Hyper_attention, self).__init__()

        self.linear1 = nn.Linear(in_ft, out_ft)
        self.linear2 = nn.Linear(out_ft, out_ft)

        self.ln1 = nn.LayerNorm(out_ft)
        self.ln2 = nn.LayerNorm(out_ft)
        self.relu = nn.ReLU(inplace=True)


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor, scaled_weights=None):
        # x_ori = x

        x_ = G.matmul(x)

        x = self.linear1(x_)
        x = self.ln1(x)
        x = self.relu(x)

        x = self.linear2(x)
        # x = x.matmul(self.weight2) + self.bias2

        x = self.ln2(x)
        x = self.relu(x)
        return x + x_

