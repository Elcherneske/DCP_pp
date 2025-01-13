import numpy as np
import torch.nn as nn
import torch
import numpy


class DeepFeatEmbeddingLayer(nn.Module):
    def __init__(self, nsample=None, in_dim=None, mlp=None):
        super().__init__()
        if mlp is None:
            mlp = [32, 32, 32]
        self.fc = nn.Sequential(
            nn.Linear(in_dim, mlp[0]),
            nn.Linear(mlp[0], mlp[1]),
            nn.Linear(mlp[1], mlp[2])
        )
        self.max_pool = nn.MaxPool1d(kernel_size=nsample)

    def forward(self, x, is_src = True):
        """
        :param x: [B, K, nsample, 35] if src,   [B, K, C, nsample, 35] if not src
        :return: [B, K, 35] if src,   [B, K, C, 35] if not src
        """
        x = x.float()
        x = self.fc(x)
        if is_src:
            B, K, _, D = x.shape
            x = x.permute(0,1,3,2)
            x = torch.flatten(x, start_dim = 1, end_dim = 2)
            x = self.max_pool(x)
            x = x.reshape(B, K, D)
            return x
        else:
            B, K, C, _, D = x.shape
            x = x.permute(0,1,2,4,3)
            x = torch.flatten(x, start_dim=1, end_dim=3)
            x = self.max_pool(x)
            x = x.reshape(B, K, C, D)
            return x

