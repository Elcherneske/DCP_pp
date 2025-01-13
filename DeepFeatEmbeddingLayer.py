import numpy as np
import torch.nn as nn
import torch
import numpy


class DeepFeatEmbeddingLayer(nn.Module):
    def __init__(self, in_dim =64, hidden_dim=64, out_dim=64, nsample=32, K=256):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(nsample * K),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(nsample * K),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(nsample * K),
            nn.Softplus()
        )
        self.maxpool = nn.MaxPool1d(nsample)



    def forward(self, x):
        """
        :param x: [B, K, nsample, in_dim]
        :return: [B, K, out_dim]
        """
        x = x.float()
        B, K, nsample, C = x.shape
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.fc(x)
        x = x.reshape(B, K, nsample, self.out_dim)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        emb_normal = self.maxpool(x.permute(0,2,1)).reshape(B, K, self.out_dim)

        return emb_normal

