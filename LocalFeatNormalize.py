import torch
import torch.nn as nn


class LocalFeatNorm(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, xyz, xyz_grouped, feat_grouped):
        """
        :param xyz: [B, topK, 3]
        :param xyz_grouped: [B, topK, nsample, 3]
        :param feat_grouped: [B, topK, nsample, C]
        :return: [B, topK, nsample, C+3]
        """
        B, K, _ = xyz.shape
        _, _, nsample, C = feat_grouped.shape

        local_xyz = xyz_grouped - xyz.unsqueeze(-2).repeat(1,1,nsample,1) #[B, topK, nsample, 3]

        pdist = nn.PairwiseDistance(p=2, keepdim=True)
        dist = pdist(torch.flatten(xyz.unsqueeze(-2).repeat(1,1,nsample,1), start_dim=0, end_dim=2),
                     torch.flatten(xyz_grouped, start_dim=0, end_dim=2)).reshape(B, K, nsample, 1) #[B, topK, nsample, 1]
        dist_ratio = (dist / torch.sum(dist, dim=2, keepdim=True)).repeat(1,1,1,C)
        local_feat = feat_grouped * dist_ratio

        return torch.cat((local_xyz, local_feat),dim=-1)





