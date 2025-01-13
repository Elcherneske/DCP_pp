import torch
import torch.nn as nn
import numpy as np
from ModelNet40DataSet import *
from LocalFeatNormalize import LocalFeatNorm


class DeepFeatExtractionLayer(nn.Module):
    def __init__(self, use_normal=False, npoints=None, radius=None, nsamples=None, in_dim=None, hidden_dim=None, out_dim=None):
        super().__init__()
        if in_dim is None:
            in_dim=4
        if hidden_dim is None:
            hidden_dim=128
        if out_dim is None:
            out_dim=64
        self.use_normal = use_normal
        self.sa1 = SetAbstractionLayer(npoint=npoints[0], radius=radius[0], nsample=nsamples[0], in_dim=in_dim,
                                       hidden_dim=hidden_dim, out_dim=out_dim)
        self.sa2 = SetAbstractionLayer(npoint=npoints[1], radius=radius[1], nsample=nsamples[1], in_dim=out_dim + 3,
                                       hidden_dim=hidden_dim, out_dim=out_dim)
        self.sa3 = SetAbstractionLayer(npoint=npoints[2], radius=radius[2], nsample=nsamples[2], in_dim=out_dim + 3,
                                       hidden_dim=hidden_dim, out_dim=out_dim)


    def forward(self, points):
        """
        param: points [B, N, 3]
        return: output_xyz [B, npoint, 3]
                output_normal [B, npoint, 32]
        """
        if self.use_normal:
            normal = points[:, :, 3:]
            xyz = points[:, :, :3]
        else:
            normal = None
            xyz = points
        output_xyz, output_normal = self.sa1(xyz, normal)
        output_xyz, output_normal = self.sa2(output_xyz, output_normal)
        output_xyz, output_normal = self.sa3(output_xyz, output_normal)

        return output_xyz, output_normal





class SetAbstractionLayer(nn.Module):
    def __init__(self, npoint, nsample, radius, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.sampling_layer = SamplingLayer(self.npoint)
        self.grouping_layer = GroupingLayer(nsample=self.nsample, radius=radius)
        self.pointnet_layer = PointNetLayer(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, nsample=nsample, npoint=npoint)
        self.localfeatnorm = LocalFeatNorm()

    def forward(self, xyz, normal):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            normal: input points data, [B, N, D]/None
        Return:
            centroids: sampled points position data, [B, S, 3]
            pointnet_normal: sample points feature data, [B, S, D']
        """
        device = xyz.device
        B, N, _ = xyz.shape

        centroids_idx = self.sampling_layer(xyz) #[B, S]
        centroids = xyz.gather(dim = 1, index=centroids_idx.unsqueeze(-1).repeat(1, 1, 3)) #[B, S, 3]

        group_idx = self.grouping_layer(centroids, xyz) #[B, S, nsample]
        _, npoint, nsample = group_idx.shape
        grouped_xyz = xyz.gather(dim=1, index=group_idx.reshape(B, -1).unsqueeze(-1).repeat(1, 1, 3)).reshape(B, npoint, nsample, 3)

        if normal is None:
            grouped_normal = grouped_xyz - centroids.reshape(B, npoint, 1, 3)
        else:
            _, _, D = normal.shape
            grouped_normal = normal.gather(dim = 1, index=group_idx.reshape(B, -1).unsqueeze(-1).repeat(1, 1, D)).reshape(B, npoint, nsample, D)
            # grouped_xyz_normal = grouped_xyz - centroids.reshape(B, npoint, 1, 3) # [B, npoint, nsample, 3]
            grouped_normal = self.localfeatnorm(centroids, grouped_xyz, grouped_normal)



        pointnet_normal = self.pointnet_layer(grouped_normal).to(device)
        return centroids, pointnet_normal



class SamplingLayer(nn.Module):
    def __init__(self, npoint):
        super().__init__()
        self.npoint = npoint

    def forward(self, all_xyz):
        """
        Input:
            xyz: point cloud data, [B, N, C]
        Return:
            centroids: sampled point cloud index, [B, npoint]
        """
        device = all_xyz.device
        B, N, C = all_xyz.shape
        if N == self.npoint:
            return torch.arange(self.npoint).unsqueeze(0).to(device)

        centroids = torch.zeros(B, self.npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.min(torch.sum((all_xyz - torch.mean(all_xyz, dim=1))**2, dim=-1), dim=-1, keepdim=True).indices #[B, 1]

        for i in range(self.npoint):
            centroids[:, i] = farthest
            centroid = all_xyz.gather(dim = 1, index=farthest.unsqueeze(-1).repeat(1,1,C)) #[B, 1, 3]
            dist = torch.sum((all_xyz - centroid) ** 2, -1) #[B, N]
            mask = dist < distance #[B, N]
            distance[mask] = dist[mask].float() # only the index for mask=true will be replaced
            farthest = torch.max(distance, -1)[1] #indices
        return centroids

class GroupingLayer(nn.Module):
    def __init__(self, radius, nsample):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
    def forward(self, centroids_xyz, all_xyz):
        """
        Input:
            radius: local region radius
            nsample: max sample number in local region
            all_xyz: all points, [B, N, 3]
            centroids_xyz: query points, [B, S, 3]
        Return:
            group_idx: grouped points index, [B, S, nsample]
        """
        device = centroids_xyz.device
        B, N, _ = all_xyz.shape
        _, S, _ = centroids_xyz.shape
        sqrdists = self.square_distance(centroids_xyz, all_xyz) # [B, S, N]
        sqrdists_sort_k, sqrdists_indices = torch.topk(sqrdists, dim=-1, k=self.nsample, largest=False) #[B, S, nsample]
        _, _, nsample = sqrdists_indices.shape
        group_idx = sqrdists_indices.to(device)
        mask = (sqrdists_sort_k > self.radius ** 2) #[B, S, nsample]

        mask[:, :, 0] = False

        group_first = group_idx[:, :, 0].reshape(B, S, 1).repeat([1, 1, nsample])  # [B, S, N]
        group_idx[mask] = group_first[mask]

        return group_idx

    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape

        dist = -2 * (src.float() @ dst.permute(0, 2, 1).float())
        dist += torch.sum(src ** 2, -1).reshape(B, N, 1)
        dist += torch.sum(dst ** 2, -1).reshape(B, 1, M)
        return dist


class PointNetLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, nsample, npoint):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.nsample = nsample
        self.npoint = npoint
        self.out_dim = out_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(nsample * npoint),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(nsample * npoint),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(nsample * npoint),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(nsample * npoint),
            nn.Softplus()
        )
        self.maxpool = nn.MaxPool1d(nsample)

    def forward(self, x):
        """
            Input:
                normal: [B, npoint, nsample, D]
            Return:
                pointnet_normal: [B, npoint, D']
        """
        x = x.float()
        B, K, nsample, C = x.shape
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.fc(x)
        x = x.reshape(B, K, nsample, self.out_dim)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        emb_normal = self.maxpool(x.permute(0, 2, 1)).reshape(B, K, self.out_dim)
        return emb_normal

# class PointNetLayer(nn.Module):
#     def __init__(self, normal_dim, emb_dim, nhead = 6, layer_num = 2, dropout=0.2):
#         super().__init__()
#         self.emb_dim = emb_dim
#         self.normal_linear = nn.Linear(normal_dim, emb_dim)
#         self.encoding = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=emb_dim, dropout=dropout), num_layers=layer_num)
#         self.xyz_linear = nn.Linear(3, emb_dim)
#
#     def forward(self, normal):
#         """
#             Input:
#                 if consider_xyz:
#                     normal: [B, npoint, nsample, 3+D]
#                 else:
#                     normal: [B, npoint, nsample, D]
#             Return:
#                 pointnet_normal: [B, npoint, D']
#         """
#         B, npoint, nsample, _ = normal.shape
#         normal = normal.float()
#         xyz = self.xyz_linear(normal[:,:,:,:3])
#         processed_normal = self.normal_linear(normal[:,:,:,3:])
#         pointnet_normal = processed_normal + xyz
#
#         pointnet_normal = torch.flatten(pointnet_normal, start_dim=0, end_dim=1) #[B x npoint, nsample, D']
#         pointnet_normal = self.encoding(pointnet_normal.permute(1,0,2)).permute(1,0,2).reshape(B, npoint, nsample, self.emb_dim)
#         pointnet_normal = torch.max(pointnet_normal, dim=2)[0]
#
#         # print(pointnet_normal)
#
#         return pointnet_normal



if __name__ == '__main__':
    root = "../brainDataset/"




