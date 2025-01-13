import torch.nn as nn
import torch
import numpy as np



class LinearPointWeightLayer(nn.Module):
    def __init__(self, K = 256, in_dim = 64, hidden_dim=64):
        super().__init__()
        self.K = K

        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )


    def forward(self, x):
        B, _, _ = x.shape
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = x.squeeze(-1)

        _, topk_indices = torch.topk(x, self.K, dim=-1)
        topk_indices.reshape(B, self.K)

        return topk_indices

class SrcPointWeightLayer(nn.Module):
    def __init__(self, K = 256):
        super().__init__()
        self.K = K



    def forward(self, src_normal):
        """
        :param src_normal: [B, npoint, 64]
        :return:
        """
        B, N, C = src_normal.shape
        device = src_normal.device
        dist = torch.min(torch.cdist(src_normal, src_normal, p=2) + 1e10 * torch.eye(N).unsqueeze(0).to(device), dim=-1).values #[B, N, 1] #寻找与其他点差距最大的点

        _, topk_indices = torch.topk(dist, self.K, dim=-1)
        topk_indices = topk_indices.reshape(B, self.K)

        return topk_indices


class TarPointWeightLayer(nn.Module):
    def __init__(self, K=256):
        super().__init__()
        self.K = K


    def forward(self, src_key_normal, tar_normal):
        """
        :param src_key_normal: [B, K, 64]
               tar_normal: [B, npoint, 64]
        :return:
        """
        B, K, C = src_key_normal.shape
        _, N, _ = tar_normal.shape
        device = src_key_normal.device

        dist_matrix = torch.cdist(src_key_normal, tar_normal, p=2)  # [B, K, N]

        tar_min_dist = torch.min(dist_matrix, dim=1).values.reshape(B, N)

        topk_values, topk_indices = torch.topk(tar_min_dist, self.K, dim=-1, largest=False)
        topk_indices = topk_indices.reshape(B, self.K)

        return topk_indices



if __name__ == '__main__':
    x = torch.from_numpy(np.array([1,2,3,4,5,6,7]))

    result = torch.topk(x, 2)


    print(result)



