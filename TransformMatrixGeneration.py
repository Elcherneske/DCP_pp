import numpy
import torch
import torch.nn as nn


class CNNLayer(nn.Module):
    def __init__(self, channels = [1, 8, 16, 1]):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        last_channel = channels[0]
        assert last_channel == 1
        for out_channel in channels[1:]:
            self.convs.append(nn.Conv2d(in_channels=last_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0))
            self.bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel


    def forward(self, K_matrix):
        B, K, K = K_matrix.shape
        K_matrix = K_matrix.unsqueeze(1)
        for i, conv in enumerate(self.convs):
            bn = self.bns[i]
            K_matrix = bn(conv(K_matrix.float()))

        K_matrix = K_matrix.reshape(B, K, K)

        return K_matrix


class TransformMatrixGenration(nn.Module):
    def __init__(self, K, hidden_dim, CNN_channels=None):
        super().__init__()
        if CNN_channels is None:
            CNN_channels = [1, 8, 16, 1]
        self.CNN_layer = CNNLayer(channels=CNN_channels)
        self.fc = nn.Sequential(
            nn.Linear(K, hidden_dim),
            nn.BatchNorm1d(K),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(K),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(K),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )


    def forward(self, src, tar):
        """
        :param src: [B, topK, C]
        :param tar: [B, topK, C]
        :return:[B, topK, topK]
        """
        src = src.float()
        tar = tar.float()
        _, K, _ = src.shape

        dist_matrix = -1 * torch.cdist(src, tar, p=2)

        # print(torch.max(dist_matrix, dim=-1).values - torch.min(dist_matrix, dim=-1).values)
        ratio = self.fc(dist_matrix)
        # print(ratio)
        matrix = ratio * self.CNN_layer(dist_matrix)
        # matrix = dist_matrix

        matrix = torch.softmax(matrix, dim=-1)

        return matrix



if __name__ == "__main__":
    a = [
        [[1,2,3],
         [4,5,6]]
    ]
    b = [
        [[3,2,1],
         [7,1,2]]
    ]
    a = torch.from_numpy(numpy.array(a))
    b = torch.from_numpy(numpy.array(b))
    B, topk, C = a.shape
    print((B, topk, C))
    model = TransformMatrixGenration(feat_dim=3)


    print(model(a, b))



