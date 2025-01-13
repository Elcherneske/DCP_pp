import torch.nn as nn
import torch
import numpy as np



class PointWeightLayer(nn.Module):
    def __init__(self, in_dim=None, mlp=None, norm_num=None):
        super().__init__()
        if mlp is None:
            mlp = [16, 8, 1]
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, mlp[0]),
            nn.BatchNorm1d(norm_num),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(mlp[0], mlp[1]),
            nn.BatchNorm1d(norm_num),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(mlp[1], mlp[2]),
            nn.Softplus()
        )


    def forward(self, x, k = 64):
        B, _, _ = x.shape
        x = x.float()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        topk_indices = torch.topk(x, k, dim=1).indices.reshape(B, k)
        #topk_indices = topk_indices.flatten()
        return topk_indices



if __name__ == '__main__':
    x = torch.from_numpy(np.array([1,2,3,4,5,6,7]))

    result = torch.topk(x, 2)


    print(result)



