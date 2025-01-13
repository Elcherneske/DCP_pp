import torch
import torch.nn as nn

class KNN(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, X, Y):
        """
        :param X: [B, N1, 3]
        :param Y: [B, N2, 3]
        :return: distance:[B, N1, N2]
                indices:[B, N1, k]
        """
        X = X.float()
        Y = Y.float()
        distance = torch.cdist(X, Y)
        k_distance, indices = torch.topk(distance, self.K)

        return k_distance, indices



if __name__ == "__main__":
    X = torch.rand(5, 3).unsqueeze(0)
    Y = torch.rand(7, 3).unsqueeze(0)

    knn = KNN(K = 2)

    result = knn(X, Y)

    print(result)
