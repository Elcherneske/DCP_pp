import torch.nn as nn
import torch
from KNN import KNN
from Utils import *

class Loss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, src_points, R_pred, t_pred, R_true, t_true, K_matrix, ratios):
        '''
        Combine L1 loss function with
        @param  src_points: BxNx3 source points
                y_pred: BxNx3 transformed source points
                R_true: Bx3x3 ground truth rotation
                t_true: Bx1x3 ground truth translation
                K_matrix: BxKxk
                alpha: loss balancing factor
                beta: loss balancing factor
        @return
                loss:
                R: Bx3x3 calculated rotation matrix
                t: BxNx3 calcualted translation
        '''
        B, K, _ = K_matrix.shape
        device = src_points.device
        alpha, beta, eta, gamma, kapa = ratios
        loss_fc = nn.MSELoss(reduction="mean")
        cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        #point_loss
        src_points = src_points.float()
        y_pred = (torch.matmul(src_points.float(), R_pred.float()).float() + t_pred).squeeze(0)
        y_true = (torch.matmul(src_points.float(), R_true.float()).float() + t_true).squeeze(0)
        point_loss = loss_fc(y_true, y_pred)

        #R_loss
        fro_norm = torch.norm(R_true - R_pred, "fro")
        MRAE = torch.acos((torch.trace(torch.matmul(torch.inverse(R_pred), R_true).squeeze(0)) - 1)/2)
        R_loss = fro_norm

        #t_loss
        MRTE = loss_fc(t_pred, t_true)
        t_loss = MRTE


        #matrix_cross_entropy
        topk_indices = torch.max(K_matrix, dim=-1).indices.squeeze(0) #[B, K]
        K_matrix_loss = cross_entropy(K_matrix.squeeze(0), topk_indices.squeeze(0))

        #distribute
        distribute_loss = loss_fc(torch.ones(B, K).to(device), torch.sum(K_matrix, dim=1))

        loss = alpha * point_loss + beta*R_loss + eta * t_loss + gamma * K_matrix_loss + kapa * distribute_loss

        my_dict = {'point_loss': point_loss, 'fro_norm': fro_norm, 'MRAE': MRAE, 'MRTE': MRTE, 'matrix_loss': K_matrix_loss, 'distribute_loss': distribute_loss}


        return loss, my_dict




if __name__ ==  "__main__":

    # 定义矩阵的大小
    N = 5

    # 创建一个N*N的零矩阵
    matrix = torch.zeros(1, N, N)

    # 定义索引
    indices = torch.randint(0, N, (1, N, 1))

    print(indices.squeeze())

    # 将对应的索引位置设置为1
    matrix[:, torch.arange(N), indices.squeeze()] = 1

    # 打印生成的矩阵
    print(matrix)
