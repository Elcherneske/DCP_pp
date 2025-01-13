import torch.nn as nn
import torch
#import KNN

class Loss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, src, tar, src_key, tar_key_pred, R_true, t_true, alpha):
        '''
        Combine L1 loss function with
        @param  x: BxNx3 source points
                y_pred: BxNx3 transformed source points
                R_true: Bx3x3 ground truth rotation
                t_true: BxNx3 ground truth translation
                alpha:  loss balancing factor
        @return
                loss:
                R: Bx3x3 calculated rotation matrix
                t: BxNx3 calcualted translation
        '''

        loss_fc = nn.L1Loss(reduction="mean")
        inlier_ratio = 0.8

        tar_key_true = torch.matmul(R_true, src_key.permute(0,2,1)).permute(0,2,1) + t_true
        key_point_loss = loss_fc(tar_key_pred.squeeze(0), tar_key_true.squeeze(0))

        dist = torch.sum((tar_key_pred - tar_key_true)**2, dim=-1)
        top_indices = torch.topk(dist, k=int(src_key.shape[1] *inlier_ratio), dim=-1).indices
        src_key_inlier = src_key.gather(dim=1, index=top_indices.unsqueeze(-1).repeat(1,1,3))
        tar_key_inlier = tar_key_pred.gather(dim=1, index=top_indices.unsqueeze(-1).repeat(1,1,3))

        R, t = self.svd_optimize(src_key_inlier, tar_key_inlier)

        tar_pred = torch.matmul(R, src.permute(0,2,1)).permute(0,2,1) + t
        points_loss = loss_fc(tar_pred.squeeze(0), tar.squeeze(0))

        loss = alpha * key_point_loss + (1 - alpha) * points_loss


        # R_loss
        fro_norm = torch.norm(R_true - R, "fro")
        MRAE = torch.acos((torch.trace(torch.matmul(torch.inverse(R), R_true).squeeze(0)) - 1) / 2)

        # t_loss
        MRTE = loss_fc(t, t_true)

        my_dict = {'point_loss': points_loss, 'fro_norm': fro_norm, 'MRAE': MRAE, 'MRTE': MRTE}

        return loss, my_dict, R, t

    def svd_optimize(self, src, tar):
        """
            :param src: [B, K, 3]
            :param tar: [B, K, 3]
            :return:
        """
        B, K, _ = src.shape
        device = src.device

        X = src - torch.mean(src, dim=1, keepdim=True).repeat(1, K, 1)  # [B, K, 3]
        Y = tar - torch.mean(tar, dim=1, keepdim=True).repeat(1, K, 1)  # [B, K, 3]

        H = torch.matmul(X.permute(0, 2, 1).float(), Y.float())

        try:
            u, _, v = torch.svd(H)
        except RuntimeError as e:
            print(e.args)
            print(H)
            u = v = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(device)

        C = torch.eye(u.shape[-1]).unsqueeze(0).repeat(B, 1, 1).to(device)  # B x 3 x 3
        d = torch.sign(torch.det(torch.matmul(v.float(), u.permute(0, 2, 1).float())))
        print(d)
        C[:, -1, -1] = d

        R = torch.matmul(torch.matmul(v.float(), C.float()), u.permute(0, 2, 1).float())

        # solve for translation: BxNx3
        t = (torch.mean(tar, dim=1, keepdim=True) - torch.mean(
            torch.matmul(R.float(), src.permute(0, 2, 1).float()).permute(0, 2, 1), dim=1, keepdim=True))

        return R, t


if __name__ ==  "__main__":
    # 创建一个实对称矩阵
    A = torch.tensor([[1.0, 2.0, 3.0],
                      [2.0, 4.0, 5.0],
                      [3.0, 5.0, 6.0]])

    # 对矩阵进行特征值和特征向量分解
    eigenvalues, eigenvectors = torch.eig(A, eigenvectors=True)

    print("特征值：", eigenvalues)
    print("特征向量：", eigenvectors)