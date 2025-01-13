import torch
import torch.nn as nn


class RecoverTransformLayer(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, src, tar, K_matrix, ratio=0.6):
        B, K, _ = src.shape
        device = src.device

        max_similar = torch.max(K_matrix, dim=-1, keepdim=False).values # [B, K]
        topk_num = max(int(ratio * K), torch.sum((max_similar > 0.95).squeeze(0), dim=-1).item())
        indices = torch.topk(max_similar, k=topk_num, dim=-1).indices  # [B, K/2]
        batch_index = torch.arange(B, dtype=torch.long).unsqueeze(-1).to(device)
        ratio_matrix = torch.zeros(B, K, K).to(device)
        ratio_matrix[batch_index, indices, :] = 1
        K1_matrix = K_matrix * ratio_matrix


        R1, t1 = self.svd_optimize(src, tar, K1_matrix)

        tar_1 = torch.matmul(R1, src.float().permute(0,2,1)).permute(0,2,1) + t1

        dist = torch.sum(torch.cdist(tar_1, tar, p=2) * K1_matrix, dim=-1)#[B, K]
        topk_num = max(int(ratio * K), torch.sum((dist < 0.5).squeeze(0), dim=-1).item())
        indices = torch.topk(dist, k=topk_num, largest=False, dim=-1).indices #[B, K/2]
        batch_index = torch.arange(B, dtype=torch.long).unsqueeze(-1).to(device)

        ratio_matrix = torch.zeros(B, K, K).to(device)
        ratio_matrix[batch_index, indices, :] = 1

        K2_matrix = K1_matrix * ratio_matrix

        R2, t2 = self.svd_optimize(tar_1, tar, K2_matrix)

        R = R2 @ R1
        t = torch.matmul(R2, t1.permute(0,2,1)).permute(0,2,1) + t2

        return R, t

    def svd_optimize(self, src, tar, K_matrix):
        """
            :param src: [B, K, 3]
            :param tar: [B, K, 3]
            :return:
        """
        B, K, _ = src.shape
        device = src.device

        X = src - torch.mean(src, dim=1, keepdim=True).repeat(1, K, 1)  # [B, K, 3]
        Y = tar - torch.mean(tar, dim=1, keepdim=True).repeat(1, K, 1)  # [B, K, 3]

        U_x = (torch.matmul(torch.matmul(X.permute(0, 2, 1).float(), K_matrix.float()),
                            torch.ones(B, K, 1).to(device).float()) / torch.sum(torch.sum(K_matrix, dim=2), dim=1)
               .unsqueeze(-1).unsqueeze(-1).repeat(1, 3, 1))  # [B, 3, 1]
        U_x = U_x.permute(0, 2, 1).repeat(1, K, 1)  # [B, K, 3]

        U_y = (torch.matmul(torch.matmul(Y.permute(0, 2, 1).float(), K_matrix.permute(0, 2, 1).float())
                            , torch.ones(B, K, 1).to(device).float()) / torch.sum(torch.sum(K_matrix, dim=2), dim=1)
               .unsqueeze(-1).unsqueeze(-1).repeat(1, 3, 1))  # [B, 3, 1]
        U_y = U_y.permute(0, 2, 1).repeat(1, K, 1)  # [B, K, 3]

        X = X - U_x
        Y = Y - U_y

        H = torch.matmul(torch.matmul(X.permute(0, 2, 1).float(), K_matrix.float()), Y.float())

        try:
            u, _, v = torch.svd(H)
        except RuntimeError as e:
            print(e.args)
            print(H)
            u = v = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(device)

        C = torch.eye(u.shape[-1]).unsqueeze(0).repeat(B, 1, 1).to(device)  # B x 3 x 3
        d = torch.sign(torch.det(torch.matmul(v.float(), u.permute(0, 2, 1).float())))
        print(d)
        # d = torch.det(torch.matmul(v.float(), u.permute(0, 2, 1).float()))
        C[:, -1, -1] = d

        R = torch.matmul(torch.matmul(v.float(), C.float()), u.permute(0, 2, 1).float())

        # print(R)

        # solve for translation: BxNx3
        t = U_y - torch.matmul(R.float(), U_x.permute(0, 2, 1).float()).permute(0, 2, 1)
        # print(f"addition_t: {t[:,-1:,:]}")

        t = torch.mean(t, dim=1, keepdim=True) + (torch.mean(tar, dim=1, keepdim=True) - torch.mean(
            torch.matmul(R.float(), src.permute(0, 2, 1).float()).permute(0, 2, 1), dim=1, keepdim=True))  # [B, 1, 3]

        return R, t


if __name__ == "__main__":
    model = RecoverTransformLayer()

    a = torch.randn(1, 5, 3)
    b = torch.randn(1, 5, 3)
    K = torch.randn(1, 5, 5)
    print(model(a, b, K))

