import random
import shutil

import torch
from ModelNet40DataSet import ModelNet40DataSet
from ADNIDataset import ADNIDataset
from torch.utils.data import DataLoader
from Loss import Loss
from DeepVCP_POT import DeepVCP_POT
import torch.nn as nn
import numpy as np
import os
from Utils import *


def readFile(dir):
    with open(dir) as file:
        data = np.array(file.readlines())
        loss = float(data[0])
        R_gt = np.loadtxt(data[1:4])
        t_gt = np.loadtxt(data[4:5])
        R_pred = np.loadtxt(data[5:8])
        t_pred = np.loadtxt(data[8:9])
        src_points = np.loadtxt(data[9:])
        src_points = src_points[:int(src_points.shape[0] / 2)]
    return loss, R_gt, t_gt, R_pred, t_pred, src_points


def samplingPoint(src, radius=1):
    B, N, _ = src.shape
    src = src.squeeze(0).detach().numpy().tolist()
    choice = random.randint(0,N)
    center = src[choice]
    result = []
    for point in src:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        dz = point[2] - center[2]
        if dx*dx+dy*dy+dz*dz>radius*radius:
            result.append(point)

    result = torch.tensor(result).unsqueeze(0)


    return result


def main():
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    ratios = (1, 1, 1, 1e-5, 1e-3)  # alpha beta eta gamma
    model = DeepVCP_POT(use_normal=True)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load("epoch_0_model.pt"))
    loss_fn = Loss()

    root = "./points/"

    print(root)

    for cur_path, dirs, files in os.walk(root):
        for file in files:
            loss, R_gt, t_gt, R_pred, t_pred, src_points = readFile(os.path.join(cur_path, file))
            print(file)
            src = torch.from_numpy(src_points).unsqueeze(0)
            src = samplingPoint(src, radius=1)
            R_gt = torch.from_numpy(R_gt).unsqueeze(0)
            t_gt = torch.from_numpy(t_gt).unsqueeze(0)
            target = torch.matmul(R_gt.float(), src[:,:,:3].float().permute(0, 2, 1)).permute(0, 2, 1) + t_gt
            target = torch.cat([target, src[:, :, 3:]], dim=-1)

            src, target, R_gt, t_gt = src.to(device), target.to(device), R_gt.to(device), t_gt.to(device)

            R, t, K_matrix = model(src, target)  # R [B, 3, 3]  t [B, 1, 3]

            loss, loss_detail = loss_fn(src[:, :, :3], R_pred=R, t_pred=t, R_true=R_gt, t_true=t_gt, K_matrix=K_matrix,
                                        ratios=ratios)

            print(
                f"Loss: {loss.item()}, PointLoss: {loss_detail['point_loss'].item()}, fro_norm: {loss_detail['fro_norm'].item()}, MRAE: {loss_detail['MRAE'].item()}"
                f"MRTE: {loss_detail['MRTE'].item()}, matrix_loss: {loss_detail['matrix_loss'].item()}, distribute_loss: {loss_detail['distribute_loss'].item()}")

    # root = "../subADNI/"
    # batch_size = 1
    # test_data = ADNIDataset(root=root, augment=True, split='test', partition=1)
    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    #
    # loss_fn = Loss()
    # model.eval()
    # loss_test = []
    # with torch.no_grad():
    #     for n_batch, (src, target, R_gt, t_gt, src_file) in enumerate(test_loader):
    #         # mini batch
    #         src, target, R_gt, t_gt = src.to(device), target.to(device), R_gt.to(device), t_gt.to(device)
    #
    #         R, t, K_matrix = model(src, target) # R [B, 3, 3]  t [B, 1, 3]
    #
    #         loss, loss_detail = loss_fn(src[:,:,:3], R_pred=R, t_pred=t, R_true=R_gt, t_true=t_gt, K_matrix=K_matrix, ratios=ratios)
    #
    #         loss_test.append([loss.item(), loss_detail['point_loss'].item(), loss_detail['fro_norm'].item(),
    #                            loss_detail['MRAE'].item(), loss_detail['MRTE'].item(),
    #                            loss_detail['matrix_loss'].item(), loss_detail['distribute_loss'].item()])
    #         print(
    #             f"n_batch: {n_batch}, Loss: {loss.item()}, PointLoss: {loss_detail['point_loss'].item()}, fro_norm: {loss_detail['fro_norm'].item()}, MRAE: {loss_detail['MRAE'].item()}"
    #             f"MRTE: {loss_detail['MRTE'].item()}, matrix_loss: {loss_detail['matrix_loss'].item()}, distribute_loss: {loss_detail['distribute_loss'].item()}")
    #
    #         if loss_detail['point_loss'].item() < 3:
    #             shutil.copy(src_file[0], "../subADNI/")


if __name__ == "__main__":
    main()