import torch
import numpy as np
import math


def RotX(theta):
    Rx = np.matrix([[1, 0, 0],
                    [0, math.cos(theta), -math.sin(theta)],
                    [0, math.sin(theta), math.cos(theta)]])
    return Rx


# rotation about y axis
def RotY(theta):
    Ry = np.matrix([[math.cos(theta), 0, math.sin(theta)],
                    [0, 1, 0],
                    [-math.sin(theta), 0, math.cos(theta)]])
    return Ry


# rotation about z axis
def RotZ(theta):
    Rz = np.matrix([[math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1]])
    return Rz


def writePoints(src, tar, R_gt, t_gt, R_pred, t_pred, dir, loss):
    """
    :param src: [B, N, 4]
    :param tar: [B, N, 4]
    :param R_gt: [B, 3, 3]
    :param t_gt: [B, 1, 3]
    :param R_pred: [B, 3, 3]
    :param t_pred: [B, 1, 3]
    :param dir:
    :param loss:
    :return:
    """
    src = src.squeeze(0).to('cpu').detach().numpy()
    tar = tar.squeeze(0).to('cpu').detach().numpy()
    R_gt = R_gt.squeeze(0).to('cpu').detach().numpy()
    t_gt = t_gt.squeeze(0).to('cpu').detach().numpy()
    R_pred = R_pred.squeeze(0).to('cpu').detach().numpy()
    t_pred = t_pred.squeeze(0).to('cpu').detach().numpy()
    loss = loss.item()
    output_file = dir

    # 打开文件以写入模式
    with open(output_file, "w") as f:
        f.write(str(loss) + "\n")
        # 遍历二维数组的每一行
        for row in R_gt:
            f.write("\t".join(map(str, row)) + "\n")
        for row in t_gt:
            f.write("\t".join(map(str, row)) + "\n")
        for row in R_pred:
            f.write("\t".join(map(str, row)) + "\n")
        for row in t_pred:
            f.write("\t".join(map(str, row)) + "\n")
        for row in src:
            f.write("\t".join(map(str, row)) + "\n")
        for row in tar:
            f.write("\t".join(map(str, row)) + "\n")