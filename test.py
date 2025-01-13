import argparse
import time

import torch
from ModelNet40DataSet import ModelNet40DataSet
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from scipy.spatial.transform import Rotation
from ADNIDataset import ADNIDataset
import pickle
from Utils import *
import numpy as np
from DeepVCP import DeepVCP
from Loss import Loss

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ratios = (1, 1, 1, 1e-5, 1e-3)  # alpha beta eta gamma
    model = DeepVCP(use_normal=True)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load("epoch_0_model.pt"))

    root = "../subADNI/"
    batch_size = 1
    test_data = ADNIDataset(root=root, augment=True, split='test', partition=1)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    loss_fn = Loss()
    running_loss = 0
    cons_time = 0
    for n_batch, (src, target, R_gt, t_gt, src_file) in enumerate(test_loader):
        # mini batch
        src, target, R_gt, t_gt = src.to(device), target.to(device), R_gt.to(device), t_gt.to(device)
        t_init = torch.zeros(1, 3).to(device).unsqueeze(0)
        theta_x = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
        theta_y = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
        theta_z = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
        R_init = torch.from_numpy(RotX(theta_x) @ RotY(theta_y) @ RotZ(theta_z)).float().unsqueeze(0).to(device)

        start_time = time.time()
        src_keypts, target_vcp = model(src, target, R_init, t_init)
        end_time = time.time()
        cons_time = end_time - start_time

        loss, loss_detail, R, t = loss_fn(src[:, :, :3], target[:, :, :3], src_keypts, target_vcp, R_gt, t_gt,
                                          alpha=0.5)

        r_pred = Rotation.from_matrix(R.squeeze(0).cpu().detach().numpy())
        r_pred_arr = torch.tensor(r_pred.as_euler('xyz', degrees=True)).reshape(1, 3)
        r_gt = Rotation.from_matrix(R_gt.squeeze(0).cpu().detach().numpy())
        r_gt_arr = torch.tensor(r_gt.as_euler('xyz', degrees=True)).reshape(1, 3)
        pdist = nn.PairwiseDistance(p=2)

        print("")
        print(f"Batch: {n_batch}, Loss: {loss.item()}, average_Loss: {running_loss / (n_batch + 1)}")
        print(
            f"rotation error(euler): {pdist(r_pred_arr, r_gt_arr).item()}, rotation error(loss): {2 * torch.asin(loss_detail['fro_norm'] / 2.828427).item()}")
        print(
            f"translation error(pdist): {pdist(t.reshape(1, 3), t_gt.reshape(1, 3)).item()}, translation error(tensor): {(t_gt - t).reshape(3).detach()}")
        print("")

        running_loss += loss.item()


    print(f"times: {cons_time/len(test_loader)}")

if __name__ == "__main__":
    main()