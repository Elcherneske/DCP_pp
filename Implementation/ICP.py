import numpy as np
import open3d as o3d
import torch
from Utils import *
from ADNIDataset import ADNIDataset
from ModelNet40DataSet import ModelNet40DataSet
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle

def main():
    # hyper-parameters
    num_epochs = 1
    batch_size = 1
    print(f"Params: epochs: {num_epochs}, batch: {batch_size}")
    dataset = "ADNI"
    # dataset
    if dataset == "modelnet":
        root = "../ModelNet40/"
        train_data = ModelNet40DataSet(root=root, augment=True, split='train')
        test_data = ModelNet40DataSet(root=root, augment=True, split='test')
        use_normal = False
    if dataset == "ADNI":
        root = "../subADNI/"
        train_data = ADNIDataset(root=root, augment=True, split='train', partition=6)
        test_data = ADNIDataset(root=root, augment=True, split='test', partition=6)
        use_normal = True
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    print('Train dataset size: ', len(train_data))
    print('Test dataset size: ', len(test_data))

    for epoch in range(num_epochs):
        print(f"epoch #{epoch}")
        loss_epoch = []
        running_loss = 0.0
        for n_batch, (src, tar, R_gt, t_gt, src_file) in enumerate(test_loader):
            src_xyz = src[:, :, :3].squeeze(0).detach().numpy()
            tar_xyz = tar[:, :, :3].squeeze(0).detach().numpy()
            src_color = src[:, :, 3:].repeat(1, 1, 3).squeeze(0).detach().numpy()
            tar_color = tar[:, :, 3:].repeat(1, 1, 3).squeeze(0).detach().numpy()
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(src_xyz)
            pcd1.colors = o3d.utility.Vector3dVector(src_color)
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(tar_xyz)
            pcd2.colors = o3d.utility.Vector3dVector(tar_color)
            source = pcd1
            target = pcd2

            translation_max = 4.0
            translation_min = -4.0
            t_init = (translation_max - translation_min) * torch.rand(1, 3) + translation_min
            t_init = t_init.unsqueeze(0).permute(0,2,1)
            theta_x = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
            theta_y = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
            theta_z = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
            R_init = torch.from_numpy(RotX(theta_x) @ RotY(theta_y) @ RotZ(theta_z)).float().unsqueeze(0)
            init = torch.cat(
                [torch.cat([R_init, t_init], dim=-1), torch.tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(0).unsqueeze(0)],
                dim=1)
            init = init.squeeze(0).detach().numpy()

            icp_result = o3d.pipelines.registration.registration_icp(
                source, target, max_correspondence_distance=0.02, init=init,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.cuda.pybind.pipelines.registration.ICPConvergenceCriteria(max_iteration=400)
            )

            transform = torch.tensor(icp_result.transformation).unsqueeze(0)

            loss, loss_detail = Loss(src=src[:,:,:3], tar=tar[:,:,:3], transform=transform, R_gt=R_gt, t_gt=t_gt)


            print(
                f"n_batch: {n_batch}, Loss: {loss.item()}, PointLoss: {loss_detail['point_loss'].item()}, fro_norm: {loss_detail['fro_norm'].item()}, MRAE: {loss_detail['MRAE'].item()}, "
                f"MRTE: {loss_detail['MRTE'].item()}")

            running_loss += loss.item()
            loss_epoch.append([loss.item(), loss_detail['point_loss'].item(), loss_detail['fro_norm'].item(),
                               loss_detail['MRAE'].item(), loss_detail['MRTE'].item()])

        with open("training_loss_" + str(epoch) + ".txt", "wb") as fp:  # Pickling
            pickle.dump(loss_epoch, fp)

def Loss(src, tar, transform, R_gt, t_gt):
    loss_fc = nn.MSELoss(reduction="mean")
    R = transform[:, :3, :3].float()
    t = transform[:, :3, 3:].float()
    tar_pred = (torch.matmul(R, src.float().permute(0,2,1)) + t).permute(0,2,1)
    point_loss = loss_fc(tar.squeeze(0), tar_pred.squeeze(0))
    # R_loss
    fro_norm = torch.norm(R_gt - R, "fro")
    MRAE = torch.acos((torch.trace(torch.matmul(torch.inverse(R), R_gt).squeeze(0)) - 1) / 2)

    # t_loss
    MRTE = loss_fc(t, t_gt)

    my_dict = {'point_loss': point_loss, 'fro_norm': fro_norm, 'MRAE': MRAE, 'MRTE': MRTE}

    return point_loss, my_dict



if __name__ == "__main__":
    main()



