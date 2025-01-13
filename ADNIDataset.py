import torch
import os
import torch.nn as nn
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

from Utils import *

# class ADNIDataset(nn.Module):
#     def __init__(self, root, augment = True):
#         super().__init__()
#         self.root = root
#         self.augment = augment
#
#         self.dirlists = [entry.name for entry in os.scandir(self.root)]
#
#         self.pointsdir = [os.path.join(self.root, dir) for dir in self.dirlists]
#
#     def __len__(self):
#         return len(self.pointsdir)
#
#     def __getitem__(self, index):
#         src_file = self.pointsdir[index]
#         with open(src_file) as file:
#             data = np.array(file.readlines())
#             num = int(data[0])
#             src_points = np.loadtxt(data[1:1 + num])
#
#             src_points_xyz = src_points[:, :3] / np.sqrt(np.sqrt(3) * np.mean(src_points[:, :3]**2)) * 10  # normalize the point cloud
#
#         print('Processing file:', src_file)
#
#         if (self.augment):
#             theta_x = np.random.uniform(0 + np.pi * 0.1, np.pi * 2 - np.pi * 0.1)
#             theta_y = np.random.uniform(0 + np.pi * 0.1, np.pi * 2 - np.pi * 0.1)
#             theta_z = np.random.uniform(0 + np.pi * 0.1, np.pi * 2 - np.pi * 0.1)
#
#             # generate random translation
#             translation_max = 10.0
#             translation_min = -10.0
#             t = (translation_max - translation_min) * torch.rand(1, 3) + translation_min
#
#             # Generate target point cloud by doing a series of random
#             # rotations on source point cloud
#             Rx = RotX(theta_x)
#             Ry = RotY(theta_y)
#             Rz = RotZ(theta_z)
#             R = Rx @ Ry @ Rz
#
#             # rotate source point cloud and normals
#             target_points_xyz = src_points_xyz @ R
#
#         src_points_normal = torch.from_numpy(src_points[:, 3:])
#         src_points = torch.cat([torch.from_numpy(src_points_xyz), src_points_normal], dim=1)
#         target_points = torch.cat([torch.from_numpy(target_points_xyz) + t, src_points_normal], dim=1)
#
#         R = torch.from_numpy(R)
#
#         return (src_points, target_points, R, t)

class ADNIDataset(nn.Module):
    def __init__(self, root, augment=True, split='train', partition=6):
        super().__init__()
        self.root = root
        self.augment = augment
        self.Imgsdir = []
        for cur_path, dirs, files in os.walk(root):
            for file in files:
                self.Imgsdir.append(os.path.join(cur_path, file))
        sub_dir = self.Imgsdir[::partition]
        if split == 'train':
            self.Imgsdir = [x for x in self.Imgsdir if x not in sub_dir]
        else:
            self.Imgsdir = sub_dir

    def __len__(self):
        return int(len(self.Imgsdir))

    def __getitem__(self, index, threadhold=25):
        src_file = self.Imgsdir[index]
        print('Processing file:', src_file)
        img_nib = nib.load(src_file)
        img_data = img_nib.get_fdata() #[x_shape, y_shape, z_shape]
        img_data = torch.tensor(img_data)
        zooms = img_nib.header.get_zooms()  # similar to the spacing settings in the 3D compounding

        img, zooms = self.CNN3Layer(img_data, zooms)
        img, zooms = self.CNN3Layer(img, zooms)
        img, zooms = self.CNN3Layer(img, zooms) #temp use for small set

        img = img / torch.max(torch.flatten(img, start_dim=0, end_dim=2), dim=-1).values * 1000

        x, y, z = img.shape
        x_width, y_width, z_width = zooms
        X = (torch.flatten(torch.arange(0, x).unsqueeze(-1).unsqueeze(-1).repeat(1, y, z), start_dim=0, end_dim=-1) * x_width).unsqueeze(-1)
        Y = (torch.flatten(torch.arange(0, y).unsqueeze(0).unsqueeze(-1).repeat(x, 1, z), start_dim=0, end_dim=-1) * y_width).unsqueeze(-1)
        Z = (torch.flatten(torch.arange(0, z).unsqueeze(0).unsqueeze(0).repeat(x, y, 1), start_dim=0, end_dim=-1) * z_width).unsqueeze(-1)
        value = (torch.flatten(img, start_dim=0, end_dim=-1)).unsqueeze(-1)
        src_points = torch.cat([X, Y, Z, value], dim=-1)
        src_points = src_points[src_points[:, 3] > threadhold]
        print(f"Item: {index}, Points shape: {src_points.shape}")
        # print('zooms of the voxel: ', zooms)

        # src_points_xyz = src_points[:, :3] / torch.mean(torch.sqrt(torch.sum(src_points[:, :3] ** 2, dim=-1)), dim=-1) * 10  # normalize the point cloud
        src_points_xyz = (src_points[:, :3] - torch.mean(src_points[:, :3], dim=0, keepdim=True)) / torch.max(torch.sqrt(torch.sum(src_points[:, :3] ** 2, dim=-1)), dim=-1).values * 10  # normalize the point cloud
        src_points = torch.cat([src_points_xyz, src_points[:,3:]],dim=-1)

        if (self.augment):
            theta_x = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
            theta_y = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
            theta_z = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
            print(f"angle:({theta_x/np.pi * 180}, {theta_y/np.pi * 180}, {theta_z/np.pi * 180})")

            # generate random translation
            translation_max = 4.0
            translation_min = -4.0
            t = (translation_max - translation_min) * torch.rand(1, 3) + translation_min

            # Generate target point cloud by doing a series of random
            # rotations on source point cloud
            Rx = RotX(theta_x)
            Ry = RotY(theta_y)
            Rz = RotZ(theta_z)
            R = Rx @ Ry @ Rz

            R = torch.from_numpy(R).float()
            # rotate source point cloud and normals
            target_points_xyz = torch.matmul(R, src_points[:,:3].permute(1,0).float()).permute(1,0)

            src_points_normal = src_points[:,3:] + torch.randn(src_points.shape[0], 3) * 1000 * 0.1
            target_points = torch.cat([target_points_xyz + t, src_points_normal], dim=-1)
            return (src_points, target_points, R, t, src_file)

        else:
            return src_points, src_file

    def CNN3Layer(self, img, zooms, kernel_size=3, stride=2, padding=1, grad=False):
        img = img.unsqueeze(0).unsqueeze(0).float()

        kernel = torch.ones(1, 1, 3, 3, 3) / 27.0
        # 创建卷积层
        conv_layer = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # 将自定义的卷积核加载到卷积层中
        conv_layer.weight.data = kernel
        conv_layer.weight.requires_grad=grad

        img = conv_layer(img)

        img = img.squeeze(0).squeeze(0)

        zooms = (zooms[0] * stride, zooms[1] * stride, zooms[2] * stride)

        return img, zooms


if __name__ == '__main__':
    root = "../ADNI/"
    print(ADNIDataset(root, augment=True)[0])
    # x=2
    # y=2
    # z=2
    # x_width=1
    # y_width = 2
    # z_width=3
    # tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 2, 2)
    # print(tensor)
    # X = torch.flatten(torch.arange(0, x).unsqueeze(-1).unsqueeze(-1).repeat(1, y, z), start_dim=0, end_dim=-1) * x_width
    # Y = torch.flatten(torch.arange(0, y).unsqueeze(0).unsqueeze(-1).repeat(x, 1, z), start_dim=0, end_dim=-1) * y_width
    # Z = torch.flatten(torch.arange(0, z).unsqueeze(0).unsqueeze(0).repeat(x, y, 1), start_dim=0, end_dim=-1) * z_width
    #
    # X = X.unsqueeze(-1)
    # Y = Y.unsqueeze(-1)
    # Z = Z.unsqueeze(-1)
    # value = torch.flatten(tensor, start_dim=0, end_dim=-1).unsqueeze(-1)
    # print(X.shape)
    # result  = torch.cat([X,Y,Z,value], dim=-1)
    #
    # print(result)
    #
    # print(result[result[:, 3]>3])

