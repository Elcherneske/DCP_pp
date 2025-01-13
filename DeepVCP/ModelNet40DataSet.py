
import torch
import os
import numpy as np

from torch.utils.data import Dataset
from Utils import *

class ModelNet40DataSet(Dataset):

    def __init__(self, root, augment = True, split = "train"):
        self.root = root
        self.augment = augment
        self.split = split

        #The class name in ModelNet40 set
        self.dirlists = [entry.name for entry in os.scandir(self.root)]

        #The file name of points for a single itme
        self.pointsdir = []
        for dir in self.dirlists:
            folder_name = os.path.join(self.root, dir, split)
            for entry in os.scandir(folder_name):
                self.pointsdir.append(entry)


    def __len__(self):
        return int(len(self.pointsdir)/20)


    ###structure: [N, 3]
    def __getitem__(self, index):
        src_file = self.pointsdir[index]
        with open(src_file) as file:
            data = np.array(file.readlines())
            num = int(data[1].split()[0])


            src_points = np.loadtxt(data[2:2+num])

        print('Processing file:', src_file)

        if(self.augment):
            theta_x = np.random.uniform(0 + np.pi * 0.1, np.pi * 2 - np.pi * 0.1)
            theta_y = np.random.uniform(0 + np.pi * 0.1, np.pi * 2 - np.pi * 0.1)
            theta_z = np.random.uniform(0 + np.pi * 0.1, np.pi * 2 - np.pi * 0.1)

            # generate random translation
            translation_max = 1.0
            translation_min = -1.0
            t = (translation_max - translation_min) * torch.rand(1, 3) + translation_min

            # Generate target point cloud by doing a series of random
            # rotations on source point cloud
            Rx = RotX(theta_x)
            Ry = RotY(theta_y)
            Rz = RotZ(theta_z)
            R = Rx @ Ry @ Rz

            # rotate source point cloud and normals
            target_points = src_points @ R

        src_points = torch.from_numpy(src_points)
        target_points = torch.from_numpy(target_points) + t

        R = torch.from_numpy(R)

        return (src_points, target_points, R, t)




if __name__ == '__main__':
    root = "./ModelNet40/"
    print(ModelNet40DataSet(root, split='test')[0])