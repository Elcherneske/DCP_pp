import argparse
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


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default="ADNI", help='dataset (specify modelnet or kitti)')
parser.add_argument('-f', '--full_dataset', default="full", help='specify to train on full or partial dataset')
parser.add_argument('-r', '--retrain_path', action = "store", type = str, help='specify a saved model to retrain on')
parser.add_argument('-m', '--model_path', default="./model/final_model.pt", action = "store", type = str, help='specify path to save final model')

args = parser.parse_args()
dataset = args.dataset
retrain_path = args.retrain_path
model_path = args.model_path
full_dataset = True if args.full_dataset == "full" else False


def main():
    # hyper-parameters
    num_epochs = 1
    batch_size = 1
    lr = 0.01
    # loss balancing factor
    alpha = 0.5

    print(f"Params: epochs: {num_epochs}, batch: {batch_size}, lr: {lr}, alpha: {alpha}\n")

    # check if cuda is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f"device: {device}")

    # dataset
    if dataset == "modelnet":
        root = "../ModelNet40/"
        train_data = ModelNet40DataSet(root=root, augment=True, split='train')
        test_data = ModelNet40DataSet(root=root, augment=True, split='test')
        use_normal = False
    if dataset == "ADNI":
        root = "../ADNI/"
        train_data = ADNIDataset(root=root, augment=True, split='train', partition=6)
        test_data = ADNIDataset(root=root, augment=True, split='test', partition=6)
        use_normal = True
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    num_train = len(train_data)
    num_test = len(test_data)
    print('Train dataset size: ', num_train)
    print('Test dataset size: ', num_test)

    # Initialize the model
    model = DeepVCP(use_normal=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    loss_fn = Loss()

    # Retrain
    if retrain_path:
        print("Retrain on ", retrain_path)
        model.load_state_dict(torch.load(retrain_path))
    else:
        print("No retrain")

    # Define the optimizer
    optim = Adam(model.parameters(), lr=lr)

    # begin train
    model.train()
    for epoch in range(num_epochs):
        print(f"epoch #{epoch}")
        loss_epoch = []
        running_loss = 0.0

        for n_batch, (src, target, R_gt, t_gt, src_file) in enumerate(train_loader):
            # mini batch
            if n_batch % 200 == 0:
                lr = lr * 0.6
                optim = Adam(model.parameters(), lr=lr)

            src, target, R_gt, t_gt = src.to(device), target.to(device), R_gt.to(device), t_gt.to(device)
            t_init = torch.zeros(1, 3).to(device).unsqueeze(0)
            theta_x = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
            theta_y = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
            theta_z = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
            R_init = torch.from_numpy(RotX(theta_x) @ RotY(theta_y) @ RotZ(theta_z)).float().unsqueeze(0).to(device)
            src_keypts, target_vcp = model(src, target, R_init, t_init)

            # zero gradient
            optim.zero_grad()

            loss, loss_detail, R, t = loss_fn(src[:,:,:3], target[:,:,:3], src_keypts, target_vcp, R_gt, t_gt, alpha=0.5)

            print(
                f"n_batch: {n_batch}, Loss: {loss.item()}, PointLoss: {loss_detail['point_loss'].item()}, fro_norm: {loss_detail['fro_norm'].item()}, MRAE: {loss_detail['MRAE'].item()}, "
                f"MRTE: {loss_detail['MRTE'].item()}")
            loss.backward()
            optim.step()

            running_loss += loss.item()
            loss_epoch.append([loss.item(), loss_detail['point_loss'].item(), loss_detail['fro_norm'].item(),
                               loss_detail['MRAE'].item(), loss_detail['MRTE'].item()])


        torch.save(model.state_dict(), "epoch_" + str(epoch) + "_model.pt")
        with open("training_loss_" + str(epoch) + ".txt", "wb") as fp:  # Pickling
            pickle.dump(loss_epoch, fp)

    print("Finished Training")

    # begin test
    loss_test = []
    running_loss = 0.0
    for n_batch, (src, target, R_gt, t_gt, src_file) in enumerate(test_loader):
        # mini batch
        src, target, R_gt, t_gt = src.to(device), target.to(device), R_gt.to(device), t_gt.to(device)
        t_init = torch.zeros(1, 3).to(device).unsqueeze(0)
        theta_x = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
        theta_y = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
        theta_z = np.random.uniform(0 + np.pi * 0.1, np.pi * 0.5 - np.pi * 0.1)
        R_init = torch.from_numpy(RotX(theta_x) @ RotY(theta_y) @ RotZ(theta_z)).float().unsqueeze(0).to(device)
        src_keypts, target_vcp = model(src, target, R_init, t_init)

        loss, loss_detail, R, t = loss_fn(src_keypts, target_vcp, R_gt, t_gt, alpha=0.5)

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
        loss_test.append([loss.item(), loss_detail['point_loss'].item(), loss_detail['fro_norm'].item(),
                          loss_detail['MRAE'].item(), loss_detail['MRTE'].item(),
                          loss_detail['matrix_loss'].item(), loss_detail['distribute_loss'].item()])
        if loss < 0.1:
            dir = "./points/nbatch_" + str(n_batch) + "_points.txt"
            writePoints(src=src, tar=target, R_gt=R_gt, R_pred=R, t_gt=t_gt, t_pred=t, dir=dir,
                        loss=loss_detail['point_loss'])

    with open("test_loss.txt", "wb") as fp_test:  # Pickling
        pickle.dump(loss_test, fp_test)

if __name__ == "__main__":
    main()