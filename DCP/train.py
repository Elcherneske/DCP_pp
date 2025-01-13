import argparse
import torch
from ADNIDataset import ADNIDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, SGD
from scipy.spatial.transform import Rotation
import pickle
import shutil
from model import DCP
from Loss import Loss
from Utils import *


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default="ADNI", help='dataset (specify modelnet or kitti)')
parser.add_argument('-f', '--full_dataset', default="full", help='specify to train on full or partial dataset')
parser.add_argument('-r', '--retrain_path', action = "store", type = str, help='specify a saved model to retrain on')
parser.add_argument('-m', '--model_path', default="./final_model.pt", action = "store", type = str, help='specify path to save final model')
parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='Name of the experiment')
parser.add_argument('--model', type=str, default='dcp', metavar='N', choices=['dcp'], help='Model to use, [dcp]')
parser.add_argument('--emb_nn', type=str, default='pointnet', metavar='N', choices=['pointnet', 'dgcnn'], help='Embedding nn to use, [pointnet, dgcnn]')
parser.add_argument('--pointer', type=str, default='transformer', metavar='N', choices=['identity', 'transformer'], help='Attention-based pointer generator to use, [identity, transformer]')
parser.add_argument('--head', type=str, default='svd', metavar='N', choices=['mlp', 'svd', ], help='Head to use, [mlp, svd]')
parser.add_argument('--emb_dims', type=int, default=128, metavar='N', help='Dimension of embeddings')
parser.add_argument('--n_blocks', type=int, default=1, metavar='N', help='Num of blocks of encoder&decoder')
parser.add_argument('--n_heads', type=int, default=4, metavar='N', help='Num of heads in multiheadedattention')
parser.add_argument('--ff_dims', type=int, default=512, metavar='N', help='Num of dimensions of fc in transformer')
parser.add_argument('--dropout', type=float, default=0.0, metavar='N', help='Dropout ratio in transformer')
parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of episode to train ')
parser.add_argument('--use_sgd', action='store_true', default=False, help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1)')
parser.add_argument('--eval', action='store_true', default=False, help='evaluate the model')
parser.add_argument('--cycle', type=bool, default=False, metavar='N', help='Whether to use cycle consistency')
parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N', help='Wheter to add gaussian noise')
parser.add_argument('--unseen', type=bool, default=False, metavar='N', help='Wheter to test on unseen category')
parser.add_argument('--num_points', type=int, default=1024, metavar='N', help='Num of points to use')
parser.add_argument('--factor', type=float, default=4, metavar='N', help='Divided factor for rotations')


args = parser.parse_args()
dataset = args.dataset
retrain_path = args.retrain_path
model_path = args.model_path
full_dataset = True if args.full_dataset == "full" else False


def main():
    # hyper-parameters
    num_epochs = 1
    batch_size = 1
    lr = 0.001

    print(f"Params: epochs: {num_epochs}, batch: {batch_size}, lr: {lr}")

    # check if cuda is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f"device: {device}")

    # dataset
    if dataset == "ADNI":
        root = "../ADNI/"
        train_data = ADNIDataset(root=root, augment=True, split='train', partition=6)
        test_data = ADNIDataset(root=root, augment=True, split='test', partition=6)
        use_normal = True
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


    print('Train dataset size: ', len(train_data))
    print('Test dataset size: ', len(test_data))



    # Initialize the model
    model = DCP(args)
    print(f"parameter number: {sum(p.numel() for p in model.parameters())}")
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

            src, target, R_gt, t_gt = src.to(device), target.to(device), R_gt.to(device), t_gt.to(device)
            R, t, _, _ = model(src.permute(0,2,1), target.permute(0,2,1)) # R [B, 3, 3]  t [B, 1, 3]
            optim.zero_grad()
            loss, loss_detail = loss_fn(src[:,:,:3], R_pred=R, t_pred=t, R_true=R_gt, t_true=t_gt)
            loss.backward()
            optim.step()

            print(
                f"n_batch: {n_batch}, Loss: {loss.item()}, PointLoss: {loss_detail['point_loss'].item()}, fro_norm: {loss_detail['fro_norm'].item()}, MRAE: {loss_detail['MRAE'].item()}, "
                f"MRTE: {loss_detail['MRTE'].item()}")

            running_loss += loss.item()
            loss_epoch.append([loss.item(), loss_detail['point_loss'].item(), loss_detail['fro_norm'].item(),
                               loss_detail['MRAE'].item(), loss_detail['MRTE'].item()])

        torch.save(model.state_dict(), "epoch_" + str(epoch) + "_model.pt")
        with open("training_loss_" + str(epoch) + ".txt", "wb") as fp:  # Pickling
            pickle.dump(loss_epoch, fp)

    # save
    print("Finished Training")

    # begin test
    loss_test = []
    running_loss = 0.0
    for n_batch, (src, target, R_gt, t_gt, src_file) in enumerate(test_loader):
        # mini batch
        src, target, R_gt, t_gt = src.to(device), target.to(device), R_gt.to(device), t_gt.to(device)
        optim.zero_grad()
        R, t, _, _ = model(src.premute(0,2,1), target.permute(0,2,1)) # R [B, 3, 3]  t [B, 1, 3]

        loss, loss_detail = loss_fn(src[:,:,:3], R_pred=R, t_pred=t, R_true=R_gt, t_true=t_gt)
        # error metric for rigid body transformation
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
                          loss_detail['MRAE'].item(), loss_detail['MRTE'].item()])
        if loss < 0.1:
            dir = "./points/nbatch_" + str(n_batch) + "_points.txt"
            writePoints(src=src, tar=target, R_gt=R_gt, R_pred=R, t_gt=t_gt, t_pred=t, dir=dir,
                        loss=loss_detail['point_loss'])


    with open("test_loss.txt", "wb") as fp_test:  # Pickling
        pickle.dump(loss_test, fp_test)



if __name__ == "__main__":
    main()