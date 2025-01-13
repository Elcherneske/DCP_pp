import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DeepFeatExtractionLayer import DeepFeatExtractionLayer, SamplingLayer, GroupingLayer
from PointWeight_Layer import SrcPointWeightLayer, TarPointWeightLayer, LinearPointWeightLayer
from DeepFeatEmbeddingLayer import DeepFeatEmbeddingLayer
from LocalFeatNormalize import LocalFeatNorm
from TransformMatrixGeneration import TransformMatrixGenration
from RecoverTransformLayer import RecoverTransformLayer
from ModelNet40DataSet import ModelNet40DataSet


class DeepVCP_POT(nn.Module):
    def __init__(self, use_normal = False):
        super().__init__()
        self.feat_extraction = DeepFeatExtractionLayer(use_normal=use_normal, npoints=[2500, 2500],
                                                       radius=[1.5, 3], nsamples=[16, 8], in_dim=4, hidden_dim=512, out_dim=256)
        self.src_weight_layer = SrcPointWeightLayer(K=256)
        self.tar_weight_layer = TarPointWeightLayer(K=256)

        # self.deep_feat_emb = DeepFeatEmbeddingLayer(in_dim=4, hidden_dim=256, out_dim=64, nsample=64, K=256)
        # self.local_feat_norm = LocalFeatNorm()
        self.matrix_generation = TransformMatrixGenration(K=256, hidden_dim=256, CNN_channels = [1, 16, 16, 1])
        self.recover_transform = RecoverTransformLayer()


    def forward(self, source, target):
        """
        :param source: [B, N, 3+C] xyz + normal(optinal)
        :param target: [B, N, 3+C] xyz + normal(optinal)
        :param R_init: [3,3]
        :param t_init: [1,3]
        :return: pre_target_xyz[B, N, 3]
                R [3, 3]
                t [1, 3]
        """
        B, _, _ = source.shape
        device = source.device

        # feature extraction
        src_feat_xyz, src_feat_normal = self.feat_extraction(source)  #[B, npoint, 3] [B, npoint, 64]
        _, _, C = src_feat_normal.shape
        #key points
        src_key_indices = self.src_weight_layer(src_feat_normal) # [B, topK]
        src_key_xyz = src_feat_xyz.gather(dim=1, index=src_key_indices.unsqueeze(-1).repeat(1, 1, 3))  # [B, topK, 3]
        src_key_normal = src_feat_normal.gather(dim=1, index=src_key_indices.unsqueeze(-1).repeat(1,1,C))
        _, K, _ = src_key_xyz.shape



        tar_feat_xyz, tar_feat_normal = self.feat_extraction(target) #[B, npoint, 3] [B, npoint, 64]
        #key points
        tar_key_indices = self.tar_weight_layer(src_key_normal, tar_feat_normal)  # [B, topK]  #todo:可以换成没有deep feat emb的srckeynormal
        tar_key_xyz = tar_feat_xyz.gather(dim=1, index=tar_key_indices.unsqueeze(-1).repeat(1, 1, 3))  # [B, topK, 3]
        tar_key_normal = tar_feat_normal.gather(dim=1, index=tar_key_indices.unsqueeze(-1).repeat(1, 1, C))



        src_key_deep_feat = src_key_normal
        tar_key_deep_feat = tar_key_normal


        K_matrix = self.matrix_generation(src_key_deep_feat, tar_key_deep_feat)  # [B, K, K]
        R, t = self.recover_transform(src_key_xyz, tar_key_xyz, K_matrix)

        return R, t, K_matrix, src_key_xyz, tar_key_xyz





if __name__ == "__main__":
    root = "../ModelNet40/"
    train_data = ModelNet40DataSet(root, split="train")
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
    model = DeepVCP_POT(use_normal=False)
    for n_batch, (src, target, R_gt, t_gt) in enumerate(train_loader):
        model(src, target, R_gt, t_gt)

