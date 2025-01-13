import torch
import torch.nn as nn
from DeepFeatExtractionLayer import DeepFeatExtractionLayer, SamplingLayer, GroupingLayer
from PointWeight_Layer import PointWeightLayer
from DeepFeatEmbeddingLayer import DeepFeatEmbeddingLayer
from CorPointGenerationLayer import CorPointGenerationLayer
from voxelize import *
from get_cat_feat_src import Get_Cat_Feat_Src
from get_cat_feat_tgt import Get_Cat_Feat_Tgt
class DeepVCP(nn.Module):
    def __init__(self, use_normal = False):
        super().__init__()
        self.feat_extraction = DeepFeatExtractionLayer(use_normal=use_normal, npoints=[2500, 2500, 2500],
                                                       radius=[1.5, 3, 5], nsamples=[16, 8, 8], in_dim=4, hidden_dim=128, out_dim=32)
        self.point_weight = PointWeightLayer(in_dim=32, mlp = [16, 8, 1], norm_num=2500)
        self.deep_feat_embedding = DeepFeatEmbeddingLayer(in_dim=32+3, nsample=64, mlp = [32, 32, 32])
        self.cor_point_generation_layer = CorPointGenerationLayer()



    def forward(self, source, target, R_init, t_init):
        B, _, _ = source.shape
        device = source.device

        src_feat_xyz, src_feat_normal = self.feat_extraction(source)    #[B, npoint, 3] [B, npoint, 32]

        #source -> weighting layer and get the topK points
        _, _, C = src_feat_normal.shape
        src_key_points_indices = self.point_weight(src_feat_normal) #[B, topK]
        src_key_points_xyz = src_feat_xyz.gather(dim=1, index=src_key_points_indices.unsqueeze(-1).repeat(1,1,3)) #[B, topK, 3]
        _, K, _ = src_key_points_xyz.shape


        #use the topKpoints as centroids to group the feature points
        grouping_layer = GroupingLayer(radius=2, nsample=64)
        src_key_points_grouped_indices = grouping_layer(src_key_points_xyz, src_feat_xyz) #not sure, [B, topK, nsample]
        _, _, nsample = src_key_points_grouped_indices.shape
        src_key_points_grouped_xyz = src_feat_xyz.gather(dim=1, index=torch.flatten(src_key_points_grouped_indices, start_dim=1, end_dim=2).unsqueeze(-1).repeat(1,1,3)).reshape(B, K, nsample, 3)
        src_key_points_grouped_normal = src_feat_normal.gather(dim=1, index=torch.flatten(src_key_points_grouped_indices, start_dim=1, end_dim=2).unsqueeze(-1).repeat(1,1,C)).reshape(B, K, nsample, C)
        src_gcf = Get_Cat_Feat_Src()
        src_key_points_feats_cat = src_gcf(src_key_points_xyz, src_key_points_grouped_xyz, src_key_points_grouped_normal)
        src_feat_emb = self.deep_feat_embedding(src_key_points_feats_cat) #[B, K, 32]



        #process the target point cloud
        tar_feat_xyz, tar_feat_normal = self.feat_extraction(target)  # [B, npoint, 3] [B, npoint, 32]
        # t_init: (B x 1 x 3)
        # R_init: (B x 3 x 3)
        # src_key_points_xyz: (B x K x 3)
        r = 1.0
        s = 0.4
        src_key_points_transformed = (torch.matmul(R_init.float(), src_key_points_xyz.float().permute(0,2,1)).permute(0,2,1)) + t_init #[B, K, 3]
        candidate_points_xyz = voxelize(src_key_points_transformed, r, s) #[B, K, C, 3]
        _, _, D, _ = candidate_points_xyz.shape

        target_gcf = Get_Cat_Feat_Tgt()
        tgt_key_points_feats_cat = target_gcf(candidate_points_xyz, tar_feat_xyz, tar_feat_normal)
        tar_feat_emb = self.deep_feat_embedding(tgt_key_points_feats_cat, is_src = False) #[B, K, C, 32]
        src_feat_emb = src_feat_emb.unsqueeze(2)

        result = self.cor_point_generation_layer(src_feat_emb, tar_feat_emb, candidate_points_xyz, r, s) #[B, K, 3]

        return src_key_points_xyz, result



















