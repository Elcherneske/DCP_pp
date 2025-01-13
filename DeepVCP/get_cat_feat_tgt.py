import torch
import torch.nn as nn

from KNN import KNN

'''
Get concatenated local coordinates and normalized features for candidate points

    1) Find K nearest neighbors for candidate_pts in tgt_pts_xyz
    2) Convert the xyz coordinates for KNN into local w.r.t. candidate_pts
    3) Normalize the deep features in src_keyfeats based on distance between
       src_keypts with its K nearest neighbors
'''
class Get_Cat_Feat_Tgt(nn.Module):
    def __init__(self):
        super(Get_Cat_Feat_Tgt, self).__init__()

    def forward(self, candidate_xyz, tar_xyz, tar_normal):
        """
        Input:
            candidate_xyz: candidate corresponding points (B x K_topk x C x 3)
            src_key_xyz: keypoints in src point cloud (B x K_topk x 3)
            tar_xyz: original points in target point cloud (B x N x 3)
            tar_normal: deep features for tgt point cloud (B x N x num_feats)
        
        Output: 
            tgt_keyfeats_cat: concatenated local coordinates of candidate points and 
                              normalized deep features (B x K_topk x C x nsample x (3 + num_feats))
        """
        B, K, C, _ = candidate_xyz.shape
        _, _, D = tar_normal.shape
        nsample = 64
        knn = KNN(K = nsample)
        dist, idx = knn(torch.flatten(candidate_xyz, start_dim = 1, end_dim = 2), tar_xyz) #  (B x (K_topk x C) x nsample)

        tar_grouped_xyz = tar_xyz.gather(dim=1, index=torch.flatten(idx, start_dim= 1, end_dim=2).unsqueeze(-1).repeat(1,1,3)).reshape(B, K, C, nsample, 3) #(B x K_topk x C x nsample x 3)
        tar_grouped_normal = tar_normal.gather(dim=1, index=torch.flatten(idx, start_dim= 1, end_dim=2).unsqueeze(-1).repeat(1,1,D)).reshape(B, K, C, nsample, D) #(B x K_topk x C x nsample x num_feat)
        candidates_local_xyz = tar_grouped_xyz - candidate_xyz.unsqueeze(3).repeat(1,1,1,nsample,1)


        dist_normalize = (dist / torch.sum(dist, dim = -1, keepdim = True)).reshape(B, K, C, nsample).unsqueeze(-1).repeat(1,1,1,1,D) #(B x (K_topk x C) x nsample x num_feat)

        result = torch.cat((candidates_local_xyz, tar_grouped_normal * dist_normalize), dim = -1)

        return result
