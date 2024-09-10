import torch
import torch.nn as nn
import numpy as np
import random
import copy

from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from ..detection3d.decoder import W, L, H, SIN_YAW, COS_YAW

@PLUGIN_LAYERS.register_module()
class Denoise2D(nn.Module):
    def __init__(self,
                 num_cams=6,
                 num_dn_groups=0,
                 with_attn_mask=False):
        super().__init__()
        self.num_cams = num_cams
        self.num_dn_groups = num_dn_groups
        self.with_attn_mask = with_attn_mask


    def get_dn_project(
            self,
            anchor3d,
            metas,
            dn_trans_mask,
            dn_valid_mask2d,
            dn_cls_target2d,
            dn_box_target2d,
            dn_alpha_target2d=None,
            dn_depth_target2d=None
    ):
        # project dn_anchor3d and create 2d-gt
        device = anchor3d.device
        anchor3d_center = anchor3d[..., :3]
        lidar2imgs = torch.tile(metas['projection_mat'][:, None], (1, anchor3d.shape[1], 1, 1, 1))
        batch_size, num_anchor3d, num_cams = lidar2imgs.shape[:3]
        img_w, img_h = map(int, metas['image_wh'][0, 0].tolist())

        # get rotation_mat
        rotation_mat = anchor3d.new_zeros([batch_size, num_anchor3d, 3, 3])
        rotation_mat[:, :, 0, 0] = anchor3d[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor3d[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor3d[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor3d[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1

        # get anchor corners
        corners_norm = anchor3d.new_tensor(np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1))
        corners_norm = corners_norm - anchor3d.new_tensor([0.5, 0.5, 0.5])

        anchor3d_size = anchor3d[..., [W, L, H]].exp()

        anchor3d_corners = anchor3d_size[:, :, None, :] * corners_norm[None, None, :, :]
        anchor3d_corners = torch.matmul(rotation_mat[:, :, None], anchor3d_corners[..., None]).squeeze(-1)
        anchor3d_corners = anchor3d_corners + anchor3d_center[:, :, None, :]
        anchor3d_corners = torch.cat([anchor3d_corners, anchor3d_center[:, :, None, :]], dim=-2)

        # get points in camera and image plane
        coord_pts3d = torch.cat([anchor3d_corners, torch.ones_like(anchor3d_corners[..., :1])], -1)
        coord_pts3d = coord_pts3d.view(batch_size, num_anchor3d, 1, 9, 4, 1).repeat(1, 1, num_cams, 1, 1, 1)
        coord_pts2d = torch.matmul(lidar2imgs[:, :, :, None], coord_pts3d).squeeze(-1)

        center_pts2d = coord_pts2d[..., -1, :]
        corner_pts2d = coord_pts2d[..., :-1, :]
        center_depth2d = center_pts2d[..., 2:3]
        corner_depth2d = corner_pts2d[..., 2:3]

        center_pts2d = center_pts2d[..., :2] / center_depth2d.clamp(1e-5)
        corner_pts2d = corner_pts2d[..., :2] / corner_depth2d.clamp(1e-5)

        center_valid = ((0 < center_pts2d[..., 0]) & (center_pts2d[..., 0] < img_w) &
                        (0 < center_pts2d[..., 1]) & (center_pts2d[..., 1] < img_h))
        center_valid = torch.logical_and(center_valid, dn_trans_mask)

        # project corners to get corner-centers
        x_min = torch.clamp(corner_pts2d[..., 0].min(-1).values, min=0, max=img_w)
        x_max = torch.clamp(corner_pts2d[..., 0].max(-1).values, min=0, max=img_w)
        y_min = torch.clamp(corner_pts2d[..., 1].min(-1).values, min=0, max=img_h)
        y_max = torch.clamp(corner_pts2d[..., 1].max(-1).values, min=0, max=img_h)

        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        select_centers = torch.stack([cx, cy], dim=-1)
        select_centers[center_valid] = center_pts2d[center_valid]  # overwrite center points

        dn_trans_shape = dn_trans_mask.sum(1)
        dn_trans_meta_shape = dn_trans_shape.max(0).values
        dn_trans_meta_start = torch.cat([torch.zeros_like(dn_trans_meta_shape[:1]), dn_trans_meta_shape])
        dn_trans_meta_cumsum = dn_trans_meta_start.cumsum(-1).tolist()

        dn_trans_start = dn_trans_meta_start.cumsum(-1)[:self.num_cams].unsqueeze(0).repeat(batch_size, 1)
        dn_trans_end = dn_trans_start + dn_trans_shape

        dn_query_groups = [(qs, qe) for qs, qe in zip(dn_trans_meta_cumsum[:-1], dn_trans_meta_cumsum[1:])]
        dn_num_anchor2d = dn_trans_meta_shape.sum()

        # create reference points
        trans_mask_tmp = dn_trans_mask.permute(0, 2, 1).flatten(0, 1)
        select_centers = select_centers.permute(0, 2, 1, 3).flatten(0, 1)
        select_depths = center_depth2d.permute(0, 2, 1, 3).flatten(0, 1)  # corner depth is fake

        select_centers = select_centers[trans_mask_tmp]
        select_depths = select_depths[trans_mask_tmp]

        # create dn target
        dn_box_target2d = dn_box_target2d.permute(0, 2, 1, 3).flatten(0, 1)
        dn_cls_target2d = dn_cls_target2d.unsqueeze(-1).permute(0, 2, 1, 3).flatten(0, 1)
        dn_valid_mask2d = dn_valid_mask2d.unsqueeze(-1).permute(0, 2, 1, 3).flatten(0, 1)
        dn_alpha_target2d = dn_alpha_target2d.unsqueeze(-1).permute(0, 2, 1, 3).flatten(0, 1) if dn_alpha_target2d is not None else None
        dn_depth_target2d = dn_depth_target2d.unsqueeze(-1).permute(0, 2, 1, 3).flatten(0, 1) if dn_depth_target2d is not None else None

        dn_box_target2d = dn_box_target2d[trans_mask_tmp]
        dn_cls_target2d = dn_cls_target2d[trans_mask_tmp]
        dn_valid_mask2d = dn_valid_mask2d[trans_mask_tmp]
        dn_alpha_target2d = dn_alpha_target2d[trans_mask_tmp] if dn_alpha_target2d is not None else None
        dn_depth_target2d = dn_depth_target2d[trans_mask_tmp] if dn_depth_target2d is not None else None

        selected_mask = torch.zeros((batch_size, dn_num_anchor2d), device=device)
        dn_attn_mask = torch.ones((batch_size, dn_num_anchor2d, dn_num_anchor2d), device=device).fill_(float("-inf"))
        for bs in range(batch_size):
            for st, ed in zip(dn_trans_start[bs], dn_trans_end[bs]):
                selected_mask[bs, st:ed] = 1.0
                if self.with_attn_mask:
                    dn_attn_mask[bs, st:ed, st:ed] = 0.0
        selected_mask = selected_mask.unsqueeze(-1).repeat(1, 1, 4).to(torch.bool)

        # scatter dn reference points
        dn_pts2d = torch.zeros((batch_size, dn_num_anchor2d, 2), device=device)
        dn_depth2d = torch.zeros((batch_size, dn_num_anchor2d, 1), device=device)

        dn_pts2d = torch.masked_scatter(dn_pts2d, selected_mask[..., :2], select_centers)
        dn_depth2d = torch.masked_scatter(dn_depth2d, selected_mask[..., :1], select_depths.abs())

        # scatter dn target
        dn_box_lvl_target2d = torch.zeros((batch_size, dn_num_anchor2d, 4), device=device)
        dn_cls_lvl_target2d = -torch.ones((batch_size, dn_num_anchor2d, 1), device=device).to(torch.long)
        dn_valid_lvl_mask2d = torch.zeros((batch_size, dn_num_anchor2d, 1), device=device).to(torch.bool)
        dn_alpha_lvl_target2d = torch.zeros((batch_size, dn_num_anchor2d, 1), device=device) if dn_alpha_target2d is not None else None
        dn_depth_lvl_target2d = torch.zeros((batch_size, dn_num_anchor2d, 1), device=device) if dn_depth_target2d is not None else None

        dn_box_lvl_target2d = torch.masked_scatter(dn_box_lvl_target2d, selected_mask[..., :4], dn_box_target2d)
        dn_cls_lvl_target2d = torch.masked_scatter(dn_cls_lvl_target2d, selected_mask[..., :1], dn_cls_target2d)
        dn_valid_lvl_mask2d = torch.masked_scatter(dn_valid_lvl_mask2d, selected_mask[..., :1], dn_valid_mask2d)
        dn_alpha_lvl_target2d = torch.masked_scatter(dn_alpha_lvl_target2d, selected_mask[..., :1], dn_alpha_target2d) if dn_alpha_target2d is not None else None
        dn_depth_lvl_target2d = torch.masked_scatter(dn_depth_lvl_target2d, selected_mask[..., :1], dn_depth_target2d) if dn_depth_target2d is not None else None

        # create trans matrix
        dn_trans_matrix = nn.Parameter(
            torch.zeros((batch_size, dn_num_anchor2d, num_anchor3d), device=device), requires_grad=False)

        dn_meta_mask = dn_trans_mask.to(torch.float) + center_valid.to(torch.float)
        dn_meta_mask = dn_meta_mask.permute(0, 2, 1)

        for bs in range(batch_size):
            cam_index, pts3d_index = torch.nonzero(dn_meta_mask[bs]).chunk(2, dim=1)
            cam_index, pts3d_index = cam_index[:, 0], pts3d_index[:, 0]
            pts2d_index = torch.cat(
                [torch.arange(st, ed, device=device) for st, ed in zip(dn_trans_start[bs], dn_trans_end[bs])])
            dn_trans_matrix[bs, pts2d_index, pts3d_index] = dn_meta_mask[bs, cam_index, pts3d_index]

        dn_center_matrix = (dn_trans_matrix == 2).to(torch.float)
        dn_trans_matrix = (dn_trans_matrix >= 1).to(torch.float)  # include center and corner

        # post-normalize
        dn_pts2d = dn_pts2d / dn_pts2d.new_tensor([img_w, img_h])

        # organize dim
        dn_valid_lvl_mask2d = dn_valid_lvl_mask2d[..., 0]
        dn_cls_lvl_target2d = dn_cls_lvl_target2d[..., 0]
        dn_alpha_lvl_target2d = dn_alpha_lvl_target2d[..., 0] if dn_alpha_target2d is not None else None
        dn_depth_lvl_target2d = dn_depth_lvl_target2d[..., 0] if dn_depth_target2d is not None else None

        return dn_pts2d, dn_depth2d, dn_trans_mask, dn_trans_shape, dn_trans_matrix, dn_center_matrix, dn_query_groups, \
            dn_attn_mask, dn_valid_lvl_mask2d, dn_cls_lvl_target2d, dn_box_lvl_target2d, dn_alpha_lvl_target2d, dn_depth_lvl_target2d


    def get_self_dn_query_groups(self, ref_query_groups, dn_query_groups):
        self_dn_query_groups = copy.deepcopy(ref_query_groups)
        last = copy.deepcopy(self_dn_query_groups[-1])
        for dn_qg in dn_query_groups:
            self_dn_query_groups.append((last[1] + dn_qg[0], last[1] + dn_qg[1]))

        return self_dn_query_groups


    def get_cross_dn_query_groups(self, ref_query_groups, dn_query_groups):
        cross_dn_query_groups = []
        for qg, dn_qg in zip(ref_query_groups, dn_query_groups):
            cross_dn_query_groups.append((qg[0] + dn_qg[0], qg[1] + dn_qg[1]))

        return cross_dn_query_groups


    def permute_instance(self, instance_feature, anchor, anchor_embed, self_dn_query_groups):
        nc = self.num_cams
        qg = self_dn_query_groups
        anchor = torch.cat([torch.cat([
            anchor[:, qg[i][0]: qg[i][1]],
            anchor[:, qg[i + nc][0]: qg[i + nc][1]]], dim=1) for i in range(nc)], dim=1)

        anchor_embed = torch.cat([torch.cat([
            anchor_embed[:, qg[i][0]: qg[i][1]],
            anchor_embed[:, qg[i + nc][0]: qg[i + nc][1]]], dim=1) for i in range(nc)], dim=1)

        instance_feature = torch.cat([torch.cat([
            instance_feature[:, qg[i][0]: qg[i][1]],
            instance_feature[:, qg[i + nc][0]: qg[i + nc][1]]], dim=1) for i in range(nc)], dim=1)

        return instance_feature, anchor, anchor_embed


    def depermute_instance(self, instance_feature, anchor, anchor_embed, ref_query_groups, dn_query_groups):
        qgs = ref_query_groups
        dn_qgs = dn_query_groups

        anchor = torch.cat(
            [anchor[:, qg[0] + dn_qg[0]: qg[1] + dn_qg[0]] for qg, dn_qg in zip(qgs, dn_qgs)] + \
            [anchor[:, qg[1] + dn_qg[0]: qg[1] + dn_qg[1]] for qg, dn_qg in zip(qgs, dn_qgs)], dim=1)

        anchor_embed = torch.cat(
            [anchor_embed[:, qg[0] + dn_qg[0]: qg[1] + dn_qg[0]] for qg, dn_qg in zip(qgs, dn_qgs)] + \
            [anchor_embed[:, qg[1] + dn_qg[0]: qg[1] + dn_qg[1]] for qg, dn_qg in zip(qgs, dn_qgs)], dim=1)

        instance_feature = torch.cat(
            [instance_feature[:, qg[0] + dn_qg[0]: qg[1] + dn_qg[0]] for qg, dn_qg in zip(qgs, dn_qgs)] + \
            [instance_feature[:, qg[1] + dn_qg[0]: qg[1] + dn_qg[1]] for qg, dn_qg in zip(qgs, dn_qgs)], dim=1)

        return instance_feature, anchor, anchor_embed
