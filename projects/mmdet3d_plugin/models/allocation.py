import torch
import torch.nn as nn
import random
import numpy as np

from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from .detection3d.decoder import W, L, H, SIN_YAW, COS_YAW

@PLUGIN_LAYERS.register_module()
class DynamicQueryAllocation(nn.Module):
    def __init__(self,
                 with_attn_mask=False,
                 with_project_wh=False,
                 limit_anchor_size=[35, 35, 10],
                 limit_corners_num=[100] * 6,
                 ):
        super().__init__()
        self.with_attn_mask = with_attn_mask
        self.with_project_wh = with_project_wh
        self.limit_anchor_size = limit_anchor_size
        self.limit_corners_num = limit_corners_num

    def forward(self, anchor3d, metas):
        outputs = self.projection_allocation(anchor3d, metas)
        return outputs

    def projection_allocation(self, anchor3d, metas):
        device = anchor3d.device
        anchor3d_center = anchor3d[..., :3]
        lidar2imgs = torch.tile(metas['projection_mat'][:, None], (1, anchor3d.shape[1], 1, 1, 1))
        batch_size, num_anchor3d, num_cams = lidar2imgs.shape[:3]
        img_w, img_h = map(int, metas['image_wh'][0, 0].tolist())

        # get rotation mat
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
        anchor3d_size = anchor3d_size.clamp(max=torch.tensor(self.limit_anchor_size, device=device).view(1, 1, -1))

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

        corner_valid1 = (corner_depth2d[..., 0] > 0)
        corner_valid2 = ((0 < corner_pts2d[..., 0]) & (corner_pts2d[..., 0] < img_w) &
                         (0 < corner_pts2d[..., 1]) & (corner_pts2d[..., 1] < img_h))
        corner_valid = torch.logical_and(corner_valid1, corner_valid2).any(-1)

        # project corners to get corner-centers
        x_min = torch.clamp(corner_pts2d[..., 0].min(-1).values, min=0, max=img_w)
        x_max = torch.clamp(corner_pts2d[..., 0].max(-1).values, min=0, max=img_w)
        y_min = torch.clamp(corner_pts2d[..., 1].min(-1).values, min=0, max=img_h)
        y_max = torch.clamp(corner_pts2d[..., 1].max(-1).values, min=0, max=img_h)

        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        select_centers = torch.stack([cx, cy], dim=-1)
        select_centers[center_valid] = center_pts2d[center_valid]  # overwrite center points

        if self.training and self.limit_corners_num:
            corner_valid = torch.where(torch.logical_and(corner_valid, center_valid), False, corner_valid)
            corner_valid = self.random_sample_corner_mask(corner_valid)

        # divide to groups
        trans_mask = torch.logical_or(center_valid, corner_valid)
        trans_shape = trans_mask.sum(1)
        trans_meta_shape = trans_shape.max(0).values
        trans_meta_start = torch.cat([torch.zeros_like(trans_meta_shape[:1]), trans_meta_shape])
        trans_meta_cumsum = trans_meta_start.cumsum(-1).tolist()

        trans_start = trans_meta_start.cumsum(-1)[:num_cams].unsqueeze(0).repeat(batch_size, 1)
        trans_end = trans_start + trans_shape

        query_groups = [(qs, qe) for qs, qe in zip(trans_meta_cumsum[:-1], trans_meta_cumsum[1:])]
        num_anchor2d = trans_meta_shape.sum()

        # create reference points
        trans_mask_tmp = trans_mask.permute(0, 2, 1).flatten(0, 1)
        select_centers = select_centers.permute(0, 2, 1, 3).flatten(0, 1)
        select_depths = center_depth2d.permute(0, 2, 1, 3).flatten(0, 1)  # corner depth is fake

        select_centers = select_centers[trans_mask_tmp]
        select_depths = select_depths[trans_mask_tmp]

        selected_mask = torch.zeros((batch_size, num_anchor2d), device=device)
        attn_mask = torch.ones((batch_size, num_anchor2d, num_anchor2d), device=device).fill_(float("-inf"))
        for bs in range(batch_size):
            for st, ed in zip(trans_start[bs], trans_end[bs]):
                selected_mask[bs, st:ed] = 1.0
                if self.with_attn_mask:
                    attn_mask[bs, st:ed, st:ed] = 0.0
        selected_mask = selected_mask.unsqueeze(-1).repeat(1, 1, 2).to(torch.bool)

        ref_pts2d = torch.zeros((batch_size, num_anchor2d, 2), device=device)
        ref_depth2d = torch.zeros((batch_size, num_anchor2d, 1), device=device)

        ref_pts2d = torch.masked_scatter(ref_pts2d, selected_mask[..., :2], select_centers)
        ref_depth2d = torch.masked_scatter(ref_depth2d, selected_mask[..., :1], select_depths.abs())

        ref_pts2d = ref_pts2d / ref_pts2d.new_tensor([img_w, img_h])

        # create trans matrix
        trans_matrix = nn.Parameter(
            torch.zeros((batch_size, num_anchor2d, num_anchor3d), device=device), requires_grad=False)

        meta_mask = trans_mask.to(torch.float) + center_valid.to(torch.float)
        meta_mask = meta_mask.permute(0, 2, 1)

        for bs in range(batch_size):
            cam_index, pts3d_index = torch.nonzero(meta_mask[bs]).chunk(2, dim=1)
            cam_index, pts3d_index = cam_index[:, 0], pts3d_index[:, 0]
            pts2d_index = torch.cat(
                [torch.arange(st, ed, device=device) for st, ed in zip(trans_start[bs], trans_end[bs])])
            trans_matrix[bs, pts2d_index, pts3d_index] = meta_mask[bs, cam_index, pts3d_index]

        center_matrix = (trans_matrix == 2).to(torch.float)
        trans_matrix = (trans_matrix >= 1).to(torch.float)  # include center and corner points

        return ref_pts2d, ref_depth2d, trans_mask, trans_shape, trans_matrix, center_matrix, query_groups, attn_mask

    def random_sample_corner_mask(self, corner_valid):
        batch_size, num_anchor3d, num_cams = corner_valid.shape
        corner_view = corner_valid.permute(0, 2, 1).reshape(batch_size * num_cams, -1)
        corner_nums = corner_view.sum(-1).detach().cpu().numpy()
        limit_nums = np.array(self.limit_corners_num * batch_size)

        remove_ids = np.where(corner_nums > limit_nums)[0]
        for remove_id in remove_ids:
            bs = remove_id // num_cams
            cam_id = remove_id % num_cams
            remove_num = max(corner_nums[remove_id] - limit_nums[remove_id], 0)
            remove_index = np.random.permutation(corner_nums[remove_id])[remove_num:]
            corner_valid[bs, :, cam_id][remove_index] = False

        return corner_valid