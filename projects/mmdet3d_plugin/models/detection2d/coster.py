import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import HungarianAssigner
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.match_costs import build_match_cost

__all__ = ["SparseBox2DCoster"]


@BBOX_SAMPLERS.register_module()
class SparseBox2DCoster(object):
    def __init__(self,
                 eps=1e-12,
                 cls_cost=None,
                 reg_cost=None,
                 iou_cost=None):
        super(SparseBox2DCoster, self).__init__()
        self.eps = eps
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)


    def cost(self, cls_pred, box_pred, cls_target, box_target,
             data, trans_shape=None, query_groups=None):
        bs, num_pred, num_cls = cls_pred.shape
        img_w, img_h = map(int, data['image_wh'][0, 0].tolist())
        factor = box_pred[0].new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

        all_reg_weights = []
        for i in range(bs):
            reg_weights = []
            for j in range(len(query_groups)):
                weights = torch.logical_not(box_target[i][j].isnan()).to(dtype=box_target[i][j].dtype)
                reg_weights.append(weights)
            all_reg_weights.append(reg_weights)

        cls_cost = self._cls_cost(cls_pred, cls_target, query_groups)
        reg_cost = self._reg_cost(box_pred, box_target, query_groups, factor)
        iou_cost = self._iou_cost(box_pred, box_target, query_groups, factor)

        all_costs = []
        for i in range(bs):
            costs = []
            for j in range(len(query_groups)):
                if cls_cost[i][j] is not None and reg_cost[i][j] is not None and iou_cost[i][j] is not None:
                    cost = (cls_cost[i][j] + reg_cost[i][j] + iou_cost[i][j]).detach().cpu().numpy()
                    if cost.size > 0 and trans_shape is not None:
                        cost[trans_shape[i][j]:, :] = cost.max()
                    cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
                    costs.append(cost)
                else:
                    costs.append(None)
            all_costs.append(costs)

        return all_costs

    def trans_cost(self, all_costs, pred_cls_2d, gt_cls_2d, gt_cls_3d,
                   ref_trans_matrix, query_groups, gt_2d_3d_map):

        bs, num_query2d, num_cls2d = pred_cls_2d.shape
        num_query3d = ref_trans_matrix.shape[-1]

        cost2d_map3d_list = []
        for i in range(bs):
            num_target2d = sum([len(x) for x in gt_cls_2d[i]])
            num_target3d = len(gt_cls_3d[i])

            if num_target2d >0 and num_target3d>0:
                # 1. extend cost2d to large map
                cost2d_extend = np.ones((num_query2d, num_target2d), dtype=np.float32) * (-1 / self.eps)

                gt_qg_shape = np.cumsum([0] + [len(x) for x in gt_cls_2d[i]])
                gt_qg_groups = [(qg[0], qg[1]) for qg in zip(gt_qg_shape[:-1], gt_qg_shape[1:])]

                for j, (qg, gp) in enumerate(zip(query_groups, gt_qg_groups)):
                    if all_costs[i][j] is not None:
                        cost2d_extend[qg[0]:qg[1], gp[0]:gp[1]] = all_costs[i][j]

                if cost2d_extend.max() == (-1 / self.eps):
                    cost2d_extend = 0

                cost2d_extend[cost2d_extend == (-1 / self.eps)] = cost2d_extend.max()

                # 2. trans cost2d_extend to cost2d_map3d
                map_trans_matrix = torch.zeros((num_target2d, num_target3d))
                map_trans_matrix[torch.arange(num_target2d), torch.cat(gt_2d_3d_map[i]).cpu()] = 1
                ref_trans_matrix_t = ref_trans_matrix[i].cpu().T

                cost2d_extend = torch.from_numpy(cost2d_extend)
                cost2d_map3d = (cost2d_extend @ map_trans_matrix) / torch.clamp(map_trans_matrix.sum(0), 1e-5).unsqueeze(0)
                cost2d_map3d = (ref_trans_matrix_t @ cost2d_map3d) / torch.clamp(ref_trans_matrix_t.sum(-1), 1e-5).unsqueeze(-1)

                map_mask = torch.logical_or((cost2d_map3d.sum(0) == 0)[None, :],
                                            (cost2d_map3d.sum(1) == 0)[:, None])

                cost2d_map3d[map_mask] = cost2d_map3d.max()
                cost2d_map3d_list.append(cost2d_map3d.numpy())

            else:
                cost2d_map3d = np.zeros((num_query3d, num_target3d), dtype=np.float32)
                cost2d_map3d_list.append(cost2d_map3d)

        return cost2d_map3d_list


    def sample(self, cls_pred, box_pred, depth_pred, cls_target, box_target, depth_target,
               data, trans_matrix = None, query_groups = None, cost_list=None,
               alpha=None, alpha_targets=None):

        bs, num_pred, num_cls = cls_pred.shape

        all_reg_weights = []
        for i in range(bs):
            reg_weights = []
            for j in range(len(query_groups)):
                weights = torch.logical_not(box_target[i][j].isnan()).to(dtype=box_target[i][j].dtype)
                reg_weights.append(weights)
            all_reg_weights.append(reg_weights)


        all_indices = []
        for i in range(bs):
            indices = []
            for j in range(len(query_groups)):
                if cost_list[i][j] is not None and cost_list[i][j].size>0:
                    cost = cost_list[i][j]
                    cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
                    indices.append(
                        [
                            cls_pred.new_tensor(x, dtype=torch.int64)
                            for x in linear_sum_assignment(cost)
                        ]
                    )
                else:
                    indices.append(None)
            all_indices.append(indices)

        output_cls_target = cls_target[0][0].new_ones([bs, num_pred], dtype=torch.long) * -1
        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_reg_weights = box_pred.new_zeros(box_pred.shape)

        output_box_depth = None
        if depth_pred is not None:
            output_box_depth = box_pred.new_zeros(box_pred[..., 1].shape)

        output_alpha_target = None
        if alpha_targets is not None and alpha is not None:
            output_alpha_target = alpha.new_zeros(alpha.shape)

        for i in range(bs):
            for j, qg in enumerate(query_groups):
                if len(cls_target[i][j]) > 0 and all_indices[i][j] is not None:
                    pred_idx, target_idx = all_indices[i][j]

                    output_cls_target[i, pred_idx + qg[0]] = cls_target[i][j][target_idx]
                    output_box_target[i, pred_idx + qg[0]] = box_target[i][j][target_idx]
                    output_reg_weights[i, pred_idx + qg[0]] = all_reg_weights[i][j][target_idx]

                    if alpha_targets is not None:
                        assigned_alpha = alpha_targets[i][j][target_idx]
                        if len(assigned_alpha.shape) == 1: # for scalar alpha, not multibin
                            assigned_alpha_target = torch.stack((torch.sin(assigned_alpha), torch.cos(assigned_alpha))).transpose(1,0)#view(-1, 2)

                        if output_alpha_target is not None:
                            output_alpha_target[i, pred_idx + qg[0]] = assigned_alpha_target

                    if depth_pred is not None:
                        output_box_depth[i, pred_idx + qg[0]] = depth_target[i][j][target_idx]

        return output_cls_target, output_box_target, output_alpha_target, output_box_depth, output_reg_weights


    def _cls_cost(self, cls_pred, cls_target, query_groups):
        bs = cls_pred.shape[0]
        all_cost = []
        for i in range(bs):
            cost = []
            for j, qg in enumerate(query_groups):
                if len(cls_target[i][j]) > 0:
                    cls_cost = self.cls_cost(cls_pred[i][qg[0]:qg[1]],
                                             cls_target[i][j])
                    cost.append(cls_cost)
                else:
                    cost.append(None)
            all_cost.append(cost)
        return all_cost


    def _reg_cost(self, box_pred, box_target, query_groups, factor):
        bs = box_pred.shape[0]
        all_cost = []
        for i in range(bs):
            cost = []
            for j, qg in enumerate(query_groups):
                if len(box_target[i][j]) > 0:
                    # normalized resolution
                    reg_cost = self.reg_cost(box_pred[i][qg[0]:qg[1]],
                                             box_target[i][j][:, 0:4] / factor)
                    cost.append(reg_cost)
                else:
                    cost.append(None)
            all_cost.append(cost)
        return all_cost


    def _iou_cost(self, box_pred, box_target, query_groups, factor):
        bs = box_pred.shape[0]
        all_cost = []
        for i in range(bs):
            cost = []
            for j, qg in enumerate(query_groups):
                if len(box_target[i][j])>0:
                    # original resolution
                    iou_cost = self.iou_cost(bbox_cxcywh_to_xyxy(box_pred[i][qg[0]:qg[1]]) * factor,
                                             box_target[i][j][:, 0:4])
                    cost.append(iou_cost)
                else:
                    cost.append(None)
            all_cost.append(cost)
        return all_cost

