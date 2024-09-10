import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import HungarianAssigner
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.match_costs import build_match_cost


__all__ = ["SparseBox2DTarget", "MultiviewHungarianAssigner"]



@BBOX_SAMPLERS.register_module()
class SparseBox2DTarget(object):
    def __init__(self,
                 eps=1e-12,
                 cls_cost=None,
                 reg_cost=None,
                 iou_cost=None):
        super(SparseBox2DTarget, self).__init__()
        self.eps = eps
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)

    def sample(self,
               cls_pred,
               box_pred,
               cls_target,
               box_target,
               data = None,
               trans_matrix = None,
               query_groups = None,
               ):
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
        all_indices = []
        for i in range(bs):
            costs = []
            indices = []
            for j in range(len(query_groups)):
                if cls_cost[i][j] is not None and reg_cost[i][j] is not None and iou_cost[i][j] is not None:
                    cost = (cls_cost[i][j] + reg_cost[i][j] + iou_cost[i][j]).detach().cpu().numpy()
                    cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
                    costs.append(cost)
                    indices.append(
                        [
                            cls_pred.new_tensor(x, dtype=torch.int64)
                            for x in linear_sum_assignment(cost)
                        ]
                    )
                else:
                    costs.append(None)
                    indices.append(None)

            all_costs.append(costs)
            all_indices.append(indices)


        output_cls_target = cls_target[0][0].new_ones([bs, num_pred], dtype=torch.long) * -1
        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_reg_weights = box_pred.new_zeros(box_pred.shape)

        for i in range(bs):
            for j, qg in enumerate(query_groups):
                if len(cls_target[i][j]) > 0:
                    pred_idx, target_idx = all_indices[i][j]

                    output_cls_target[i, pred_idx + qg[0]] = cls_target[i][j][target_idx]
                    output_box_target[i, pred_idx + qg[0]] = box_target[i][j][target_idx]
                    output_reg_weights[i, pred_idx + qg[0]] = all_reg_weights[i][j][target_idx]

        # get cost2d_maps
        cost2d_maps = []
        for i in range(bs):
            num_gts = sum([len(x) for x in cls_target[i]])
            cost2d_map = np.ones((num_pred, num_gts)) * (-1/self.eps)

            if num_gts>0:
                gt_qg_shape = np.cumsum([0] + [len(x) for x in cls_target[i]])
                gt_qg_groups = [(qg[0], qg[1]) for qg in zip(gt_qg_shape[:-1], gt_qg_shape[1:])]

                for j, (qg, gp) in enumerate(zip(query_groups, gt_qg_groups)):
                    if all_costs[i][j] is not None:
                        cost2d_map[qg[0]:qg[1], gp[0]:gp[1]] = all_costs[i][j]

                cost2d_map[cost2d_map==(-1/self.eps)] = cost2d_map.max()

            cost2d_maps.append(cost2d_map)

        return output_cls_target, output_box_target, output_reg_weights, cost2d_maps



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


@BBOX_ASSIGNERS.register_module()
class MultiviewHungarianAssigner(HungarianAssigner):
    def __init__(self,
                 query_groups=[None],
                 cls_cost=dict(type='ClassificationCost', weight=1),
                 reg_cost=dict(type='BBoxL1Cost', weight=1),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1)):
        self.query_groups = query_groups
        super().__init__(cls_cost, reg_cost, iou_cost)

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               data,
               ref_trans_matrix=None,
               ref_query_groups=None,
               eps=1e-7):

        if ref_query_groups is not None and self.query_groups != ref_query_groups:
            self.query_groups = ref_query_groups

        num_gts = sum([cam_boxes.shape[0] for cam_boxes in gt_bboxes])
        num_bboxes = bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0

            assign_result = AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
            if ref_query_groups is not None:
                fake_cost2d = torch.ones((num_bboxes, num_gts)) * 1e-6
                return assign_result, fake_cost2d
            else:
                return assign_result
        img_w, img_h = map(int, data['image_wh'][0, 0].tolist())
        factor = gt_bboxes[0].new_tensor([img_w, img_h, img_w,
                                          img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        gt_num_list = torch.tensor([0] + [cam_boxes.shape[0] for cam_boxes in gt_bboxes], dtype=torch.long)
        gt_num_list = gt_num_list.cumsum(0)[:-1]

        cost_list = []
        matched_row_col_list = []
        for cam_idx, qg in enumerate(self.query_groups):
            if gt_bboxes[cam_idx].shape[0] != 0:
                cls_cost = self.cls_cost(cls_pred[qg[0]:qg[1]], gt_labels[cam_idx])
                cam_gt_bboxes = gt_bboxes[cam_idx][:, 0:4]
                # regression L1 cost
                normalize_gt_bboxes = cam_gt_bboxes /factor
                normalize_pred_bboxes = bbox_pred[qg[0]:qg[1]]
                reg_cost = self.reg_cost(normalize_pred_bboxes, normalize_gt_bboxes.to(bbox_pred.dtype))
                # regression iou cost, defaultly giou is used in official DETR.
                bboxes = bbox_cxcywh_to_xyxy(bbox_pred[qg[0]:qg[1]]) * factor
                iou_cost = self.iou_cost(bboxes, cam_gt_bboxes)
                # weighted sum of above three costs
                cost = cls_cost + reg_cost + iou_cost

                # 3. do Hungarian matching on CPU using linear_sum_assignment
                cost = cost.detach().cpu()
                cost_list.append(cost)
                if linear_sum_assignment is None:
                    raise ImportError('Please run "pip install scipy" '
                                      'to install scipy first.')
                matched_row_col_inds = linear_sum_assignment(cost)
                matched_row_col_list.append(matched_row_col_inds)
            else:
                cost_list.append(None)
                matched_row_col_list.append(None)

        # matched_row_inds = torch.from_numpy(matched_row_inds).to(
        #     bbox_pred.device)
        # matched_col_inds = torch.from_numpy(matched_col_inds).to(
        #     bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # Combine all camera gt_labels
        combine_gt_labels = torch.cat(gt_labels, dim=0)
        for i, r_c in enumerate(matched_row_col_list):
            if r_c is None:
                continue
            mt_row, mt_col = r_c
            mt_row = mt_row + self.query_groups[i][0]
            mt_col = mt_col + gt_num_list[i].item()
            mt_row = torch.from_numpy(mt_row).to(bbox_pred.device)
            mt_col = torch.from_numpy(mt_col).to(bbox_pred.device)
            # assign foregrounds based on matching results
            assigned_gt_inds[mt_row] = mt_col + 1
            assigned_labels[mt_row] = combine_gt_labels[mt_col]

        assign_result = AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        if ref_query_groups is not None:
            # ================================
            # cal cost2d for assisting 3d match
            cost2d = torch.ones((num_bboxes, num_gts)) * -999
            gt_shape = [len(gt_labels[i]) for i in range(len(gt_labels))]
            gt_shape.insert(0, 0)
            gt_start = np.cumsum(gt_shape)
            gt_groups = [(gs, ge) for gs, ge in zip(gt_start[:-1], gt_start[1:])]
            for cam_idx in range(len(self.query_groups)):
                (qs, qe), (gs, ge) = self.query_groups[cam_idx], gt_groups[cam_idx]
                if cost_list[cam_idx] is not None:
                    cost2d[qs: qe, gs: ge] = cost_list[cam_idx]

            cost2d[cost2d == -999] = cost2d.max()

            return assign_result, cost2d

        else:
            return assign_result