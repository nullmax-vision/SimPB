from typing import Optional

import torch

from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from projects.mmdet3d_plugin.core.box3d import *


@BBOX_CODERS.register_module()
class SparseBox3DDecoder(object):
    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
        sorted: bool = True,
    ):
        super(SparseBox3DDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.sorted = sorted

    def decode_box(self, box):
        yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
        box = torch.cat(
            [
                box[:, [X, Y, Z]],
                box[:, [W, L, H]].exp(),
                yaw[:, None],
                box[:, VX:],
            ],
            dim=-1,
        )
        return box

    def decode_box2d(self, box, aug_config):
        crop = aug_config['crop']
        scale_factor = aug_config['resize']

        crop_img_size = (crop[2] - crop[0], crop[3] - crop[1])

        box = bbox_cxcywh_to_xyxy(box)

        box[..., 0::2] = box[..., 0::2] * crop_img_size[0]
        box[..., 1::2] = box[..., 1::2] * crop_img_size[1]
        box[..., 0::2].clamp_(min=0, max=crop_img_size[0])
        box[..., 1::2].clamp_(min=0, max=crop_img_size[1])

        box[..., 1::2] += crop[1]
        box /= box.new_tensor(scale_factor)
        return box

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        qulity=None,
        output_idx=-1,
    ):
        squeeze_cls = instance_id is not None

        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        if not squeeze_cls:
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if qulity is not None:
            centerness = qulity[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if qulity is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = self.decode_box(box)
            output.append(
                {
                    "boxes_3d": box.cpu(),
                    "scores_3d": scores.cpu(),
                    "labels_3d": category_ids.cpu(),
                }
            )
            if qulity is not None:
                output[-1]["cls_scores"] = scores_origin.cpu()
            if instance_id is not None:
                ids = instance_id[i, indices[i]]
                if self.score_threshold is not None:
                    ids = ids[mask[i]]
                output[-1]["instance_ids"] = ids
        return output

    def decode_with2d(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        qulity=None,
        output_idx=-1,
        cls_scores2d=None,
        box_preds2d=None,
        trans_matrix=None,
        query_groups=None,
        output_idx2d=-1,
        aug_configs=None,
        with_association=False
    ):
        squeeze_cls = instance_id is not None

        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        if not squeeze_cls:
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if qulity is not None:
            centerness = qulity[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        cls_scores2d = cls_scores2d[output_idx2d]
        box_preds2d = box_preds2d[output_idx2d]
        trans_matrix_t = trans_matrix[output_idx2d].permute(0, 2, 1)
        query_groups = query_groups[output_idx2d]
        aug_config = aug_configs[0]

        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]

            # 2d
            assert num_cls == 1

            # get indices2d and trans_t
            if with_association:
                trans_t = trans_matrix_t[i, indices[i]]
                indices2d = torch.where(trans_t.any(0))[0]
                trans_t = torch.index_select(trans_t, 1, indices2d).cpu()
            else:
                indices2d = torch.arange(len(box_preds2d[i]))
                trans_t = None

            # get new query_groups
            camidx_2d = []
            query_groups_new = []
            for cam_idx, qg in enumerate(query_groups):
                parts_index = torch.where(torch.logical_and(qg[0] <= indices2d, indices2d < qg[1]))[0]

                if len(parts_index) > 0:
                    qg_new = (parts_index[0].cpu().item(), parts_index[-1].cpu().item() + 1)
                elif len(query_groups_new) > 0:
                    qg_new = (query_groups_new[-1][-1], query_groups_new[-1][-1])
                else:
                    qg_new = (0, 0)

                camidx_2d.append(torch.ones((len(parts_index))) * cam_idx)
                query_groups_new.append(qg_new)

            camidx_2d = torch.cat(camidx_2d, dim=0)
            query_groups = query_groups_new

            # resize box to original img
            scores2d, category_ids2d = cls_scores2d[i, indices2d].sigmoid().max(dim=-1)
            box2d = box_preds2d[i, indices2d]
            box2d = self.decode_box2d(box2d, aug_config)

            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if qulity is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = self.decode_box(box)

            output.append(
                {
                    # 3d
                    "boxes_3d": box.cpu(),
                    "scores_3d": scores.cpu(),
                    "labels_3d": category_ids.cpu(),
                    # 2d
                    "boxes_2d": box2d.cpu(),
                    "scores_2d": scores2d.cpu(),
                    "labels_2d": category_ids2d.cpu(),
                    "camidx_2d": camidx_2d,
                    "trans_matrix": trans_t,
                    "query_groups": query_groups
                }
            )
            if qulity is not None:
                output[-1]["cls_scores"] = scores_origin.cpu()
            if instance_id is not None:
                ids = instance_id[i, indices[i]]
                if self.score_threshold is not None:
                    ids = ids[mask[i]]
                output[-1]["instance_ids"] = ids
        return output
