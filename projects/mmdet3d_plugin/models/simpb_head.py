from typing import List, Optional, Tuple, Union
import warnings
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
    TRANSFORMER_LAYER_SEQUENCE
)
from mmcv.utils import build_from_cfg
from mmcv.runner import BaseModule, force_fp32
from mmdet.core import reduce_mean
from mmdet.models import HEADS, LOSSES
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

from .utils import get_valid_ratio, get_reference_points, pos2posemb2d


__all__ = ["SimPBHead"]


@HEADS.register_module()
class SimPBHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        num_cams: int = 6,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        init_cfg: dict = None,
        enable2d=False,
        enable3d=True,
        embed_dims=256,
        num_levels=4,
        num_anchor=900,
        encoder2d=None,
        share_encoder2d=False,
        anchor_encoder2d=None,
        positional_encoding=None,
        qg_self_attn=None,
        qg_cross_attn=None,
        refine_layer2d=None,
        refine_layer3d=None,
        decouple_attn2d=False,
        with_allocate_attn_mask=False,
        dynamic_allocation=None,
        adaptive_aggregation=None,
        coster2d=None,
        coster3d=None,
        denoise2d=None,
        loss_cls2d=None,
        loss_iou2d=None,
        loss_bbox2d=None,
        loss_alpha2d=None,
        loss_depth2d=None,
        **kwargs,
    ):
        super(SimPBHead, self).__init__(init_cfg)
        self.enable2d = enable2d
        self.enable3d = enable3d
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.num_anchor = num_anchor
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn
        self.decouple_attn2d = decouple_attn2d
        self.cls_threshold_to_reg = cls_threshold_to_reg

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine3d",
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        if self.enable2d:

            if encoder2d:
                self.encoder2d = build(encoder2d, TRANSFORMER_LAYER_SEQUENCE)
                self.positional_encoding = build(positional_encoding, POSITIONAL_ENCODING)
                self.level_embeds = nn.Parameter(torch.Tensor(self.num_levels, self.embed_dims))
            else:
                self.encoder2d = None

            if anchor_encoder2d is not None:
                self.anchor_encoder2d = build(anchor_encoder2d, POSITIONAL_ENCODING)
            else:
                self.query_embeddings2d = nn.Sequential(
                    nn.Linear(in_features=self.embed_dims, out_features=self.embed_dims, bias=True),
                    nn.ReLU(),
                    nn.Linear(in_features=self.embed_dims, out_features=self.embed_dims, bias=True),
                )
                self.anchor_encoder2d = lambda x: self.query_embeddings2d(pos2posemb2d(x))

            self.instance_status = '3d'

            self.share_encoder2d = share_encoder2d
            self.with_allocate_attn_mask = with_allocate_attn_mask

            self.loss_cls2d = build(loss_cls2d, LOSSES)
            self.loss_iou2d = build(loss_iou2d, LOSSES)
            self.loss_bbox2d = build(loss_bbox2d, LOSSES)
            self.loss_alpha2d = build(loss_alpha2d, LOSSES)
            self.loss_depth2d = build(loss_depth2d, LOSSES)

            if denoise2d is not None:
                self.with_denoise2d = True
                self.denoise2d = build(denoise2d, PLUGIN_LAYERS)

            self.coster2d = build(coster2d, BBOX_SAMPLERS)

        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)
        self.op_config_map = {
            # common
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "norm": [norm_layer, NORM_LAYERS],
            # 2d
            "allocation": [dynamic_allocation, PLUGIN_LAYERS],
            "aggregation": [adaptive_aggregation, PLUGIN_LAYERS],
            "qg_self_attn": [qg_self_attn, ATTENTION],
            "qg_cross_attn": [qg_cross_attn, ATTENTION],
            "refine2d": [refine_layer2d, PLUGIN_LAYERS],
            # 3d
            "gnn": [graph_model, ATTENTION],
            "temp_gnn": [temp_graph_model, ATTENTION],
            "deformable": [deformable_model, ATTENTION],
            "refine3d": [refine_layer3d, PLUGIN_LAYERS],
        }

        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )

        if self.decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False) if self.enable3d else None
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        if self.decouple_attn2d and self.enable2d:
            self.fc_before2d = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            self.fc_after2d = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)
        else:
            self.fc_before2d = nn.Identity()
            self.fc_after2d = nn.Identity()

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine2d" or op != "refine3d":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def prepare2d(self, feature_maps, metas):
        if self.encoder2d is not None:
            if self.use_deformable_func:
                bs, _, dim = feature_maps[0].shape
                nc = len(feature_maps[1])
                spatial_shape = feature_maps[1][0]
                split_size = (spatial_shape[:, 0] * spatial_shape[:, 1]).tolist()
                feature_maps = list(torch.split(feature_maps[0].reshape(bs, nc, -1, dim).permute(0, 1, 3, 2), split_size, dim=-1))
                for i, feat in enumerate(feature_maps):
                    feature_maps[i] = feat.reshape(feat.shape[:3] + (spatial_shape[i, 0], spatial_shape[i, 1]))

            mlvl_feats = feature_maps
            mlvl_feats = [x.flatten(0, 1) for x in mlvl_feats]

            batch_size = mlvl_feats[0].size(0)  # BN,C,H,W
            img_w, img_h = map(int, metas['image_wh'][0, 0].tolist())
            img_masks = mlvl_feats[0].new_zeros((batch_size, img_w, img_h))

            mlvl_masks = [F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0) for feat in
                          mlvl_feats]
            mlvl_pos_embeds = [self.positional_encoding(mlvl_masks[i]) for i in range(len(mlvl_feats))]

            feat_flatten = []
            mask_flatten = []
            spatial_shapes = []
            lvl_pos_embed_flatten = []

            for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
                bs, c, h, w = feat.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                feat = feat.flatten(2).transpose(1, 2)
                mask = mask.flatten(1)
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
                feat_flatten.append(feat)
                mask_flatten.append(mask)

            feat_flatten = torch.cat(feat_flatten, 1)
            mask_flatten = torch.cat(mask_flatten, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack([get_valid_ratio(m) for m in mlvl_masks], 1)

            reference_points = get_reference_points(spatial_shapes, valid_ratios, device=feat.device)

            memory = self.encoder2d(
                query=feat_flatten,
                key=None,
                value=None,
                query_pos=lvl_pos_embed_flatten,
                query_key_padding_mask=mask_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
            )

            encoder2d_dict = {
                'value': memory,
                'key_padding_mask': mask_flatten,
                'spatial_shapes': spatial_shapes,
                'level_start_index': level_start_index,
            }
        else:
            if self.use_deformable_func:
                bs, _, dim = feature_maps[0].shape
                nc = len(feature_maps[1])
                feat_flatten = feature_maps[0].reshape(bs, nc, -1, dim).flatten(0, 1)
                mask_flatten = feat_flatten.new_zeros(feat_flatten.shape[:2]).to(torch.bool)
                encoder2d_dict = {
                    'value': feat_flatten,
                    'key_padding_mask': mask_flatten,
                    'spatial_shapes': feature_maps[1][0].long(),
                    'level_start_index': feature_maps[2][0].long(),
                }
            else:
                raise RuntimeError

        return encoder2d_dict

    def graph_model(self, index, query, key=None, value=None, query_pos=None, key_pos=None, **kwargs):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            key = torch.cat([key, key_pos], dim=-1) if key is not None else None
            query_pos, key_pos = None, None
        value = self.fc_before(value) if value is not None else None

        if isinstance(index, int):
            output = self.layers[index](query, key, value, query_pos=query_pos, key_pos=key_pos, **kwargs)
        else:
            output = index(query, key, value, query_pos=query_pos, key_pos=key_pos, **kwargs)

        return self.fc_after(output)

    def graph_model2d(self, index, query, key=None, value=None, query_pos=None, key_pos=None, **kwargs):
        if self.decouple_attn2d:
            query = torch.cat([query, query_pos], dim=-1)
            key = torch.cat([key, key_pos], dim=-1) if key is not None else None
            query_pos, key_pos = None, None
        value = self.fc_before2d(value) if value is not None else None

        output = self.layers[index](query, key, value, query_pos=query_pos, key_pos=key_pos, **kwargs)

        return self.fc_after2d(output)

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # ========= get instance info ============
        if self.sampler.dn_metas is not None and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size:
            self.sampler.dn_metas = None

        instance_feature, anchor, \
        temp_instance_feature, temp_anchor, time_interval \
            = self.instance_bank.get(batch_size, metas, dn_metas=self.sampler.dn_metas)

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            if "instance_id" in metas["img_metas"][0]:
                gt_instance_id = [torch.from_numpy(x["instance_id"]).cuda() for x in metas["img_metas"]]
            else:
                gt_instance_id = None

            dn_metas = self.sampler.get_dn_anchors(
                metas["gt_labels_3d"], metas["gt_bboxes_3d"], gt_instance_id, metas=metas if self.with_denoise2d else None,
            )

        if dn_metas is not None:
            if not self.with_denoise2d:
                dn_anchor, dn_reg_target, dn_cls_target, \
                    dn_attn_mask, valid_mask, dn_id_target = dn_metas
            else:
                dn_anchor, dn_reg_target, dn_cls_target, dn_attn_mask, valid_mask, dn_id_target, dn_trans_mask2d, \
                    dn_valid_mask2d, dn_cls_target2d, dn_box_target2d, dn_alpha_target2d, dn_depth_target2d = dn_metas

            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [dn_anchor, dn_anchor.new_zeros(batch_size, num_dn_anchor, remain_state_dims)], dim=-1)

            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [instance_feature, instance_feature.new_zeros(batch_size, num_dn_anchor, instance_feature.shape[-1])], dim=1)

            num_instance = instance_feature.shape[1]
            num_anchor3d = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones((num_instance, num_instance), dtype=torch.bool)
            attn_mask[:num_anchor3d, :num_anchor3d] = False
            attn_mask[num_anchor3d:, num_anchor3d:] = dn_attn_mask
        else:
            num_anchor3d = instance_feature.shape[1]

        anchor_embed = self.anchor_encoder(anchor)
        temp_anchor_embed = self.anchor_encoder(temp_anchor) if temp_anchor is not None else None

        # =================== forward the layers ====================
        quality = []
        prediction = []
        classification = []

        prediction2d = []
        classification2d = []
        prediction_alpha2d = []
        prediction_depth2d = []

        ref_pts2d_list = []
        ref_trans_shape_list = []
        ref_trans_matrix_list = []
        ref_query_groups_list = []

        num_query2d_list = []
        num_dn_query2d_list = []

        if self.with_denoise2d:
            dn_trans_mask2d_list = []
            dn_valid_mask2d_list = []
            dn_cls_target2d_list = []
            dn_box_target2d_list = []
            dn_alpha_target2d_list = []
            dn_depth_target2d_list = []

        temp_attn_instance = instance_feature

        if self.enable2d:
            encoder2d_dict = self.prepare2d(feature_maps, metas)
            if self.share_encoder2d:
                feature_maps[0] = encoder2d_dict['value'].view(batch_size, -1, self.embed_dims)

        for i, op in enumerate(self.operation_order):
            # common op
            if self.layers[i] is None:
                continue
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)

            # 2D
            elif op == "allocation":
                assert self.instance_status == '3d'
                if dn_metas is not None:
                    dn_instance_feature = instance_feature[:, num_anchor3d:]
                    instance_feature = instance_feature[:, :num_anchor3d]
                    dn_anchor = anchor[:, num_anchor3d:]
                    anchor = anchor[:, :num_anchor3d]

                anchor2d, ref_depth2d, ref_trans_mask, ref_trans_shape, ref_trans_matrix, \
                    ref_center_matrix, ref_query_groups, allocate_attn_mask = self.layers[i](anchor, metas)

                instance_feature = torch.matmul(ref_trans_matrix, instance_feature)

                num_anchor2d = anchor2d.size(1)

                # with denoise2d
                if self.with_denoise2d and dn_metas is not None:
                    (dn_anchor2d, dn_depth2d, dn_trans_mask2d, dn_trans_shape, dn_trans_matrix,
                     dn_center_matrix, dn_query_groups, dn_allocate_attn_mask, dn_valid_lvl_mask2d,
                     dn_cls_lvl_target2d, dn_box_lvl_target2d, dn_alpha_lvl_target2d, dn_depth_lvl_target2d) = \
                        self.denoise2d.get_dn_project(dn_anchor, metas, dn_trans_mask2d, dn_valid_mask2d,
                                                      dn_cls_target2d, dn_box_target2d, dn_alpha_target2d, dn_depth_target2d)

                    dn_instance_feature = torch.matmul(dn_trans_matrix, dn_instance_feature)

                    num_dn_anchor2d = dn_anchor2d.size(1)

                    num_query2d_list.append(num_anchor2d)
                    num_dn_query2d_list.append(num_dn_anchor2d)

                    self_dn_query_groups = self.denoise2d.get_self_dn_query_groups(ref_query_groups, dn_query_groups)
                    cross_dn_query_groups = self.denoise2d.get_cross_dn_query_groups(ref_query_groups, dn_query_groups)

                    if self.with_allocate_attn_mask:
                        meta_num_query2d = num_anchor2d + num_dn_anchor2d
                        meta_attn_mask = anchor2d.new_ones((batch_size, meta_num_query2d, meta_num_query2d)).fill_(float("-inf"))
                        meta_attn_mask[:, :num_anchor2d, :num_anchor2d] = allocate_attn_mask
                        meta_attn_mask[:, num_anchor2d:, num_anchor2d:] = dn_allocate_attn_mask
                        allocate_attn_mask = meta_attn_mask

                    instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)
                    anchor2d = torch.cat([anchor2d, dn_anchor2d], dim=1)

                anchor_embed2d = self.anchor_encoder2d(anchor2d)

                if not self.training:  # just for visualizing reference points
                    ref_pts2d_list.append(anchor2d[..., :2])

                self.instance_status = '2d'

            elif op == "aggregation":
                assert self.instance_status == '2d'
                instance_feature, anchor_embed, anchor = self.layers[i](
                    # 2d
                    query2d=instance_feature[:, :num_anchor2d],
                    query_pos2d=anchor_embed2d[:, :num_anchor2d],
                    anchor2d=anchor2d[:, :num_anchor2d],
                    dn_query2d=instance_feature[:, num_anchor2d:] if self.with_denoise2d and dn_metas is not None else None,
                    dn_query_pos2d=anchor_embed2d[:, num_anchor2d:] if self.with_denoise2d and dn_metas is not None else None,
                    dn_anchor2d=anchor2d[:, num_anchor2d:] if self.with_denoise2d and dn_metas is not None else None,
                    # 3d
                    query3d=temp_attn_instance[:, :num_anchor3d],
                    query_pos3d=anchor_embed[:, :num_anchor3d],
                    anchor3d=anchor,
                    dn_query3d=temp_attn_instance[:, num_anchor3d:] if dn_metas is not None else None,
                    dn_query_pos3d=anchor_embed[:, num_anchor3d:] if dn_metas is not None else None,
                    dn_anchor3d=dn_anchor if dn_metas is not None else None,
                    # 2d-to-3d matrix
                    trans_matrix=ref_trans_matrix,
                    center_matrix=ref_center_matrix,
                    dn_trans_matrix=dn_trans_matrix if self.with_denoise2d and dn_metas is not None else None,
                    dn_center_matrix=dn_center_matrix if self.with_denoise2d and dn_metas is not None else None,
                    # 3d attn-mask
                    attn_mask=attn_mask,
                    graph_model=self.graph_model,
                )
                self.instance_status = '3d'

            elif op == "qg_self_attn":
                instance_feature = self.graph_model2d(
                    i,
                    query=instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed2d,
                    query_groups=self_dn_query_groups if self.with_denoise2d and dn_metas is not None else ref_query_groups,
                    group_attn_mask=allocate_attn_mask if self.with_allocate_attn_mask else None
                )

            elif op == 'qg_cross_attn':
                if self.with_denoise2d and dn_metas is not None:
                    instance_feature, anchor2d, anchor_embed2d = self.denoise2d.permute_instance(
                        instance_feature, anchor2d, anchor_embed2d, self_dn_query_groups)

                instance_feature = self.layers[i](
                    query=instance_feature,
                    query_pos=anchor_embed2d,
                    reference_points=anchor2d.unsqueeze(2),
                    query_groups=cross_dn_query_groups if self.with_denoise2d and dn_metas is not None else ref_query_groups,
                    **encoder2d_dict)

                if self.with_denoise2d and dn_metas is not None:
                    instance_feature, anchor2d, anchor_embed2d = self.denoise2d.depermute_instance(
                        instance_feature, anchor2d, anchor_embed2d, ref_query_groups, dn_query_groups)

            elif op == 'refine2d':
                anchor2d, cls2d, depth2d, alpha2d = self.layers[i](
                    instance_feature,
                    anchor2d,
                    anchor_embed2d,
                    metas=metas,
                    query_groups=ref_query_groups,
                )

                prediction2d.append(anchor2d)
                classification2d.append(cls2d)
                prediction_alpha2d.append(alpha2d)
                prediction_depth2d.append(depth2d)

                ref_trans_shape_list.append(ref_trans_shape)
                ref_trans_matrix_list.append(ref_trans_matrix)
                ref_query_groups_list.append(ref_query_groups)

                if self.training and self.with_denoise2d and dn_metas is not None:
                    dn_trans_mask2d_list.append(dn_trans_mask2d)
                    dn_valid_mask2d_list.append(dn_valid_lvl_mask2d)
                    dn_cls_target2d_list.append(dn_cls_lvl_target2d)
                    dn_box_target2d_list.append(dn_box_lvl_target2d)
                    dn_alpha_target2d_list.append(dn_alpha_lvl_target2d)
                    dn_depth_target2d_list.append(dn_depth_lvl_target2d)

            # 3D
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )

            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask
                    if temp_instance_feature is None
                    else None,
                )
                temp_attn_instance = instance_feature

            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )

            elif op == "refine3d":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(instance_feature, anchor, cls)
                    # temporal denoise
                    if (dn_metas is not None and self.sampler.num_temp_dn_groups > 0 and dn_id_target is not None):
                        if not self.with_denoise2d:
                            instance_feature, anchor, temp_dn_reg_target, temp_dn_cls_target, temp_valid_mask, \
                                dn_id_target = self.sampler.update_dn(
                                instance_feature, anchor, dn_reg_target, dn_cls_target, valid_mask, dn_id_target,
                                num_anchor3d, self.instance_bank.mask)
                        else:
                            instance_feature, anchor, temp_dn_reg_target, temp_dn_cls_target, temp_valid_mask, \
                            dn_id_target, dn_trans_mask2d, dn_valid_mask2d, dn_cls_target2d, dn_box_target2d,  \
                            dn_alpha_target2d, dn_depth_target2d = self.sampler.update_dn(
                                instance_feature, anchor, dn_reg_target, dn_cls_target, valid_mask, dn_id_target,
                                num_anchor3d, self.instance_bank.mask, dn_trans_mask2d, dn_valid_mask2d,
                                dn_cls_target2d, dn_box_target2d, dn_alpha_target2d, dn_depth_target2d)
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if len(prediction) > self.num_single_frame_decoder and temp_anchor_embed is not None:
                    temp_anchor_embed = anchor_embed[:, : self.instance_bank.num_temp_instances]

            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            if self.with_denoise2d:
                dn_classification2d = [x[:, n:] for x, n in zip(classification2d, num_query2d_list)]
                classification2d = [x[:, :n] for x, n in zip(classification2d, num_query2d_list)]
                dn_prediction2d = [x[:, n:] for x, n in zip(prediction2d, num_query2d_list)]
                prediction2d = [x[:, :n] for x, n in zip(prediction2d, num_query2d_list)]
                if self.loss_alpha2d is not None:
                    dn_prediction_alpha2d = [x[:, n:] for x, n in zip(prediction_alpha2d, num_query2d_list)]
                    prediction_alpha2d = [x[:, :n] for x, n in zip(prediction_alpha2d, num_query2d_list)]
                else:
                    dn_prediction_alpha2d = prediction_alpha2d

                if self.loss_depth2d is not None:
                    dn_prediction_depth2d = [x[:, n:] for x, n in zip(prediction_depth2d, num_query2d_list)]
                    prediction_depth2d = [x[:, :n] for x, n in zip(prediction_depth2d, num_query2d_list)]
                else:
                    dn_prediction_depth2d = prediction_depth2d


                output.update(
                    {
                        "dn_classification2d": dn_classification2d,
                        "dn_prediction2d": dn_prediction2d,
                        "dn_prediction_alpha2d": dn_prediction_alpha2d,
                        "dn_prediction_depth2d": dn_prediction_depth2d,

                        "dn_valid_mask2d_list": dn_valid_mask2d_list,
                        "dn_cls_target2d_list": dn_cls_target2d_list,
                        "dn_box_target2d_list": dn_box_target2d_list,
                        "dn_alpha_target2d_list": dn_alpha_target2d_list,
                        "dn_depth_target2d_list": dn_depth_target2d_list,
                    }
                )

            dn_classification = [x[:, num_anchor3d:] for x in classification]
            classification = [x[:, :num_anchor3d] for x in classification]
            dn_prediction = [x[:, num_anchor3d:] for x in prediction]
            prediction = [x[:, :num_anchor3d] for x in prediction]
            quality = [x[:, :num_anchor3d] if x is not None else None for x in quality]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_anchor3d:]
            dn_anchor = anchor[:, num_anchor3d:]
            instance_feature = instance_feature[:, :num_anchor3d]
            anchor = anchor[:, :num_anchor3d]
            cls = cls[:, :num_anchor3d]

            # cache dn_metas for temporal denoising
            if not self.with_denoise2d:
                self.sampler.cache_dn(
                    dn_instance_feature,
                    dn_anchor,
                    dn_cls_target,
                    valid_mask,
                    dn_id_target,
                )
            else:
                self.sampler.cache_dn(
                    dn_instance_feature,
                    dn_anchor,
                    dn_cls_target,
                    valid_mask,
                    dn_id_target,
                    dn_trans_mask2d,
                    dn_valid_mask2d,
                    dn_cls_target2d,
                    dn_alpha_target2d,
                    dn_depth_target2d,
                )

        output.update(
            {
                "quality": quality,
                "prediction": prediction,
                "classification": classification,

                "prediction2d": prediction2d,
                "classification2d": classification2d,
                "prediction_alpha2d": prediction_alpha2d,
                "prediction_depth2d": prediction_depth2d,

                "ref_pts2d_list": ref_pts2d_list,
                "ref_trans_shape_list": ref_trans_shape_list,
                "ref_trans_matrix_list": ref_trans_matrix_list,
                "ref_query_groups_list": ref_query_groups_list,
            }
        )

        # cache current instances for temporal modeling
        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        if not self.training:
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id
        return output

    def loss(self, model_outs, data):
        losses = {}
        loss3d = self.get_loss3d(model_outs, data)
        losses.update(loss3d)

        if hasattr(self, 'coster2d'):
            cost2d_list = self.get_cost2d(model_outs, data)
            loss2d = self.get_loss2d(model_outs, data, cost2d_list)
            losses.update(loss2d)

        loss_dn = self.get_dn_loss(model_outs, data)
        losses.update(loss_dn)

        return losses

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(end_dim=1)[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(end_dim=1)[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(dn_reg_target.shape[0], 1)
        # num_dn_pos = max(reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)), 1.0,)
        num_dn_pos = max(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype), 1.0,)

        return dn_valid_mask, dn_cls_target, dn_reg_target, dn_pos_mask, reg_weights, num_dn_pos

    def get_cost2d(self, outputs, data):
        if not hasattr(self, 'coster2d'):
            return None

        reg_preds = outputs['prediction2d']
        cls_scores = outputs['classification2d']
        reg_targets = data['gt_bboxes_2d']
        cls_targets = data['gt_labels_2d']

        trans_shape_list = outputs['ref_trans_shape_list']
        query_groups_list = outputs['ref_query_groups_list']

        cost2d_list = []
        for decoder_idx, (cls, reg, trans_shape, query_groups) in enumerate(
                zip(cls_scores, reg_preds, trans_shape_list, query_groups_list)):
            cost2d_map = self.coster2d.cost(cls, reg, cls_targets, reg_targets,
                                            data, trans_shape, query_groups)
            cost2d_list.append(cost2d_map)

        return cost2d_list

    @force_fp32(apply_to=("model_outs"))
    def get_loss2d(self, model_outs, data, cost_list=None):
        if cost_list is None or len(cost_list) == 0:
            return dict()
        reg_preds = model_outs['prediction2d']
        cls_scores = model_outs['classification2d']
        alpha_preds = model_outs['prediction_alpha2d']
        depth_preds = model_outs['prediction_depth2d']

        cls_targets = data['gt_labels_2d']
        reg_targets = data['gt_bboxes_2d']
        alpha_targets = data['gt_alphas_2d'] if 'gt_alphas_2d' in data else None
        depth_targets = data['gt_depths_2d'] if 'gt_depths_2d' in data else None

        trans_shape_list = model_outs['ref_trans_shape_list']
        trans_matrix_list = model_outs['ref_trans_matrix_list']
        query_groups_list = model_outs['ref_query_groups_list']

        img_w, img_h = map(int, data['image_wh'][0, 0].tolist())
        factor = reg_preds[0][0].new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

        loss_out = {}
        for decoder_idx, (cls, reg, alpha, depth, trans_matrix, query_groups, trans_shape) in enumerate(
                zip(cls_scores, reg_preds, alpha_preds, depth_preds, trans_matrix_list, query_groups_list, trans_shape_list)):

            cls_target, reg_target, alpha_target, depth_target, reg_weights = self.coster2d.sample(
                cls, reg, depth, cls_targets, reg_targets, depth_targets,
                data, trans_matrix, query_groups, cost_list[decoder_idx], alpha, alpha_targets)

            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))

            # remove none-query3d
            qg_mask = torch.zeros_like(mask)
            for bs in range(qg_mask.shape[0]):
                for cam_idx, qg in enumerate(query_groups):
                    qg_mask[bs, qg[0]:qg[0] + trans_shape[bs][cam_idx]] = 1
            mask = torch.logical_and(mask, qg_mask.to(torch.bool))

            num_pos = max(reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls2d(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg = reg.flatten(end_dim=1)[mask]
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]

            iou_loss = self.loss_iou2d(
                bbox_cxcywh_to_xyxy(reg) * factor, reg_target, weight=reg_weights, avg_factor=num_pos
            )

            box_loss = self.loss_bbox2d(
                reg, bbox_xyxy_to_cxcywh(reg_target) / factor, weight=reg_weights, avg_factor=num_pos
            )

            loss_out.update(
                {
                    f"loss_cls2d_{decoder_idx}": cls_loss,
                    f"loss_iou2d_{decoder_idx}": iou_loss,
                    f"loss_box2d_{decoder_idx}": box_loss,
                }
            )

            if self.loss_alpha2d is not None:
                alpha = alpha.flatten(end_dim=1)[mask]
                alpha_target = alpha_target.flatten(end_dim=1)[mask]

                alpha_loss = self.loss_alpha2d(alpha, alpha_target, weight=reg_weights[:, :2], avg_factor=num_pos)

                loss_out.update(
                    {
                        f"loss_alpha2d_{decoder_idx}": alpha_loss,
                    }
                )

            if self.loss_depth2d is not None:
                depth = depth.flatten(end_dim=1)[mask].squeeze(1)
                depth_target = depth_target.flatten(end_dim=1)[mask]

                depth_loss = self.loss_depth2d(depth, depth_target, weight=reg_weights[:, 0], avg_factor=num_pos)

                loss_out.update(
                    {
                        f"loss_depth2d_{decoder_idx}": depth_loss
                    }
                )

        return loss_out

    @force_fp32(apply_to=("model_outs"))
    def get_loss3d(self, model_outs, data):
        quality = model_outs["quality"]
        reg_preds = model_outs["prediction"]
        cls_scores = model_outs["classification"]

        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(zip(cls_scores, reg_preds, quality)):

            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(cls, reg, data["gt_labels_3d"], data["gt_bboxes_3d"])

            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0)

            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(mask, cls.max(dim=-1).values.sigmoid() > threshold)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]

            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(reg_target.isnan(), reg.new_tensor(0.0), reg_target)

            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"3d_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"loss_cls3d_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        return output

    @force_fp32(apply_to=("model_outs"))
    def get_dn_loss(self, model_outs, data):
        output = dict()
        # 3d
        if "dn_prediction" in model_outs:
            dn_reg_preds = model_outs["dn_prediction"]
            dn_cls_scores = model_outs["dn_classification"]

            dn_valid_mask, dn_cls_target, dn_reg_target, \
                dn_pos_mask, reg_weights, num_dn_pos = self.prepare_for_dn_loss(model_outs)

            for decoder_idx, (cls, reg) in enumerate(zip(dn_cls_scores, dn_reg_preds)):
                if "temp_dn_valid_mask" in model_outs and decoder_idx == self.num_single_frame_decoder:
                    dn_valid_mask, dn_cls_target, dn_reg_target, \
                        dn_pos_mask, reg_weights, num_dn_pos = self.prepare_for_dn_loss(model_outs, prefix="temp_")

                cls_loss = self.loss_cls(cls.flatten(end_dim=1)[dn_valid_mask],
                                         dn_cls_target,
                                         avg_factor=num_dn_pos)

                reg_loss = self.loss_reg(reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][..., : len(self.reg_weights)],
                                         dn_reg_target,
                                         avg_factor=num_dn_pos,
                                         weight=reg_weights,
                                         suffix=f"3d_dn_{decoder_idx}")

                output.update({f"loss_cls3d_dn_{decoder_idx}": cls_loss})
                output.update(reg_loss)

        elif self.sampler.num_dn_groups > 0:
            for decoder_idx, cls in enumerate(model_outs["classification"]):
                cls_loss = cls.new_tensor(0., requires_grad=False)
                reg_loss = {f"loss_box3d_dn_{decoder_idx}": cls.new_tensor(0., requires_grad=False)}

                output.update({f"loss_cls3d_dn_{decoder_idx}": cls_loss})
                output.update(reg_loss)

        # 2d
        if self.with_denoise2d and self.sampler.num_dn_groups > 0:
            if "dn_prediction2d" in model_outs:
                dn_box_preds2d = model_outs["dn_prediction2d"]
                dn_cls_preds2d = model_outs["dn_classification2d"]
                dn_alpha_preds2d = model_outs["dn_prediction_alpha2d"]
                dn_depth_preds2d = model_outs["dn_prediction_depth2d"]

                dn_valid_mask2d_list = model_outs["dn_valid_mask2d_list"]
                dn_cls_target2d_list = model_outs["dn_cls_target2d_list"]
                dn_box_target2d_list = model_outs["dn_box_target2d_list"]
                dn_alpha_target2d_list = model_outs["dn_alpha_target2d_list"]
                dn_depth_target2d_list = model_outs["dn_depth_target2d_list"]

                img_w, img_h = map(int, data['image_wh'][0, 0].tolist())
                factor = dn_box_preds2d[0][0].new_tensor([img_w, img_h, img_w, img_h])

                for decoder_idx, (dn_cls_pred, dn_box_pred, dn_alpha_pred, dn_depth_pred, dn_cls_target2d,
                                  dn_box_target2d, dn_alpha_target, dn_depth_target, dn_valid_mask2d) in enumerate(
                    zip(dn_cls_preds2d, dn_box_preds2d, dn_alpha_preds2d, dn_depth_preds2d, dn_cls_target2d_list,
                        dn_box_target2d_list, dn_alpha_target2d_list, dn_depth_target2d_list, dn_valid_mask2d_list)):

                    num_pos = max(torch.sum(dn_valid_mask2d).to(dtype=reg.dtype), 1.0)

                    dn_cls_pred = dn_cls_pred[dn_valid_mask2d]
                    dn_cls_target = dn_cls_target2d[dn_valid_mask2d]
                    dn_cls_loss2d = self.loss_cls2d(dn_cls_pred, dn_cls_target)

                    dn_valid_mask2d = torch.logical_and(dn_valid_mask2d, dn_cls_target2d >= 0)
                    dn_box_pred = dn_box_pred[dn_valid_mask2d]
                    dn_box_target = dn_box_target2d[dn_valid_mask2d]
                    dn_box_weights = torch.ones_like(dn_box_pred)

                    dn_iou_loss2d = self.loss_iou2d(
                        bbox_cxcywh_to_xyxy(dn_box_pred) * factor, dn_box_target, weight=dn_box_weights, avg_factor=num_pos
                    )

                    dn_box_loss2d = self.loss_bbox2d(
                        dn_box_pred, bbox_xyxy_to_cxcywh(dn_box_target) / factor, weight=dn_box_weights, avg_factor=num_pos
                    )

                    output.update(
                        {
                            f"loss_cls2d_dn_{decoder_idx}": dn_cls_loss2d,
                            f"loss_iou2d_dn_{decoder_idx}": dn_iou_loss2d,
                            f"loss_box2d_dn_{decoder_idx}": dn_box_loss2d,
                        }
                    )

                    if self.loss_alpha2d is not None:
                        dn_alpha_pred = dn_alpha_pred[dn_valid_mask2d]
                        dn_alpha_target = dn_alpha_target[dn_valid_mask2d]
                        if len(dn_alpha_target.shape) == 1:
                            dn_alpha_target = torch.stack((torch.sin(dn_alpha_target), torch.cos(dn_alpha_target)), dim=-1)

                        dn_alpha_loss = self.loss_alpha2d(dn_alpha_pred, dn_alpha_target, weight=dn_box_weights[:, :2], avg_factor=num_pos)
                        output.update(
                            {
                                f"loss_alpha2d_dn_{decoder_idx}": dn_alpha_loss,
                            }
                        )

                    if self.loss_depth2d is not None:
                        dn_depth_pred = dn_depth_pred[dn_valid_mask2d]
                        dn_depth_target = dn_depth_target[dn_valid_mask2d]

                        dn_depth_pred = dn_depth_pred.flatten(end_dim=1).squeeze(1)
                        dn_depth_target = dn_depth_target.flatten(end_dim=1)

                        dn_depth_loss = self.loss_depth2d(
                            dn_depth_pred, dn_depth_target, weight=dn_box_weights[:, 0], avg_factor=num_pos
                        )

                        output.update(
                            {
                                f"loss_depth2d_dn_{decoder_idx}": dn_depth_loss,
                            }
                        )

            else:
                for decoder_idx, cls in enumerate(model_outs["classification2d"]):
                    dn_cls_loss2d = cls.new_tensor(0., requires_grad=False)
                    dn_box_loss2d = cls.new_tensor(0., requires_grad=False)
                    dn_iou_loss2d = cls.new_tensor(0., requires_grad=False)

                    output.update(
                        {
                            f"loss_cls2d_dn_{decoder_idx}": dn_cls_loss2d,
                            f"loss_iou2d_dn_{decoder_idx}": dn_iou_loss2d,
                            f"loss_box2d_dn_{decoder_idx}": dn_box_loss2d,
                        }
                    )

                    if self.loss_alpha2d is not None:
                        dn_alpha_loss = cls.new_tensor(0., requires_grad=False)
                        output.update(
                            {
                                f"loss_alpha2d_dn_{decoder_idx}": dn_alpha_loss
                            }
                        )
                    if self.loss_depth2d is not None:
                        dn_depth_loss = cls.new_tensor(0., requires_grad=False)
                        output.update(
                            {
                                f"loss_depth2d_dn_{decoder_idx}": dn_depth_loss
                            }
                        )

        return output


    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, data, output_idx=-1, output_idx2d=-1):
        results = [dict() for _ in data['img_metas']]

        if self.enable2d:
            aug_configs = [img_metas['aug_config'] for img_metas in data['img_metas']]
            results_3d = self.decoder.decode_with2d(
                # 3d res
                model_outs["classification"],
                model_outs["prediction"],
                model_outs.get("instance_id"),
                model_outs.get("quality"),
                output_idx,
                # 2d res
                model_outs["classification2d"],
                model_outs["prediction2d"],
                model_outs['ref_trans_matrix_list'],
                model_outs['ref_query_groups_list'],
                output_idx2d,
                aug_configs,
                with_association=True,
            )
        else:
            results_3d = self.decoder.decode(
                model_outs["classification"],
                model_outs["prediction"],
                model_outs.get("instance_id"),
                model_outs.get("quality"),
                output_idx=output_idx,
            )

        for i, result_3d in enumerate(results_3d):
            results[i]['img_bbox'] = result_3d

        return results