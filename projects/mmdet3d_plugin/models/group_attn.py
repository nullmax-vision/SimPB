import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import warnings
from mmcv.cnn import build_norm_layer, xavier_init
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.registry import (ATTENTION, TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule
from mmcv.utils import deprecated_api_warning
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@ATTENTION.register_module()
class QueryGroupMultiheadAttention(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 query_groups=None,
                 **kwargs):
        super(QueryGroupMultiheadAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.query_groups = query_groups
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)
        self.attn_mask = None

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='QueryGroupMultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                query_groups=None,
                group_attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        ## initial multihead attention
        ## Keep this query/key + xx_pos, for enhanced position info
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # initial group_attn_mask
        if group_attn_mask is None:

            if query_groups is not None and self.query_groups != query_groups:
                self.query_groups = query_groups

            if len(self.query_groups) > 1:
                group_attn_mask = torch.ones((query.shape[0], query.shape[0]),
                                             dtype=torch.float, device=query.device).fill_(float("-inf"))
                for qg in self.query_groups:
                    group_attn_mask[qg[0]:qg[1], qg[0]:qg[1]] = 0

        else:
            if group_attn_mask.dim() == 2:
                group_attn_mask = group_attn_mask
            else:
                group_attn_mask = torch.repeat_interleave(group_attn_mask, self.num_heads, dim=0)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=group_attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        out = torch.nan_to_num(out)

        return identity + self.dropout_layer(self.proj_drop(out))


@ATTENTION.register_module()
class QueryGroupMultiScaleDeformableAttention(MultiScaleDeformableAttention):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4, num_cams=6, query_groups=None,
                 im2col_step=64, dropout=0.1, batch_first=False, norm_cfg=None, init_cfg=None, residual_mode='add'):
        self.num_cams = num_cams
        self.query_groups = query_groups
        self.residual_mode = residual_mode
        super().__init__(embed_dims, num_heads, num_levels, num_points, im2col_step, dropout, batch_first, norm_cfg,
                         init_cfg)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        # bs, num_query, _
        bs, num_query, _ = query.shape
        # bs*num_cams, num_query, _
        bcs, num_value, _ = value.shape
        bvs = bcs // self.num_cams
        assert bvs == bs
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, self.num_cams, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets \
                                 / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 3:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets \
                                 / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.num_points \
                                 * reference_points[:, :, None, :, None, 2:] \
                                 * 0.5
        elif reference_points.shape[-1] == 5:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.num_points \
                                 * reference_points[:, :, None, :, None, 2:4] \
                                 * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        ref_depth = kwargs.get('ref_depth2d', None)
        if ref_depth is not None:
            xs, ys, _ = torch.where(ref_depth == 0)
            sampling_locations[xs, ys] = 0
        if torch.cuda.is_available() and value.is_cuda:
            output_list = []
            if kwargs.get('query_groups', None) is not None:
                self.query_groups = kwargs['query_groups']

            for i, qg in enumerate(self.query_groups):
                if (qg[1] - qg[0]) > 0:
                    output_i = MultiScaleDeformableAttnFunction.apply(
                        value[:, i].contiguous(), spatial_shapes, level_start_index,
                        sampling_locations[:, qg[0]:qg[1]].contiguous(),
                        attention_weights[:, qg[0]:qg[1]].contiguous(), self.im2col_step)
                    output_list.append(output_i)

            output = torch.cat(output_list, dim=1)

        else:
            raise ValueError(
                f'Only support gpu ms_deformable_attn, forbidden cpu implementation')
            # output = multi_scale_deformable_attn_pytorch(
            #     value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        output = self.dropout(output)

        if self.residual_mode == "add":
            output = output + identity
        elif self.residual_mode == "cat":
            output = torch.cat([output, identity], dim=-1)

        return output


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class QueryGroupDeformableDetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, batch_first=False, **kwargs):

        super(QueryGroupDeformableDetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.batch_first = batch_first
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)

            if not self.batch_first:
                output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if not self.batch_first:
                output = output.permute(1, 0, 2)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points