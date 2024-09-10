# ================ base config ===================
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None

num_gpus = 8
batch_size = 4
num_iters_per_epoch = int(28130 // (num_gpus * batch_size))
num_epochs = 100
checkpoint_epoch_interval = 20
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=1)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)

load_from = None
resume_from = None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)

tracking_test = True
tracking_threshold = 0.2

num_dn_groups = 5
num_temp_dn_groups = 3
with_alpha_angle = True

# ================== model ========================
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

num_classes = len(class_names)
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
decouple_attn = True
decouple_attn2d = True
with_quality_estimation = True

single_decoder_layer3d = ["gnn", "norm", "deformable", "ffn", "norm", "refine3d"]
single_decoder_layer2d = ["allocation", "qg_self_attn", "norm", "qg_cross_attn", "ffn", "norm", "refine2d", "aggregation", "refine3d"]
decoder_layer3d = ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine3d"]
decoder_layer2d = ["temp_gnn", "allocation", "qg_self_attn", "norm", "qg_cross_attn", "ffn", "norm", "refine2d", "aggregation", "refine3d"]

operation_order = single_decoder_layer2d + decoder_layer3d +\
                  decoder_layer2d + decoder_layer3d +\
                  decoder_layer2d + decoder_layer3d


model = dict(
    type="SimPB",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="ckpts/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    head=dict(
        type="SimPBHead",
        enable2d=True,
        num_levels=num_levels,
        embed_dims=embed_dims,
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn,
        decouple_attn2d=decouple_attn2d,
        with_denoise2d=True,
        denoise2d=dict(
            type="Denoise2D",
            num_dn_groups=num_dn_groups,
        ),
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor="./data/nuscenes/nuscenes_kmeans900.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=600 if temporal else -1,
            confidence_decay=0.6,
            feat_grad=False,
        ),
        anchor_encoder2d=dict(
            type="SparseBox2DEncoder",
            embed_dims=embed_dims,
            with_sin_embed=True,
            in_loops=1,
            out_loops=2,
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            vel_dims=3,
            embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
            mode="cat" if decouple_attn else "add",
            output_fc=not decouple_attn,
            in_loops=1,
            out_loops=4 if decouple_attn else 2,
        ),
        encoder2d=None,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=operation_order,
        # common op
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            pre_norm=dict(type="LN"),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        # 2d op
        dynamic_allocation=dict(
            type="DynamicQueryAllocation",
            limit_corners_num=[100, 100, 100, 100, 100, 100],
        ),
        adaptive_aggregation=dict(
            type="AdaptiveQueryAggregation",
            self_attn=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            reweight=True,
            with_pos=True,
        ),
        qg_self_attn=dict(
            type='QueryGroupMultiheadAttention',
            batch_first=True,
            embed_dims=embed_dims if not decouple_attn2d else embed_dims * 2,
            num_heads=num_groups,
            attn_drop=drop_out,
            dropout_layer=dict(type='Dropout', drop_prob=0.1),
        ),
        qg_cross_attn=dict(
            type='QueryGroupMultiScaleDeformableAttention',
            batch_first=True,
            num_levels=num_levels,
            embed_dims=embed_dims,
            num_points=4,
            residual_mode='cat',
        ),
        refine_layer2d=dict(
            type="SparseBox2DRefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
            with_alpha_branch=True,
        ),
        # 3d op
        temp_graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        )
        if temporal
        else None,
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregation",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ),
        ),
        refine_layer3d=dict(
            type="SparseBox3DRefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
            refine_yaw=True,
            with_quality_estimation=with_quality_estimation,
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        coster2d=dict(
            type='SparseBox2DCoster',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)
        ),
        sampler=dict(
            type="SparseBox3DTargetWith2D",
            num_dn_groups=num_dn_groups,
            num_temp_dn_groups=num_temp_dn_groups,
            with_alpha_angle=with_alpha_angle,
            dn_noise_scale=[2.0] * 3 + [0.5] * 7,
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
            cls_wise_reg_weights={
                class_names.index("traffic_cone"): [
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
            },
        ),
        # 2d
        loss_cls2d=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
        loss_alpha2d=dict(type='L1Loss', loss_weight=0.5),
        # 3d
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(
            type="SparseBox3DLoss",
            loss_box=dict(type="L1Loss", loss_weight=0.25),
            loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
            loss_yawness=dict(type="GaussianFocalLoss"),
            cls_allow_reverse=[class_names.index("barrier")],
        ),
        decoder=dict(type="SparseBox3DDecoder"),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),
)

# ================== data ========================
dataset_type = "NuScenes3DDetTrackDataset"
data_root = "data/nuscenes/"
anno_root = "data/nuscenes/"
file_client_args = dict(backend="disk")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile",
         coord_type="LIDAR",
         load_dim=5,
         use_dim=5,
         file_client_args=file_client_args
    ),
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator",
         downsample=strides[:num_depth_layers]
    ),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter",
         class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "gt_bboxes_3d", "gt_labels_3d", "gt_bboxes_2d", "gt_labels_2d", "gt_depths_2d", "gt_alphas_2d", "gt_2d_3d_map"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "aug_config"]
    ),
]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "timestamp", "projection_mat", "image_wh"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "aug_config"],
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    version="v1.0-trainval",
)

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0, 0), #(-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [-0.3925, 0.3925],
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "simpb_nuscenes_infos_train.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        with_info2d=True,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "simpb_nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        tracking=tracking_test,
        tracking_threshold=tracking_threshold,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "simpb_nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        tracking=tracking_test,
        tracking_threshold=tracking_threshold,
    ),
)

# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=4e-4,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)

# ================== eval ========================
vis_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="Collect",
        keys=["img"],
        meta_keys=["timestamp", "lidar2img"],
    ),
]
evaluation = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval,
    pipeline=vis_pipeline,
    # out_dir="./vis",  # for visualization
)

# load pretrained weights
load_from = 'ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]


'''
mAP: 0.4896
mATE: 0.5274
mASE: 0.2621
mAOE: 0.3566
mAVE: 0.2065
mAAE: 0.1868
NDS: 0.5909
Eval time: 81.7s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.685	0.353	0.145	0.047	0.179	0.199
truck	0.385	0.544	0.189	0.046	0.178	0.215
bus	0.455	0.694	0.195	0.050	0.366	0.246
trailer	0.227	1.001	0.262	0.775	0.161	0.099
construction_vehicle	0.144	0.829	0.462	0.979	0.132	0.344
pedestrian	0.597	0.486	0.281	0.359	0.253	0.143
motorcycle	0.538	0.437	0.234	0.277	0.247	0.242
bicycle	0.461	0.401	0.252	0.585	0.137	0.006
traffic_cone	0.726	0.246	0.319	nan	nan	nan
barrier	0.677	0.284	0.283	0.091	nan	nan
'''
