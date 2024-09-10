import os
import cv2
import copy
import mmcv
import math
import torch
import random
import tempfile
import numpy as np
import pyquaternion

from torch.utils.data import Dataset
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.detection.config import config_factory as det_configs
from nuscenes.eval.common.config import config_factory as track_configs

from os import path as osp
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from .utils import draw_lidar_bbox3d_on_img, draw_lidar_bbox3d_on_bev, draw_image_bbox2d_on_img


@DATASETS.register_module()
class NuScenes3DDetTrackDataset(Dataset):
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )
    ID_COLOR_MAP = [
        (59, 59, 238),
        # (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 127, 255),
        (71, 130, 255),
        (127, 127, 0),
    ]

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        test_mode=False,
        det3d_eval_version="detection_cvpr_2019",
        track3d_eval_version="tracking_nips_2019",
        version="v1.0-trainval",
        use_valid_flag=False,
        vis_score_threshold=0.25,
        data_aug_conf=None,
        sequences_split_num=1,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
        tracking=False,
        tracking_threshold=0.2,
        with_info2d=False,
    ):
        self.version = version
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.box_mode_3d = 0

        if classes is not None:
            self.CLASSES = classes
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.with_info2d = with_info2d
        self.with_velocity = with_velocity
        self.det3d_eval_version = det3d_eval_version
        self.det3d_eval_configs = det_configs(self.det3d_eval_version)
        self.track3d_eval_version = track3d_eval_version
        self.track3d_eval_configs = track_configs(self.track3d_eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.vis_score_threshold = vis_score_threshold

        self.data_aug_conf = data_aug_conf
        self.tracking = tracking
        self.tracking_threshold = tracking_threshold
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.current_aug = None
        self.last_id = None
        if with_seq_flag:
            self._set_sequence_group_flag()

    def __len__(self):
        return len(self.data_infos)

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]["sweeps"]) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
            scale_3d = np.random.uniform(*self.data_aug_conf.get("scale_ratio_range", [1.0, 1.0]))
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
            scale_3d = 1
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
            "final_dim": (fH, fW),
            "scale_3d": scale_3d,
        }
        return aug_config

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        data = self.pipeline(data)
        return data

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def match_2d_3d_bbox(self, input_dict, info, gt_labels_2d):
        gt_depths_2d = info['depths']
        gt_centers_2d = info['centers2d']
        gt_names_3d = info["gt_names"]
        gt_bboxes_3d = info['gt_boxes']
        lidar2imgs = input_dict['lidar2img']
        gt_2d_3d_map = [np.ones(len(x)) * -1 for x in gt_centers_2d]

        for cam_idx, cam_type in enumerate(info['cams']):
            if len(gt_centers_2d[cam_idx]) > 0:
                for index2d, (bbox_center2d, bbox_depth2d) in enumerate(
                        zip(gt_centers_2d[cam_idx], gt_depths_2d[cam_idx])):

                    best_dist = float('inf')
                    match_index3d = -1
                    for index3d, bbox3d in enumerate(gt_bboxes_3d):
                        bbox_homo = np.concatenate([bbox3d[:3], np.ones_like(bbox3d[:1])])
                        bbox_cam = lidar2imgs[cam_idx] @ bbox_homo

                        dist_match = np.linalg.norm(bbox_cam[2] - bbox_depth2d) + \
                                     np.linalg.norm(bbox_cam[:2] / bbox_cam[2] - bbox_center2d)

                        if dist_match < 1e-2 and dist_match < best_dist:
                            best_dist = dist_match
                            match_index3d = index3d

                    if match_index3d >= 0:
                        cat_2d = gt_labels_2d[cam_idx][index2d]
                        cat_3d = self.CLASSES.index(gt_names_3d[match_index3d])
                        if cat_2d == cat_3d:
                            gt_2d_3d_map[cam_idx][index2d] = match_index3d

        gt_2d_3d_map = [gm.astype(np.long) for gm in gt_2d_3d_map]

        return gt_2d_3d_map

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format="pkl")
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        print(self.metadata)
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
        )
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = pyquaternion.Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
        ego2global = np.eye(4)
        ego2global[:3, :3] = pyquaternion.Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        input_dict["lidar2global"] = ego2global @ lidar2ego

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            cam_intrinsic = []
            intrinsics = []
            extrinsics = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = (cam_info["sensor2lidar_translation"] @ lidar2cam_r.T)
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = copy.deepcopy(cam_info["cam_intrinsic"])
                cam_intrinsic.append(intrinsic)
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsic,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                )
            )

        if not self.test_mode:
            annos = self.get_ann_info(input_dict, index)
            input_dict.update(annos)
        return input_dict

    def matrix2eular(self, rot_mat):
        sy = math.sqrt(rot_mat[0][0]*rot_mat[0][0] + rot_mat[1][0]*rot_mat[1][0])
        YPR = []
        if sy > 1e-6:
            gamma = math.atan2(rot_mat[2][1], rot_mat[2][2])
            YPR.append(gamma) 
            YPR.append(math.atan2(-rot_mat[2][0], sy))

            YPR.append(math.atan2(rot_mat[1][0], rot_mat[0][0]))
            
        else:
            YPR.append(math.atan2(-rot_mat[1][2], rot_mat[1][1]))
            YPR.append(math.atan2(-rot_mat[2][0], sy))
            YPR.append(0)
        return YPR

    def get_alphas(self, gt_bboxes_3d, input_dict):
        lidar2camera_axis = np.array([1,  0,  0,  0,
                                      0,  0, -1,  0,
                                      0,  1,  0,  0,
                                      0,  0,  0,  1]).reshape(4, 4)
        extrins = input_dict['extrinsics']
        alpha_cameras = []
        for extrin_idx, extrinsic in enumerate(extrins):
            center_homos = np.concatenate([gt_bboxes_3d[:, :3], np.ones((gt_bboxes_3d.shape[0], 1))], axis=-1)
            center_cams = extrinsic @ center_homos.T
            masks = center_cams[2, :] > 0.2
            alphas = np.zeros(gt_bboxes_3d.shape[0], dtype=np.float32)
            for i in range(gt_bboxes_3d.shape[0]):
                if not masks[i]:
                    continue
                yaw = gt_bboxes_3d[i][6]
                rotation_yaw_matrix = np.asarray([[+np.cos(yaw), -np.sin(yaw), 0, 0],
                                                 [+np.sin(yaw),  +np.cos(yaw), 0, 0],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]])
                lidar2cam_without_axisalign = (np.linalg.inv(lidar2camera_axis) @ extrinsic) @ rotation_yaw_matrix
                rpy = self.matrix2eular(lidar2cam_without_axisalign)
                ry = rpy[2] * -1.0

                x, y, z = center_cams[0, i], center_cams[1, i], center_cams[2, i]
                beta = np.arctan(z / x)
                if beta < 0:
                    beta = beta + math.pi
                beta -= math.pi / 2
                alpha = ry + beta
                if alpha < -math.pi:
                    alpha += 2*math.pi
                if alpha > math.pi:
                    alpha -= 2*math.pi
                alphas[i] = alpha
                
            alpha_cameras.append(alphas)
        return alpha_cameras

    def get_ann_info(self, input_dict, index):
        info = self.data_infos[index]
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        if "instance_inds" in info:
            instance_inds = np.array(info["instance_inds"], dtype=np.int)[mask]
            anns_results["instance_inds"] = instance_inds

        if self.with_info2d:
            gt_bboxes_2d = copy.deepcopy(info['bboxes2d'])
            gt_labels_2d = copy.deepcopy(info['labels2d'])
            gt_centers_2d = copy.deepcopy(info['centers2d'])
            gt_depths_2d = copy.deepcopy(info['depths'])
            gt_alphas_2d = self.get_alphas(info["gt_boxes"], input_dict)

            if 'gt_2d_3d_map' not in info:
                info['gt_2d_3d_map'] = self.match_2d_3d_bbox(input_dict, info, gt_labels_2d)
            gt_2d_3d_map = copy.deepcopy(info['gt_2d_3d_map'])

            maskes_2d = [np.ones(len(x), dtype=np.bool) for x in gt_bboxes_2d]
            gt_alphas_aligned = []
            for cam_idx, map_2d_3d in enumerate(gt_2d_3d_map):
                gt_alphas_aligned.append(gt_alphas_2d[cam_idx][map_2d_3d])
                assert len(map_2d_3d) == len(gt_bboxes_2d[cam_idx])
                for index_2d, index_3d in enumerate(map_2d_3d):
                    if index_3d in np.where(~mask)[0]:
                        maskes_2d[cam_idx][index_2d] = False

            trans_index = np.ones((len(mask)), dtype=np.long) * -1
            trans_index[mask] = 1
            trans_index[mask] = np.cumsum(trans_index[mask]) - 1
            trans_index = np.concatenate([trans_index, -np.ones(1, dtype=np.long)])

            for cam_idx, mask_2d in enumerate(maskes_2d):
                gt_bboxes_2d[cam_idx] = gt_bboxes_2d[cam_idx][mask_2d]
                gt_labels_2d[cam_idx] = gt_labels_2d[cam_idx][mask_2d]
                gt_centers_2d[cam_idx] = gt_centers_2d[cam_idx][mask_2d]
                gt_depths_2d[cam_idx] = gt_depths_2d[cam_idx][mask_2d]
                gt_alphas_aligned[cam_idx] = gt_alphas_aligned[cam_idx][mask_2d]
                gt_2d_3d_map[cam_idx] = trans_index[gt_2d_3d_map[cam_idx][mask_2d]]

            anns_results.update(
                dict(
                    gt_bboxes_2d=gt_bboxes_2d,
                    gt_labels_2d=gt_labels_2d,
                    gt_centers_2d=gt_centers_2d,
                    gt_depths_2d=gt_depths_2d,
                    gt_alphas_2d=gt_alphas_aligned,
                    gt_2d_3d_map=gt_2d_3d_map,
                )
            )

        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(
                det, threshold=self.tracking_threshold if tracking else None
            )
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenes3DDetTrackDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenes3DDetTrackDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )

                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
        self, result_path, logger=None, result_name="img_bbox", tracking=False
    ):
        from nuscenes import NuScenes

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False
        )
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        if not tracking:
            from nuscenes.eval.detection.evaluate import NuScenesEval

            nusc_eval = NuScenesEval(
                nusc,
                config=self.det3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
            )
            nusc_eval.main(render_curves=False)

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            for name in self.CLASSES:
                for k, v in metrics["label_aps"][name].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}_AP_dist_{}".format(metric_prefix, name, k)
                    ] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["tp_errors"].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}".format(metric_prefix, self.ErrNameMapping[k])
                    ] = val

            detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
            detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]
        else:
            from nuscenes.eval.tracking.evaluate import TrackingEval

            nusc_eval = TrackingEval(
                config=self.track3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root,
            )
            metrics = nusc_eval.main()

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            print(metrics)
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            keys = [
                "amota",
                "amotp",
                "recall",
                "motar",
                "gt",
                "mota",
                "motp",
                "mt",
                "ml",
                "faf",
                "tp",
                "fp",
                "fn",
                "ids",
                "frag",
                "tid",
                "lgd",
            ]
            for key in keys:
                detail["{}/{}".format(metric_prefix, key)] = metrics[key]

        return detail

    def format_results(self, results, jsonfile_prefix=None, tracking=False):
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(
                results, jsonfile_prefix, tracking=tracking
            )
        else:
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {
                        name: self._format_bbox(
                            results_, tmp_file_, tracking=tracking
                        )
                    }
                )
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        for metric in ["detection", "tracking"]:
            if metric == "tracking": continue ## TODO
            tracking = metric == "tracking"
            if tracking and not self.tracking:
                continue
            result_files, tmp_dir = self.format_results(
                results, jsonfile_prefix, tracking=tracking
            )

            if isinstance(result_files, dict):
                results_dict = dict()
                for name in result_names:
                    ret_dict = self._evaluate_single(
                        result_files[name], tracking=tracking
                    )
                results_dict.update(ret_dict)
            elif isinstance(result_files, str):
                results_dict = self._evaluate_single(
                    result_files, tracking=tracking
                )
            if tmp_dir is not None:
                tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, save_dir=out_dir, show=show, pipeline=pipeline)
        return results_dict

    def show(self, results, save_dir=None, show=False, pipeline=None):
        save_dir = "./" if save_dir is None else save_dir
        save_dir = os.path.join(save_dir, "visual")
        print_log(os.path.abspath(save_dir))
        pipeline = Compose(pipeline)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        videoWriter = None

        from tqdm import tqdm
        for i, result in enumerate(tqdm(results)):
            if "img_bbox" in result.keys():
                result = result["img_bbox"]
            data_info = pipeline(self.get_data_info(i))
            annos = self.get_ann_info(data_info, i)

            raw_imgs = data_info["img"]
            lidar2img = data_info["img_metas"].data["lidar2img"]
            pred_bboxes_3d = result["boxes_3d"][result["scores_3d"] > self.vis_score_threshold]
            gt_bboxes_3d = annos["gt_bboxes_3d"]

            if "instance_ids" in result and self.tracking:
                color = []
                for id in result["instance_ids"].cpu().numpy().tolist():
                    color.append(self.ID_COLOR_MAP[int(id % len(self.ID_COLOR_MAP))])
            elif "labels_3d" in result:
                color = []
                for id in result["labels_3d"].cpu().numpy().tolist():
                    color.append(self.ID_COLOR_MAP[id])
            else:
                color = (255, 0, 0)

            gt_color = (0, 255, 0)

            # ===== draw boxes_3d to images =====
            imgs = []
            for j, img_origin in enumerate(raw_imgs):
                img = img_origin.copy()
                # if len(gt_bboxes_3d) != 0:
                #     img = draw_lidar_bbox3d_on_img(gt_bboxes_3d, img, lidar2img[j],
                #                                    img_metas=None, color=gt_color, thickness=3)
                if len(pred_bboxes_3d) != 0:
                    img = draw_lidar_bbox3d_on_img(pred_bboxes_3d, img, lidar2img[j],
                                                   img_metas=None, color=(255, 0, 0), thickness=3)

                imgs.append(img)

            # ===== draw boxes_2d to images =====
            for j, img in enumerate(imgs):
                qp = result['query_groups'][j]
                pred_bboxes_2d = result["boxes_2d"][qp[0]: qp[1]][result["scores_2d"][qp[0]: qp[1]] > self.vis_score_threshold]
                if len(pred_bboxes_2d) != 0:
                    img = draw_image_bbox2d_on_img(pred_bboxes_2d, img,
                                                   color=(70, 255, 255), thickness=2)

            # ===== draw boxes_3d to BEV =====
            bev = draw_lidar_bbox3d_on_bev(gt_bboxes_3d, bev_size=img.shape[0] * 2, color=gt_color)
            bev = draw_lidar_bbox3d_on_bev(pred_bboxes_3d, bev_size=img.shape[0] * 2, color=color, bev_img=bev)

            # ===== put text and concat =====
            for j, name in enumerate(["FRONT", "FRONT_RIGHT", "FRONT_LEFT", "REAR", "REAR LEFT", "REAR RIGHT"]):
                cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                imgs[j] = cv2.putText(imgs[j], name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

            image = np.concatenate([
                np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
            ], axis=0)
            # image = np.concatenate([image, bev], axis=1)

            # ===== save video =====
            if videoWriter is None:
                videoWriter = cv2.VideoWriter(os.path.join(save_dir, "video.avi"), fourcc, 7, image.shape[:2][::-1])
            cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), image)
            videoWriter.write(image)
        videoWriter.release()


def output_to_nusc_box(detection, threshold=None):
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "instance_ids" in detection:
        ids = detection["instance_ids"]  # .numpy()
    if threshold is not None:
        if "cls_scores" in detection:
            mask = detection["cls_scores"].numpy() >= threshold
        else:
            mask = scores >= threshold
        box3d = box3d[mask]
        scores = scores[mask]
        labels = labels[mask]
        ids = ids[mask]

    if hasattr(box3d, "gravity_center"):
        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        nus_box_dims = box_dims[:, [1, 0, 2]]
        box_yaw = box3d.yaw.numpy()
    else:
        box3d = box3d.numpy()
        box_gravity_center = box3d[..., :3].copy()
        box_dims = box3d[..., 3:6].copy()
        nus_box_dims = box_dims[..., [1, 0, 2]]
        box_yaw = box3d[..., 6].copy()

    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if hasattr(box3d, "gravity_center"):
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (*box3d[i, 7:9], 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        if "instance_ids" in detection:
            box.token = ids[i]
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info,
    boxes,
    classes,
    eval_configs,
    eval_version="detection_cvpr_2019",
):
    box_list = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list
