import copy
import cv2
import mmcv
import torch
import numpy as np
from PIL import Image
from numpy import random
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    def __init__(self, filter_invisible=True):
        self.min_size = 2
        self.filter_invisible = filter_invisible

    def __call__(self, results):
        aug_config = results.get("aug_config")
        if aug_config is None:
            return results
        imgs = results["img"]
        aug_config["ori_shape"] = imgs[0].shape

        N = len(imgs)
        new_imgs = []
        new_gt_bboxes_2d = []
        new_gt_labels_2d = []
        new_gt_centers_2d = []
        new_gt_depths_2d = []
        new_gt_2d_3d_map = []
        new_gt_alphas_2d = []

        for i in range(N):
            img, mat = self._img_transform(np.uint8(imgs[i]), aug_config)

            if "gt_bboxes_2d" in results:
                gt_bboxes = results['gt_bboxes_2d'][i]
                centers2d = results['gt_centers_2d'][i]
                gt_labels = results['gt_labels_2d'][i]
                depths = results['gt_depths_2d'][i]
                gt_2d_3d_map = results['gt_2d_3d_map'][i]
                gt_alphas_2d = copy.deepcopy(results['gt_alphas_2d'][i])
                if len(gt_bboxes) != 0:
                    gt_bboxes, centers2d, gt_labels, depths, gt_2d_3d_map, gt_alphas_2d = self._bboxes_transform(
                        gt_bboxes, centers2d, gt_labels, depths, gt_2d_3d_map, gt_alphas_2d, aug_config)
                else:
                    gt_bboxes = gt_bboxes.reshape(0, 4)
                    gt_labels = gt_labels.reshape(0)
                    centers2d = centers2d.reshape(0, 2)
                    depths = depths.reshape(0)
                    gt_2d_3d_map = gt_2d_3d_map.reshape(0)
                    gt_alphas_2d = gt_alphas_2d.reshape(0)

                if len(gt_bboxes) != 0 and self.filter_invisible:
                    gt_bboxes, centers2d, gt_labels, depths, gt_2d_3d_map, gt_alphas_2d = self._filter_invisible(
                        gt_bboxes, centers2d, gt_labels, depths, gt_2d_3d_map, gt_alphas_2d, aug_config)

                new_gt_bboxes_2d.append(gt_bboxes)
                new_gt_labels_2d.append(gt_labels)
                new_gt_centers_2d.append(centers2d)
                new_gt_depths_2d.append(depths)
                new_gt_2d_3d_map.append(gt_2d_3d_map)
                new_gt_alphas_2d.append(gt_alphas_2d)

            # results["lidar2img"][i] = mat @ results["lidar2img"][i]
            results['intrinsics'][i] = mat @ results['intrinsics'][i]
            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= aug_config["resize"]
                # results["cam_intrinsic"][i][:3, :3] = mat[:3, :3] @ results["cam_intrinsic"][i][:3, :3]

            new_imgs.append(np.array(img).astype(np.float32))

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]

        results['gt_bboxes_2d'] = new_gt_bboxes_2d
        results['gt_labels_2d'] = new_gt_labels_2d
        results['gt_centers_2d'] = new_gt_centers_2d
        results['gt_depths_2d'] = new_gt_depths_2d
        results['gt_2d_3d_map'] = new_gt_2d_3d_map
        results['gt_alphas_2d'] = new_gt_alphas_2d

        return results

    def _img_transform(self, img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array(
                [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths, gt_2d_3d_map, alphas, aug_configs):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths) == len(gt_2d_3d_map)
        H, W = aug_configs.get("ori_shape", [900, 1600, 3])[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)
        fH, fW = aug_configs["final_dim"]

        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)

        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1

            neg_alphas = alphas < 0
            alphas[neg_alphas] = -1.0 * alphas[neg_alphas] - np.pi
            alphas[~neg_alphas] = -1.0 * alphas[~neg_alphas] + np.pi

        if rotate != 0:
            rot_matrix = cv2.getRotationMatrix2D([fW / 2, fH / 2], rotate, 1)
            new_bboxes = []
            for i, bbox in enumerate(bboxes):
                x_min, y_min, x_max, y_max = bbox
                w, h = x_max - x_min, y_max - y_min
                cx, cy = x_min + w / 2, y_min + h / 2
                new_cx, new_cy = np.dot(rot_matrix, np.array([cx, cy, 1]))
                x_min, y_min, x_max, y_max = max(new_cx - w / 2, 0), max(new_cy - h / 2, 0), \
                                             min(new_cx + w / 2, fW), min(new_cy + h / 2, fH)
                new_bboxes.append([x_min, y_min, x_max, y_max])
            bboxes = np.array(new_bboxes, dtype=np.float32)

        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)

        bboxes = bboxes[keep]

        centers2d = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH)
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        if rotate != 0:
            new_centers2d = []
            for center2d in centers2d:
                new_cx, new_cy = np.dot(rot_matrix, np.array([center2d[0], center2d[1], 1]))
                new_cx, new_cy = min(max(new_cx, 0), fW), min(max(new_cy, 0), fH)
                new_centers2d.append([new_cx, new_cy])
            centers2d = np.array(new_centers2d)

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]
        gt_2d_3d_map = gt_2d_3d_map[keep]
        alphas = alphas[keep]

        return bboxes, centers2d, gt_labels, depths, gt_2d_3d_map, alphas

    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths, gt_2d_3d_map, alphas, aug_configs):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths) == len(gt_2d_3d_map)
        fH, fW = aug_configs["final_dim"]
        indices_maps = np.zeros((fH,fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        gt_2d_3d_map = gt_2d_3d_map[sort_idx]
        alphas = alphas[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]
        gt_2d_3d_map = gt_2d_3d_map[indices_res]
        alphas = alphas[indices_res]

        return bboxes, centers2d, gt_labels, depths, gt_2d_3d_map, alphas


@PIPELINES.register_module()
class BBoxRotation(object):
    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = results["lidar2img"][view] @ rot_mat_inv
            results["extrinsics"][view] = (results["extrinsics"][view] @ rot_mat_inv)
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = self.box_rotate(results["gt_bboxes_3d"], angle)
        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d


@PIPELINES.register_module()
class BBoxScale(object):
    def __call__(self, results):
        scale_ratio = results["aug_config"]["scale_3d"]

        scale_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )
        scale_mat_inv = np.linalg.inv(scale_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = results["lidar2img"][view] @ scale_mat_inv
            results["extrinsics"][view] = (results["extrinsics"][view] @ scale_mat_inv)
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ scale_mat_inv
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = self.box_scale(results["gt_bboxes_3d"], scale_ratio)
        return results

    @staticmethod
    def box_scale(bbox_3d, scale_ratio):
        bbox_3d[:,:6] *= scale_ratio
        bbox_3d[:,7:] *= scale_ratio
        return bbox_3d


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str
