# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
from mmcv.transforms import RandomFlip
from mmseg.registry import TRANSFORMS
from .transforms import RandomCrop, Resize


@TRANSFORMS.register_module()
class RandomCropMulti(RandomCrop):
    def transform(self, results: dict) -> dict:
        img = results["img"]
        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)
        if results.get("img2") is not None:
            results["img2"] = self.crop(results["img2"], crop_bbox)

        # crop semantic seg
        for key in results.get("seg_fields", []):
            results[key] = self.crop(results[key], crop_bbox)

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class ResizeMulti(Resize):
    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get("img") is not None:
            if self.keep_ratio:
                img = mmcv.imrescale(
                    results["img"],
                    results["scale"],
                    interpolation=self.interpolation,
                    backend=self.backend,
                )
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results["img"].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results["img"],
                    results["scale"],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend,
                )
            results["img"] = img
            results["img_shape"] = img.shape[:2]
            results["scale_factor"] = (w_scale, h_scale)
            results["keep_ratio"] = self.keep_ratio

        if results.get("img2") is not None:
            if self.keep_ratio:
                results["img2"] = mmcv.imrescale(
                    results["img2"],
                    results["scale"],
                    interpolation=self.interpolation,
                    backend=self.backend,
                )
            else:
                results["img2"] = mmcv.imresize(
                    results["img2"],
                    results["scale"],
                    interpolation=self.interpolation,
                    backend=self.backend,
                )


@TRANSFORMS.register_module()
class RandomFlipMulti(RandomFlip):
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes and semantic segmentation map."""
        # flip image
        results["img"] = mmcv.imflip(
            results["img"], direction=results["flip_direction"]
        )
        if results.get("img2") is not None:
            results["img2"] = mmcv.imflip(
                results["img2"], direction=results["flip_direction"]
            )

        img_shape = results["img"].shape[:2]

        # flip bboxes
        if results.get("gt_bboxes", None) is not None:
            results["gt_bboxes"] = self._flip_bbox(
                results["gt_bboxes"], img_shape, results["flip_direction"]
            )

        # flip seg map
        for key in results.get("seg_fields", []):
            if results.get(key, None) is not None:
                results[key] = self._flip_seg_map(
                    results[key], direction=results["flip_direction"]
                ).copy()
                results["swap_seg_labels"] = self.swap_seg_labels
