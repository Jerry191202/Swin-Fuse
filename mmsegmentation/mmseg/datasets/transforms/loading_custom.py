# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
import mmcv
from mmengine import fileio
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadMultipleFIBImageFromFile(BaseTransform):
    """Load two FIB-SEM images from file."""

    def __init__(
        self, color_type: str = "color", imdecode_backend: str = "cv2"
    ) -> None:
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results["img_path"]
        filename2 = results["img_path2"]
        img_bytes = fileio.get(filename)
        img_bytes2 = fileio.get(filename2)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend
        )
        img2 = mmcv.imfrombytes(
            img_bytes2, flag=self.color_type, backend=self.imdecode_backend
        )

        results["img"] = img
        results["img2"] = img2
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"color_type='{self.color_type}', "
            f"imdecode_backend='{self.imdecode_backend}', "
        )

        return repr_str
