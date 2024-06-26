# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseCDDataset


@DATASETS.register_module()
class FIBDualDataset(BaseCDDataset):
    METAINFO = dict(classes=("pore", "particle"), palette=[[0, 0, 0], [255, 0, 0]])

    def __init__(
        self, img_suffix=".png", img_suffix2=".png", seg_map_suffix=".png", **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            img_suffix2=img_suffix2,
            seg_map_suffix=seg_map_suffix,
            **kwargs
        )
