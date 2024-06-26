from copy import deepcopy
from typing import List, Optional

from mmcv.transforms import Compose
from mmengine.infer.infer import InputsType
from numpy import ndarray

from mmseg.utils import ConfigType
from .mmseg_inferencer import MMSegInferencer


class FIBDualInferencer(MMSegInferencer):
    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
        dc_inputs = deepcopy(inputs)
        return super().preprocess(dc_inputs, batch_size, **kwargs)

    def visualize(
        self,
        inputs: list,
        preds: List[dict],
        return_vis: bool = False,
        show: bool = False,
        wait_time: int = 0,
        img_out_dir: str = "",
        opacity: float = 0.8,
        with_labels: Optional[bool] = True,
    ) -> List[ndarray]:
        return super().visualize(
            [inp["img_path"] for inp in inputs],
            preds,
            return_vis,
            show,
            wait_time,
            img_out_dir,
            opacity,
            with_labels,
        )

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        for transform in ("LoadAnnotations", "LoadDepthAnnotation"):
            idx = self._get_transform_idx(pipeline_cfg, transform)
            if idx != -1:
                del pipeline_cfg[idx]
        return Compose(pipeline_cfg)
