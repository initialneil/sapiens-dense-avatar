# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import os.path as osp
import shutil
from collections import OrderedDict
from typing import Dict, Optional, Sequence
import numpy as np
from mmengine.dist import is_main_process, master_only
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image

from mmseg.registry import METRICS
from ...datasets.dense_avatar_general import save_xyz_iuv_to_png

@METRICS.register_module()
class XYZIUVMetric(BaseMetric):
    """XYZIUV evaluation metric.

    Args:
        output_dir (str): The directory for output prediction
        keep_results (bool): Whether to keep the results. When ``format_only``
            is True, ``keep_results`` must be True. Defaults to False.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 output_dir: str,
                 keep_results: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.output_dir = output_dir

        self.keep_results = keep_results
        self.prefix = prefix
        if is_main_process():
            mkdir_or_exist(self.output_dir)

    @master_only
    def __del__(self) -> None:
        """Clean up."""
        if not self.keep_results:
            shutil.rmtree(self.output_dir)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        mkdir_or_exist(self.output_dir)

        for input_img, data_sample in zip(data_batch['inputs'], data_samples):
            basename = osp.splitext(osp.basename(data_sample['img_path']))[0]

            # # input image
            # png_filename = osp.abspath(osp.join(self.output_dir, f'{basename}.png'))
            # cv2.imwrite(png_filename, input_img.permute(1, 2, 0).numpy())
            
            # prediction
            pred_label = data_sample['pred_depth_map']['data'].permute(1, 2, 0).cpu().numpy()
            pred_xyz = pred_label[..., :3]
            pred_i = pred_label[..., 3:6].argmax(axis=-1, keepdims=True)
            pred_uv = pred_label[..., 6:9]

            png_filename = osp.abspath(osp.join(self.output_dir, f'{basename}_xyziuv.png'))
            save_xyz_iuv_to_png(png_filename, pred_xyz, pred_i, pred_uv)
            
            self.results.append((png_filename, data_sample['img_path']))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): Testing results of the dataset.

        Returns:
            dict[str: float]: XYZIUV evaluation results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        # if self.format_only:
        #     logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
        #     return OrderedDict()

        msg = 'Evaluating in XYZIUV style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        eval_results = dict()
        print_log(
            f'Evaluating results under {self.output_dir} ...', logger=logger)

        pred_list, in_list = zip(*results)
        metric = dict()

        return metric

    # @staticmethod
    # def _convert_to_label_id(result):
    #     """Convert trainId to id for XYZIUV."""
    #     if isinstance(result, str):
    #         result = np.load(result)
    #     result_copy = result.copy()
    #     for trainId, label in CSLabels.trainId2label.items():
    #         result_copy[result == trainId] = label.id

    #     return result_copy
