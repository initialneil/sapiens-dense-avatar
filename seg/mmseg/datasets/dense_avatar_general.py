# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

import numpy as np
import os
import cv2
import pickle
from PIL import ImageDraw
from tqdm import tqdm
import io
import json
import copy
import os.path as osp
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import random
from matplotlib import pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import mmengine.fileio as fileio
from multiprocessing import Pool
import gzip

# write xyziuv to 6 grids in png
# xyz(gbt), mask(fb), uv
XYZ_SCALE, XYZ_SHIFT = 10000.0, 32768.0
MASK_SCALE = 32767.0
UV_SCALE = 50000.0
def write_xyz_iuv_to_png(xyziuv_fn, xyz, mask_fb, uv_map):
    xyziuv = np.concatenate([
        np.concatenate([
            xyz[..., 0:1] * XYZ_SCALE + XYZ_SHIFT,
            xyz[..., 1:2] * XYZ_SCALE + XYZ_SHIFT,
            xyz[..., 2:3] * XYZ_SCALE + XYZ_SHIFT,
        ], axis=1),
        np.concatenate([
            mask_fb * MASK_SCALE,
            uv_map[..., 0:1] * UV_SCALE,
            uv_map[..., 1:2] * UV_SCALE,
        ], axis=1),
    ], axis=0)

    cv2.imwrite(xyziuv_fn, xyziuv.detach().cpu().numpy().astype(np.uint16))

def read_xyz_iuv_from_png(xyziuv_fn):
    img = cv2.imread(xyziuv_fn, cv2.IMREAD_UNCHANGED)
    xyziuv = img.astype(float)
    h, w = xyziuv.shape[:2]
    xyz = (xyziuv[:h//2, ...] - XYZ_SHIFT) / XYZ_SCALE
    xyz = np.stack(xyz.split(w//3, dim=1), axis=-1)

    mask_fb = xyziuv[h//2:h, :w//3, None] / MASK_SCALE
    uv_map = xyziuv[h//2:h, w//3:] / UV_SCALE
    uv_map = np.stack(uv_map.split(w//3, dim=1), dim=-1)
    return xyz, mask_fb, uv_map

##-----------------------------------------------------------------------
@DATASETS.register_module()
class DenseAvatarGeneralDataset(BaseSegDataset):
    def __init__(self,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        return

    def load_data_list(self, use_cache=True) -> List[dict]:
        """Load annotation from directory or annotation file.
        modify this function to load a list of all sample information in training

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        self.rgb_dir = os.path.join(self.data_root, 'png')
        self.gt_dir = os.path.join(self.data_root, 'ground_truth')
        self.cam_dir = os.path.join(self.data_root, 'ground_truth/camera')

        print('\033[92mLoading DenseAvatarGeneralDataset! {}\033[0m'.format(self.data_root))

        # Create data list from sequence list
        seq_list = [d for d in os.listdir(self.rgb_dir) if d.startswith('seq_')]
        for seq_dir in seq_list:
            fns = [fn for fn in os.listdir(f'{self.rgb_dir}/{seq_dir}') if fn.endswith('.png') and fn[11] != '-']
            for img_fn in fns:
                xyziuv_fn = img_fn.replace('.png', '_xyz(gbt)_iuv.png')
                data_list.append({
                    'rgb_path': os.path.join(self.rgb_dir, f'{seq_dir}/{img_fn}'),
                    'xyziuv_path': os.path.join(self.gt_dir, f'{seq_dir}/{xyziuv_fn}'),
                })

        print('\033[92mDone! DenseAvatarGeneralDataset. Loaded total samples: {}.\033[0m'.format(len(data_list)))

        return data_list

    def get_data_info(self, idx):
        """Get data info according to the given index.
        modify this function to load a the sample at index idx

        Args:
            idx (int): Index of the sample data to get.
        
        Returns:
            dict: Data information. Following keys are important:
                - img : numpy array. in BGR channel order. in range [0, 255]. H x W x 3
                - gt_xyziuv : numpy array. in H x W x 6. 
                  - xyz in range [-1, 1]. X, Y, Z
                  - i(mask_fb) in set {0, 1, 2}. None/Back/Front
                  - uv in range [0, 1], U, V
                - mask: numpy array. in H x W. in range [0, 255]. 0 is background, 255 is foreground. H x W.
                  - mask <= (mask_fb > 0)
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])

        img = cv2.imread(data_info['rgb_path']) ## bgr image is default
        xyz, mask_fb, uv_map = read_xyz_iuv_from_png(data_info['xyziuv_path'])
        gt_xyziuv = np.concatenate([xyz, mask_fb, uv_map], axis=-1)
        mask = (mask_fb > 0).squeeze(-1)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        # Find the bounding box's bounds
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        data_info = {
            'img': img,
            'img_id': os.path.basename(data_info['rgb_path']),
            'img_path': data_info['rgb_path'],
            'gt_xyziuv': gt_xyziuv,
            'mask': mask,
            'id': idx,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
        }

        return data_info
