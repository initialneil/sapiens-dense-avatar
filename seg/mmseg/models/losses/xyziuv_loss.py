# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss
from .cross_entropy_loss import CrossEntropyLoss

##################################################
@MODELS.register_module()
class XYZIUV_XYZ_MSELoss(nn.Module):
    """XYZIUV_XYZ_MSELoss. MSE Loss for XYZ in the XYZIUV output.

    Args:
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_xyziuv.xyz_mse'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred_full, target_full, weight=None, avg_factor=None, reduction_override=None):
        assert pred_full.shape == target_full.shape, f'The shapes of pred ({pred_full.shape}) and target ({target_full.shape}) are mismatched'
        assert reduction_override in (None, 'none', 'mean', 'sum'), 'Invalid reduction_override value'

        reduction = reduction_override if reduction_override else self.reduction
        
        # uv from xyziuv.uv
        # pred: [xyz 012 uv]
        # target: [xyz label uv]
        # valid_mask from xyziuv.i
        pred = pred_full[..., :3]
        target = target_full[..., :3]
        valid_mask = target_full[..., 3:4] > 1e-3

        if valid_mask.sum() == 0:
            return 0.0 * pred.sum()

        loss = F.mse_loss(pred, target, reduction='none') * valid_mask

        if reduction == 'mean':
            loss = loss.sum() / valid_mask.sum().clamp(min=1)
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'none':
            pass  # Keep per-pixel loss
        else:
            raise ValueError(f'Invalid reduction type: {reduction}')

        loss = weight_reduce_loss(loss, weight, reduction, avg_factor) * self.loss_weight

        ## convert nan to 0
        loss = torch.nan_to_num(loss, 
                nan=torch.tensor(0, dtype=pred.dtype, device=pred.device), 
                posinf=torch.tensor(0, dtype=pred.dtype, device=pred.device), 
                neginf=torch.tensor(0, dtype=pred.dtype, device=pred.device))

        return loss

    @property
    def loss_name(self):
        """Returns the name of this loss function."""
        return self._loss_name

##################################################
@MODELS.register_module()
class XYZIUV_I_CELoss(CrossEntropyLoss):
    """XYZIUV_I_CELoss. Cross Entropy Loss for I in the XYZIUV output.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_xyziuv.i_ce',
                 avg_non_ignore=False):
        super().__init__(use_sigmoid=use_sigmoid,
                         use_mask=use_mask,
                         reduction=reduction,
                         class_weight=class_weight,
                         loss_weight=loss_weight,
                         loss_name=loss_name,
                         avg_non_ignore=avg_non_ignore)

    def forward(self,
                pred_full,
                target_full,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        # uv from xyziuv.uv
        # pred: [xyz 012 uv]
        # target: [xyz label uv]
        # valid_mask from xyziuv.i
        cls_score = pred_full[..., 3:6]
        label = target_full[..., 3:4]
        loss_cls = super().forward(cls_score,
                                   label,
                                   weight=weight,
                                   avg_factor=avg_factor,
                                   reduction_override=reduction_override,
                                   ignore_index=ignore_index,
                                   **kwargs)
        return loss_cls

##################################################
@MODELS.register_module()
class XYZIUV_UV_MSELoss(nn.Module):
    """XYZIUV_UV_MSELoss. MSE Loss for UV in the XYZIUV output.

    Args:
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_xyziuv.uv_mse'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred_full, target_full, weight=None, avg_factor=None, reduction_override=None):
        assert pred_full.shape == target_full.shape, f'The shapes of pred ({pred_full.shape}) and target ({target_full.shape}) are mismatched'
        assert reduction_override in (None, 'none', 'mean', 'sum'), 'Invalid reduction_override value'

        reduction = reduction_override if reduction_override else self.reduction
        
        # uv from xyziuv.uv
        # pred: [xyz 012 uv]
        # target: [xyz label uv]
        # valid_mask from xyziuv.i
        pred = pred_full[..., 6:8]
        target = target_full[..., 4:6]
        valid_mask = target_full[..., 3:4] > 1e-3

        if valid_mask.sum() == 0:
            return 0.0 * pred.sum()

        loss = F.mse_loss(pred, target, reduction='none') * valid_mask

        if reduction == 'mean':
            loss = loss.sum() / valid_mask.sum().clamp(min=1)
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'none':
            pass  # Keep per-pixel loss
        else:
            raise ValueError(f'Invalid reduction type: {reduction}')

        loss = weight_reduce_loss(loss, weight, reduction, avg_factor) * self.loss_weight

        ## convert nan to 0
        loss = torch.nan_to_num(loss, 
                nan=torch.tensor(0, dtype=pred.dtype, device=pred.device), 
                posinf=torch.tensor(0, dtype=pred.dtype, device=pred.device), 
                neginf=torch.tensor(0, dtype=pred.dtype, device=pred.device))

        return loss

    @property
    def loss_name(self):
        """Returns the name of this loss function."""
        return self._loss_name

