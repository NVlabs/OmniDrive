import torch
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.hooks import OptimizerHook
from torch.cuda.amp import autocast
import copy
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model, LossScaler, allreduce_grads)
from typing import Optional, Union
import torch.nn as nn
from torch.nn.utils import clip_grad
from torch import Tensor
import logging
from collections import defaultdict
from itertools import chain

@HOOKS.register_module()
class CustomOptimizerHook(OptimizerHook):  # type: ignore
    """FP16 optimizer hook (mmcv's implementation).

    The steps of fp16 optimizer is as follows.
    1. Scale the loss value.
    2. BP in the fp16 model.
    2. Copy gradients from fp16 model to fp32 weights.
    3. Update fp32 weights.
    4. Copy updated parameters from fp32 weights to fp16 model.

    Refer to https://arxiv.org/abs/1710.03740 for more details.

    Args:
        loss_scale (float | str | dict): Scale factor configuration.
            If loss_scale is a float, static loss scaling will be used with
            the specified scale. If loss_scale is a string, it must be
            'dynamic', then dynamic loss scaling will be used.
            It can also be a dict containing arguments of LossScaler.
            Defaults to 512.
    """
    def before_run(self, runner) -> None:
        """Preparing steps before Mixed Precision Training."""
        # wrap model mode to fp16
        # runner.optimizer.param_groups = copy.deepcopy(
        #         runner.optimizer.param_groups.to(torch.bfloat16))
        runner.model = runner.model.to(torch.bfloat16)

    # def after_train_iter(self, runner):
    #     runner.optimizer.zero_grad()
    #     if self.detect_anomalous_params:
    #         self.detect_anomalous_parameters(runner.outputs['loss'], runner)
    #     runner.outputs['loss'].backward()

    #     if self.grad_clip is not None:
    #         grad_norm = self.clip_grads(runner.model.parameters())
    #         if grad_norm is not None:
    #             # Add grad norm to the logger
    #             runner.log_buffer.update({'grad_norm': float(grad_norm)},
    #                                      runner.outputs['num_samples'])
    #     runner.optimizer.step()