from typing import Optional, Union, Callable, Dict

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.classifier import BasicClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import (
    at_least_ndim,
    SUPPORTED_NOISE_SCHEDULES, SUPPORTED_DISCRETIZATIONS, SUPPORTED_SAMPLING_STEP_SCHEDULE)
from .basic import DiffusionModel


class DiscreteRectifiedFlow(DiffusionModel):

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,

            # ----------------- Masks ----------------- #
            # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`

            # ------------------ Plugins ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier: Optional[BasicClassifier] = None,

            # ------------------ Training Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,

            # ------------------- Diffusion Params ------------------- #
            diffusion_steps: int = 1000,

            discretization: Union[str, Callable] = "uniform",

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)
        
        assert classifier is None, "Rectified Flow does not support classifier-guidance."

        self.x_max, self.x_min = x_max, x_min

        # ================= Discretization =================
        # - Map the continuous range [0., 1.] to the discrete range [0, T-1]
        if isinstance(discretization, str):
            if discretization in SUPPORTED_DISCRETIZATIONS.keys():
                self.t_diffusion = SUPPORTED_DISCRETIZATIONS[discretization](diffusion_steps, 0.).to(device)
            else:
                Warning(f"Discretization method {discretization} is not supported. "
                        f"Using uniform discretization instead.")
                self.t_diffusion = SUPPORTED_DISCRETIZATIONS["uniform"](diffusion_steps, 0.).to(device)
        elif callable(discretization):
            self.t_diffusion = discretization(diffusion_steps, 0.).to(device)
        else:
            raise ValueError("discretization must be a callable or a string")

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ==================== Training: Straighten Flow ======================

    def loss(self, x0, x1=None, condition=None):

        # x1 is the samples of source distribution.
        # If x1 is None, then we assume x1 is from a standard Gaussian distribution.
        if x1 is None:
            x1 = torch.randn_like(x0)
        else:
            assert x0.shape == x1.shape, "x0 and x1 must have the same shape"

        t = torch.randint(self.diffusion_steps, (x0.shape[0],), device=self.device)
        t_c = self.t_diffusion[t]

        xt = t_c * x1 + (1 - t_c) * x0

        condition = self.model["condition"](condition) if condition is not None else None

        loss = (self.model["diffusion"](xt, t, condition) - (x0 - x1)) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

    def update(self, x0, condition=None, update_ema=True, x1=None, **kwargs):

        loss = self.loss(x0, x1, condition)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        log = {"loss": loss.item(), "grad_norm": grad_norm}

        return log

    # ==================== Sampling: Solving a straight ODE flow ======================

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            x1: torch.Tensor = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform",
            use_ema: bool = True,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        assert w_cg == 0.0 and condition_cg is None, "Rectified Flow does not support classifier-guidance."

        if x1 is None:
            x1 = torch.randn_like(prior) * temperature
        else:
            assert prior.shape == x1.shape, "prior and x1 must have the same shape"

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        xt = x1.clone()
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None

        # ===================== Sampling Schedule ====================
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    self.diffusion_steps, sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule(self.diffusion_steps, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        # ===================== Denoising Loop ========================

        for i in reversed(range(1, sample_steps + 1)):

            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.long, device=self.device)

            start_t, end_t = self.t_diffusion[t], self.t_diffusion[t - 1]
            delta_t = start_t - end_t

            # velocity
            if w_cfg != 0.0 and w_cfg != 1.0 and condition_vec_cfg is not None:
                repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]
                condition_vec_cfg = torch.cat([condition_vec_cfg, torch.zeros_like(condition_vec_cfg)], 0)
                vel_all = model["diffusion"](
                    xt.repeat(*repeat_dim), t.repeat(2), condition_vec_cfg)
                vel_cond, vel_uncond = vel_all.chunk(2, dim=0)
                vel = w_cfg * vel_cond + (1 - w_cfg) * vel_uncond
            elif w_cfg == 0.0 or condition_vec_cfg is None:
                vel = model["diffusion"](xt, t, None)
            else:
                vel = model["diffusion"](xt, t, condition_vec_cfg)

            # one-step update
            xt = xt + delta_t * vel

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log
