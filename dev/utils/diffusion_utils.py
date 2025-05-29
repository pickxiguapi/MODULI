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
from cleandiffuser.diffusion.basic import DiffusionModel

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

SUPPORTED_SOLVERS = [
    "ddpm", "ddim", "ode_dpmsolver++_1", "sde_dpmsolver++_1", "ode_dpmsolver++_2M"
]

def epstheta_to_xtheta(x, alpha, sigma, eps_theta):
    """
    x_theta = (x - sigma * eps_theta) / alpha
    """
    return (x - sigma * eps_theta) / alpha


def xtheta_to_epstheta(x, alpha, sigma, x_theta):
    """
    eps_theta = (x - alpha * x_theta) / sigma
    """
    return (x - alpha * x_theta) / sigma

class ContinuousVPSDEControlNet(DiffusionModel):

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_controlnet: BaseNNDiffusion,
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
            epsilon: float = 1e-3,

            discretization: Union[str, Callable] = "uniform",
            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            predict_noise: bool = True,

            device: Union[torch.device, str] = "cpu",

            # ------------------- ControlNet Params ------------------- #
            dropout_percent: float = None,
            dropout_points: int = None,
            pref_upper_bound: float = None,
            pref_lower_bound: float = None,
    ):
        super().__init__(
            nn_diffusion, nn_controlnet, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            0, ema_rate, optim_params, device)

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.x_max, self.x_min = x_max, x_min
        self.dropout_percent = dropout_percent
        self.dropout_points = dropout_points
        self.pref_upper_bound = pref_upper_bound
        self.pref_lower_bound = pref_lower_bound

        # ==================== ControlNet Pref Bounds ====================

        pref_ul_bounds = torch.Tensor([pref_lower_bound, pref_upper_bound])
        dropout_centers = torch.linspace(pref_lower_bound, pref_upper_bound, self.dropout_points + 2)[1:-1]
        dropout_lower_bounds = dropout_centers - (self.pref_upper_bound - self.pref_lower_bound) * self.dropout_percent / self.dropout_points / 2
        dropout_upper_bounds = dropout_centers + (self.pref_upper_bound - self.pref_lower_bound) * self.dropout_percent / self.dropout_points / 2
        self.pref_bounds = torch.cat([pref_ul_bounds, dropout_lower_bounds, dropout_upper_bounds], dim=-1).to(device)

        # ==================== Continuous Time-step Range ====================
        if noise_schedule == "cosine":
            self.t_diffusion = [epsilon, 0.9946]
        else:
            self.t_diffusion = [epsilon, 1.]

        # ===================== Noise Schedule ======================
        if isinstance(noise_schedule, str):
            if noise_schedule in SUPPORTED_NOISE_SCHEDULES.keys():
                self.noise_schedule_funcs = SUPPORTED_NOISE_SCHEDULES[noise_schedule]
                self.noise_schedule_params = noise_schedule_params
            else:
                raise ValueError(f"Noise schedule {noise_schedule} is not supported.")
        elif isinstance(noise_schedule, dict):
            self.noise_schedule_funcs = noise_schedule
            self.noise_schedule_params = noise_schedule_params
        else:
            raise ValueError("noise_schedule must be a callable or a string")

        # ===================== ControlNet ======================
        for name, param in self.model["diffusion"].named_parameters():
            param.requires_grad = False
        for name, param in self.model["condition"].named_parameters():
            param.requires_grad = False

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ==================== Training: Score Matching ======================

    def add_noise(self, x0, t=None, eps=None):

        t = (torch.rand((x0.shape[0],), device=self.device) *
             (self.t_diffusion[1] - self.t_diffusion[0]) + self.t_diffusion[0]) if t is None else t

        eps = torch.randn_like(x0) if eps is None else eps

        alpha, sigma = self.noise_schedule_funcs["forward"](t, **(self.noise_schedule_params or {}))
        alpha = at_least_ndim(alpha, x0.dim())
        sigma = at_least_ndim(sigma, x0.dim())

        xt = alpha * x0 + sigma * eps
        xt = (1. - self.fix_mask) * xt + self.fix_mask * x0

        return xt, t, eps

    def loss(self, x0, condition=None):

        xt, t, eps = self.add_noise(x0)

        # bias_size = torch.Tensor([torch.min(torch.abs(self.pref_bounds - condition[i, 0])) for i in range(condition.shape[0])]).reshape((-1, 1)).to(self.device)

        bias_size = torch.min(torch.tensor(0.1), torch.min(self.pref_upper_bound - condition[:, 0], condition[:, 0] - self.pref_lower_bound)).unsqueeze(1)
        # bias_size = 0.01
        bias_single = (torch.rand_like(condition[:, 0]).reshape((-1, 1)) * 2 - 1) * bias_size 

        bias = torch.cat([bias_single, -bias_single, torch.zeros_like(condition[:,-2:])], dim=-1)

        condition_pos = self.model["condition"](condition + bias) if condition is not None else None
        condition_neg = self.model["condition"](condition - bias) if condition is not None else None
        condition_center = self.model["condition"](condition) if condition is not None else None

        xt_cond_pos = self.model["diffusion"](xt, t, condition_pos)
        xt_cond_neg = self.model["diffusion"](xt, t, condition_neg)

        if np.random.rand() < 0.5:
            delta_x = self.model["controlnet"](xt, t)
        else:
            delta_x = self.model["controlnet"](xt, t, condition_center)

        loss = (delta_x * bias_single.unsqueeze(1) * 2 - (xt_cond_pos - xt_cond_neg)) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema["controlnet"].parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1. - self.ema_rate)
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema99["controlnet"].parameters()):
                p_ema.data.mul_(0.99).add_(p.data, alpha=0.01)
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema999["controlnet"].parameters()):
                p_ema.data.mul_(0.999).add_(p.data, alpha=0.001)
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema9999["controlnet"].parameters()):
                p_ema.data.mul_(0.9999).add_(p.data, alpha=0.0001)

    def update(self, x0, condition=None, update_ema=True, **kwargs):

        loss = self.loss(x0, condition)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        log = {"loss": loss.item(), "grad_norm": grad_norm}

        return log

    def update_classifier(self, x0, condition):

        xt, t, eps = self.add_noise(x0)

        log = self.classifier.update(xt, t, condition)

        return log

    # ==================== Sampling: Solving SDE/ODE ======================

    def classifier_guidance(
            self, xt, t, alpha, sigma,
            model, condition=None, w: float = 1.0,
            pred=None):
        """
        Guided Sampling CG:
        bar_eps = eps - w * sigma * grad
        bar_x0  = x0 + w * (sigma ** 2) * alpha * grad
        """
        if pred is None:
            pred = model["diffusion"](xt, t, None)
        if self.classifier is None or w == 0.0 or condition is None:
            return pred, None
        else:
            log_p, grad = self.classifier.gradients(xt.clone(), t, condition)
            if self.predict_noise:
                pred = pred - w * sigma * grad
            else:
                pred = pred + w * ((sigma ** 2) / alpha) * grad

        return pred, log_p

    def classifier_free_guidance(
            self, xt, t,
            model, condition=None, w: float = 1.0,
            pred=None, pred_uncond=None,
            requires_grad: bool = False,
            w_controlnet: torch.Tensor = None,
            w_controlnet_cond: float = 0.0):
        """
        Guided Sampling CFG:
        bar_eps = w * pred + (1 - w) * pred_uncond
        bar_x0  = w * pred + (1 - w) * pred_uncond
        """
        with torch.set_grad_enabled(requires_grad):
            if w != 0.0 and w != 1.0:
                if pred is None or pred_uncond is None:
                    b = xt.shape[0]
                    repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]
                    # condition = torch.cat([condition, torch.zeros_like(condition)], 0)

                    pred_all = model["diffusion"](
                        xt.repeat(*repeat_dim), t.repeat(2), torch.cat([condition, torch.zeros_like(condition)], 0))
                    pred, pred_uncond = pred_all[:b], pred_all[b:]
            elif w == 0.0:
                pred = 0.
                pred_uncond = model["diffusion"](xt, t, None)
            else:
                pred = model["diffusion"](xt, t, condition)
                pred_uncond = 0.

        if self.predict_noise or not self.predict_noise:
            bar_pred = w * pred + (1 - w) * pred_uncond
        else:
            bar_pred = pred
        
        if w_controlnet is not None:
            b = xt.shape[0]
            repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]

            controlnet_pred_all = model["controlnet"](xt.repeat(*repeat_dim), t.repeat(2), torch.cat([condition, torch.zeros_like(condition)], 0))
            controlnet_pred, controlnet_pred_uncond = controlnet_pred_all[:b], controlnet_pred_all[b:]
            delta_pred = w_controlnet_cond * controlnet_pred + (1 - w_controlnet_cond) * controlnet_pred_uncond
            
            bar_pred += delta_pred * w_controlnet.reshape((-1, 1, 1))
        return bar_pred

    def clip_prediction(self, pred, xt, alpha, sigma):
        """
        Clip the prediction at each sampling step to stablize the generation.
        (xt - alpha * x_max) / sigma <= eps <= (xt - alpha * x_min) / sigma
                               x_min <= x0  <= x_max
        """
        if self.clip_pred:
            if self.predict_noise:
                upper_bound = (xt - alpha * self.x_min) / sigma if self.x_min is not None else None
                lower_bound = (xt - alpha * self.x_max) / sigma if self.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
            else:
                pred = pred.clip(self.x_min, self.x_max)
        return pred

    def guided_sampling(
            self, xt, t, alpha, sigma,
            model,
            condition_cfg=None, w_cfg: float = 0.0,
            condition_cg=None, w_cg: float = 0.0,
            requires_grad: bool = False,
            w_controlnet: torch.Tensor = None,
            w_controlnet_cond: float = 0.0):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad, w_controlnet, w_controlnet_cond)

        pred, logp = self.classifier_guidance(
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        return pred, logp

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            # --------------- control weight -------------- #
            w_controlnet: torch.Tensor = None,
            w_controlnet_cond: float = 0.0,
            # ----------------- sampling ----------------- #
            solver: str = "ddpm",
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform_continuous",
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
        assert solver in SUPPORTED_SOLVERS, f"Solver {solver} is not supported."

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        xt = torch.randn_like(prior, device=self.device) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # ===================== Sampling Schedule ====================
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    self.t_diffusion, sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule(self.t_diffusion, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {}))
        logSNRs = torch.log(alphas / sigmas)
        hs = torch.zeros_like(logSNRs)
        hs[1:] = logSNRs[:-1] - logSNRs[1:]  # hs[0] is not correctly calculated, but it will not be used.
        stds = torch.zeros((sample_steps + 1,), device=self.device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()

        buffer = []

        # ===================== Denoising Loop ========================

        for i in reversed(range(1, sample_steps + 1)):

            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.float32, device=self.device)

            # guided sampling
            pred, logp = self.guided_sampling(
                xt, t, alphas[i], sigmas[i],
                model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad, w_controlnet, w_controlnet_cond)

            # clip the prediction
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            # transform to eps_theta
            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            x_theta = pred if not self.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)

            # one-step update
            if solver == "ddpm":
                xt = (
                        (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
                        (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta)
                if i > 1:
                    xt += (stds[i] * torch.randn_like(xt))

            elif solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)

            elif solver == "ode_dpmsolver++_1":
                xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "sde_dpmsolver++_1":
                xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                      alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                      sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            elif solver == "ode_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * D
                else:
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.classifier is not None:
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=torch.long, device=self.device)
                logp = self.classifier.logp(xt, t, condition_vec_cg)
            log["log_p"] = logp

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log
    
class ContinuousVPSDEControlNet3Obj(DiffusionModel):

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_controlnet: BaseNNDiffusion,
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
            epsilon: float = 1e-3,

            discretization: Union[str, Callable] = "uniform",
            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            predict_noise: bool = True,

            device: Union[torch.device, str] = "cpu",

            # ------------------- ControlNet Params ------------------- #
            dropout_percent: float = None,
            dropout_points: int = None,
            pref_upper_bound: float = None,
            pref_lower_bound: float = None,
    ):
        super().__init__(
            nn_diffusion, nn_controlnet, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            0, ema_rate, optim_params, device)

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.x_max, self.x_min = x_max, x_min
        self.dropout_percent = dropout_percent
        self.dropout_points = dropout_points
        self.pref_upper_bound = pref_upper_bound
        self.pref_lower_bound = pref_lower_bound

        # ==================== ControlNet Pref Bounds ====================

        pref_ul_bounds = torch.Tensor([pref_lower_bound, pref_upper_bound])
        dropout_centers = torch.linspace(pref_lower_bound, pref_upper_bound, self.dropout_points + 2)[1:-1]
        dropout_lower_bounds = dropout_centers - (self.pref_upper_bound - self.pref_lower_bound) * self.dropout_percent / self.dropout_points / 2
        dropout_upper_bounds = dropout_centers + (self.pref_upper_bound - self.pref_lower_bound) * self.dropout_percent / self.dropout_points / 2
        self.pref_bounds = torch.cat([pref_ul_bounds, dropout_lower_bounds, dropout_upper_bounds], dim=-1).to(device)

        # ==================== Continuous Time-step Range ====================
        if noise_schedule == "cosine":
            self.t_diffusion = [epsilon, 0.9946]
        else:
            self.t_diffusion = [epsilon, 1.]

        # ===================== Noise Schedule ======================
        if isinstance(noise_schedule, str):
            if noise_schedule in SUPPORTED_NOISE_SCHEDULES.keys():
                self.noise_schedule_funcs = SUPPORTED_NOISE_SCHEDULES[noise_schedule]
                self.noise_schedule_params = noise_schedule_params
            else:
                raise ValueError(f"Noise schedule {noise_schedule} is not supported.")
        elif isinstance(noise_schedule, dict):
            self.noise_schedule_funcs = noise_schedule
            self.noise_schedule_params = noise_schedule_params
        else:
            raise ValueError("noise_schedule must be a callable or a string")

        # ===================== ControlNet ======================
        for name, param in self.model["diffusion"].named_parameters():
            param.requires_grad = False
        for name, param in self.model["condition"].named_parameters():
            param.requires_grad = False

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ==================== Training: Score Matching ======================

    def add_noise(self, x0, t=None, eps=None):

        t = (torch.rand((x0.shape[0],), device=self.device) *
             (self.t_diffusion[1] - self.t_diffusion[0]) + self.t_diffusion[0]) if t is None else t

        eps = torch.randn_like(x0) if eps is None else eps

        alpha, sigma = self.noise_schedule_funcs["forward"](t, **(self.noise_schedule_params or {}))
        alpha = at_least_ndim(alpha, x0.dim())
        sigma = at_least_ndim(sigma, x0.dim())

        xt = alpha * x0 + sigma * eps
        xt = (1. - self.fix_mask) * xt + self.fix_mask * x0

        return xt, t, eps

    def loss(self, x0, condition=None):

        xt, t, eps = self.add_noise(x0)

        # bias_size = torch.Tensor([torch.min(torch.abs(self.pref_bounds - condition[i, 0])) for i in range(condition.shape[0])]).reshape((-1, 1)).to(self.device)

        # bias_size = torch.min(torch.tensor(0.1), torch.min(self.pref_upper_bound - condition[:, 0], condition[:, 0] - self.pref_lower_bound)).unsqueeze(1)
        bias_size = 0.1
        bias_single = (torch.rand_like(condition[:, 0]).reshape((-1, 1)) * 2 - 1) * bias_size 
        bias_single_2 = (torch.rand_like(condition[:, 0]).reshape((-1, 1)) * 2 - 1) * bias_size 

        bias = torch.cat([bias_single, bias_single_2, -(bias_single+bias_single_2), torch.zeros_like(condition[:,-3:])], dim=-1)

        condition_pos = self.model["condition"](condition + bias) if condition is not None else None
        condition_neg = self.model["condition"](condition - bias) if condition is not None else None
        condition_center = self.model["condition"](condition) if condition is not None else None

        xt_cond_pos = self.model["diffusion"](xt, t, condition_pos)
        xt_cond_neg = self.model["diffusion"](xt, t, condition_neg)

        if np.random.rand() < 0.5:
            delta_x = self.model["controlnet"](xt, t)
        else:
            delta_x = self.model["controlnet"](xt, t, condition_center)

        loss = (delta_x * bias_single.unsqueeze(1) * 2 - (xt_cond_pos - xt_cond_neg)) ** 2
        # if self.predict_noise:
        #     loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        # else:
        #     loss = (self.model["diffusion"](xt, t, condition) - x0) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema["controlnet"].parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1. - self.ema_rate)
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema99["controlnet"].parameters()):
                p_ema.data.mul_(0.99).add_(p.data, alpha=0.01)
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema999["controlnet"].parameters()):
                p_ema.data.mul_(0.999).add_(p.data, alpha=0.001)
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema9999["controlnet"].parameters()):
                p_ema.data.mul_(0.9999).add_(p.data, alpha=0.0001)

    def update(self, x0, condition=None, update_ema=True, **kwargs):

        loss = self.loss(x0, condition)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        log = {"loss": loss.item(), "grad_norm": grad_norm}

        return log

    def update_classifier(self, x0, condition):

        xt, t, eps = self.add_noise(x0)

        log = self.classifier.update(xt, t, condition)

        return log

    # ==================== Sampling: Solving SDE/ODE ======================

    def classifier_guidance(
            self, xt, t, alpha, sigma,
            model, condition=None, w: float = 1.0,
            pred=None):
        """
        Guided Sampling CG:
        bar_eps = eps - w * sigma * grad
        bar_x0  = x0 + w * (sigma ** 2) * alpha * grad
        """
        if pred is None:
            pred = model["diffusion"](xt, t, None)
        if self.classifier is None or w == 0.0 or condition is None:
            return pred, None
        else:
            log_p, grad = self.classifier.gradients(xt.clone(), t, condition)
            if self.predict_noise:
                pred = pred - w * sigma * grad
            else:
                pred = pred + w * ((sigma ** 2) / alpha) * grad

        return pred, log_p

    def classifier_free_guidance(
            self, xt, t,
            model, condition=None, w: float = 1.0,
            pred=None, pred_uncond=None,
            requires_grad: bool = False,
            w_controlnet: torch.Tensor = None,
            w_controlnet_cond: float = 0.0):
        """
        Guided Sampling CFG:
        bar_eps = w * pred + (1 - w) * pred_uncond
        bar_x0  = w * pred + (1 - w) * pred_uncond
        """
        with torch.set_grad_enabled(requires_grad):
            if w != 0.0 and w != 1.0:
                if pred is None or pred_uncond is None:
                    b = xt.shape[0]
                    repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]
                    # condition = torch.cat([condition, torch.zeros_like(condition)], 0)

                    pred_all = model["diffusion"](
                        xt.repeat(*repeat_dim), t.repeat(2), torch.cat([condition, torch.zeros_like(condition)], 0))
                    pred, pred_uncond = pred_all[:b], pred_all[b:]
            elif w == 0.0:
                pred = 0.
                pred_uncond = model["diffusion"](xt, t, None)
            else:
                pred = model["diffusion"](xt, t, condition)
                pred_uncond = 0.

        if self.predict_noise or not self.predict_noise:
            bar_pred = w * pred + (1 - w) * pred_uncond
        else:
            bar_pred = pred
        
        if w_controlnet is not None:
            b = xt.shape[0]
            repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]

            controlnet_pred_all = model["controlnet"](xt.repeat(*repeat_dim), t.repeat(2), torch.cat([condition, torch.zeros_like(condition)], 0))
            controlnet_pred, controlnet_pred_uncond = controlnet_pred_all[:b], controlnet_pred_all[b:]
            delta_pred = w_controlnet_cond * controlnet_pred + (1 - w_controlnet_cond) * controlnet_pred_uncond
            
            bar_pred += delta_pred * w_controlnet.reshape((-1, 1, 1))
        return bar_pred

    def clip_prediction(self, pred, xt, alpha, sigma):
        """
        Clip the prediction at each sampling step to stablize the generation.
        (xt - alpha * x_max) / sigma <= eps <= (xt - alpha * x_min) / sigma
                               x_min <= x0  <= x_max
        """
        if self.clip_pred:
            if self.predict_noise:
                upper_bound = (xt - alpha * self.x_min) / sigma if self.x_min is not None else None
                lower_bound = (xt - alpha * self.x_max) / sigma if self.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
            else:
                pred = pred.clip(self.x_min, self.x_max)
        return pred

    def guided_sampling(
            self, xt, t, alpha, sigma,
            model,
            condition_cfg=None, w_cfg: float = 0.0,
            condition_cg=None, w_cg: float = 0.0,
            requires_grad: bool = False,
            w_controlnet: torch.Tensor = None,
            w_controlnet_cond: float = 0.0):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad, w_controlnet, w_controlnet_cond)

        pred, logp = self.classifier_guidance(
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        return pred, logp

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            # --------------- control weight -------------- #
            w_controlnet: torch.Tensor = None,
            w_controlnet_cond: float = 0.0,
            # ----------------- sampling ----------------- #
            solver: str = "ddpm",
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform_continuous",
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
        assert solver in SUPPORTED_SOLVERS, f"Solver {solver} is not supported."

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        xt = torch.randn_like(prior, device=self.device) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # ===================== Sampling Schedule ====================
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    self.t_diffusion, sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule(self.t_diffusion, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {}))
        logSNRs = torch.log(alphas / sigmas)
        hs = torch.zeros_like(logSNRs)
        hs[1:] = logSNRs[:-1] - logSNRs[1:]  # hs[0] is not correctly calculated, but it will not be used.
        stds = torch.zeros((sample_steps + 1,), device=self.device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()

        buffer = []

        # ===================== Denoising Loop ========================

        for i in reversed(range(1, sample_steps + 1)):

            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.float32, device=self.device)

            # guided sampling
            pred, logp = self.guided_sampling(
                xt, t, alphas[i], sigmas[i],
                model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad, w_controlnet, w_controlnet_cond)

            # clip the prediction
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            # transform to eps_theta
            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            x_theta = pred if not self.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)

            # one-step update
            if solver == "ddpm":
                xt = (
                        (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
                        (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta)
                if i > 1:
                    xt += (stds[i] * torch.randn_like(xt))

            elif solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)

            elif solver == "ode_dpmsolver++_1":
                xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "sde_dpmsolver++_1":
                xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                      alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                      sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            elif solver == "ode_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * D
                else:
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.classifier is not None:
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=torch.long, device=self.device)
                logp = self.classifier.logp(xt, t, condition_vec_cg)
            log["log_p"] = logp

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log

class ContinuousVPSDEControlNetLinear(DiffusionModel):

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_controlnet: BaseNNDiffusion,
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
            epsilon: float = 1e-3,

            discretization: Union[str, Callable] = "uniform",
            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            predict_noise: bool = True,

            device: Union[torch.device, str] = "cpu",

            # ------------------- ControlNet Params ------------------- #
            dropout_percent: float = None,
            dropout_points: int = None,
            pref_upper_bound: float = None,
            pref_lower_bound: float = None,
    ):
        super().__init__(
            nn_diffusion, nn_controlnet, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            0, ema_rate, optim_params, device)

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.x_max, self.x_min = x_max, x_min
        self.dropout_percent = dropout_percent
        self.dropout_points = dropout_points
        self.pref_upper_bound = pref_upper_bound
        self.pref_lower_bound = pref_lower_bound

        # ==================== ControlNet Pref Bounds ====================

        pref_ul_bounds = torch.Tensor([pref_lower_bound, pref_upper_bound])
        dropout_centers = torch.linspace(pref_lower_bound, pref_upper_bound, self.dropout_points + 2)[1:-1]
        dropout_lower_bounds = dropout_centers - (self.pref_upper_bound - self.pref_lower_bound) * self.dropout_percent / self.dropout_points / 2
        dropout_upper_bounds = dropout_centers + (self.pref_upper_bound - self.pref_lower_bound) * self.dropout_percent / self.dropout_points / 2
        self.pref_bounds = torch.cat([pref_ul_bounds, dropout_lower_bounds, dropout_upper_bounds], dim=-1).to(device)

        # ==================== Continuous Time-step Range ====================
        if noise_schedule == "cosine":
            self.t_diffusion = [epsilon, 0.9946]
        else:
            self.t_diffusion = [epsilon, 1.]

        # ===================== Noise Schedule ======================
        if isinstance(noise_schedule, str):
            if noise_schedule in SUPPORTED_NOISE_SCHEDULES.keys():
                self.noise_schedule_funcs = SUPPORTED_NOISE_SCHEDULES[noise_schedule]
                self.noise_schedule_params = noise_schedule_params
            else:
                raise ValueError(f"Noise schedule {noise_schedule} is not supported.")
        elif isinstance(noise_schedule, dict):
            self.noise_schedule_funcs = noise_schedule
            self.noise_schedule_params = noise_schedule_params
        else:
            raise ValueError("noise_schedule must be a callable or a string")

        # ===================== ControlNet ======================
        for name, param in self.model["diffusion"].named_parameters():
            param.requires_grad = False
        for name, param in self.model["condition"].named_parameters():
            param.requires_grad = False

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ==================== Training: Score Matching ======================

    def add_noise(self, x0, t=None, eps=None):

        t = (torch.rand((x0.shape[0],), device=self.device) *
             (self.t_diffusion[1] - self.t_diffusion[0]) + self.t_diffusion[0]) if t is None else t

        eps = torch.randn_like(x0) if eps is None else eps

        alpha, sigma = self.noise_schedule_funcs["forward"](t, **(self.noise_schedule_params or {}))
        alpha = at_least_ndim(alpha, x0.dim())
        sigma = at_least_ndim(sigma, x0.dim())

        xt = alpha * x0 + sigma * eps
        xt = (1. - self.fix_mask) * xt + self.fix_mask * x0

        return xt, t, eps

    def loss(self, x0, condition=None):

        xt, t, eps = self.add_noise(x0)

        # bias_size = torch.Tensor([torch.min(torch.abs(self.pref_bounds - condition[i, 0])) for i in range(condition.shape[0])]).reshape((-1, 1)).to(self.device)

        bias_size = torch.min(torch.tensor(0.1), torch.min(self.pref_upper_bound - condition[:, 0], condition[:, 0] - self.pref_lower_bound)).unsqueeze(1)
        # bias_size = 0.01
        bias_single = (torch.rand_like(condition[:, 0]).reshape((-1, 1)) * 2 - 1) * bias_size 

        bias = torch.cat([bias_single, -bias_single, torch.zeros_like(condition[:,-2:])], dim=-1)

        condition_pos = self.model["condition"](condition + bias) if condition is not None else None
        condition_neg = self.model["condition"](condition - bias) if condition is not None else None
        condition_center = self.model["condition"](condition) if condition is not None else None

        xt_cond_pos = self.model["diffusion"](xt, t, condition_pos)
        xt_cond_neg = self.model["diffusion"](xt, t, condition_neg)

        if np.random.rand() < 0.5:
            delta_x = self.model["controlnet"](xt, t)
        else:
            delta_x = self.model["controlnet"](xt, t, condition_center)

        loss = (delta_x * bias_single.unsqueeze(1) * 2 - (xt_cond_pos - xt_cond_neg)) ** 2
        # if self.predict_noise:
        #     loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        # else:
        #     loss = (self.model["diffusion"](xt, t, condition) - x0) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema["controlnet"].parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1. - self.ema_rate)
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema99["controlnet"].parameters()):
                p_ema.data.mul_(0.99).add_(p.data, alpha=0.01)
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema999["controlnet"].parameters()):
                p_ema.data.mul_(0.999).add_(p.data, alpha=0.001)
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema9999["controlnet"].parameters()):
                p_ema.data.mul_(0.9999).add_(p.data, alpha=0.0001)

    def update(self, x0, condition=None, update_ema=True, **kwargs):

        loss = self.loss(x0, condition)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        log = {"loss": loss.item(), "grad_norm": grad_norm}

        return log

    def update_classifier(self, x0, condition):

        xt, t, eps = self.add_noise(x0)

        log = self.classifier.update(xt, t, condition)

        return log

    # ==================== Sampling: Solving SDE/ODE ======================
    def classifier_guidance(
            self, xt, t, alpha, sigma,
            model, condition=None, w: float = 1.0,
            pred=None):
        """
        Guided Sampling CG:
        bar_eps = eps - w * sigma * grad
        bar_x0  = x0 + w * (sigma ** 2) * alpha * grad
        """
        if pred is None:
            pred = model["diffusion"](xt, t, None)
        if self.classifier is None or w == 0.0 or condition is None:
            return pred, None
        else:
            log_p, grad = self.classifier.gradients(xt.clone(), t, condition)
            if self.predict_noise:
                pred = pred - w * sigma * grad
            else:
                pred = pred + w * ((sigma ** 2) / alpha) * grad

        return pred, log_p

    def classifier_free_guidance(
            self, xt, t,
            model, condition=None, w: float = 1.0,
            pred=None, pred_uncond=None,
            requires_grad: bool = False,
            w_controlnet: torch.Tensor = None,
            w_controlnet_cond: float = 0.0,
            sample_step: int = 0,
            total_sample_steps: int = 0):
        """
        Guided Sampling CFG:
        bar_eps = w * pred + (1 - w) * pred_uncond
        bar_x0  = w * pred + (1 - w) * pred_uncond
        """

        with torch.set_grad_enabled(requires_grad):
            if w != 0.0 and w != 1.0:
                if pred is None or pred_uncond is None:
                    b = xt.shape[0]
                    repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]
                    # condition = torch.cat([condition, torch.zeros_like(condition)], 0)

                    pred_all = model["diffusion"](
                        xt.repeat(*repeat_dim), t.repeat(2), torch.cat([condition, torch.zeros_like(condition)], 0))
                    pred, pred_uncond = pred_all[:b], pred_all[b:]
            elif w == 0.0:
                pred = 0.
                pred_uncond = model["diffusion"](xt, t, None)
            else:
                pred = model["diffusion"](xt, t, condition)
                pred_uncond = 0.

        if self.predict_noise or not self.predict_noise:
            bar_pred = w * pred + (1 - w) * pred_uncond
        else:
            bar_pred = pred
        
        if w_controlnet is not None:
            b = xt.shape[0]
            repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]

            controlnet_pred_all = model["controlnet"](xt.repeat(*repeat_dim), t.repeat(2), torch.cat([condition, torch.zeros_like(condition)], 0))
            controlnet_pred, controlnet_pred_uncond = controlnet_pred_all[:b], controlnet_pred_all[b:]
            delta_pred = w_controlnet_cond * controlnet_pred + (1 - w_controlnet_cond) * controlnet_pred_uncond
            
            bar_pred += delta_pred * w_controlnet.reshape((-1, 1, 1)) / total_sample_steps
        return bar_pred

    def clip_prediction(self, pred, xt, alpha, sigma):
        """
        Clip the prediction at each sampling step to stablize the generation.
        (xt - alpha * x_max) / sigma <= eps <= (xt - alpha * x_min) / sigma
                               x_min <= x0  <= x_max
        """
        if self.clip_pred:
            if self.predict_noise:
                upper_bound = (xt - alpha * self.x_min) / sigma if self.x_min is not None else None
                lower_bound = (xt - alpha * self.x_max) / sigma if self.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
            else:
                pred = pred.clip(self.x_min, self.x_max)
        return pred

    def guided_sampling(
            self, xt, t, alpha, sigma,
            model,
            condition_cfg=None, w_cfg: float = 0.0,
            condition_cg=None, w_cg: float = 0.0,
            requires_grad: bool = False,
            w_controlnet: torch.Tensor = None,
            w_controlnet_cond: float = 0.0,
            sample_step: int = 0,
            total_sample_steps: int = 0):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad, w_controlnet, w_controlnet_cond, sample_step, total_sample_steps)

        pred, logp = self.classifier_guidance(
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        return pred, logp

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            # --------------- control weight -------------- #
            w_controlnet: torch.Tensor = None,
            w_controlnet_cond: float = 0.0,
            # ----------------- sampling ----------------- #
            solver: str = "ddpm",
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform_continuous",
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
        assert solver in SUPPORTED_SOLVERS, f"Solver {solver} is not supported."

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        xt = torch.randn_like(prior, device=self.device) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        # with torch.set_grad_enabled(requires_grad):
        #     condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
        #     condition_vec_cg = condition_cg

        # ===================== Sampling Schedule ====================
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    self.t_diffusion, sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule(self.t_diffusion, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {}))
        logSNRs = torch.log(alphas / sigmas)
        hs = torch.zeros_like(logSNRs)
        hs[1:] = logSNRs[:-1] - logSNRs[1:]  # hs[0] is not correctly calculated, but it will not be used.
        stds = torch.zeros((sample_steps + 1,), device=self.device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()

        buffer = []

        # ===================== Denoising Loop ========================

        for i in reversed(range(1, sample_steps + 1)):
            with torch.set_grad_enabled(requires_grad):
                step_condition_cfg = condition_cfg+torch.cat([w_controlnet.reshape((-1, 1)), -w_controlnet.reshape((-1, 1)), torch.zeros((w_controlnet.shape[0], 2), device=self.device)], dim=-1)*((sample_steps-i)/sample_steps)
                condition_vec_cfg = model["condition"](step_condition_cfg, mask_cfg) if condition_cfg is not None else None
                condition_vec_cg = condition_cg
            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.float32, device=self.device)

            # guided sampling
            pred, logp = self.guided_sampling(
                xt, t, alphas[i], sigmas[i],
                model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad, w_controlnet, w_controlnet_cond, i, sample_steps)

            # clip the prediction
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            # transform to eps_theta
            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            x_theta = pred if not self.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)

            # one-step update
            if solver == "ddpm":
                xt = (
                        (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
                        (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta)
                if i > 1:
                    xt += (stds[i] * torch.randn_like(xt))

            elif solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)

            elif solver == "ode_dpmsolver++_1":
                xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "sde_dpmsolver++_1":
                xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                      alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                      sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            elif solver == "ode_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * D
                else:
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.classifier is not None:
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=torch.long, device=self.device)
                logp = self.classifier.logp(xt, t, condition_vec_cg)
            log["log_p"] = logp

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log
    
class ContinuousVPSDEControlNet3ObjLinear(DiffusionModel):

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_controlnet: BaseNNDiffusion,
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
            epsilon: float = 1e-3,

            discretization: Union[str, Callable] = "uniform",
            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            predict_noise: bool = True,

            device: Union[torch.device, str] = "cpu",

            # ------------------- ControlNet Params ------------------- #
            dropout_percent: float = None,
            dropout_points: int = None,
            pref_upper_bound: float = None,
            pref_lower_bound: float = None,
    ):
        super().__init__(
            nn_diffusion, nn_controlnet, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            0, ema_rate, optim_params, device)

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.x_max, self.x_min = x_max, x_min
        self.dropout_percent = dropout_percent
        self.dropout_points = dropout_points
        self.pref_upper_bound = pref_upper_bound
        self.pref_lower_bound = pref_lower_bound

        # ==================== ControlNet Pref Bounds ====================

        pref_ul_bounds = torch.Tensor([pref_lower_bound, pref_upper_bound])
        dropout_centers = torch.linspace(pref_lower_bound, pref_upper_bound, self.dropout_points + 2)[1:-1]
        dropout_lower_bounds = dropout_centers - (self.pref_upper_bound - self.pref_lower_bound) * self.dropout_percent / self.dropout_points / 2
        dropout_upper_bounds = dropout_centers + (self.pref_upper_bound - self.pref_lower_bound) * self.dropout_percent / self.dropout_points / 2
        self.pref_bounds = torch.cat([pref_ul_bounds, dropout_lower_bounds, dropout_upper_bounds], dim=-1).to(device)

        # ==================== Continuous Time-step Range ====================
        if noise_schedule == "cosine":
            self.t_diffusion = [epsilon, 0.9946]
        else:
            self.t_diffusion = [epsilon, 1.]

        # ===================== Noise Schedule ======================
        if isinstance(noise_schedule, str):
            if noise_schedule in SUPPORTED_NOISE_SCHEDULES.keys():
                self.noise_schedule_funcs = SUPPORTED_NOISE_SCHEDULES[noise_schedule]
                self.noise_schedule_params = noise_schedule_params
            else:
                raise ValueError(f"Noise schedule {noise_schedule} is not supported.")
        elif isinstance(noise_schedule, dict):
            self.noise_schedule_funcs = noise_schedule
            self.noise_schedule_params = noise_schedule_params
        else:
            raise ValueError("noise_schedule must be a callable or a string")

        # ===================== ControlNet ======================
        for name, param in self.model["diffusion"].named_parameters():
            param.requires_grad = False
        for name, param in self.model["condition"].named_parameters():
            param.requires_grad = False

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ==================== Training: Score Matching ======================

    def add_noise(self, x0, t=None, eps=None):

        t = (torch.rand((x0.shape[0],), device=self.device) *
             (self.t_diffusion[1] - self.t_diffusion[0]) + self.t_diffusion[0]) if t is None else t

        eps = torch.randn_like(x0) if eps is None else eps

        alpha, sigma = self.noise_schedule_funcs["forward"](t, **(self.noise_schedule_params or {}))
        alpha = at_least_ndim(alpha, x0.dim())
        sigma = at_least_ndim(sigma, x0.dim())

        xt = alpha * x0 + sigma * eps
        xt = (1. - self.fix_mask) * xt + self.fix_mask * x0

        return xt, t, eps

    def loss(self, x0, condition=None):

        xt, t, eps = self.add_noise(x0)

        # bias_size = torch.Tensor([torch.min(torch.abs(self.pref_bounds - condition[i, 0])) for i in range(condition.shape[0])]).reshape((-1, 1)).to(self.device)

        # bias_size = torch.min(torch.tensor(0.1), torch.min(self.pref_upper_bound - condition[:, 0], condition[:, 0] - self.pref_lower_bound)).unsqueeze(1)
        bias_size = 0.1
        bias_single = (torch.rand_like(condition[:, 0]).reshape((-1, 1)) * 2 - 1) * bias_size 
        bias_single_2 = (torch.rand_like(condition[:, 0]).reshape((-1, 1)) * 2 - 1) * bias_size 

        bias = torch.cat([bias_single, bias_single_2, -(bias_single+bias_single_2), torch.zeros_like(condition[:,-3:])], dim=-1)

        condition_pos = self.model["condition"](condition + bias) if condition is not None else None
        condition_neg = self.model["condition"](condition - bias) if condition is not None else None
        condition_center = self.model["condition"](condition) if condition is not None else None

        xt_cond_pos = self.model["diffusion"](xt, t, condition_pos)
        xt_cond_neg = self.model["diffusion"](xt, t, condition_neg)

        if np.random.rand() < 0.5:
            delta_x = self.model["controlnet"](xt, t)
        else:
            delta_x = self.model["controlnet"](xt, t, condition_center)

        loss = (delta_x * bias_single.unsqueeze(1) * 2 - (xt_cond_pos - xt_cond_neg)) ** 2
        # if self.predict_noise:
        #     loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        # else:
        #     loss = (self.model["diffusion"](xt, t, condition) - x0) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema["controlnet"].parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1. - self.ema_rate)
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema99["controlnet"].parameters()):
                p_ema.data.mul_(0.99).add_(p.data, alpha=0.01)
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema999["controlnet"].parameters()):
                p_ema.data.mul_(0.999).add_(p.data, alpha=0.001)
            for p, p_ema in zip(self.model["controlnet"].parameters(), self.model_ema9999["controlnet"].parameters()):
                p_ema.data.mul_(0.9999).add_(p.data, alpha=0.0001)

    def update(self, x0, condition=None, update_ema=True, **kwargs):

        loss = self.loss(x0, condition)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        log = {"loss": loss.item(), "grad_norm": grad_norm}

        return log

    def update_classifier(self, x0, condition):

        xt, t, eps = self.add_noise(x0)

        log = self.classifier.update(xt, t, condition)

        return log

    # ==================== Sampling: Solving SDE/ODE ======================

    def classifier_guidance(
            self, xt, t, alpha, sigma,
            model, condition=None, w: float = 1.0,
            pred=None):
        """
        Guided Sampling CG:
        bar_eps = eps - w * sigma * grad
        bar_x0  = x0 + w * (sigma ** 2) * alpha * grad
        """
        if pred is None:
            pred = model["diffusion"](xt, t, None)
        if self.classifier is None or w == 0.0 or condition is None:
            return pred, None
        else:
            log_p, grad = self.classifier.gradients(xt.clone(), t, condition)
            if self.predict_noise:
                pred = pred - w * sigma * grad
            else:
                pred = pred + w * ((sigma ** 2) / alpha) * grad

        return pred, log_p

    def classifier_free_guidance(
            self, xt, t,
            model, condition=None, w: float = 1.0,
            pred=None, pred_uncond=None,
            requires_grad: bool = False,
            w_controlnet: torch.Tensor = None,
            w_controlnet_cond: float = 0.0,
            sample_step: int = 0,
            total_sample_steps: int = 0):
        """
        Guided Sampling CFG:
        bar_eps = w * pred + (1 - w) * pred_uncond
        bar_x0  = w * pred + (1 - w) * pred_uncond
        """

        with torch.set_grad_enabled(requires_grad):
            if w != 0.0 and w != 1.0:
                if pred is None or pred_uncond is None:
                    b = xt.shape[0]
                    repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]
                    # condition = torch.cat([condition, torch.zeros_like(condition)], 0)

                    pred_all = model["diffusion"](
                        xt.repeat(*repeat_dim), t.repeat(2), torch.cat([condition, torch.zeros_like(condition)], 0))
                    pred, pred_uncond = pred_all[:b], pred_all[b:]
            elif w == 0.0:
                pred = 0.
                pred_uncond = model["diffusion"](xt, t, None)
            else:
                pred = model["diffusion"](xt, t, condition)
                pred_uncond = 0.

        if self.predict_noise or not self.predict_noise:
            bar_pred = w * pred + (1 - w) * pred_uncond
        else:
            bar_pred = pred
        
        if w_controlnet is not None:
            b = xt.shape[0]
            repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]

            controlnet_pred_all = model["controlnet"](xt.repeat(*repeat_dim), t.repeat(2), torch.cat([condition, torch.zeros_like(condition)], 0))
            controlnet_pred, controlnet_pred_uncond = controlnet_pred_all[:b], controlnet_pred_all[b:]
            delta_pred = w_controlnet_cond * controlnet_pred + (1 - w_controlnet_cond) * controlnet_pred_uncond
            
            bar_pred += delta_pred * w_controlnet.reshape((-1, 1, 1)) / total_sample_steps
        return bar_pred

    def clip_prediction(self, pred, xt, alpha, sigma):
        """
        Clip the prediction at each sampling step to stablize the generation.
        (xt - alpha * x_max) / sigma <= eps <= (xt - alpha * x_min) / sigma
                               x_min <= x0  <= x_max
        """
        if self.clip_pred:
            if self.predict_noise:
                upper_bound = (xt - alpha * self.x_min) / sigma if self.x_min is not None else None
                lower_bound = (xt - alpha * self.x_max) / sigma if self.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
            else:
                pred = pred.clip(self.x_min, self.x_max)
        return pred

    def guided_sampling(
            self, xt, t, alpha, sigma,
            model,
            condition_cfg=None, w_cfg: float = 0.0,
            condition_cg=None, w_cg: float = 0.0,
            requires_grad: bool = False,
            w_controlnet: torch.Tensor = None,
            w_controlnet_cond: float = 0.0,
            sample_step: int = 0,
            total_sample_steps: int = 0):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad, w_controlnet, w_controlnet_cond, sample_step, total_sample_steps)

        pred, logp = self.classifier_guidance(
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        return pred, logp

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            # --------------- control weight -------------- #
            w_controlnet: torch.Tensor = None,
            w_controlnet_cond: float = 0.0,
            # ----------------- sampling ----------------- #
            solver: str = "ddpm",
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform_continuous",
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
        assert solver in SUPPORTED_SOLVERS, f"Solver {solver} is not supported."

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        xt = torch.randn_like(prior, device=self.device) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        # with torch.set_grad_enabled(requires_grad):
        #     condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
        #     condition_vec_cg = condition_cg

        # ===================== Sampling Schedule ====================
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    self.t_diffusion, sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule(self.t_diffusion, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {}))
        logSNRs = torch.log(alphas / sigmas)
        hs = torch.zeros_like(logSNRs)
        hs[1:] = logSNRs[:-1] - logSNRs[1:]  # hs[0] is not correctly calculated, but it will not be used.
        stds = torch.zeros((sample_steps + 1,), device=self.device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()

        buffer = []

        # ===================== Denoising Loop ========================

        for i in reversed(range(1, sample_steps + 1)):
            with torch.set_grad_enabled(requires_grad):
                step_condition_cfg = condition_cfg+torch.cat([w_controlnet.reshape((-1, 1)), -w_controlnet.reshape((-1, 1))/2, -w_controlnet.reshape((-1, 1))/2, torch.zeros((w_controlnet.shape[0], 3), device=self.device)], dim=-1)*((sample_steps-i)/sample_steps)
                condition_vec_cfg = model["condition"](step_condition_cfg, mask_cfg) if condition_cfg is not None else None
                condition_vec_cg = condition_cg

            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.float32, device=self.device)

            # guided sampling
            pred, logp = self.guided_sampling(
                xt, t, alphas[i], sigmas[i],
                model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad, w_controlnet, w_controlnet_cond, i, sample_steps)
            # clip the prediction
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            # transform to eps_theta
            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            x_theta = pred if not self.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)

            # one-step update
            if solver == "ddpm":
                xt = (
                        (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
                        (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta)
                if i > 1:
                    xt += (stds[i] * torch.randn_like(xt))

            elif solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)

            elif solver == "ode_dpmsolver++_1":
                xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "sde_dpmsolver++_1":
                xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                      alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                      sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            elif solver == "ode_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * D
                else:
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.classifier is not None:
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=torch.long, device=self.device)
                logp = self.classifier.logp(xt, t, condition_vec_cg)
            log["log_p"] = logp

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log