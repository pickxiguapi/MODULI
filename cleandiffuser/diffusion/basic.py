from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.utils import to_tensor
from cleandiffuser.classifier import BasicClassifier
from cleandiffuser.nn_condition import BaseNNCondition, IdentityCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion


class DiffusionModel:

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            # NN backbone for the diffusion model
            nn_diffusion: BaseNNDiffusion,
            nn_controlnet: Optional[BaseNNCondition] = None,
            # Add a condition-process NN to enable classifier-free-guidance
            nn_condition: Optional[BaseNNCondition] = None,

            # ----------------- Masks ----------------- #
            # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`

            # ------------------ Plugs ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier: Optional[BasicClassifier] = None,

            # ------------------ Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            diffusion_steps: int = 1000,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,

            device: Union[torch.device, str] = "cpu"
    ):
        if optim_params is None:
            optim_params = {"lr": 2e-4, "weight_decay": 1e-5}

        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.diffusion_steps = diffusion_steps
        self.ema_rate = ema_rate

        # nn_condition is None means that the model is not conditioned on any input.
        if nn_condition is None:
            nn_condition = IdentityCondition()

        # In the code implementation of Diffusion models, it is common to maintain an exponential
        # moving average (EMA) version of the model for inference, as it has been observed that
        # this approach can result in more stable generation outcomes.

        if nn_controlnet is None:
            self.model = nn.ModuleDict({
                "diffusion": nn_diffusion.to(self.device),
                "condition": nn_condition.to(self.device)})
        else:
            self.model = nn.ModuleDict({
                "diffusion": nn_diffusion.to(self.device),
                "controlnet": nn_controlnet.to(self.device), 
                "condition": nn_condition.to(self.device)})
        self.model_ema = deepcopy(self.model).requires_grad_(False)
        self.model_ema99 = deepcopy(self.model).requires_grad_(False)
        self.model_ema999 = deepcopy(self.model).requires_grad_(False)
        self.model_ema9999 = deepcopy(self.model).requires_grad_(False)

        self.model.train()
        self.model_ema.eval()
        self.model_ema99.eval()
        self.model_ema999.eval()
        self.model_ema9999.eval()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optim_params)

        self.classifier = classifier

        self.fix_mask = to_tensor(fix_mask, self.device)[None, ] if fix_mask is not None else 0.
        self.loss_weight = to_tensor(loss_weight, self.device)[None, ] if loss_weight is not None else 1.

        # self.default_x_shape = fix_mask.shape if fix_mask != 0. else None

    def train(self):
        self.model.train()
        if self.classifier is not None:
            self.classifier.model.train()

    def eval(self):
        self.model.eval()
        if self.classifier is not None:
            self.classifier.model.eval()

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1. - self.ema_rate)
            for p, p_ema in zip(self.model.parameters(), self.model_ema99.parameters()):
                p_ema.data.mul_(0.99).add_(p.data, alpha=0.01)
            for p, p_ema in zip(self.model.parameters(), self.model_ema999.parameters()):
                p_ema.data.mul_(0.999).add_(p.data, alpha=0.001)
            for p, p_ema in zip(self.model.parameters(), self.model_ema9999.parameters()):
                p_ema.data.mul_(0.9999).add_(p.data, alpha=0.0001)

    def update(self, x0, condition=None, update_ema=True, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "model_ema": self.model_ema.state_dict(),
            "model_ema99": self.model_ema99.state_dict(),
            "model_ema999": self.model_ema999.state_dict(),
            "model_ema9999": self.model_ema9999.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model_ema.load_state_dict(checkpoint["model_ema"], strict=False)
        self.model_ema99.load_state_dict(checkpoint["model_ema99"], strict=False)
        self.model_ema999.load_state_dict(checkpoint["model_ema999"], strict=False)
        self.model_ema9999.load_state_dict(checkpoint["model_ema9999"], strict=False)
