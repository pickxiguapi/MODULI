from typing import Optional, List

import torch
import torch.nn as nn
import numpy as np
from math import *

from cleandiffuser.classifier.basic import BasicClassifier
from cleandiffuser.utils import at_least_ndim
from cleandiffuser.nn_condition import IdentityCondition, get_mask
from collections import OrderedDict
import pickle
import os

def hv(points: np.ndarray, ref_point: np.ndarray):
    """
    Calculate the hypervolume of a set of points with respect to a reference point.
    """
    n_points = points.shape[0]
    n_dim = points.shape[1]
    hv = 0
    for i in range(n_points):
        hv += np.prod(ref_point - points[i, :])
    return hv

class LRUDatasetCache():
    def __init__(self, capacity=12, save_path:str="dev/data/cache/"):
        self.capacity = capacity
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        initial_dict = {(i,):f"cached_dataset_{i}.pkl" for i in range(self.capacity)}
        self.cache = OrderedDict(initial_dict)
        print(self.cache)
    
    def exists(self, key):
        return key in self.cache and os.path.exists(self.save_path + self.cache[key])

    def get(self, key):
        if key in self.cache:
            # self.cache.move_to_end(key)
            with open(self.save_path + self.cache[key], 'rb') as f:
                return pickle.load(file=f)
        return None

    def put(self, key, value):
        if key in self.cache:
            with open(self.save_path + self.cache[key], 'wb') as f:
                pickle.dump(value, file=f)
            self.cache.move_to_end(key)
        else:
            first_key, first_value = self.cache.popitem(last=False)
            self.cache[key] = first_value
            with open(self.save_path + first_value, 'wb') as f:
                pickle.dump(value, file=f)
    
    def __repr__(self):
        return repr(self.cache)


class TrajSparseTable():
    """
    A class to create a sparse table for traj data in shape [n_trajs, max_path_length, n_features],
    support interval maximum and minimum value queries with O(1) time complexity.
    """
    def __init__(self, traj_data: np.ndarray):
        self.level = 9
        self.ceiling = ceil(log2(traj_data.shape[0])) + 1
        self.n_trajs = traj_data.shape[0]
        self.st = np.zeros((traj_data.shape[0], traj_data.shape[1], traj_data.shape[2], self.ceiling, 2), dtype=np.float32)
        
        self.st[:, :, :, 0, 0] = traj_data.copy()
        self.st[:, :, :, 0, 1] = traj_data.copy()

        for l in range(1, self.level + 1):
            for i in range(self.n_trajs - (1 << l) + 1):
                self.st[i, :, :, l, 0] = np.maximum(self.st[i, :, :, l-1, 0], self.st[i + (1 << (l-1)), :, :, l-1, 0])
                self.st[i, :, :, l, 1] = np.minimum(self.st[i, :, :, l-1, 1], self.st[i + (1 << (l-1)), :, :, l-1, 1])

    def max(self, l: int, r: int):
        """
        Query the maximum value in the interval [l, r].
        """
        if l < 0 or r >= self.n_trajs or r < l:
            raise ValueError("Invalid interval.")

        k = int(floor(log2(r - l)))
        self.extend(k)
        return np.maximum(self.st[l, :, :, k, 0], self.st[r - (1 << k) + 1, :, :, k, 0])

    def min(self, l: int, r: int):
        """
        Query the minimum value in the interval [l, r].
        """
        if l < 0 or r >= self.n_trajs or r < l:
            raise ValueError("Invalid interval.")

        k = int(floor(log2(r - l)))
        self.extend(k)
        return np.minimum(self.st[l, :, :, k, 1], self.st[r - (1 << k) + 1, :, :, k, 1])

    def extend(self, current_level: int):
        assert current_level < self.ceiling
        while (self.level < current_level):
            self.level += 1
            for i in range(self.n_trajs - (1 << self.level) + 1):
                self.st[i, :, :, self.level, 0] = np.maximum(self.st[i, :, :, self.level - 1, 0], self.st[i + (1 << (self.level - 1)), :, :, self.level - 1, 0])
                self.st[i, :, :, self.level, 1] = np.minimum(self.st[i, :, :, self.level - 1, 1], self.st[i + (1 << (self.level - 1)), :, :, self.level - 1, 1])
            
        

class MOCumRewClassifier(BasicClassifier):
    def __init__(
            self,
            nn_classifier: nn.Module,
            device: str = "cpu",
            scale_param: float = 1.,
            optim_params: Optional[dict] = None,
    ):
        super().__init__(nn_classifier, device, optim_params)
        self.s = scale_param

    def loss(self, x, noise, R):
        pred_R = self.model(x, noise, None)
        return ((pred_R - R) ** 2).mean()

    def update(self, x, noise, R):
        self.optim.zero_grad()
        loss = self.loss(x, noise, R)
        loss.backward()
        self.optim.step()
        self.ema_update()
        return {"loss": loss.item()}

    def logp(self, x, noise, c=None):
        pred_R = self.model_ema(x, noise, None)
        pref_loss = (pred_R[:,0]*c[:,1] - pred_R[:,1]*c[:,0])**2
        cum_reward = pred_R
        return torch.cat([cum_reward, pref_loss.unsqueeze(-1)], dim=-1)

class MOWeightedCumRewClassifier(BasicClassifier):
    def __init__(
            self,
            nn_classifier: nn.Module,
            device: str = "cpu",
            pref_dim: int = 2,
            optim_params: Optional[dict] = None,
    ):
        super().__init__(nn_classifier, device, optim_params)
        self.pref_dim = pref_dim

    def loss(self, x, noise, R):
        pred_R = self.model(x, noise, R[0])
        return ((pred_R - (R[0]*R[1]).sum(axis=1, keepdim=True)) ** 2).mean()

    def update(self, x, noise, R):
        self.optim.zero_grad()
        loss = self.loss(x, noise, R)
        loss.backward()
        self.optim.step()
        self.ema_update()
        return {"loss": loss.item()}

    def logp(self, x, noise, c=None):
        pred_R = self.model_ema(x, noise, c)
        return pred_R

class MOWeightedRTGClassifier(BasicClassifier):
    def __init__(
            self,
            nn_classifier: nn.Module,
            device: str = "cpu",
            pref_dim: int = 2,
            optim_params: Optional[dict] = None,
    ):
        super().__init__(nn_classifier, device, optim_params)
        self.pref_dim = pref_dim

    def loss(self, x, noise, R):
        pred_R = self.model(x, noise, R[0])
        return ((pred_R - R[1]) ** 2).mean()

    def update(self, x, noise, R):
        self.optim.zero_grad()
        loss = self.loss(x, noise, R)
        loss.backward()
        self.optim.step()
        self.ema_update()
        return {"loss": loss.item()}

    def logp(self, x, noise, c=None):
        pred_R = self.model_ema(x, noise, c) * c
        return pred_R

class MOPrefClassifier(BasicClassifier):
    def __init__(
            self,
            nn_classifier: nn.Module,
            device: str = "cpu",
            pref_weight: float = 1.,
            optim_params: Optional[dict] = None,
    ):
        super().__init__(nn_classifier, device, optim_params)
        self.pref_weight = pref_weight

    def loss(self, x, noise, R):
        pred_R = self.model(x, noise, None)
        return ((pred_R - R) ** 2).mean()

    def update(self, x, noise, R):
        self.optim.zero_grad()
        loss = self.loss(x, noise, R)
        loss.backward()
        self.optim.step()
        self.ema_update()
        return {"loss": loss.item()}

    def logp(self, x, noise, c=None):
        pred_R = self.model_ema(x, noise, None)
        pref_loss = -(pred_R[:, :pred_R.shape[1]//2]-c) ** 2 * self.pref_weight
        val = pred_R[:, -pred_R.shape[1]//2:]
        return torch.cat([pref_loss, val], dim=-1)
    
class Mlp(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dims: List[int],
                 out_dim: int,
                 activation: nn.Module = nn.ReLU(),
                 out_activation: nn.Module = nn.Identity()):

        super().__init__()
        self.mlp = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i]),
                activation)
            for i in range(len(hidden_dims))
        ], nn.Linear(in_dim if len(hidden_dims) == 0 else hidden_dims[-1], out_dim), out_activation)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)
    
class MLPConditionSep(IdentityCondition):
    """
    MLP condition sep is a simple multi-layer perceptron to process the input condition for multi-object tasks, 
    which processes preference and reward condition seperatly.

    Input:
        - condition: (b, *(pref_in_shape + rew_in_shape))
        - mask:      (b, ) or None, None means no mask
    
    Output:
        - condition: (b, *cond_out_shape)
    """
    
    def __init__(self, pref_dim: int, reward_dim: int, out_dim: int, hidden_dims: List[int],
                 act=nn.LeakyReLU(), dropout: float = 0.25):
        super().__init__(dropout)
        self.pref_dim, self.reward_dim = pref_dim, reward_dim
        hidden_dims = [hidden_dims, ] if isinstance(hidden_dims, int) else hidden_dims
        self.pref_mlp = Mlp(
            pref_dim, [], pref_dim, nn.Identity())
        self.reward_mlp = Mlp(
            reward_dim, [], reward_dim, nn.Identity())
        self.mlp = Mlp(
            pref_dim + reward_dim, hidden_dims, out_dim, act)
    
    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        prefs_emb = self.pref_mlp(condition[:, :self.pref_dim])
        rewards_emb = self.reward_mlp(condition[:, -self.reward_dim:])
        condition = torch.cat([prefs_emb, rewards_emb], dim=-1)

        mask = at_least_ndim(get_mask(
            mask, (condition.shape[0],), self.dropout, self.training, condition.device), condition.dim())
        return self.mlp(condition) * mask

def hv(data: np.ndarray):
    """
    Calculate hypervolume of a given data.
    """
    assert data.ndim == 2 or data.ndim == 3, "Data must be 2D or 3D."

    if data.ndim == 2:
        data = data[None, :]

    hv = np.zeros(data.shape[0])

    for i in range(data.shape[0]):
        sorted_idx = np.argsort(data[i, :, 0])
        data_slice = data[i, sorted_idx]
        num = data_slice.shape[0]
        front = np.ones(num)
        for j in range(num):
            for k in range(num):
                if j != k and np.all(data_slice[j] < data_slice[k]):
                    front[j] = 0
                    break
        x = 0.
        for j in range(num):
            if front[j] == 1:
                hv[i] += data_slice[j, 1] * (data_slice[j, 0] - x)
                x = data_slice[j, 0]
    
    return hv

from cleandiffuser.nn_classifier import HalfDiT1d
from cleandiffuser.nn_condition import MLPCondition

class ConditionalHalfDiT1d(HalfDiT1d):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 cond_dim: int,
                 emb_dim: int,
                 d_model: int = 384,
                 n_heads: int = 6,
                 depth: int = 12,
                 dropout: float = 0.0,
                 timestep_emb_type: str = "positional",
                 ):
        super().__init__(in_dim, out_dim, emb_dim, d_model, n_heads, depth, dropout, timestep_emb_type)
        self.nn_condition = MLPCondition(cond_dim, emb_dim, [128, ], nn.SiLU(), 0.25)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, horizon, in_dim * 2), where the first half is the reference signal
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        condition = self.nn_condition(condition)
        return super().forward(x, noise, condition)

from cleandiffuser.utils import PositionalEmbedding, FourierEmbedding

class BatchFourierEmbedding(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(emb_dim // 8) * scale, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim // 4 * in_dim, emb_dim), nn.Mish(), nn.Linear(emb_dim, emb_dim))

    def forward(self, x: torch.Tensor):
        emb = torch.einsum('bi,j->bij', x, (2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1).reshape((x.shape[0], -1))
        emb = self.mlp(emb)
        return emb
    
class EmbMLPCondition(IdentityCondition):
    """
    MLP condition is a simple multi-layer perceptron to process the input condition.

    Input:
        - condition: (b, *cond_in_shape)
        - mask :     (b, ) or None, None means no mask

    Output:
        - condition: (b, *cond_out_shape)
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int], 
                 act=nn.LeakyReLU(), dropout: float = 0.25, cond_emb_type: str = "None"):
        super().__init__(dropout)
        hidden_dims = [hidden_dims, ] if isinstance(hidden_dims, int) else hidden_dims
        if cond_emb_type == "positional":
            raise NotImplementedError("Positional embedding is not implemented yet")
            # self.cond_embedding = PositionalEmbedding(in_dim, endpoint=True)
        elif cond_emb_type == "fourier":
            self.cond_embedding = BatchFourierEmbedding(in_dim, 128)
            self.mlp = Mlp(
                128, hidden_dims, out_dim, act)
        else:
            self.cond_embedding = nn.Identity()
            self.mlp = Mlp(
                in_dim, hidden_dims, out_dim, act)
        

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        condition = self.cond_embedding(condition)
        mask = at_least_ndim(get_mask(
            mask, (condition.shape[0],), self.dropout, self.training, condition.device), condition.dim())
        return self.mlp(condition) * mask

class MlpDynamic:
    def __init__(
            self,
            o_dim: int,
            a_dim: int,
            hidden_dim: int = 512,
            out_activation: nn.Module = nn.Identity(),
            optim_params: dict = {},
            device: str = "cpu",
    ):
        self.device = device
        self.o_dim, self.a_dim, self.hidden_dim = o_dim, a_dim, hidden_dim
        self.out_activation = out_activation
        self.optim_params = optim_params
        params = {"lr": 5e-4}
        params.update(optim_params)
        self.mlp = Mlp(
            o_dim + a_dim, [hidden_dim, hidden_dim], o_dim,
            nn.ReLU(), out_activation).to(device)
        self.optim = torch.optim.Adam(self.mlp.parameters(), **optim_params)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, o, a):
        return self.mlp(torch.cat([o, a], dim=-1))

    def update(self, o, a, o_next):
        self.optim.zero_grad()
        o_next_pred = self.forward(o, a)
        loss = ((o_next_pred - o_next) ** 2).mean()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def predict(self, o, a):
        return self.forward(o, a)

    def __call__(self, o, a):
        return self.predict(o, a)

    def train(self):
        self.mlp.train()

    def eval(self):
        self.mlp.eval()

    def save(self, path):
        torch.save(self.mlp.state_dict(), path)

    def load(self, path):
        self.mlp.load_state_dict(torch.load(path, self.device))

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

class DynaContinuousVPSDE(DiffusionModel):

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,
            nn_dyna: MlpDynamic = None,

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
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            0, ema_rate, optim_params, device)

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.x_max, self.x_min = x_max, x_min

        self.dyna = nn_dyna
        self.o_dim, self.a_dim = self.dyna.o_dim, self.dyna.a_dim

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

        condition = self.model["condition"](condition) if condition is not None else None

        if self.predict_noise:
            loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        else:
            loss = (self.model["diffusion"](xt, t, condition) - x0) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

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
            requires_grad: bool = False):
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
                    condition = torch.cat([condition, torch.zeros_like(condition)], 0)
                    pred_all = model["diffusion"](
                        xt.repeat(*repeat_dim), t.repeat(2), condition)
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
            requires_grad: bool = False):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad)

        pred, logp = self.classifier_guidance(
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        return pred, logp

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
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
                model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad)

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

            for i in range(1, self.fix_mask.shape[1]):
                xt[:, i, :self.o_dim] = self.dyna.predict(xt[:, i-1, :self.o_dim], xt[:, i-1, self.o_dim:self.o_dim+self.a_dim])

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

from lora_pytorch import LoRA
from cleandiffuser.nn_diffusion import BaseNNDiffusion

class ContinuousVPSDELoRA(DiffusionModel):
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
            epsilon: float = 1e-3,

            discretization: Union[str, Callable] = "uniform",
            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            predict_noise: bool = True,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            0, ema_rate, optim_params, device)

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.x_max, self.x_min = x_max, x_min

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

        bias_size = 0.1
        bias_single = torch.rand_like(condition[:, 0]).reshape((-1, 1)) * bias_size
        bias = torch.cat([bias_single, -bias_single, torch.zeros_like(condition[:,-2:])], dim=-1)

        condition_origin = self.model["condition"](condition) if condition is not None else None
        condition_pos = self.model["condition"](condition + bias) if condition is not None else None
        condition_neg = self.model["condition"](condition - bias) if condition is not None else None

        self.model["diffusion"].disable_lora()

        non_lora_uncond = self.model["diffusion"](xt, t, condition_origin)
        non_lora_cond_pos = self.model["diffusion"](xt, t, condition_pos)
        non_lora_cond_neg = self.model["diffusion"](xt, t, condition_neg)

        self.model["diffusion"].enable_lora()

        lora_cond = self.model["diffusion"](xt, t, condition_origin)

        loss = ((lora_cond - non_lora_uncond) * bias_single.unsqueeze_(1) - (non_lora_cond_pos - non_lora_cond_neg)) ** 2

        # if self.predict_noise:
        #     loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        # else:
        #     loss = (self.model["diffusion"](xt, t, condition) - x0) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

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
            requires_grad: bool = False):
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
                    condition = torch.cat([condition, torch.zeros_like(condition)], 0)

                    model["diffusion"].set_lora_weight(0.00)

                    # model["diffusion"].disable_lora()
                    # value = model["diffusion"](
                    #     xt.repeat(*repeat_dim), t.repeat(2), condition)
                    # model["diffusion"].enable_lora()
                    # bias =  model["diffusion"](
                    #     xt.repeat(*repeat_dim), t.repeat(2), condition)
                    
                    # bias_size = 0.00
                    # pred_all = value + bias * bias_size

                    pred_all = model["diffusion"](
                        xt.repeat(*repeat_dim), t.repeat(2), condition)
                    
                    model["diffusion"].set_lora_weight(1.0)

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
            requires_grad: bool = False):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad)

        pred, logp = self.classifier_guidance(
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        return pred, logp

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
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
                model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad)

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

    def apply_lora(self):
        for name, param in self.model["diffusion"]:
            param.requires_grad = False
        for name, param in self.model_ema["diffusion"]:
            param.requires_grad = False
        for name, param in self.model_ema99["diffusion"]:
            param.requires_grad = False
        for name, param in self.model_ema999["diffusion"]:
            param.requires_grad = False
        for name, param in self.model_ema9999["diffusion"]:
            param.requires_grad = False
        
        for name, param in self.model["condition"]:
            param.requires_grad = False
        for name, param in self.model_ema["condition"]:
            param.requires_grad = False
        for name, param in self.model_ema99["condition"]:
            param.requires_grad = False
        for name, param in self.model_ema999["condition"]:
            param.requires_grad = False
        for name, param in self.model_ema9999["condition"]:
            param.requires_grad = False

        self.model["diffusion"] = LoRA.from_module(self.model["diffusion"])
        self.model_ema["diffusion"] = LoRA.from_module(self.model_ema["diffusion"])
        self.model_ema99["diffusion"] = LoRA.from_module(self.model_ema99["diffusion"])
        self.model_ema999["diffusion"] = LoRA.from_module(self.model_ema999["diffusion"])
        self.model_ema9999["diffusion"] = LoRA.from_module(self.model_ema9999["diffusion"])
    
    def set_lora_weight(self, w: float):
        self.model["diffusion"].set_lora_weight(w)
        self.model_ema["diffusion"].set_lora_weight(w)
        self.model_ema99["diffusion"].set_lora_weight(w)
        self.model_ema999["diffusion"].set_lora_weight(w)
        self.model_ema9999["diffusion"].set_lora_weight(w)