from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn


class BasicClassifier:
    """
    Basic classifier for classifier-guidance.
    Generally, the classifier predicts the logp(c|x_t, noise),
    and then uses the gradient with respect to x_t to guide the diffusion model in sampling the distribution p(x_0|c).
    """

    def __init__(
            self,
            nn_classifier: nn.Module,
            device: str = "cpu",
            optim_params: Optional[dict] = None,
    ):
        if optim_params is None:
            optim_params = {"lr": 2e-4, "weight_decay": 1e-4}
        self.device = device
        self.model = nn_classifier
        self.model_ema = deepcopy(self.model).eval()
        self.optim = torch.optim.Adam(self.model.parameters(), **optim_params)

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(0.995).add_(p.data, alpha=1. - 0.995)

    def loss(self, x: torch.Tensor, noise: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError

    def update(self, x: torch.Tensor, noise: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError

    def logp(self, x: torch.Tensor, noise: torch.Tensor, c: torch.Tensor):
        """
        Calculate logp(c|x_t / scale, noise) for classifier-guidance.

        Input:
            - x:         (batch, *x_shape)
            - noise:     (batch, )
            - c:         (batch, *c_shape)

        Output:
            - logp(c|x, noise): (batch, 1)
        """
        raise NotImplementedError

    def gradients(self, x: torch.Tensor, noise: torch.Tensor, c: torch.Tensor):
        x.requires_grad_()
        logp = self.logp(x, noise, c)
        grad = torch.autograd.grad([logp.sum()], [x])[0]
        x.detach()
        return logp.detach(), grad.detach()

    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "model_ema": self.model_ema.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])



# class CategoricalClassifier(BasicClassifier):
#     """
#     CategoricalClassifier is used for finite discrete conditional sets.
#     In this case, the training of the classifier can be transformed into a classification task.
#     """
#     def __init__(self, nn_classifier: nn.Module):
#         super().__init__()
#
#     def logp(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, scale: Union[torch.Tensor, float] = 1.) -> torch.Tensor:
#         """
#         Calculate logp(c|x_t / scale, t) for classifier-guidance.
#
#         Input:
#             - x:         (batch, *x_shape)
#             - t:         (batch, *t_shape)
#             - c:         (batch, *c_shape)
#             - scale      (batch, *x_shape) or float
#
#         Output:
#             - logp(c|x / scale, t): (batch, 1)
#         """
#         raise NotImplementedError
