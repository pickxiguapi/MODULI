import argparse
import os
from datetime import datetime

import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DDPM, EDM, DiffusionModel, DPMSolver
from cleandiffuser.diffusion.vpsde import DiscreteVPSDE, ContinuousVPSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d, JannerUNet1d
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE
import environments
from dev.utils.utils import EmbMLPCondition, MOCumRewClassifier, LRUDatasetCache, hv, MlpDynamic

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, default="mo_dd_dyna")
    parser.add_argument("--dataset_name", type=str, default="MO-Hopper-v2_50000_amateur_uniform")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--redirect", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    env_name, _, _, _ = dataset_name.split("_")
    device = args.device
    seed = args.seed

    save_path = f"results/{args.pipeline_name}/{dataset_name}/"
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    if args.redirect:
        import sys
        logfile = save_path+"log.txt"
        sys.stdout = open(logfile, 'w')
        sys.stderr = sys.stdout

    set_seed(seed)

    if "Hopper" in dataset_name or "Walker2d" in dataset_name:
        args.horizon = 32
    else:
        args.horizon = 4

    # ---------------- Create Dataset ----------------
    dataset = PEDAMuJoCoDataset(
        dataset_name, terminal_penalty=0., horizon=args.horizon, avg=True, gamma=1.,
        normalize_rewards=True, eps=1e-3, discount=0.997)
    dataloader = DataLoader(
        dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Create Diffusion Model -----------------
    dyna = MlpDynamic(obs_dim, act_dim, 1024, optim_params={"lr": 5e-4}, device=device)

    # ---------------- Train ----------------
    dyna.train()

    n_gradient_step = 0
    log = {"time": 0., "avg_loss_dyna": 0., "gradient_steps": 0}

    for batch in loop_dataloader(dataloader):

        obs = batch["obs"]["state"].to(device)
        act = batch["act"].to(device)

        log["avg_loss_dyna"] += dyna.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']

        if (n_gradient_step + 1) % 1000 == 0:
            log["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log["gradient_steps"] = n_gradient_step + 1
            log["avg_loss_dyna"] /= 1000
            print(log)
            log = {"time":0., "avg_loss_dyna": 0., "gradient_steps": 0}

        if (n_gradient_step + 1) % 100_000 == 0:
            dyna.save(save_path + f"dyna_ckpt{n_gradient_step + 1}.pt")

        n_gradient_step += 1
        if n_gradient_step >= 500_000:
            break