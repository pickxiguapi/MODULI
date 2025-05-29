import argparse
import os
from datetime import datetime
from time import time
import sys
sys.path.append("/mnt/dataset/zzr/modiff")
import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset, PEDAMuJoCoDataset3Obj
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DDPM, EDM, DiffusionModel, DPMSolver
from cleandiffuser.diffusion.vpsde import DiscreteVPSDE, ContinuousVPSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d, JannerUNet1d
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE
import environments
from dev.PEDA.modt.hypervolume import InnerHyperVolume
hv = InnerHyperVolume([0, 0])
from dev.utils.utils import EmbMLPCondition, MOCumRewClassifier, LRUDatasetCache

import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, default="front_approx")
    parser.add_argument("--dataset_name", type=str, default="MO-Hopper-v2_50000_amateur_uniform")
    parser.add_argument("--device", type=str, default="cuda:1")
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
    pref_dim = 2
    approximator = nn.Sequential(
        nn.Linear(pref_dim, 128),
        nn.Mish(),
        nn.Linear(128, 512),
        nn.Mish(),
        nn.Linear(512, 512),
        nn.Mish(),
        nn.Linear(512, pref_dim)
    ).to(args.device)
    report_parameters(approximator)

    dataset = PEDAMuJoCoDataset3Obj(
        dataset_name, terminal_penalty=100., horizon=args.horizon, avg=True, gamma=1.,
        normalize_rewards=True, eps=3e-3, discount=0.997, force_override=True)
    obs_dim, act_dim, pref_dim = dataset.o_dim, dataset.a_dim, dataset.pref_dim

    train_dataset = TensorDataset(torch.from_numpy(dataset.front_pref).to(args.device), torch.from_numpy(dataset.front_rew).to(args.device))
    dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # --------------- Create Approx Model -----------------

    optim = torch.optim.AdamW(approximator.parameters(), lr=1e-3)

    # ---------------- Train ----------------
    approximator.train()

    n_gradient_step = 0
    log = {"time": 0., "avg_loss_approx": 0., "gradient_steps": 0}

    for pref, rew in loop_dataloader(dataloader):
        loss = nn.MSELoss()(approximator(pref), rew)
        log["avg_loss_approx"] += loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (n_gradient_step + 1) % 1000 == 0:
            log["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log["gradient_steps"] = n_gradient_step + 1
            log["avg_loss_approx"] /= 1000
            print(log)
            log = {"time":0., "avg_loss_approx": 0., "gradient_steps": 0}

        if (n_gradient_step + 1) % 100_000 == 0:
            torch.save(approximator.state_dict(), save_path + f"approx_ckpt{n_gradient_step + 1}.pt")
            # eval_prefs = torch.linspace(0, 1, 100).reshape(-1, 1)
            # eval_prefs = torch.cat([eval_prefs, torch.ones_like(eval_prefs)-eval_prefs], dim=-1)
            pred_rew = approximator(torch.from_numpy(dataset.front_pref).to(args.device)).detach().cpu().numpy()
            np.save(save_path + f"approx_rew_{n_gradient_step + 1}.npy", pred_rew)

            fig, ax = plt.subplots()
            sum_seq_rew = dataset.seq_rew.sum(1)[::50]
            # ax.scatter(sum_seq_rew[:, 0], sum_seq_rew[:, 1], c='b', label='True')
            ax.scatter(dataset.front_rew[:, 0], dataset.front_rew[:, 1], c='b', label='True')
            ax.scatter(pred_rew[:, 0], pred_rew[:, 1], c='r', label='Predicted')
            ax.legend()
            fig.savefig(save_path + f"approx_{n_gradient_step + 1}.pdf")

        n_gradient_step += 1
        if n_gradient_step >= 100_000:
            break