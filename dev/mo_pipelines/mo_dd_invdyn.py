import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from time import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.utils import report_parameters

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, default="mo_dd_invdyn")
    parser.add_argument("--dataset_name", type=str, default="MO-Ant-v2_50000_amateur_uniform")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--redirect", action="store_true")
    parser.add_argument("--n_gradient_steps", type=int, default=500_000)
    parser.add_argument("--save_interval", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.n_gradient_steps, args.save_interval, args.batch_size) < 1:
        parser.error("training steps, save interval and batch size must be positive")

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

    torch.manual_seed(seed)
    np.random.seed(seed)

    if "Hopper" in dataset_name or "Walker2d" in dataset_name:
        args.horizon = 32
    else:
        args.horizon = 4

    # ---------------- Create Dataset ----------------
    dataset = PEDAMuJoCoDataset(
        dataset_name, terminal_penalty=0., horizon=args.horizon, avg=True, gamma=1.,
        normalize_rewards=True, eps=1e-3, discount=0.997)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    if len(dataset) < args.batch_size:
        raise ValueError("Dataset contains fewer trajectory segments than batch_size")
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Create Diffusion Model -----------------
    invdyn = MlpInvDynamic(obs_dim, act_dim, 1024, optim_params={"lr": 5e-4}, device=device)
    report_parameters(invdyn.mlp)
    # ---------------- Train ----------------
    invdyn.train()

    n_gradient_step = 0
    log = {"time": 0., "avg_loss_invdyn": 0., "gradient_steps": 0}
    start_time = time()
    for batch in loop_dataloader(dataloader):

        obs = batch["obs"]["state"].to(device)
        act = batch["act"].to(device)

        log["avg_loss_invdyn"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']

        if (n_gradient_step + 1) % 1000 == 0:
            log["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log["gradient_steps"] = n_gradient_step + 1
            log["avg_loss_invdyn"] /= 1000
            current_time = time()
            log["time_per_1000_steps"] = (current_time - start_time) / (n_gradient_step + 1) * 1000
            print(f"Step {n_gradient_step + 1}, loss: {log['avg_loss_invdyn']:.6f}, s/kiter: {log['time_per_1000_steps']:.2f}")
            # print(log)
            log = {"time":0., "avg_loss_invdyn": 0., "gradient_steps": 0}

        if (n_gradient_step + 1) % args.save_interval == 0:
            invdyn.save(save_path + f"invdyn_ckpt{n_gradient_step + 1}.pt")

        n_gradient_step += 1
        if n_gradient_step >= args.n_gradient_steps:
            break
    if n_gradient_step % args.save_interval:
        invdyn.save(save_path + f"invdyn_ckpt{n_gradient_step}.pt")
