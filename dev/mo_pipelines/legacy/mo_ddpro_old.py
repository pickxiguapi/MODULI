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
from dev.utils.utils import MOCumRewClassifier

def inference(
        env: gym.Env,
        n_envs: int,
        dataset: Dataset,
        agent: DiscreteVPSDE,
        invdyn: MlpInvDynamic,
        n_episodes: int,
        args
):
    # -------------------- Start Rollout -----------------------
    episode_rewards, episode_dones = [], []
    normalizer = dataset.get_normalizer()
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    pref_dim = dataset.pref_dim
    pref_min = dataset.seq_pref[:,0].min()
    pref_max = dataset.seq_pref[:,0].max()
    test_prefs = torch.linspace(pref_min, pref_max, n_envs).reshape(-1, 1)
    test_prefs = torch.cat([test_prefs, torch.ones_like(test_prefs)-test_prefs], dim=-1)
    device, horizon = args.device, args.horizon
    condition = test_prefs.copy().to(device)

    for i in range(n_episodes):
        obs, done, ep_reward, t = env.reset(seed=seed), False, 0, 0
        cum_done = None
        ep_done = 0

        while not np.all(cum_done) and t <= (1000 + 1):
            # normalize obs
            obs = torch.tensor(normalizer.normalize(obs), device=device, dtype=torch.float32)

            # sample trajectories
            prior = torch.zeros((n_envs, horizon, obs_dim + act_dim + pref_dim), device=device)
            prior[:, 0, :obs_dim] = obs
            prior[:, :, -pref_dim:] = 1.

            traj, log = agent.ddim_solver(
                prior, n_samples=n_envs, sample_steps=args.sample_steps, sample_step_schedule=args.sample_step_schedule, use_ema=True,
                condition_cfg=condition, w_cfg=args.w_cfg, temperature=0.5)

            # select the best plan
            with torch.no_grad():
                act = invdyn.predict(obs, traj[:, 1, :obs_dim]).cpu().numpy()

            # step env
            obs, rew, done, info = env.step(act)
            
            for i in range(n_envs):
                rew[i] = (test_prefs[i] * info[i]['obj']).sum()

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            ep_reward += (rew * (1 - cum_done))
            ep_done += (1-cum_done)
            
            # print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

        episode_rewards.append(ep_reward)
        episode_dones.append(ep_done)

    return episode_rewards, episode_dones


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, default="mo_ddpro")
    parser.add_argument("--dataset_name", type=str, default="MO-Hopper-v2_50000_amateur_uniform")
    parser.add_argument("--save_name", type=str, default="debug")
    parser.add_argument("--mode", type=str, default="train")  # train / inference
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nn", type=str, default="dit")
    parser.add_argument("--time_emb_type", type=str, default="positional")
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    env_name, _, _, _ = dataset_name.split("_")
    save_name = args.save_name
    device = args.device
    seed = args.seed
    mode = args.mode
    sample_steps = args.sample_steps

    save_path = f"results/{args.pipeline_name}/{dataset_name}/{save_name}/"
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    set_seed(seed)

    if "HalfCheetah" in dataset_name:
        args.horizon = 4
        dim_mult = [1, 4, 2]
        if "expert" in dataset_name:
            args.w_cfg = 1.5
        elif "amateur" in dataset_name:
            args.w_cfg = 1.5

    elif "Hopper" in dataset_name:
        args.horizon = 32
        dim_mult = [1, 2, 2, 2]
        attn = False
        if "expert" in dataset_name:
            args.w_cfg = 1.5
        elif "amateur" in dataset_name:
            args.w_cfg = 1.5

    elif "Walker2d" in dataset_name:
        args.horizon = 32
        dim_mult = [1, 2, 2, 2]
        attn = False
        if "expert" in dataset_name:
            args.w_cfg = 1.5
        elif "amateur" in dataset_name:
            args.w_cfg = 1.5
    
    else:
        args.horizon = 4
        dim_mult = [1, 4, 2]
        if "expert" in dataset_name:
            args.w_cfg = 1.5
        elif "amateur" in dataset_name:
            args.w_cfg = 1.5

    # args.w_cfg = 1.5
    # ---------------- Create Dataset ----------------
    env = gym.make(env_name)
    dataset = PEDAMuJoCoDataset(
        dataset_name, terminal_penalty=0., horizon=args.horizon, avg=True, gamma=1.,
        normalize_rewards=True, eps=1e-3, discount=0.997)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim, pref_dim = dataset.o_dim, dataset.a_dim, dataset.pref_dim

    # --------------- Create Diffusion Model -----------------
    if args.nn == "dit":
        nn_diffusion = DiT1d(
            obs_dim + act_dim + pref_dim , emb_dim=128, d_model=384, n_heads=6, depth=args.depth,
            timestep_emb_type=args.time_emb_type).to(device)
        nn_condition = MLPCondition(
            2, 128, [128, ], nn.SiLU(), 0.25).to(device)
    elif args.nn == "unet":
        raise NotImplementedError

    invdyn = MlpInvDynamic(obs_dim, act_dim, 1024, optim_params={"lr": 5e-4}, device=device)
    invdyn.load("results/mo_dd_invdyn/MO-Hopper-v2_50000_amateur_uniform/invdyn_ckpt100000.pt")

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    fix_mask = torch.zeros((args.horizon, obs_dim + act_dim + pref_dim), device=device)
    fix_mask[0, :obs_dim] = 1.
    fix_mask[:, -pref_dim:] = 1.
    loss_weight = torch.ones((args.horizon, obs_dim + act_dim + pref_dim), device=device)
    loss_weight[1, :obs_dim] = 10.

    agent = DiscreteVPSDE(
            nn_diffusion, nn_condition, predict_noise=False,
            fix_mask=fix_mask, loss_weight=loss_weight, noise_schedule="linear",
            device=device, diffusion_steps=20)

    # ---------------- Train ----------------
    if mode == "train":
        n_gradient_step = 0
        log = {"time":0., "avg_loss_diffusion": 0., "avg_loss_invdyn": 0., "gradient_steps": 0}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(device)
            act = batch["act"].to(device)
            score = batch["score"].to(device)
            pref = batch["pref"].to(device)

            x = torch.cat([obs, act, score], -1)
            
            cond = pref
            if np.random.rand() < 0.5:
                cond[:] = -1.

            log["avg_loss_diffusion"] += agent.update(x, cond)['loss']

            if (n_gradient_step + 1) % 1000 == 0:
                log["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= 1000
                log["avg_loss_invdyn"] /= 1000
                print(log)
                log = {"time":0., "avg_loss_diffusion": 0., "avg_loss_invdyn": 0., "gradient_steps": 0}

            if (n_gradient_step + 1) % 20_000 == 0:
                agent.eval()
                num_envs = 20
                num_episodes = 1
                args.n_elites = 4
                args.solver = "ddpm"
                env_eval = gym.vector.make(env_name, num_envs)
                episode_rewards = inference(env_eval, num_envs, dataset, agent, invdyn, num_episodes, args)
                episode_rewards = np.array(episode_rewards)
                print(f"mean:{np.mean(episode_rewards, -1)}, std:{np.std(episode_rewards, -1)}")
                print(episode_rewards)
                agent.train()
            
            if (n_gradient_step + 1) % 100_000 == 0:
                agent.save(save_path + f"diffusion_ckpt{n_gradient_step + 1}.pt")

            n_gradient_step += 1
            if n_gradient_step >= 500_000:
                break

    # ---------------- Evaluation ----------------
    elif mode == "inference":
        
        args.w_cfg = 1.2
        # args.solver = "ddpm"
        args.solver = "ddpm"
        # args.solver = "ode_dpmsolver++_2M"
        # args.solver = "sde_dpmsolver++_1"

        num_envs = 20
        num_episodes = 1
        env_eval = gym.vector.make(env_name, num_envs)

        for ckpt in [20000]:

            agent.load(save_path + f"diffusion_ckpt{ckpt}.pt")
            agent.eval()
            invdyn.eval()

            episode_rewards, episode_dones = inference(env_eval, num_envs, dataset, agent, invdyn, num_episodes, args)
            # episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
            episode_rewards, episode_dones = np.array(episode_rewards), np.array(episode_dones)
            print(f"mean:{np.mean(episode_rewards, -1)}, std:{np.std(episode_rewards, -1)}")
            print(episode_rewards)
            print(episode_dones)

        # import pickle
        #
        # with open(f'{args.pipeline_name}_{env_name}_{args.solver}_diffss.pkl', 'wb') as f:
        #     pickle.dump(log, f)

    else:
        raise ValueError("Invalid mode")
