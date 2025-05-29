import argparse
import os
from datetime import datetime

import d4rl
import gym
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import EDM, DiffusionModel, DDIM, DDPM, DPMSolver, DPMSolverDiscrete
from cleandiffuser.diffusion.vpsde import DiscreteVPSDE
from cleandiffuser.nn_classifier import HalfDiT1d, HalfJannerUNet1d
from cleandiffuser.nn_diffusion import DiT1d, JannerUNet1d
from torch.optim.lr_scheduler import CosineAnnealingLR
from cleandiffuser.utils import IQL
import environments


from pathlib import Path
from cleandiffuser.utils import count_parameters, report_parameters
from dev.utils.utils import MOCumRewClassifier

def inference(
        env: gym.Env,
        n_envs: int,
        dataset: Dataset,
        agent: DiscreteVPSDE,
        logger,
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
    n_elites = args.n_elites
    test_prefs_n_elites = test_prefs.repeat(n_elites, 1).to(args.device)
    test_prefs = test_prefs.cpu().numpy()

    for i in range(n_episodes):
        obs, done, ep_reward, t = env.reset(seed=seed), False, 0, 0
        cum_done = None
        ep_done = 0

        while not np.all(cum_done) and t <= (1000 + 1):
            # normalize obs
            obs = torch.tensor(normalizer.normalize(obs), device=device, dtype=torch.float32)

            # sample trajectories
            prior = torch.zeros((n_envs, horizon, obs_dim + act_dim), device=device)
            prior[:, 0, :obs_dim] = obs
            prior = prior.repeat(n_elites, 1, 1)
            traj, log = agent.ddim_solver(
                prior, n_elites * n_envs, args.sample_steps, use_ema=True,
                condition_cg=test_prefs_n_elites, w_cg=args.w_cg, temperature=0.5)

            # select the best plan
            logp = log["log_p"].view(n_elites, n_envs, -1).sum(-1)
            idx = logp.argmax(0)
            act = traj.view(n_elites, n_envs, horizon, -1)[idx, torch.arange(n_envs), 0, obs_dim:]
            act = act.cpu().numpy()

            # step env
            obs, rew, done, info = env.step(act)
            
            for i in range(n_envs):
                rew[i] = (test_prefs[i] * info[i]['obj']).sum()

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            ep_reward += (rew * (1 - cum_done))
            ep_done += (1-cum_done)
            
            # print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}, logp: {logp[idx, torch.arange(n_envs)]}')

        episode_rewards.append(ep_reward)
        episode_dones.append(ep_done)

    return episode_rewards, episode_dones


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, default="mo_diffuser")
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
            args.w_cg = 0.01
        elif "amateur" in dataset_name:
            args.w_cg = 0.0001

    elif "Hopper" in dataset_name:
        args.horizon = 32
        dim_mult = [1, 2, 2, 2]
        attn = False
        if "expert" in dataset_name:
            args.w_cg = 0.00001
        elif "amateur" in dataset_name:
            args.w_cg = 0.3

    elif "Walker2d" in dataset_name:
        args.horizon = 32
        dim_mult = [1, 2, 2, 2]
        attn = False
        if "expert" in dataset_name:
            args.w_cg = 0.001
        elif "amateur" in dataset_name:
            args.w_cg = 0.02
    
    else:
        args.horizon = 4
        dim_mult = [1, 4, 2]
        if "expert" in dataset_name:
            args.w_cg = 0.01
        elif "amateur" in dataset_name:
            args.w_cg = 0.0001

    # args.w_cg = 0.0001
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
            obs_dim + act_dim, emb_dim=128, d_model=384, n_heads=6, depth=args.depth,
            timestep_emb_type=args.time_emb_type).to(device)
        nn_classifier = HalfDiT1d(
            obs_dim + act_dim, out_dim=pref_dim, emb_dim=128, d_model=192, n_heads=6, depth=2,
            timestep_emb_type=args.time_emb_type).to(device)
    elif args.nn == "unet":
        nn_diffusion = JannerUNet1d(
            obs_dim + act_dim, model_dim=32, emb_dim=32, dim_mult=dim_mult,
            timestep_emb_type="positional", attention=False, kernel_size=5).to(device)
        nn_classifier = HalfJannerUNet1d(
            args.horizon, obs_dim + act_dim, out_dim=pref_dim, model_dim=32, emb_dim=32, dim_mult=dim_mult,
            timestep_emb_type="positional", kernel_size=3).to(device)

    print(f"======================= Parameter Report of Diffusion Model =======================",)
    report_parameters(nn_diffusion)
    print(f"======================= Parameter Report of Classifier =======================")
    report_parameters(nn_classifier)
    print(f"==============================================================================")

    classifier = MOCumRewClassifier(nn_classifier, device=device, scale_param=1.)

    fix_mask = torch.zeros((args.horizon, obs_dim + act_dim), device=device)
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.horizon, obs_dim + act_dim), device=device)
    loss_weight[0, obs_dim:] = 10.
    
    agent = DiscreteVPSDE(
            nn_diffusion, None, predict_noise=False,
            fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, noise_schedule="linear",
            device=device, diffusion_steps=20)
    
    # ckpt = 380000
    # agent.load(save_path + f"diffusion_ckpt{ckpt}.pt")
    # agent.classifier.load(save_path + f"classifier_ckpt{ckpt}.pt")
    
    diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, 1_000_000, eta_min=1e-6)
    classifier_lr_scheduler = CosineAnnealingLR(agent.classifier.optim, 1_000_000, eta_min=1e-6)

    # ---------------- Train ----------------
    if mode == "train":
        n_gradient_step = 0
        log = {"time":0., "avg_loss_diffusion": 0., "avg_loss_classifier": 0., "gradient_steps": 0}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(device)
            act = batch["act"].to(device)
            val = batch["val"].to(device)

            x = torch.cat([obs, act], -1)

            log["avg_loss_diffusion"] += agent.update(x)['loss']
            
            if n_gradient_step <= 1000_000:
                log["avg_loss_classifier"] += agent.update_classifier(x, val)['loss']
                
            if (n_gradient_step + 1) % 1000 == 0:
                log["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= 1000
                log["avg_loss_classifier"] /= 1000
                print(log)
                log = {"time":0., "avg_loss_diffusion": 0., "avg_loss_classifier": 0., "gradient_steps": 0}

            if (n_gradient_step + 1) % 20_000 == 0:
                agent.eval()
                num_envs = 20
                num_episodes = 1
                args.n_elites = 4
                args.solver = "ddpm"
                env_eval = gym.vector.make(env_name, num_envs)
                episode_rewards = inference(env_eval, num_envs, dataset, agent, None, num_episodes, args)
                episode_rewards = np.array(episode_rewards)
                print(f"mean:{np.mean(episode_rewards, -1)}, std:{np.std(episode_rewards, -1)}")
                print(episode_rewards)
                agent.train()
            
            if (n_gradient_step + 1) % 100_000 == 0:
                agent.save(save_path + f"diffusion_ckpt{n_gradient_step + 1}.pt")
                agent.classifier.save(save_path + f"classifier_ckpt{n_gradient_step + 1}.pt")

            n_gradient_step += 1
            if n_gradient_step >= 500_000:
                break

    # ---------------- Evaluation ----------------
    elif mode == "inference":

        num_envs = 20
        num_episodes = 1
        args.n_elites = 4
        args.solver = "ddpm"
        env_eval = gym.vector.make(env_name, num_envs)

        for ckpt in [40000]:

            agent.load(save_path + f"diffusion_ckpt{ckpt}.pt")
            agent.classifier.load(save_path + f"classifier_ckpt{ckpt}.pt")
            # agent.classifier.s = 0.
            
            agent.eval()
            episode_rewards, episode_dones = inference(env_eval, num_envs, dataset, agent, None, num_episodes, args)
            # episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
            episode_rewards, episode_dones = np.array(episode_rewards), np.array(episode_dones)
            print(f"mean:{np.mean(episode_rewards, -1)}, std:{np.std(episode_rewards, -1)}")
            print(episode_rewards)
            print(episode_dones)

        # import pickle
        # with open(f'{args.pipeline_name}_{env_name}_{args.nn}_{args.diffusion}_{args.sample_steps}.pkl', 'wb') as f:
        #     pickle.dump(episode_rewards, f)

    else:
        raise ValueError("Invalid mode")
