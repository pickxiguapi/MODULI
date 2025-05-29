import argparse
import os
from datetime import datetime
import json

import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset, ShatteredPEDAMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import EDM, DiffusionModel, DDIM, DDPM, DPMSolver, DPMSolverDiscrete
from cleandiffuser.diffusion.vpsde import DiscreteVPSDE, ContinuousVPSDE
from cleandiffuser.nn_classifier import HalfDiT1d, HalfJannerUNet1d
from cleandiffuser.nn_diffusion import DiT1d, JannerUNet1d
from torch.optim.lr_scheduler import CosineAnnealingLR
from cleandiffuser.utils import IQL
import environments
from dev.PEDA.modt.hypervolume import InnerHyperVolume
hv = InnerHyperVolume([0, 0])
from cleandiffuser.utils import count_parameters, report_parameters
from dev.utils.utils import EmbMLPCondition, MOCumRewClassifier, MOPrefClassifier, LRUDatasetCache

def inference(
        dataset,
        agent,
        args,
        uniform=False,
):
    # --------------- Create Preference Grid -----------------
    if uniform:
        eval_prefs = torch.linspace(0, 1, args.num_prefs).reshape(-1, 1)
    else:
        eval_prefs = torch.linspace(0.5, 0.5, args.num_prefs).reshape(-1, 1)
    eval_prefs = torch.cat([eval_prefs, torch.ones_like(eval_prefs)-eval_prefs], dim=-1)
    eval_rewards = np.zeros_like(eval_prefs.repeat(args.num_episodes, 1, 1).cpu().numpy())
    eval_lens = np.zeros((args.num_episodes, args.num_prefs))
    val = torch.ones_like(eval_prefs)*args.cond_val
    condition = eval_prefs.clone().to(args.device)
    eval_prefs = eval_prefs.cpu().numpy()

    normalizer = dataset.get_normalizer()
    seed = args.seed

    # --------------- Evaluate -----------------
    for ep in range(args.num_episodes):
        # evaluate args.num_envs preferences each time
        idx = 0
        set_seed(seed)
        while idx < args.num_prefs:
            current_num_envs = min(args.num_envs, args.num_prefs-idx)
            env = gym.vector.make(args.env_name, current_num_envs)

            obs, done, ep_reward, t = env.reset(seed=seed), False, 0, 0
            cum_dones = None

            while not np.all(cum_dones) and t <= 500:
                # normalize obs
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                # sample trajectories
                prior = torch.zeros((current_num_envs, args.horizon, obs_dim + act_dim), device=args.device)
                prior[:, 0, :obs_dim] = obs
                prior = prior.repeat(args.n_elites, 1, 1)

                traj, log = agent.sample(
                    prior, n_samples= args.n_elites * current_num_envs, sample_steps=args.sample_steps, sample_step_schedule=args.sample_step_schedule, use_ema=True, solver=args.solver,
                    condition_cg=condition[idx:idx + current_num_envs].repeat(args.n_elites, 1), w_cg=args.w_cg, temperature=args.temperature)

                # select the best plan
                logp = log["log_p"].view(args.n_elites, current_num_envs, -1).sum(-1)
                idx_act = logp.argmax(0)
                act = traj.view(args.n_elites, current_num_envs, args.horizon, -1)[idx_act, torch.arange(current_num_envs), 0, obs_dim:]
                act = act.cpu().numpy()

                # step env
                obs, _, done, info = env.step(act)
                
                if cum_dones is None:
                    cum_dones = done

                for i in range(current_num_envs):
                    eval_rewards[ep, idx + i] += info[i]['obj'] * (1 - cum_dones[i])
                    eval_lens[ep, idx + i] += 1 - cum_dones[i]

                cum_dones = np.logical_or(cum_dones, done)

                t += 1
            
            idx += current_num_envs
        seed += 1
    # eval_rewards /= args.num_episodes
    # eval_lens /= args.num_episodes    
    
    return eval_rewards, eval_lens, eval_prefs

                
if __name__ == "__main__":
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, default="mo_diffuser_prefguide_shattered")
    parser.add_argument("--dataset_name", type=str, default="MO-Hopper-v2_50000_amateur_uniform")
    parser.add_argument("--save_name", type=str, default="debug")
    parser.add_argument("--eval_name", type=str, default="",
                        help="The name of the evaluation, for easy identification")
    parser.add_argument("--mode", type=str, default="train")  # train / eval
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--load_model", type=str, default=None,
                        help=("A string represents the model to be read, include save name and file name, leave it blank to ignore it."
                              "e.g.: debug/100000 represents loading debug/diffusion_ckpt100000.pt and debug/classifier_ckpt100000.pt, depends on the pipeline"))
    parser.add_argument("--n_gradient_steps", type=int, default=200_000)
    parser.add_argument("--redirect", action="store_true",
                        help="redirect stdout to log file")
    parser.add_argument("--resume", action="store_true",
                        help="continue training from the loaded model's step")

    # diffusion parameters
    parser.add_argument("--solver", type=str, default="ddim")
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--sample_step_schedule", type=str, default="uniform_continuous")
    parser.add_argument("--nn", type=str, default="dit")
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--time_emb_type", type=str, default="positional")
    parser.add_argument("--cond_emb_type", type=str, default="None")

    # dataset parameters
    parser.add_argument("--terminal_penalty", type=float, default=100.)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--avg", type=str2bool, nargs='?', default=True, choices=[True, False], 
                        help="Whether to divide cumulate reward by (max steps - current steps)")
    parser.add_argument("--gamma", type=float, default=1.,
                        help="reward discount factor when calculating cumulate reward")
    parser.add_argument("--normalize_rewards", type=str2bool, nargs='?', default=True, choices=[True, False],
                        help="Whether to normalize rewards when calculate score")
    parser.add_argument("--eps", type=float, default=1e-3,
                        help="Range of preferences when normalizing cumulate rewards in neighborhood")
    parser.add_argument("--discount", type=int, default=0.997,
                        help="DON'T MESS WITH GAMMA, discount factor when calculating trajectory segment values")
    parser.add_argument("--weighted_score", type=str2bool, nargs='?', default=False, choices=[True, False],
                        help="Whether to weight the score by the preference. \
                              If true, the score of a trajectory ranges from 0 to its preference; otherwise, the score ranges from 0 to 1")
    parser.add_argument("--force_override", type=str2bool, nargs='?', default=False, choices=[True, False],
                        help="Whether to force override the existing dataset cache")
    parser.add_argument("--dropout_percent", type=float, default=0.3,
                        help="The percentage of trajectories to be dropped out")
    parser.add_argument("--dropout_points", type=int, default=3,
                        help="The number of centre points to be dropped out in preference space")
    parser.add_argument("--side", type=str2bool, nargs='?', default=False, choices=[True, False])
    
    # eval parameters
    parser.add_argument("--w_cg", type=float, default=0.01,
                        help="actually would be covered by dataset parameters")
    parser.add_argument("--num_envs", type=int, default=20,
                        help="The number of environments used for parallel evaluating")
    parser.add_argument("--num_prefs", type=int, default=100,
                        help="number of different input preferences when evaluating")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="number of evaluate episodes for each preference, use different seeds and calculate average")
    parser.add_argument("--n_elites", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="temperature for sampling")
    parser.add_argument("--uniform", action="store_true",
                        help="Whether sample preference points uniformly in [0, 1]")
    parser.add_argument("--cond_val", type=float, default=1.,
                        help="The value of the conditional sampling, usually set to 1 for best performance")
    parser.add_argument("--pref_weight", type=float, default=64.,
                        help="guidance weight of preference term in the loss function")
    
    args = parser.parse_args()
    assert args.normalize_rewards, "Diffuser only support normalized rewards for now, due to estimating return scales"

    dataset_name = args.dataset_name
    env_name, _, _, _ = dataset_name.split("_")
    args.env_name = env_name
    save_name = args.save_name
    save_path = f"results/{args.pipeline_name}/{dataset_name}/{save_name}/"

    if args.mode == "train":
        if args.load_model is not None:
            load_save_name = args.load_model.split("/")[:-1].join("/")
            load_ckpt  = args.load_model.split("/")[-1]
            if not os.path.exists(f"results/{args.pipeline_name}/{dataset_name}/{load_save_name}/diffusion_ckpt{load_ckpt}.pt") \
            or not os.path.exists(f"results/{args.pipeline_name}/{dataset_name}/{load_save_name}/classifier_ckpt{load_ckpt}.pt"):
                raise FileNotFoundError(f"Model {args.load_model} not found")
            
            if load_save_name == save_name:
                raise ValueError(f"Can't save model to a folder while load from it. Please simply use '--resume'")

            if os.path.exists(save_path) is False:
                os.makedirs(save_path)
                os.makedirs(save_path + 'rewards/')
                os.makedirs(save_path + 'prefs/')
                os.makedirs(save_path + 'plots/')
            elif args.save_name == "debug":
                import shutil
                shutil.rmtree(save_path)
                os.makedirs(save_path)
                os.makedirs(save_path + 'rewards/')
                os.makedirs(save_path + 'prefs/')
                os.makedirs(save_path + 'plots/')
            else:
                raise FileExistsError(f"Save path {save_path} already exists")

        else:
            if args.resume:
                if os.path.exists(save_path) is False:
                    os.makedirs(save_path)
                    os.makedirs(save_path + 'rewards/')
                    os.makedirs(save_path + 'prefs/')
                    os.makedirs(save_path + 'plots/')
                    max_available_ckpt = 0
                else:
                    max_diffusion_ckpt = 0
                    max_classifier_ckpt = 0
                    for file in os.listdir(save_path):
                        if file.startswith("diffusion_ckpt"):
                            if file.split("diffusion_ckpt")[1].split(".pt")[0].isnumeric():
                                max_diffusion_ckpt = max(int(file.split("diffusion_ckpt")[1].split(".pt")[0]), max_diffusion_ckpt)
                        elif file.startswith("classifier_ckpt"):
                            if file.split("classifier_ckpt")[1].split(".pt")[0].isnumeric():
                                max_classifier_ckpt = max(int(file.split("classifier_ckpt")[1].split(".pt")[0]), max_classifier_ckpt)
                    max_available_ckpt = min(max_diffusion_ckpt, max_classifier_ckpt)
            
            else:
                if os.path.exists(save_path) is False:
                    os.makedirs(save_path)
                    os.makedirs(save_path + 'rewards/')
                    os.makedirs(save_path + 'prefs/')
                    os.makedirs(save_path + 'plots/')
                elif args.save_name == "debug":
                    import shutil
                    shutil.rmtree(save_path)
                    os.makedirs(save_path)
                    os.makedirs(save_path + 'rewards/')
                    os.makedirs(save_path + 'prefs/')
                    os.makedirs(save_path + 'plots/')
                else:
                    raise FileExistsError(f"Save path {save_path} already exists")
    elif args.mode == "eval":
        if os.path.exists(save_path) is False:
            try:
                os.makedirs(save_path)
                os.makedirs(save_path + 'rewards/')
                os.makedirs(save_path + 'prefs/')
                os.makedirs(save_path + 'plots/')
            except:
                pass
    else:
        raise ValueError("Invalid mode")

    if "Hopper" in dataset_name or "Walker2d" in dataset_name:
        args.horizon = 32
    else:
        args.horizon = 4
    
    best_w_cgs = {
        "MO-Ant-v2_50000_amateur_uniform": 0.3,
        "MO-HalfCheetah-v2_50000_amateur_uniform": 0.5,
        "MO-Hopper-v2_50000_amateur_uniform": 0.5,
        "MO-Swimmer-v2_50000_amateur_uniform": 0.2,
        "MO-Walker2d-v2_50000_amateur_uniform": 0.2,
        "MO-Ant-v2_50000_expert_uniform": 1.0,
        "MO-HalfCheetah-v2_50000_expert_uniform": 2.0,
        "MO-Hopper-v2_50000_expert_uniform": 0.01,
        "MO-Swimmer-v2_50000_expert_uniform": 0.5,
        "MO-Walker2d-v2_50000_expert_uniform": 0.03,
    }
    args.w_cg = best_w_cgs[dataset_name]

    args_dict = vars(args)
    if args.mode == "train":
        with open(save_path+"args.json", "w") as f:
            json.dump(args_dict, f, indent=4)
    else:
        with open(save_path+f"args_{args.eval_name}.json", "w") as f:
            json.dump(args_dict, f, indent=4)
    
    set_seed(args.seed)

    if args.redirect:
        print("redirected stdout to log file")
        import sys
        if args.mode == "train":
            f = open(save_path+"log.txt", 'a', buffering=1)
        else:
            f = open(save_path+f"eval_log_{args.eval_name}.txt", 'w', buffering=1)
        sys.stdout=f
        print(f"Trial start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ---------------- Create Dataset ----------------

    env = gym.make(env_name)
    dataset = ShatteredPEDAMuJoCoDataset(
        dataset_name, terminal_penalty=args.terminal_penalty, horizon=args.horizon, avg=args.avg, gamma=args.gamma,
        normalize_rewards=args.normalize_rewards, eps=args.eps, discount=args.discount, force_override=args.force_override, weighted_score=args.weighted_score,
        dropout_percent=args.dropout_percent, dropout_points=args.dropout_points, side=args.side)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    obs_dim, act_dim, pref_dim = dataset.o_dim, dataset.a_dim, dataset.pref_dim

    # --------------- Create Diffusion Model -----------------

    if args.nn == "dit":
        nn_diffusion = DiT1d(
            obs_dim + act_dim , emb_dim=args.emb_dim, d_model=args.d_model, n_heads=args.n_heads, depth=args.depth,
            timestep_emb_type=args.time_emb_type).to(args.device)
        nn_classifier = HalfDiT1d(
            obs_dim + act_dim, out_dim=pref_dim*2, emb_dim=args.emb_dim, d_model=args.d_model//2, n_heads=args.n_heads, depth=max(args.depth//2, 1),
            timestep_emb_type=args.time_emb_type).to(args.device)
    elif args.nn == "unet":
        raise NotImplementedError

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"======================= Parameter Report of Classifier =======================")
    report_parameters(nn_classifier)
    print(f"==============================================================================")

    classifier = MOPrefClassifier(nn_classifier, device=args.device, pref_weight=args.pref_weight)

    fix_mask = torch.zeros((args.horizon, obs_dim + act_dim), device=args.device)
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.horizon, obs_dim + act_dim), device=args.device)
    loss_weight[1, :obs_dim] = 10.

    agent = ContinuousVPSDE(
            nn_diffusion, None, predict_noise=False,
            fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, noise_schedule="linear",
            device=args.device)

    if args.load_model:
        print(f"Loading cached model from: {f'results/{args.pipeline_name}/{dataset_name}/' + '/'.join(args.load_model.split('/')[:-1]) + '/diffusion_ckpt' + args.load_model.split('/')[-1] + '.pt'}")
        agent.load(f"results/{args.pipeline_name}/{dataset_name}/" + '/'.join(args.load_model.split('/')[:-1]) + '/diffusion_ckpt' + args.load_model.split('/')[-1] + '.pt')
        agent.classifier.load(f"results/{args.pipeline_name}/{dataset_name}/" + '/'.join(args.load_model.split('/')[:-1]) + '/classifier_ckpt' + args.load_model.split('/')[-1] + '.pt')
    elif args.resume and max_available_ckpt > 0:
        print(f"Loading cached model from: {save_path + 'diffusion_ckpt' + str(max_available_ckpt) + '.pt'}")
        agent.load(save_path + f"diffusion_ckpt{max_available_ckpt}.pt")
        agent.classifier.load(save_path + f"classifier_ckpt{max_available_ckpt}.pt")

    return_scale = (args.discount ** np.arange(args.horizon, dtype=np.float32)).sum()

    # --------------- Train -----------------

    if args.mode == "train":
        if args.resume:
            if args.load_model:
                step = int(args.load_model.split("/")[1])
            else:
                step = max_available_ckpt
            best_rewards = np.load(save_path + f"rewards/best.npy")
            best_hv = np.array([hv.compute(best_rewards[i]) for i in range(best_rewards.shape[0])]).mean()
        else:
            step = 0
            best_hv = 0.

        log = {"time":0., "gradient_steps": 0, "avg_loss_diffusion": 0., "avg_loss_classifier": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device)
            pref = batch["pref"].to(args.device)

            val = val / return_scale
            cond = torch.cat([pref, val], dim=-1)

            x = torch.cat([obs, act], -1)
            
            log["avg_loss_diffusion"] += agent.update(x)['loss']

            if step <= 500_000:
                log["avg_loss_classifier"] += agent.update_classifier(x, cond)['loss']

            if (step + 1) % 1000 == 0:
                log["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log["gradient_steps"] = step + 1
                log["avg_loss_diffusion"] /= 1000
                log["avg_loss_classifier"] /= 1000
                print(log)
                log = {"time":0., "gradient_steps": 0, "avg_loss_diffusion": 0., "avg_loss_classifier": 0.}
            
            if (step + 1) % 100_000 == 0:
                agent.eval()
                eval_rewards, eval_lens, eval_prefs = inference(dataset, agent, args, args.uniform)
                eval_hv = np.array([hv.compute(eval_rewards[i]) for i in range(eval_rewards.shape[0])]).mean()
                # print(eval_rewards, eval_lens, (eval_rewards * eval_prefs).sum(-1)/np.linalg.norm(eval_rewards, ord=2, axis=-1)/np.linalg.norm(eval_prefs, ord=2, axis=-1))
                print(f"average rewards: {(eval_rewards * eval_prefs).sum(-1).mean()}, average lens: {eval_lens.mean()}, average hvs: {eval_hv.mean()} ", end="")
                print(f"average dis: {((eval_rewards * eval_prefs).sum(-1)/np.linalg.norm(eval_rewards, ord=2, axis=-1)/np.linalg.norm(eval_prefs, ord=2, axis=-1)).mean()}")

                if eval_hv.mean() > best_hv:
                    best_hv = eval_hv.mean()
                    
                    np.save(save_path + "rewards/" + f"best.npy", eval_rewards)
                    np.save(save_path + "prefs/"   + f"best.npy", eval_prefs)

                    fig, ax = plt.subplots()
                    ax.scatter(dataset.seq_rew[::50, :, 0].sum(1), dataset.seq_rew[::50, :, 1].sum(1), c="y", alpha=0.5, edgecolor='none')
                    ax.scatter(eval_rewards[:, :, 0].mean(0), eval_rewards[:, :, 1].mean(0), c="r", edgecolor='none')
                    ax.set_title(f"{dataset_name}, step {step + 1}")
                    fig.savefig(save_path + "plots/" + f"best.pdf")

                    agent.save(save_path + f"diffusion_ckpt_best.pt")
                    agent.classifier.save(save_path + f"classifier_ckpt_best.pt")


                if (step + 1) % 100_000 == 0:
                    np.save(save_path + "rewards/" + f"{step + 1}.npy", eval_rewards)
                    np.save(save_path + "prefs/"   + f"{step + 1}.npy", eval_prefs)

                    fig, ax = plt.subplots()
                    ax.scatter(dataset.seq_rew[::50, :, 0].sum(1), dataset.seq_rew[::50, :, 1].sum(1), c="y", alpha=0.5, edgecolor='none')
                    ax.scatter(eval_rewards[:, :, 0].mean(0), eval_rewards[:, :, 1].mean(0), c="r", edgecolor='none')
                    ax.set_title(f"{dataset_name}, step {step + 1}")
                    fig.savefig(save_path + "plots/" + f"{step + 1}.pdf")

                agent.train()
            
            if (step + 1) % 100_000 == 0:
                agent.save(save_path + f"diffusion_ckpt{step + 1}.pt")
                agent.classifier.save(save_path + f"classifier_ckpt{step + 1}.pt")
            
            step += 1
            if step >= args.n_gradient_steps:
                break
        
    elif args.mode == "eval":
        if not args.load_model and not args.resume:
            import warnings
            warnings.warn("You are running evaluation without loading any model")

        agent.eval()
        eval_rewards, eval_lens, eval_prefs = inference(dataset, agent, args, args.uniform)
        eval_hv = np.array([hv.compute(eval_rewards[i]) for i in range(eval_rewards.shape[0])]).mean()
        # print(eval_rewards, eval_lens, (eval_rewards * eval_prefs).sum(-1)/np.linalg.norm(eval_rewards, ord=2, axis=-1)/np.linalg.norm(eval_prefs, ord=2, axis=-1))
        print(f"average rewards: {(eval_rewards * eval_prefs).sum(-1).mean()}, average lens: {eval_lens.mean()}, average hvs: {eval_hv.mean()} ", end="")
        print(f"average dis: {((eval_rewards * eval_prefs).sum(-1)/np.linalg.norm(eval_rewards, ord=2, axis=-1)/np.linalg.norm(eval_prefs, ord=2, axis=-1)).mean()}")

        np.save(save_path + f"rewards/{args.eval_name}.npy", eval_rewards)
        np.save(save_path + f"prefs/{args.eval_name}.npy", eval_prefs)
        
        fig, ax = plt.subplots()
        ax.scatter(dataset.seq_rew[::50, :, 0].sum(1), dataset.seq_rew[::50, :, 1].sum(1), c="y", alpha=0.5, edgecolor='none')
        ax.scatter(eval_rewards[:, :, 0].mean(0), eval_rewards[:, :, 1].mean(0), c="r", edgecolor='none')
        ax.set_title(f"{dataset_name}")
        fig.savefig(save_path + f"plots/{args.eval_name}.pdf")