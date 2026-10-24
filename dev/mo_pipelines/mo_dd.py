import argparse
import os
from datetime import datetime
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion.vpsde import ContinuousVPSDE
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.utils import report_parameters
import environments
from dev.PEDA.modt.hypervolume import InnerHyperVolume
hv = InnerHyperVolume([0, 0])
from dev.utils.utils import EmbMLPCondition
from time import time

def inference(dataset, agent, invdyn, args, uniform=False):
    first_pref = torch.linspace(0, 1, args.num_prefs) if uniform else torch.full((args.num_prefs,), 0.5)
    prefs = torch.stack([first_pref, 1 - first_pref], dim=-1)
    condition = torch.cat([prefs, torch.full_like(prefs, args.cond_val)], dim=-1).to(args.device)
    rewards = np.zeros((args.num_episodes, args.num_prefs, 2))
    lengths = np.zeros((args.num_episodes, args.num_prefs), dtype=int)
    normalizers = dataset.get_normalizer()

    for episode in range(args.num_episodes):
        seed = args.seed + episode
        set_seed(seed)
        for start in range(0, args.num_prefs, args.num_envs):
            count = min(args.num_envs, args.num_prefs - start)
            env = gym.vector.make(args.env_name, count)
            try:
                obs = env.reset(seed=seed)
                finished = np.zeros(count, dtype=bool)
                for _ in range(500):
                    state = torch.as_tensor(normalizers["state"].normalize(obs), device=args.device, dtype=torch.float32)
                    prior = torch.zeros((count, args.horizon, dataset.o_dim), device=args.device)
                    prior[:, 0] = state
                    trajectory, _ = agent.sample(
                        prior, n_samples=count, sample_steps=args.sample_steps,
                        sample_step_schedule=args.sample_step_schedule, use_ema=True, solver=args.solver,
                        condition_cfg=condition[start:start + count], w_cfg=args.w_cfg, temperature=args.temperature)
                    action = invdyn.predict(state, trajectory[:, 1]).cpu().numpy()
                    action = normalizers["action"].unnormalize(action)
                    obs, _, done, infos = env.step(action)
                    for index, info in enumerate(infos):
                        if not finished[index]:
                            rewards[episode, start + index] += info["obj"]
                            lengths[episode, start + index] += 1
                    finished |= done
                    if finished.all():
                        break
            finally:
                env.close()
    return rewards, lengths, prefs.numpy()


def save_evaluation(dataset, rewards, prefs, save_path, name):
    np.save(save_path + f"rewards/{name}.npy", rewards)
    np.save(save_path + f"prefs/{name}.npy", prefs)
    fig, ax = plt.subplots()
    front = dataset.seq_rew[::50].sum(1)
    ax.scatter(front[:, 0], front[:, 1], c="y", alpha=0.5, edgecolor="none")
    ax.scatter(*rewards.mean(0).T, c="r", edgecolor="none")
    fig.savefig(save_path + f"plots/{name}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, default="mo_dd")
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
    parser.add_argument("--eval_interval", type=int, default=20_000, help="0 disables evaluation during training")
    parser.add_argument("--save_interval", type=int, default=100_000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--redirect", action="store_true",
                        help="redirect stdout to log file")
    parser.add_argument("--resume", action="store_true",
                        help="continue training from the loaded model's step")
    parser.add_argument("--invdyn", type=str, default="500000")

    # diffusion parameters
    parser.add_argument("--solver", type=str, default="ddim")
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--sample_step_schedule", type=str, default="uniform_continuous")
    parser.add_argument("--nn", choices=["dit"], default="dit")
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--time_emb_type", type=str, default="positional")
    parser.add_argument("--cond_emb_type", type=str, default="None")

    # dataset parameters
    parser.add_argument("--terminal_penalty", type=float, default=100.,
                        help="positive value for negtive reward.")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--avg", type=str2bool, nargs='?', default=True, choices=[True, False], 
                        help="Whether to divide cumulate reward by (max steps - current steps)")
    parser.add_argument("--gamma", type=float, default=1.,
                        help="reward discount factor when calculating cumulate reward")
    parser.add_argument("--normalize_rewards", type=str2bool, nargs='?', default=True, choices=[True, False],
                        help="Whether to normalize rewards when calculate score")
    parser.add_argument("--eps", type=float, default=1e-3,
                        help="Range of preferences when normalizing cumulate rewards in neighborhood")
    parser.add_argument("--discount", type=float, default=0.997,
                        help="DON'T MESS WITH GAMMA, discount factor when calculating trajectory segment values")
    parser.add_argument("--weighted_score", type=str2bool, nargs='?', default=False, choices=[True, False],
                        help="Whether to weight the score by the preference. \
                              If true, the score of a trajectory ranges from 0 to its preference; otherwise, the score ranges from 0 to 1")
    parser.add_argument("--force_override", type=str2bool, nargs='?', default=False, choices=[True, False],
                        help="Whether to force override the existing dataset cache")
    
    # eval parameters
    parser.add_argument("--w_cfg", type=float, default=1.5)
    parser.add_argument("--num_envs", type=int, default=20,
                        help="The number of environments used for parallel evaluating")
    parser.add_argument("--num_prefs", type=int, default=100,
                        help="number of different input preferences when evaluating")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="number of evaluate episodes for each preference, use different seeds and calculate average")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="temperature for sampling")
    parser.add_argument("--uniform", action="store_true",
                        help="Whether sample preference points uniformly in [0, 1]")
    parser.add_argument("--cond_val", type=float, default=1.,
                        help="The value of the conditional sampling, usually set to 1 for best performance")

    args = parser.parse_args()
    assert args.normalize_rewards, "DD only support normalized rewards for now, due to estimating return scales"

    dataset_name = args.dataset_name
    env_name, _, _, _ = dataset_name.split("_")
    args.env_name = env_name
    save_name = args.save_name
    save_path = f"results/{args.pipeline_name}/{dataset_name}/{save_name}/"

    if args.mode not in ("train", "eval"):
        parser.error("mode must be train or eval")
    if args.resume and args.load_model:
        parser.error("use either --resume or --load_model")
    if args.n_gradient_steps < 1 or args.save_interval < 1 or args.eval_interval < 0:
        parser.error("steps/save interval must be positive and eval interval nonnegative")
    if min(args.num_envs, args.num_prefs, args.num_episodes, args.batch_size) < 1:
        parser.error("batch and evaluation sizes must be positive")
    if args.mode == "eval" and not (args.load_model or args.resume):
        parser.error("evaluation requires --load_model or --resume")
    output = Path(save_path)
    max_available_ckpt = max(
        (int(path.stem.removeprefix("diffusion_ckpt")) for path in output.glob("diffusion_ckpt*.pt")
         if path.stem.removeprefix("diffusion_ckpt").isdigit()), default=0)
    if args.mode == "eval" and args.resume and not max_available_ckpt:
        parser.error("no checkpoint available to evaluate")
    if args.mode == "train" and output.exists() and not args.resume:
        raise FileExistsError(f"Save path {save_path} already exists; use a new --save_name or --resume")
    checkpoint = None
    if args.load_model:
        source = Path(args.load_model)
        checkpoint = Path("results") / args.pipeline_name / dataset_name / source.parent / f"diffusion_ckpt{source.name}.pt"
    elif args.resume and max_available_ckpt:
        checkpoint = output / f"diffusion_ckpt{max_available_ckpt}.pt"
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for subdir in ("rewards", "prefs", "plots"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    if args.horizon is None:
        args.horizon = 32 if "Hopper" in dataset_name or "Walker2d" in dataset_name else 4
    if args.horizon < 2:
        parser.error("horizon must be at least 2 for inverse dynamics")

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

    dataset = PEDAMuJoCoDataset(
        dataset_name, terminal_penalty=args.terminal_penalty, horizon=args.horizon, avg=args.avg, gamma=args.gamma,
        normalize_rewards=args.normalize_rewards, eps=args.eps, discount=args.discount, force_override=args.force_override, weighted_score=args.weighted_score)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    if len(dataset) < args.batch_size:
        raise ValueError("Dataset contains fewer trajectory segments than batch_size")
    obs_dim, act_dim, pref_dim = dataset.o_dim, dataset.a_dim, dataset.pref_dim

    # --------------- Create Diffusion Model -----------------

    if args.nn == "dit":
        nn_diffusion = DiT1d(
            obs_dim , emb_dim=args.emb_dim, d_model=args.d_model, n_heads=args.n_heads, depth=args.depth,
            timestep_emb_type=args.time_emb_type).to(args.device)
        nn_condition = EmbMLPCondition(
            2 * pref_dim, 128, [128, ], nn.SiLU(), 0.25, args.cond_emb_type).to(args.device)

    invdyn = None
    if args.mode == "eval" or args.eval_interval:
        invdyn = MlpInvDynamic(obs_dim, act_dim, 1024, optim_params={"lr": 5e-4}, device=args.device)
        invdyn.load(f"results/mo_dd_invdyn/{dataset_name}/invdyn_ckpt{args.invdyn}.pt")
        invdyn.eval()

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    fix_mask = torch.zeros((args.horizon, obs_dim), device=args.device)
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.horizon, obs_dim), device=args.device)
    loss_weight[1, :obs_dim] = 10.

    agent = ContinuousVPSDE(
            nn_diffusion, nn_condition, predict_noise=False,
            fix_mask=fix_mask, loss_weight=loss_weight, noise_schedule="linear",
            device=args.device)
    
    if checkpoint is not None:
        print(f"Loading model from: {checkpoint}")
        agent.load(str(checkpoint))

    return_scale = (args.discount ** np.arange(args.horizon, dtype=np.float32)).sum()

    # --------------- Train -----------------

    if args.mode == "train":
        step = max_available_ckpt if args.resume else 0
        if step >= args.n_gradient_steps:
            raise ValueError("Checkpoint already reached n_gradient_steps")
        best_hv = 0.
        if args.resume and os.path.exists(save_path + "rewards/best.npy"):
            best_rewards = np.load(save_path + "rewards/best.npy")
            best_hv = np.mean([hv.compute(rewards) for rewards in best_rewards])
        start_step = step

        log = {"time":0., "gradient_steps": 0, "avg_loss_diffusion": 0.}
        start_time = time()
        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            val = batch["val"].to(args.device)
            pref = batch["pref"].to(args.device)

            val = val / return_scale
            cond = torch.cat([pref, val], dim=-1)
            
            
            log["avg_loss_diffusion"] += agent.update(obs, cond)['loss']

            if (step + 1) % 1000 == 0:
                log["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log["gradient_steps"] = step + 1
                log["avg_loss_diffusion"] /= 1000
                current_time = time()
                log["time_per_1000_steps"] = (current_time - start_time) / (step + 1 - start_step) * 1000
                print(f"Step {step + 1}, loss: {log['avg_loss_diffusion']:.6f}, s/kiter: {log['time_per_1000_steps']:.2f}")
                log["avg_loss_diffusion"] = 0.
            
            if args.eval_interval and (step + 1) % args.eval_interval == 0:
                agent.eval()
                eval_rewards, eval_lens, eval_prefs = inference(dataset, agent, invdyn, args, args.uniform)
                eval_hv = np.array([hv.compute(eval_rewards[i]) for i in range(eval_rewards.shape[0])]).mean()
                print(f"average rewards: {(eval_rewards * eval_prefs).sum(-1).mean()}, average lens: {eval_lens.mean()}, average hvs: {eval_hv.mean()} ", end="")
                print(f"average dis: {((eval_rewards * eval_prefs).sum(-1)/np.linalg.norm(eval_rewards, ord=2, axis=-1)/np.linalg.norm(eval_prefs, ord=2, axis=-1)).mean()}")

                if eval_hv.mean() > best_hv:
                    best_hv = eval_hv.mean()
                    
                    save_evaluation(dataset, eval_rewards, eval_prefs, save_path, "best")

                    agent.save(save_path + f"diffusion_ckpt_best.pt")


                save_evaluation(dataset, eval_rewards, eval_prefs, save_path, str(step + 1))

                agent.train()
            
            if (step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt{step + 1}.pt")
            
            step += 1
            if step >= args.n_gradient_steps:
                break
        if step % args.save_interval:
            agent.save(save_path + f"diffusion_ckpt{step}.pt")
        
    elif args.mode == "eval":
        agent.eval()
        eval_rewards, eval_lens, eval_prefs = inference(dataset, agent, invdyn, args, args.uniform)
        eval_hv = np.array([hv.compute(eval_rewards[i]) for i in range(eval_rewards.shape[0])]).mean()
        print(f"average rewards: {(eval_rewards * eval_prefs).sum(-1).mean()}, average lens: {eval_lens.mean()}, average hvs: {eval_hv.mean()} ", end="")
        print(f"average dis: {((eval_rewards * eval_prefs).sum(-1)/np.linalg.norm(eval_rewards, ord=2, axis=-1)/np.linalg.norm(eval_prefs, ord=2, axis=-1)).mean()}")
        
        save_evaluation(dataset, eval_rewards, eval_prefs, save_path, args.eval_name)
