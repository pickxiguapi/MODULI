import argparse
import numpy as np
import torch
import h5py
import time
from tqdm import tqdm
import gym
import environments
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
import matplotlib.pyplot as plt
import os
import shutil
import torch
import torch.nn as nn
import pickle
from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pipeline_name", type=str, default="mo_bc")
    parser.add_argument("--dataset_name", type=str, default="MO-Hopper-v2_50000_expert_uniform")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--save_name", type=str, default="debug")
    parser.add_argument("--model_name", type=str, default="bc_MO-Hopper-v2_expert_uniform")
    parser.add_argument("--eval_cnt", type=int, default=2)
    args = parser.parse_args()

    args.horizon=32
    seed = args.seed
    dataset_name = args.dataset_name
    device = args.device
    save_name = args.save_name
    model_name = args.model_name
    eval_cnt = args.eval_cnt

    env_name, _, _, _ = dataset_name.split("_")
    save_path = f"results/{args.pipeline_name}/{env_name}/"
    set_seed(seed)

    model = nn.Sequential(
        nn.Linear(15, 1024),
        nn.ReLU(),
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 3)
    ).to(device)
    model.load_state_dict(torch.load(save_path+model_name+".pt"))
    model.eval()
    
    if os.path.exists(f"eval/{save_name}"):
        shutil.rmtree(f"eval/{save_name}")
    os.makedirs(f"eval/{save_name}")

    with open(f"eval/{save_name}/log.txt", "a") as f:
        print(f"bc, env:{env_name}, dataset:{dataset_name}, seed:{seed}, device:{device}", file=f)
    
    dataset = PEDAMuJoCoDataset(
        dataset_name, terminal_penalty=0., horizon=args.horizon, avg=True, gamma=0,
        normalize_rewards=True, eps=1e-3, discount=0.997)

    normalizer = dataset.get_normalizer()
    
    env = gym.make(env_name)

    # cal_prefs = torch.FloatTensor(np.random.uniform(0.3208, 0.5647, eval_cnt)).reshape((-1, 1))
    cal_prefs = torch.linspace(0.3208, 0.5647, eval_cnt).reshape((-1, 1))
    cal_prefs = torch.cat([cal_prefs, 1-cal_prefs], dim=1).to(device)
    test_prefs = cal_prefs.clone().to(device)

    lens = np.zeros((eval_cnt,))
    raw_rewards = np.zeros((eval_cnt, 500, 2))

    for i, test_pref in enumerate(test_prefs):
        print(i+1, test_pref)
        test_pref = test_pref.unsqueeze(0)
        obs = env.reset()
        obs_ds = torch.FloatTensor(obs)

        obs_ds = normalizer.normalize(obs_ds).reshape(1, -1).to(device)

        # nor_obs_ds = torch.cat([nor_obs_ds, torch.FloatTensor([1.0, 1.0]).reshape(1,-1).to(device)], dim=-1)

        eval_step = 0
        while True:
            # print(obs_ds.shape, test_pref.shape, torch.ones((1, 2)).shape)
            a = model(torch.cat([obs_ds, test_pref, torch.ones((1, 2), device=device)], axis=1))
            a = a.detach().cpu().numpy().reshape(-1)
            print(a)
            obs, _, terminated, raw_reward = env.step(a)
            obs_ds = torch.FloatTensor(obs)
            obs_ds = normalizer.normalize(obs_ds).reshape(1, -1).to(device)

            raw_rewards[i, eval_step, :] = raw_reward["obj"]
            # print(raw_reward["obj"])
            eval_step += 1
            if terminated:
                break
        lens[i] = eval_step
    
    print(lens)
    # print(raw_rewards.sum(axis=1))
    print((raw_rewards.sum(axis=1)*test_prefs.cpu().numpy()).sum(-1))
    # print((raw_rewards.sum(axis=1)*test_prefs.cpu().numpy()).sum(-1).mean())
    # print(test_prefs.cpu().numpy())

    # np.save(f"eval/{save_name}/lens.npy", lens)
    # np.save(f"eval/{save_name}/raw_rewards.npy", raw_rewards)
    # np.save(f"eval/{save_name}/preference.npy", test_prefs.cpu().numpy())
            
main()