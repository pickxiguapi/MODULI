import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from tqdm import tqdm
import time
import os
import gym
import environments
from diffusion.ode import ODE
from diffusion.utils import GaussianNormalizer, count_parameters, set_seed
import pickle

device = "cuda"
env_name = "MO-Hopper-v2"
dataset = "expert_uniform"

with h5py.File(f"data/{env_name}/{env_name}_50000_{dataset}_4.hdf5", "r") as f:
    _obs, _act = f["observations"][:], f["actions"][:]
    _pref, _raw_reward = f["preferences"][:], f["raw_rewards"][:]
    _len = f["traj_lengths"][:]
    _score = f["score"][:]

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
optim = optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True)

_pref = np.repeat(_pref.reshape((-1, 1, 2)), 500, axis=1)
obs_ds, act_ds = torch.FloatTensor(_obs).to(device), torch.FloatTensor(_act).to(device)
pref_ds, score_ds = torch.FloatTensor(_pref).to(device), torch.FloatTensor(_score).to(device)
normalizer = GaussianNormalizer(obs_ds)
nor_obs_ds = normalizer.normalize(obs_ds)
nor_obs_ds = torch.cat([nor_obs_ds, pref_ds, score_ds], dim=-1)
obs_ds.cpu()
nor_obs_ds = nor_obs_ds.reshape((-1, 15))
act_ds = act_ds.reshape((-1, 3))

batch_size = 16384
n_steps = 200000

pbar = tqdm(range(n_steps))
env = gym.make(env_name)
loss_avg = 0.
start_step = 2e4
max_score = 0.
for step in range(n_steps):
    idx = np.random.randint(0, len(nor_obs_ds), batch_size)
    obs, act = nor_obs_ds[idx], act_ds[idx]
    pred = model(obs)
    loss = ((pred - act) ** 2).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()

    if (step) % 1000 == 0:
        model.eval()
        length, reward = 0, 0
        eval_cnt = 30
        test_prefs = torch.linspace(_pref[:,0].min(), _pref[:,0].max(), eval_cnt).reshape((-1, 1))
        test_prefs = torch.cat([test_prefs, 1-test_prefs], dim=1).to(device)
        pref_distances, weighted_rewards, eval_steps = [], [], []
        for i in range(eval_cnt):
            obs = env.reset()
            obs_ds = torch.FloatTensor(obs).to(device)
            obs_ds = normalizer.normalize(obs_ds).reshape(1, -1)
            reward = np.zeros((2,))
            weighted_reward = 0.
            eval_step = 0
            
            while True:
                eval_step += 1
                a = model(torch.cat([obs_ds, test_prefs[i].reshape((1, 2)), torch.ones((1, 2), device=device)], axis=1))
                a = a.detach().cpu().numpy().reshape(-1)
                obs, _, terminated, raw_reward = env.step(a)
                obs_ds = torch.FloatTensor(obs).to(device)
                obs_ds = normalizer.normalize(obs_ds).reshape(1, -1)

                # reward += raw_reward["obj"].mean()
                reward += raw_reward["obj"]
                weighted_reward += (raw_reward["obj"] * test_prefs[i].cpu().numpy()).sum()
                if terminated:
                    break

            reward /= reward.sum()
            cal_pref_cpu = test_prefs[i].cpu().numpy()
            pref_distances.append((reward * cal_pref_cpu).sum()/np.linalg.norm(reward)/np.linalg.norm(cal_pref_cpu))
            weighted_rewards.append(weighted_reward)
            eval_steps.append(eval_step)
        pbar.set_description(f'step: {step+1} loss: {loss_avg / 1000.}, reward: {sum(weighted_rewards)/eval_cnt}')
        pbar.update(1000)
        if sum(weighted_rewards)/eval_cnt > max_score:
            print("new max score", sum(weighted_rewards)/eval_cnt, "saving model")
            max_score = sum(weighted_rewards)/eval_cnt
            torch.save(model.state_dict(), f"bc_{env_name}_{dataset}.pt")
        with open("bc_log.txt", "a") as f:
            print(f"Step {step+1}, dis: {np.mean(np.array(pref_distances))}, len: {np.mean(np.array(eval_steps))}, reward: {np.mean(np.array(weighted_rewards))}", file=f)
        model.train()


