import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset

dataset = PEDAMuJoCoDataset(
        "MO-HalfCheetah-v2_50000_amateur_uniform", terminal_penalty=100, horizon=4, avg=True, gamma=1.,
        normalize_rewards=True, eps=1e-3, discount=0.997)


sum_rew = dataset.seq_rew[dataset.front_idx].sum(1)
prefs = dataset.seq_pref[dataset.front_idx]

fig, ax = plt.subplots()
max_1 = np.max(sum_rew[:, 0])
max_2 = np.max(sum_rew[:, 1])
ax.scatter(sum_rew[:, 0]/max_1, sum_rew[:, 1]/max_2, label="front1")
# fig.savefig("plots/front.pdf")

ax.scatter(sum_rew[:, 0]/prefs[:, 0]/max_1, sum_rew[:, 1]/prefs[:, 1]/max_2, label="front2")

approx = nn.Sequential(
    nn.Linear(2, 128),
    nn.Mish(),
    nn.Linear(128, 512),
    nn.Mish(),
    nn.Linear(512, 512),
    nn.Mish(),
    nn.Linear(512, 2)
)

approx.load_state_dict(torch.load("results/front_approx/MO-HalfCheetah-v2_50000_amateur_uniform/approx_ckpt100000.pt"))

front = approx(torch.from_numpy(prefs)).detach().numpy()
ax.scatter(sum_rew[:, 0]/front[:, 0], sum_rew[:, 1]/front[:, 1], label="front3")
ax.legend()

fig.savefig("plots/front.pdf")

fig, ax = plt.subplots()
ax.scatter(sum_rew[:, 0], sum_rew[:, 1], label="ground truth")
ax.scatter(prefs[:, 0]*max_1, prefs[:, 1]*max_2, label="pref weighted max")
ax.scatter(front[:, 0], front[:, 1], label="approx")
ax.legend()
fig.savefig("plots/front_approx.pdf")

fig, ax = plt.subplots()
ax.plot(dataset.seq_score[0, :, 0])
fig.savefig("plots/score.pdf")

fig, ax = plt.subplots()
ax.plot(dataset.seq_score_copy[1000, :200, 0])
ax.plot(dataset.neighbor_max_score[1000, :200, 0])
ax.plot(dataset.neighbor_min_score[1000, :200, 0])

fig.savefig("plots/norm_returns.pdf")

