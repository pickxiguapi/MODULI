import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset

d4morl_data = np.load(f"dev/data/raw_rewards/MO-Walker2d-v2_50000_amateur_uniform.npy")
prefs = np.load(f"dev/data/dataset_prefs/MO-Walker2d-v2_50000_amateur_uniform.npy")

d4morl_data = d4morl_data[::100]
prefs = prefs[::100]

approx = nn.Sequential(
    nn.Linear(2, 128),
    nn.Mish(),
    nn.Linear(128, 512),
    nn.Mish(),
    nn.Linear(512, 512),
    nn.Mish(),
    nn.Linear(512, 2)
)
approx.load_state_dict(torch.load("results/front_approx/MO-Walker2d-v2_50000_amateur_uniform/approx_ckpt100000.pt"))
approx.eval()
pred = approx(torch.from_numpy(prefs)).detach().numpy()

fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharex=True)
plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.17, wspace=0.05)
axs[1].set_yticks([])
axs[2].set_yticks([])

axs[0].scatter(d4morl_data[:, 0], d4morl_data[:, 1], edgecolor='lightgrey', facecolor='none')
axs[1].scatter(d4morl_data[:, 0], d4morl_data[:, 1], edgecolor='lightgrey', facecolor='none')
axs[2].scatter(d4morl_data[:, 0], d4morl_data[:, 1], edgecolor='lightgrey', facecolor='none')

axs[0].scatter(pred[:, 0], pred[:, 1], edgecolor='crimson', facecolor='none', label='PPN')

npn = []
from math import *
for i in range(len(prefs)):
    npn.append(np.max(d4morl_data[max(0, i-5):min(len(d4morl_data), i+5)], axis=0))

npn = np.array(npn)
axs[1].scatter(npn[:, 0], npn[:, 1], edgecolor='crimson', facecolor='none', label='NPN')  

axs[2].scatter(np.max(d4morl_data[:, 0]), np.max(d4morl_data[:, 1]), color='crimson', label='Global')

# axs[0].set_xlabel("Objective 1", fontsize=15)
axs[0].set_ylabel("Objective 2", fontsize=20)
axs[1].set_xlabel("Objective 1", fontsize=20)
# axs[2].set_xlabel("Objective 1", fontsize=15)

axs[0].legend(fontsize="20", loc="lower right")
axs[1].legend(fontsize="20", loc="lower right")
axs[2].legend(fontsize="20", loc="lower right")
fig.savefig("plots/train_sample_compare.pdf")

