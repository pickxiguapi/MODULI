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

fig, axs = plt.subplots(1, 1, figsize=(6, 6))
plt.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.1)
# axs.set_yticks([])
# axs.set_yticks([])

# axs.scatter(d4morl_data[:, 0], d4morl_data[:, 1], edgecolor='lightgrey', facecolor='none', label='train target')
# axs.scatter(d4morl_data[:, 0], d4morl_data[:, 1], edgecolor='lightgrey', facecolor='none', label='train target')
# axs.scatter(d4morl_data[:, 0], d4morl_data[:, 1], edgecolor='lightgrey', facecolor='none', label='train target')

axs.scatter(d4morl_data[:, 0]/pred[:, 0], d4morl_data[:, 1]/pred[:, 1], ec="green", fc="none", label='PPN')

npn = []
npn_min = []
from math import *
for i in range(len(prefs)):
    npn.append(np.max(d4morl_data[max(0, i-5):min(len(d4morl_data), i+5)], axis=0))
    npn_min.append(np.min(d4morl_data[max(0, i-5):min(len(d4morl_data), i+5)], axis=0))

npn = np.array(npn)
npn_min = np.array(npn_min)
axs.scatter((d4morl_data[:, 0]-npn_min[:, 0])/(npn[:, 0]-npn_min[:,0]), (d4morl_data[:, 1]-npn_min[:, 1])/(npn[:, 1]-npn_min[:,1]), ec="royalblue", fc="none", label='NPN')  

axs.scatter((d4morl_data[:, 0]-np.min(d4morl_data[:, 0]))/np.max(d4morl_data[:, 0]), (d4morl_data[:, 1]-np.min(d4morl_data[:, 1]))/np.max(d4morl_data[:, 1]), ec="crimson", fc="none", label='Global Norm')
axs.scatter(1, 1, fc="orange", ec='none', s=100, label='Sample target')

axs.set_xlabel("Objective 1", fontsize=20)
axs.set_ylabel("Objective 2", fontsize=20)

axs.set_xlim(-0.05, 1.15)
axs.set_ylim(-0.05, 1.15)

axs.legend(fontsize="18", loc="lower right")

fig.savefig("plots/train_sample_compare_2.pdf")

