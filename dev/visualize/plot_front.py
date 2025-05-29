import numpy as np
import matplotlib.pyplot as plt
from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset

fig, ax = plt.subplots()
raw_data = np.load("dev/data/raw_rewards/MO-Ant-v2_50000_amateur_uniform.npy")
data1 = np.load("results/mo_dd/MO-Ant-v2_50000_amateur_uniform/model/nheads6_dmodel384_depth8/rewards/best.npy").reshape((-1, 2))
data2 = np.load("results/mo_diffuser_prefguide/MO-Ant-v2_50000_amateur_uniform/model/22M/rewards/best.npy").reshape((-1, 2))

ax.scatter(raw_data[:, 0], raw_data[:, 1], c="y", alpha=0.5, edgecolor='none')
ax.scatter(data1[:, 0], data1[:, 1], c="r", edgecolor='none', label="dd")
ax.scatter(data2[:, 0], data2[:, 1], c="b", edgecolor='none', label="diffuser")
ax.legend()
fig.savefig("plots/ant_dd_diffuser.pdf")