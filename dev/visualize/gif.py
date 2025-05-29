import numpy as np
from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation

eval_data = np.load("results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/model/nheads6_dmodel384_depth8_fourier/rewards/best.npy")
eval_pref = np.load("results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/model/nheads6_dmodel384_depth8_fourier/prefs/best.npy")
raw_data = np.load("dev/data/raw_rewards/MO-HalfCheetah-v2_50000_expert_uniform.npy")

fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# 定义一个更新函数，用于更新每一帧的点的位置
def update(i):
    ax.clear()
    ax.scatter(raw_data[:, 0], raw_data[:, 1], c="y", alpha=0.5, edgecolor='none')
    ax.scatter(eval_data[0, i, 0], eval_data[0, i, 1], c="r", edgecolor='none', label=eval_pref[i, 0])
    ax.legend()

# 创建动画对象
ani = FuncAnimation(fig, update, frames=range(100), interval=300)
ani.save('sin_wave.gif', writer='imagemagick')