import numpy as np
import matplotlib.pyplot as plt
from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset

fig, ax = plt.subplots(2, 5, figsize=(30, 12))

list1 = [
    "results/mo_dd_shattered/MO-Ant-v2_50000_amateur_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_shattered/MO-HalfCheetah-v2_50000_amateur_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_shattered/MO-Hopper-v2_50000_amateur_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_shattered/MO-Swimmer-v2_50000_amateur_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_shattered/MO-Walker2d-v2_50000_amateur_uniform/model/11M_45/rewards/best.npy",
    "results/mo_dd_shattered/MO-Ant-v2_50000_expert_uniform/model/11M_45/rewards/best.npy",
    "results/mo_dd_shattered/MO-HalfCheetah-v2_50000_expert_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_shattered/MO-Hopper-v2_50000_expert_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_shattered/MO-Swimmer-v2_50000_expert_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_shattered/MO-Walker2d-v2_50000_expert_uniform/model/11M_45/rewards/best.npy",
]

list2 = [
    "results/mo_dd_sepguide_pro_shattered/MO-Ant-v2_50000_amateur_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_sepguide_pro_shattered/MO-HalfCheetah-v2_50000_amateur_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_sepguide_pro_shattered/MO-Hopper-v2_50000_amateur_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_sepguide_pro_shattered/MO-Swimmer-v2_50000_amateur_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_sepguide_pro_shattered/MO-Walker2d-v2_50000_amateur_uniform/model/11M_45/rewards/best.npy",
    "results/mo_dd_sepguide_pro_shattered/MO-Ant-v2_50000_expert_uniform/model/11M_45/rewards/best.npy",
    "results/mo_dd_sepguide_pro_shattered/MO-HalfCheetah-v2_50000_expert_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_sepguide_pro_shattered/MO-Hopper-v2_50000_expert_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_sepguide_pro_shattered/MO-Swimmer-v2_50000_expert_uniform/model/11M_45/rewards/best.npy",
    # "results/mo_dd_sepguide_pro_shattered/MO-Walker2d-v2_50000_expert_uniform/model/11M_45/rewards/best.npy",
]

for i in range(len(list1)):
    eval_data = np.load(list1[i])
    eval_data = eval_data.reshape((-1, eval_data.shape[-1]))
    dataset_name = list1[i].split("/")[2]
    raw_data = np.load(f"dev/data/raw_rewards/{dataset_name}.npy")

    ax[i // 5, i % 5].scatter(raw_data[:, 0], raw_data[:, 1], c="y", alpha=0.5, edgecolor='none')
    ax[i // 5, i % 5].scatter(eval_data[:, 0], eval_data[:, 1], c="r", edgecolor='none', label='CFG')

    eval_data = np.load(list2[i])
    eval_data = eval_data.reshape((-1, eval_data.shape[-1]))
    ax[i // 5, i % 5].scatter(eval_data[:, 0], eval_data[:, 1], c="b", edgecolor='none', label='CFG+CG')
    ax[i // 5, i % 5].set_title(dataset_name.split("-")[1] + ' ' + dataset_name.split("_")[-2])

ax[0, 0].legend()
fig.savefig("compare.pdf")