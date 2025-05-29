import numpy as np
import matplotlib.pyplot as plt
from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset

# Standard

# fig, ax = plt.subplots(4, 8, figsize=(32, 16), sharex="col", sharey="col")

# set = [
#     "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/standard/eval_500000.npy",
#     "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_dd/MO-Hopper-v2_50000_expert_uniform/standard/eval_500000.npy",
#     "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_dd/MO-Swimmer-v2_50000_expert_uniform/standard/eval_500000.npy",
#     "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_dd/MO-Walker2d-v2_50000_expert_uniform/standard/eval_500000.npy",
#     "results/mo_diffuser/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_diffuser/MO-HalfCheetah-v2_50000_expert_uniform/standard/eval_500000.npy",
#     "results/mo_diffuser/MO-Hopper-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_diffuser/MO-Hopper-v2_50000_expert_uniform/standard/eval_500000.npy",
#     "results/mo_diffuser/MO-Swimmer-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_diffuser/MO-Swimmer-v2_50000_expert_uniform/standard/eval_500000.npy",
#     "results/mo_diffuser/MO-Walker2d-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_diffuser/MO-Walker2d-v2_50000_expert_uniform/standard/eval_500000.npy",
#     "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/standard/eval_500000.npy",
#     "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/standard/eval_500000.npy",
#     "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/standard/eval_500000.npy",
#     "results/mo_ddpro/MO-Walker2d-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_ddpro/MO-Walker2d-v2_50000_expert_uniform/standard/eval_500000.npy",
#     "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_wo_invdyn/eval_rewards_500000.npy",
#     "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_wo_invdyn/eval_rewards_500000.npy",
#     "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/eval_wo_invdyn/eval_rewards_500000.npy",
#     "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/eval_wo_invdyn/eval_rewards_500000.npy",
#     "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/eval_wo_invdyn/eval_rewards_500000.npy",
#     "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/eval_wo_invdyn/eval_rewards_500000.npy",
#     "results/mo_ddpro/MO-Walker2d-v2_50000_amateur_uniform/eval_wo_invdyn/eval_rewards_500000.npy",
#     "results/mo_ddpro/MO-Walker2d-v2_50000_expert_uniform/eval_wo_invdyn/eval_rewards_500000.npy",
# ]

# for i, path in enumerate(set):
#     eval_data = np.load(path)
#     dataset_name = path.split("/")[2]
#     raw_data = np.load(f"dev/data/raw_rewards/{dataset_name}.npy")

#     ax[i // 8, i % 8].scatter(raw_data[:, 0], raw_data[:, 1], c="y", alpha=0.5, edgecolor='none')
#     ax[i // 8, i % 8].scatter(eval_data[:, 0], eval_data[:, 1], c="r", edgecolor='none')

#     if i // 8 == 0:
#         ax[i // 8, i % 8].set_title(dataset_name.split("_")[0]+"_"+dataset_name.split("_")[2])

# ax[0, 0].set_ylabel("DD")
# ax[1, 0].set_ylabel("Diffuser")
# ax[2, 0].set_ylabel("DDPro")
# ax[3, 0].set_ylabel("DDPro without invdyn")
# fig.savefig("plots/standard.png")

# temperature

# fig, ax = plt.subplots(4, 4, figsize=(16, 16), sharex="row", sharey="row")

# set = [
#     "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/eval_temperature/eval_rewards_0.1.npy",
#     "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/eval_temperature/eval_rewards_0.2.npy",
#     "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/eval_temperature/eval_rewards_0.5.npy",
#     "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/eval_temperature/eval_rewards_1.0.npy",
    
#     "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/eval_temperature/eval_rewards_0.1.npy",
#     "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/eval_temperature/eval_rewards_0.2.npy",
#     "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/eval_temperature/eval_rewards_0.5.npy",
#     "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/eval_temperature/eval_rewards_1.0.npy",

#     "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_temperature/eval_rewards_0.1.npy",
#     "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_temperature/eval_rewards_0.2.npy",
#     "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_temperature/eval_rewards_0.5.npy",
#     "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_temperature/eval_rewards_1.0.npy",

#     "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_temperature/eval_rewards_0.1.npy",
#     "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_temperature/eval_rewards_0.2.npy",
#     "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_temperature/eval_rewards_0.5.npy",
#     "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_temperature/eval_rewards_1.0.npy",
# ]

# for i, path in enumerate(set):
#     eval_data = np.load(path)
#     dataset_name = path.split("/")[2]
#     raw_data = np.load(f"dev/data/raw_rewards/{dataset_name}.npy")

#     ax[i // 4, i % 4].scatter(raw_data[:, 0], raw_data[:, 1], c="y", alpha=0.5, edgecolor='none')
#     ax[i // 4, i % 4].scatter(eval_data[:, 0], eval_data[:, 1], c="r", edgecolor='none')
#     if i // 4 == 0:
#         ax[i // 4, i % 4].set_title(path.split("_")[-1][:-4])

# ax[0, 0].set_ylabel("DD")
# ax[2, 0].set_ylabel("DDPro")
# fig.savefig("plots/temperature_halfcheetah&hopper.png")

# dd on halfcheetah along training

fig, ax = plt.subplots(4, 5, figsize=(20, 16), sharex="row", sharey="row")

set = [
    "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_100000.npy",
    "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_200000.npy",
    # "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_100000.npy",
    # "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_200000.npy",
    # "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_300000.npy",
    # "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_400000.npy",
    # "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_500000.npy",
    # "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/standard/eval_100000.npy",
    # "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/standard/eval_200000.npy",
    # "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/standard/eval_300000.npy",
    # "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/standard/eval_400000.npy",
    # "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/standard/eval_500000.npy",
    # "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_100000.npy",
    # "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_200000.npy",
    # "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_300000.npy",
    # "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_400000.npy",
    # "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_500000.npy",
    # "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/standard/eval_100000.npy",
    # "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/standard/eval_200000.npy",
    # "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/standard/eval_300000.npy",
    # "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/standard/eval_400000.npy",
    # "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/standard/eval_500000.npy",
]

for i, path in enumerate(set):
    eval_data = np.load(path)
    dataset_name = path.split("/")[2]
    raw_data = np.load(f"dev/data/raw_rewards/{dataset_name}.npy")

    ax[i // 5, i % 5].scatter(raw_data[:, 0], raw_data[:, 1], c="y", alpha=0.5, edgecolor='none')
    ax[i // 5, i % 5].scatter(eval_data[:, 0], eval_data[:, 1], c="r", edgecolor='none')

    if i // 5 == 0:
        ax[i // 5, i % 5].set_title(path.split("_")[-1][:-4])

ax[0, 0].set_ylabel("DD")
ax[2, 0].set_ylabel("DDPro")
fig.savefig("plots/dd_along_training.png")

# ddpro along model size

# fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex="row", sharey="row")

# set = [
#     "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/depth12/eval_500000.npy",
#     "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/depth16/eval_500000.npy",
#     "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/standard/eval_500000.npy",
#     "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/depth12/eval_500000.npy",
#     "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/depth16/eval_500000.npy",
# ]

# for i, path in enumerate(set):
#     eval_data = np.load(path)
#     dataset_name = path.split("/")[2]
#     raw_data = np.load(f"dev/data/raw_rewards/{dataset_name}.npy")

#     ax[i // 3, i % 3].scatter(raw_data[:, 0], raw_data[:, 1], c="y", alpha=0.5, edgecolor='none')
#     ax[i // 3, i % 3].scatter(eval_data[:, 0], eval_data[:, 1], c="r", edgecolor='none')

#     # if i // 5 == 0:
#     #     ax[i // 5, i % 5].set_title(dataset_name.split("_")[0]+"_"+dataset_name.split("_")[2])

# ax[0, 0].set_title("depth8")
# ax[0, 1].set_title("depth12")
# ax[0, 2].set_title("depth16")
# fig.savefig("plots/ddpro_along_model_size.png")