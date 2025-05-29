import matplotlib.pyplot as plt
import numpy as np
import pickle

def check_dominated(obj_batch, obj, tolerance=0):
    return (np.logical_and((obj_batch * (1-tolerance) >= obj).all(axis=1), (obj_batch * (1-tolerance) > obj).any(axis=1))).any()

# return sorted indices of nondominated objs
def undominated_indices(obj_batch_input, tolerance=0):
    obj_batch = np.array(obj_batch_input)
    sorted_indices = np.argsort(obj_batch.T[0])
    indices = []
    for idx in sorted_indices:
        if (obj_batch[idx] >= 0).all() and not check_dominated(obj_batch, obj_batch[idx], tolerance):
            indices.append(idx)
    return indices



dataset_names = ["Ant", "HalfCheetah", "Swimmer", "Hopper", "Walker2d"]

fig, axs = plt.subplots(6, 5, figsize=(20, 24), sharex='col', sharey='col')
plt.subplots_adjust(left=0.06, right=0.98, top=0.97, bottom=0.02, wspace=0.15, hspace=0.08)

# for i in range(6):
#     for j in range(1, 5):
#         axs[i, j].set_yticks([])

# for i in range(1, 5):
#     for j in range(5):
#         axs[i, j].set_yticks([])

for idx, dataset_name in enumerate(dataset_names):
    d4morl_data = np.load(f"dev/data/raw_rewards/MO-{dataset_name}-v2_50000_amateur_uniform.npy")
    axs[0, idx].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
    diff_data = np.load(f"results/mo_dd/MO-{dataset_name}-v2_50000_amateur_uniform/eval_standard/11M/rewards/500k.npy")
    diff_data = diff_data.mean(axis=0)
    dominated_indices = np.setdiff1d(np.arange(len(diff_data)), undominated_indices(diff_data, tolerance=0.05))
    dominated_data = diff_data[dominated_indices]
    axs[0, idx].scatter(dominated_data[:, 0], dominated_data[:, 1], edgecolor='royalblue', facecolor='none')
    front_data = diff_data[undominated_indices(diff_data, tolerance=0.05)]
    axs[0, idx].scatter(front_data[:, 0], front_data[:, 1], edgecolor='crimson', facecolor='none')
    axs[0, idx].set_title(dataset_name, fontsize=24)
    # axs[0, idx].set_xlabel("Objective 1", fontsize=15)

for idx, dataset_name in enumerate(dataset_names):
    d4morl_data = np.load(f"dev/data/raw_rewards/MO-{dataset_name}-v2_50000_expert_uniform.npy")
    axs[1, idx].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
    diff_data = np.load(f"results/mo_dd/MO-{dataset_name}-v2_50000_expert_uniform/eval_standard/11M/rewards/500k.npy")
    diff_data = diff_data.mean(axis=0)
    dominated_indices = np.setdiff1d(np.arange(len(diff_data)), undominated_indices(diff_data, tolerance=0.05))
    dominated_data = diff_data[dominated_indices]
    axs[1, idx].scatter(dominated_data[:, 0], dominated_data[:, 1], edgecolor='royalblue', facecolor='none')
    front_data = diff_data[undominated_indices(diff_data, tolerance=0.05)]
    axs[1, idx].scatter(front_data[:, 0], front_data[:, 1], edgecolor='crimson', facecolor='none')

for idx, dataset_name in enumerate(dataset_names):
    d4morl_data = np.load(f"dev/data/raw_rewards_30/MO-{dataset_name}-v2_50000_amateur_uniform.npy")
    axs[2, idx].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
    diff_data = np.load(f"results/mo_dd_shattered/MO-{dataset_name}-v2_50000_amateur_uniform/eval_standard/11M_30_controlnet_linear/rewards/200k.npy")
    diff_data = diff_data.mean(axis=0)
    dominated_indices = np.setdiff1d(np.arange(len(diff_data)), undominated_indices(diff_data, tolerance=0.05))
    dominated_data = diff_data[dominated_indices]
    axs[2, idx].scatter(dominated_data[:, 0], dominated_data[:, 1], edgecolor='royalblue', facecolor='none')
    front_data = diff_data[undominated_indices(diff_data, tolerance=0.05)]
    axs[2, idx].scatter(front_data[:, 0], front_data[:, 1], edgecolor='crimson', facecolor='none')

for idx, dataset_name in enumerate(dataset_names):
    d4morl_data = np.load(f"dev/data/raw_rewards_30/MO-{dataset_name}-v2_50000_expert_uniform.npy")
    axs[3, idx].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
    diff_data = np.load(f"results/mo_dd_shattered/MO-{dataset_name}-v2_50000_expert_uniform/eval_standard/11M_30_controlnet_linear/rewards/200k.npy")
    diff_data = diff_data.mean(axis=0)
    dominated_indices = np.setdiff1d(np.arange(len(diff_data)), undominated_indices(diff_data, tolerance=0.05))
    dominated_data = diff_data[dominated_indices]
    axs[3, idx].scatter(dominated_data[:, 0], dominated_data[:, 1], edgecolor='royalblue', facecolor='none')
    front_data = diff_data[undominated_indices(diff_data, tolerance=0.05)]
    axs[3, idx].scatter(front_data[:, 0], front_data[:, 1], edgecolor='crimson', facecolor='none')

for idx, dataset_name in enumerate(dataset_names):
    d4morl_data = np.load(f"dev/data/raw_rewards_S30/MO-{dataset_name}-v2_50000_amateur_uniform.npy")
    axs[4, idx].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
    diff_data = np.load(f"results/mo_dd_shattered/MO-{dataset_name}-v2_50000_amateur_uniform/eval_standard/11M_S30_controlnet_linear/rewards/200k.npy")
    diff_data = diff_data.mean(axis=0)
    dominated_indices = np.setdiff1d(np.arange(len(diff_data)), undominated_indices(diff_data, tolerance=0.05))
    dominated_data = diff_data[dominated_indices]
    axs[4, idx].scatter(dominated_data[:, 0], dominated_data[:, 1], edgecolor='royalblue', facecolor='none')
    front_data = diff_data[undominated_indices(diff_data, tolerance=0.05)]
    axs[4, idx].scatter(front_data[:, 0], front_data[:, 1], edgecolor='crimson', facecolor='none')

axs[4, 3].clear()
for idx, dataset_name in enumerate(["Hopper"]):
    d4morl_data = np.load(f"dev/data/raw_rewards_S30/MO-{dataset_name}-v2_50000_amateur_uniform.npy")
    axs[4, 3].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
    diff_data = np.load(f"results/mo_dd_shattered/MO-{dataset_name}-v2_50000_amateur_uniform/eval_standard/11M_S30_controlnet/side_uniform/rewards/slider.npy")
    diff_data = diff_data.mean(axis=0)
    dominated_indices = np.setdiff1d(np.arange(len(diff_data)), undominated_indices(diff_data, tolerance=0.05))
    dominated_data = diff_data[dominated_indices]
    axs[4, 3].scatter(dominated_data[:, 0], dominated_data[:, 1], edgecolor='royalblue', facecolor='none')
    front_data = diff_data[undominated_indices(diff_data, tolerance=0.05)]
    axs[4, 3].scatter(front_data[:, 0], front_data[:, 1], edgecolor='crimson', facecolor='none')

for idx, dataset_name in enumerate(dataset_names):
    d4morl_data = np.load(f"dev/data/raw_rewards_S30/MO-{dataset_name}-v2_50000_expert_uniform.npy")
    axs[5, idx].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
    diff_data = np.load(f"results/mo_dd_shattered/MO-{dataset_name}-v2_50000_expert_uniform/eval_standard/11M_S30_controlnet_linear/rewards/200k.npy")
    diff_data = diff_data.mean(axis=0)
    dominated_indices = np.setdiff1d(np.arange(len(diff_data)), undominated_indices(diff_data, tolerance=0.05))
    dominated_data = diff_data[dominated_indices]
    axs[5, idx].scatter(dominated_data[:, 0], dominated_data[:, 1], edgecolor='royalblue', facecolor='none')
    front_data = diff_data[undominated_indices(diff_data, tolerance=0.05)]
    axs[5, idx].scatter(front_data[:, 0], front_data[:, 1], edgecolor='crimson', facecolor='none')

for idx, dataset_name in enumerate(["Amateur-Complete", "Expert-Complete", "Amateur-Shattered", "Expert-Shattered", "Amateur-Narrow", "Expert-Narrow"]):
    axs[idx, 0].set_ylabel(dataset_name, fontsize=24)

fig.savefig("plots/full_result.pdf")
