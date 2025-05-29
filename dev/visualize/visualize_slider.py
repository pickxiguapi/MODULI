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


dataset_names = ["Hopper", "Swimmer"]

fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
plt.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.15)
axs[1].set_yticks([])

d4morl_data = np.load(f"dev/data/raw_rewards_S30/MO-Hopper-v2_50000_amateur_uniform.npy")
axs[0].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
axs[1].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')

non_slider_data = np.load("results/mo_dd_shattered/MO-Hopper-v2_50000_amateur_uniform/eval_standard/11M_S30/side_uniform/rewards/non_slider.npy")
non_slider_data = non_slider_data.mean(axis=0)
undominated_indice = undominated_indices(non_slider_data, tolerance=0.05)
dominated_indice = np.setdiff1d(np.arange(len(non_slider_data)), undominated_indice)

axs[0].scatter(non_slider_data[dominated_indice, 0], non_slider_data[dominated_indice, 1], edgecolor='royalblue', facecolor='none')
axs[0].scatter(non_slider_data[undominated_indice, 0], non_slider_data[undominated_indice, 1], edgecolor='crimson', facecolor='none')

slider_data = np.load("results/mo_dd_shattered/MO-Hopper-v2_50000_amateur_uniform/eval_standard/11M_S30_controlnet/side_uniform/rewards/slider.npy")
slider_data = slider_data.mean(axis=0)
undominated_indice = undominated_indices(slider_data, tolerance=0.05)
dominated_indice = np.setdiff1d(np.arange(len(slider_data)), undominated_indice)

axs[1].scatter(slider_data[dominated_indice, 0], slider_data[dominated_indice, 1], edgecolor='royalblue', facecolor='none')
axs[1].scatter(slider_data[undominated_indice, 0], slider_data[undominated_indice, 1], edgecolor='crimson', facecolor='none')

axs[0].set_title("w/o Slider", fontsize=20)
axs[1].set_title("with Slider", fontsize=20)
axs[0].set_xlabel("Objective 1", fontsize=15)
axs[0].set_ylabel("Objective 2", fontsize=15)
axs[1].set_xlabel("Objective 1", fontsize=15)

axs[0].set_xlim(2100, 4200)
axs[0].set_ylim(2000, 5400)
axs[1].set_ylim(2000, 5400)
# set fig title:
fig.suptitle("Hopper-amateur-Narrow", fontsize=20)



# for idx, dataset_name in enumerate(dataset_names):
#     d4morl_data = np.load(f"dev/data/raw_rewards/MO-{dataset_name}-v2_50000_amateur_uniform.npy")
#     axs[0, idx].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
#     diff_data = np.load(f"results/mo_dd/MO-{dataset_name}-v2_50000_amateur_uniform/eval_standard/11M/rewards/500k.npy")
#     diff_data = diff_data.mean(axis=0)
#     dominated_indices = np.setdiff1d(np.arange(len(diff_data)), undominated_indices(diff_data, tolerance=0.05))
#     dominated_data = diff_data[dominated_indices]
#     axs[0, idx].scatter(dominated_data[:, 0], dominated_data[:, 1], edgecolor='royalblue', facecolor='none')
#     front_data = diff_data[undominated_indices(diff_data, tolerance=0.05)]
#     axs[0, idx].scatter(front_data[:, 0], front_data[:, 1], edgecolor='crimson', facecolor='none')
#     axs[0, idx].set_title(dataset_name+"-amateur", fontsize=20)
#     # axs[0, idx].set_xlabel("Objective 1", fontsize=15)
#     if idx == 0:
#         axs[0, idx].set_ylabel("Objective 2", fontsize=15)

# for idx, dataset_name in enumerate(dataset_names):
#     d4morl_data = np.load(f"dev/data/raw_rewards/MO-{dataset_name}-v2_50000_expert_uniform.npy")
#     axs[1, idx].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
#     diff_data = np.load(f"results/mo_dd/MO-{dataset_name}-v2_50000_expert_uniform/eval_standard/11M/rewards/500k.npy")
#     diff_data = diff_data.mean(axis=0)
#     dominated_indices = np.setdiff1d(np.arange(len(diff_data)), undominated_indices(diff_data, tolerance=0.05))
#     dominated_data = diff_data[dominated_indices]
#     axs[1, idx].scatter(dominated_data[:, 0], dominated_data[:, 1], edgecolor='royalblue', facecolor='none')
#     front_data = diff_data[undominated_indices(diff_data, tolerance=0.05)]
#     axs[1, idx].scatter(front_data[:, 0], front_data[:, 1], edgecolor='crimson', facecolor='none')
#     axs[1, idx].set_title(dataset_name+"-expert", fontsize=20)
#     axs[1, idx].set_xlabel("Objective 1", fontsize=15)
#     if idx == 0:
#         axs[1, idx].set_ylabel("Objective 2", fontsize=15)

# d4morl_data = np.load(f"dev/data/raw_rewards/MO-Hopper-v3_50000_expert_uniform.npy")
# # delete 6-th ax
# fig.delaxes(fig.axes[5])
# # add a 3d ax
# ax = fig.add_subplot(166, projection='3d')
# ax.scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], d4morl_data[::100, 2], edgecolor='lightgrey', facecolor='none')
# diff_data = np.load(f"results/mo_dd/MO-Hopper-v3_50000_expert_uniform/eval_standard/22M/eval_reward_.npy")
# diff_data = diff_data.mean(axis=0)
# dominated_indices = np.setdiff1d(np.arange(len(diff_data)), undominated_indices(diff_data))
# dominated_data = diff_data[dominated_indices]
# ax.scatter(dominated_data[:, 0], dominated_data[:, 1], dominated_data[:, 2], edgecolor='royalblue', facecolor='none')
# front_data = diff_data[undominated_indices(diff_data)]
# ax.scatter(front_data[:, 0], front_data[:, 1], front_data[:, 2], edgecolor='crimson', facecolor='none')
# ax.set_title("Hopper-3obj-expert", fontsize=20)
# ax.set_xlabel("Objective 1", fontsize=15)
# ax.set_ylabel("Objective 2", fontsize=15)
# ax.set_zlabel("Objective 3", fontsize=15)
fig.savefig("plots/visual_slider.pdf")
