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

dataset_name = "Hopper"
dataset_type = "amateur"

fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)

# ------------- Complete
data = np.load(f"dev/data/raw_rewards/MO-{dataset_name}-v2_50000_{dataset_type}_uniform.npy")
axs[0].scatter(data[::100, 0], data[::100, 1], edgecolor='grey', facecolor='none')
axs[0].set_title("Complete Dataset", fontsize=15)
axs[0].set_xlabel("Objective 1", fontsize=13)
axs[0].set_ylabel("Objective 2", fontsize=13)

# ------------- Shattered
data = np.load(f"dev/data/raw_rewards_50_2/MO-{dataset_name}-v2_50000_{dataset_type}_uniform.npy")
axs[1].scatter(data[::100, 0], data[::100, 1], edgecolor='grey', facecolor='none')
axs[1].set_title("Shattered Dataset", fontsize=15)
axs[1].set_xlabel("Objective 1", fontsize=13)
axs[1].set_ylabel("Objective 2", fontsize=13)

# ------------- Narrow
data = np.load(f"dev/data/raw_rewards_S50/MO-{dataset_name}-v2_50000_{dataset_type}_uniform.npy")
axs[2].scatter(data[::100, 0], data[::100, 1], edgecolor='grey', facecolor='none')
axs[2].set_title("Narrow Dataset", fontsize=15)
axs[2].set_xlabel("Objective 1", fontsize=13)
axs[2].set_ylabel("Objective 2", fontsize=13)

# ------------- RvS
data = np.load(f"dev/data/raw_rewards_50_2/MO-{dataset_name}-v2_50000_{dataset_type}_uniform.npy")
axs[3].scatter(data[::100, 0], data[::100, 1], edgecolor='lightgrey', facecolor='none')
axs[3].set_title("Approximate Pareto Front", fontsize=15)
axs[3].set_xlabel("Objective 1", fontsize=13)
axs[3].set_ylabel("Objective 2", fontsize=13)

with open(f"experiment_runs/shattered_30/MO-{dataset_name}-v2/{dataset_type}_uniform/rvs/K=20/mo_rtg=True/rtg_scale=100/norm_rew=False/concat_state_pref=1/concat_rtg_pref=0/concat_act_pref=0/percent=1/batch=64/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam/seed=1/logs/step=200000_rollout.pkl", "rb") as f:
    rvs_data = pickle.load(f)

rvs_data = np.array(rvs_data["rewards"])
# average over episodes
rvs_data = rvs_data.mean(axis=1)
rvs_data = rvs_data[undominated_indices(rvs_data)]

# -------------- Diff
diff_data = np.load(f"results/mo_dd_shattered/MO-{dataset_name}-v2_50000_{dataset_type}_uniform/eval_standard/11M_30_controlnet/shatter_uniform/rewards/slider.npy")
diff_data = diff_data.mean(axis=0)
diff_data = diff_data[undominated_indices(diff_data)]

axs[3].scatter(diff_data[:, 0], diff_data[:, 1], edgecolor='crimson', facecolor='none', label='MODULI')
axs[3].scatter(rvs_data[:, 0], rvs_data[:, 1], edgecolor='royalblue', facecolor='none', label='MORvS')

# Remove tick labels
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

# Add legend to the fourth subplot
axs[3].legend()

fig.tight_layout()
fig.savefig("plots/motivation.pdf")