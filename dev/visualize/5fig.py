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

dataset_name = "Walker2d"
dataset_type = "expert"

fig, axs = plt.subplots(1, 5, figsize=(22, 4), sharex=True, sharey=True)

# ------------- Complete
data = np.load(f"dev/data/raw_rewards/MO-{dataset_name}-v2_50000_{dataset_type}_uniform.npy")
axs[0].scatter(data[::100, 0], data[::100, 1], edgecolor='royalblue', facecolor='none')

# ------------- Shattered
data = np.load(f"dev/data/raw_rewards_30/MO-{dataset_name}-v2_50000_{dataset_type}_uniform.npy")
axs[1].scatter(data[::100, 0], data[::100, 1], edgecolor='royalblue', facecolor='none')

# ------------- Narrow
data = np.load(f"dev/data/raw_rewards_S30/MO-{dataset_name}-v2_50000_{dataset_type}_uniform.npy")
axs[2].scatter(data[::100, 0], data[::100, 1], edgecolor='royalblue', facecolor='none')

# ------------- RvS
data = np.load(f"dev/data/raw_rewards_30/MO-{dataset_name}-v2_50000_{dataset_type}_uniform.npy")
axs[3].scatter(data[::100, 0], data[::100, 1], facecolor='lightgrey', lw=0.2)

with open(f"experiment_runs/shattered_30/MO-{dataset_name}-v2/{dataset_type}_uniform/rvs/K=20/mo_rtg=True/rtg_scale=100/norm_rew=False/concat_state_pref=1/concat_rtg_pref=0/concat_act_pref=0/percent=1/batch=64/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam/seed=1/logs/step=200000_rollout.pkl", "rb") as f:
    rvs_data = pickle.load(f)

rvs_data = np.array(rvs_data["rewards"])
# average over episodes
rvs_data = rvs_data.mean(axis=1)
rvs_data = rvs_data[undominated_indices(rvs_data)]

axs[3].scatter(rvs_data[:, 0], rvs_data[:, 1], edgecolor='royalblue', facecolor='none')

# -------------- Diff
data = np.load(f"dev/data/raw_rewards_30/MO-{dataset_name}-v2_50000_{dataset_type}_uniform.npy")
axs[4].scatter(data[::20, 0], data[::20, 1], facecolor='grey', lw=0.2)

diff_data = np.load(f"results/mo_dd_shattered/MO-{dataset_name}-v2_50000_{dataset_type}_uniform/eval_standard/11M_30_controlnet/shatter_uniform/rewards/slider.npy")
diff_data = diff_data.mean(axis=0)
diff_data = diff_data[undominated_indices(diff_data)]

axs[4].scatter(diff_data[:, 0], diff_data[:, 1], edgecolor='royalblue', facecolor='none')

fig.savefig("plots/5fig.pdf")

fig, axs = plt.subplots(10, 3, sharex="row", sharey="row", figsize=(12, 40))
cnt = 0
for dataset_type in ["expert", "amateur"]:
    for dataset_name in ["Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]:
        data = np.load(f"results/mo_dd_shattered/MO-{dataset_name}-v2_50000_{dataset_type}_uniform/eval_standard/11M_30/shatter_uniform/rewards/non_slider.npy")
        data = data.mean(axis=0)
        data = data[undominated_indices(data)]
        axs[cnt, 0].scatter(data[:, 0], data[:, 1], edgecolor='royalblue', facecolor='none')

        with open(f"experiment_runs/shattered_30/MO-{dataset_name}-v2/{dataset_type}_uniform/rvs/K=20/mo_rtg=True/rtg_scale=100/norm_rew=False/concat_state_pref=1/concat_rtg_pref=0/concat_act_pref=0/percent=1/batch=64/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam/seed=1/logs/step=200000_rollout.pkl", "rb") as f:
            rvs_data = pickle.load(f)

        rvs_data = np.array(rvs_data["rewards"])
        # average over episodes
        rvs_data = rvs_data.mean(axis=1)
        rvs_data = rvs_data[undominated_indices(rvs_data)]

        axs[cnt, 1].scatter(rvs_data[:, 0], rvs_data[:, 1], edgecolor='royalblue', facecolor='none')

        # with open(f"experiment_runs/shattered_30/MO-{dataset_name}-v2/{dataset_type}_uniform/bc/K=20/mo_rtg=True/rtg_scale=100/norm_rew=False/concat_state_pref=1/concat_rtg_pref=0/concat_act_pref=0/percent=1/batch=64/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam/seed=1/logs/step=200000_rollout.pkl", "rb") as f:
        #     rvs_data = pickle.load(f)

        # rvs_data = np.array(rvs_data["rewards"])
        # # average over episodes
        # rvs_data = rvs_data.mean(axis=1)
        # rvs_data = rvs_data[undominated_indices(rvs_data)]

        # axs[cnt, 2].scatter(rvs_data[:, 0], rvs_data[:, 1], edgecolor='royalblue', facecolor='none')
        axs[cnt, 0].set_title(f"{dataset_name} {dataset_type}")
        cnt += 1

fig.savefig("plots/all_env.pdf")

fig, axs = plt.subplots(1, 2, sharex="row", sharey="row", figsize=(8, 4))
data = np.load(f"dev/data/raw_rewards/MO-Walker2d-v2_50000_amateur_uniform.npy")
axs[0].scatter(data[:, 0], data[:, 1], facecolor='lightgrey', lw=0.2)
axs[1].scatter(data[:, 0], data[:, 1], facecolor='lightgrey', lw=0.2)

data = np.load(f"results/mo_dd_shattered/MO-Walker2d-v2_50000_amateur_uniform/eval_standard/11M_30/shatter_uniform/rewards/non_slider.npy")
data = data.mean(axis=0)
data = data[undominated_indices(data)]
axs[0].scatter(data[:, 0], data[:, 1], edgecolor='royalblue', facecolor='none')

with open(f"experiment_runs/shattered_30/MO-Walker2d-v2/amateur_uniform/rvs/K=20/mo_rtg=True/rtg_scale=100/norm_rew=False/concat_state_pref=1/concat_rtg_pref=0/concat_act_pref=0/percent=1/batch=64/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam/seed=1/logs/step=200000_rollout.pkl", "rb") as f:
    rvs_data = pickle.load(f)

rvs_data = np.array(rvs_data["rewards"])
# average over episodes
rvs_data = rvs_data.mean(axis=1)
rvs_data = rvs_data[undominated_indices(rvs_data)]

axs[1].scatter(rvs_data[:, 0], rvs_data[:, 1], edgecolor='royalblue', facecolor='none')

fig.savefig("plots/Walker2d.pdf")