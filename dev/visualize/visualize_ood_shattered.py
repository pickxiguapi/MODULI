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

dropped_prefs = np.load("dev/data/dropped_prefs/MO-Walker2d-v2_50000_expert_uniform.npy")
use_prefs = np.linspace(0, 1, 501).reshape((-1, 1))
use_prefs = np.concatenate([use_prefs, 1-use_prefs], axis=1)
use_indices = []

for i in range(1, len(use_prefs)):
    if np.abs(dropped_prefs - use_prefs[i]).min() <= 1e-3:
        use_indices.append(i)

use_indices = np.array(use_indices)

fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex='row')
plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.1, wspace=0.05, hspace=0.09)

from matplotlib import ticker
axs[0, 1].yaxis.set_major_locator(ticker.NullLocator())
axs[0, 2].yaxis.set_major_locator(ticker.NullLocator())
axs[1, 1].yaxis.set_major_locator(ticker.NullLocator())
axs[1, 2].yaxis.set_major_locator(ticker.NullLocator())


d4morl_data = np.load(f"dev/data/raw_rewards_30/MO-Walker2d-v2_50000_expert_uniform.npy")
axs[0, 0].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
axs[0, 1].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
axs[0, 2].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')

slider_data = np.load("results/mo_dd_shattered/MO-Walker2d-v2_50000_expert_uniform/eval_standard/11M_30_controlnet_linear/rewards/200k.npy")
slider_data = slider_data.mean(axis=0)
undominated_indice = undominated_indices(slider_data, tolerance=0.05)
dominated_indice = np.setdiff1d(np.arange(len(slider_data)), undominated_indice)

axs[0, 0].scatter(slider_data[use_indices, 0], slider_data[use_indices, 1], edgecolor='royalblue', facecolor='none')
# axs[0, 0].scatter(slider_data[:, 0], slider_data[:, 1], edgecolor='royalblue', facecolor='none')

import pickle
with open("experiment_runs/side_30/MO-Walker2d-v2/expert_uniform/rvs/K=20/mo_rtg=True/rtg_scale=100/norm_rew=False/concat_state_pref=1/concat_rtg_pref=0/concat_act_pref=0/percent=1/batch=64/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam/seed=1/logs/step=200000_rollout.pkl", "rb") as f:
    rvs_data = pickle.load(f)

rvs_data = np.array(rvs_data["rewards"]).mean(1)
axs[0, 1].scatter(rvs_data[use_indices, 0], rvs_data[use_indices, 1], edgecolor='royalblue', facecolor='none')
# axs[0, 1].scatter(rvs_data[:, 0], rvs_data[:, 1], edgecolor='royalblue', facecolor='none')

with open("experiment_runs/side_30/MO-Walker2d-v2/expert_uniform/bc/K=20/mo_rtg=True/rtg_scale=100/norm_rew=False/concat_state_pref=1/concat_rtg_pref=0/concat_act_pref=0/percent=1/batch=64/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam/seed=1/logs/step=200000_rollout.pkl", "rb") as f:
    bc_data = pickle.load(f)

bc_data = np.array(bc_data["rewards"]).mean(1)
axs[0, 2].scatter(bc_data[use_indices, 0], bc_data[use_indices, 1], edgecolor='royalblue', facecolor='none')
# axs[0, 2].scatter(bc_data[:, 0], bc_data[:, 1], edgecolor='royalblue', facecolor='none')

axs[0, 0].set_title("MODULI", fontsize=20)
axs[0, 1].set_title("MORvS(P)", fontsize=20)
axs[0, 2].set_title("BC(P)", fontsize=20)

axs[0, 0].set_ylabel("Objective 2", fontsize=18)
axs[1, 0].set_ylabel("Objective 2", fontsize=18)
axs[1, 0].set_xlabel("Objective 1", fontsize=18)
axs[1, 1].set_xlabel("Objective 1", fontsize=18)
axs[1, 2].set_xlabel("Objective 1", fontsize=18)


# ----------------------------

dropped_prefs = np.load("dev/data/dropped_prefs_S/MO-Hopper-v2_50000_amateur_uniform.npy")
use_prefs = np.linspace(0, 1, 501).reshape((-1, 1))
use_prefs = np.concatenate([use_prefs, 1-use_prefs], axis=1)
use_indices = []

for i in range(1, len(use_prefs)):
    if np.abs(dropped_prefs - use_prefs[i]).min() <= 1e-3:
        use_indices.append(i)

use_indices = np.array(use_indices)

d4morl_data = np.load(f"dev/data/raw_rewards_S30/MO-Hopper-v2_50000_amateur_uniform.npy")
axs[1, 0].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
axs[1, 1].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')
axs[1, 2].scatter(d4morl_data[::100, 0], d4morl_data[::100, 1], edgecolor='lightgrey', facecolor='none')

slider_data = np.load("results/mo_dd_shattered/MO-Hopper-v2_50000_amateur_uniform/eval_standard/11M_S30_controlnet/side_uniform/rewards/slider.npy")
slider_data = slider_data.mean(axis=0)
undominated_indice = undominated_indices(slider_data, tolerance=0.05)
dominated_indice = np.setdiff1d(np.arange(len(slider_data)), undominated_indice)

axs[1, 0].scatter(slider_data[use_indices, 0], slider_data[use_indices, 1], edgecolor='royalblue', facecolor='none')

import pickle
with open("experiment_runs/side_30/MO-Hopper-v2/amateur_uniform/rvs/K=20/mo_rtg=True/rtg_scale=100/norm_rew=False/concat_state_pref=1/concat_rtg_pref=0/concat_act_pref=0/percent=1/batch=64/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam/seed=1/logs/step=200000_rollout.pkl", "rb") as f:
    rvs_data = pickle.load(f)

rvs_data = np.array(rvs_data["rewards"]).mean(1)
axs[1, 1].scatter(rvs_data[use_indices, 0], rvs_data[use_indices, 1], edgecolor='royalblue', facecolor='none')

with open("experiment_runs/side_30/MO-Hopper-v2/amateur_uniform/bc/K=20/mo_rtg=True/rtg_scale=100/norm_rew=False/concat_state_pref=1/concat_rtg_pref=0/concat_act_pref=0/percent=1/batch=64/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam/seed=1/logs/step=200000_rollout.pkl", "rb") as f:
    bc_data = pickle.load(f)

bc_data = np.array(bc_data["rewards"]).mean(1)
axs[1, 2].scatter(bc_data[use_indices, 0], bc_data[use_indices, 1], color='royalblue', facecolor='none')

fig.savefig("plots/visual_ood_2x3.pdf")
