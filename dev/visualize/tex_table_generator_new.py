import numpy as np
import os
from dev.utils.utils import hv
from copy import deepcopy

from dev.PEDA.modt.hypervolume import InnerHyperVolume

def compute_hypervolume(ep_objs_batch):
    n = len(ep_objs_batch[0])
    HV = InnerHyperVolume(np.zeros(n))
    return HV.compute(ep_objs_batch)

def compute_sparsity(ep_objs_batch):
    if len(ep_objs_batch) < 2:
        return 0.0

    sparsity = 0.0
    m = len(ep_objs_batch[0])
    ep_objs_batch_np = np.array(ep_objs_batch)
    for dim in range(m):
        objs_i = np.sort(deepcopy(ep_objs_batch_np.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity += np.square(objs_i[i] - objs_i[i - 1])
    sparsity /= (len(ep_objs_batch) - 1)
    return sparsity

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

scale_hv = {
    "MO-Ant-v2_50000_amateur_uniform":1e6,
    "MO-Ant-v2_50000_expert_uniform":1e6,
    "MO-HalfCheetah-v2_50000_amateur_uniform":1e6,
    "MO-HalfCheetah-v2_50000_expert_uniform":1e6,
    "MO-Hopper-v2_50000_amateur_uniform":1e7,
    "MO-Hopper-v2_50000_expert_uniform":1e7,
    "MO-Hopper-v3_50000_amateur_uniform":1e10,
    "MO-Hopper-v3_50000_expert_uniform":1e10,
    "MO-Swimmer-v2_50000_amateur_uniform":1e4,
    "MO-Swimmer-v2_50000_expert_uniform":1e4,
    "MO-Walker2d-v2_50000_amateur_uniform":1e6,
    "MO-Walker2d-v2_50000_expert_uniform":1e6,
}

scale_sp = {
    "MO-Ant-v2_50000_amateur_uniform":1e4,
    "MO-Ant-v2_50000_expert_uniform":1e4,
    "MO-HalfCheetah-v2_50000_amateur_uniform":1e4,
    "MO-HalfCheetah-v2_50000_expert_uniform":1e4,
    "MO-Hopper-v2_50000_amateur_uniform":1e5,
    "MO-Hopper-v2_50000_expert_uniform":1e5,
    "MO-Hopper-v3_50000_amateur_uniform":1e5,
    "MO-Hopper-v3_50000_expert_uniform":1e5,
    "MO-Swimmer-v2_50000_amateur_uniform":1,
    "MO-Swimmer-v2_50000_expert_uniform":1,
    "MO-Walker2d-v2_50000_amateur_uniform":1e4,
    "MO-Walker2d-v2_50000_expert_uniform":1e4,
}

baseline_table_hv = {
    "MO-Ant-v2_50000_amateur_uniform":        {"B":[5.61, -1], "MODT(P)":[5.92, 0.04], "MORvS(P)":[6.07, 0.02], "BC(P)":[4.37, 0.06], "CQL(P)":[5.62, 0.23], "MODT":[4.88, 0.60], "MORvS":[4.37, 0.56], "BC":[2.34, 0.15], "CQL":[2.80, 0.68]},
    "MO-HalfCheetah-v2_50000_amateur_uniform":{"B":[5.68, -1], "MODT(P)":[5.69, 0.01], "MORvS(P)":[5.77, 0.00], "BC(P)":[5.46, 0.02], "CQL(P)":[5.54, 0.02], "MODT":[5.51, 0.01], "MORvS":[4.66, 0.05], "BC":[2.92, 0.38], "CQL":[4.41, 0.08]},
    "MO-Hopper-v2_50000_amateur_uniform":     {"B":[1.97, -1], "MODT(P)":[1.81, 0.05], "MORvS(P)":[1.76, 0.03], "BC(P)":[1.35, 0.03], "CQL(P)":[1.64, 0.01], "MODT":[1.54, 0.08], "MORvS":[1.57, 0.01], "BC":[0.01, 0.01], "CQL":[0.00, 0.06]},
    "MO-Hopper-v3_50000_amateur_uniform":     {"B":[3.09, -1], "MODT(P)":[1.04, 0.16], "MORvS(P)":[2.77, 0.24], "BC(P)":[2.42, 0.18], "CQL(P)":[0.59, 0.42], "MODT":[1.61, 0.23], "MORvS":[1.30, 0.22], "BC":[0.03, 0.01], "CQL":[0.10, 0.16]},
    "MO-Swimmer-v2_50000_amateur_uniform":    {"B":[2.11, -1], "MODT(P)":[1.67, 0.22], "MORvS(P)":[2.79, 0.03], "BC(P)":[2.82, 0.04], "CQL(P)":[1.69, 0.93], "MODT":[0.96, 0.19], "MORvS":[2.93, 0.03], "BC":[0.46, 0.15], "CQL":[0.74, 0.47]},
    "MO-Walker2d-v2_50000_amateur_uniform":   {"B":[4.99, -1], "MODT(P)":[3.10, 0.34], "MORvS(P)":[4.98, 0.01], "BC(P)":[3.42, 0.42], "CQL(P)":[1.78, 0.33], "MODT":[3.76, 0.34], "MORvS":[4.32, 0.05], "BC":[0.91, 0.36], "CQL":[0.76, 0.81]},
    "MO-Ant-v2_50000_expert_uniform":         {"B":[6.32, -1], "MODT(P)":[6.21, 0.01], "MORvS(P)":[6.41, 0.01], "BC(P)":[4.88, 0.17], "CQL(P)":[5.76, 0.10], "MODT":[5.52, 0.16], "MORvS":[5.52, 0.02], "BC":[0.84, 0.60], "CQL":[3.52, 0.45]},
    "MO-HalfCheetah-v2_50000_expert_uniform": {"B":[5.79, -1], "MODT(P)":[5.73, 0.00], "MORvS(P)":[5.78, 0.00], "BC(P)":[5.54, 0.05], "CQL(P)":[5.63, 0.04], "MODT":[5.59, 0.03], "MORvS":[4.19, 0.74], "BC":[1.53, 0.09], "CQL":[3.78, 0.46]},
    "MO-Hopper-v2_50000_expert_uniform":      {"B":[2.09, -1], "MODT(P)":[2.00, 0.02], "MORvS(P)":[2.02, 0.02], "BC(P)":[1.23, 0.10], "CQL(P)":[0.33, 0.39], "MODT":[1.68, 0.03], "MORvS":[1.73, 0.07], "BC":[0.28, 0.21], "CQL":[0.02, 0.02]},
    "MO-Hopper-v3_50000_expert_uniform":      {"B":[3.73, -1], "MODT(P)":[3.38, 0.05], "MORvS(P)":[3.42, 0.10], "BC(P)":[2.29, 0.07], "CQL(P)":[0.78, 0.24], "MODT":[1.05, 0.43], "MORvS":[2.53, 0.06], "BC":[0.06, 0.02], "CQL":[0.00, 0.00]},
    "MO-Swimmer-v2_50000_expert_uniform":     {"B":[3.25, -1], "MODT(P)":[3.15, 0.02], "MORvS(P)":[3.24, 0.00], "BC(P)":[3.21, 0.00], "CQL(P)":[3.22, 0.08], "MODT":[2.49, 0.19], "MORvS":[3.19, 0.01], "BC":[1.68, 0.38], "CQL":[2.08, 0.08]},
    "MO-Walker2d-v2_50000_expert_uniform":    {"B":[5.21, -1], "MODT(P)":[4.89, 0.05], "MORvS(P)":[5.14, 0.01], "BC(P)":[3.74, 0.11], "CQL(P)":[3.21, 0.32], "MODT":[0.65, 0.46], "MORvS":[5.10, 0.02], "BC":[0.07, 0.02], "CQL":[0.82, 0.62]},
}

baseline_table_sp = {
    "MO-Ant-v2_50000_amateur_uniform":        {"MODT(P)":[8.72, 0.77],  "MORvS(P)":[5.24, 0.52], "BC(P)":[25.9, 16.4], "CQL(P)":[1.06, 0.28]},
    "MO-HalfCheetah-v2_50000_amateur_uniform":{"MODT(P)":[1.16, 0.42],  "MORvS(P)":[0.57, 0.09], "BC(P)":[2.22, 0.91], "CQL(P)":[0.45, 0.27]},
    "MO-Hopper-v2_50000_amateur_uniform":     {"MODT(P)":[1.61, 0.29],  "MORvS(P)":[3.50, 1.54], "BC(P)":[2.42, 1.08], "CQL(P)":[3.30, 5.25]},
    "MO-Hopper-v3_50000_amateur_uniform":     {"MODT(P)":[10.23, 2.78], "MORvS(P)":[1.03, 0.11], "BC(P)":[0.87, 0.29], "CQL(P)":[2.00, 1.72]},
    "MO-Swimmer-v2_50000_amateur_uniform":    {"MODT(P)":[2.87, 1.32],  "MORvS(P)":[1.03, 0.20], "BC(P)":[5.05, 1.82], "CQL(P)":[8.87, 6.24]},
    "MO-Walker2d-v2_50000_amateur_uniform":   {"MODT(P)":[164.2, 13.5], "MORvS(P)":[1.94, 0.06], "BC(P)":[53.1, 34.6], "CQL(P)":[7.33, 5.89]},
    "MO-Ant-v2_50000_expert_uniform":         {"MODT(P)":[8.26, 2.22], "MORvS(P)":[6.50, 0.81], "BC(P)":[46.2, 16.4], "CQL(P)":[0.58, 0.10]},
    "MO-HalfCheetah-v2_50000_expert_uniform": {"MODT(P)":[1.24, 0.23], "MORvS(P)":[0.67, 0.05], "BC(P)":[1.78, 0.39], "CQL(P)":[0.10, 0.00]},
    "MO-Hopper-v2_50000_expert_uniform":      {"MODT(P)":[16.3, 10.6], "MORvS(P)":[3.03, 0.36], "BC(P)":[52.5, 4.88], "CQL(P)":[2.84, 2.46]},
    "MO-Hopper-v3_50000_expert_uniform":      {"MODT(P)":[1.40, 0.44], "MORvS(P)":[2.72, 1.93], "BC(P)":[0.72, 0.09], "CQL(P)":[2.60, 3.14]},
    "MO-Swimmer-v2_50000_expert_uniform":     {"MODT(P)":[15.0, 7.49], "MORvS(P)":[4.39, 0.07], "BC(P)":[4.50, 0.39], "CQL(P)":[13.6, 5.31]},
    "MO-Walker2d-v2_50000_expert_uniform":    {"MODT(P)":[0.99, 0.44], "MORvS(P)":[3.22, 0.73], "BC(P)":[75.6, 52.3], "CQL(P)":[6.23, 10.7]},
}

# "B":[5.61, -1], "MODT(P)":[5.92, 0.04], "MORvS(P)":[6.07, 0.02], "BC(P)":[4.37, 0.06], "CQL(P)":[5.62, 0.23], "MODT":[4.88, 0.60], "MORvS":[4.37, 0.56], "BC":[2.34, 0.15], "CQL":[2.80, 0.68]
# "B":[5.68, -1], "MODT(P)":[5.69, 0.01], "MORvS(P)":[5.77, 0.00], "BC(P)":[5.46, 0.02], "CQL(P)":[5.54, 0.02], "MODT":[5.51, 0.01], "MORvS":[4.66, 0.05], "BC":[2.92, 0.38], "CQL":[4.41, 0.08]
# "B":[1.97, -1], "MODT(P)":[1.81, 0.05], "MORvS(P)":[1.76, 0.03], "BC(P)":[1.35, 0.03], "CQL(P)":[1.64, 0.01], "MODT":[1.54, 0.08], "MORvS":[1.57, 0.01], "BC":[0.01, 0.01], "CQL":[0.00, 0.06]
# "B":[3.09, -1], "MODT(P)":[1.04, 0.16], "MORvS(P)":[2.77, 0.24], "BC(P)":[2.42, 0.18], "CQL(P)":[0.59, 0.42], "MODT":[1.61, 0.23], "MORvS":[1.30, 0.22], "BC":[0.03, 0.01], "CQL":[0.10, 0.16]
# "B":[2.11, -1], "MODT(P)":[1.67, 0.22], "MORvS(P)":[2.79, 0.03], "BC(P)":[2.82, 0.04], "CQL(P)":[1.69, 0.93], "MODT":[0.96, 0.19], "MORvS":[2.93, 0.03], "BC":[0.46, 0.15], "CQL":[0.74, 0.47]
# "B":[4.99, -1], "MODT(P)":[3.10, 0.34], "MORvS(P)":[4.98, 0.01], "BC(P)":[3.42, 0.42], "CQL(P)":[1.78, 0.33], "MODT":[3.76, 0.34], "MORvS":[4.32, 0.05], "BC":[0.91, 0.36], "CQL":[0.76, 0.81]

# "B":[6.32, -1], "MODT(P)":[6.21, 0.01], "MORvS(P)":[6.41, 0.01], "BC(P)":[4.88, 0.17], "CQL(P)":[5.76, 0.10], "MODT":[5.52, 0.16], "MORvS":[5.52, 0.02], "BC":[0.84, 0.60], "CQL":[3.52, 0.45]
# "B":[5.79, -1], "MODT(P)":[5.73, 0.00], "MORvS(P)":[5.78, 0.00], "BC(P)":[5.54, 0.05], "CQL(P)":[5.63, 0.04], "MODT":[5.59, 0.03], "MORvS":[4.19, 0.74], "BC":[1.53, 0.09], "CQL":[3.78, 0.46]
# "B":[2.09, -1], "MODT(P)":[2.00, 0.02], "MORvS(P)":[2.02, 0.02], "BC(P)":[1.23, 0.10], "CQL(P)":[0.33, 0.39], "MODT":[1.68, 0.03], "MORvS":[1.73, 0.07], "BC":[0.28, 0.21], "CQL":[0.02, 0.02]
# "B":[3.73, -1], "MODT(P)":[3.38, 0.05], "MORvS(P)":[3.42, 0.10], "BC(P)":[2.29, 0.07], "CQL(P)":[0.78, 0.24], "MODT":[1.05, 0.43], "MORvS":[2.53, 0.06], "BC":[0.06, 0.02], "CQL":[0.00, 0.00]
# "B":[3.25, -1], "MODT(P)":[3.15, 0.02], "MORvS(P)":[3.24, 0.00], "BC(P)":[3.21, 0.00], "CQL(P)":[3.22, 0.08], "MODT":[2.49, 0.19], "MORvS":[3.19, 0.01], "BC":[1.68, 0.38], "CQL":[2.08, 0.08]
# "B":[5.21, -1], "MODT(P)":[4.89, 0.05], "MORvS(P)":[5.14, 0.01], "BC(P)":[3.74, 0.11], "CQL(P)":[3.21, 0.32], "MODT":[0.65, 0.46], "MORvS":[5.10, 0.02], "BC":[0.07, 0.02], "CQL":[0.82, 0.62]

table_hv = {
    "MO-Ant-v2_50000_amateur_uniform":{},
    "MO-HalfCheetah-v2_50000_amateur_uniform":{},
    "MO-Hopper-v2_50000_amateur_uniform":{},
    "MO-Hopper-v3_50000_amateur_uniform":{},
    "MO-Swimmer-v2_50000_amateur_uniform":{},
    "MO-Walker2d-v2_50000_amateur_uniform":{},
    "MO-Ant-v2_50000_expert_uniform":{},
    "MO-HalfCheetah-v2_50000_expert_uniform":{},
    "MO-Hopper-v2_50000_expert_uniform":{},
    "MO-Hopper-v3_50000_expert_uniform":{},
    "MO-Swimmer-v2_50000_expert_uniform":{},
    "MO-Walker2d-v2_50000_expert_uniform":{},
}

table_sp = {
    "MO-Ant-v2_50000_amateur_uniform":{},
    "MO-HalfCheetah-v2_50000_amateur_uniform":{},
    "MO-Hopper-v2_50000_amateur_uniform":{},
    "MO-Hopper-v3_50000_amateur_uniform":{},
    "MO-Swimmer-v2_50000_amateur_uniform":{},
    "MO-Walker2d-v2_50000_amateur_uniform":{},
    "MO-Ant-v2_50000_expert_uniform":{},
    "MO-HalfCheetah-v2_50000_expert_uniform":{},
    "MO-Hopper-v2_50000_expert_uniform":{},
    "MO-Hopper-v3_50000_expert_uniform":{},
    "MO-Swimmer-v2_50000_expert_uniform":{},
    "MO-Walker2d-v2_50000_expert_uniform":{},
}

envs = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Swimmer-v2", "Walker2d-v2"]
dataset_types = ["amateur", "expert"]

# datasets = [
#     [f"SCORE", f"results/mo_dd_sepguide_pro/MO-{env}_50000_{dataset_type}_uniform/model/22M/rewards/best.npy"] for env in envs for dataset_type in dataset_types
# ] + [
#     [f"RTG", f"results/mo_dd_sepguide_pro_rtg/MO-{env}_50000_{dataset_type}_uniform/model/11M/rewards/best.npy"] for env in envs for dataset_type in dataset_types
# ]

# standard
# datasets = [
#     [f"best", f"results/mo_dd/MO-{env}_50000_amateur_uniform/eval_standard/11M/rewards/standard.npy"] for env in envs 
# ] + [
#     [f"best", f"results/mo_dd/MO-Hopper-v3_50000_amateur_uniform/eval_standard/22M/eval_reward_.npy"]
# ] + [
#     [f"best", f"results/mo_dd/MO-{env}_50000_expert_uniform/eval_standard/11M/rewards/standard.npy"] for env in envs 
# ] + [
#     [f"best", f"results/mo_dd/MO-Hopper-v3_50000_expert_uniform/eval_standard/22M/eval_reward_.npy"]
# ] + [
#     [f"500k", f"results/mo_dd/MO-{env}_50000_amateur_uniform/eval_standard/11M/rewards/standard.npy"] for env in envs 
# ] + [
#     [f"500k", f"results/mo_dd/MO-{env}_50000_expert_uniform/eval_standard/11M/rewards/standard.npy"] for env in envs 
# ]

datasets = [
    [f"best", f"results/mo_dd/MO-{env}_50000_amateur_uniform/eval_standard/11M/rewards/standard.npy"] for env in envs 
] + [
    [f"best", f"results/mo_dd/MO-{env}_50000_expert_uniform/eval_standard/11M/rewards/standard.npy"] for env in envs 
] + [
    [f"500k", f"results/mo_dd/MO-{env}_50000_amateur_uniform/eval_standard/11M/rewards/500k.npy"] for env in envs 
] + [
    [f"500k", f"results/mo_dd/MO-{env}_50000_expert_uniform/eval_standard/11M/rewards/500k.npy"] for env in envs 
]


# datasets = [
#     [f"Diffusion", f"results/mo_dd/MO-{env}_50000_{dataset_type}_uniform/eval_standard/11M/rewards/standard.npy"] for dataset_type in dataset_types for env in envs 
# ] + [
#     [f"wo norm", f"results/mo_dd_wo_norm/MO-{env}_50000_{dataset_type}_uniform/eval_standard/22M/rewards/standard.npy"] for dataset_type in dataset_types for env in envs 
# ] + [
#     [f"global norm", f"results/mo_dd/MO-{env}_50000_{dataset_type}_uniform/eval_standard/11M_global_norm/rewards/standard.npy"] for dataset_type in dataset_types for env in envs 
# ]


# datasets = [
#     [f"DDIM+UNIFORM", f"results/mo_dd/MO-{env}_50000_{dataset_type}_uniform/model/11M/rewards/best.npy"] for env in envs for dataset_type in dataset_types
# ] + [
#     [f"DDIM+QUAD", f"results/mo_dd/MO-{env}_50000_{dataset_type}_uniform/eval_quick/ddim_quad/rewards/.npy"] for env in envs for dataset_type in dataset_types
# ] + [
#     [f"DDPM+UNIFORM", f"results/mo_dd/MO-{env}_50000_{dataset_type}_uniform/eval_quick/ddpm_uniform/rewards/.npy"] for env in envs for dataset_type in dataset_types
# ] + [
#     [f"DDPM+QUAD", f"results/mo_dd/MO-{env}_50000_{dataset_type}_uniform/eval_quick/ddpm_quad/rewards/ddpm_quad.npy"] for env in envs for dataset_type in dataset_types
# ]
# datasets = [
#     [f"{horizon.split('_')[-1]}", f"results/mo_dd/MO-{env}-v2_50000_{training}/model/11M{horizon if horizon != '_H4' else ''}/rewards/best.npy"]
#     for horizon in ["_H4", "_H8", "_H16", "_H32"]
#     for env in ["Ant", "HalfCheetah", "Swimmer"]
#     for training in ["amateur_uniform", "expert_uniform"]
# ] + [
#     [f"{horizon.split('_')[-1]}", f"results/mo_dd/MO-{env}-v2_50000_{training}/model/11M{horizon if horizon != '_H32' else ''}/rewards/best.npy"]
#     for horizon in ["_H4", "_H8", "_H16", "_H32"]
#     for env in ["Hopper", "Walker2d"]
#     for training in ["amateur_uniform", "expert_uniform"]
# ]

# datasets = [
#     [f"{wcg}", f"results/mo_dd_sepguide_pro_rtg/MO-{env}-v2_50000_{training}/eval_quick/11M/wcg/rewards/{wcg}.npy"]
#     for wcg in ["0.0", "0.00001", "0.0001", "0.001", "0.01", "0.1", "0.2", "0.5", "1.0", "2.0", "5.0", "10.0"]
#     for env in ["Ant", "HalfCheetah", "Swimmer"]
#     for training in ["amateur_uniform", "expert_uniform"]
# ] + [
#     [f"{wcg}", f"results/mo_dd_sepguide_pro_rtg/MO-{env}-v2_50000_{training}/eval_quick/11M/wcg/rewards/{wcg}.npy"]
#     for wcg in ["0.0", "0.00001", "0.0001", "0.001", "0.01", "0.1", "0.2", "0.5", "1.0", "2.0", "5.0", "10.0"]
#     for env in ["Hopper",]
#     for training in ["expert_uniform"]
# ]

use_baselines_hv = [["B"], ["MODT(P)", "MORvS(P)", "BC(P)"]]
# use_baselines_hv = []
use_baselines_sp = [["MODT(P)", "MORvS(P)", "BC(P)", "CQL(P)"]]

use_algos, use_datasets = [], []

for i, [algo_name, path] in enumerate(datasets):
    if "v3" in path:
        HyperVolume = InnerHyperVolume([0, 0, 0])
    else:
        HyperVolume = InnerHyperVolume([0, 0])

    dataset_name = path.split('/')[2]

    use_algos.append(algo_name)
    use_datasets.append(dataset_name)

    eval_data = np.load(path)

    # tmp = np.array([HyperVolume.compute(eval_data[i]) for i in range(eval_data.shape[0])])
    # print(eval_data.shape)
    # tmp = np.array([eval_data[i][undominated_indices(eval_data[i])] for i in range(eval_data.shape[0])])
    # print(tmp, tmp.shape)
    sp = np.array([compute_sparsity(eval_data[i][undominated_indices(eval_data[i])]) for i in range(eval_data.shape[0])])
    # print(tmp.mean(), tmp.shape)
                
    table_hv[dataset_name][algo_name] = [round(np.array([HyperVolume.compute(eval_data[i]) for i in range(eval_data.shape[0])]).mean()/scale_hv[dataset_name], 2), round(np.array([HyperVolume.compute(eval_data[i]) for i in range(eval_data.shape[0])]).std()/scale_hv[dataset_name], 2)]
    table_sp[dataset_name][algo_name] = [round(sp.mean()/scale_sp[dataset_name], 2), round(sp.std()/scale_sp[dataset_name], 2)]

from collections import OrderedDict
use_algos = list(OrderedDict.fromkeys(use_algos))
use_datasets = list(OrderedDict.fromkeys(use_datasets))

# HV

caption = "HV"
std = True

width = sum([len(i) for i in use_baselines_hv]) + len(use_algos)
height = len(use_datasets)

if std:
    head = "".join(["p{0.1\\textwidth}"*len(i) + "|" for i in use_baselines_hv]) + "p{0.1\\textwidth}"*len(use_algos)
else:
    head = "".join(["p{0.08\\textwidth}"*len(i) + "|" for i in use_baselines_hv]) + "p{0.08\\textwidth}"*len(use_algos)
env = " & ".join(["Environments"] + [item for sublist in use_baselines_hv for item in sublist] + use_algos)

# expert, amateur = False, False
# for i in use_datasets:
#     if "expert" in i:
#         expert = True
#     else:
#         amateur = True
# flag = amateur and expert

body = ""

for idx_d, dataset in enumerate(use_datasets):
    if idx_d > 0:
        body += "    "
    body += f"{dataset.split('-')[1]}-{dataset.split('-')[2].split('_')[0]}-{dataset.split('_')[-2]}-{dataset.split('_')[-1]}{' '*(18-len(dataset.split('_')[-2]+dataset.split('-')[1]))} & "

    line_max, line_max_std = -1e9, 0

    for j in use_baselines_hv:
        for k in j:
            if k != "B":
                if baseline_table_hv[dataset][k][0] > line_max:
                    line_max = baseline_table_hv[dataset][k][0]
                    line_max_std = baseline_table_hv[dataset][k][1]
                elif baseline_table_hv[dataset][k][0] == line_max:
                    line_max_std = min(line_max_std, baseline_table_hv[dataset][k][1])
    
    for j in use_algos:
        if table_hv[dataset][j][0] > line_max:
            line_max = table_hv[dataset][j][0]
            line_max_std = table_hv[dataset][j][1]
        elif table_hv[dataset][j][0] == line_max:
            line_max_std = min(line_max_std, table_hv[dataset][j][1])
    
    for j in use_baselines_hv:
        for k in j:
            if std:
                if baseline_table_hv[dataset][k][0] >= line_max and k != 'B':
                    body += "\\textbf{" + f"{baseline_table_hv[dataset][k][0]:.2f}±" + '.' + f"{baseline_table_hv[dataset][k][1]:2f}".split(".")[1][:2] + "}" + " & "
                else:
                    body += f"{baseline_table_hv[dataset][k][0]:.2f}±" + '.' + f"{baseline_table_hv[dataset][k][1]:2f}".split(".")[1][:2] + " & "
            else:
                if baseline_table_hv[dataset][k][0] >= line_max and k != 'B':
                    body += "\\textbf{" + f"{baseline_table_hv[dataset][k][0]:.2f}" + "}" + " & "
                else:
                    body += f"{baseline_table_hv[dataset][k][0]:.2f}" + " & "
    
    for idx_j, j in enumerate(use_algos):
        if table_hv[dataset].get(j) is not None:
            if std:
                if table_hv[dataset][j][0] >= line_max:
                    body += "\\textbf{" + f"{table_hv[dataset][j][0]:.2f}±" + '.' + f"{table_hv[dataset][j][1]:2f}".split(".")[1][:2] + "}"
                else:
                    body += f"{table_hv[dataset][j][0]:.2f}±" + '.' + f"{table_hv[dataset][j][1]:2f}".split(".")[1][:2]
            else:
                if table_hv[dataset][j][0] >= line_max:
                    body += "\\textbf{" + f"{table_hv[dataset][j][0]:.2f}" + "}"
                else:
                    body += f"{table_hv[dataset][j][0]:.2f}"
        else:
            body += "/"
        
        if idx_j != len(use_algos)-1:
            body += " & "
        else:
            body += r" \\"
    
    if idx_d < height-1:
        body += "\n"


s = f"""
\\begin{{table}}[H]
  \\caption{{{caption}}}
  \\label{{sample-table}}
  \\Scentering
  \\begin{{tabular}}{{p{{0.3\\textwidth}}||{head}}}
    \\toprule
    {env} \\\\
    \\midrule
    {body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""

print(s)


#SP
caption = "SP"
std = True

width = sum([len(i) for i in use_baselines_sp]) + len(use_algos)
height = len(use_datasets)

if std:
    head = "".join(["p{0.1\\textwidth}"*len(i) + "|" for i in use_baselines_sp]) + "p{0.1\\textwidth}"*len(use_algos)
else:
    head = "".join(["p{0.08\\textwidth}"*len(i) + "|" for i in use_baselines_sp]) + "p{0.08\\textwidth}"*len(use_algos)
env = " & ".join(["Environments"] + [item for sublist in use_baselines_sp for item in sublist] + use_algos)

# expert, amateur = False, False
# for i in use_datasets:
#     if "expert" in i:
#         expert = True
#     else:
#         amateur = True
# flag = amateur and expert


body = ""

for idx_d, dataset in enumerate(use_datasets):
    if idx_d > 0:
        body += "    "
    body += f"{dataset.split('-')[1]}-{dataset.split('-')[2].split('_')[0]}-{dataset.split('_')[-2]}-{dataset.split('_')[-1]}{' '*(18-len(dataset.split('_')[-2]+dataset.split('-')[1]))} & "

    line_min, line_min_std = 1e9, 0

    for j in use_baselines_sp:
        for k in j:
            if k != "B":
                if baseline_table_sp[dataset][k][0] < line_min:
                    line_min = baseline_table_sp[dataset][k][0]
                    line_min_std = baseline_table_sp[dataset][k][1]
                elif baseline_table_sp[dataset][k][0] == line_min:
                    line_min_std = min(line_min_std, baseline_table_sp[dataset][k][1])
    
    for j in use_algos:
        if table_sp[dataset][j][0] < line_min:
            line_min = table_sp[dataset][j][0]
            line_min_std = table_sp[dataset][j][1]
        elif table_sp[dataset][j][0] == line_min:
            line_min_std = min(line_min_std, table_sp[dataset][j][1])
    
    for j in use_baselines_sp:
        for k in j:
            if std:
                if baseline_table_sp[dataset][k][0] <= line_min and k != 'B':
                    body += "\\textbf{" + f"{baseline_table_sp[dataset][k][0]:.2f}±" + '.' + f"{baseline_table_sp[dataset][k][1]:2f}".split(".")[1][:2] + "}" + " & "
                else:
                    body += f"{baseline_table_sp[dataset][k][0]:.2f}±" + '.' + f"{baseline_table_sp[dataset][k][1]:2f}".split(".")[1][:2] + " & "
            else:
                if baseline_table_sp[dataset][k][0] <= line_min and k != 'B':
                    body += "\\textbf{" + f"{baseline_table_sp[dataset][k][0]:.2f}" + "}" + " & "
                else:
                    body += f"{baseline_table_sp[dataset][k][0]:.2f}" + " & "
    
    for idx_j, j in enumerate(use_algos):
        if table_sp[dataset].get(j) is not None:
            if std:
                if table_sp[dataset][j][0] <= line_min:
                    body += "\\textbf{" + f"{table_sp[dataset][j][0]:.2f}±" + '.' + f"{table_sp[dataset][j][1]:2f}".split(".")[1][:2] + "}"
                else:
                    body += f"{table_sp[dataset][j][0]:.2f}±" + '.' + f"{table_sp[dataset][j][1]:2f}".split(".")[1][:2]
            else:
                if table_sp[dataset][j][0] <= line_min:
                    body += "\\textbf{" + f"{table_sp[dataset][j][0]:.2f}" + "}"
                else:
                    body += f"{table_sp[dataset][j][0]:.2f}"
        else:
            body += "/"
        
        if idx_j != len(use_algos)-1:
            body += " & "
        else:
            body += r" \\"
    
    if idx_d < height-1:
        body += "\n"


s = f"""
\\begin{{table}}[H]
  \\caption{{{caption}}}
  \\label{{sample-table}}
  \\Scentering
  \\begin{{tabular}}{{p{{0.3\\textwidth}}||{head}}}
    \\toprule
    {env} \\\\
    \\midrule
    {body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""

print(s)