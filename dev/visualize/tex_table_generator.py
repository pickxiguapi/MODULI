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

scale = {
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

baseline_table = {
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

table = {
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


datasets = [
    ["dd", "results/mo_dd/MO-Ant-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["dd", "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["dd", "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["dd", "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["dd", "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["dd", "results/mo_dd/MO-Ant-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["dd", "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["dd", "results/mo_dd/MO-Hopper-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["dd", "results/mo_dd/MO-Swimmer-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["dd", "results/mo_dd/MO-Walker2d-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],

    ["seperate condition", "results/mo_dd_sepcond/MO-HalfCheetah-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["seperate condition", "results/mo_dd_sepcond/MO-Hopper-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["seperate condition", "results/mo_dd_sepcond/MO-Swimmer-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["seperate condition", "results/mo_dd_sepcond/MO-HalfCheetah-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["seperate condition", "results/mo_dd_sepcond/MO-Hopper-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["seperate condition", "results/mo_dd_sepcond/MO-Swimmer-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],

    ["seperate guidance", "results/mo_dd_sepguide/MO-HalfCheetah-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["seperate guidance", "results/mo_dd_sepguide/MO-Hopper-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["seperate guidance", "results/mo_dd_sepguide/MO-Swimmer-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["seperate guidance", "results/mo_dd_sepguide/MO-HalfCheetah-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["seperate guidance", "results/mo_dd_sepguide/MO-Hopper-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["seperate guidance", "results/mo_dd_sepguide/MO-Swimmer-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],

    ["diffuser", "results/mo_diffuser/MO-HalfCheetah-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["diffuser", "results/mo_diffuser/MO-Hopper-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["diffuser", "results/mo_diffuser/MO-Swimmer-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["diffuser", "results/mo_diffuser/MO-HalfCheetah-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["diffuser", "results/mo_diffuser/MO-Hopper-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["diffuser", "results/mo_diffuser/MO-Swimmer-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],

    ["ddpro", "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["ddpro", "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["ddpro", "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["ddpro", "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["ddpro", "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],
    ["ddpro", "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/eval_standard/22M/eval_reward_best.npy"],
]

datasets = [
    ["dd", "results/mo_dd/MO-Ant-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
    ["dd", "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
    ["dd", "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
    # ["dd_3obj", "results/mo_dd/MO-Hopper-v3_50000_amateur_uniform/model/11M/rewards/best.npy"],
    ["dd", "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
    ["dd", "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
    ["dd", "results/mo_dd/MO-Ant-v2_50000_expert_uniform/model/11M/rewards/best.npy"],
    ["dd", "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/model/11M/rewards/best.npy"],
    ["dd", "results/mo_dd/MO-Hopper-v2_50000_expert_uniform/model/11M/rewards/best.npy"],
    # ["dd_3obj", "results/mo_dd/MO-Hopper-v3_50000_expert_uniform/model/11M/rewards/best.npy"],
    ["dd", "results/mo_dd/MO-Swimmer-v2_50000_expert_uniform/model/11M/rewards/best.npy"],
    ["dd", "results/mo_dd/MO-Walker2d-v2_50000_expert_uniform/model/11M/rewards/best.npy"],

    # ["16M", "results/mo_dd/MO-Ant-v2_50000_amateur_uniform/model/16M/rewards/best.npy"],
    # ["16M", "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/model/16M/rewards/best.npy"],
    # ["16M", "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/model/16M/rewards/best.npy"],
    # # ["16M_3obj", "results/mo_dd/MO-Hopper-v3_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["16M", "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/model/16M/rewards/best.npy"],
    # ["16M", "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/model/16M/rewards/best.npy"],
    # ["16M", "results/mo_dd/MO-Ant-v2_50000_expert_uniform/model/16M/rewards/best.npy"],
    # ["16M", "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/model/16M/rewards/best.npy"],
    # ["16M", "results/mo_dd/MO-Hopper-v2_50000_expert_uniform/model/16M/rewards/best.npy"],
    # # ["16M_3obj", "results/mo_dd/MO-Hopper-v3_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["16M", "results/mo_dd/MO-Swimmer-v2_50000_expert_uniform/model/16M/rewards/best.npy"],
    # ["16M", "results/mo_dd/MO-Walker2d-v2_50000_expert_uniform/model/16M/rewards/best.npy"],

    # ["22M", "results/mo_dd/MO-Ant-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["22M", "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["22M", "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # # ["22M_3obj", "results/mo_dd/MO-Hopper-v3_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["22M", "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["22M", "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["22M", "results/mo_dd/MO-Ant-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["22M", "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["22M", "results/mo_dd/MO-Hopper-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # # ["22M_3obj", "results/mo_dd/MO-Hopper-v3_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["22M", "results/mo_dd/MO-Swimmer-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["22M", "results/mo_dd/MO-Walker2d-v2_50000_expert_uniform/model/22M/rewards/best.npy"],

    # ["7M", "results/mo_dd/MO-Ant-v2_50000_amateur_uniform/model/7M/rewards/best.npy"],
    # ["7M", "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/model/7M/rewards/best.npy"],
    # ["7M", "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/model/7M/rewards/best.npy"],
    # # ["7M_3obj", "results/mo_dd/MO-Hopper-v3_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["7M", "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/model/7M/rewards/best.npy"],
    # ["7M", "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/model/7M/rewards/best.npy"],
    # ["7M", "results/mo_dd/MO-Ant-v2_50000_expert_uniform/model/7M/rewards/best.npy"],
    # ["7M", "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/model/7M/rewards/best.npy"],
    # ["7M", "results/mo_dd/MO-Hopper-v2_50000_expert_uniform/model/7M/rewards/best.npy"],
    # # ["7M_3obj", "results/mo_dd/MO-Hopper-v3_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["7M", "results/mo_dd/MO-Swimmer-v2_50000_expert_uniform/model/7M/rewards/best.npy"],
    # ["7M", "results/mo_dd/MO-Walker2d-v2_50000_expert_uniform/model/7M/rewards/best.npy"],

    # ["5M", "results/mo_dd/MO-Ant-v2_50000_amateur_uniform/model/5M/rewards/best.npy"],
    # ["5M", "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/model/5M/rewards/best.npy"],
    # ["5M", "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/model/5M/rewards/best.npy"],
    # # ["5M_3obj", "results/mo_dd/MO-Hopper-v3_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["5M", "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/model/5M/rewards/best.npy"],
    # ["5M", "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/model/5M/rewards/best.npy"],
    # ["5M", "results/mo_dd/MO-Ant-v2_50000_expert_uniform/model/5M/rewards/best.npy"],
    # ["5M", "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/model/5M/rewards/best.npy"],
    # ["5M", "results/mo_dd/MO-Hopper-v2_50000_expert_uniform/model/5M/rewards/best.npy"],
    # # ["5M_3obj", "results/mo_dd/MO-Hopper-v3_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["5M", "results/mo_dd/MO-Swimmer-v2_50000_expert_uniform/model/5M/rewards/best.npy"],
    # ["5M", "results/mo_dd/MO-Walker2d-v2_50000_expert_uniform/model/5M/rewards/best.npy"],

    # ["dd w dyna", "results/mo_dd_w_dyna/MO-Ant-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd w dyna", "results/mo_dd_w_dyna/MO-HalfCheetah-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd w dyna", "results/mo_dd_w_dyna/MO-Hopper-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd w dyna", "results/mo_dd_w_dyna/MO-Swimmer-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd w dyna", "results/mo_dd_w_dyna/MO-Walker2d-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd w dyna", "results/mo_dd_w_dyna/MO-Ant-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["dd w dyna", "results/mo_dd_w_dyna/MO-HalfCheetah-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["dd w dyna", "results/mo_dd_w_dyna/MO-Hopper-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["dd w dyna", "results/mo_dd_w_dyna/MO-Swimmer-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["dd w dyna", "results/mo_dd_w_dyna/MO-Walker2d-v2_50000_expert_uniform/model/22M/rewards/best.npy"],

    # ["dd wo score", "results/mo_dd_wo_score/MO-Ant-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd wo score", "results/mo_dd_wo_score/MO-HalfCheetah-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd wo score", "results/mo_dd_wo_score/MO-Hopper-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd wo score", "results/mo_dd_wo_score/MO-Swimmer-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd wo score", "results/mo_dd_wo_score/MO-Walker2d-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd wo score", "results/mo_dd_wo_score/MO-Ant-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["dd wo score", "results/mo_dd_wo_score/MO-HalfCheetah-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["dd wo score", "results/mo_dd_wo_score/MO-Hopper-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["dd wo score", "results/mo_dd_wo_score/MO-Swimmer-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["dd wo score", "results/mo_dd_wo_score/MO-Walker2d-v2_50000_expert_uniform/model/22M/rewards/best.npy"],

    # ["dd wo norm", "results/mo_dd_wo_norm/MO-Ant-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd wo norm", "results/mo_dd_wo_norm/MO-HalfCheetah-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd wo norm", "results/mo_dd_wo_norm/MO-Hopper-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd wo norm", "results/mo_dd_wo_norm/MO-Swimmer-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd wo norm", "results/mo_dd_wo_norm/MO-Walker2d-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["dd wo norm", "results/mo_dd_wo_norm/MO-Ant-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["dd wo norm", "results/mo_dd_wo_norm/MO-HalfCheetah-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["dd wo norm", "results/mo_dd_wo_norm/MO-Hopper-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["dd wo norm", "results/mo_dd_wo_norm/MO-Swimmer-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["dd wo norm", "results/mo_dd_wo_norm/MO-Walker2d-v2_50000_expert_uniform/model/22M/rewards/best.npy"],

    # ["seperate condition", "results/mo_dd_sepcond/MO-HalfCheetah-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["seperate condition", "results/mo_dd_sepcond/MO-Hopper-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["seperate condition", "results/mo_dd_sepcond/MO-Swimmer-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["seperate condition", "results/mo_dd_sepcond/MO-HalfCheetah-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["seperate condition", "results/mo_dd_sepcond/MO-Hopper-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["seperate condition", "results/mo_dd_sepcond/MO-Swimmer-v2_50000_expert_uniform/model/22M/rewards/best.npy"],

    # ["seperate guidance", "results/mo_dd_sepguide/MO-HalfCheetah-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["seperate guidance", "results/mo_dd_sepguide/MO-Hopper-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["seperate guidance", "results/mo_dd_sepguide/MO-Swimmer-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["seperate guidance", "results/mo_dd_sepguide/MO-HalfCheetah-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["seperate guidance", "results/mo_dd_sepguide/MO-Hopper-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["seperate guidance", "results/mo_dd_sepguide/MO-Swimmer-v2_50000_expert_uniform/model/22M/rewards/best.npy"],

    ["seperate guidance pro", "results/mo_dd_sepguide_pro_rtg/MO-Ant-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
    ["seperate guidance pro", "results/mo_dd_sepguide_pro_rtg/MO-HalfCheetah-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
    ["seperate guidance pro", "results/mo_dd_sepguide_pro_rtg/MO-Hopper-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
    ["seperate guidance pro", "results/mo_dd_sepguide_pro_rtg/MO-Swimmer-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
    ["seperate guidance pro", "results/mo_dd_sepguide_pro_rtg/MO-Walker2d-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
    ["seperate guidance pro", "results/mo_dd_sepguide_pro_rtg/MO-Ant-v2_50000_expert_uniform/model/11M/rewards/best.npy"],
    ["seperate guidance pro", "results/mo_dd_sepguide_pro_rtg/MO-HalfCheetah-v2_50000_expert_uniform/model/11M/rewards/best.npy"],
    ["seperate guidance pro", "results/mo_dd_sepguide_pro_rtg/MO-Hopper-v2_50000_expert_uniform/model/11M/rewards/best.npy"],
    ["seperate guidance pro", "results/mo_dd_sepguide_pro_rtg/MO-Swimmer-v2_50000_expert_uniform/model/11M/rewards/best.npy"],
    ["seperate guidance pro", "results/mo_dd_sepguide_pro_rtg/MO-Walker2d-v2_50000_expert_uniform/model/11M/rewards/best.npy"],

    # ["diffuser", "results/mo_diffuser/MO-HalfCheetah-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["diffuser", "results/mo_diffuser/MO-Hopper-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["diffuser", "results/mo_diffuser/MO-Swimmer-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["diffuser", "results/mo_diffuser/MO-HalfCheetah-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["diffuser", "results/mo_diffuser/MO-Hopper-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["diffuser", "results/mo_diffuser/MO-Swimmer-v2_50000_expert_uniform/model/22M/rewards/best.npy"],

    # ["diffuser pref", "results/mo_diffuser_prefguide/MO-Ant-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["diffuser pref", "results/mo_diffuser_prefguide/MO-HalfCheetah-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["diffuser pref", "results/mo_diffuser_prefguide/MO-Hopper-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["diffuser pref", "results/mo_diffuser_prefguide/MO-Swimmer-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["diffuser pref", "results/mo_diffuser_prefguide/MO-Walker2d-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["diffuser pref", "results/mo_diffuser_prefguide/MO-Ant-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["diffuser pref", "results/mo_diffuser_prefguide/MO-HalfCheetah-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["diffuser pref", "results/mo_diffuser_prefguide/MO-Hopper-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["diffuser pref", "results/mo_diffuser_prefguide/MO-Swimmer-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["diffuser pref", "results/mo_diffuser_prefguide/MO-Walker2d-v2_50000_expert_uniform/model/22M/rewards/best.npy"],

    # ["ddpro", "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["ddpro", "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["ddpro", "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/model/22M/rewards/best.npy"],
    # ["ddpro", "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["ddpro", "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
    # ["ddpro", "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/model/22M/rewards/best.npy"],
]

# datasets = [
#     ["dd", "results/mo_dd/MO-Ant-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
#     ["dd", "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
#     ["dd", "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
#     ["dd", "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
#     ["dd", "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/model/11M/rewards/best.npy"],
#     ["dd", "results/mo_dd/MO-Ant-v2_50000_expert_uniform/model/11M/rewards/best.npy"],
#     ["dd", "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/model/11M/rewards/best.npy"],
#     ["dd", "results/mo_dd/MO-Hopper-v2_50000_expert_uniform/model/11M/rewards/best.npy"],
#     ["dd", "results/mo_dd/MO-Swimmer-v2_50000_expert_uniform/model/11M/rewards/best.npy"],
#     ["dd", "results/mo_dd/MO-Walker2d-v2_50000_expert_uniform/model/11M/rewards/best.npy"],

#     ["quad ddpm", "results/mo_dd/MO-Ant-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/ddpm_quad.npy"],
#     ["quad ddpm", "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/ddpm_quad.npy"],
#     ["quad ddpm", "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/ddpm_quad.npy"],
#     ["quad ddpm", "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/ddpm_quad.npy"],
#     ["quad ddpm", "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/ddpm_quad.npy"],
#     ["quad ddpm", "results/mo_dd/MO-Ant-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/ddpm_quad.npy"],
#     ["quad ddpm", "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/ddpm_quad.npy"],
#     ["quad ddpm", "results/mo_dd/MO-Hopper-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/ddpm_quad.npy"],
#     ["quad ddpm", "results/mo_dd/MO-Swimmer-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/ddpm_quad.npy"],
#     ["quad ddpm", "results/mo_dd/MO-Walker2d-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/ddpm_quad.npy"],

#     ["5", "results/mo_dd/MO-Ant-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/5.npy"],
#     ["5", "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/5.npy"],
#     ["5", "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/5.npy"],
#     ["5", "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/5.npy"],
#     ["5", "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/5.npy"],
#     ["5", "results/mo_dd/MO-Ant-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/5.npy"],
#     ["5", "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/5.npy"],
#     ["5", "results/mo_dd/MO-Hopper-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/5.npy"],
#     ["5", "results/mo_dd/MO-Swimmer-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/5.npy"],
#     ["5", "results/mo_dd/MO-Walker2d-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/5.npy"],

#     ["20", "results/mo_dd/MO-Ant-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/20.npy"],
#     ["20", "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/20.npy"],
#     ["20", "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/20.npy"],
#     ["20", "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/20.npy"],
#     ["20", "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/eval_quick/ddpm_quad/rewards/20.npy"],
#     ["20", "results/mo_dd/MO-Ant-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/20.npy"],
#     ["20", "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/20.npy"],
#     ["20", "results/mo_dd/MO-Hopper-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/20.npy"],
#     ["20", "results/mo_dd/MO-Swimmer-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/20.npy"],
#     ["20", "results/mo_dd/MO-Walker2d-v2_50000_expert_uniform/eval_quick/ddpm_quad/rewards/20.npy"],
#     # ["dd global", "results/mo_dd/MO-Ant-v2_50000_amateur_uniform/eval_quick/11M_global_norm/rewards/best.npy"],
#     # ["dd global", "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/11M_global_norm/rewards/best.npy"],
#     # ["dd global", "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/eval_quick/11M_global_norm/rewards/best.npy"],
#     # ["dd global", "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/11M_global_norm/rewards/best.npy"],
#     # ["dd global", "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/eval_quick/11M_global_norm/rewards/best.npy"],
#     # ["dd global", "results/mo_dd/MO-Ant-v2_50000_expert_uniform/eval_quick/11M_global_norm/rewards/best.npy"],
#     # ["dd global", "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/11M_global_norm/rewards/best.npy"],
#     # ["dd global", "results/mo_dd/MO-Hopper-v2_50000_expert_uniform/eval_quick/11M_global_norm/rewards/best.npy"],
#     # ["dd global", "results/mo_dd/MO-Swimmer-v2_50000_expert_uniform/eval_quick/11M_global_norm/rewards/best.npy"],
#     # ["dd global", "results/mo_dd/MO-Walker2d-v2_50000_expert_uniform/eval_quick/11M_global_norm/rewards/best.npy"],

#     # ["dd wo norm", "results/mo_dd_wo_norm/MO-Ant-v2_50000_amateur_uniform/eval_quick/11M/rewards/best.npy"],
#     # ["dd wo norm", "results/mo_dd_wo_norm/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/11M/rewards/best.npy"],
#     # ["dd wo norm", "results/mo_dd_wo_norm/MO-Hopper-v2_50000_amateur_uniform/eval_quick/11M/rewards/best.npy"],
#     # ["dd wo norm", "results/mo_dd_wo_norm/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/11M/rewards/best.npy"],
#     # ["dd wo norm", "results/mo_dd_wo_norm/MO-Walker2d-v2_50000_amateur_uniform/eval_quick/11M/rewards/best.npy"],
#     # ["dd wo norm", "results/mo_dd_wo_norm/MO-Ant-v2_50000_expert_uniform/eval_quick/11M/rewards/best.npy"],
#     # ["dd wo norm", "results/mo_dd_wo_norm/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/11M/rewards/best.npy"],
#     # ["dd wo norm", "results/mo_dd_wo_norm/MO-Hopper-v2_50000_expert_uniform/eval_quick/11M/rewards/best.npy"],
#     # ["dd wo norm", "results/mo_dd_wo_norm/MO-Swimmer-v2_50000_expert_uniform/eval_quick/11M/rewards/best.npy"],
#     # ["dd wo norm", "results/mo_dd_wo_norm/MO-Walker2d-v2_50000_expert_uniform/eval_quick/11M/rewards/best.npy"],
# ]

# wcgs = ["0.0", "0.00001", "0.01","0.2", "0.3", "0.5", "1.0", "2.0", "3.0", "5.0", "10.0"]
# envs = ["MO-Ant-v2_50000_amateur_uniform", "MO-HalfCheetah-v2_50000_amateur_uniform", "MO-Hopper-v2_50000_amateur_uniform", "MO-Swimmer-v2_50000_amateur_uniform", "MO-Walker2d-v2_50000_amateur_uniform", "MO-Ant-v2_50000_expert_uniform", "MO-HalfCheetah-v2_50000_expert_uniform", "MO-Hopper-v2_50000_expert_uniform", "MO-Swimmer-v2_50000_expert_uniform", "MO-Walker2d-v2_50000_expert_uniform"]

# datasets = [
#     [f"{wcg}", f"results/mo_diffuser_prefguide/{env}/eval_quick/22M/wcg/eval_reward_{wcg}.npy"] for wcg in wcgs for env in envs
# ]

# datasets = [
#     [f"diffuser", f"results/mo_diffuser_prefguide/{env}/model/11M/rewards/best.npy"] for env in envs
# ] + [
#     [f"dd", f"results/mo_dd/{env}/model/11M/rewards/best.npy"] for env in envs
# ]

# datasets = [
#     ["wcfg=0.0", "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_0.0.npy"],
#     ["wcfg=0.0", "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_0.0.npy"],
#     ["wcfg=0.0", "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_0.0.npy"],
#     ["wcfg=0.0", "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_0.0.npy"],
#     ["wcfg=0.0", "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_0.0.npy"],
#     ["wcfg=0.0", "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_0.0.npy"],

#     ["0.1", "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_0.1.npy"],
#     ["0.1", "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_0.1.npy"],
#     ["0.1", "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_0.1.npy"],
#     ["0.1", "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_0.1.npy"],
#     ["0.1", "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_0.1.npy"],
#     ["0.1", "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_0.1.npy"],

#     ["0.2", "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_0.2.npy"],
#     ["0.2", "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_0.2.npy"],
#     ["0.2", "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_0.2.npy"],
#     ["0.2", "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_0.2.npy"],
#     ["0.2", "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_0.2.npy"],
#     ["0.2", "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_0.2.npy"],

#     ["0.5", "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_0.5.npy"],
#     ["0.5", "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_0.5.npy"],
#     ["0.5", "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_0.5.npy"],
#     ["0.5", "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_0.5.npy"],
#     ["0.5", "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_0.5.npy"],
#     ["0.5", "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_0.5.npy"],

#     ["1.0", "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_1.0.npy"],
#     ["1.0", "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_1.0.npy"],
#     ["1.0", "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_1.0.npy"],
#     ["1.0", "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_1.0.npy"],
#     ["1.0", "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_1.0.npy"],
#     ["1.0", "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_1.0.npy"],

#     ["1.5", "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_1.5.npy"],
#     ["1.5", "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_1.5.npy"],
#     ["1.5", "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_1.5.npy"],
#     ["1.5", "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_1.5.npy"],
#     ["1.5", "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_1.5.npy"],
#     ["1.5", "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_1.5.npy"],

#     ["2.0", "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_2.0.npy"],
#     ["2.0", "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_2.0.npy"],
#     ["2.0", "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_2.0.npy"],
#     ["2.0", "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_2.0.npy"],
#     ["2.0", "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_2.0.npy"],
#     ["2.0", "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_2.0.npy"],

#     ["3.0", "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_3.0.npy"],
#     ["3.0", "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_3.0.npy"],
#     ["3.0", "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_3.0.npy"],
#     ["3.0", "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_3.0.npy"],
#     ["3.0", "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_3.0.npy"],
#     ["3.0", "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_3.0.npy"],

#     ["6.0", "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_6.0.npy"],
#     ["6.0", "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_6.0.npy"],
#     ["6.0", "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_6.0.npy"],
#     ["6.0", "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_6.0.npy"],
#     ["6.0", "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_6.0.npy"],
#     ["6.0", "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_6.0.npy"],

#     ["10.0", "results/mo_ddpro/MO-HalfCheetah-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_10.0.npy"],
#     ["10.0", "results/mo_ddpro/MO-Hopper-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_10.0.npy"],
#     ["10.0", "results/mo_ddpro/MO-Swimmer-v2_50000_amateur_uniform/eval_quick/22M/wcfg/eval_reward_10.0.npy"],
#     ["10.0", "results/mo_ddpro/MO-HalfCheetah-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_10.0.npy"],
#     ["10.0", "results/mo_ddpro/MO-Hopper-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_10.0.npy"],
#     ["10.0", "results/mo_ddpro/MO-Swimmer-v2_50000_expert_uniform/eval_quick/22M/wcfg/eval_reward_10.0.npy"],
# ]


# datasets = [
#     [f"{reward}", f"results/mo_dd_sepguide/MO-{env}-v2_50000_{training}/eval_quick/22M/wcg/eval_reward_{reward}.npy"]
#     for reward in ["0.0", "0.00001", "0.0001", "0.001", "0.01", "0.1", "0.2", "0.3", "0.5", "1.0", "2.0", "3.0", "5.0", "10.0"]
#     for env in ["HalfCheetah", "Hopper", "Swimmer"]
#     for training in ["amateur_uniform", "expert_uniform"]
# ]

datasets = [
    [f"{horizon.split('_')[-1]}", f"results/mo_dd/MO-{env}-v2_50000_{training}/model/11M{horizon if horizon != '_H4' else ''}/rewards/best.npy"]
    for horizon in ["_H4", "_H8", "_H16", "_H32"]
    for env in ["Ant", "HalfCheetah", "Swimmer"]
    for training in ["amateur_uniform", "expert_uniform"]
] + [
    [f"{horizon.split('_')[-1]}", f"results/mo_dd/MO-{env}-v2_50000_{training}/model/11M{horizon if horizon != '_H32' else ''}/rewards/best.npy"]
    for horizon in ["_H4", "_H8", "_H16", "_H32"]
    for env in ["Hopper", "Walker2d"]
    for training in ["amateur_uniform", "expert_uniform"]
]
algo_callnames, algos = [], []

use_baselines = [["B"]]

for i, [name ,path] in enumerate(datasets):
    print(name, path)
    if "v3" in path:
        HyperVolume = InnerHyperVolume([0, 0, 0])
    else:
        HyperVolume = InnerHyperVolume([0, 0])
    algo_name = path.split("/")[1]
    dataset_name = path.split("/")[2]

    eval_data = np.load(path)

    # tmp = np.array([HyperVolume.compute(eval_data[i]) for i in range(eval_data.shape[0])])
    # print(eval_data.shape)
    # tmp = np.array([eval_data[i][undominated_indices(eval_data[i])] for i in range(eval_data.shape[0])])
    # print(tmp, tmp.shape)
    # tmp = np.array([compute_sparsity(eval_data[i][undominated_indices(eval_data[i])]) for i in range(eval_data.shape[0])])
    # print(tmp.mean(), tmp.shape)
                
    table[dataset_name]['/'.join(path.split('/')[:2]+path.split('/')[3:])] = [round(np.array([HyperVolume.compute(eval_data[i]) for i in range(eval_data.shape[0])]).mean()/scale[dataset_name], 2), round(np.array([HyperVolume.compute(eval_data[i]) for i in range(eval_data.shape[0])]).std()/scale[dataset_name], 2)]
    # table[dataset_name]['/'.join(path.split('/')[:2]+path.split('/')[3:])] = [-round(tmp.mean(), 2), round(tmp.std(), 2)]
    if name not in algo_callnames:
        algo_callnames.append(name)
    if '/'.join(path.split('/')[:2]+path.split('/')[3:]) not in algos:
        algos.append('/'.join(path.split('/')[:2]+path.split('/')[3:]))
    # raw_data = np.load(f"dev/data/raw_rewards/{dataset_name}.npy")
    # dataset = PEDAMuJoCoDataset(
    #     dataset_name, terminal_penalty=0., horizon=32 if "Hopper" in dataset_name or "HalfCheetah" in dataset_name else 4 , avg=True, gamma=1.,
    #     normalize_rewards=True, eps=1e-3, discount=0.997, read_only=True)
    
    # raw_data = dataset.seq_rew[::100, :, :].sum(1)
    # print(round(HyperVolume.compute(eval_data).mean()/scale[dataset_name], 2))

print(table)
# process data
# algo_callnames = list(set(algo_callnames))
# algos = list(set(algos))

print(algo_callnames, algos)
caption = "test"
std = False

width = sum([len(i) for i in use_baselines]) + max([len(j) for (i, j) in table.items()])
height = sum([1 for (i, j) in table.items() if len(j)])
print(width, height)

if std:
    head = "".join(["p{0.1\\textwidth}"*len(i) + "|" for i in use_baselines]) + "p{0.1\\textwidth}"*max([len(j) for (i, j) in table.items()])
else:
    head = "".join(["p{0.08\\textwidth}"*len(i) + "|" for i in use_baselines]) + "p{0.1\\textwidth}"*max([len(j) for (i, j) in table.items()])
env = " & ".join(["Environments"] + [item for sublist in use_baselines for item in sublist] + algo_callnames)

expert, amateur = False, False
for i, j in table.items():
    if len(j) > 0:
        if "expert" in i:
            expert = True
        else:
            amateur = True
flag = amateur and expert

body = ""
cnt = 0
for i, j in table.items():
    if len(j) > 0:
        if cnt > 0:
            body += "    "
        cnt += 1
        if flag:
            body += f"{i.split('-')[1]}-{i.split('_')[-2]} & "
        else:
            body += f"{i.split('-')[1]} & "

        line_max, float_range = -1e9, 0
        for k in use_baselines:
            for l in k:
                if l != "B":
                    line_max = max(line_max, baseline_table[i][l][0])
                    float_range = baseline_table[i][l][1]

        for k in algos:
            if k in j.keys():
                if j[k][0] > line_max:
                    line_max = max(line_max, j[k][0])
                    float_range = j[k][1]
                elif j[k][0] == line_max:
                    float_range = max(float_range, j[k][1])
        
        for k in use_baselines:
            for l in k:
                if std:
                    if baseline_table[i][l][0]+baseline_table[i][l][1] >= line_max-float_range:
                        body += "\\textbf{" + f"{baseline_table[i][l][0]:.2f}±" + '.' + f"{baseline_table[i][l][1]:.2f}".split(".")[1] + "}" + " & "
                    else:
                        body += f"{baseline_table[i][l][0]:.2f}±" + '.' + f"{baseline_table[i][l][1]:.2f}".split(".")[1] + " & "
                else:
                    if baseline_table[i][l][0]+baseline_table[i][l][1] >= line_max-float_range:
                        body += "\\textbf{" + f"{baseline_table[i][l][0]:.2f}" + "}" + " & "
                    else:
                        body += f"{baseline_table[i][l][0]:.2f}" + " & "
        
        for k in algos:
            if k in j.keys():
                if std:
                    if j[k][0]+j[k][1] >= line_max-float_range:
                        body += "\\textbf{" + f"{j[k][0]:.2f}±" + '.' + f"{j[k][1]:.2f}".split(".")[1] + "}"
                    else:
                        body += f"{j[k][0]:.2f}±" + '.' + f"{j[k][1]:.2f}".split(".")[1]
                else:
                    if j[k][0]+j[k][1] >= line_max-float_range:
                        body += "\\textbf{" + f"{j[k][0]:.2f}" + "}"
                    else:
                        body += f"{j[k][0]:.2f}"
            else:
                body += "/"
            if k != algos[-1]:
                body += " & "
            else:
                body += r" \\"
        
        if cnt < height:
            body += "\n"

# print(body)

s = f"""
\\begin{{table}}[H]
  \\caption{{{caption}}}
  \\label{{sample-table}}
  \\Scentering
  \\begin{{tabular}}{{p{{0.25\\textwidth}}||{head}}}
    \\toprule
    {env} \\\\
    \\midrule
    {body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""

print(s)