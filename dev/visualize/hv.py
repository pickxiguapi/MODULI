import numpy as np
from dev.dataset.peda_mujoco_dataset import PEDAMuJoCoDataset

def hv(data: np.ndarray):
    """
    Calculate hypervolume of a given data.
    """
    sorted_idx = np.argsort(data[:, 0])
    data = data[sorted_idx]
    num = data.shape[0]
    print(num)
    front = np.ones(num)
    for i in range(num):
        for j in range(num):
            if i != j and np.all(data[i] < data[j]):
                front[i] = 0
                break
    hv = 0.
    x = 0.
    for i in range(num):
        if front[i] == 1:
            hv += data[i, 1] * (data[i, 0] - x)
            x = data[i, 0]
    
    return hv

set = [
    # "results/mo_dd/MO-Ant-v2_50000_amateur_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    # "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    # "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    # "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    # "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    # "results/mo_dd/MO-Ant-v2_50000_expert_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    # "results/mo_dd/MO-HalfCheetah-v2_50000_expert_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    # "results/mo_dd/MO-Hopper-v2_50000_expert_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    # "results/mo_dd/MO-Swimmer-v2_50000_expert_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    # "results/mo_dd/MO-Walker2d-v2_50000_expert_uniform/model_size/head12_dim480_depth8//eval_500000.npy",

    "results/mo_dd_sepcond/MO-Ant-v2_50000_amateur_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    "results/mo_dd_sepcond/MO-HalfCheetah-v2_50000_amateur_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    "results/mo_dd_sepcond/MO-Hopper-v2_50000_amateur_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    "results/mo_dd_sepcond/MO-Swimmer-v2_50000_amateur_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    "results/mo_dd_sepcond/MO-Walker2d-v2_50000_amateur_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    "results/mo_dd_sepcond/MO-Ant-v2_50000_expert_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    "results/mo_dd_sepcond/MO-HalfCheetah-v2_50000_expert_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    "results/mo_dd_sepcond/MO-Hopper-v2_50000_expert_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    "results/mo_dd_sepcond/MO-Swimmer-v2_50000_expert_uniform/model_size/head12_dim480_depth8/eval_500000.npy",
    "results/mo_dd_sepcond/MO-Walker2d-v2_50000_expert_uniform/model_size/head12_dim480_depth8//eval_500000.npy",
]

# set = [
#     "results/mo_dd/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_dd/MO-Hopper-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_dd/MO-Swimmer-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_dd/MO-Walker2d-v2_50000_amateur_uniform/standard/eval_500000.npy",
# ]

# set = [
#     "results/mo_diffuser/MO-HalfCheetah-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_diffuser/MO-Hopper-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_diffuser/MO-Swimmer-v2_50000_amateur_uniform/standard/eval_500000.npy",
#     "results/mo_diffuser/MO-Walker2d-v2_50000_amateur_uniform/standard/eval_500000.npy",
# ]

for i, path in enumerate(set):
    eval_data = np.load(path)
    dataset_name = path.split("/")[2]
    # raw_data = np.load(f"dev/data/raw_rewards/{dataset_name}.npy")
    # dataset = PEDAMuJoCoDataset(
    #     dataset_name, terminal_penalty=0., horizon=32 if "Hopper" in dataset_name or "HalfCheetah" in dataset_name else 4 , avg=True, gamma=1.,
    #     normalize_rewards=True, eps=1e-3, discount=0.997, read_only=True)
    
    # raw_data = dataset.seq_rew[::100, :, :].sum(1)
    print(dataset_name, hv(eval_data))
    