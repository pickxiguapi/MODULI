# MODULI: Unlocking Preference Generalization via Diffusion Models for Offline Multi-Objective Reinforcement Learning (ICML2025)
Official code for **MODULI: Unlocking Preference Generalization via Diffusion Models for Offline Multi-Objective Reinforcement Learning** (ICML 2025). MODULI leverages diffusion models to enable robust generalization in offline multi-objective RL. 

Authors: [Yifu Yuan](https://yifu-yuan.github.io/)*, [Zhenrui Zheng](https://scholar.google.com/citations?user=KPpd1pYAAAAJ&hl=zh-CN)*, [Zibin Dong](https://scholar.google.com/citations?user=JQ6881QAAAAJ&hl=zh-CN), [JianYe Hao](http://www.icdai.org/jianye.html)

## Setup
#### 1. Environment Setup

```bash
git clone https://github.com/pickxiguapi/MODULI.git
cd MODULI
conda create -n MODULI python=3.9 pip
conda activate MODULI
pip install -r requirements.txt
```

These dependencies cover `mo_dd` and `mo_dd_invdyn`; `environment.yml` is the historical full experiment environment. On Linux, install MuJoCo 2.1.0 under `~/.mujoco/mujoco210` and the system packages `build-essential libosmesa6-dev libglew-dev libgl-dev patchelf`. For headless evaluation, set `MUJOCO_PY_FORCE_CPU=1` and add `~/.mujoco/mujoco210/bin` to `LD_LIBRARY_PATH`.
#### 2. Data Download
   
MODULI uses the D4MORL dataset from [PEDA](https://github.com/baitingzbt/PEDA), which can be downloaded as:

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1wfd6BwAu-hNLC9uvsI1WPEOmPpLQVT9k?usp=sharing --output dev/data
```

## Usage
#### 1. Dataset Preparation
Since many of MODULI's experiments are based on variations of original D4MORL dataset, the processing of which is very time and disk space consuming, so MODULI adopts a cache-based dataset preparation mechanism.

The dataset loader creates `dev/data/cache/` automatically and reuses processed datasets without rewriting their payloads. To rebuild a corrupted cache, remove `dev/data/cache/LRUDatasetCache.pkl` before running the pipeline again.

#### 2. Train and Evaluate

MODULi's ablation experiments are highly parameterized and are usually run in batches in `.sh` files. You can find the running scripts for most experiments in the `/script` folder (including but not limited to the experiments shown in the paper). Model training and evaluation are usually separate, controlled by the `--mode` parameter.

From the repository root, first train inverse dynamics, then the standard Multi-Objective Decision Diffuser on `HalfCheetah expert-uniform`:

```bash
python -u -m dev.mo_pipelines.mo_dd_invdyn --dataset_name MO-HalfCheetah-v2_50000_expert_uniform --device cuda:0
python -u -m dev.mo_pipelines.mo_dd --save_name model/22M --dataset_name MO-HalfCheetah-v2_50000_expert_uniform --device cuda:0 --mode train
```

and then evaluate it:

```bash
python -u -m dev.mo_pipelines.mo_dd --load_model model/22M/200000 --eval_name standard_eval --save_name model/22M --dataset_name MO-HalfCheetah-v2_50000_expert_uniform --device cuda:0 --mode eval --uniform
```

For a short pipeline check, use `--n_gradient_steps 1000` for both trainers, `--eval_interval 0` for diffusion training, and `--invdyn 1000 --load_model model/22M/1000 --num_prefs 5 --num_envs 5` for evaluation. Both trainers save their final step. Short runs check execution, not paper-level performance; `--resume` reloads model weights but starts a fresh optimizer.

Information about the meaning of parameters and the parsing rules for strings (e.g., the `--save_name` parameter) can be found in any experiment pipeline code file.

#### 3. Metric Calculation and Visualization

Since each evaluate experiment pipeline only records raw data points (i.e., an array of multi-dimensional returns for test points), metrics are calculated and visualized afterwards. The code for which can be found in `dev/visualize`. The calculation code for some metrics and baseline data are referenced from the PEDA paper's source code, which can be referred to if anything is unclear.

#### 4. Evaluation of baseline algorithms on *incomplete* datasets
We conducted comparative experiments on the *incomplete* dataset for MODULI and baseline algorithms (`MORvS`, `MODT`, etc.). Some baseline algorithms were referenced from the PEDA paper's source code and slightly modified to adapt to the *incomplete* dataset. Code details can be found in `dev/PEDA`.

#### 5. Implementation of `sliding guidance`

We implement the `sliding guidance` from the paper using `Low-Rank Adaptation`. Its implementation relies on a lightweight LoRA library [lora-pytorch](https://github.com/lucidrains/lora-pytorch), which we use to train additional fine-tuning networks for the planner's neural network backbone (a 1D Diffusion Transformer). Coding details can be found in `dev/utils/diffusion_utils.py` and needs to work with the [cleandiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) library.

## Citation
If you find this work useful in your research, please consider citing:

```
@article{yuan2025moduli,
      title={MODULI: Unlocking Preference Generalization via Diffusion Models for Offline Multi-Objective Reinforcement Learning}, 
      author={Yifu Yuan and Zhenrui Zheng and Zibin Dong and Jianye Hao},
      year={2025},
      url={https://arxiv.org/abs/2408.15501}, 
}
```

