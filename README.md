# MODULI: Unlocking Preference Generalization via Diffusion Models for Offline Multi-Objective Reinforcement Learning
Official code for **MODULI: Unlocking Preference Generalization via Diffusion Models for Offline Multi-Objective Reinforcement Learning** (ICML 2025). MODULI leverages diffusion models to enable robust generalization in offline multi-objective RL.

Authors: [Yifu Yuan](https://yifu-yuan.github.io/), [Zhenrui Zheng](https://scholar.google.com/citations?user=KPpd1pYAAAAJ&hl=zh-CN), [Zibin Dong](https://scholar.google.com/citations?user=JQ6881QAAAAJ&hl=zh-CN), [JianYe Hao](http://www.icdai.org/jianye.html)


> 🚧The repository is still under development, codes and manuals will be organized and updated within two weeks. Please stay tuned!🚧


## Setup
#### 1. Environment Setup

```bash
git clone https://github.com/pickxiguapi/MODULI.git
cd MODULI
conda env create -f environment.yml
conda activate MODULI
```
#### 2. Data Download
   
MODULI uses the D4MORL dataset from [PEDA](https://github.com/baitingzbt/PEDA), which can be downloaded as:

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1wfd6BwAu-hNLC9uvsI1WPEOmPpLQVT9k?usp=sharing --output dev/data
```

## Usage
#### 1. Dataset Preparation
Since many of MODULI's experiments are based on variations of original D4MORL dataset, the processing of which is very time and disk space consuming, so MODULI adopts a cache-based dataset preparation mechanism.

Although the repository comes with an pickled empty cache management file (`dev/data/cache/LRUDatasetCache.pkl`), sometimes (when it is corrupted or lost) you may need to manually recreate one:

```python
from dev.utils.utils import LRUDatasetCache

cache = LRUDatasetCache(capacity=12, save_path:="dev/data/cache/")
with open("dev/data/cache/LRUDatasetCache.pkl", "wb") as f:
    pickle.dump(cache, f)
```

Subsequently, each time an experiment pipeline is run, the code will read the cache and determine whether a dataset needs to be reprocessed. The details can be found within any code file in `dev/mo_pipelines`.

#### 2. Train and Evaluate

MODULi's ablation experiments are highly parameterized and are usually run in batches in `.sh` files. You can find the running scripts for most experiments in the `/script` folder (including but not limited to the experiments shown in the paper). Model training and evaluation are usually separate, controlled by the `--mode` parameter.

For example, you can run the following command to train a standard Multi-Objective Decision Diffuser on the `HalfCheetah expert-uniform` dataset like:

```bash
python -u dev/mo_pipelines/mo_dd.py --save_name model/22M --dataset_name MO-HalfCheetah-v2_50000_expert_uniform --device cuda:0 --mode train
```

and then evaluate it:

```bash
python -u dev/mo_pipelines/mo_dd.py --load_model model/22M/200000 --eval_name standard_eval --save_name model/22M --dataset_name MO-HalfCheetah-v2_50000_expert_uniform --device cuda:0 --mode eval
```

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

