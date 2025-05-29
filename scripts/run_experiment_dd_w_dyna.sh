nohup python -u dev/mo_pipelines/mo_dd_w_dyna.py --save_name model/22M --dataset_name MO-Ant-v2_50000_amateur_uniform          --mode train --uniform --device cuda:6 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_w_dyna.py --save_name model/22M --dataset_name MO-Ant-v2_50000_expert_uniform           --mode train --uniform --device cuda:6 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_w_dyna.py --save_name model/22M --dataset_name MO-HalfCheetah-v2_50000_amateur_uniform  --mode train --uniform --device cuda:7 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_w_dyna.py --save_name model/22M --dataset_name MO-HalfCheetah-v2_50000_expert_uniform   --mode train --uniform --device cuda:7 --redirect & wait
# nohup python -u dev/mo_pipelines/mo_dd_w_dyna.py --save_name model/22M --dataset_name MO-Hopper-v2_50000_amateur_uniform       --mode train --pref_weight 64  --train_name single_point --uniform --device cuda:6 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd_w_dyna.py --save_name model/22M --dataset_name MO-Hopper-v2_50000_expert_uniform        --mode train --pref_weight 64  --train_name single_point --uniform --device cuda:6 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_w_dyna.py --save_name model/22M --dataset_name MO-Swimmer-v2_50000_amateur_uniform      --mode train --uniform --device cuda:6 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_w_dyna.py --save_name model/22M --dataset_name MO-Swimmer-v2_50000_expert_uniform       --mode train --uniform --device cuda:6 --redirect & wait
# nohup python -u dev/mo_pipelines/mo_dd_w_dyna.py --save_name model/22M --dataset_name MO-Walker2d-v2_50000_amateur_uniform     --mode train --pref_weight 64  --train_name single_point --uniform --device cuda:7 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd_w_dyna.py --save_name model/22M --dataset_name MO-Walker2d-v2_50000_expert_uniform      --mode train --pref_weight 64  --train_name single_point --uniform --device cuda:7 --redirect & wait
