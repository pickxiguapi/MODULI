
nohup python -u dev/mo_pipelines/mo_dd_3obj.py --save_name model/22M --dataset_name MO-Hopper-v3_50000_expert_wide    --device cuda:1 --mode train --uniform --redirect &
nohup python -u dev/mo_pipelines/mo_dd_3obj.py --save_name model/22M --dataset_name MO-Hopper-v3_50000_expert_narrow  --device cuda:1 --mode train --uniform --redirect & wait
