nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --dataset_name MO-Hopper-v2_50000_expert_uniform --device cuda:6 --mode train --uniform --redirect &
nohup python -u dev/mo_pipelines/mo_dd_shattered_3obj.py --dataset_name MO-Hopper-v3_50000_expert_uniform    --device cuda:6 --mode eval --uniform --redirect &
nohup python -u dev/mo_pipelines/mo_dd_shattered_3obj.py --save_name model/11M --dataset_name MO-Hopper-v3_50000_amateur_uniform   --device cuda:1 --mode train --uniform --redirect & wait
