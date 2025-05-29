
# for DATASET in MO-Ant-v2_50000_amateur_uniform MO-Ant-v2_50000_expert_uniform MO-HalfCheetah-v2_50000_amateur_uniform MO-HalfCheetah-v2_50000_expert_uniform \
#                MO-Hopper-v2_50000_amateur_uniform MO-Hopper-v2_50000_expert_uniform MO-Swimmer-v2_50000_amateur_uniform MO-Swimmer-v2_50000_expert_uniform \
#                MO-Walker2d-v2_50000_amateur_uniform MO-Walker2d-v2_50000_expert_uniform
# do
#     nohup python -u dev/mo_pipelines/mo_dd_wo_norm.py --save_name model/22M --dataset_name $DATASET --device cuda:3 --mode train --uniform --redirect &
# done

# nohup python -u dev/mo_pipelines/mo_diffuser_prefguide.py --save_name model/22M --dataset_name MO-Ant-v2_50000_amateur_uniform --device cuda:6 --mode train --uniform --redirect &
# nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/22M --dataset_name MO-Ant-v2_50000_expert_uniform  --device cuda:1 --mode train --avg False --normalize_rewards False --uniform --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/22M --dataset_name MO-HalfCheetah-v2_50000_amateur_narrow --device cuda:1 --mode train --uniform --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/22M --dataset_name MO-HalfCheetah-v2_50000_amateur_wide --device cuda:1 --mode train   --uniform --redirect &
sleep 180
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/22M --dataset_name MO-HalfCheetah-v2_50000_expert_narrow --device cuda:2 --mode train  --uniform --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/22M --dataset_name MO-HalfCheetah-v2_50000_expert_wide --device cuda:2 --mode train    --uniform --redirect &
sleep 180
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/22M --dataset_name MO-Hopper-v2_50000_expert_narrow --device cuda:3 --mode train       --uniform --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/22M --dataset_name MO-Hopper-v2_50000_expert_wide --device cuda:3 --mode train         --uniform --redirect &
sleep 180
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/22M --dataset_name MO-Swimmer-v2_50000_expert_narrow --device cuda:4 --mode train      --uniform --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/22M --dataset_name MO-Swimmer-v2_50000_expert_wide --device cuda:4 --mode train        --uniform --redirect &
sleep 180
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/22M --dataset_name MO-Walker2d-v2_50000_expert_narrow --device cuda:5 --mode train     --uniform --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/22M --dataset_name MO-Walker2d-v2_50000_expert_wide --device cuda:5 --mode train       --uniform --redirect & wait