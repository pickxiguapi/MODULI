nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_30 --dataset_name MO-Ant-v2_50000_amateur_uniform          --uniform --device cuda:1 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_30 --dataset_name MO-Ant-v2_50000_expert_uniform           --uniform --device cuda:1 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_30 --dataset_name MO-HalfCheetah-v2_50000_amateur_uniform  --uniform --device cuda:4 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_30 --dataset_name MO-HalfCheetah-v2_50000_expert_uniform   --uniform --device cuda:4 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_30 --dataset_name MO-Hopper-v2_50000_amateur_uniform       --uniform --device cuda:5 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_30 --dataset_name MO-Hopper-v2_50000_expert_uniform        --uniform --device cuda:5 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_30 --dataset_name MO-Swimmer-v2_50000_amateur_uniform      --uniform --device cuda:6 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_30 --dataset_name MO-Swimmer-v2_50000_expert_uniform       --uniform --device cuda:6 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_30 --dataset_name MO-Walker2d-v2_50000_amateur_uniform     --uniform --device cuda:7 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_30 --dataset_name MO-Walker2d-v2_50000_expert_uniform      --uniform --device cuda:7 --redirect --force_override True & wait

nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_S30 --dataset_name MO-Ant-v2_50000_amateur_uniform         --side True  --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_S30 --dataset_name MO-Ant-v2_50000_expert_uniform          --side True  --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_S30 --dataset_name MO-HalfCheetah-v2_50000_amateur_uniform --side True  --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_S30 --dataset_name MO-HalfCheetah-v2_50000_expert_uniform  --side True  --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_S30 --dataset_name MO-Hopper-v2_50000_amateur_uniform      --side True  --uniform --device cuda:5 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_S30 --dataset_name MO-Hopper-v2_50000_expert_uniform       --side True  --uniform --device cuda:5 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_S30 --dataset_name MO-Swimmer-v2_50000_amateur_uniform     --side True  --uniform --device cuda:6 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_S30 --dataset_name MO-Swimmer-v2_50000_expert_uniform      --side True  --uniform --device cuda:6 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_S30 --dataset_name MO-Walker2d-v2_50000_amateur_uniform    --side True  --uniform --device cuda:7 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_shattered.py --save_name model/11M_S30 --dataset_name MO-Walker2d-v2_50000_expert_uniform     --side True  --uniform --device cuda:7 --redirect & wait

nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_30 --dataset_name MO-Ant-v2_50000_amateur_uniform          --uniform --device cuda:1 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_30 --dataset_name MO-Ant-v2_50000_expert_uniform           --uniform --device cuda:1 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_30 --dataset_name MO-HalfCheetah-v2_50000_amateur_uniform  --uniform --device cuda:4 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_30 --dataset_name MO-HalfCheetah-v2_50000_expert_uniform   --uniform --device cuda:4 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_30 --dataset_name MO-Hopper-v2_50000_amateur_uniform       --uniform --device cuda:5 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_30 --dataset_name MO-Hopper-v2_50000_expert_uniform        --uniform --device cuda:5 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_30 --dataset_name MO-Swimmer-v2_50000_amateur_uniform      --uniform --device cuda:6 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_30 --dataset_name MO-Swimmer-v2_50000_expert_uniform       --uniform --device cuda:6 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_30 --dataset_name MO-Walker2d-v2_50000_amateur_uniform     --uniform --device cuda:7 --redirect --force_override True &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_30 --dataset_name MO-Walker2d-v2_50000_expert_uniform      --uniform --device cuda:7 --redirect --force_override True & wait

nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_S30 --dataset_name MO-Ant-v2_50000_amateur_uniform         --side True  --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_S30 --dataset_name MO-Ant-v2_50000_expert_uniform          --side True  --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_S30 --dataset_name MO-HalfCheetah-v2_50000_amateur_uniform --side True  --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_S30 --dataset_name MO-HalfCheetah-v2_50000_expert_uniform  --side True  --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_S30 --dataset_name MO-Hopper-v2_50000_amateur_uniform      --side True  --uniform --device cuda:5 --redirect &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_S30 --dataset_name MO-Hopper-v2_50000_expert_uniform       --side True  --uniform --device cuda:5 --redirect &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_S30 --dataset_name MO-Swimmer-v2_50000_amateur_uniform     --side True  --uniform --device cuda:6 --redirect &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_S30 --dataset_name MO-Swimmer-v2_50000_expert_uniform      --side True  --uniform --device cuda:6 --redirect &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_S30 --dataset_name MO-Walker2d-v2_50000_amateur_uniform    --side True  --uniform --device cuda:7 --redirect &
nohup python -u dev/mo_pipelines/mo_diffuser_prefguide_shattered.py --save_name model/11M_S30 --dataset_name MO-Walker2d-v2_50000_expert_uniform     --side True  --uniform --device cuda:7 --redirect & wait