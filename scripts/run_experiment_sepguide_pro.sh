nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_rtg.py --save_name model/11M --dataset_name MO-Ant-v2_50000_amateur_uniform         --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_rtg.py --save_name model/11M --dataset_name MO-Ant-v2_50000_expert_uniform          --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_rtg.py --save_name model/11M --dataset_name MO-HalfCheetah-v2_50000_amateur_uniform --uniform --device cuda:2 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_rtg.py --save_name model/11M --dataset_name MO-HalfCheetah-v2_50000_expert_uniform  --uniform --device cuda:2 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_rtg.py --save_name model/11M --dataset_name MO-Hopper-v2_50000_amateur_uniform      --uniform --device cuda:3 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_rtg.py --save_name model/11M --dataset_name MO-Hopper-v2_50000_expert_uniform       --uniform --device cuda:3 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_rtg.py --save_name model/11M --dataset_name MO-Swimmer-v2_50000_amateur_uniform     --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_rtg.py --save_name model/11M --dataset_name MO-Swimmer-v2_50000_expert_uniform      --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_rtg.py --save_name model/11M --dataset_name MO-Walker2d-v2_50000_amateur_uniform    --uniform --device cuda:5 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_sepguide_pro_rtg.py --save_name model/11M --dataset_name MO-Walker2d-v2_50000_expert_uniform     --uniform --device cuda:5 --redirect & wait