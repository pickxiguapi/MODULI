# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-Ant-v2_50000_amateur_uniform             --mode train --uniform --device cuda:0 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-Ant-v2_50000_expert_uniform              --mode train --uniform --device cuda:0 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-Hopper-v2_50000_amateur_uniform          --mode train --uniform --device cuda:1 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-Hopper-v2_50000_expert_uniform           --mode train --uniform --device cuda:1 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-Swimmer-v2_50000_amateur_uniform         --mode train --uniform --device cuda:2 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-Swimmer-v2_50000_expert_uniform          --mode train --uniform --device cuda:2 --redirect &
# wait
nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name eval/11M --eval_name standard --dataset_name MO-Ant-v2_50000_amateur_uniform             --load_model model/11M/200000 --mode eval --uniform --device cuda:0 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name eval/11M --eval_name standard --dataset_name MO-Ant-v2_50000_expert_uniform              --load_model model/11M/200000 --mode eval --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name eval/11M --eval_name standard --dataset_name MO-Hopper-v2_50000_amateur_uniform          --load_model model/11M/200000 --mode eval --uniform --device cuda:2 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name eval/11M --eval_name standard --dataset_name MO-Hopper-v2_50000_expert_uniform           --load_model model/11M/200000 --mode eval --uniform --device cuda:3 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name eval/11M --eval_name standard --dataset_name MO-Swimmer-v2_50000_amateur_uniform         --load_model model/11M/200000 --mode eval --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name eval/11M --eval_name standard --dataset_name MO-Swimmer-v2_50000_expert_uniform          --load_model model/11M/200000 --mode eval --uniform --device cuda:5 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-HalfCheetah-v2_50000_amateur_uniform  --mode train --uniform --device cuda:7 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-HalfCheetah-v2_50000_expert_uniform   --mode train --uniform --device cuda:7 --redirect & wait
# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-Hopper-v2_50000_amateur_uniform       --mode train --pref_weight 64  --train_name single_point --uniform --device cuda:6 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-Hopper-v2_50000_expert_uniform        --mode train --pref_weight 64  --train_name single_point --uniform --device cuda:6 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-Swimmer-v2_50000_amateur_uniform      --mode train --uniform --device cuda:6 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-Swimmer-v2_50000_expert_uniform       --mode train --uniform --device cuda:6 --redirect & wait
# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-Walker2d-v2_50000_amateur_uniform     --mode train --pref_weight 64  --train_name single_point --uniform --device cuda:7 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd_wo_invdyn.py --save_name model/11M --dataset_name MO-Walker2d-v2_50000_expert_uniform      --mode train --pref_weight 64  --train_name single_point --uniform --device cuda:7 --redirect & wait
