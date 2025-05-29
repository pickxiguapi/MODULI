nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H8 --horizon 8 --dataset_name MO-Ant-v2_50000_amateur_uniform --uniform --device cuda:3 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H8 --horizon 8 --dataset_name MO-Ant-v2_50000_expert_uniform --uniform --device cuda:3 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H8 --horizon 8 --dataset_name MO-HalfCheetah-v2_50000_amateur_uniform --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H8 --horizon 8 --dataset_name MO-HalfCheetah-v2_50000_expert_uniform --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H8 --horizon 8 --dataset_name MO-Swimmer-v2_50000_amateur_uniform --uniform --device cuda:5 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H8 --horizon 8 --dataset_name MO-Swimmer-v2_50000_expert_uniform --uniform --device cuda:5 --redirect & wait

nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H16 --horizon 16 --dataset_name MO-Ant-v2_50000_amateur_uniform --uniform --device cuda:3 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H16 --horizon 16 --dataset_name MO-Ant-v2_50000_expert_uniform --uniform --device cuda:3 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H16 --horizon 16 --dataset_name MO-HalfCheetah-v2_50000_amateur_uniform --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H16 --horizon 16 --dataset_name MO-HalfCheetah-v2_50000_expert_uniform --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H16 --horizon 16 --dataset_name MO-Swimmer-v2_50000_amateur_uniform --uniform --device cuda:5 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H16 --horizon 16 --dataset_name MO-Swimmer-v2_50000_expert_uniform --uniform --device cuda:5 --redirect & wait

nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H32 --horizon 32 --dataset_name MO-Ant-v2_50000_amateur_uniform --uniform --device cuda:3 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H32 --horizon 32 --dataset_name MO-Ant-v2_50000_expert_uniform --uniform --device cuda:3 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H32 --horizon 32 --dataset_name MO-HalfCheetah-v2_50000_amateur_uniform --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H32 --horizon 32 --dataset_name MO-HalfCheetah-v2_50000_expert_uniform --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H32 --horizon 32 --dataset_name MO-Swimmer-v2_50000_amateur_uniform --uniform --device cuda:5 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H32 --horizon 32 --dataset_name MO-Swimmer-v2_50000_expert_uniform --uniform --device cuda:5 --redirect & wait

nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H4 --horizon 4 --dataset_name MO-Hopper-v2_50000_amateur_uniform --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H4 --horizon 4 --dataset_name MO-Hopper-v2_50000_expert_uniform --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H4 --horizon 4 --dataset_name MO-Walker2d-v2_50000_amateur_uniform --uniform --device cuda:5 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H4 --horizon 4 --dataset_name MO-Walker2d-v2_50000_expert_uniform --uniform --device cuda:5 --redirect & wait

nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H8 --horizon 8 --dataset_name MO-Hopper-v2_50000_amateur_uniform --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H8 --horizon 8 --dataset_name MO-Hopper-v2_50000_expert_uniform --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H8 --horizon 8 --dataset_name MO-Walker2d-v2_50000_amateur_uniform --uniform --device cuda:5 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H8 --horizon 8 --dataset_name MO-Walker2d-v2_50000_expert_uniform --uniform --device cuda:5 --redirect & wait

nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H16 --horizon 16 --dataset_name MO-Hopper-v2_50000_amateur_uniform --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H16 --horizon 16 --dataset_name MO-Hopper-v2_50000_expert_uniform --uniform --device cuda:4 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H16 --horizon 16 --dataset_name MO-Walker2d-v2_50000_amateur_uniform --uniform --device cuda:5 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M_H16 --horizon 16 --dataset_name MO-Walker2d-v2_50000_expert_uniform --uniform --device cuda:5 --redirect & wait