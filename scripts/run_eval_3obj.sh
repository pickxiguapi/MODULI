
nohup python -u dev/mo_pipelines/mo_dd_3obj.py --mode eval --save_name eval_standard/22M --load_model model/22M/200000 --dataset_name MO-Hopper-v3_50000_expert_uniform    --device cuda:1 --num_episodes 3 --uniform --redirect &
nohup python -u dev/mo_pipelines/mo_dd_3obj.py --mode eval --save_name eval_standard/22M --load_model model/22M/200000 --dataset_name MO-Hopper-v3_50000_amateur_uniform  --device cuda:1 --num_episodes 3 --uniform --redirect & wait
