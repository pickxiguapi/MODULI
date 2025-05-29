# 16M model
# for ALGO in dev/mo_pipelines/mo_dd.py
# do
#     for DATASET in MO-Ant-v2_50000_amateur_uniform MO-Ant-v2_50000_expert_uniform MO-HalfCheetah-v2_50000_amateur_uniform MO-HalfCheetah-v2_50000_expert_uniform MO-Hopper-v2_50000_amateur_uniform \
#         MO-Hopper-v2_50000_expert_uniform MO-Swimmer-v2_50000_amateur_uniform MO-Swimmer-v2_50000_expert_uniform MO-Walker2d-v2_50000_amateur_uniform MO-Walker2d-v2_50000_expert_uniform
#     do
#         nohup python -u $ALGO --save_name eval_quick/22M/wcfg --dataset_name $DATASET --load_model model/nheads6_dmodel384_depth8/200000  --n_heads 6 --d_model 384 --depth 8 --mode eval --terminal_penalty -100 --num_envs 20 --num_prefs 100 --num_episodes 1 --w_cfg 0.0  --eval_name 0.0  --uniform --device=cuda:1 --redirect &
#         nohup python -u $ALGO --save_name eval_quick/22M/wcfg --dataset_name $DATASET --load_model model/nheads6_dmodel384_depth8/200000  --n_heads 6 --d_model 384 --depth 8 --mode eval --terminal_penalty -100 --num_envs 20 --num_prefs 100 --num_episodes 1 --w_cfg 0.1  --eval_name 0.1  --uniform --device=cuda:1 --redirect &
#         nohup python -u $ALGO --save_name eval_quick/22M/wcfg --dataset_name $DATASET --load_model model/nheads6_dmodel384_depth8/200000  --n_heads 6 --d_model 384 --depth 8 --mode eval --terminal_penalty -100 --num_envs 20 --num_prefs 100 --num_episodes 1 --w_cfg 0.2  --eval_name 0.2  --uniform --device=cuda:2 --redirect &
#         nohup python -u $ALGO --save_name eval_quick/22M/wcfg --dataset_name $DATASET --load_model model/nheads6_dmodel384_depth8/200000  --n_heads 6 --d_model 384 --depth 8 --mode eval --terminal_penalty -100 --num_envs 20 --num_prefs 100 --num_episodes 1 --w_cfg 0.5  --eval_name 0.5  --uniform --device=cuda:2 --redirect & 
#         nohup python -u $ALGO --save_name eval_quick/22M/wcfg --dataset_name $DATASET --load_model model/nheads6_dmodel384_depth8/200000  --n_heads 6 --d_model 384 --depth 8 --mode eval --terminal_penalty -100 --num_envs 20 --num_prefs 100 --num_episodes 1 --w_cfg 1.0  --eval_name 1.0  --uniform --device=cuda:3 --redirect &
#         nohup python -u $ALGO --save_name eval_quick/22M/wcfg --dataset_name $DATASET --load_model model/nheads6_dmodel384_depth8/200000  --n_heads 6 --d_model 384 --depth 8 --mode eval --terminal_penalty -100 --num_envs 20 --num_prefs 100 --num_episodes 1 --w_cfg 1.5  --eval_name 1.5  --uniform --device=cuda:3 --redirect &
#         nohup python -u $ALGO --save_name eval_quick/22M/wcfg --dataset_name $DATASET --load_model model/nheads6_dmodel384_depth8/200000  --n_heads 6 --d_model 384 --depth 8 --mode eval --terminal_penalty -100 --num_envs 20 --num_prefs 100 --num_episodes 1 --w_cfg 2.0  --eval_name 2.0  --uniform --device=cuda:4 --redirect &
#         nohup python -u $ALGO --save_name eval_quick/22M/wcfg --dataset_name $DATASET --load_model model/nheads6_dmodel384_depth8/200000  --n_heads 6 --d_model 384 --depth 8 --mode eval --terminal_penalty -100 --num_envs 20 --num_prefs 100 --num_episodes 1 --w_cfg 3.0  --eval_name 3.0  --uniform --device=cuda:4 --redirect &
#         nohup python -u $ALGO --save_name eval_quick/22M/wcfg --dataset_name $DATASET --load_model model/nheads6_dmodel384_depth8/200000  --n_heads 6 --d_model 384 --depth 8 --mode eval --terminal_penalty -100 --num_envs 20 --num_prefs 100 --num_episodes 1 --w_cfg 6.0  --eval_name 6.0  --uniform --device=cuda:5 --redirect &
#         nohup python -u $ALGO --save_name eval_quick/22M/wcfg --dataset_name $DATASET --load_model model/nheads6_dmodel384_depth8/200000  --n_heads 6 --d_model 384 --depth 8 --mode eval --terminal_penalty -100 --num_envs 20 --num_prefs 100 --num_episodes 1 --w_cfg 10.0 --eval_name 10.0 --uniform --device=cuda:5 --redirect & wait
#     done
# done

# 16M
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/16M --dataset_name MO-Ant-v2_50000_amateur_uniform         --n_heads 6 --d_model 384 --depth 6 --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/16M --dataset_name MO-Ant-v2_50000_expert_uniform         --n_heads 6 --d_model 384 --depth 6 --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/16M --dataset_name MO-HalfCheetah-v2_50000_amateur_uniform --n_heads 6 --d_model 384 --depth 6 --uniform --device cuda:2 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/16M --dataset_name MO-HalfCheetah-v2_50000_expert_uniform --n_heads 6 --d_model 384 --depth 6 --uniform --device cuda:2 --redirect & wait
# nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/16M --dataset_name MO-Hopper-v2_50000_amateur_uniform      --n_heads 6 --d_model 384 --depth 6 --uniform --device cuda:1 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/16M --dataset_name MO-Hopper-v2_50000_expert_uniform      --n_heads 6 --d_model 384 --depth 6 --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/16M --dataset_name MO-Swimmer-v2_50000_amateur_uniform     --n_heads 6 --d_model 384 --depth 6 --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/16M --dataset_name MO-Swimmer-v2_50000_expert_uniform     --n_heads 6 --d_model 384 --depth 6 --uniform --device cuda:1 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/16M --dataset_name MO-Walker2d-v2_50000_amateur_uniform    --n_heads 6 --d_model 384 --depth 6 --uniform --device cuda:1 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/16M --dataset_name MO-Walker2d-v2_50000_expert_uniform    --n_heads 6 --d_model 384 --depth 6 --uniform --device cuda:1 --redirect &
# 11M
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Ant-v2_50000_amateur_uniform         --n_heads 6 --d_model 384 --depth 4 --uniform --device cuda:2 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Ant-v2_50000_expert_uniform         --n_heads 6 --d_model 384 --depth 4 --uniform --device cuda:2 --redirect & wait
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-HalfCheetah-v2_50000_amateur_uniform --n_heads 6 --d_model 384 --depth 4 --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-HalfCheetah-v2_50000_expert_uniform --n_heads 6 --d_model 384 --depth 4 --uniform --device cuda:1 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Hopper-v2_50000_amateur_uniform      --n_heads 6 --d_model 384 --depth 4 --uniform --device cuda:1 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Hopper-v2_50000_expert_uniform      --n_heads 6 --d_model 384 --depth 4 --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Swimmer-v2_50000_amateur_uniform     --n_heads 6 --d_model 384 --depth 4 --uniform --device cuda:2 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Swimmer-v2_50000_expert_uniform     --n_heads 6 --d_model 384 --depth 4 --uniform --device cuda:2 --redirect & wait
# nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Walker2d-v2_50000_amateur_uniform    --n_heads 6 --d_model 384 --depth 4 --uniform --device cuda:1 --redirect &
# nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Walker2d-v2_50000_expert_uniform    --n_heads 6 --d_model 384 --depth 4 --uniform --device cuda:1 --redirect &