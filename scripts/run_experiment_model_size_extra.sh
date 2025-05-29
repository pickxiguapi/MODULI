# 11M model
# COUNT=0
# for ALGO in dev/mo_pipelines/mo_diffuser_prefguide.py
# do
#     for DATASET in MO-Ant-v2_50000_amateur_uniform MO-Ant-v2_50000_expert_uniform MO-HalfCheetah-v2_50000_amateur_uniform MO-HalfCheetah-v2_50000_expert_uniform MO-Hopper-v2_50000_amateur_uniform \
#         MO-Hopper-v2_50000_expert_uniform MO-Swimmer-v2_50000_amateur_uniform MO-Swimmer-v2_50000_expert_uniform MO-Walker2d-v2_50000_amateur_uniform MO-Walker2d-v2_50000_expert_uniform
#     do
#         COUNT=$((COUNT+1))
#         echo $COUNT
#         # if count % 2 == 1:
#         if [ $((COUNT % 2)) -eq 1 ]
#         then
#             echo "sleeping"
#             sleep 3
#         fi
#         # nohup python -u $ALGO --save_name model/11M --dataset_name $DATASET --eval_name 2.0  --uniform --device cuda:4 --redirect &
#         # nohup python -u $ALGO --save_name model/11M --dataset_name $DATASET --eval_name 3.0  --uniform --device cuda:4 --redirect &
#         # nohup python -u $ALGO --save_name model/11M --dataset_name $DATASET --eval_name 6.0  --uniform --device cuda:5 --redirect &
#         # nohup python -u $ALGO --save_name model/11M --dataset_name $DATASET --eval_name 10.0 --uniform --device cuda:5 --redirect & wait
#     done
# done

nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Ant-v2_50000_amateur_uniform --uniform --device cuda:0 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Ant-v2_50000_expert_uniform --uniform --device cuda:0 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-HalfCheetah-v2_50000_amateur_uniform --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-HalfCheetah-v2_50000_expert_uniform --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Hopper-v2_50000_amateur_uniform --uniform --device cuda:2 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Hopper-v2_50000_expert_uniform --uniform --device cuda:2 --redirect & wait
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Swimmer-v2_50000_amateur_uniform --uniform --device cuda:0 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Swimmer-v2_50000_expert_uniform --uniform --device cuda:0 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Walker2d-v2_50000_amateur_uniform --uniform --device cuda:1 --redirect &
nohup python -u dev/mo_pipelines/mo_dd.py --save_name model/11M --dataset_name MO-Walker2d-v2_50000_expert_uniform --uniform --device cuda:1 --redirect & wait

#         # nohup python -u $ALGO --save_name model/11M --dataset_name $DATASET --eval_name 0.0  --uniform --device cuda:1 --redirect &
#         # nohup python -u $ALGO --save_name model/11M --dataset_name $DATASET --eval_name 0.1  --uniform --device cuda:1 --redirect &
#         # nohup python -u $ALGO --save_name model/11M --dataset_name $DATASET --eval_name 0.2  --uniform --device cuda:2 --redirect &
#         # nohup python -u $ALGO --save_name model/11M --dataset_name $DATASET --eval_name 0.5  --uniform --device cuda:2 --redirect & 
#         # nohup python -u $ALGO --save_name model/11M --dataset_name $DATASET --eval_name 1.0  --uniform --device cuda:3 --redirect &
#         # nohup python -u $ALGO --save_name model/11M --dataset_name $DATASET --eval_name 1.5  --uniform --device cuda:3 --redirect &