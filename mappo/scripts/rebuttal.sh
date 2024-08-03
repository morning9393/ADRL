#!/bin/sh

llama="/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/muning/models/Llama-2-7b-hf"
codellama="/path/to/CodeLlama-7b-hf"

# CUDA_VISIBLE_DEVICES=0 python -u train_virtualhome.py --env_name "VirtualHome-v1" --algorithm_name "POAD" --experiment_name "full_scale" --num_env_steps 6200 --seed 10 --lr 2e-7 --entropy_coef 0.001 --ppo_epoch 5 --num_mini_batch 2 --gradient_cp_steps 16 --model_name ${llama} --use_full_scale
# CUDA_VISIBLE_DEVICES=0 python -u train_virtualhome.py --env_name "VirtualHome-v1" --algorithm_name "POAD" --experiment_name "full_scale" --num_env_steps 6200 --seed 20 --lr 2e-7 --entropy_coef 0.001 --ppo_epoch 5 --num_mini_batch 2 --gradient_cp_steps 16 --model_name ${llama} --use_full_scale
# CUDA_VISIBLE_DEVICES=0 python -u train_virtualhome.py --env_name "VirtualHome-v1" --algorithm_name "POAD" --experiment_name "full_scale" --num_env_steps 6200 --seed 30 --lr 2e-7 --entropy_coef 0.001 --ppo_epoch 5 --num_mini_batch 2 --gradient_cp_steps 16 --model_name ${llama} --use_full_scale
# CUDA_VISIBLE_DEVICES=0 python -u train_virtualhome.py --env_name "VirtualHome-v1" --algorithm_name "POAD" --experiment_name "test" --num_env_steps 6000 --seed 10 --entropy_coef 0.0001 --ppo_epoch 5 --num_mini_batch 2 --gradient_cp_steps 16 --model_name ${llama}
CUDA_VISIBLE_DEVICES=0 python -u train_virtualhome.py --env_name "VirtualHome-v1" --algorithm_name "ARCHER" --experiment_name "archer" --num_env_steps 6200 --seed 10 --entropy_coef 0.0001 --ppo_epoch 5 --num_mini_batch 2 --gradient_cp_steps 1 --model_name ${llama}

