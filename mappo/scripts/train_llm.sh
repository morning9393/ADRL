#!/bin/sh

llama="/path/to/Llama-2-7b-hf"
codellama="/path/to/CodeLlama-7b-hf"

CUDA_VISIBLE_DEVICES=4 python -u train_overcooked.py --env_name "Overcooked-LLMA-v4" --algorithm_name "POAD" --experiment_name "test" --num_env_steps 40000 --seed 1 --entropy_coef 0.0001 --gradient_cp_steps 1 --ppo_epoch 5 --num_mini_batch 2 --gamma 0.99 --model_name ${llama}
# CUDA_VISIBLE_DEVICES=0 python -u train_virtualhome.py --env_name "VirtualHome-v1" --algorithm_name "TWOSOME" --experiment_name "test" --num_env_steps 6000 --seed 10 --entropy_coef 0.01 --ppo_epoch 1 --num_mini_batch 4 --gradient_cp_steps 8 --model_name ${llama}
# CUDA_VISIBLE_DEVICES=0 python -u train_datascience.py --env_name "scikit" --algorithm_name "POAD" --experiment_name "test" --dataset_name "balance_scale" --flag "balance_scale_poad_10" --seed 10 --model_name ${codellama}
