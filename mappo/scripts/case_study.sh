#!/bin/sh

# "Cheese", "Hamburger", "Apple Pie", "Pizza", "Washing Plate", "Laundry"
tokenizer_path="/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/muning/models/Llama-2-7b-hf"
peft_path="./results/test/VirtualHome-v1/POAD/run1/models/episode_0045"

CUDA_VISIBLE_DEVICES=0 python -u train_case_study.py --model_name ${tokenizer_path} --experiment_name "case_study" --algorithm_name "NTPO" --seed 10 --num_env_steps 50000 --ppo_epoch 1 --num_mini_batch 1
CUDA_VISIBLE_DEVICES=0 python -u train_case_study.py --model_name ${tokenizer_path} --experiment_name "case_study" --algorithm_name "TWOSOME" --seed 10 --num_env_steps 50000 --ppo_epoch 1 --num_mini_batch 1
CUDA_VISIBLE_DEVICES=0 python -u train_case_study.py --model_name ${tokenizer_path} --experiment_name "case_study" --algorithm_name "POAD" --seed 10 --num_env_steps 50000 --ppo_epoch 1 --num_mini_batch 1