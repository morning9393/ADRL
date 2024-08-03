#!/bin/sh

# "Cheese", "Hamburger", "Apple Pie", "Pizza", "Washing Plate", "Laundry"
tokenizer_path="/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/muning/models/Llama-2-7b-hf"
peft_path="./results/full_scale/VirtualHome-v1/TWOSOME/run1/models/episode_0047"


CUDA_VISIBLE_DEVICES=1 python -u test_virtualhome.py --env_name "VirtualHome-v1" --model_name ${tokenizer_path} --peft_path ${peft_path} --variant "Laundry" --seed 10 --n_eval_rollout_threads 10 --eval_episodes 100 --use_full_scale
# CUDA_VISIBLE_DEVICES=1 python -u test_virtualhome.py --env_name "VirtualHome-v1" --model_name ${tokenizer_path} --peft_path ${peft_path} --variant "Laundry" --seed 10 --n_eval_rollout_threads 10 --eval_episodes 100