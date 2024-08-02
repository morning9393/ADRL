#!/bin/sh

# "Cheese", "Hamburger", "Apple Pie", "Pizza", "Washing Plate", "Laundry"
tokenizer_path="/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/muning/models/Llama-2-7b-hf"
peft_path="./results/test/VirtualHome-v1/POAD/run1/models/episode_0045"

CUDA_VISIBLE_DEVICES=0 python -u test_virtualhome.py --env_name "VirtualHome-v1" --model_name ${tokenizer_path} --peft_path ${peft_path} --variant "Laundry" --seed 10 --n_eval_rollout_threads 10 --eval_episodes 100