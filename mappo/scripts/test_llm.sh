#!/bin/sh

# "Cheese", "Hamburger", "Apple Pie", "Pizza", "Washing Plate", "Laundry"
peft_path="./path/to/lora/model"

CUDA_VISIBLE_DEVICES=0 python -u test_virtualhome.py --env_name "VirtualHome-v1" --peft_path ${peft_path} --variant "Washing Plate" --seed 10 --n_eval_rollout_threads 10 --eval_episodes 100