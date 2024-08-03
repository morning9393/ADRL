#!/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
import random
from pathlib import Path
import torch
sys.path.append("../../")
from mappo.config import get_config
from mappo.envs.virtualhome.virtualhome_env import VirtualHomeEnv
from mappo.agents.llama_lora_agent import LlamaLoRAgent
from mappo.agents.llama_full_agent import LlamaFullAgent


def parse_args(args, parser):
    parser.add_argument('--env_name', type=str, default="VirtualHome-v1", choices=["VirtualHome-v1", "VirtualHome-v2"], help="Which env to run on")
    parser.add_argument('--model_name', type=str, default='', help="Which model to uese")
    parser.add_argument('--peft_path', type=str, default='', help="Which model to uese")
    parser.add_argument('--variant', type=str, default='Cheese', help="Which model to uese")
    parser.add_argument('--max_new_tokens', type=int, default=11, help="max_new_tokens")
    parser.add_argument('--vacab_size', type=int, default=32000)
    parser.add_argument("--use_full_scale", action='store_true', default=False, help="by default False, whether to use full scale model.")
    all_args = parser.parse_known_args(args)[0]

    return all_args

@torch.no_grad()
def eval(agent, eval_envs, threads, episodes):
    eval_episode = 0
    success_count = 0
    eval_episode_rewards = []
    eval_episode_lengths = []

    eval_obs, eval_ava = eval_envs.reset()
    while True:
        eval_actions = agent.get_actions(np.concatenate(eval_obs), np.concatenate(eval_ava), greedy=False)[0]
        eval_actions = np.array(np.split(eval_actions, threads))
        # print("eval_obs: ", eval_obs)
        # print("eval_actions: ", eval_actions)
        eval_obs, eval_rewards, eval_dones, eval_ava, eval_infos = eval_envs.step(eval_actions)

        eval_dones_env = np.all(eval_dones, axis=1)

        for eval_i in range(threads):
            if eval_dones_env[eval_i]:
                eval_episode += 1
                episodic_return = eval_infos[eval_i]["episode"]["r"]
                episodic_length = eval_infos[eval_i]["episode"]["l"]
                print("episodic_return: {}, length: {}".format(episodic_return, episodic_length))
                eval_episode_rewards.append(episodic_return)
                eval_episode_lengths.append(episodic_length)
                if episodic_return > 0:
                    success_count += 1

        if eval_episode >= episodes:
            eval_episode_rewards = np.array(eval_episode_rewards)
            eval_episode_lengths = np.array(eval_episode_lengths)
            print("average eval reward is {}.".format(np.mean(eval_episode_rewards)))   
            print("average eval lengths is {}.".format(np.mean(eval_episode_lengths)))
            print("eval success count is {}, rate is {}.".format(success_count, success_count/episodes))        
            break


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # seed
    random.seed(all_args.seed)
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)  # If you're using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if all_args.use_full_scale:
        agent = LlamaFullAgent(all_args.model_name, all_args.max_new_tokens, "TWOSOME", all_args.peft_path)
    else:
        agent = LlamaLoRAgent(all_args.model_name, all_args.max_new_tokens, "TWOSOME", all_args.peft_path)
    
    # agent = LlamaLoRAgent(all_args.model_name, all_args.max_new_tokens, "APPO")
    eval_envs = VirtualHomeEnv(all_args.env_name, all_args.n_eval_rollout_threads, all_args.seed, variant=all_args.variant)
    # eval_envs = VirtualHomeEnv(all_args.env_name, all_args.n_eval_rollout_threads, all_args.seed)

    eval(agent, eval_envs, all_args.n_eval_rollout_threads, all_args.eval_episodes)

    # post process
    eval_envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])
