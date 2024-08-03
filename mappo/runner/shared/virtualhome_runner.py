import time
import os
import numpy as np
from functools import reduce
import torch
from tensorboardX import SummaryWriter
from mappo.models.codellama import Llama
from mappo.agents.llama_lora_agent import LlamaLoRAgent
from mappo.agents.llama_full_agent import LlamaFullAgent
from mappo.utils.language_buffer import LanguageBuffer
from mappo.trainers.llm_trainer_appo import APPOTrainer
from mappo.trainers.llm_trainer_tppo import TPPOTrainer
import pickle
from mappo.envs.datascience.prompts.scikit_prompts import *
import json

def _t2n(x):
    return x.detach().cpu().numpy()

def cal_token_mask(action_tokens_batch, pad_token):
    token_mask = (action_tokens_batch != pad_token).astype(np.int64)
    return token_mask

class VirtualHomeRunner:
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        self.num_agents = config['num_agents']
        self.all_args = config['all_args']
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.log_interval = self.all_args.log_interval
        self.algo = self.all_args.algorithm_name
        self.use_full_scale = self.all_args.use_full_scale

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models/')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        if self.use_full_scale:
            self.agent = LlamaFullAgent(self.all_args.model_name, self.all_args.max_new_tokens, self.algo)
        else:
            self.agent = LlamaLoRAgent(self.all_args.model_name, self.all_args.max_new_tokens, self.algo)
        self.buffer = LanguageBuffer(self.all_args, self.num_agents, self.agent.tokenizer.pad_token_id)
        

        if self.algo == "TWOSOME":
            self.trainer = APPOTrainer(self.all_args, self.agent, self.num_agents)
        elif self.algo in ["POAD", "NTPO", "ARCHER"]:
            self.trainer = TPPOTrainer(self.all_args, self.agent, self.num_agents)
        else:
            raise NotImplementedError
        
        self.trajectories = None
        

    def run(self):
        
        obs, ava = self.envs.reset()
        self.buffer.obs[self.buffer.cur_batch_index, 0] = obs.copy()
        self.buffer.available_actions[self.buffer.cur_batch_index, 0] = ava.copy()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        total_num_steps = 0
        for episode in range(episodes):
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_tokens, log_probs = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, ava, infos = self.envs.step(actions)
                
                for i in range(self.n_rollout_threads):
                    if "episode" in infos[i].keys():
                        global_step = total_num_steps + step * self.n_rollout_threads + i
                        print(f"global_step={global_step}, episodic_return={infos[i]['episode']['r']}, episodic_length={infos[i]['episode']['l']}")
                        self.writter.add_scalar("charts/episodic_return", infos[i]["episode"]["r"], global_step)
                        self.writter.add_scalar("charts/episodic_length", infos[i]["episode"]["l"], global_step)
                        break
                
                # insert data into buffer
                data = obs, rewards, dones, ava, values, \
                       actions, action_tokens, log_probs
                self.insert(data)
                
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # compute return and update network
            self.before_update()
            # self.trainer.prep_training()
            train_infos = self.trainer.train(self.buffer)      
            self.buffer.after_update()
            
            # save model
            if (episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                print("total_num_steps: ", total_num_steps)
                self.log_train(train_infos, total_num_steps)
        

    @torch.no_grad()
    def collect(self, step):
        # self.trainer.prep_rollout()
        
        behaviour_data = self.agent.infer_for_rollout(np.concatenate(self.buffer.obs[self.buffer.cur_batch_index, step]),
                                                np.concatenate(self.buffer.available_actions[self.buffer.cur_batch_index, step]))
        actions, action_tokens, values, log_probs = behaviour_data
        
        # [self.envs, agents]
        values = np.array(np.split(values, self.n_rollout_threads))
        actions = np.array(np.split(actions, self.n_rollout_threads))
        action_tokens = np.array(np.split(action_tokens, self.n_rollout_threads))
        log_probs = np.array(np.split(log_probs, self.n_rollout_threads))
            
        return values, actions, action_tokens, log_probs

    def insert(self, data):
        obs, rewards, dones, ava, values, actions, action_tokens, log_probs = data
            
        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents), dtype=np.float32)
        
        if self.algo == "TWOSOME":
            self.buffer.insert_appo(obs, ava, actions, values, rewards, masks, action_tokens, log_probs)
        elif self.algo in ["POAD", "NTPO", "ARCHER"]:
            self.buffer.insert_tppo(obs, ava, actions, values, rewards, masks, action_tokens, log_probs)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def before_update(self):
        """Calculate returns for the collected data."""
        if self.algo == "TWOSOME":
            next_values = self.agent.get_next_values(np.concatenate(self.buffer.obs[self.buffer.cur_batch_index, -1]))
            next_values = np.array(np.split(next_values, self.n_rollout_threads))
            self.buffer.batch_process_appo(next_values)
        elif self.algo in ["POAD", "NTPO", "ARCHER"]:
            next_values = self.agent.get_next_values(np.concatenate(self.buffer.obs[self.buffer.cur_batch_index, -1]))
            next_values = np.array(np.split(next_values, self.n_rollout_threads))
            self.buffer.batch_process_tppo(next_values)
        else:
            raise NotImplementedError

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards[self.buffer.cur_batch_index])
        for k, v in train_infos.items():
            # print("k: ", k, ", v: ", v)
            self.writter.add_scalars(k, {k: v}, total_num_steps)
                
    def save(self, episode):
        """Save policy's actor and critic networks."""
        self.agent.save(self.save_dir, episode)


