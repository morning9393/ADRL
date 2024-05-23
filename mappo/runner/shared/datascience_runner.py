import time
import os
import numpy as np
from functools import reduce
import torch
from tensorboardX import SummaryWriter
from mappo.models.codellama import Llama
from mappo.agents.llama_lora_code_agent import LlamaLoRAgent
from mappo.utils.code_buffer import CodeBuffer
from mappo.trainers.llm_trainer_tppo import TPPOTrainer
import pickle
from mappo.envs.datascience.prompts.scikit_prompts import *

def _t2n(x):
    return x.detach().cpu().numpy()

class DataScienceRunner:
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        self.num_agents = config['num_agents']
        self.all_args = config['all_args']
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.log_interval = self.all_args.log_interval
        self.eval_interval = self.all_args.eval_interval
        self.algo = self.all_args.algorithm_name

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
        self.agent = LlamaLoRAgent(self.all_args.model_name, self.all_args.max_new_tokens, self.algo)
        self.buffer = CodeBuffer(self.all_args, self.num_agents, self.agent.tokenizer.pad_token_id)
        self.trainer = TPPOTrainer(self.all_args, self.agent, self.num_agents)


        self.best_reward = -np.inf
        self.best_std = -1.0
        self.best_action = None
        self.stds = []

    def run(self):
        obs = self.envs.reset()
        self.buffer.obs[self.buffer.cur_batch_index, 0] = obs.copy()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads  
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_tokens, log_probs = self.collect(step)
                
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)
                
                # insert data into buffer
                data = obs, rewards, dones, values, \
                       actions, action_tokens, log_probs
                self.insert(data)

                squeezed_reward = np.squeeze(rewards, axis=-1)
                best_index = np.argmax(squeezed_reward)
                if squeezed_reward[best_index] > self.best_reward:
                    self.best_reward = squeezed_reward[best_index]
                    self.best_std = infos[best_index][0]["std"]
                    self.best_action = np.squeeze(actions, axis=-1)[best_index]
                    self.log_code(total_num_steps)
                    
                mean_std = np.mean([info[0]["std"] for info in infos])
                self.stds.append(mean_std)

            # compute return and update network
            self.before_update()
            train_infos = self.trainer.train(self.buffer)      
            self.buffer.after_update()

            # log information
            if episode % self.log_interval == 0:
                print("total_num_steps: ", total_num_steps)
                print("average_step_rewards: ", np.mean(self.buffer.rewards[self.buffer.pre_batch_index]))
                self.log_train(train_infos, total_num_steps)
                self.stds = []

        

    @torch.no_grad()
    def collect(self, step):
        behaviour_data = self.agent.infer_for_rollout(np.concatenate(self.buffer.obs[self.buffer.cur_batch_index, step]))
        
        actions, action_tokens, values, log_probs = behaviour_data
        # [self.envs, agents]
        values = np.array(np.split(values, self.n_rollout_threads))
        actions = np.array(np.split(actions, self.n_rollout_threads))
        action_tokens = np.array(np.split(action_tokens, self.n_rollout_threads))
        log_probs = np.array(np.split(log_probs, self.n_rollout_threads))

        return values, actions, action_tokens, log_probs

    def insert(self,data):
        obs, rewards, dones, values, actions, action_tokens, log_probs = data

        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents), dtype=np.float32)
        self.buffer.insert_tppo(obs, actions, values, rewards, masks, action_tokens, log_probs)


    @torch.no_grad()
    def before_update(self):
        """Calculate returns for the collected data."""
        next_values = self.agent.get_next_values(np.concatenate(self.buffer.obs[self.buffer.cur_batch_index, -1]))
        next_values = np.array(np.split(next_values, self.n_rollout_threads))
        self.buffer.batch_process_tppo(next_values)


    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards[self.buffer.cur_batch_index])
        train_infos["best_step_rewards"] = self.best_reward
        train_infos["best_step_std"] = self.best_std
        train_infos["average_step_std"] = np.mean(self.stds)
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_code(self, total_num_steps):
        best_code = LOG_CODE.format(reward=self.best_reward, step=total_num_steps,
                                    std=self.best_std, action=self.best_action)
        log_code_file = self.log_dir + "/best_code.txt"
        with open(log_code_file, "a") as f:
            f.write(best_code)


