import torch
import numpy as np
import torch.nn.functional as F
from mappo.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class LanguageBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    """

    def __init__(self, args, num_agents, pad_token_id):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self.algo = args.algorithm_name
        self.num_agents = num_agents

        self.max_new_tokens = args.max_new_tokens
        self.vacab_size = args.vacab_size
        self.pad_token_id = pad_token_id
        
        # when max_batch = 1, this is an on-policy buffer, otherwise it is a replaybuffer
        self.max_batch = 1
        self.cur_num_batch = 0
        self.cur_batch_index = 0
        self.pre_batch_index = None

        self.obs = np.empty((self.max_batch, self.episode_length + 1, self.n_rollout_threads, num_agents), dtype=np.object_)
        self.available_actions = np.empty_like(self.obs, dtype=np.object_)
        self.actions = np.empty((self.max_batch, self.episode_length, self.n_rollout_threads, num_agents), dtype=np.object_)
        self.action_tokens = np.empty((self.max_batch, self.episode_length, self.n_rollout_threads, num_agents, self.max_new_tokens), dtype=np.int64)
        self.rewards = np.zeros(
            (self.max_batch, self.episode_length, self.n_rollout_threads, num_agents), dtype=np.float32)
        self.masks = np.ones((self.max_batch, self.episode_length + 1, self.n_rollout_threads, num_agents), dtype=np.float32)
        
        # for action-level ppo
        self.action_level_v_values = np.zeros(
            (self.max_batch, self.episode_length + 1, self.n_rollout_threads, num_agents), dtype=np.float32)
        self.action_level_returns = np.zeros(
            (self.max_batch, self.episode_length, self.n_rollout_threads, num_agents), dtype=np.float32)
        self.action_level_advantages = np.zeros_like(self.action_level_returns)
        self.action_level_log_probs = np.zeros_like(self.action_level_returns)
        
        # for token-level ppo
        self.tppo_values = np.zeros(
            (self.max_batch, self.episode_length + 1, self.n_rollout_threads, num_agents, self.max_new_tokens), dtype=np.float32)
        self.tppo_returns = np.zeros(
            (self.max_batch, self.episode_length, self.n_rollout_threads, num_agents, self.max_new_tokens), dtype=np.float32)
        self.tppo_advantages = np.zeros_like(self.tppo_returns)
        self.tppo_log_probs = np.zeros_like(self.tppo_returns)
        
        self.step = 0
   
    
    def insert_appo(self, obs, ava, actions, value_preds, rewards, masks, action_tokens, action_log_probs):
        """
        Insert data into the buffer.
        """
        self.obs[self.cur_batch_index, self.step + 1] = obs.copy()
        self.available_actions[self.cur_batch_index, self.step + 1] = ava.copy()
        self.actions[self.cur_batch_index, self.step] = actions.copy()
        self.rewards[self.cur_batch_index, self.step] = rewards.copy()
        self.masks[self.cur_batch_index, self.step + 1] = masks.copy()
        self.action_tokens[self.cur_batch_index, self.step] = action_tokens.copy()
        self.action_level_v_values[self.cur_batch_index, self.step] = value_preds.copy()
        self.action_level_log_probs[self.cur_batch_index, self.step] = action_log_probs.copy()

        self.step = (self.step + 1) % self.episode_length    
        
    def insert_tppo(self, obs, ava, actions, value_preds, rewards, masks, action_tokens, token_log_probs):
        """
        Insert data into the buffer.
        """
        self.obs[self.cur_batch_index, self.step + 1] = obs.copy()
        self.available_actions[self.cur_batch_index, self.step + 1] = ava.copy()
        self.actions[self.cur_batch_index, self.step] = actions.copy()
        self.rewards[self.cur_batch_index, self.step] = rewards.copy()
        self.masks[self.cur_batch_index, self.step + 1] = masks.copy()
        self.action_tokens[self.cur_batch_index, self.step] = action_tokens.copy()
        self.tppo_values[self.cur_batch_index, self.step] = value_preds.copy()
        self.tppo_log_probs[self.cur_batch_index, self.step] = token_log_probs.copy()

        self.step = (self.step + 1) % self.episode_length  

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.pre_batch_index = self.cur_batch_index
        self.cur_batch_index = (self.cur_batch_index + 1) % self.max_batch
        self.obs[self.cur_batch_index, 0] = self.obs[self.pre_batch_index, -1].copy()
        self.available_actions[self.cur_batch_index, 0] = self.available_actions[self.pre_batch_index, -1].copy()
    
    def get_last_token_position(self, action_tokens):
        pos = len(action_tokens) - 1
        while action_tokens[pos] == self.pad_token_id:
            pos -= 1
        return pos
        
    def batch_process_appo(self, next_value):
        self.action_level_v_values[self.cur_batch_index, -1] = next_value
        gae = 0
        for step in reversed(range(self.episode_length)):
            delta = self.rewards[self.cur_batch_index, step] \
                + self.gamma * self.action_level_v_values[self.cur_batch_index, step + 1] * self.masks[self.cur_batch_index, step + 1] \
                    - self.action_level_v_values[self.cur_batch_index, step]
            gae = delta + self.gamma * self.gae_lambda * self.masks[self.cur_batch_index, step + 1] * gae
            self.action_level_returns[self.cur_batch_index, step] = self.action_level_v_values[self.cur_batch_index, step] + gae
            self.action_level_advantages[self.cur_batch_index, step] = gae
            
        self.cur_num_batch = self.cur_num_batch + 1 if self.cur_num_batch < self.max_batch else self.max_batch
        
    def batch_process_tppo(self, next_value):
        self.tppo_values[self.cur_batch_index, -1, :, :, 0] = next_value
        
        for thread in range(self.n_rollout_threads):
            gae = 0
            for step in reversed(range(self.episode_length)):
                last_token = self.get_last_token_position(self.action_tokens[self.cur_batch_index, step, thread, 0, :])
                for token in reversed(range(last_token + 1)):
                    rew = self.rewards[self.cur_batch_index, step, thread, :]
                    v = self.tppo_values[self.cur_batch_index, step, thread, :, token]
                    if token == last_token:
                        v_next = self.tppo_values[self.cur_batch_index, step + 1, thread, :, 0]
                        mask_next = self.masks[self.cur_batch_index, step + 1, thread, :]
                        delta = rew + self.gamma * v_next * mask_next - v
                        gae = delta + self.gamma * self.gae_lambda * mask_next * gae
                    else:
                        v_next = self.tppo_values[self.cur_batch_index, step, thread, :, token + 1]
                        if self.algo == "POAD":
                            delta = v_next - v
                            gae = delta + self.gae_lambda * gae
                        else:
                            # for NTPO
                            delta = self.gamma * v_next - v
                            gae = delta + self.gamma * self.gae_lambda * gae
                        
                    self.tppo_returns[self.cur_batch_index, step, thread, :, token] = gae + v
                    self.tppo_advantages[self.cur_batch_index, step, thread, :, token] = gae
        self.cur_num_batch = self.cur_num_batch + 1 if self.cur_num_batch < self.max_batch else self.max_batch
                
    def appo_sampler(self, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for APPO.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        batch_size = self.n_rollout_threads * self.episode_length * self.cur_num_batch
        # num_mini_batch is the number of mini batches to split per single batch into thus should multiply cur_num_batch
        num_mini_batch *= self.cur_num_batch

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch
            mini_batch_size = batch_size // num_mini_batch

        # rand = torch.randperm(batch_size).numpy()
        rand = np.arange(batch_size)
        np.random.shuffle(rand)
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        # keep (num_agent, dim)
        obs = self.obs[:, :-1].reshape(-1, *self.obs.shape[3:])
        ava = self.available_actions[:, :-1].reshape(-1, *self.available_actions.shape[3:])
        actions = self.actions.reshape(-1, *self.actions.shape[3:])
        value_preds = self.action_level_v_values[:, :-1].reshape(-1, *self.action_level_v_values.shape[3:])
        returns = self.action_level_returns.reshape(-1, *self.action_level_returns.shape[3:])
        advantages = self.action_level_advantages.reshape(-1, *self.action_level_advantages.shape[3:])
        log_prob = self.action_level_log_probs.reshape(-1, *self.action_level_log_probs.shape[3:])
        action_tokens = self.action_tokens.reshape(-1, *self.action_tokens.shape[3:])

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            # value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            # return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            # o_a_embd_batch = o_a_embds[indices].reshape(-1, *o_a_embds.shape[2:])
            obs_batch = obs[indices]
            ava_batch = ava[indices]
            action_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            advantages_batch = advantages[indices]
            log_prob_batch = log_prob[indices]
            action_tokens_batch = action_tokens[indices]
            yield obs_batch, ava_batch, action_batch, log_prob_batch, value_preds_batch, return_batch, advantages_batch, action_tokens_batch
            
    def tppo_sampler(self, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for TPPO.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        batch_size = self.n_rollout_threads * self.episode_length * self.cur_num_batch
        # num_mini_batch is the number of mini batches to split per single batch into thus should multiply cur_num_batch
        num_mini_batch *= self.cur_num_batch

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch
            mini_batch_size = batch_size // num_mini_batch

        # rand = torch.randperm(batch_size).numpy()
        rand = np.arange(batch_size)
        np.random.shuffle(rand)
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        # keep (num_agent, dim)
        obs = self.obs[:, :-1].reshape(-1, *self.obs.shape[3:])
        ava = self.available_actions[:, :-1].reshape(-1, *self.available_actions.shape[3:])
        actions = self.actions.reshape(-1, *self.actions.shape[3:])
        value_preds = self.tppo_values[:, :-1].reshape(-1, *self.tppo_values.shape[3:])
        returns = self.tppo_returns.reshape(-1, *self.tppo_returns.shape[3:])
        advantages = self.tppo_advantages.reshape(-1, *self.tppo_advantages.shape[3:])
        log_prob = self.tppo_log_probs.reshape(-1, *self.tppo_log_probs.shape[3:])
        action_tokens = self.action_tokens.reshape(-1, *self.action_tokens.shape[3:])

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            # value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            # return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            # o_a_embd_batch = o_a_embds[indices].reshape(-1, *o_a_embds.shape[2:])
            obs_batch = obs[indices]
            ava_batch = ava[indices]
            action_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            advantages_batch = advantages[indices]
            log_prob_batch = log_prob[indices]
            action_tokens_batch = action_tokens[indices]
            yield obs_batch, ava_batch, action_batch, log_prob_batch, value_preds_batch, return_batch, advantages_batch, action_tokens_batch
                