import sys
from time import sleep
from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import torch
import torch.nn.functional as F
import numpy as np
# from mappo.utils.util import update_linear_schedule
# from transformers import GemmaForCausalLM, LlamaForCausalLM
import copy
from torch.distributions.categorical import Categorical
import gc
import random
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import os
from peft import PeftModel
from mappo.models.critic import APPOCritic, ETPOCritic, TPPOCritic


class LlamaCritic:

    def __init__(self, model_name, max_new_tokens, algo, load_path=None):
        self.device = "cuda"
        self.algo = algo
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        
        self.base_model = LlamaForCausalLM.from_pretrained(model_name, 
                                                           torch_dtype=torch.float16,
                                                           device_map="auto")
        self.base_model.half().to(self.device)
        
        self.max_new_tokens = max_new_tokens
        
        if load_path is None:
            self.critic = self._init_critic().to(self.device)
        else:
            self.load(load_path)
    
    def _init_critic(self, critic_weights = None):
        if self.algo == "TWOSOME":
            critic = APPOCritic(self.base_model, self.tokenizer)
        elif self.algo in ["POAD", "NTPO"]:
            critic = TPPOCritic(self.base_model, self.tokenizer)
        else:
            raise NotImplementedError
        if critic_weights is not None:
            critic.v_head.load_state_dict(torch.load(critic_weights, map_location= "cpu"))
        return critic
    
    def sample_actions(self, input_ids, seq_token_lengths, act_token_lengths, 
                       action_num_list, action_list):
        assert act_token_lengths.max() <= self.max_new_tokens, \
            f"The length of action tokens {act_token_lengths.max()} exceeds the maximum length {self.max_new_tokens}."
        
        action_token_list = []
        for i in range(len(act_token_lengths)):
            start_idx = seq_token_lengths[i] - act_token_lengths[i]
            end_idx = seq_token_lengths[i]
            token_slice = input_ids[i, start_idx:end_idx]
            action_token_list.append(token_slice)
        
        actions = []
        action_tokens = torch.ones((len(action_num_list), self.max_new_tokens), 
                                   dtype=torch.int64).to("cuda") * self.tokenizer.pad_token_id

        for i in range(len(action_num_list)):
            start = sum(action_num_list[:i])
            end = sum(action_num_list[:i+1])
            act_token_lengths_i = act_token_lengths[start:end]
            action_token_list_i = action_token_list[start:end]
            
            dist_i = Categorical(logits=torch.ones_like(act_token_lengths_i)) # uniform distribution
            action_i_idx = dist_i.sample()
            
            action_i = action_list[i][action_i_idx]
            actions.append(action_i)
            action_token_i = action_token_list_i[action_i_idx]
            action_tokens[i, :act_token_lengths_i[action_i_idx]] = action_token_i

        actions = np.array(actions, dtype=np.object_)
        return actions, action_tokens
    
    def get_actions(self, obs, ava):
        """
        Compute actions and value function predictions for the given inputs.
        """
        prompts = obs.tolist()
        action_list = [act.split(",") for act in ava.tolist()]
        action_num_list = [len(ac) for ac in action_list]
        
        sequences = []
        action_sequences = []
        for p, ac in zip(prompts, action_list):
            sequences += [p + " " + a for a in ac]
            action_sequences += [a for a in ac]  # for llama
        
        token_seq = self.tokenizer(sequences, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        seq_token_lengths = attn_mask.sum(dim=1)
        
        # outputs = self.actor(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
        # token_logits = outputs.logits[:, :-1, :]
        # input_ids = input_ids[: , 1:]  # align logits and ids
        
        act_token_seq = self.tokenizer(action_sequences, return_tensors="pt", padding=True)
        act_attn_mask = act_token_seq["attention_mask"].to("cuda")
        act_token_lengths = act_attn_mask.sum(dim=1) - 1  # ignore the <bos> token
            
        actions, action_tokens = self.sample_actions(input_ids, seq_token_lengths, 
                                                     act_token_lengths, action_num_list, action_list)
        # print("selected action prob: ", action_log_probs.exp().mean())
        
        return actions, action_tokens
        
    def get_action_values(self, obs):
        obs = obs.tolist()
        inputs = self.tokenizer(obs, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        values = self.critic(input_ids, attention_mask=attention_mask)
        # values = values.detach().float().cpu().numpy()
        return values
    
    def get_slice(self, logits, seq_token_lengths, act_token_lengths):
        action_slice = torch.zeros((logits.shape[0], self.max_new_tokens, logits.shape[-1])).to("cuda")
        for i in range(logits.shape[0]):
            start_idx = seq_token_lengths[i] - act_token_lengths[i] - 1
            end_idx = seq_token_lengths[i] - 1
            action_slice[i, :act_token_lengths[i]] = logits[i, start_idx:end_idx]
        return action_slice
    
    def get_token_values(self, obs, actions):
        obs_act = [obs[i] + " " + actions[i] for i in range(len(obs))]
        
        token_seq = self.tokenizer(obs_act, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        seq_token_lengths = attn_mask.sum(dim=1)
        
        act_token_seq = self.tokenizer(actions.tolist(), return_tensors="pt", padding=True)
        act_attn_mask = act_token_seq["attention_mask"].to("cuda")
        act_token_lengths = act_attn_mask.sum(dim=1) - 1  # ignore the <bos> token
        
        values = self.critic(input_ids, attention_mask=attn_mask)
        values = self.get_slice(values, seq_token_lengths, act_token_lengths)
        return values

    @torch.no_grad()
    def infer_for_rollout(self, obs, ava):
        actions, action_tokens = self.get_actions(obs, ava)
        
        if self.algo == "TWOSOME":
            values = self.get_action_values(obs)
            values = values.float().cpu().numpy()
            action_tokens = action_tokens.int().cpu().numpy()
            return actions, action_tokens, values
        elif self.algo in ["POAD", "NTPO"]:
            values = self.get_token_values(obs, actions).squeeze(-1)
            values = values.float().cpu().numpy()
            action_tokens = action_tokens.int().cpu().numpy()
            return actions, action_tokens, values
        else:
            raise NotImplementedError
    
    def get_next_tppo_values(self, obs): 
        token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        token_idx = attn_mask.sum(dim=1) - 1
        
        # values
        values = self.critic(input_ids, attention_mask=attn_mask)
        values = values[torch.arange(obs.shape[0]), token_idx]
        return values
    
    @torch.no_grad()
    def get_next_values(self, obs):
        """
        Get value function predictions.
        """
        if self.algo == "TWOSOME":
            values = self.get_action_values(obs)
            values = values.cpu().float().numpy()
            return values
        elif self.algo in ["POAD", "NTPO"]:
            values = self.get_next_tppo_values(obs).squeeze(-1)
            values = values.cpu().float().numpy()
            return values
        else: 
            raise NotImplementedError

    def save(self, save_dir, episode):
        print("save model")

    def load(self, save_dir):
        print("load model on path: ", save_dir)


