import sys
from time import sleep
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
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
from mappo.models.critic import TPPOCritic
from mappo.envs.datascience.prompts.scikit_prompts import *


class LlamaLoRAgent:

    def __init__(self, model_name, max_new_tokens, algo, load_path=None):
        self.device = "cuda"
        self.algo = algo
        # self.tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side='left')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                           torch_dtype=torch.float16,
                                                           device_map="auto")
        self.base_model.half().to(self.device)
        
        # self.device = next(self.generator.parameters()).device
        self.max_new_tokens = max_new_tokens
        
        if load_path is None:
            self.actor = self._init_actor().to(self.device)
            self.critic = self._init_critic().to(self.device)
        else:
            self.load(load_path)
    
    
    def _init_actor(self, lora_weights = None):
        if lora_weights is None:
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj",],
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(self.base_model, config)

            model.print_trainable_parameters()

            old_state_dict = model.state_dict
            model.state_dict = (
                lambda self, *_, **__: get_peft_model_state_dict(
                    self, old_state_dict()
                )
            ).__get__(model, type(model))
        else:
            model = PeftModel.from_pretrained(
                self.base_model,
                lora_weights,
                torch_dtype=torch.float16,
            )

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)
        
        model.half()

        return model
    
    def _init_critic(self, critic_weights = None):
        
        critic = TPPOCritic(self.actor, self.tokenizer)

        if critic_weights is not None:
            critic.v_head.load_state_dict(torch.load(critic_weights, map_location= "cpu"))
        return critic
    
    def sample_actions(self, input_ids, token_logits, seq_token_lengths, act_token_lengths, 
                       action_num_list, action_list, action_ids=None):
        assert act_token_lengths.max() <= self.max_new_tokens, \
            f"The length of action tokens {act_token_lengths.max()} exceeds the maximum length {self.max_new_tokens}."
            
        pi_log_softmax = torch.log_softmax(token_logits, dim=-1)
        
        action_logits = []
        action_token_list = []
        flatten_action_list = np.array([a for ac in action_list for a in ac], dtype=np.object_)
        for i in range(len(act_token_lengths)):
            start_idx = seq_token_lengths[i] - act_token_lengths[i] - 1
            end_idx = seq_token_lengths[i] - 1
            logit_slice = pi_log_softmax[i, start_idx:end_idx, :]
            token_slice = input_ids[i, start_idx:end_idx]
            action_token_list.append(token_slice)
            
            act_logit_seq = torch.gather(logit_slice, 1, token_slice[:, None]).squeeze(-1)
            # action_log_softmax = act_logit_seq.sum() / act_token_lengths[i]  # token normalization
            action_word_length = len(flatten_action_list[i].split())
            action_log_softmax = act_logit_seq.sum() / action_word_length  # word normalization
            action_logits.append(action_log_softmax)
        action_logits = torch.stack(action_logits)
        
        actions = []
        action_tokens = torch.ones((len(action_num_list), self.max_new_tokens), 
                                   dtype=torch.int64).to("cuda") * self.tokenizer.pad_token_id
        action_log_probs = []
        entropies = []
        for i in range(len(action_num_list)):
            start = sum(action_num_list[:i])
            end = sum(action_num_list[:i+1])
            act_token_lengths_i = act_token_lengths[start:end]
            action_logits_i = action_logits[start:end]
            action_token_list_i = action_token_list[start:end]
            
            dist_i = Categorical(logits=action_logits_i)
            if action_ids is None:
                action_i_idx = dist_i.sample()  # for rollout
            else:
                action_i_idx = action_ids[i]  # for training
            # print("selected action probs: ", dist_i.probs[action_i_idx])
            
            action_i = action_list[i][action_i_idx]
            actions.append(action_i)
            action_token_i = action_token_list_i[action_i_idx]
            action_tokens[i, :act_token_lengths_i[action_i_idx]] = action_token_i
            action_log_probs.append(dist_i.log_prob(action_i_idx))
            entropies.append(dist_i.entropy())
        
        action_log_probs = torch.stack(action_log_probs)
        entropies = torch.stack(entropies)
        actions = np.array(actions, dtype=np.object_)
        return actions, action_tokens, action_log_probs, entropies
    
    def get_actions(self, obs, actions=None, greedy=False):
        """
        Compute actions and value function predictions for the given inputs.
        """
        prompts = obs.tolist()
        token_seq = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        if not greedy:
            output = self.actor.generate(
                input_ids,
                attention_mask=attn_mask,
                do_sample=True,
                top_k=50,
                temperature=0.5,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
        else:
            output = self.actor.generate(
                input_ids,
                attention_mask=attn_mask,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
        sequences = output.sequences
        
        actions = []
        action_tokens = torch.ones((sequences.shape[0], self.max_new_tokens), 
                                   dtype=torch.int64).to("cuda") * self.tokenizer.pad_token_id
        for i in range(sequences.shape[0]):
            action_token = sequences[i][input_ids[i].shape[0]:]
            action_tokens[i, :action_token.shape[0]] = action_token
            action = self.tokenizer.decode(action_token, skip_special_tokens=True)
            actions.append(action)
        actions = np.array(actions, dtype=np.object_)
        
        return actions, action_tokens
        
    def get_action_values(self, obs):
        obs = obs.tolist()
        inputs = self.tokenizer(obs, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with self.actor.disable_adapter():
            values = self.critic(input_ids, attention_mask=attention_mask)
        # values = values.detach().float().cpu().numpy()
        return values
    
    def get_slice(self, logits, seq_token_lengths, obs_token_lengths):
        act_token_lengths = seq_token_lengths - obs_token_lengths
        
        action_slice = torch.zeros((logits.shape[0], self.max_new_tokens, logits.shape[-1])).to("cuda")
        for i in range(logits.shape[0]):
            start_idx = -act_token_lengths[i] - 1
            end_idx = -1 # the last one is invalid
            action_slice[i, :act_token_lengths[i]] = logits[i, start_idx:end_idx]
        return action_slice
    
    def get_token_values(self, obs, actions):
        obs_act = [obs[i] + actions[i] for i in range(len(obs))]
        
        token_seq = self.tokenizer(obs_act, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        seq_token_lengths = attn_mask.sum(dim=1)
        
        obs_token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        obs_attn_mask = obs_token_seq["attention_mask"].to("cuda")
        obs_token_lengths = obs_attn_mask.sum(dim=1)
        
        with self.actor.disable_adapter():
            values = self.critic(input_ids, attention_mask=attn_mask)
        values = self.get_slice(values, seq_token_lengths, obs_token_lengths)
        return values
    
    def get_token_logits(self, obs, actions):
        obs_act = [obs[i] + actions[i] for i in range(len(obs))]
        
        token_seq = self.tokenizer(obs_act, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        seq_token_lengths = attn_mask.sum(dim=1)
        
        obs_token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        obs_attn_mask = obs_token_seq["attention_mask"].to("cuda")
        obs_token_lengths = obs_attn_mask.sum(dim=1)
        
        with self.actor.disable_adapter():
            rho_outputs = self.actor(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
            rho_logits = self.get_slice(rho_outputs.logits, seq_token_lengths, obs_token_lengths)
            
        pi_outputs = self.actor(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
        pi_logits = self.get_slice(pi_outputs.logits, seq_token_lengths, obs_token_lengths)
        
        return pi_logits, rho_logits
    
    def kl_cal(self, pi_logits, rho_logits, values, action_tokens=None):
        pi = F.softmax(pi_logits, dim=-1)
        rho = F.softmax(rho_logits, dim=-1)
        kl = torch.sum(F.kl_div(torch.log(pi), rho, reduction='none'), dim=-1)
        # kl = torch.sum(pi * (torch.log_softmax(pi_logits, dim=-1) - torch.log_softmax(rho_logits, dim=-1)), 
        #                dim=-1)
        expected_values = torch.sum(pi * values, dim=-1)
        if action_tokens is not None:
            token_pi = torch.gather(pi, dim=-1, index=action_tokens.unsqueeze(-1)).squeeze()
            token_values = torch.gather(values, dim=-1, index=action_tokens.unsqueeze(-1)).squeeze()
            return kl, expected_values, token_pi, token_values
        else:
            return kl, expected_values
        
    def get_last_token_position(self, action_tokens):
        pos = len(action_tokens) - 1
        while action_tokens[pos] == self.tokenizer.pad_token_id:
            pos -= 1
        return pos

    def get_joint_action_log_probs(self, obs, actions, action_tokens):
        pi_logits, _ = self.get_token_logits(obs, actions)
        pi_log_softmax = torch.log_softmax(pi_logits, dim=-1)
        action_log_probs = []
        entropies = []
        for i in range(pi_logits.shape[0]):
            act_token_length = self.get_last_token_position(action_tokens[i]) + 1
            log_softmax_slice = pi_log_softmax[i, :act_token_length, :]
            action_token_slice = action_tokens[i, :act_token_length]
            token_log_probs = torch.gather(log_softmax_slice, -1, action_token_slice.unsqueeze(-1)).squeeze(-1)
            action_log_prob = token_log_probs.sum()
            action_log_probs.append(action_log_prob)
            
            entropy = Categorical(logits=pi_logits[i, :act_token_length, :]).entropy().mean()
            entropies.append(entropy)
        action_log_probs = torch.stack(action_log_probs)
        entropies = torch.stack(entropies)
        return action_log_probs, entropies
            

    @torch.no_grad()
    def infer_for_rollout(self, obs):
        actions, action_tokens = self.get_actions(obs)
        
        values = self.get_token_values(obs, actions).squeeze(-1)
        pi_logits, _ = self.get_token_logits(obs, actions)
        pi_log_softmax = torch.log_softmax(pi_logits, dim=-1)
        token_log_probs = torch.gather(pi_log_softmax, -1, action_tokens.unsqueeze(-1)).squeeze(-1)
        
        values = values.float().cpu().numpy()
        action_tokens = action_tokens.int().cpu().numpy()
        token_log_probs = token_log_probs.float().cpu().numpy()
        
        return actions, action_tokens, values, token_log_probs

    
    def get_next_etpo_values(self, obs):
        
        token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        token_idx = attn_mask.sum(dim=1) - 1
        
        # values
        with self.actor.disable_adapter():
            values = self.critic(input_ids, attention_mask=attn_mask)
            values = values[torch.arange(obs.shape[0]), token_idx]
            
        # rho logits
        with self.actor.disable_adapter():
            rho_outputs = self.actor(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
            rho_logits = rho_outputs.logits[torch.arange(obs.shape[0]), token_idx]
        
        # pi logits
        pi_outputs = self.actor(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
        pi_logits = pi_outputs.logits[torch.arange(obs.shape[0]), token_idx]
        
        kl, expected_values = self.kl_cal(pi_logits, rho_logits, values)
        
        return expected_values, kl
    
    def get_next_tppo_values(self, obs): 
        token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        
        # values
        with self.actor.disable_adapter():
            values = self.critic(input_ids, attention_mask=attn_mask)
            values = values[:, -1]
        return values
    
    @torch.no_grad()
    def get_next_values(self, obs):
        """
        Get value function predictions.
        """
        values = self.get_next_tppo_values(obs).squeeze(-1)
        values = values.cpu().float().numpy()
        return values
        
    def infer_for_action_update(self, obs, actions, ava=None, action_tokens= None):
        assert action_tokens is not None, "action_tokens could not be none"
        action_log_probs, entropies = self.get_joint_action_log_probs(obs, actions, action_tokens)
        return action_log_probs, entropies
    
    def infer_for_token_update(self, obs, actions):
        pi_logits, rho_logits = self.get_token_logits(obs, actions)
        return pi_logits, rho_logits

    def save(self, save_dir, episode):
        print("save model")
        exp_path = os.path.join(save_dir, "episode_{:04d}".format(episode))

        os.makedirs(exp_path, exist_ok=True)
        # save lora
        self.actor.save_pretrained(exp_path)
        # save critic
        torch.save(self.critic.v_head.state_dict(), os.path.join(exp_path, "critic.pth"))

    def load(self, save_dir):
        print("load model")
        self.actor = self._init_actor(save_dir).to(self.device)
        critic_weights = os.path.join(save_dir, "critic.pth")
        self.critic = self._init_critic(critic_weights).to(self.device)

    def train(self):
        self.generator.train()
        self.critic.train()

    def eval(self):
        self.generator.eval()
        self.critic.eval()

