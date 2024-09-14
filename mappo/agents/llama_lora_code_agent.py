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


class CodeLlamaLoRAgent:

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
                eos_token_id=[self.tokenizer.eos_token_id, 736],  # 736 is the "return"
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
        else:
            output = self.actor.generate(
                input_ids,
                attention_mask=attn_mask,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=[self.tokenizer.eos_token_id, 736],  # 736 is the "return"
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
    
    def get_slice(self, logits, obs_full_lengths, act_real_lengths):            
        action_slice = torch.zeros((logits.shape[0], self.max_new_tokens, logits.shape[-1])).to("cuda")
        for i in range(logits.shape[0]):
            start_idx = obs_full_lengths - 1
            end_idx = obs_full_lengths + act_real_lengths[i] - 1
            action_slice[i, :act_real_lengths[i]] = logits[i, start_idx:end_idx]
        return action_slice
    
    def get_token_values(self, obs, action_tokens):        
        obs_token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        obs_input_ids = obs_token_seq["input_ids"].to("cuda")
        obs_attn_mask = obs_token_seq["attention_mask"].to("cuda")
        obs_full_lengths = obs_input_ids.shape[1]
        
        act_attn_mask = (action_tokens != 0)
        act_real_lengths = act_attn_mask.sum(dim=1)
        
        obs_act_ids = torch.cat([obs_input_ids, action_tokens], dim=1)
        obs_act_mask = torch.cat([obs_attn_mask, act_attn_mask], dim=1)
        
        with self.actor.disable_adapter():
            values = self.critic(obs_act_ids, attention_mask=obs_act_mask)
        values = self.get_slice(values, obs_full_lengths, act_real_lengths)
        return values
    
    def get_token_logits(self, obs, action_tokens):
        obs_token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        obs_input_ids = obs_token_seq["input_ids"].to("cuda")
        obs_attn_mask = obs_token_seq["attention_mask"].to("cuda")
        obs_full_lengths = obs_input_ids.shape[1]
        
        act_attn_mask = (action_tokens != 0)
        act_real_lengths = act_attn_mask.sum(dim=1)
        
        obs_act_ids = torch.cat([obs_input_ids, action_tokens], dim=1)
        obs_act_mask = torch.cat([obs_attn_mask, act_attn_mask], dim=1)
        
        with self.actor.disable_adapter():
            rho_outputs = self.actor(input_ids=obs_act_ids, attention_mask=obs_act_mask, return_dict=True)
            rho_logits = self.get_slice(rho_outputs.logits, obs_full_lengths, act_real_lengths)
            
        pi_outputs = self.actor(input_ids=obs_act_ids, attention_mask=obs_act_mask, return_dict=True)
        pi_logits = self.get_slice(pi_outputs.logits, obs_full_lengths, act_real_lengths)
        
        return pi_logits, rho_logits
        
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
        
        values = self.get_token_values(obs, action_tokens).squeeze(-1)
        pi_logits, _ = self.get_token_logits(obs, action_tokens)
        pi_log_softmax = torch.log_softmax(pi_logits, dim=-1)
        token_log_probs = torch.gather(pi_log_softmax, -1, action_tokens.unsqueeze(-1)).squeeze(-1)
        
        values = values.float().cpu().numpy()
        action_tokens = action_tokens.int().cpu().numpy()
        token_log_probs = token_log_probs.float().cpu().numpy()
        
        return actions, action_tokens, values, token_log_probs
    
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
    
    def infer_for_token_update(self, obs, action_tokens):
        pi_logits, rho_logits = self.get_token_logits(obs, action_tokens)
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

