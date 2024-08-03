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


class LlamaLoRAgent:

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
            print("init with weights")
            model = LlamaForCausalLM.from_pretrained(lora_weights, 
                                                     torch_dtype=torch.float16,
                                                     device_map="auto")

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)
        
        model.half()

        return model
    
    def _init_critic(self, critic_weights = None):
        if self.algo == "TWOSOME":
            critic = APPOCritic(self.actor, self.tokenizer)
        elif self.algo in ["POAD", "NTPO"]:
            critic = TPPOCritic(self.actor, self.tokenizer)
        else:
            raise NotImplementedError
        if critic_weights is not None:
            critic.v_head.load_state_dict(torch.load(critic_weights, map_location= "cpu"))
        return critic
    
    def sample_actions(self, input_ids, token_logits, seq_token_lengths, act_token_lengths, 
                       action_num_list, action_list, action_ids=None, greedy=False):
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
            token_slice = input_ids[i, start_idx+1:end_idx+1]
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
                if greedy:
                    action_i_idx = action_logits_i.argmax()
                else:
                    action_i_idx = dist_i.sample()  # for rollout
            else:
                action_i_idx = action_ids[i]  # for training
            
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
    
    def get_actions(self, obs, ava, actions=None, greedy=False):
        """
        Compute actions and value function predictions for the given inputs.
        """
        prompts = obs.tolist()
        action_list = [act.split(",") for act in ava.tolist()]
        action_num_list = [len(ac) for ac in action_list]
        if actions is not None:
            action_ids = []
            for i in range(len(actions)):
                action_ids.append(action_list[i].index(actions[i]))
            action_ids = torch.tensor(action_ids).to("cuda")
        else:
            action_ids = None
        
        sequences = []
        action_sequences = []
        for p, ac in zip(prompts, action_list):
            sequences += [p + " " + a for a in ac]
            action_sequences += [a for a in ac]  # for llama
        
        token_seq = self.tokenizer(sequences, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        seq_token_lengths = attn_mask.sum(dim=1)
        
        outputs = self.actor(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
        token_logits = outputs.logits[:, :-1, :]
        input_ids = input_ids[: , 1:]  # align logits and ids
        
        act_token_seq = self.tokenizer(action_sequences, return_tensors="pt", padding=True)
        act_attn_mask = act_token_seq["attention_mask"].to("cuda")
        act_token_lengths = act_attn_mask.sum(dim=1) - 1  # ignore the <bos> token
            
        actions, action_tokens, action_log_probs, entropies = self.sample_actions(input_ids, 
                                                                                  token_logits, 
                                                                                  seq_token_lengths,
                                                                                  act_token_lengths, 
                                                                                  action_num_list, 
                                                                                  action_list,
                                                                                  action_ids,
                                                                                  greedy)
        # print("selected action prob: ", action_log_probs.exp().mean())
        
        return actions, action_tokens, action_log_probs, entropies
        
    def get_action_values(self, obs):
        obs = obs.tolist()
        inputs = self.tokenizer(obs, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with self.actor.disable_adapter():
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
        
        with self.actor.disable_adapter():
            values = self.critic(input_ids, attention_mask=attn_mask)
        values = self.get_slice(values, seq_token_lengths, act_token_lengths)
        return values
    
    def get_token_logits(self, obs, actions):
        obs_act = [obs[i] + " " + actions[i] for i in range(len(obs))]
        
        token_seq = self.tokenizer(obs_act, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        seq_token_lengths = attn_mask.sum(dim=1)
        
        act_token_seq = self.tokenizer(actions.tolist(), return_tensors="pt", padding=True)
        act_attn_mask = act_token_seq["attention_mask"].to("cuda")
        act_token_lengths = act_attn_mask.sum(dim=1) - 1  # ignore the <bos> token
        
        with self.actor.disable_adapter():
            rho_outputs = self.actor(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
            rho_logits = self.get_slice(rho_outputs.logits, seq_token_lengths, act_token_lengths)
            
        pi_outputs = self.actor(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
        pi_logits = self.get_slice(pi_outputs.logits, seq_token_lengths, act_token_lengths)
        
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

    @torch.no_grad()
    def infer_for_rollout(self, obs, ava):
        actions, action_tokens, action_log_probs, _ = self.get_actions(obs, ava)
        
        if self.algo == "TWOSOME":
            values = self.get_action_values(obs)
            values = values.float().cpu().numpy()
            action_tokens = action_tokens.int().cpu().numpy()
            action_log_probs = action_log_probs.float().cpu().numpy()
            return actions, action_tokens, values, action_log_probs
        elif self.algo in ["POAD", "NTPO"]:
            values = self.get_token_values(obs, actions).squeeze(-1)
            pi_logits, _ = self.get_token_logits(obs, actions)
            pi_log_softmax = torch.log_softmax(pi_logits, dim=-1)
            token_log_probs = torch.gather(pi_log_softmax, -1, action_tokens.unsqueeze(-1)).squeeze(-1)
            
            values = values.float().cpu().numpy()
            action_tokens = action_tokens.int().cpu().numpy()
            token_log_probs = token_log_probs.float().cpu().numpy()
            
            return actions, action_tokens, values, token_log_probs
        else:
            raise NotImplementedError
    
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
        token_idx = attn_mask.sum(dim=1) - 1
        
        # values
        with self.actor.disable_adapter():
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
        
    def infer_for_action_update(self, obs, actions, ava=None, action_tokens= None):
        assert ava is not None, "ava could not be none"
        _, _, action_log_probs, entropies = self.get_actions(obs, ava, actions=actions)
        return action_log_probs, entropies
    
    def infer_for_token_update(self, obs, actions):
        pi_logits, rho_logits = self.get_token_logits(obs, actions)
        return pi_logits, rho_logits

    def save(self, save_dir, episode):
        print("save model")
        exp_path = os.path.join(save_dir, "episode_{:04d}".format(episode))

        os.makedirs(exp_path, exist_ok=True)
        # save lora
        print("active adapter: ", self.actor.active_adapter)
        model = self.actor.merge_and_unload()
        model.save_pretrained(exp_path)

    def load(self, save_dir):
        print("load model on path: ", save_dir)
        self.actor = self._init_actor(save_dir).to(self.device)


