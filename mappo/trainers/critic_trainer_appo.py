import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mappo.utils.util import get_gard_norm, huber_loss, mse_loss
from mappo.envs.case_study.case_study_env import STATES, AVAILABLE_ACTIONS

class CriticAPPOTrainer:

    def __init__(self, args, agent, num_agents):
        self.tpdv = dict(dtype=torch.float32, device=torch.device("cuda:0"))
        self.agent = agent

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.entropy_coef = args.entropy_coef
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.gradient_cp_steps = args.gradient_cp_steps

        self.critic_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.agent.critic.parameters()), lr=self.critic_lr, eps=1e-5)
        
    
    def cal_value_loss(self, values_infer, value_preds_batch, return_batch):
        
        value_pred_clipped = value_preds_batch + (values_infer - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        error_clipped = return_batch - value_pred_clipped
        error_unclipped = return_batch - values_infer
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_unclipped = huber_loss(error_unclipped, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_unclipped = mse_loss(error_unclipped)
        value_loss = torch.max(value_loss_clipped, value_loss_unclipped).mean()
        return value_loss * self.value_loss_coef

    def ppo_update(self, sample):
        obs_batch, ava_batch, action_batch, log_prob_batch, \
            value_preds_batch, return_batch, advantages_batch, action_tokens_batch = sample
            
        # print("obs_batch: ", obs_batch)
        # print("action_batch: ", action_batch)
        # print("advantages_batch: ", advantages_batch)
        
        s1_a1_adv = []
        s1_a2_adv = []
        s3_a1_adv = []
        s3_a2_adv = []
        for i in range(len(obs_batch)):
            if obs_batch[i][0] == STATES[0]:
                if action_batch[i][0] == AVAILABLE_ACTIONS[0][0]:
                    s1_a1_adv.append(advantages_batch[i])
                elif action_batch[i][0] == AVAILABLE_ACTIONS[0][1]:
                    s1_a2_adv.append(advantages_batch[i])
                else:
                    raise ValueError("Invalid action")
            elif obs_batch[i][0] == STATES[2]:
                if action_batch[i][0] == AVAILABLE_ACTIONS[2][0]:
                    s3_a1_adv.append(advantages_batch[i])
                elif action_batch[i][0] == AVAILABLE_ACTIONS[2][1]:
                    s3_a2_adv.append(advantages_batch[i])
                else:
                    raise ValueError("Invalid action")
            elif obs_batch[i][0] == STATES[1]:
                pass
            else:
                raise ValueError("Invalid state")
        
        # print("s1_a1_adv mean: {}, std: {}".format(np.mean(s1_a1_adv), np.std(s1_a1_adv)))
        # print("s1_a2_adv mean: {}, std: {}".format(np.mean(s1_a2_adv), np.std(s1_a2_adv)))
        # print("s3_a1_adv mean: {}, std: {}".format(np.mean(s3_a1_adv), np.std(s3_a1_adv)))
        # print("s3_a2_adv mean: {}, std: {}".format(np.mean(s3_a2_adv), np.std(s3_a2_adv)))

        log_prob_batch = torch.from_numpy(log_prob_batch).to("cuda")
        value_preds_batch = torch.from_numpy(value_preds_batch).to("cuda")
        return_batch = torch.from_numpy(return_batch).to("cuda")
        # advantages_batch = torch.from_numpy(advantages_batch).to("cuda")
        action_tokens_batch = torch.from_numpy(action_tokens_batch).to("cuda")
        batch_size = obs_batch.shape[0]
        
        # critic update
        values_infer = self.agent.get_action_values(np.concatenate(obs_batch))
        values_infer = values_infer.view(batch_size, -1)
        
        value_loss = self.cal_value_loss(values_infer, value_preds_batch, return_batch)
        # print("value_loss: ", value_loss)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.agent.critic.parameters())
        self.critic_optimizer.step()
        value_loss = value_loss.item()
        self.critic_optimizer.zero_grad()
        critic_grad_norm = critic_grad_norm.item()
        
        return value_loss, critic_grad_norm, np.mean(s1_a1_adv), np.mean(s1_a2_adv), np.mean(s3_a1_adv), np.mean(s3_a2_adv)

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info['value_loss'] = 0
        train_info['value_grad_norm'] = 0
        train_info['s1_a1_adv'] = 0
        train_info['s1_a2_adv'] = 0
        train_info['s3_a1_adv'] = 0
        train_info['s3_a2_adv'] = 0

        update_time = 0
        for _ in range(self.ppo_epoch):
            data_generator = buffer.appo_sampler(self.num_mini_batch)
            for sample in data_generator:
                value_loss, value_grad_norm, s1_a1_adv, s1_a2_adv, s3_a1_adv, s3_a2_adv = self.ppo_update(sample)
                train_info['value_loss'] += value_loss
                train_info['value_grad_norm'] += value_grad_norm
                train_info['s1_a1_adv'] += s1_a1_adv
                train_info['s1_a2_adv'] += s1_a2_adv
                train_info['s3_a1_adv'] += s3_a1_adv
                train_info['s3_a2_adv'] += s3_a2_adv
                update_time += 1

        for k in train_info.keys():
            train_info[k] /= update_time
 
        return train_info

    def prep_training(self):
        self.agent.actor().train()
        self.agent.critic().train()

    def prep_rollout(self):
        self.agent.actor().eval()
        self.agent.critic().eval()
