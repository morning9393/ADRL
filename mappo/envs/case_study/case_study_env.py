import virtual_home
import gym
import numpy as np
from mappo.envs.virtualhome.virtualhome_utils import init_env

STATES = [
    'There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the kitchen. You notice pancake and microwave. Currently, you have grabbed the pancake in hand. The microwave is close to you. The microwave is not opend. In order to heat up the pancake in the microwave, your next step is to',
    'There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the kitchen. You notice pancake and microwave. Currently, you have grabbed the pancake in hand. The microwave is close to you. The microwave is opened. In order to heat up the pancake in the microwave, your next step is to',
    'There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the kitchen. You notice pancake and microwave. The microwave is opened. In order to heat up the pancake in the microwave, your next step is to'
]

ACTION_OPEN = 'open the microwave'
ACTION_CLOASE = 'close the microwave'
ACTION_PUT = 'put the pancake in the microwave'

AVAILABLE_ACTIONS = [
    # ['walk to the living room,walk to the bathroom,walk to the bedroom,walk to the pancake,put the pancake in the microwave,open the microwave,close the microwave']
    [ACTION_OPEN, ACTION_CLOASE],
    # ['walk to the living room,walk to the bathroom,walk to the bedroom,walk to the pancake,put the pancake in the microwave,open the microwave,close the microwave']
    [ACTION_PUT],
    # ['walk to the living room,walk to the bathroom,walk to the bedroom,open the microwave,close the microwave']
    [ACTION_OPEN, ACTION_CLOASE]
]


class BaseEnv:
    
    def __init__(self) -> None:
        self.max_step = 5
        self.stage = 0
        self.cur_step = 0

        
    def reset(self):
        self.cur_step = 0
        self.stage = 0
        obs = STATES[self.stage]
        ava = AVAILABLE_ACTIONS[self.stage]
        
        return obs, ava
        
    def step(self, action):
        self.cur_step += 1
        if self.cur_step >= self.max_step:
            done = True
            reward = -1.0
        else:
            reward = 0.0
            done = False
        
        if self.stage == 0:
            if ACTION_OPEN == action:
                self.stage = 1
        elif self.stage == 1:
            if ACTION_PUT == action:
                self.stage = 2
        elif self.stage == 2:
            if ACTION_CLOASE == action:
                reward = 1.0
                done = True
        else:
            raise ValueError("Invalid stage")
        
        if done:
            next_obs, ava = self.reset()
        else:
            next_obs = STATES[self.stage]
            ava = AVAILABLE_ACTIONS[self.stage]
        
        return next_obs, reward, done, ava
    
    
class CaseStudyEnv:
    
    def __init__(self, num_envs) -> None:
        self.num_agents = 1
        
        self.envs = [BaseEnv() for _ in range(num_envs)]

    def reset(self):
        obs = np.empty((len(self.envs), self.num_agents), dtype=np.object_)
        ava = np.empty_like(obs, dtype=np.object_)
        for i in range(len(self.envs)):
            env_obs, env_ava = self.envs[i].reset()
            obs[i, 0] = env_obs
            ava[i, 0] = ",".join(env_ava)
        return obs, ava
    
    def step(self, actions):
        next_obs = np.empty((len(self.envs), self.num_agents), dtype=np.object_)
        ava = np.empty_like(next_obs, dtype=np.object_)
        reward = []
        done = []
        for i in range(len(self.envs)):
            env_obs, env_rew, env_done, env_ava = self.envs[i].step(actions[i])
            next_obs[i, 0] = env_obs
            ava[i, 0] = ",".join(env_ava)
            reward.append([env_rew])
            done.append([env_done])
        reward = np.array(reward)
        done = np.array(done)
        
        info = [{} for _ in range(len(self.envs))]
        return next_obs, reward, done, ava, info
            
            