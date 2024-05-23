import virtual_home
import gym
import numpy as np
from mappo.envs.virtualhome.virtualhome_utils import init_env
from mappo.envs.virtualhome.virtualhome_variants import init_variant_env

def make_env(env_id, seed, idx, env_params):
    def thunk():

        env = gym.make(env_id, **env_params)
        return env

    return thunk

class VirtualHomeEnv:
    
    def __init__(self, env_id, num_envs, seed, variant=None) -> None:
        env_params = {'seed': seed, 'debug': False}
        self.num_envs = num_envs
        self.num_agents = 1
        self.envs = gym.vector.SyncVectorEnv([make_env(env_id, seed + i, i, env_params) for i in range(num_envs)])
        print("env_id: ", env_id)
        
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete)
        
        if variant is None:
            self.action_template, self.obs2text = init_env(env_id=env_id)
        else:
            self.action_template, self.obs2text = init_variant_env(env_id=env_id, variant=variant)

        self.template2action = {
            k:i for i,k in enumerate(self.action_template)
        }
        
    def reset(self):
        ori_obs = self.envs.reset()
        obs, ava = self.handle_obs(ori_obs)

        return obs, ava
        
    def step(self, ori_action):
        action = self.handle_action(ori_action)
        ori_next_obs, reward, done, info = self.envs.step(action)
        next_obs, ava = self.handle_obs(ori_next_obs)
        reward = np.repeat(reward[:, None], self.num_agents, axis=1)
        done = np.repeat(done[:, None], self.num_agents, axis=1)
        
        # for i in range(done.shape[0]):
        #     if done[i][0] and reward[i][0] == 0:
        #         reward[i][0] = -1
        
        return next_obs, reward, done, ava, info
    
    def handle_action(self, ori_action):
        action = np.zeros((self.num_envs,), dtype=np.int64)
        for i in range(self.num_envs):
            action[i] = self.template2action[ori_action[i][0]]
        return action
    
    def handle_obs(self, ori_obs):
        obs = np.empty((self.num_envs, self.num_agents), dtype=np.object_)
        ava = np.empty_like(obs, dtype=np.object_)
        for i in range(self.num_envs):
            text_obs = self.obs2text(ori_obs[i], self.action_template)
            obs[i, 0] = text_obs["prompt"]
            ava[i, 0] = ",".join(text_obs["avaliable_action"])
        return obs, ava
    
    def close(self):
        self.envs.close()