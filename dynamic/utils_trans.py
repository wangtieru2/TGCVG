
import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
import torch
import numpy as np 
import gym 
import d4rl 
import numpy as np
from dynamic.agent import AGENT
from dynamic.components.static_fns import STATICFUNC


class RewardPredictingModel:
    def __init__(self, 
        device = torch.device('cuda:0'), 
        env_name = 'walker2d-medium-replay-v2', 
        load_path = ''
    ): 
        self.env_name = env_name
        self.env = gym.make(env_name)

        self.device = device 
        # create dynamics model
        task = env_name.split('-')[0]
        if task == "antmaze": task = task + "-" + env_name.split('-')[1]
        static_fn = STATICFUNC[task.lower()]
        self.agent = AGENT["admpo"](
            obs_shape=self.env.observation_space.shape,
            action_dim=np.prod(self.env.action_space.shape),
            static_fn=static_fn,
            max_arm_step=2,
            arm_hidden_dim=400 if "halfcheetah-medium-expert" in self.env_name else 200,
            model_lr=3e-4,
            device=self.device,
        )
        self.agent.load_model(load_path+"/model.pth")

    def predict(self, obs, act): 
        next_obs, reward = self.agent.dynamics.step(obs, act)
        if "maze" in self.env_name:
            reward -= 1
        return { 'next_obs': next_obs, 'reward':  reward }

# debug 
if __name__ == '__main__': 
    sys.exit(0)




