import gym
import d4rl
from .mujoco_env import make_mujoco_env

ENV = {
    "mujoco": make_mujoco_env,
    "d4rl": lambda env_name: gym.make(env_name),
    "adroit": lambda env_name: gym.make(env_name),
    "maze": lambda env_name: gym.make(env_name),
    "kitchen": lambda env_name: gym.make(env_name)
}
