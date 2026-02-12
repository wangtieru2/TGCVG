import torch
import numpy as np
from tqdm import tqdm

from dynamic.agent import AGENT
from dynamic.buffer import BUFFER
from dynamic.components.static_fns import STATICFUNC

from .base_trainer import BASETrainer

class OFFTrainer(BASETrainer):
    """ offline MBRL trainer """

    def __init__(self, args):
        super(OFFTrainer, self).__init__(args)

        # init armpo agent
        task = args.env_name.split('-')[0]
        if args.env == "maze" and task == "antmaze": task = task + "-" + args.env_name.split('-')[1]
        # if args.env == "maze" and task == "maze2d": task = task + "-" + args.env_name.split('-')[1]
        self.task = task
        static_fn = STATICFUNC[task.lower()]
        self.agent = AGENT["admpo"](
            obs_shape=args.obs_shape,
            action_dim=args.action_dim,
            static_fn=static_fn,
            max_arm_step=args.max_arm_step,
            arm_hidden_dim=args.arm_hidden_dim,
            model_lr=args.model_lr,
            device=args.device
        )
        self.agent.train()

        # init replay buffer to store environmental data
        self.memory = BUFFER["seq-sample"](
            buffer_size=args.buffer_size,
            obs_shape=args.obs_shape,
            action_dim=args.action_dim
        )
        rew_bias = 1 if args.env == "maze" else 0
        self.memory.load_dataset(self.env.get_dataset(), self.env._max_episode_steps, rew_bias)


    def run(self):
        """ train {args.algo} on {args.env} for {args.n_epochs} epochs"""
        model_loss = self.agent.learn_dynamics_from(self.memory, self.batch_size)
        self._save()
        print(f"{self.task} final model loss: {model_loss}")
