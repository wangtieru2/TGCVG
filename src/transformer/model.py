"""
Main diffusion code.
Code was adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
"""
import sys 
import math
import pathlib
from multiprocessing import cpu_count
from typing import Optional, Sequence, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchdiffeq import odeint
from tqdm import tqdm

import pyrootutils

path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)


from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy
from torch.distributions import Normal
from torch.distributions import Normal, TanhTransform, TransformedDistribution

# helpers
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data
# tensor helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T) 

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))  
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out

class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x

class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
            self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def forward(
            self, mean: torch.Tensor, log_std: torch.Tensor, training: bool = True, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)  
        std = torch.exp(log_std)


        if not training:  
            has_nan = torch.isnan(mean).any() or torch.isnan(std).any()
            if has_nan:
                has_nan_per_sample = torch.isnan(mean).any(dim=-1) | torch.isnan(std).any(dim=-1)

                action_sample = torch.full_like(mean, float("nan"))
                log_prob = torch.full_like(std[:, :, 0], float("nan"))
                
                valid_indices = torch.where(~has_nan_per_sample)[0]  
                if valid_indices.numel() > 0:     
                    valid_mean = mean[valid_indices]
                    valid_std = std[valid_indices]

                    if self.no_tanh:
                        valid_dist = Normal(valid_mean, valid_std)
                    else:
                        valid_dist = TransformedDistribution(
                            Normal(valid_mean, valid_std), TanhTransform(cache_size=1)
                        )

                    if deterministic:
                        valid_action = torch.tanh(valid_mean)
                    else:
                        valid_action = valid_dist.rsample()

                    valid_log_prob = torch.sum(valid_dist.log_prob(valid_action), dim=-1)

                    action_sample[valid_indices] = valid_action
                    log_prob[valid_indices] = valid_log_prob 

                return action_sample, log_prob
                

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)  
            )                          

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)  

        return action_sample, log_prob

# main class
class ElucidatedTransformer(nn.Module):
    def __init__( 
        self,
        obs_normalizer,
        state_dim,
        act_dim,
        hidden_size,
        max_length=19,
        n_layer=3,
        n_head=1,
        drop_p=0.1,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        no_tanh: bool = False,
        max_action: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.obs_normalizer = obs_normalizer
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

        ### transformer blocks
        input_seq_len = max_length
        blocks = [Block(hidden_size, input_seq_len, n_head, drop_p) for _ in range(n_layer)]
        self.transformer = nn.Sequential(*blocks)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        ### CQL 
        self.max_action = max_action
        self.base_network = nn.Linear(hidden_size, 2 * self.act_dim)
        
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)
        

    def forward(self, states, training=True):
        state_embeddings = self.embed_state(states)
        stacked_inputs = self.embed_ln(state_embeddings)
        x = self.transformer(stacked_inputs)  #(batch_size, T, hidden_size)  or (batch_size, repeat, T, hidden_size)
        base_network_output = self.base_network(x)

        mean, log_std = torch.split(base_network_output, self.act_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, training)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def sample_back_and_forth(
            self,
            transformer_seq_len,
            dynamics_model,
            samples: torch.Tensor,
            critic1,
            critic2,
    ):
        rewards = samples[:, :-1, -1, None]
        states = samples[:, :-1, :self.state_dim]     
        actions = samples[:, :-1, self.state_dim:self.state_dim+self.act_dim]

        batch_size, seq_len, _ = samples.shape
        sample_len = seq_len - transformer_seq_len

        norm_states =  self.obs_normalizer.normalize(states)

        rec_norm_states = norm_states[:, :transformer_seq_len, :]  # (batch_size, T, state_dim)
        rec_actions = torch.zeros((batch_size, 0, self.act_dim), device=actions.device, dtype=torch.float32)    # (batch_size, 0, act_dim)
        rec_rewards = torch.zeros((batch_size, 0, 1), device=rewards.device, dtype=torch.float32)      # (batch_size, 0, 1)
        for i in range(sample_len):
            actions_pred = self.get_action_infer_no_q(rec_norm_states)  #for gym

            rec_states = self.obs_normalizer.unnormalize(rec_norm_states)  # (batch_size, transformer_seq_len, state_dim)

            dynamics_info = dynamics_model.predict(
                rec_states[:, -1, :],  # (batch_size, state_dim)
                actions_pred  # (batch_size, act_dim)
            )  #return { 'next_obs': pred[0], 'reward':  pred[1], 'done': pred[2]}

            next_state_pred = torch.from_numpy(dynamics_info['next_obs']).to(rec_norm_states.device)  # (batch_size, state_dim)
            predict_reward = torch.from_numpy(dynamics_info['reward']).to(rec_norm_states.device)  # (batch_size, 1)
            next_state_pred = self.obs_normalizer.normalize(next_state_pred)  # (batch_size, state_dim)

            rec_norm_states = torch.cat(
                [rec_norm_states, next_state_pred[:, None, :]],
                dim=1
            ).to(dtype=torch.float32)  # (batch_size, transformer_seq_len + 1, state_dim)

            rec_actions = torch.cat(
                [rec_actions, actions_pred[:, None, :]],
                dim=1
            ).to(dtype=torch.float32)  # (batch_size, 0 , act_dim)

            rec_rewards = torch.cat(
                [rec_rewards, predict_reward[:, None, :]],
                dim=1
            ).to(dtype=torch.float32)  # (batch_size, 0 , 1)

        output_states = torch.clone(self.obs_normalizer.unnormalize(rec_norm_states[:, transformer_seq_len-1:, :]))  
        output_actions = torch.cat((rec_actions, torch.zeros_like(rec_actions[:,-1, None,:])), dim=1)
        output_rewards = torch.cat((rec_rewards, torch.zeros_like(rec_rewards[:,-1, None,:])), dim=1)

        outputs = torch.cat([output_states, output_actions, output_rewards], dim=-1)  
        return outputs
    
    def get_action_infer_no_q(self, states):
        states = states[:,-self.max_length:]
        states = torch.cat(
            [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
            dim=1).to(dtype=torch.float32)

        action_preds, _ = self.forward(states, training=False)  # shape: (B, T, act_dim)

        return action_preds[:, -1, :]  # shape: (B, act_dim)

    def get_action_infer_with_q(self, states, critic1, critic2):
        states = states[:, -self.max_length:]
        states = torch.cat(
            [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
            dim=1).to(dtype=torch.float32)  # shape: (B, T, state_dim)

        B, T, D = states.shape
        
        states_repeat = states.unsqueeze(1).repeat(1, 50, 1, 1)  # shape: (B, 50, T, state_dim)
        states_reshape = states_repeat.reshape(-1, T, D)  # shape: (B*50, T, state_dim)

        action_preds, _ = self.forward(states_reshape, training=False)  # shape: (B*50, T, act_dim)
        
        action_preds = action_preds.reshape(B, 50, T, -1)  # shape: (B, 50, T, act_dim)

        state_rpt = states_repeat[:, :, -1, :]
        action_rpt = action_preds[:, :, -1, :]   # shape: (B, 50, act_dim)

        q_value = torch.min(critic1(state_rpt, action_rpt), critic2(state_rpt, action_rpt)) #shape: (B, 50, 1)
        
        max_indices = q_value.squeeze(-1).argmax(dim=1)
        batch_indices = torch.arange(action_rpt.size(0), device=action_rpt.device)  #shape: (B,)
        
        return action_rpt[batch_indices, max_indices]   #shape: (B, act_dim)



def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim) 

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant
    
def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)

class FullyConnectedQFunction(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            orthogonal_init: bool = False,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        self.network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 1),
        )
        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.network[-1], False)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        return q_values

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)



class Trainer(object):
    def __init__(
            self,
            transformer_model,
            warmup_steps=10000,
            dataset: Optional[torch.utils.data.Dataset] = None,
            train_batch_size: int = 64,
            train_num_steps: int = 100000,
            save_and_sample_every: int = 10000,
            weight_decay: float = 0.,
            results_folder: str = './results',
            modalities: List[str] = ['observations', 'actions'],
            discounted_return : bool = True,
            gamma=0.99,
            state_dim=17,
            act_dim=6,
            lr_decay=True,
            lr_maxt=100000,
            lr_min=0.,
            device='cuda',
            use_automatic_entropy_tuning=True,
            alpha_multiplier=1.0,
            target_entropy=6,
            cql_n_actions=5,   
            cql_temp=1.0,
            cql_clip_diff_min=-np.inf,
            cql_clip_diff_max=np.inf,
            cql_min_q_weight=10.0,
    ):
        super().__init__()
        self.modalities = modalities  #["observations", "actions", "rewards"]
        self.discounted_return = discounted_return  #True
        self.warmup_steps = warmup_steps
    
        self.save_and_sample_every = save_and_sample_every        
        self.train_num_steps = train_num_steps

        if dataset is not None:
            self.batch_size = train_batch_size
            print(f'Using batch size: {self.batch_size}')
            # dataset and dataloader
            dl = DataLoader(dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True, 
                num_workers=0)
            self.dl = cycle(dl)        

        self.results_folder = pathlib.Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.step = 0

        '''CQL'''
        self.device = device
        self.actor = transformer_model.to(self.device)
        num_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {num_params}.')

        self.critic_1 = FullyConnectedQFunction(state_dim, act_dim, True).to(self.device)
        self.critic_2 = FullyConnectedQFunction(state_dim, act_dim, True).to(self.device)
        self.target_critic_1 = deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = deepcopy(self.critic_2).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, weight_decay=weight_decay)
        self.critic_1_optimizer = torch.optim.Adam(list(self.critic_1.parameters()), 3e-4)
        self.critic_2_optimizer = torch.optim.Adam(list(self.critic_2.parameters()), 3e-4)

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(self.log_alpha_prime.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=lr_min)
            self.critic_1_lr_scheduler = CosineAnnealingLR(self.critic_1_optimizer, T_max=lr_maxt, eta_min=lr_min)
            self.critic_2_lr_scheduler = CosineAnnealingLR(self.critic_2_optimizer, T_max=lr_maxt, eta_min=lr_min)

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=3e-4,
            )
        else:
            self.log_alpha = None

        self.lr_decay = lr_decay
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = act_dim
        self.soft_target_update_rate = 0.005
        self.num_steps_per_iter = 1000

        self.alpha_multiplier = alpha_multiplier
        self.target_entropy = target_entropy 
        self.cql_n_actions = cql_n_actions
        self.cql_temp = cql_temp
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.cql_min_q_weight = cql_min_q_weight


    def train(self):
        device = self.device
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                self.step += 1
                states, actions, rewards, next_states, terminals, time_steps, rtg = next(self.dl)
                states = states.to(device)    #(B, T, state_dim)
                states = self.actor.obs_normalizer.normalize(states)
                next_states = next_states.to(device)
                next_states = self.actor.obs_normalizer.normalize(next_states)
                actions = actions.to(device)
                rewards = rewards.to(device)
                terminals = terminals.to(dtype=torch.long, device=device)

                
                """ alpha loss """
                new_actions, log_pi = self.actor(states)
                alpha, alpha_loss = self._alpha_and_alpha_loss(states, log_pi)

                """ Policy loss """
                new_actions = new_actions.reshape(-1, self.action_dim)
                log_pi = log_pi.reshape(-1)
                policy_loss = self._policy_loss(
                    states.reshape(-1, self.state_dim), actions.reshape(-1, self.action_dim), new_actions, alpha, log_pi
                )

                """ Q function loss """
                qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
                    states, actions, next_states, rewards, terminals, alpha
                )

                if self.use_automatic_entropy_tuning:
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

                self.critic_1_optimizer.zero_grad()
                self.critic_2_optimizer.zero_grad()
                qf_loss.backward(retain_graph=True)
                self.critic_1_optimizer.step()
                self.critic_2_optimizer.step()

                pbar.set_description(f'policy loss: {policy_loss.item():.4f}')

                self.update_target_network(self.soft_target_update_rate)

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.save(self.step)

                pbar.update(1) 

                if self.step != 0 and self.step % self.num_steps_per_iter == 0:  
                    if self.lr_decay: 
                        self.actor_lr_scheduler.step()
                        self.critic_1_lr_scheduler.step()
                        self.critic_2_lr_scheduler.step()
        print('training complete')

    def _q_loss(
            self, observations, actions, next_observations, rewards, terminals, alpha
    ):
        device = self.device
        T = actions.shape[1]
        B = actions.shape[0]

        new_next_actions, next_log_pi = self.actor(next_observations)
        target_q_values = torch.min(
            self.target_critic_1(next_observations[:, -1], new_next_actions[:, -1]),
            self.target_critic_2(next_observations[:, -1], new_next_actions[:, -1]),
        )   # [B]

        not_done =(1 - terminals.unsqueeze(-1)[:, -1]) # [B, 1]
        discount1 = (T - 1 - torch.arange(T)).to(device)  # [T]
        discount1 = (self.gamma ** discount1).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        discount1 = discount1.expand(B, -1, -1)  # [B, T, 1]
        k_rewards = torch.cumsum(rewards.flip(dims=[1]) * discount1, dim=1).flip(dims=[1])  # [B, T, 1]

        discount2 = torch.arange(T).to(device)  # [T]
        discount2 = (self.gamma ** discount2).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        discount2 = discount2.expand(B, -1, -1)  # [B, T, 1]
        k_rewards = (k_rewards / discount2).squeeze(-1)  # [B, T]

        td_target = (k_rewards + (not_done * self.gamma * discount1.squeeze(-1) * target_q_values.unsqueeze(-1))) # [B, T]

        q1_predicted = self.critic_1(observations, actions)  # [B, T]
        q2_predicted = self.critic_2(observations, actions)

        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        action_dim = self.action_dim
        obs_dim = self.state_dim

        cql_random_actions = actions.new_empty(             # [B*T, cql_n_actions, action_dim]
            (B*T, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)

        observations = extend_and_repeat(observations, 1, self.cql_n_actions)  #[B, cql_n_actions, T, obs_dim]
        observations = observations.reshape(-1, T, obs_dim) #[B*cql_n_actions, T, obs_dim]
        cql_current_actions, cql_current_log_pis = self.actor(observations)  #[B*cql_n_actions, T, action_dim]

        next_observations = extend_and_repeat(next_observations, 1, self.cql_n_actions) 
        next_observations = next_observations.reshape(-1, T, obs_dim)
        cql_next_actions, cql_next_log_pis = self.actor(next_observations)
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        observations = observations.reshape(B, self.cql_n_actions, T, obs_dim)
        observations = observations.permute(0, 2, 1, 3).reshape(-1, self.cql_n_actions, obs_dim)  # [B*T, cql_n_actions, obs_dim]

        cql_current_actions = cql_current_actions.reshape(B, self.cql_n_actions, T, action_dim)
        cql_current_actions = cql_current_actions.permute(0, 2, 1, 3).reshape(-1, self.cql_n_actions, action_dim)  # [B*T, cql_n_actions, action_dim]
        cql_next_actions = cql_next_actions.reshape(B, self.cql_n_actions, T, action_dim)
        cql_next_actions = cql_next_actions.permute(0, 2, 1, 3).reshape(-1, self.cql_n_actions, action_dim)  # [B*T, cql_n_actions, action_dim]

        cql_next_log_pis = cql_next_log_pis.reshape(B, self.cql_n_actions, T)
        cql_next_log_pis = cql_next_log_pis.permute(0, 2, 1).reshape(-1, self.cql_n_actions)  # [B*T, cql_n_actions]

        cql_current_log_pis = cql_current_log_pis.reshape(B, self.cql_n_actions, T)
        cql_current_log_pis = cql_current_log_pis.permute(0, 2, 1).reshape(-1, self.cql_n_actions)  # [B*T, cql_n_actions]

        cql_q1_rand = self.critic_1(observations, cql_random_actions)   #[B*T, cql_n_actions]
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        random_density = np.log(0.5 ** action_dim)
        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand - random_density,
                cql_q1_next_actions - cql_next_log_pis.detach(),
                cql_q1_current_actions - cql_current_log_pis.detach(),
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand - random_density,
                cql_q2_next_actions - cql_next_log_pis.detach(),
                cql_q2_current_actions - cql_current_log_pis.detach(),
            ],
            dim=1,
        )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted.reshape(-1),
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted.reshape(-1),
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
        cql_min_qf2_loss = cql_qf2_diff * self.cql_min_q_weight
        alpha_prime_loss = observations.new_tensor(0.0)
        alpha_prime = observations.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        return qf_loss, alpha_prime, alpha_prime_loss

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                    self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss
    
    def _policy_loss(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            new_actions: torch.Tensor,
            alpha: torch.Tensor,
            log_pi: torch.Tensor,
    ) -> torch.Tensor:
        q_new_actions = torch.min(
            self.critic_1(observations, new_actions),
            self.critic_2(observations, new_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)


    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.actor.state_dict(),
            'opt': self.actor_optimizer.state_dict(),
            'critic_1': self.target_critic_1.state_dict(),
            'critic_2': self.target_critic_2.state_dict(),
            'scaler': None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone: Optional[str] = None, ckpt_path: Optional[str] = None):
        device = self.device

        if ckpt_path is not None:
            data = torch.load(ckpt_path, map_location=device)
        else:    
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        self.actor.load_state_dict(data['model'])
        self.step = data['step']
        self.actor_optimizer.load_state_dict(data['opt'])
        self.target_critic_1.load_state_dict(data['critic_1'])
        self.target_critic_2.load_state_dict(data['critic_2'])

        if hasattr(self, 'scaler') and exists(data['scaler']):
            self.scaler.load_state_dict(data['scaler'])

    

