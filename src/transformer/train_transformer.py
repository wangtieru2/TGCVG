import os
import argparse
from datetime import datetime
import pickle
from typing import List
import gym
import numpy as np
import torch
from torch.utils.data import DataLoader
from hydra import initialize, compose
from omegaconf import OmegaConf

import pyrootutils

path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")  # find the root path of the project
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)


from src.transformer.model import Trainer
from src.data.norm import normalizer_factory
from src.transformer.utils import split_transformer_trajectory, construct_transformer_model
from corl.shared.utils  import merge_dictionary
from corl.shared.buffer import TransformerTrajectoryDataset
from corl.shared.utils  import merge_dictionary

from dynamic.utils_trans import RewardPredictingModel
#eval

class SimpleTransformerGenerator:         
    def __init__(
            self,
            data_name,
            env: gym.Env,
            ema_model,
            sample_batch_size: int = 1000,
            modalities : List[str] = ["observations", "actions", "rewards"]
    ):
        self.data_name = data_name
        self.env = env
        self.transformer = ema_model
        self.transformer.eval()
        self.sample_batch_size = sample_batch_size
        self.modalities = modalities
        print(f'Sampling using: {self.sample_batch_size} batch size.')

    
    def prepare_sampling_data(
        self,
        states,
        actions,
        rewards,
        next_states,
        device,
    ):         
        B, T, D = states.shape
        rewards = rewards.reshape(B, T, 1)

        data = []
        for mod in self.modalities:
            if mod == 'observations':
                data.append(states)
            elif mod == 'actions':
                data.append(actions)
            elif mod == 'rewards':
                data.append(rewards) 
        last_state = next_states[:,-1, None,:]
        last_action = torch.zeros_like(actions[:,-1, None,:])
        last_reward = torch.zeros_like(rewards[:,-1, None,:])
        last_transition = torch.cat([last_state, last_action, last_reward], dim=-1).to(device)
            
        data = torch.cat(data, dim=-1).to(device)
        data = torch.cat([data, last_transition], dim=1)

        return data

    def sample_back_and_forth(
            self,
            min_reward,
            max_reward,
            transformer_seq_len,
            dynamics_model,
            data_loader,
            num_samples: int,
            device ,
            retain_original,
            critic1,
            critic2,
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        # assert num_samples % self.sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'
        num_batches = num_samples // self.sample_batch_size
        if num_samples % self.sample_batch_size != 0:
            num_batches += 1
        
        generated_samples = []
        loader_iterator = iter(data_loader)
        for i in range(num_batches):
            try:
                states, actions, rewards, next_states, terminals, time_steps, rtg = next(loader_iterator) 
            except StopIteration:
                loader_iterator = iter(data_loader)
                states, actions, rewards, next_states, terminals, time_steps, rtg = next(loader_iterator)
            samples = self.prepare_sampling_data(states,
                actions,
                rewards,
                next_states,
                device,
                )    

            print(f'Generating split {i + 1} of {num_batches}')
            sampled_outputs = self.transformer.sample_back_and_forth(
                transformer_seq_len=transformer_seq_len,
                dynamics_model=dynamics_model,
                samples=samples,
                critic1=critic1,
                critic2=critic2,
            )
            sampled_outputs = sampled_outputs.cpu().numpy()


            if "rewards" in self.modalities:
                gen_state, gen_action, gen_reward, gen_next_obs = split_transformer_trajectory(
                    samples = sampled_outputs, 
                    env = self.env,
                    modalities = self.modalities,)
                
                for b in range(gen_state.shape[0]):
                    if retain_original:       
                        temp = {
                        "observations": gen_state[b,:,:],
                        "actions": gen_action[b,:,:],
                        "next_observations": gen_next_obs[b,:,:],
                        "rewards": gen_reward[b,:,:].squeeze(),
                        "terminals": terminals[b,transformer_seq_len-1:].cpu().numpy(),
                        # for DT
                        "timesteps": time_steps[b,transformer_seq_len-1:].cpu().numpy(),
                        "RTG" : rtg[b,transformer_seq_len-1:].squeeze().cpu().numpy(),
                        }
                    generated_samples.append(temp)

        filtered_samples = [s for s in generated_samples if not np.isnan(s['rewards']).any()]
        if "maze" in self.data_name:
            filtered_samples = [s for s in filtered_samples if s['rewards'].mean() >= min_reward - 1]
        else:
            filtered_samples = [s for s in filtered_samples if s['rewards'].mean() >= min_reward]
        filtered_samples = [s for s in filtered_samples if s['rewards'].mean() <= 2 * max_reward]
        print(len(filtered_samples))


        return filtered_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--config_path', type=str, default='../../configs')
    parser.add_argument('--config_name', type=str, default='config.yaml')
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_samples', action='store_true', default=True)
    parser.add_argument('--load_checkpoint', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--back_and_forth', action='store_true')
    parser.add_argument('--infer_with_q', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
        
    with initialize(version_base=None, config_path=args.config_path):
        cfg = compose(config_name=args.config_name)  # load the config file

    # Set seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)
    device = args.device
    # Create the environment and dataset.
    env = gym.make(args.dataset)
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    with open(f'xx.pkl','rb') as f:
        dataset_pkl = pickle.load(f) 
    
    data = merge_dictionary(dataset_pkl)   

    for k in cfg.Dataset.modalities :   #modalities : ["observations", "actions", "rewards"]
        if k == "observations":
            obs_normalizer = normalizer_factory(cfg.Dataset.normalizer_type, torch.from_numpy(data[k]).float(), skip_dims=[])
        elif k == "actions":
            act_normalizer = normalizer_factory(cfg.Dataset.normalizer_type, torch.from_numpy(data[k]).float(), skip_dims=[])
        elif k == "rewards":
            min_reward = np.min(data[k])
            max_reward = np.max(data[k])
            print(f"min_reward: {min_reward}, max_reward: {max_reward}")
            data[k] = data[k].reshape(-1,1)

    if "penalty" in cfg.Dataset:
        penalty = cfg.Dataset.penalty 
        print("terminal penalty : ",penalty)
    else:
        penalty = None
        print("terminal penalty : None")


    print("load with Transformer Trajectory Dataset")
    dataset = TransformerTrajectoryDataset(
        dataset_pkl,
        args.dataset,
        seq_len = cfg.Dataset.seq_len,
        discounted_return = cfg.Dataset.discounted_return,
        restore_rewards=cfg.Dataset.restore_rewards,
        penalty = penalty,  
    )


    now = datetime.now()
    date_ = now.strftime("%Y-%m-%d")
    time_ = now.strftime("%H:%M")
    model_nm = args.config_name.split('.')[0]
    
    fname = args.dataset+"/"+model_nm+"/"+date_+"/"+time_
    
    resfolder = os.path.join(args.results_folder, fname)

    if not os.path.exists(resfolder):
            os.makedirs(resfolder)
        
    with open(os.path.join(resfolder, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    
    # Create the transformer model and trainer.
    transformer = construct_transformer_model(
        state_dim=obs_dim,
        act_dim=action_dim,
        max_action=float(env.action_space.high[0]),
        obs_normalizer=obs_normalizer,
        edm_config=cfg.ElucidatedTransformer,  
        )
    

    trainer = Trainer(
        transformer_model=transformer,
        dataset=dataset,
        results_folder=resfolder,
        train_batch_size=cfg.Trainer.train_batch_size,
        weight_decay=cfg.Trainer.weight_decay,
        train_num_steps=cfg.Trainer.train_num_steps,
        save_and_sample_every=cfg.Trainer.save_and_sample_every,
        modalities = cfg.Dataset.modalities,
        discounted_return = cfg.Dataset.discounted_return,
        state_dim = obs_dim,
        act_dim = action_dim,
        device=device,
        gamma=cfg.Trainer.gamma,
        use_automatic_entropy_tuning=True,
        target_entropy=-np.prod(env.action_space.shape).item(),
        cql_min_q_weight=cfg.Trainer.cql_min_q_weight,
    )

    if not args.load_checkpoint:
        trainer.train()
    else:
        trainer.load(ckpt_path=args.ckpt_path)

    if args.load_checkpoint:        
        # Generate samples and save them.   
        if args.save_samples:
            sample_batch_size = min(len(dataset),cfg.SimpleTransformerGenerator.sample_batch_size)
            generator = SimpleTransformerGenerator(
                data_name = args.dataset,
                env=env,
                ema_model=trainer.actor,
                modalities=cfg.Dataset.modalities,
                sample_batch_size=sample_batch_size,
            )
            if args.back_and_forth:
                dataset = TransformerTrajectoryDataset(
                    dataset_pkl,
                    args.dataset,
                    seq_len = cfg.SimpleTransformerGenerator.max_sample_length,
                    discounted_return = cfg.Dataset.discounted_return,
                    restore_rewards=cfg.Dataset.restore_rewards,
                    penalty = penalty,  
                )
                sample_loader = DataLoader(dataset, shuffle=True, batch_size=sample_batch_size)                
                
                num_transitions = cfg.SimpleTransformerGenerator.save_num_transitions

                
                lcm = (cfg.SimpleTransformerGenerator.max_sample_length+1 - cfg.Dataset.seq_len) * sample_batch_size
                num_transitions = ((num_transitions + lcm - 1) // lcm) * lcm  
                num_samples = num_transitions // (cfg.SimpleTransformerGenerator.max_sample_length+1 - cfg.Dataset.seq_len)             
                
                dynamics_model = RewardPredictingModel(
                    device = device,
                    env_name = args.dataset,   
                    load_path = cfg.SimpleTransformerGenerator.dynamic_model_path+args.dataset,
                ) 

                generated_samples = generator.sample_back_and_forth(
                    min_reward = min_reward,
                    max_reward = max_reward,
                    transformer_seq_len=cfg.Dataset.seq_len,
                    dynamics_model=dynamics_model,
                    data_loader=sample_loader,
                    num_samples=num_samples,
                    device = device,
                    retain_original=True,
                    critic1=trainer.target_critic_1,
                    critic2=trainer.target_critic_2,
                )

            save_dir = f'data/generated_data/{args.dataset}'      
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file_name = f"tgcvg_smaples.npz"
            
            gen_sample = np.array(generated_samples)
            np.random.shuffle(gen_sample)
            gen_sample = gen_sample[:(cfg.SimpleTransformerGenerator.save_num_transitions//(cfg.SimpleTransformerGenerator.max_sample_length+1 - cfg.Dataset.seq_len))+1]
            savepath = os.path.join(save_dir, save_file_name)

            np.savez(savepath,       
                data = gen_sample,
                config = dict(cfg))
    else:
        print("transformer training is done")

            