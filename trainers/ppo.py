from collections.abc import Iterable
from itertools import chain
from typing import SupportsFloat
from torch import Tensor
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .utils.critic_network import Critic

from .trainer import Trainer


EPS = 1e-8


class RolloutDataset(Dataset):
    def __init__(self, obsns, acts, advgs, lgprobs, returns=None):
        self.obsns = obsns
        self.acts = acts
        self.advgs = advgs
        self.lgprobs = lgprobs
        self.returns = returns

    def __len__(self):
        return len(self.obsns)

    def __getitem__(self, idx):
        if self.returns is not None:
            return self.obsns[idx], self.acts[idx], self.advgs[idx], self.lgprobs[idx], self.returns[idx]
        else:
            return self.obsns[idx], self.acts[idx], self.advgs[idx], self.lgprobs[idx]


# def collate_fn(batch):
#     obsns, acts, advgs, lgprobs = zip(*batch)
#     obsns = collate_obsns(obsns)
#     acts = torch.stack(acts)
#     advgs = torch.stack(advgs)
#     lgprobs = torch.stack(lgprobs)
#     return obsns, acts, advgs, lgprobs


class PPO(Trainer):

    def __init__(self, agent_cfg, env_cfg, train_cfg):
        super().__init__(agent_cfg, env_cfg, train_cfg)

        self.entropy_coeff = train_cfg.get("entropy_coeff", 0.4)
        self.clip_range = train_cfg.get("clip_range", 0.2)
        self.target_kl = train_cfg.get("target_kl", 0.01)
        self.num_epochs = train_cfg.get("num_epochs", 10)
        self.num_batches = train_cfg.get("num_batches", 3)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 0.5) 
        
        self.value_coeff = train_cfg.get("value_coeff", 0.5) 

        self.use_critic = train_cfg.get('use_critic', False)
        self.combined_loss = train_cfg.get('combined_loss', True) 
        self.current_iteration = 0  
        
        if self.use_critic:
            self.compare_baselines = train_cfg.get('compare_baselines', True)
            self.critic_warmup_iterations = train_cfg.get('critic_warmup_iterations', 10)
            
            # Initialize critic network
            critic_input_dim = train_cfg.get('critic_input_dim', 1)
            critic_hidden_dim1 = train_cfg.get('critic_hidden_dim1', 64)
            critic_hidden_dim2 = train_cfg.get('critic_hidden_dim2', 32)
            critic_lr = train_cfg.get('critic_lr', 3e-4)
            num_sequences = train_cfg.get('num_sequences', 1)
            num_rollouts = train_cfg.get('num_rollouts', 1)
            
            self.critic = Critic(
                input_dim=critic_input_dim,
                hidden_dim1=critic_hidden_dim1,
                hidden_dim2=critic_hidden_dim2,
                lr=critic_lr,
                num_sequences=num_sequences,
                num_rollouts=num_rollouts,
                feature_extractor=None  
            )

            if self.combined_loss:
                self.optimizer = torch.optim.Adam(
                    list(self.scheduler.parameters()) + list(self.critic.network.parameters()),
                    lr=train_cfg.get('combined_lr', 3e-4)
                )
                print(f'Initialized combined actor-critic optimizer with lr: {train_cfg.get("combined_lr", 3e-4)}')
            else:
                # Separate optimizers
                self.optimizer = torch.optim.Adam(
                    self.scheduler.parameters(),
                    lr=train_cfg.get("combined_lr", 3e-4)
                )
                print(f'Initialized separate actor optimizer with lr: {train_cfg.get("lr", 3e-4)}')

            self.critic.scheduler = self.scheduler

        else:
            # No critic, just initialize optimizer for actor only
            self.optimizer = torch.optim.Adam(
                self.scheduler.parameters(),
                lr=train_cfg.get("combined_lr", 3e-4)
            )

    def _preprocess_rollouts(self, rollout_buffers):
        data = super()._preprocess_rollouts(rollout_buffers)

        data['original_baselines_list'] = data['baselines_list'].copy()

        if self.use_critic:
            obsns_list = data['obsns_list']
            returns_list = data['returns_list']
            
            ts_list = []
            for rollout_buffer in rollout_buffers:
                if rollout_buffer is not None:
                    # Remove last timestamp (same as base class does)
                    ts_list.append(rollout_buffer.wall_times[:-1])
            
            # Use critic to get baselines
            if self.current_iteration >= self.critic_warmup_iterations:
                critic_baselines_list = self.critic(ts_list, returns_list)
                data['critic_baselines_list'] = critic_baselines_list
                
                if self.compare_baselines:
                    # Use critic baselines instead of original
                    data['baselines_list'] = critic_baselines_list
                    print(f"Using critic baselines (iteration {self.current_iteration})")
                else:
                    print(f"Critic baselines computed but not used (iteration {self.current_iteration})")
            else:
                print(f"Using original baselines during warmup (iteration {self.current_iteration}/{self.critic_warmup_iterations})")

        return data

    def train_on_rollouts(self, rollout_buffers):
        # Store rollout buffers for critic training
        self.last_rollout_buffers = rollout_buffers
        
        data = self._preprocess_rollouts(rollout_buffers) # turns rollout buffers in a list.

        returns_flat = np.array(list(chain(*data["returns_list"])))
        returns = np.array([b.cpu().numpy() if torch.is_tensor(b) else b for b in returns_flat])
        baselines = np.concatenate([b.cpu().numpy() if torch.is_tensor(b) else b for b in data['baselines_list']])

        advantages = returns - baselines

        if self.use_critic and 'critic_baselines_list' in data:
            original_baselines = np.concatenate([b.cpu().numpy() if torch.is_tensor(b) else b for b in data['original_baselines_list']])
            original_advantages = returns - original_baselines
            original_advantage_var = np.var(original_advantages)

            critic_baselines = np.concatenate([b.cpu().numpy() if torch.is_tensor(b) else b for b in data['critic_baselines_list']])
            critic_advantages = returns - critic_baselines
            critic_advantage_var = np.var(critic_advantages)

        dataset = RolloutDataset(
            obsns=list(chain(*data["obsns_list"])),
            acts=list(chain(*data["actions_list"])),
            advgs=advantages,
            lgprobs=list(chain(*data["lgprobs_list"])),
            returns=returns if self.use_critic and self.combined_loss else None
        )

        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset) // self.num_batches + 1,
            shuffle=True,
            collate_fn=lambda batch: list(zip(*batch))
        )

        training_metrics = self._train(dataloader, data)

        # Train critic if using critic and separate training
        if self.use_critic and hasattr(self, 'critic') and not self.combined_loss:
            # For compatibility with standalone critic
            critic_loss = self._train_critic(data)
            print(f'DEBUG: using standalone critic with separate training, critic loss: {critic_loss}')
        elif self.use_critic and self.combined_loss:
            # Combined training already handled in _compute_loss
            critic_loss = training_metrics.get("value loss", 0.0)
            print(f'DEBUG: using combined actor-critic training, value loss: {critic_loss}')
        else:
            critic_loss = 0.0
        
        if self.use_critic and 'critic_baselines_list' in data:
            training_metrics.update({
                'original_advantage_var': original_advantage_var,
                'critic_advantage_var': critic_advantage_var,
                'variance_reduction': ((original_advantage_var - critic_advantage_var) / original_advantage_var),
                'critic_loss': critic_loss if self.use_critic and hasattr(self, 'critic') else 0.0
            })
            
        # Increment iteration counter
        self.current_iteration += 1

        return training_metrics

    def _train(self, dataloader, data):
        policy_losses = []
        entropy_losses = []
        value_losses = []
        approx_kl_divs = []
        continue_training = True

        for i in range(self.num_epochs):
            if not continue_training:
                break

            print(f"current self epochs: {i}, total batches: {self.num_batches}")

            for batch_data in dataloader:
                if len(batch_data) == 5:  # With returns
                    obsns, actions, advgs, old_lgprobs, returns = batch_data
                    loss, info = self._compute_loss(obsns, actions, advgs, old_lgprobs, returns, data)
                else:  # Without returns (backward compatibility)
                    obsns, actions, advgs, old_lgprobs = batch_data
                    loss, info = self._compute_loss(obsns, actions, advgs, old_lgprobs)

                print(f'current Total loss: {loss}')

                kl = info["approx_kl_div"]

                policy_losses += [info["policy_loss"]]
                entropy_losses += [info["entropy_loss"]]
                value_losses += [info.get("value_loss", 0.0)]
                approx_kl_divs.append(kl)

                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    print(f"Early stopping due to reaching max kl: {kl:.3f}")
                    continue_training = False
                    break

                self.optimizer.zero_grad()
                loss.backward()
                if self.use_critic and self.combined_loss:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.scheduler.parameters()) + list(self.critic.network.parameters()), 
                        max_norm=self.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(self.scheduler.parameters(), self.max_grad_norm)
                self.optimizer.step()

        metrics = {
            "policy loss": np.abs(np.mean(policy_losses)),
            "entropy": np.abs(np.mean(entropy_losses)),
            "approx kl div": np.abs(np.mean(approx_kl_divs)),
        }
        
        if value_losses and any(v != 0.0 for v in value_losses):
            metrics["value loss"] = np.abs(np.mean(value_losses))
            
        return metrics

    def _train_critic(self, data):
        print('training critic with non combined loss')
        if hasattr(self, 'critic') and self.current_iteration >= self.critic_warmup_iterations:
            # Extract states and returns for critic training
            all_obs = []
            all_times = []
            all_returns = []
            
            for obs_seq, returns_seq in zip(data['obsns_list'], data['returns_list']):
                all_obs.extend(obs_seq)
                all_returns.extend(returns_seq)
            
            # Extract times from rollout buffers
            for rollout_buffer in self.last_rollout_buffers:
                if rollout_buffer is not None:
                    # Use wall_times (excluding last timestamp)
                    all_times.extend(rollout_buffer.wall_times[:-1])
            
            states = self.critic.extract_features_from_observations(all_obs, all_times)
            
            critic_loss = self.critic.update(states, all_returns)
            print(f"Critic loss: {critic_loss:.4f}")
            
            return critic_loss
        return 0.0

    def _compute_loss(
        self,
        obsns: Iterable[dict],
        acts: Iterable[tuple],
        advantages: Iterable[SupportsFloat],
        old_lgprobs: Iterable[SupportsFloat],
        returns: Iterable[SupportsFloat] = None,
        data = None,
    ) -> tuple[Tensor, dict[str, SupportsFloat]]:
        eval_res = self.scheduler.evaluate_actions(obsns, acts)

        advgs = torch.tensor(advantages).float()
        advgs = (advgs - advgs.mean()) / (advgs.std() + EPS)

        log_ratio = eval_res["lgprobs"] - torch.tensor(old_lgprobs)
        ratio = log_ratio.exp()

        policy_loss1 = advgs * ratio
        policy_loss2 = advgs * torch.clamp(
            ratio, 1 - self.clip_range, 1 + self.clip_range
        )
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

        # Entropy loss
        entropy_loss = -eval_res["entropies"].mean()

        total_loss = policy_loss + self.entropy_coeff * entropy_loss
        
        value_loss = torch.tensor(0.0)
        if self.use_critic and returns is not None and self.combined_loss:
            try:
                value_loss = self._calc_critic_loss(data)
                total_loss = total_loss + self.value_coeff * value_loss
                print(f'Value loss computed: {value_loss}')
            except Exception as e:
                print(f'Error computing value loss: {e}')
                value_loss = torch.tensor(0.0)
        elif not self.use_critic and not self.combined_loss:
            print('not critic and not combined loss, thus value loss 0')

        with torch.no_grad():
            approx_kl_div = ((ratio - 1) - log_ratio).mean().item()

        return total_loss, {
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "value_loss": value_loss.item() if torch.is_tensor(value_loss) else value_loss,
            "approx_kl_div": approx_kl_div,
        }
    
    def _calc_critic_loss(self, data):
        if hasattr(self, 'critic') and self.current_iteration >= self.critic_warmup_iterations:
            all_obs = []
            all_times = []
            all_returns = []
            
            for obs_seq, returns_seq in zip(data['obsns_list'], data['returns_list']):
                all_obs.extend(obs_seq)
                all_returns.extend(returns_seq)
            
            for rollout_buffer in self.last_rollout_buffers:
                if rollout_buffer is not None:
                    all_times.extend(rollout_buffer.wall_times[:-1])
            
            states = self.critic.extract_features_from_observations(all_obs, all_times)
            
            critic_loss = self.critic.calc_loss(states, all_returns)

            print(f"Critic loss: {critic_loss.item()}")
            
            return critic_loss
        return 0.0


    def save(self, path):

        torch.save(self.scheduler.state_dict(), path)
        print(f'Policy saved to {path}') # saves policy

        if self.use_critic:
            critic_path = path.replace('.pt', '_critic.pt')
            torch.save({
                'model_state_dict': self.critic.network.state_dict(),
                'optimizer_state_dict': self.critic.optimizer.state_dict(),
                'state_mean': self.critic.state_mean,
                'state_std': self.critic.state_std,
            }, critic_path)
            print(f'Critic model saved to {critic_path}')

    def load(self, path):
        self.scheduler.load_state_dict(torch.load(path, map_location=self.device))
        print(f'policy loaded from {path}')

        if self.use_critic:
            critic_path = path.replace('.pt', '_critic.pt')
            if os.path.exists(critic_path):
                checkpoint = torch.load(critic_path, map_location=self.device)

                self.critic.network.load_state_dict(checkpoint['model_state_dict'])
                self.critic.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.critic.state_mean = checkpoint['state_mean']
                self.critic.state_std = checkpoint['state_std']
                print(f'Critic model loaded from {critic_path}')

            else:
                print(f'no critic model found at {critic_path}')

