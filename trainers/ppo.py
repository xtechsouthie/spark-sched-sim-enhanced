from collections.abc import Iterable
from itertools import chain
from typing import SupportsFloat
from torch import Tensor
import sys
from .utils.meta_critic import MetaCritic
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .trainer import Trainer


EPS = 1e-8


class RolloutDataset(Dataset):
    def __init__(self, obsns, acts, advgs, lgprobs):
        self.obsns = obsns
        self.acts = acts
        self.advgs = advgs
        self.lgprobs = lgprobs

    def __len__(self):
        return len(self.obsns)

    def __getitem__(self, idx):
        return self.obsns[idx], self.acts[idx], self.advgs[idx], self.lgprobs[idx]


# def collate_fn(batch):
#     obsns, acts, advgs, lgprobs = zip(*batch)
#     obsns = collate_obsns(obsns)
#     acts = torch.stack(acts)
#     advgs = torch.stack(advgs)
#     lgprobs = torch.stack(lgprobs)
#     return obsns, acts, advgs, lgprobs


class PPO(Trainer):
    """Proximal Policy Optimization"""

    def __init__(self, agent_cfg, env_cfg, train_cfg):
        super().__init__(agent_cfg, env_cfg, train_cfg)

        self.entropy_coeff = train_cfg.get("entropy_coeff", 0.0)
        self.clip_range = train_cfg.get("clip_range", 0.2)
        self.target_kl = train_cfg.get("target_kl", 0.01)
        self.num_epochs = train_cfg.get("num_epochs", 10)
        self.num_batches = train_cfg.get("num_batches", 3)
        self.current_iteration = 0
        self.use_meta_critic = train_cfg.get("use_meta_critic", True)
        if self.use_meta_critic:
            self.compare_baselines = train_cfg.get("compare_baselines", True)
            self.critic_warmup_iterations = train_cfg.get("critic_warmup_iterations", 20)

            meta_input_dim = train_cfg.get("meta_input_dim", 16)
            meta_hidden_dim1 = train_cfg.get("meta_hidden_dim1", 64)
            meta_hidden_dim2 = train_cfg.get("meta_hidden_dim2", 32)
            meta_lr = train_cfg.get("meta_lr", 3e-4)
            meta_inner_lr = train_cfg.get("meta_inner_lr", 3e-4)

            self.meta_critic = MetaCritic(
                input_dim=meta_input_dim,
                hidden_dim1=meta_hidden_dim1,
                hidden_dim2=meta_hidden_dim2,
                meta_lr=meta_lr,
                inner_lr=meta_inner_lr,
                num_sequences=self.num_sequences,
                num_rollouts=self.num_rollouts
            )

            self.meta_critic.scheduler = self.scheduler

    def _preprocess_rollouts(self, rollout_buffers):
        data = super()._preprocess_rollouts(rollout_buffers)
    
        data["original_baselines_list"] = data["baselines_list"].copy()

        if self.use_meta_critic:
            all_obs = []
            for obs_seq in data["obsns_list"]:
                all_obs.extend(obs_seq)
            
            ts_list = []
            for rollout_buffer in rollout_buffers:
                if rollout_buffer is not None:
                    ts_list.extend(rollout_buffer.wall_times[:-1])
            
            if self.current_iteration >= self.critic_warmup_iterations:
                meta_critic_baselines = self.meta_critic(ts_list, all_obs)  # Single array
                
                # Fix: Convert single array back to list of arrays matching original structure
                meta_critic_baselines_list = []
                start_idx = 0
                
                for baseline_seq in data["original_baselines_list"]:
                    seq_length = len(baseline_seq)
                    if seq_length > 0:
                        meta_critic_baselines_list.append(meta_critic_baselines[start_idx:start_idx + seq_length])
                        start_idx += seq_length
                    else:
                        meta_critic_baselines_list.append(np.array([]))
                
                data["meta_critic_baselines_list"] = meta_critic_baselines_list

                if self.compare_baselines:
                    data["baselines_list"] = meta_critic_baselines_list
                    print(f"Using meta critic baselines (iteration {self.current_iteration})")

            else:
                print(f"Using original baselines during warmup (iteration {self.current_iteration}/{self.critic_warmup_iterations})")

        return data

    def train_on_rollouts(self, rollout_buffers):
        self.last_rollout_buffers = rollout_buffers
        data = self._preprocess_rollouts(rollout_buffers)

        returns = np.array(list(chain(*data["returns_list"])))
        baselines = np.concatenate(data["baselines_list"])

        dataset = RolloutDataset(
            obsns=list(chain(*data["obsns_list"])),
            acts=list(chain(*data["actions_list"])),
            advgs=returns - baselines,
            lgprobs=list(chain(*data["lgprobs_list"])),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset) // self.num_batches + 1,
            shuffle=True,
            collate_fn=lambda batch: zip(*batch),
        )

        training_metrics =  self._train_actor(dataloader)

        if self.use_meta_critic and hasattr(self, "meta_critic"):
            meta_critic_loss = self._train_meta_critic(data)
            training_metrics.update({
                "meta_critic_loss": meta_critic_loss
            })

        self.current_iteration += 1

        return training_metrics

    def _train_actor(self, dataloader):
        policy_losses = []
        entropy_losses = []
        approx_kl_divs = []
        continue_training = True

        for _ in range(self.num_epochs):
            if not continue_training:
                break

            for obsns, actions, advgs, old_lgprobs in dataloader:
                loss, info = self._compute_loss(obsns, actions, advgs, old_lgprobs)

                print(f"actor loss: {loss}")
                kl = info["approx_kl_div"]

                policy_losses += [info["policy_loss"]]
                entropy_losses += [info["entropy_loss"]]
                approx_kl_divs.append(kl)

                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    print(f"Early stopping due to reaching max kl: " f"{kl:.3f}")
                    continue_training = False
                    break

                self.scheduler.update_parameters(loss)

        return {
            "policy loss": np.abs(np.mean(policy_losses)),
            "entropy": np.abs(np.mean(entropy_losses)),
            "approx kl div": np.abs(np.mean(approx_kl_divs)),
        }

    def _compute_loss(
        self,
        obsns: Iterable[dict],
        acts: Iterable[tuple],
        advantages: Iterable[SupportsFloat],
        old_lgprobs: Iterable[SupportsFloat],
    ) -> tuple[Tensor, dict[str, SupportsFloat]]:
        """CLIP loss"""
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

        entropy_loss = -eval_res["entropies"].mean()

        loss = policy_loss + self.entropy_coeff * entropy_loss

        with torch.no_grad():
            approx_kl_div = ((ratio - 1) - log_ratio).mean().item()

        return loss, {
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "approx_kl_div": approx_kl_div,
        }

    def _train_meta_critic(self, data):
        if self.use_meta_critic and hasattr(self, 'meta_critic'):
            obs = []
            times = []
            returns = []
            meta_critic_loss = []

            for obs_seq, returns_seq in zip(data['obsns_list'], data['returns_list']):
                obs.extend(obs_seq)
                returns.extend(returns_seq)
            
            for rollout_buffer in self.last_rollout_buffers:
                if rollout_buffer is not None:
                    times.extend(rollout_buffer.wall_times[:-1])
            
            # this is for training meta_critic with obs which have same job input sequences.
            for j in range(self.num_sequences): 
                start1 = j * self.num_rollouts
                end1 = start1 + (self.num_rollouts // 2)
                start2 = end1
                end2 = start1 + self.num_rollouts
                loss = self.meta_critic.update(times[start1:end1], times[start2:end2], obs[start1:end1], obs[start2:end2],
                                               returns[start1:end1], returns[start2:end2])
                meta_critic_loss.append(loss)

            avg_loss = np.mean(meta_critic_loss)
            return avg_loss
        return 0.0
    
    def save(self, path):

        torch.save(self.scheduler.state_dict(), path)
        print(f'Policy saved to {path}') # saves policy

        if self.use_meta_critic:
            critic_path = path.replace('.pt', '_critic.pt')
            torch.save({
                'model_state_dict': self.meta_critic.network.state_dict(),
                'optimizer_state_dict': self.meta_critic.optimizer.state_dict(),
                'state_mean': self.meta_critic.state_mean,
                'state_std': self.meta_critic.state_std,
            }, critic_path)
            print(f'Critic model saved to {critic_path}')

    def load(self, path):
        self.scheduler.load_state_dict(torch.load(path, map_location=self.device))
        print(f'policy loaded from {path}')

        if self.use_meta_critic:
            critic_path = path.replace('.pt', '_critic.pt')
            if os.path.exists(critic_path):
                checkpoint = torch.load(critic_path, map_location=self.device)

                self.meta_critic.network.load_state_dict(checkpoint['model_state_dict'])
                self.meta_critic.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.meta_critic.state_mean = checkpoint['state_mean']
                self.meta_critic.state_std = checkpoint['state_std']
                print(f'Critic model loaded from {critic_path}')

            else:
                print(f'no critic model found at {critic_path}')
