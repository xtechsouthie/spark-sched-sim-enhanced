import numpy as np
import torch
import torch.profiler

from .trainer import Trainer


EPS = 1e-8


class VPG(Trainer):
    """Vanilla Policy Gradient"""

    def __init__(self, agent_cfg, env_cfg, train_cfg):
        super().__init__(agent_cfg, env_cfg, train_cfg)

        self.entropy_coeff = train_cfg.get("entropy_coeff", 0.0)

    def train_on_rollouts(self, rollout_buffers):
        data = self._preprocess_rollouts(rollout_buffers)

        policy_losses = []
        entropy_losses = []

        for obsns, actions, returns, baselines, old_lgprobs in zip(data.values()):
            eval_res = self.scheduler.evaluate_actions(obsns, actions)

            # re-computed log-probs don't exactly match the original ones,
            # but it doesn't seem to affect training
            # with torch.no_grad():
            #     diff = (lgprobs - torch.tensor(old_lgprobs)).abs()
            #     assert lgprobs.allclose(torch.tensor(old_lgprobs))

            adv = torch.from_numpy(returns - baselines).float()
            adv = (adv - adv.mean()) / (adv.std() + EPS)
            policy_loss = -(eval_res["lgprobs"] * adv).mean()
            policy_losses += [policy_loss.item()]

            entropy_loss = -eval_res["entropies"].mean()
            entropy_losses += [entropy_loss.item()]

            loss = policy_loss + self.entropy_coeff * entropy_loss
            loss.backward()

        self.scheduler.update_parameters()

        return {
            "policy loss": np.mean(policy_losses),
            "entropy": np.mean(entropy_losses),
        }
