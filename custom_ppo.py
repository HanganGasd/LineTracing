import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)

    def get_action_and_value(self, obs, action=None):
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)

        dist = Normal(mean, std)

        if action is None:
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
        else:
            action = torch.clamp(action, -0.999, 0.999)
            raw_action = 0.5 * torch.log((1 + action) / (1 - action))

        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        # tanh squashing 보정
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)

        return action, log_prob, entropy, value