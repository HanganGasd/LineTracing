"""Asymmetric camera actor-critic for continuous-control PPO."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal


def _orthogonal_init(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.zeros_(module.bias)


class _CameraHead(nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int, int],
        auxiliary_size: int,
        output_size: int,
        output_gain: float,
    ) -> None:
        super().__init__()
        channels, height, width = image_shape
        self.image_shape = image_shape
        self.image_size = channels * height * width
        self.auxiliary_size = auxiliary_size
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 3)),
            nn.Flatten(),
        )
        self.hidden = nn.Sequential(
            nn.Linear(48 * 2 * 3 + auxiliary_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(32, output_size)
        self.apply(_orthogonal_init)
        _orthogonal_init(self.output, output_gain)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        image = observation[..., : self.image_size].reshape(
            -1, *self.image_shape
        )
        features = self.features(image)
        auxiliary = observation[
            ..., self.image_size : self.image_size + self.auxiliary_size
        ].reshape(-1, self.auxiliary_size)
        features = torch.cat((features, auxiliary), dim=-1)
        return self.output(self.hidden(features))


class CameraActorCritic(nn.Module):
    """Camera-only actor and training-only privileged critic."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        policy_auxiliary_size: int,
        privileged_size: int,
        action_size: int,
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.policy_auxiliary_size = policy_auxiliary_size
        self.privileged_size = privileged_size
        self.actor = _CameraHead(
            image_shape,
            policy_auxiliary_size,
            action_size,
            output_gain=0.01,
        )
        self.critic = _CameraHead(
            image_shape,
            policy_auxiliary_size + privileged_size,
            1,
            output_gain=1.0,
        )
        self.log_std = nn.Parameter(torch.full((action_size,), -0.8))

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        mean = self.actor(observation)
        distribution = Normal(mean, self.log_std.exp().expand_as(mean))
        if action is None:
            raw_action = distribution.rsample()
            action = torch.tanh(raw_action)
        else:
            action = action.clamp(-0.999, 0.999)
            raw_action = torch.atanh(action)

        log_probability = distribution.log_prob(raw_action).sum(dim=-1)
        log_probability -= torch.log(
            1.0 - action.square() + 1e-6
        ).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return (
            action,
            log_probability,
            entropy,
            self.get_value(observation),
        )
