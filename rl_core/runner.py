"""Deterministic simulator runner for the trained camera actor."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import torch

from rl_core.ppo import CameraActorCritic


def run(
    env_factory: Callable[[], gym.Env],
    checkpoint_path: str | Path,
    *,
    privileged_size: int = 3,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = env_factory()
    model = CameraActorCritic(
        env.image_shape,
        env.auxiliary_size,
        privileged_size,
        env.action_space.shape[0],
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    observation, _ = env.reset()
    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE
                ):
                    running = False
            tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.inference_mode():
                action = torch.tanh(model.actor(tensor))[0].cpu().numpy()
            observation, _, terminated, truncated, _ = env.step(
                np.clip(action, -1.0, 1.0)
            )
            env.render()
            env.unwrapped.clock.tick(30)
            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        env.close()
