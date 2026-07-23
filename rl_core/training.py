"""Stable PPO training loop for the asymmetric camera actor-critic."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from rl_core.ppo import CameraActorCritic


@dataclass(frozen=True)
class PpoConfig:
    num_envs: int = 4
    rollout_steps: int = 256
    total_updates: int = 1000
    update_epochs: int = 4
    minibatch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.15
    value_clip_epsilon: float = 0.20
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 5e-4
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    log_std_min: float = -2.0
    log_std_max: float = -0.3


def train(
    env_factory: Callable[[], gym.Env],
    checkpoint_path: str | Path,
    *,
    config: PpoConfig = PpoConfig(),
    device: str | None = None,
) -> None:
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if selected_device.type == "cuda":
        print(
            f"Device: {selected_device} | "
            f"GPU: {torch.cuda.get_device_name(selected_device)}"
        )
    else:
        print(f"Device: {selected_device}")

    envs = [env_factory() for _ in range(config.num_envs)]
    first = envs[0]
    model = CameraActorCritic(
        first.image_shape,
        first.auxiliary_size,
        first.privileged_size,
        first.action_space.shape[0],
    ).to(selected_device)
    checkpoint = Path(checkpoint_path)
    if checkpoint.exists():
        try:
            model.load_state_dict(
                torch.load(
                    checkpoint,
                    map_location=selected_device,
                    weights_only=True,
                )
            )
            print(f"Checkpoint loaded: {checkpoint}")
        except RuntimeError:
            print(
                f"Incompatible old checkpoint ignored: {checkpoint}"
            )

    optimizer = torch.optim.Adam(
        [
            {
                "params": list(model.actor.parameters())
                + [model.log_std],
                "lr": config.actor_learning_rate,
            },
            {
                "params": model.critic.parameters(),
                "lr": config.critic_learning_rate,
            },
        ]
    )
    observations = np.stack([env.reset()[0] for env in envs])
    episode_lengths = np.zeros(config.num_envs, dtype=np.int32)
    episode_rewards = np.zeros(config.num_envs, dtype=np.float32)
    recent_lengths: deque[int] = deque(maxlen=20)
    recent_rewards: deque[float] = deque(maxlen=20)
    episode_count = 0
    global_step = 0

    try:
        for update in range(1, config.total_updates + 1):
            shape = (
                config.rollout_steps,
                config.num_envs,
            )
            obs_buffer = np.empty(
                (*shape, observations.shape[-1]), dtype=np.float32
            )
            action_buffer = np.empty(
                (*shape, first.action_space.shape[0]), dtype=np.float32
            )
            logprob_buffer = np.empty(shape, dtype=np.float32)
            reward_buffer = np.empty(shape, dtype=np.float32)
            done_buffer = np.empty(shape, dtype=np.float32)
            value_buffer = np.empty(shape, dtype=np.float32)

            for step in range(config.rollout_steps):
                obs_tensor = torch.as_tensor(
                    observations, device=selected_device
                )
                with torch.no_grad():
                    actions, logprobs, _, values = (
                        model.get_action_and_value(obs_tensor)
                    )
                action_array = actions.cpu().numpy()
                obs_buffer[step] = observations
                action_buffer[step] = action_array
                logprob_buffer[step] = logprobs.cpu().numpy()
                value_buffer[step] = values.cpu().numpy()

                next_observations = []
                for index, env in enumerate(envs):
                    next_obs, reward, terminated, truncated, info = (
                        env.step(action_array[index])
                    )
                    done = terminated or truncated
                    episode_lengths[index] += 1
                    episode_rewards[index] += reward
                    reward_buffer[step, index] = reward
                    done_buffer[step, index] = float(done)
                    if done:
                        episode_count += 1
                        length = int(episode_lengths[index])
                        total_reward = float(episode_rewards[index])
                        recent_lengths.append(length)
                        recent_rewards.append(total_reward)
                        print(
                            f"Episode {episode_count} | Env {index} | "
                            f"TotalSteps {global_step + index + 1} | "
                            f"EpisodeSteps {length} | "
                            f"Reward {total_reward:.2f} | "
                            f"Reason {info.get('done_reason', 'unknown')}"
                        )
                        episode_lengths[index] = 0
                        episode_rewards[index] = 0.0
                        next_obs, _ = env.reset()
                    next_observations.append(next_obs)
                observations = np.stack(next_observations)
                global_step += config.num_envs

            with torch.no_grad():
                next_values = model.get_value(
                    torch.as_tensor(
                        observations, device=selected_device
                    )
                )
            rewards = torch.as_tensor(
                reward_buffer, device=selected_device
            )
            dones = torch.as_tensor(done_buffer, device=selected_device)
            old_values = torch.as_tensor(
                value_buffer, device=selected_device
            )
            advantages = torch.zeros_like(rewards)
            last_advantage = torch.zeros(
                config.num_envs, device=selected_device
            )
            for step in reversed(range(config.rollout_steps)):
                following_value = (
                    next_values
                    if step == config.rollout_steps - 1
                    else old_values[step + 1]
                )
                non_terminal = 1.0 - dones[step]
                delta = (
                    rewards[step]
                    + config.gamma * following_value * non_terminal
                    - old_values[step]
                )
                last_advantage = (
                    delta
                    + config.gamma
                    * config.gae_lambda
                    * non_terminal
                    * last_advantage
                )
                advantages[step] = last_advantage
            returns = advantages + old_values

            batch_size = config.rollout_steps * config.num_envs
            batch_obs = torch.as_tensor(
                obs_buffer.reshape(batch_size, -1),
                device=selected_device,
            )
            batch_actions = torch.as_tensor(
                action_buffer.reshape(batch_size, -1),
                device=selected_device,
            )
            batch_logprobs = torch.as_tensor(
                logprob_buffer.reshape(-1), device=selected_device
            )
            batch_old_values = old_values.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_advantages = advantages.reshape(-1)
            batch_advantages = (
                batch_advantages - batch_advantages.mean()
            ) / (batch_advantages.std(unbiased=False) + 1e-8)

            indices = np.arange(batch_size)
            stopped_for_kl = False
            for _ in range(config.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, config.minibatch_size):
                    index = indices[start : start + config.minibatch_size]
                    _, new_logprobs, entropy, new_values = (
                        model.get_action_and_value(
                            batch_obs[index], batch_actions[index]
                        )
                    )
                    log_ratio = (
                        new_logprobs - batch_logprobs[index]
                    )
                    ratio = log_ratio.exp()
                    unclipped = ratio * batch_advantages[index]
                    clipped = torch.clamp(
                        ratio,
                        1.0 - config.clip_epsilon,
                        1.0 + config.clip_epsilon,
                    ) * batch_advantages[index]
                    policy_loss = -torch.min(
                        unclipped, clipped
                    ).mean()

                    old_value = batch_old_values[index]
                    clipped_value = old_value + torch.clamp(
                        new_values - old_value,
                        -config.value_clip_epsilon,
                        config.value_clip_epsilon,
                    )
                    value_loss = 0.5 * torch.maximum(
                        F.mse_loss(
                            new_values,
                            batch_returns[index],
                            reduction="none",
                        ),
                        F.mse_loss(
                            clipped_value,
                            batch_returns[index],
                            reduction="none",
                        ),
                    ).mean()
                    entropy_value = entropy.mean()
                    loss = (
                        policy_loss
                        + config.value_coefficient * value_loss
                        - config.entropy_coefficient * entropy_value
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                    optimizer.step()
                    with torch.no_grad():
                        model.log_std.clamp_(
                            config.log_std_min, config.log_std_max
                        )
                        approximate_kl = (
                            (ratio - 1.0) - log_ratio
                        ).mean()
                        clip_fraction = (
                            (ratio - 1.0).abs()
                            > config.clip_epsilon
                        ).float().mean()
                    if approximate_kl > config.target_kl:
                        stopped_for_kl = True
                        break
                if stopped_for_kl:
                    break

            with torch.no_grad():
                prediction = model.get_value(batch_obs)
                return_variance = batch_returns.var(unbiased=False)
                explained_variance = 1.0 - (
                    (batch_returns - prediction).var(unbiased=False)
                    / (return_variance + 1e-8)
                )

            if update % 10 == 0:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint)
            print(
                f"Update {update} | Step {global_step} | "
                f"Policy {policy_loss.item():.4f} | "
                f"Value {value_loss.item():.4f} | "
                f"EV {explained_variance.item():.3f} | "
                f"KL {approximate_kl.item():.5f} | "
                f"Clip {clip_fraction.item():.3f} | "
                f"Entropy {entropy_value.item():.3f} | "
                f"RewardMean {rewards.mean().item():.3f} | "
                f"ReturnMean {batch_returns.mean().item():.3f} | "
                f"VMean {prediction.mean().item():.3f} | "
                f"AvgEpReward20 "
                f"{np.mean(recent_rewards) if recent_rewards else 0.0:.2f} | "
                f"AvgLen20 "
                f"{np.mean(recent_lengths) if recent_lengths else 0.0:.1f}"
            )
    finally:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        for env in envs:
            env.close()
