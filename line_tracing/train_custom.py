from collections import deque, defaultdict
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import os

from environment import LineTracingCameraEnv
from lt_ppo import ActorCritic

#.venv\Scripts\activate.ps1

# =========================
# 하이퍼파라미터
# =========================
TOTAL_TIMESTEPS = 100_000
ROLLOUT_STEPS = 2048
UPDATE_EPOCHS = 10
MINIBATCH_SIZE = 64

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2

ACTOR_LR = 1e-4
CRITIC_LR = 5e-4

VALUE_COEF = 0.5
ENTROPY_COEF = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "custom_ppo_line_tracing.pt"


def compute_gae(rewards, values, dones, next_value):
    """
    GAE: Generalized Advantage Estimation
    advantage를 안정적으로 계산하는 방법
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]

        delta = rewards[t] + GAMMA * next_values * next_non_terminal - values[t]
        last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def main():
    env = LineTracingCameraEnv()

    obs, info = env.reset()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, action_dim).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"기존 모델 불러옴: {MODEL_PATH}")
    else:
        print("기존 모델 없음. 새 모델로 학습 시작")

    optimizer = torch.optim.Adam([
        {
            "params": list(model.actor.parameters()) + [model.log_std],
            "lr": ACTOR_LR,
        },
        {
            "params": model.critic.parameters(),
            "lr": CRITIC_LR,
        },
    ])
    global_step = 0
    episode_reward = 0
    episode_count = 1

    recent_rewards = deque(maxlen=20)
    recent_lengths = deque(maxlen=20)
    episode_length = 0
    start_stats = defaultdict(lambda: {"count": 0, "len_sum": 0, "rew_sum": 0.0})

    while global_step < TOTAL_TIMESTEPS:
        obs_buffer = []
        action_buffer = []
        logprob_buffer = []
        reward_buffer = []
        done_buffer = []
        value_buffer = []

        # =========================
        # Rollout 수집
        # =========================
        steering_sum = 0.0
        throttle_sum = 0.0
        action_count = 0
        abs_steering_sum = 0.0
        steering_change_sum = 0.0
        max_abs_steering = 0.0
        prev_logged_steering = 0.0

        for step in range(ROLLOUT_STEPS):
            if global_step%5000 == 0:
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"모델 중간 저장: {MODEL_PATH}")
            global_step += 1

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                action_tensor, logprob_tensor, _, value_tensor = model.get_action_and_value(obs_tensor)

            action = action_tensor.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(action)

            steering = info["steering"]

            steering_sum += steering
            throttle_sum += info["throttle"]
            action_count += 1

            abs_steering_sum += abs(steering)
            steering_change_sum += abs(steering - prev_logged_steering)
            max_abs_steering = max(max_abs_steering, abs(steering))
            prev_logged_steering = steering

            done = terminated or truncated

            obs_buffer.append(obs)
            action_buffer.append(action)
            logprob_buffer.append(logprob_tensor.item())
            reward_buffer.append(reward)
            done_buffer.append(float(done))
            value_buffer.append(value_tensor.item())

            episode_reward += reward
            obs = next_obs
            episode_length += 1

            if done:
                recent_rewards.append(episode_reward)
                recent_lengths.append(episode_length)

                avg_reward = sum(recent_rewards) / len(recent_rewards)
                avg_length = sum(recent_lengths) / len(recent_lengths)
                reward_per_step = episode_reward / max(episode_length, 1)
                start_id = info.get("start_id", -1)
                done_reason = info.get("done_reason", "none")

                start_stats[start_id]["count"] += 1
                start_stats[start_id]["len_sum"] += episode_length
                start_stats[start_id]["rew_sum"] += episode_reward

                total_distance = info.get("total_distance", 0.0)
                done_reason = info.get("done_reason", "none")
                start_id = info.get("start_id", -1)
                track_id = info.get("track_id", -1)

                print(
                    f"Ep {episode_count} | "
                    f"Track {track_id} | "
                    f"Start {start_id} | "
                    f"Orig {info.get('start_original_id', -1)} | "
                    f"Step {global_step} | "
                    f"Len {episode_length} | "
                    f"Reward {episode_reward:.1f} | "
                    f"AvgLen20 {avg_length:.1f} | "
                    f"Reason {done_reason} | "
                    f"Dist {total_distance:.0f}"
                )

                if episode_count % 100 == 0:
                    print("---- Start Avg ----")
                    for sid in sorted(start_stats.keys()):
                        c = start_stats[sid]["count"]
                        avg_len = start_stats[sid]["len_sum"] / max(c, 1)
                        print(f"Start {sid}: Len {avg_len:.1f}")

                episode_count += 1
                episode_reward = 0.0
                episode_length = 0
                obs, info = env.reset()

            if global_step >= TOTAL_TIMESTEPS:
                break

        # =========================
        # 다음 상태 value 계산
        # =========================
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, _, _, next_value_tensor = model.get_action_and_value(obs_tensor)
            next_value = next_value_tensor.item()

        rewards = np.array(reward_buffer, dtype=np.float32)
        values = np.array(value_buffer, dtype=np.float32)
        dones = np.array(done_buffer, dtype=np.float32)

        advantages, returns = compute_gae(rewards, values, dones, next_value)

        # advantage 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # =========================
        # Tensor 변환
        # =========================
        obs_tensor = torch.tensor(np.array(obs_buffer), dtype=torch.float32, device=DEVICE)
        action_tensor = torch.tensor(np.array(action_buffer), dtype=torch.float32, device=DEVICE)
        old_logprob_tensor = torch.tensor(np.array(logprob_buffer), dtype=torch.float32, device=DEVICE)
        advantage_tensor = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        return_tensor = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        batch_size = len(obs_buffer)
        indices = np.arange(batch_size)

        # =========================
        # PPO 업데이트
        # =========================

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        current_std = torch.exp(model.log_std).detach().cpu().numpy()
        avg_steering = steering_sum / max(action_count, 1)
        avg_abs_steering = abs_steering_sum / max(action_count, 1)
        avg_steering_change = steering_change_sum / max(action_count, 1)
        avg_throttle = throttle_sum / max(action_count, 1)
        
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(indices)

            for start in range(0, batch_size, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_idx = indices[start:end]

                mb_obs = obs_tensor[mb_idx]
                mb_actions = action_tensor[mb_idx]
                mb_old_logprob = old_logprob_tensor[mb_idx]
                mb_advantages = advantage_tensor[mb_idx]
                mb_returns = return_tensor[mb_idx]

                _, new_logprob, entropy, new_value = model.get_action_and_value(
                    mb_obs,
                    mb_actions
                )

                ratio = torch.exp(new_logprob - mb_old_logprob)

                unclipped_loss = ratio * mb_advantages
                clipped_loss = torch.clamp(
                    ratio,
                    1.0 - CLIP_COEF,
                    1.0 + CLIP_COEF
                ) * mb_advantages

                policy_loss = -torch.min(unclipped_loss, clipped_loss).mean()

                value_loss = ((new_value - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + VALUE_COEF * value_loss
                    - ENTROPY_COEF * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                last_policy_loss = policy_loss.item()
                last_value_loss = value_loss.item()
                last_entropy = entropy_loss.item()

        avg_steering = steering_sum / max(action_count, 1)
        avg_throttle = throttle_sum / max(action_count, 1)

        with torch.no_grad():
            _, _, _, all_values = model.get_action_and_value(obs_tensor)

            return_mean = return_tensor.mean().item()
            return_std = return_tensor.std().item()

            value_mean = all_values.mean().item()
            value_std = all_values.std().item()

            value_error = (all_values - return_tensor).abs().mean().item()

            var_return = torch.var(return_tensor)
            if var_return.item() == 0:
                explained_var = 0.0
            else:
                explained_var = (
                    1.0 - torch.var(return_tensor - all_values) / (var_return + 1e-8)
                ).item()

        print(
            f"[Update] Step {global_step} | "
            f"Policy {policy_loss.item():.4f} | "
            f"Value {value_loss.item():.4f} | "
            f"EV {explained_var:.3f} | "
            f"Entropy {entropy_loss.item():.3f} | "
            f"Throttle {avg_throttle:.3f} | "
            f"AbsSteer {avg_abs_steering:.3f} | "
            f"Std {current_std}"
            f"Start {start_id} | "
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"직접 구현한 PPO 모델 저장 완료: {MODEL_PATH}")

    env.close()


if __name__ == "__main__":
    main()