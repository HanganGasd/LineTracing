import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque, defaultdict

from environment import LineTracingCameraEnv
from custom_ppo import ActorCritic


# =========================
# 설정
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_ENVS = 4
ROLLOUT_STEPS = 512
TOTAL_UPDATES = 1000

GAMMA = 0.99
GAE_LAMBDA = 0.95

PPO_EPOCHS = 4
MINI_BATCH_SIZE = 256
CLIP_EPS = 0.15

ACTOR_LR = 3e-4
CRITIC_LR = 1e-3

VALUE_COEF = 0.5
ENTROPY_COEF = 0.01

#처음부터 학습할 때 최소 표준편차 유지를 위함(기존 모델에 이어서 학습하려면 제거)
LOG_STD_MIN = -1.8
LOG_STD_MAX = -0.3

MODEL_PATH = "custom_ppo_line_tracing.pt"

def get_action_and_value(model, obs_tensor):
    """
    obs_tensor: [num_envs, obs_dim]
    """
    mean = model.actor(obs_tensor)

    # model.log_std가 있다고 가정
    std = torch.exp(model.log_std).expand_as(mean)

    dist = Normal(mean, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)

    value = model.critic(obs_tensor).squeeze(-1)

    action = torch.clamp(action, -1.0, 1.0)

    return action, log_prob, entropy, value


def evaluate_action(model, obs_tensor, action_tensor):
    """
    PPO update 때 기존 action에 대한 새 log_prob/value/entropy 계산
    """
    mean = model.actor(obs_tensor)
    std = torch.exp(model.log_std).expand_as(mean)

    dist = Normal(mean, std)

    log_prob = dist.log_prob(action_tensor).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)
    value = model.critic(obs_tensor).squeeze(-1)

    return log_prob, entropy, value


def compute_gae(rewards, dones, values, next_values):
    """
    rewards: [T, N]
    dones: [T, N]
    values: [T, N]
    next_values: [N]
    """
    T, N = rewards.shape

    advantages = torch.zeros_like(rewards).to(DEVICE)
    last_gae = torch.zeros(N, device=DEVICE)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = next_values
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + GAMMA * next_value * next_non_terminal - values[t]
        last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def main():
    # =========================
    # 병렬 환경 생성
    # =========================
    envs = [
        LineTracingCameraEnv(render_mode=False)
        for _ in range(NUM_ENVS)
    ]

    obs_list = []
    for env in envs:
        obs, info = env.reset()
        obs_list.append(obs)

    obs = np.stack(obs_list)

    obs_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]

    model = ActorCritic(obs_dim, action_dim).to(DEVICE)
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

    # 기존 단일 트랙 성공 모델에서 이어 학습하고 싶으면 로드
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"기존 모델 불러옴: {MODEL_PATH}")
    except FileNotFoundError:
        print("기존 모델 없음. 새 모델로 시작.")

    recent_lengths = deque(maxlen=20)
    recent_rewards = deque(maxlen=20)

    track_stats = defaultdict(lambda: {"count": 0, "len_sum": 0, "success": 0})
    combo_stats = defaultdict(lambda: {"count": 0, "len_sum": 0, "success": 0})

    episode_count = 0
    global_step = 0

    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    episode_lengths = np.zeros(NUM_ENVS, dtype=np.int32)

    for update in range(1, TOTAL_UPDATES + 1):
        obs_buffer = []
        action_buffer = []
        logprob_buffer = []
        reward_buffer = []
        done_buffer = []
        value_buffer = []

        for step in range(ROLLOUT_STEPS):
            global_step += NUM_ENVS

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                action_tensor, log_prob_tensor, _, value_tensor = get_action_and_value(
                    model,
                    obs_tensor
                )

            actions = action_tensor.cpu().numpy()

            next_obs_list = []
            rewards = []
            dones = []

            for env_idx, env in enumerate(envs):
                next_obs, reward, terminated, truncated, info = env.step(actions[env_idx])
                done = terminated or truncated

                episode_rewards[env_idx] += reward
                episode_lengths[env_idx] += 1

                if done:
                    episode_count += 1

                    ep_len = episode_lengths[env_idx]
                    ep_rew = episode_rewards[env_idx]

                    track_id = info.get("track_id", -1)
                    start_id = info.get("start_id", -1)
                    done_reason = info.get("done_reason", "none")
                    total_dist = info.get("total_distance", 0.0)

                    recent_lengths.append(ep_len)
                    recent_rewards.append(ep_rew)

                    track_stats[track_id]["count"] += 1
                    track_stats[track_id]["len_sum"] += ep_len

                    if done_reason == "max_steps":
                        track_stats[track_id]["success"] += 1
                    
                    combo_key = (track_id, start_id)

                    combo_stats[combo_key]["count"] += 1
                    combo_stats[combo_key]["len_sum"] += ep_len

                    if done_reason == "max_steps":
                        combo_stats[combo_key]["success"] += 1

                    avg_len20 = np.mean(recent_lengths)
                    avg_rew20 = np.mean(recent_rewards)

                    print(
                        f"Ep {episode_count} | "
                        f"Env {env_idx} | "
                        f"Track {track_id} | "
                        f"Start {start_id} | "
                        f"Orig {info.get('start_original_id', -1)} | "
                        f"Step {global_step} | "
                        f"Len {ep_len} | "
                        f"Reward {ep_rew:.1f} | "
                        f"AvgLen20 {avg_len20:.1f} | "
                        f"Reason {done_reason} | "
                        f"Dist {total_dist:.0f}"
                    )

                    next_obs, reset_info = env.reset()
                    episode_rewards[env_idx] = 0.0
                    episode_lengths[env_idx] = 0

                next_obs_list.append(next_obs)
                rewards.append(reward)
                dones.append(float(done))

            obs_buffer.append(obs)
            action_buffer.append(actions)
            logprob_buffer.append(log_prob_tensor.cpu().numpy())
            reward_buffer.append(rewards)
            done_buffer.append(dones)
            value_buffer.append(value_tensor.cpu().numpy())

            obs = np.stack(next_obs_list)

        # =========================
        # buffer tensor 변환
        # =========================
        obs_tensor = torch.tensor(np.array(obs_buffer), dtype=torch.float32, device=DEVICE)
        action_tensor = torch.tensor(np.array(action_buffer), dtype=torch.float32, device=DEVICE)
        old_logprob_tensor = torch.tensor(np.array(logprob_buffer), dtype=torch.float32, device=DEVICE)
        reward_tensor = torch.tensor(np.array(reward_buffer), dtype=torch.float32, device=DEVICE)
        done_tensor = torch.tensor(np.array(done_buffer), dtype=torch.float32, device=DEVICE)
        value_tensor = torch.tensor(np.array(value_buffer), dtype=torch.float32, device=DEVICE)

        # next value
        with torch.no_grad():
            next_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            next_value = model.critic(next_obs_tensor).squeeze(-1)

        advantages, returns = compute_gae(
            reward_tensor,
            done_tensor,
            value_tensor,
            next_value
        )

        # =========================
        # flatten: [T, N, ...] -> [T*N, ...]
        # =========================
        b_obs = obs_tensor.reshape(ROLLOUT_STEPS * NUM_ENVS, obs_dim)
        b_actions = action_tensor.reshape(ROLLOUT_STEPS * NUM_ENVS, action_dim)
        b_old_logprobs = old_logprob_tensor.reshape(ROLLOUT_STEPS * NUM_ENVS)
        b_advantages = advantages.reshape(ROLLOUT_STEPS * NUM_ENVS)
        b_returns = returns.reshape(ROLLOUT_STEPS * NUM_ENVS)
        b_values = value_tensor.reshape(ROLLOUT_STEPS * NUM_ENVS)

        # advantage normalize
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = ROLLOUT_STEPS * NUM_ENVS
        indices = np.arange(batch_size)

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0

        # =========================
        # PPO update
        # =========================
        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(indices)

            for start in range(0, batch_size, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_logprobs = b_old_logprobs[mb_idx]
                mb_advantages = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]

                new_logprobs, entropy, new_values = evaluate_action(
                    model,
                    mb_obs,
                    mb_actions
                )

                ratio = torch.exp(new_logprobs - mb_old_logprobs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - CLIP_EPS,
                    1.0 + CLIP_EPS
                ) * mb_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, mb_returns)
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
                with torch.no_grad():
                    model.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)

                last_policy_loss = policy_loss.item()
                last_value_loss = value_loss.item()
                last_entropy = entropy_loss.item()

        # =========================
        # EV 계산
        # =========================
        with torch.no_grad():
            y_true = b_returns
            y_pred = b_values
            var_y = torch.var(y_true)
            ev = 1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)

        print(
            f"[Update {update}] "
            f"Step {global_step} | "
            f"Policy {last_policy_loss:.4f} | "
            f"Value {last_value_loss:.4f} | "
            f"EV {ev.item():.3f} | "
            f"Entropy {last_entropy:.3f} | "
            f"AvgLen20 {np.mean(recent_lengths) if len(recent_lengths) > 0 else 0:.1f}"
        )

        if update % 10 == 0:
            print("---- Track Avg ----")
            for tid in sorted(track_stats.keys()):
                c = track_stats[tid]["count"]
                avg_len = track_stats[tid]["len_sum"] / max(c, 1)
                success = track_stats[tid]["success"]
                success_rate = success / max(c, 1) * 100.0
                print(
                    f"Track {tid}: "
                    f"Count {c}, "
                    f"AvgLen {avg_len:.1f}, "
                    f"Success {success_rate:.1f}%"
                )

            print("---- Track-Start Avg ----")
            for (tid, sid), stat in sorted(combo_stats.items()):
                c = stat["count"]

                # 너무 적게 나온 조합은 통계 의미가 약하니까 2번 이상 나온 것만 출력
                if c < 2:
                    continue

                avg_len = stat["len_sum"] / max(c, 1)
                success_rate = stat["success"] / max(c, 1) * 100.0

                print(
                    f"Track {tid}, Start {sid}: "
                    f"Count {c}, "
                    f"AvgLen {avg_len:.1f}, "
                    f"Success {success_rate:.1f}%"
                )

            torch.save(model.state_dict(), MODEL_PATH)
            print(f"모델 저장: {MODEL_PATH}")

    torch.save(model.state_dict(), MODEL_PATH)

    for env in envs:
        env.close()


if __name__ == "__main__":
    main()