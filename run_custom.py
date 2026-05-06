import pygame
import torch
import numpy as np

from environment import LineTracingCameraEnv
from custom_ppo import ActorCritic


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "custom_ppo_line_tracing.pt"


def main():
    env = LineTracingCameraEnv()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, action_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"모델 불러옴: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    obs, info = env.reset()

    episode = 1
    total_reward = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        obs_tensor = torch.tensor(
            obs,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        # 테스트는 deterministic하게 actor 평균 action만 사용
        with torch.no_grad():
            mean_action = model.actor(obs_tensor)

        action = mean_action.squeeze(0).cpu().numpy()
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        env.clock.tick(30)

        total_reward += reward

        # 10 step마다만 출력해서 로그 줄이기
        if info["step_count"] % 10 == 0:
            print(
                f"Ep {episode} | "
                f"Track {info.get('track_id', -1)} | "
                f"Start {info.get('start_id', -1)} | "
                f"Orig {info.get('start_original_id', -1)} | "
                f"Step {info['step_count']} | "
                f"Steer {info['steering']:.3f} | "
                f"Throttle {info['throttle']:.3f} | "
                f"Reward {reward:.3f} | "
                f"Total {total_reward:.1f} | "
                f"Dist {info.get('total_distance', 0):.0f}"
            )

        if terminated or truncated:
            done_reason = info.get("done_reason", "none")
            total_distance = info.get("total_distance", 0.0)
            track_id = info.get("track_id", -1)
            start_id = info.get("start_id", -1)
            start_original_id = info.get("start_original_id", -1)

            print("=" * 70)
            print(
                f"Episode {episode} finished | "
                f"Track {track_id} | "
                f"Start {start_id} | "
                f"Orig {start_original_id} | "
                f"Length {info['step_count']} | "
                f"Total Reward {total_reward:.1f} | "
                f"Reason {done_reason} | "
                f"TotalDist {total_distance:.0f}"
            )
            print("=" * 70)

            episode += 1
            total_reward = 0.0
            obs, info = env.reset()


if __name__ == "__main__":
    main()