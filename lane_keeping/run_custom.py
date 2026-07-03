import os
import pygame
import torch
import numpy as np

from environment import LineTracingCameraEnv
from custom_ppo import ActorCritic


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "custom_ppo_lane_keeping.pt"


def load_model(model, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)

    # 그냥 state_dict로 저장한 경우 / checkpoint dict로 저장한 경우 둘 다 대응
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # DataParallel 저장 모델이면 module. 제거
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            fixed_state_dict[key[len("module."):]] = value
        else:
            fixed_state_dict[key] = value

    model.load_state_dict(fixed_state_dict)
    model.eval()


def get_action(model, obs):
    obs_tensor = torch.tensor(
        obs,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    with torch.no_grad():
        raw_mean_action = model.actor(obs_tensor)

        # 중요:
        # 학습 때 action = tanh(raw_action)이었으므로
        # 실행 때도 tanh(mean)을 사용해야 함.
        action = torch.tanh(raw_mean_action)

    action = action.squeeze(0).cpu().numpy()
    action = np.clip(action, -1.0, 1.0)

    return action


def main():
    env = LineTracingCameraEnv(render_mode=True)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, action_dim).to(DEVICE)
    load_model(model, MODEL_PATH)

    print(f"모델 불러옴: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Obs dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print("ESC: 종료 / R: 현재 에피소드 리셋")

    obs, info = env.reset()

    episode = 1
    total_reward = 0.0
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_r:
                        print("[Manual Reset]")
                        obs, info = env.reset()
                        total_reward = 0.0

            action = get_action(model, obs)

            obs, reward, terminated, truncated, info = env.step(action)

            env.render()
            env.clock.tick(30)

            total_reward += reward

            if info["step_count"] % 10 == 0:
                print(
                    f"Ep {episode} | "
                    f"Track {info.get('track_id', -1)} | "
                    f"Start {info.get('start_id', -1)} | "
                    f"Orig {info.get('start_original_id', -1)} | "
                    f"Step {info['step_count']} | "
                    f"Action [{action[0]:.3f}, {action[1]:.3f}] | "
                    f"Steer {info['steering']:.3f} | "
                    f"Throttle {info['throttle']:.3f} | "
                    f"Reward {reward:.3f} | "
                    f"Total {total_reward:.1f} | "
                    f"Dist {info.get('total_distance', 0):.0f} | "
                    f"Lap {info.get('lap_progress', 0):.0f}/"
                    f"{info.get('track_length', 1):.0f} | "
                    f"CenterDist {info.get('dist_to_center', 0):.1f} | "
                    f"ValidRows {info.get('valid_lane_rows', -1)}"
                )

            if terminated or truncated:
                done_reason = info.get("done_reason", "none")
                total_distance = info.get("total_distance", 0.0)

                print("=" * 70)
                print(
                    f"Episode {episode} finished | "
                    f"Track {info.get('track_id', -1)} | "
                    f"Start {info.get('start_id', -1)} | "
                    f"Orig {info.get('start_original_id', -1)} | "
                    f"Length {info['step_count']} | "
                    f"Total Reward {total_reward:.1f} | "
                    f"Reason {done_reason} | "
                    f"TotalDist {total_distance:.0f} | "
                    f"LapProgress {info.get('lap_progress', 0):.0f}/"
                    f"{info.get('track_length', 1):.0f}"
                )
                print("=" * 70)

                episode += 1
                total_reward = 0.0
                obs, info = env.reset()

    finally:
        env.close()


if __name__ == "__main__":
    main()