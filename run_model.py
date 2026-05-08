from stable_baselines3 import PPO
from environment import LineTracingCameraEnv


def main():
    # 환경 생성
    env = LineTracingCameraEnv(render_mode=True)

    # 학습된 PPO 모델 불러오기
    model = PPO.load("ppo_line_tracing_new")

    # 환경 초기화
    obs, info = env.reset()

    episode = 1
    total_reward = 0

    while True:
        # pygame 창 닫기 이벤트 처리
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        # 현재 observation을 보고 PPO가 action 예측
        action, _states = model.predict(obs, deterministic=True)

        # action을 환경에 넣어서 자동차 움직이기
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # 화면 출력
        env.render()

        print(
            f"Episode: {episode}, "
            f"Step: {info['step_count']}, "
            f"Steering: {info['steering']:.3f}, "
            f"Throttle: {info['throttle']:.3f}, "
            f"Reward: {reward:.3f}, "
            f"Total Reward: {total_reward:.3f}"
        )

        # 에피소드 종료 시 다시 시작
        if terminated or truncated:
            print("=" * 60)
            print(f"Episode {episode} 종료")
            print(f"Total Reward: {total_reward:.3f}")
            print("=" * 60)

            episode += 1
            total_reward = 0
            obs, info = env.reset()


if __name__ == "__main__":
    main()