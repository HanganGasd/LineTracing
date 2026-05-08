from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import LineTracingCameraEnv


def main():
    # 학습용 환경: 화면 안 띄움
    env = LineTracingCameraEnv(render_mode=False)
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cpu",   # MLPPolicy는 보통 CPU가 더 안정적
    )

    model.learn(total_timesteps=300_000)

    model.save("ppo_line_tracing_new")

    env.close()
    print("학습 완료: ppo_line_tracing_new.zip 저장됨")


if __name__ == "__main__":
    main()