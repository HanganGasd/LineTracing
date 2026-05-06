from stable_baselines3 import PPO
from environment import LineTracingCameraEnv


env = LineTracingCameraEnv()
model = PPO.load("ppo_line_tracing")

obs, info = env.reset()

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    print(
        "step:", step,
        "steering:", info["steering"],
        "throttle:", info["throttle"],
        "reward:", reward
    )

    if terminated or truncated:
        obs, info = env.reset()