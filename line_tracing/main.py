import pygame
import numpy as np
import time
from environment import LineTracingCameraEnv


env = LineTracingCameraEnv()
observation = env.reset()

CONTROL_HZ = 10
CONTROL_DT = 1.0 / CONTROL_HZ

episode = 1
MAX_EPISODES = 5

running = True

while running:
    loop_start = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 아직 PPO 모델이 없으므로 랜덤 action 사용
    steering = np.random.uniform(-1.0, 1.0)
    throttle = np.random.uniform(0.3, 1.0)

    action = np.array([steering, throttle], dtype=np.float32)

    observation, reward, done = env.step(action)

    print(
        "episode:", episode,
        "step:", env.step_count,
        "action:", action,
        "reward:", round(reward, 3),
        "done:", done
    )

    env.render()

    if done:
        print(f"Episode {episode} finished at step {env.step_count}.")

        episode += 1

        if episode > MAX_EPISODES:
            print("Test finished.")
            running = False
        else:
            observation = env.reset()

    elapsed = time.time() - loop_start

    if elapsed > CONTROL_DT:
        print("제어 주기 초과:", round(elapsed, 4), "초")
    else:
        time.sleep(CONTROL_DT - elapsed)

env.close()