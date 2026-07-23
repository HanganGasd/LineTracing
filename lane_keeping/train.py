"""Train the asymmetric camera PPO lane-keeping policy."""

from driving.training import TrainingCameraWrapper
from lane_keeping.environment import LaneKeepingEnv
from rl_core.checkpoint import model_path
from rl_core.training import train


MODEL_PATH = model_path(__file__, "camera_ppo_lane_keeping_v2.pt")


def make_env() -> TrainingCameraWrapper:
    return TrainingCameraWrapper(LaneKeepingEnv(render_mode=False))


if __name__ == "__main__":
    train(make_env, MODEL_PATH)
