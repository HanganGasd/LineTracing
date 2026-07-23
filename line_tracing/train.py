"""Train the asymmetric camera PPO line-tracing policy."""

from driving.training import TrainingCameraWrapper
from line_tracing.environment import LineTracingCameraEnv
from rl_core.checkpoint import model_path
from rl_core.training import train


MODEL_PATH = model_path(__file__, "camera_ppo_line_tracing_v2.pt")


def make_env() -> TrainingCameraWrapper:
    return TrainingCameraWrapper(
        LineTracingCameraEnv(render_mode=False)
    )


if __name__ == "__main__":
    train(make_env, MODEL_PATH)
