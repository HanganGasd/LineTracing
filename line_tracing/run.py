"""Run the trained line-tracing camera actor in simulation."""

from driving.simulation import CameraObservationWrapper
from line_tracing.environment import LineTracingCameraEnv
from rl_core.checkpoint import model_path
from rl_core.runner import run


MODEL_PATH = model_path(__file__, "camera_ppo_line_tracing_v2.pt")


def make_env() -> CameraObservationWrapper:
    return CameraObservationWrapper(LineTracingCameraEnv(render_mode=True))


if __name__ == "__main__":
    run(make_env, MODEL_PATH)
