"""Run the trained lane-keeping camera actor in simulation."""

from driving.simulation import CameraObservationWrapper
from lane_keeping.environment import LaneKeepingEnv
from rl_core.checkpoint import model_path
from rl_core.runner import run


MODEL_PATH = model_path(__file__, "camera_ppo_lane_keeping_v2.pt")


def make_env() -> CameraObservationWrapper:
    return CameraObservationWrapper(LaneKeepingEnv(render_mode=True))


if __name__ == "__main__":
    run(make_env, MODEL_PATH)
