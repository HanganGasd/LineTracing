"""Adapters that expose existing Gym driving environments as camera policies."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from driving.observation.camera_tensor import CameraTensorEncoder
from driving.types import VehicleState


class CameraObservationWrapper(gym.Wrapper):
    """Replace policy observations with RGB pixels without changing the env.

    The wrapped environment still computes its original CV observation for
    rewards, termination and diagnostics. Only the policy-facing observation
    changes, so the existing task definition and PPO algorithm are preserved.
    """

    def __init__(
        self,
        env: gym.Env,
        encoder: CameraTensorEncoder | None = None,
    ) -> None:
        super().__init__(env)
        self.encoder = encoder or CameraTensorEncoder()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.encoder.observation_size,),
            dtype=np.float32,
        )

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return self.encoder.image_shape

    @property
    def auxiliary_size(self) -> int:
        return self.encoder.auxiliary_size

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._camera_observation(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._camera_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def _camera_observation(self) -> np.ndarray:
        image = self.env.unwrapped.get_camera_image()
        base = self.env.unwrapped
        state = VehicleState(
            speed_normalized=float(
                base.current_speed / max(base.max_speed, 1e-6)
            ),
            previous_steering=float(base.prev_steering),
        )
        return self.encoder.encode(image, state)
