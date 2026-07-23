"""Camera-only policy observations with privileged simulator critic state."""

from __future__ import annotations

from types import MethodType

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from driving.simulation import CameraObservationWrapper


class TrainingCameraWrapper(CameraObservationWrapper):
    """Use RGB for the actor and geometry only for reward and the critic."""

    privileged_size = 3

    def __init__(self, env: gym.Env) -> None:
        self._task = (
            "lane" if hasattr(env.unwrapped, "lane_scan_rows") else "line"
        )
        self._disable_legacy_cv(env.unwrapped)
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(
                self.encoder.observation_size + self.privileged_size,
            ),
            dtype=np.float32,
        )

    def step(self, action):
        previous_steering = float(self.env.unwrapped.prev_steering)
        _, _, terminated, truncated, info = self.env.step(action)
        reward = self._reward(
            steering=float(info["steering"]),
            throttle=float(info["throttle"]),
            previous_steering=previous_steering,
            terminated=terminated,
            truncated=truncated,
            reason=info.get("done_reason", "none"),
        )
        return (
            self._camera_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def _camera_observation(self) -> np.ndarray:
        policy_observation = super()._camera_observation()
        return np.concatenate(
            (policy_observation, self._privileged_state())
        ).astype(np.float32, copy=False)

    def _privileged_state(self) -> np.ndarray:
        base = self.env.unwrapped
        center_error = self._center_error()
        max_step_distance = max(base.max_speed * base.dt, 1e-6)
        progress = float(
            np.clip(
                base.last_progress_delta / max_step_distance,
                -1.0,
                1.0,
            )
        )
        lap_ratio = float(
            np.clip(
                base.lap_progress / max(base.track_length, 1e-6),
                0.0,
                1.0,
            )
        )
        return np.asarray(
            (center_error, progress, lap_ratio),
            dtype=np.float32,
        )

    def _center_error(self) -> float:
        base = self.env.unwrapped
        if self._task == "lane":
            distance = float(base.dist_to_center)
            allowed = max(base.road_width * 0.8, 1e-6)
        else:
            distance = float(
                base.get_distance_to_track_center(base.car_x, base.car_y)
            )
            allowed = max(
                base.line_width / 2.0 + base.off_line_margin,
                1e-6,
            )
        return float(np.clip(distance / allowed, 0.0, 2.0))

    def _reward(
        self,
        *,
        steering: float,
        throttle: float,
        previous_steering: float,
        terminated: bool,
        truncated: bool,
        reason: str,
    ) -> float:
        base = self.env.unwrapped
        progress = float(
            np.clip(
                base.last_progress_delta
                / max(base.max_speed * base.dt, 1e-6),
                -1.0,
                1.0,
            )
        )
        center_error = self._center_error()
        center_score = max(0.0, 1.0 - center_error)

        reward = progress
        reward += 0.20 * throttle * center_score
        reward -= 0.15 * center_error * center_error
        reward -= 0.02 * abs(steering)
        reward -= 0.04 * abs(steering - previous_steering)
        if throttle < 0.20:
            reward -= 0.05

        if terminated or truncated:
            if reason in {"max_steps", "lap_complete"}:
                reward += 10.0
            elif reason == "low_speed":
                reward -= 3.0
            else:
                reward -= 10.0
        return float(np.clip(reward, -10.0, 10.0))

    @staticmethod
    def _disable_legacy_cv(base) -> None:
        if hasattr(base, "lane_scan_rows"):
            size = base.lane_scan_rows * 2 + 2

            def empty_observation(self):
                return np.zeros(size, dtype=np.float32)

            def unused_reward(self, steering, throttle):
                return 0.0

            def no_cv_termination(self):
                return False

            base.get_observation = MethodType(empty_observation, base)
            base.calculate_reward = MethodType(unused_reward, base)
            base.is_off_road_by_lane_detection = MethodType(
                no_cv_termination, base
            )
            return

        size = (
            base.observation_bins * base.observation_rows
            + base.observation_rows
            + 2
        )

        def no_preprocess(self, image):
            return np.empty((1, 1), dtype=np.uint8)

        def empty_observation(self, binary):
            return np.zeros(size, dtype=np.float32)

        def unused_reward(self, binary, steering, throttle):
            return 0.0

        def no_cv_termination(self, binary):
            return False

        base.preprocess_camera_image = MethodType(no_preprocess, base)
        base.get_observation_from_binary = MethodType(
            empty_observation, base
        )
        base.calculate_reward_from_binary = MethodType(unused_reward, base)
        base.is_off_track_from_binary = MethodType(
            no_cv_termination, base
        )
