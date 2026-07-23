"""Shared data contracts across simulation and real hardware."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """One RGB camera frame and its monotonic capture timestamp."""

    image: np.ndarray
    timestamp: float

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("SensorFrame.image must be an HxWx3 RGB array.")


@dataclass(frozen=True, slots=True)
class VehicleState:
    """Non-visual state supplied to an observation encoder."""

    speed_normalized: float = 0.0
    previous_steering: float = 0.0
    heading_normalized: float = 0.0
    yaw_rate_normalized: float = 0.0


@dataclass(frozen=True, slots=True)
class ControlCommand:
    """Normalized steering and throttle command."""

    steering: float
    throttle: float

    def clipped(
        self,
        *,
        max_steering: float = 1.0,
        min_throttle: float = -1.0,
        max_throttle: float = 1.0,
    ) -> "ControlCommand":
        return ControlCommand(
            steering=float(np.clip(self.steering, -max_steering, max_steering)),
            throttle=float(np.clip(self.throttle, min_throttle, max_throttle)),
        )
