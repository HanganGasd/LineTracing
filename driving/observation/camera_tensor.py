"""Portable RGB camera observation used by simulation and real vehicles."""

from __future__ import annotations

import cv2
import numpy as np

from driving.types import VehicleState


class CameraTensorEncoder:
    """Convert RGB frames to the flat input consumed by CameraActorCritic."""

    def __init__(
        self,
        *,
        width: int = 80,
        height: int = 60,
        auxiliary_fields: tuple[str, ...] = (
            "previous_steering",
            "speed_normalized",
        ),
    ) -> None:
        self.width = width
        self.height = height
        self.auxiliary_fields = auxiliary_fields

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (3, self.height, self.width)

    @property
    def auxiliary_size(self) -> int:
        return len(self.auxiliary_fields)

    @property
    def observation_size(self) -> int:
        return int(np.prod(self.image_shape)) + self.auxiliary_size

    def encode(self, image: np.ndarray, state: VehicleState) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an HxWx3 RGB image.")
        if (image.shape[1], image.shape[0]) != (self.width, self.height):
            image = cv2.resize(
                image,
                (self.width, self.height),
                interpolation=cv2.INTER_AREA,
            )
        pixels = (
            np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
            .reshape(-1)
            / 255.0
        )
        auxiliary = np.asarray(
            [getattr(state, field) for field in self.auxiliary_fields],
            dtype=np.float32,
        )
        return np.concatenate((pixels, auxiliary)).astype(
            np.float32, copy=False
        )
