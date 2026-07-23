"""Picamera2 adapter, importable on development machines without Picamera2."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from driving.types import SensorFrame


class RaspberryPiCamera:
    def __init__(
        self,
        *,
        width: int = 160,
        height: int = 120,
        fps: int = 15,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self._camera: Any = None

    def open(self) -> None:
        try:
            from picamera2 import Picamera2
            from libcamera import Transform
        except ImportError as error:
            raise RuntimeError(
                "Picamera2 is required on Raspberry Pi. "
                "Install it with the Raspberry Pi OS package manager."
            ) from error

        camera = Picamera2()
        config = camera.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.fps},
            transform=Transform(
                hflip=self.horizontal_flip,
                vflip=self.vertical_flip,
            ),
        )
        camera.configure(config)
        camera.start()
        self._camera = camera

    def read(self) -> SensorFrame:
        if self._camera is None:
            raise RuntimeError("Camera is not open.")
        image = np.asarray(self._camera.capture_array("main"))
        return SensorFrame(
            np.ascontiguousarray(image[:, :, :3], dtype=np.uint8),
            time.monotonic(),
        )

    def close(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera.close()
            self._camera = None
