"""OpenCV camera and recorded-video sources."""

from __future__ import annotations

import time

import cv2

from driving.types import SensorFrame


class OpenCvCamera:
    """USB/V4L camera source returning RGB frames."""

    def __init__(
        self,
        device: int | str = 0,
        *,
        width: int = 160,
        height: int = 120,
        fps: int = 15,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self.device)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open camera {self.device!r}.")
        self._capture = capture

    def read(self) -> SensorFrame:
        if self._capture is None:
            raise RuntimeError("Camera is not open.")
        ok, bgr = self._capture.read()
        if not ok:
            raise RuntimeError("Camera frame capture failed.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return SensorFrame(rgb, time.monotonic())

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class VideoFileCamera(OpenCvCamera):
    """Recorded video source for testing the real-vehicle pipeline on a PC."""

    def __init__(self, path: str, *, loop: bool = False) -> None:
        super().__init__(path)
        self.loop = loop

    def read(self) -> SensorFrame:
        try:
            return super().read()
        except RuntimeError:
            if not self.loop or self._capture is None:
                raise EOFError("End of video.")
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return super().read()
