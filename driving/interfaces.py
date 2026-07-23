"""Protocols implemented by simulation, video, and Raspberry Pi adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import ControlCommand, SensorFrame, VehicleState


@runtime_checkable
class CameraSource(Protocol):
    def open(self) -> None: ...

    def read(self) -> SensorFrame: ...

    def close(self) -> None: ...


@runtime_checkable
class ObservationEncoder(Protocol):
    @property
    def observation_size(self) -> int: ...

    def encode(
        self,
        image: np.ndarray,
        state: VehicleState,
    ) -> np.ndarray: ...


@runtime_checkable
class Policy(Protocol):
    def predict(self, observation: np.ndarray) -> ControlCommand: ...


@runtime_checkable
class VehicleController(Protocol):
    def open(self) -> None: ...

    def apply(self, command: ControlCommand) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...
