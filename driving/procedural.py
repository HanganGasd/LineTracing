"""Procedural tracks and camera domain randomization for driving tasks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def generate_closed_track(
    rng: np.random.Generator,
    *,
    width: int,
    height: int,
    margin: float,
    num_points: int = 240,
) -> list[tuple[int, int]]:
    """Generate either an organic loop or a rounded 90-degree-corner track."""
    if rng.random() < 0.55:
        return _generate_corner_track(
            rng,
            width=width,
            height=height,
            margin=margin,
            num_points=num_points,
        )
    return _generate_organic_track(
        rng,
        width=width,
        height=height,
        margin=margin,
        num_points=num_points,
    )


def _generate_organic_track(
    rng: np.random.Generator,
    *,
    width: int,
    height: int,
    margin: float,
    num_points: int,
) -> list[tuple[int, int]]:
    """Generate a smooth, non-self-intersecting, star-shaped loop."""
    center_x = width * 0.5 + rng.uniform(-35.0, 35.0)
    center_y = height * 0.5 + rng.uniform(-20.0, 20.0)
    radius_x = rng.uniform(width * 0.27, width * 0.34)
    radius_y = rng.uniform(height * 0.25, height * 0.34)
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    radial = np.ones_like(angles)
    for frequency, amplitude in ((2, 0.12), (3, 0.10), (4, 0.06)):
        radial += rng.uniform(-amplitude, amplitude) * np.sin(
            frequency * angles + rng.uniform(0.0, 2.0 * np.pi)
        )
    radial = np.clip(radial, 0.72, 1.22)
    rotation = rng.uniform(-0.25, 0.25)
    local_x = radius_x * radial * np.cos(angles)
    local_y = radius_y * radial * np.sin(angles)
    x = center_x + np.cos(rotation) * local_x - np.sin(rotation) * local_y
    y = center_y + np.sin(rotation) * local_x + np.cos(rotation) * local_y
    x = np.clip(x, margin, width - margin)
    y = np.clip(y, margin, height - margin)
    return list(zip(np.rint(x).astype(int), np.rint(y).astype(int)))


def _generate_corner_track(
    rng: np.random.Generator,
    *,
    width: int,
    height: int,
    margin: float,
    num_points: int,
) -> list[tuple[int, int]]:
    """Generate long straights joined by rounded, approximately 90° turns."""
    center_x = width * 0.5 + rng.uniform(-25.0, 25.0)
    center_y = height * 0.5 + rng.uniform(-15.0, 15.0)
    radius_x = rng.uniform(width * 0.28, width * 0.34)
    radius_y = rng.uniform(height * 0.27, height * 0.34)
    # A high superellipse exponent creates straights and 90-degree corners.
    exponent = rng.uniform(4.0, 7.0)
    power = 2.0 / exponent
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    local_x = radius_x * np.sign(cos_a) * np.abs(cos_a) ** power
    local_y = radius_y * np.sign(sin_a) * np.abs(sin_a) ** power

    # Mild asymmetry and a chicane-like bend prevent memorizing a rectangle.
    local_x += rng.uniform(12.0, 32.0) * np.sin(
        2.0 * angles + rng.uniform(0.0, 2.0 * np.pi)
    )
    local_y += rng.uniform(8.0, 22.0) * np.sin(
        3.0 * angles + rng.uniform(0.0, 2.0 * np.pi)
    )
    rotation = rng.uniform(-0.18, 0.18)
    x = center_x + np.cos(rotation) * local_x - np.sin(rotation) * local_y
    y = center_y + np.sin(rotation) * local_x + np.cos(rotation) * local_y
    x = np.clip(x, margin, width - margin)
    y = np.clip(y, margin, height - margin)
    return list(zip(np.rint(x).astype(int), np.rint(y).astype(int)))


@dataclass
class CameraDomainRandomizer:
    """Episode-level camera variation plus per-frame sensor noise."""

    enabled: bool = False
    brightness: float = 1.0
    contrast: float = 1.0
    color_gain: np.ndarray | None = None
    noise_std: float = 0.0
    blur_sigma: float = 0.0
    dropout_probability: float = 0.0

    def reset(self, rng: np.random.Generator) -> None:
        if not self.enabled:
            self.brightness = self.contrast = 1.0
            self.color_gain = np.ones(3, dtype=np.float32)
            self.noise_std = self.blur_sigma = self.dropout_probability = 0.0
            return
        self.brightness = float(rng.uniform(0.82, 1.18))
        self.contrast = float(rng.uniform(0.82, 1.20))
        self.color_gain = rng.uniform(0.90, 1.10, size=3).astype(np.float32)
        self.noise_std = float(rng.uniform(1.5, 7.0))
        self.blur_sigma = float(rng.uniform(0.0, 1.0))
        self.dropout_probability = float(rng.uniform(0.0, 0.008))

    def apply(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        if not self.enabled:
            return image
        result = (image.astype(np.float32) - 127.5) * self.contrast + 127.5
        result *= self.brightness * self.color_gain.reshape(1, 1, 3)
        result += rng.normal(0.0, self.noise_std, result.shape)
        if self.blur_sigma > 0.15:
            result = cv2.GaussianBlur(result, (3, 3), self.blur_sigma)
        mask = rng.random(result.shape[:2]) < self.dropout_probability
        if mask.any():
            result[mask] = rng.uniform(0.0, 255.0, (int(mask.sum()), 3))
        return np.clip(result, 0.0, 255.0).astype(np.uint8)
