"""Checkpoint path and loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def model_path(module_file: str, filename: str) -> Path:
    """Resolve a model next to its task module, independent of the current directory."""
    return Path(module_file).resolve().with_name(filename)


def load_state_dict(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device,
) -> None:
    """Load plain, wrapped, or DataParallel state dictionaries."""
    checkpoint: Any = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary.")

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )
    normalized = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(normalized)
