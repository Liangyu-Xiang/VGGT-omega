"""Checkpoint inspection with explicit VGGT-family compatibility diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


@dataclass(frozen=True)
class CheckpointInfo:
    """Architecture family inferred from a checkpoint state dictionary."""

    path: Path
    family: str
    key_count: int
    patch_size: int | None
    register_tokens: int | None

    @property
    def is_selftr_compatible(self) -> bool:
        return self.family == "vggt-omega"


def load_state_dict(checkpoint: str | Path) -> Mapping[str, torch.Tensor]:
    """Memory-map and normalize a state dictionary without creating a model."""

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        state = torch.load(path, mmap=True, **kwargs)
    except TypeError:
        state = torch.load(path, **kwargs)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint must contain a PyTorch state dictionary")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def inspect_checkpoint(checkpoint: str | Path) -> CheckpointInfo:
    """Identify the checkpoint family using its aggregator layout."""

    path = Path(checkpoint)
    state = load_state_dict(path)
    keys = set(state)
    if any(key.startswith("aggregator.inter_frame_blocks.") for key in keys):
        family = "vggt-omega"
    elif any(key.startswith("aggregator.global_blocks.") for key in keys):
        family = "official-vggt"
    else:
        family = "unknown"
    patch_weight = state.get("aggregator.patch_embed.patch_embed.proj.weight")
    if patch_weight is None:
        patch_weight = state.get("aggregator.patch_embed.proj.weight")
    patch_size = int(patch_weight.shape[-1]) if patch_weight is not None and patch_weight.ndim == 4 else None
    register_token = state.get("aggregator.register_token")
    register_tokens = int(register_token.shape[2]) if register_token is not None and register_token.ndim == 4 else None
    return CheckpointInfo(path=path, family=family, key_count=len(state), patch_size=patch_size, register_tokens=register_tokens)


def require_selftr_checkpoint(checkpoint: str | Path) -> CheckpointInfo:
    """Fail before model allocation when a checkpoint is not SelfTR-compatible."""

    info = inspect_checkpoint(checkpoint)
    if info.is_selftr_compatible:
        return info
    if info.family == "official-vggt":
        raise ValueError(
            "This is an official VGGT checkpoint (aggregator.global_blocks), while this "
            "SelfTR implementation wraps the bundled VGGT-Omega aggregator "
            "(aggregator.inter_frame_blocks). The two state dictionaries are not "
            "interchangeable. Supply a compatible VGGT-Omega checkpoint or use an "
            "explicit conversion released with the target backbone."
        )
    raise ValueError(
        "Could not identify a SelfTR-compatible VGGT-Omega checkpoint. Expected "
        "state-dict keys beginning with 'aggregator.inter_frame_blocks.'."
    )


__all__ = ["CheckpointInfo", "inspect_checkpoint", "load_state_dict", "require_selftr_checkpoint"]
