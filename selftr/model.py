"""Stable public model API for the official SelfTR implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_state_dict, require_selftr_checkpoint
from .config import SelfTRConfig
from .identity import METHOD_ID, canonical_frame_fusion_mode, resolve_method_name
from vggt_omega.models import VGGTOmega


class SelfTR(VGGTOmega):
    """VGGT backbone with the canonical SelfTR token-compression mode.

    SelfTR adds no trainable parameters to the released VGGT checkpoint.  Its
    state-dict keys are therefore exactly compatible with ``VGGTOmega``.
    """

    def __init__(self, *args: Any, method_name: str | None = None, **kwargs: Any) -> None:
        requested_mode = kwargs.get("frame_fusion_mode", METHOD_ID)
        resolved_mode = canonical_frame_fusion_mode(requested_mode)
        if resolved_mode == METHOD_ID:
            defaults = SelfTRConfig(method_name=method_name).model_kwargs()
            defaults.update(kwargs)
            kwargs = defaults
        kwargs["frame_fusion_mode"] = resolved_mode
        super().__init__(*args, **kwargs)
        self.method_name = resolve_method_name(method_name)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        method_name: str | None = None,
        strict: bool = True,
        **model_kwargs: Any,
    ) -> "SelfTR":
        """Load a compatible VGGT-Omega checkpoint without local-path assumptions."""

        checkpoint_path = Path(checkpoint)
        require_selftr_checkpoint(checkpoint_path)
        model = cls(method_name=method_name, **model_kwargs)
        state = load_state_dict(checkpoint_path)
        model.load_state_dict(state, strict=strict)
        return model.to(device).eval()


__all__ = ["SelfTR"]
