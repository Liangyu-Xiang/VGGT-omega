"""Public configuration for the SelfTR compression method."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .identity import METHOD_ID, resolve_method_name


@dataclass(frozen=True)
class SelfTRConfig:
    """Inference-time parameters for SelfTR token compression.

    The defaults match the previous U-M implementation.  They are deliberately
    collected here so a paper release can version one explicit configuration
    instead of relying on scattered command-line defaults.
    """

    method_name: str | None = None
    recompute_layers: tuple[int, ...] = (0, 10, 17)
    lambda_cost: float = 0.04
    min_keep_ratio: float = 0.05
    temporal_window: int = 1
    spatial_radius: int = 1
    merge_top_similarity_percent: float = 100.0
    attention_variant: str = "representative"

    @property
    def display_name(self) -> str:
        """Official method name recorded in an experiment artifact."""

        return resolve_method_name(self.method_name)

    def model_kwargs(self) -> dict[str, object]:
        """Return keyword arguments consumed by :class:`selftr.SelfTR`."""

        return {
            "merge_ratio": 0.0,
            "frame_fusion_mode": METHOD_ID,
            "frame_fusion_recompute_layers": self.recompute_layers,
            "frame_fusion_lambda_cost": self.lambda_cost,
            "frame_fusion_min_keep_ratio": self.min_keep_ratio,
            "frame_fusion_temporal_window": self.temporal_window,
            "frame_fusion_spatial_radius": self.spatial_radius,
            "frame_fusion_merge_top_similarity_percent": self.merge_top_similarity_percent,
            "frame_fusion_attention_variant": self.attention_variant,
        }

    def metadata(self) -> dict[str, object]:
        """Return JSON-serializable parameters for ``metrics.json`` files."""

        data = asdict(self)
        data["method_id"] = METHOD_ID
        data["method_name"] = self.display_name
        return data


__all__ = ["SelfTRConfig"]
