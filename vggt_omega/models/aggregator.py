# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import heapq
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from selftr.identity import canonical_frame_fusion_mode
from vggt_omega.models.adaptive_pair_scope_attention import (
    adaptive_pair_scope_attention_block,
)
from vggt_omega.models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer
from vggt_omega.models.progressive_attention import (
    ProgressiveAttentionConfig,
    ProgressiveMaskState,
    progressive_attention_block,
    progressive_config_from_dict,
    resolve_progressive_schedule,
)
from vggt_omega.models.selftr_triton import fused_selftr_edge_cost
from vggt_omega.utils.reference_frame import resolve_first_frame_token_indices


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

# Normalized features use d(x, y) = 1 - cos(x, y).  Its theoretical range is
# [0, 2], so dividing the average reconstruction distance by this constant
# puts every frame-fusion distortion curve on a common [0, 1] scale.
_FRAME_FUSION_MAX_COSINE_DISTANCE = 2.0


@dataclass(frozen=True)
class FrameFusionSegment:
    start: int
    end: int
    medoid: int
    cost: float
    mean_distance: float
    max_distance: float

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class FrameFusionPair:
    frame_a: int
    frame_b: int
    similarity: float


@dataclass(frozen=True)
class FrameFusionGroup:
    anchor: int
    members: tuple[int, ...]


@dataclass(frozen=True)
class FrameFusionBatchPlan:
    pairs: tuple[FrameFusionPair, ...]
    source_frames: torch.Tensor
    target_frames: torch.Tensor
    attention_indices: torch.Tensor
    unique_candidate_count: int
    requested_pair_count: int
    groups: tuple[FrameFusionGroup, ...] = ()
    target_keep_patch_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class TemporalRepresentativeBatchPlan:
    """Fixed temporal representative mapping for one batch element."""

    position_to_representative: torch.Tensor
    representative_source_indices: torch.Tensor
    representative_weights: torch.Tensor


@dataclass(frozen=True)
class SpatialRepresentativeBatchPlan:
    """Fixed per-frame spatial representative mapping for one batch element."""

    position_to_representative: torch.Tensor
    representative_source_indices: torch.Tensor
    representative_weights: torch.Tensor


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
        retain_only_cached_intermediates: bool = False,
        global_merging: bool = True,
        merging: int | None = 0,
        merge_ratio: float = 0.9,
        merge_random_seed: int = 33,
        fastvggt_protect_tokens: bool = True,
        first_frame_token_indices: tuple[int, ...] | list[int] | str = (0,),
        register_patch_inter_frame_mode: str = "none",
        register_patch_inter_frame_percent: float = 0.0,
        register_patch_inter_frame_seed: int = 33,
        sparse_attention: bool = False,
        sparse_ratio: float | None = None,
        sparse_cdf_threshold: float | None = None,
        sparse_pool_mode: str = "avg",
        progressive_attention: ProgressiveAttentionConfig | dict | None = None,
        inter_frame_only_layers: tuple[int, ...] = (),
        use_adaptive_kv_anchor: bool = False,
        adaptive_anchor_layers: str | int | tuple[int, ...] | list[int] = "none",
        adaptive_anchor_ratio: float = 0.25,
        adaptive_anchor_total: int | None = None,
        adaptive_anchor_min_per_frame: int = 1,
        adaptive_anchor_tau: float = 1.0,
        adaptive_anchor_uniform_mix: float = 0.0,
        adaptive_anchor_strategy: str = "lifting",
        adaptive_anchor_score_alpha_cross: float = 1.0,
        adaptive_anchor_score_beta_intra: float = 0.2,
        adaptive_anchor_score_mode: str = "intra",
        adaptive_anchor_proxy_quota_ratio: float = 0.0,
        adaptive_anchor_intra_source: str = "cached_frame_qk",
        adaptive_anchor_frame_budget_mode: str = "hybrid",
        adaptive_anchor_frame_budget_top_frac: float = 0.1,
        adaptive_anchor_frame_budget_lambda_intra: float = 0.7,
        adaptive_anchor_frame_budget_lambda_reg: float = 0.3,
        adaptive_anchor_frame_budget_reg_topm: int = 4,
        adaptive_anchor_reg_patch_topk_ratio: float = 0.1,
        adaptive_anchor_reg_patch_topk_min: int = 8,
        adaptive_anchor_reg_patch_topk_max: int = 64,
        adaptive_anchor_reg_patch_conf_power: float = 1.0,
        adaptive_anchor_reg_patch_min_conf: float = 0.05,
        adaptive_anchor_query_conditioned_eta: float = 0.1,
        adaptive_anchor_gated_anchor_ratio_per_key_frame: float = 0.1,
        adaptive_anchor_gated_min_per_key_frame: int = 4,
        adaptive_anchor_gated_max_per_key_frame: int = 64,
        adaptive_anchor_always_include_self_frame: bool = True,
        adaptive_anchor_profile: bool = False,
        adaptive_anchor_topm_frames: int | None = 4,
        adaptive_anchor_random_seed: int = 33,
        adaptive_anchor_debug: bool = False,
        adaptive_anchor_debug_dir: str | Path = "outputs/debug_register_mediated_anchor",
        frame_fusion_mode: str = "none",
        frame_fusion_k: int | None = None,
        frame_fusion_max_group_size: int = 5,
        frame_fusion_beta: float = 1.0,
        frame_fusion_start_layer: int = -1,
        frame_fusion_pair_percent: float = 25.0,
        frame_fusion_pool_size: int = 2,
        frame_fusion_group_similarity_threshold: float = 0.0,
        frame_fusion_target_keep_policy: str = "none",
        frame_fusion_target_keep_grid_size: int = 4,
        frame_fusion_target_keep_percent: float = 0.0,
        frame_fusion_target_keep_threshold: float = 0.0,
        frame_fusion_target_keep_seed: int = 33,
        frame_fusion_recompute_each_global: bool = False,
        frame_fusion_recompute_layers: tuple[int, ...] | list[int] | str | None = None,
        frame_fusion_lambda_cost: float = 0.15,
        frame_fusion_fixed_compression_ratio: float | None = None,
        frame_fusion_merge_top_similarity_percent: float = 100.0,
        frame_fusion_layer_lambdas: tuple[float, ...] | list[float] | dict[int, float] | str | None = None,
        frame_fusion_min_keep_ratio: float = 0.05,
        frame_fusion_temporal_window: int = 1,
        frame_fusion_spatial_radius: int = 1,
        frame_fusion_spatial_neighborhood: str = "N8",
        frame_fusion_time_overlap: float = 0.5,
        frame_fusion_reassignment_candidates: int = 8,
        frame_fusion_representative_update: str = "parent",
        frame_fusion_attention_variant: str = "representative",
    ) -> None:
        super().__init__()

        if not 0.0 <= merge_ratio <= 1.0:
            raise ValueError(f"merge_ratio must be between 0.0 and 1.0, got {merge_ratio}")
        if sparse_attention and self._merge_is_enabled(global_merging, merging, merge_ratio):
            raise ValueError("Sparse attention and token merging are mutually exclusive")
        if sparse_pool_mode not in {"avg", "max"}:
            raise ValueError(f"sparse_pool_mode must be 'avg' or 'max', got {sparse_pool_mode!r}")

        self.patch_embed = _build_patch_embed(patch_size=patch_size, embed_dim=embed_dim)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                    merge_ratio=merge_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                    merge_ratio=merge_ratio,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.cached_layer_indices = set(cached_layer_indices)
        # The released/legacy VGGT-Omega path retains every encoder output.
        # Keep the reduced-cache policy as an explicit memory A/B option.
        self.retain_only_cached_intermediates = bool(retain_only_cached_intermediates)
        self.global_merging = global_merging
        self.merging = merging
        self.merge_ratio = merge_ratio
        self.merge_random_seed = merge_random_seed
        self.fastvggt_protect_tokens = bool(fastvggt_protect_tokens)
        for block in (*self.frame_blocks, *self.inter_frame_blocks):
            block.attn.fastvggt_protect_tokens = self.fastvggt_protect_tokens
        self.first_frame_token_indices = first_frame_token_indices
        self.register_patch_inter_frame_mode = "none"
        self.register_patch_inter_frame_percent = 0.0
        self.register_patch_inter_frame_seed = register_patch_inter_frame_seed
        self.sparse_attention = sparse_attention
        self.sparse_ratio = sparse_ratio
        self.sparse_cdf_threshold = sparse_cdf_threshold
        self.sparse_pool_mode = sparse_pool_mode
        self.progressive_attention_config = ProgressiveAttentionConfig()
        self.progressive_layer_schedule = {}
        self._progressive_stage_states: dict[int, ProgressiveMaskState] = {}
        self.last_progressive_attention_stats: dict[int, dict[str, object]] = {}
        self.last_progressive_sample_indices: dict[int, torch.Tensor] = {}
        self.last_adaptive_pair_scope_debug: dict[
            int,
            dict[str, object],
        ] = {}
        self.inter_frame_only_layers: set[int] = set()
        self.use_adaptive_kv_anchor = False
        self.adaptive_anchor_layers: set[int] = set()
        self.adaptive_anchor_ratio = adaptive_anchor_ratio
        self.adaptive_anchor_total = adaptive_anchor_total
        self.adaptive_anchor_min_per_frame = adaptive_anchor_min_per_frame
        self.adaptive_anchor_tau = adaptive_anchor_tau
        self.adaptive_anchor_uniform_mix = adaptive_anchor_uniform_mix
        self.adaptive_anchor_strategy = adaptive_anchor_strategy
        self.adaptive_anchor_score_alpha_cross = float(adaptive_anchor_score_alpha_cross)
        self.adaptive_anchor_score_beta_intra = float(adaptive_anchor_score_beta_intra)
        self.adaptive_anchor_score_mode = adaptive_anchor_score_mode
        self.adaptive_anchor_proxy_quota_ratio = float(adaptive_anchor_proxy_quota_ratio)
        self.adaptive_anchor_intra_source = adaptive_anchor_intra_source
        self.adaptive_anchor_frame_budget_mode = adaptive_anchor_frame_budget_mode
        self.adaptive_anchor_frame_budget_top_frac = float(adaptive_anchor_frame_budget_top_frac)
        self.adaptive_anchor_frame_budget_lambda_intra = float(adaptive_anchor_frame_budget_lambda_intra)
        self.adaptive_anchor_frame_budget_lambda_reg = float(adaptive_anchor_frame_budget_lambda_reg)
        self.adaptive_anchor_frame_budget_reg_topm = int(adaptive_anchor_frame_budget_reg_topm)
        self.adaptive_anchor_reg_patch_topk_ratio = float(adaptive_anchor_reg_patch_topk_ratio)
        self.adaptive_anchor_reg_patch_topk_min = int(adaptive_anchor_reg_patch_topk_min)
        self.adaptive_anchor_reg_patch_topk_max = int(adaptive_anchor_reg_patch_topk_max)
        self.adaptive_anchor_reg_patch_conf_power = float(adaptive_anchor_reg_patch_conf_power)
        self.adaptive_anchor_reg_patch_min_conf = float(adaptive_anchor_reg_patch_min_conf)
        self.adaptive_anchor_query_conditioned_eta = float(adaptive_anchor_query_conditioned_eta)
        self.adaptive_anchor_gated_anchor_ratio_per_key_frame = float(
            adaptive_anchor_gated_anchor_ratio_per_key_frame
        )
        self.adaptive_anchor_gated_min_per_key_frame = int(adaptive_anchor_gated_min_per_key_frame)
        self.adaptive_anchor_gated_max_per_key_frame = int(adaptive_anchor_gated_max_per_key_frame)
        self.adaptive_anchor_always_include_self_frame = bool(adaptive_anchor_always_include_self_frame)
        self.adaptive_anchor_profile = bool(adaptive_anchor_profile)
        self.adaptive_anchor_topm_frames = adaptive_anchor_topm_frames
        self.adaptive_anchor_random_seed = int(adaptive_anchor_random_seed)
        self.adaptive_anchor_debug = adaptive_anchor_debug
        self.adaptive_anchor_debug_dir = Path(adaptive_anchor_debug_dir)
        self._adaptive_anchor_debug_step = 0
        self._adaptive_intra_scores: dict[int, torch.Tensor] = {}
        self.frame_fusion_mode = "none"
        self.frame_fusion_k: int | None = None
        self.frame_fusion_max_group_size = int(frame_fusion_max_group_size)
        self.frame_fusion_beta = float(frame_fusion_beta)
        self.frame_fusion_start_layer = int(frame_fusion_start_layer)
        self.frame_fusion_pair_percent = float(frame_fusion_pair_percent)
        self.frame_fusion_pool_size = int(frame_fusion_pool_size)
        self.frame_fusion_group_similarity_threshold = float(frame_fusion_group_similarity_threshold)
        self.frame_fusion_target_keep_policy = "none"
        self.frame_fusion_target_keep_grid_size = int(frame_fusion_target_keep_grid_size)
        self.frame_fusion_target_keep_percent = float(frame_fusion_target_keep_percent)
        self.frame_fusion_target_keep_threshold = float(frame_fusion_target_keep_threshold)
        self.frame_fusion_target_keep_seed = int(frame_fusion_target_keep_seed)
        self.frame_fusion_recompute_each_global = bool(frame_fusion_recompute_each_global)
        if frame_fusion_recompute_layers is None:
            frame_fusion_recompute_layers = (
                (0, 10, 17)
                if canonical_frame_fusion_mode(frame_fusion_mode) == "selftr"
                else ()
            )
        self.frame_fusion_recompute_layers = self._normalize_frame_fusion_recompute_layers(
            frame_fusion_recompute_layers
        )
        self.frame_fusion_min_keep_ratio = float(frame_fusion_min_keep_ratio)
        self.frame_fusion_temporal_window = int(frame_fusion_temporal_window)
        self.frame_fusion_spatial_radius = int(frame_fusion_spatial_radius)
        self.frame_fusion_spatial_neighborhood = str(frame_fusion_spatial_neighborhood).upper()
        self.frame_fusion_time_overlap = float(frame_fusion_time_overlap)
        self.frame_fusion_reassignment_candidates = int(frame_fusion_reassignment_candidates)
        self.frame_fusion_representative_update = str(frame_fusion_representative_update)
        frame_fusion_attention_variant = str(frame_fusion_attention_variant).replace("_", "-").lower()
        if frame_fusion_attention_variant not in {
            "representative",
            "representative-log",
            "kv-only",
            "replicated",
            "last-representative",
        }:
            raise ValueError(
                "frame_fusion_attention_variant must be representative, representative-log, "
                "kv-only, replicated, or last-representative, "
                f"got {frame_fusion_attention_variant!r}"
            )
        self.frame_fusion_attention_variant = frame_fusion_attention_variant
        self.last_frame_fusion_debug: dict[str, object] = {}
        self._frame_fusion_debug_layers: list[dict[str, object]] = []
        self._frame_fusion_plan_seconds = 0.0
        self._frame_fusion_global_attention_seconds = 0.0
        self._frame_fusion_attention_error_stats: dict[str, dict[str, float]] = {}
        self._frame_fusion_attention_error_comparisons = 0
        self._frame_fusion_edge_tensor_cache: dict[
            tuple[object, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.last_fastvggt_debug: dict[str, object] = {}
        self._fastvggt_merge_debug_layers: list[dict[str, object]] = []
        self.layer_token_swap_layer: int | None = None
        self.layer_token_swap_kind = "none"
        self.layer_token_swap_pairs: tuple[tuple[int, int], ...] = ()
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.patch_token_start = 1 + num_register_tokens
        self._register_patch_selection: dict[int, torch.Tensor] = {}
        self.set_register_patch_inter_frame(
            mode=register_patch_inter_frame_mode,
            percent=register_patch_inter_frame_percent,
            seed=register_patch_inter_frame_seed,
        )

        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"
        self.set_inter_frame_only_layers(inter_frame_only_layers)
        self.set_adaptive_kv_anchor(
            enabled=use_adaptive_kv_anchor,
            layers=adaptive_anchor_layers,
            ratio=adaptive_anchor_ratio,
            total=adaptive_anchor_total,
            min_per_frame=adaptive_anchor_min_per_frame,
            tau=adaptive_anchor_tau,
            uniform_mix=adaptive_anchor_uniform_mix,
            strategy=adaptive_anchor_strategy,
            score_alpha_cross=adaptive_anchor_score_alpha_cross,
            score_beta_intra=adaptive_anchor_score_beta_intra,
            score_mode=adaptive_anchor_score_mode,
            proxy_quota_ratio=adaptive_anchor_proxy_quota_ratio,
            intra_source=adaptive_anchor_intra_source,
            frame_budget_mode=adaptive_anchor_frame_budget_mode,
            frame_budget_top_frac=adaptive_anchor_frame_budget_top_frac,
            frame_budget_lambda_intra=adaptive_anchor_frame_budget_lambda_intra,
            frame_budget_lambda_reg=adaptive_anchor_frame_budget_lambda_reg,
            frame_budget_reg_topm=adaptive_anchor_frame_budget_reg_topm,
            reg_patch_topk_ratio=adaptive_anchor_reg_patch_topk_ratio,
            reg_patch_topk_min=adaptive_anchor_reg_patch_topk_min,
            reg_patch_topk_max=adaptive_anchor_reg_patch_topk_max,
            reg_patch_conf_power=adaptive_anchor_reg_patch_conf_power,
            reg_patch_min_conf=adaptive_anchor_reg_patch_min_conf,
            query_conditioned_eta=adaptive_anchor_query_conditioned_eta,
            gated_anchor_ratio_per_key_frame=adaptive_anchor_gated_anchor_ratio_per_key_frame,
            gated_min_per_key_frame=adaptive_anchor_gated_min_per_key_frame,
            gated_max_per_key_frame=adaptive_anchor_gated_max_per_key_frame,
            always_include_self_frame=adaptive_anchor_always_include_self_frame,
            profile=adaptive_anchor_profile,
            topm_frames=adaptive_anchor_topm_frames,
            random_seed=adaptive_anchor_random_seed,
            debug=adaptive_anchor_debug,
            debug_dir=adaptive_anchor_debug_dir,
        )
        self.set_progressive_attention(progressive_attention)
        self.set_frame_fusion(
            mode=frame_fusion_mode,
            num_groups=frame_fusion_k,
            max_group_size=frame_fusion_max_group_size,
            beta=frame_fusion_beta,
            start_layer=frame_fusion_start_layer,
            pair_percent=frame_fusion_pair_percent,
            pool_size=frame_fusion_pool_size,
            group_similarity_threshold=frame_fusion_group_similarity_threshold,
            target_keep_policy=frame_fusion_target_keep_policy,
            target_keep_grid_size=frame_fusion_target_keep_grid_size,
            target_keep_percent=frame_fusion_target_keep_percent,
            target_keep_threshold=frame_fusion_target_keep_threshold,
            target_keep_seed=frame_fusion_target_keep_seed,
            recompute_each_global=frame_fusion_recompute_each_global,
            recompute_layers=frame_fusion_recompute_layers,
            lambda_cost=frame_fusion_lambda_cost,
            fixed_compression_ratio=frame_fusion_fixed_compression_ratio,
            merge_top_similarity_percent=frame_fusion_merge_top_similarity_percent,
            layer_lambdas=frame_fusion_layer_lambdas,
            spatial_radius=frame_fusion_spatial_radius,
        )

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    @staticmethod
    def _merge_is_enabled(global_merging: bool, merging: int | None, merge_ratio: float) -> bool:
        return global_merging and merging is not None and merge_ratio > 0.0

    @staticmethod
    def _normalize_frame_fusion_recompute_layers(
        layers: tuple[int, ...] | list[int] | str | int,
    ) -> tuple[int, ...]:
        """Normalize optional layer indices used for SelfTR plan refreshes."""
        if isinstance(layers, str):
            normalized = layers.strip().lower()
            if not normalized or normalized == "none":
                return ()
            values: list[int] = []
            for part in normalized.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    start_text, end_text = part.split("-", 1)
                    start, end = int(start_text), int(end_text)
                    if end < start:
                        raise ValueError(f"Invalid frame_fusion_recompute_layers range {part!r}")
                    values.extend(range(start, end + 1))
                else:
                    values.append(int(part))
            layers = values
        elif isinstance(layers, int):
            layers = (layers,)
        result = tuple(sorted(set(int(layer) for layer in layers)))
        invalid = [layer for layer in result if layer < 0]
        if invalid:
            raise ValueError(
                "frame_fusion_recompute_layers must contain non-negative layer indices, "
                f"got {invalid}"
            )
        return result

    @staticmethod
    def _normalize_frame_fusion_layer_lambdas(
        layer_lambdas: tuple[float, ...] | list[float] | dict[int, float] | str | None,
        recompute_layers: tuple[int, ...],
    ) -> dict[int, float]:
        """Normalize optional per-recompute-layer lambda overrides."""

        if layer_lambdas is None:
            return {}
        if isinstance(layer_lambdas, dict):
            values = {int(layer): float(value) for layer, value in layer_lambdas.items()}
        elif isinstance(layer_lambdas, str):
            text = layer_lambdas.strip()
            if not text or text.lower() == "none":
                return {}
            parts = [part.strip() for part in text.split(",") if part.strip()]
            if any(":" in part for part in parts):
                values = {}
                for part in parts:
                    layer_text, value_text = part.split(":", 1)
                    values[int(layer_text.strip())] = float(value_text.strip())
            else:
                if len(parts) != len(recompute_layers):
                    raise ValueError(
                        "frame_fusion_layer_lambdas without layer prefixes must provide "
                        f"one value for each recompute layer {recompute_layers}, got {parts}"
                    )
                values = {
                    int(layer): float(value)
                    for layer, value in zip(recompute_layers, parts)
                }
        else:
            if len(layer_lambdas) != len(recompute_layers):
                raise ValueError(
                    "frame_fusion_layer_lambdas must provide one value for each "
                    f"recompute layer {recompute_layers}, got {layer_lambdas}"
                )
            values = {
                int(layer): float(value)
                for layer, value in zip(recompute_layers, layer_lambdas)
            }
        invalid = [layer for layer in values if layer < 0]
        if invalid:
            raise ValueError(f"frame_fusion_layer_lambdas has invalid layers {invalid}")
        negative = {layer: value for layer, value in values.items() if value < 0.0}
        if negative:
            raise ValueError(
                "frame_fusion_layer_lambdas must be non-negative, "
                f"got {negative}"
            )
        return values

    def _frame_fusion_lambda_for_layer(self, source_layer: int) -> float:
        return float(
            getattr(self, "frame_fusion_lambda_cost_by_layer", {}).get(
                int(source_layer),
                getattr(self, "frame_fusion_lambda_cost", 0.15),
            )
        )

    @staticmethod
    def _select_reallocation_prefix(
        removal_scores: np.ndarray,
        removable: np.ndarray,
        *,
        protected_count: int,
        min_keep: int,
        lambda_cost: float,
        cost_denominator: float | None = None,
    ) -> tuple[int, float]:
        """Select an R-mode deletion prefix with the normalized objective.

        R modes use deletion/reassignment rather than group merges.  The
        deletion path is ordered by the exact current nearest-survivor cosine
        error, and the selected prefix minimizes accumulated average cosine
        error normalized by its maximum value of 2, plus the normalized number
        of non-reference representatives. The final mapping is still
        recomputed against the selected survivor set.
        """
        scores = np.asarray(removal_scores, dtype=np.float64)
        removable = np.asarray(removable, dtype=np.int64)
        total = int(scores.size)
        compressible = max(total - int(protected_count), 1)
        cost_scale = max(
            float(compressible if cost_denominator is None else cost_denominator),
            1.0,
        )
        max_remove = max(0, int(removable.size) - max(int(min_keep) - int(protected_count), 0))
        if max_remove == 0:
            return 0, float(lambda_cost) * (compressible / cost_scale)
        order = removable[np.argsort(scores[removable], kind="stable")[:max_remove]]
        cumulative_error = 0.0
        best_remove = 0
        best_objective = float(lambda_cost) * (compressible / cost_scale)
        for remove_count, source_index in enumerate(order, start=1):
            cumulative_error += max(float(scores[source_index]), 0.0)
            distortion = cumulative_error / (
                _FRAME_FUSION_MAX_COSINE_DISTANCE * cost_scale
            )
            active_cost = (compressible - remove_count) / cost_scale
            objective = distortion + float(lambda_cost) * active_cost
            if objective < best_objective:
                best_objective = objective
                best_remove = remove_count
        return best_remove, best_objective

    def _frame_fusion_enabled(self) -> bool:
        return self.frame_fusion_mode != "none"

    def _frame_fusion_then_fastvggt_enabled(self) -> bool:
        """Whether representative fusion is followed by FastVGGT merging.

        Temporal representatives produce a full token sequence after their
        attention residual is restored, so subsequent global blocks can use
        the ordinary FastVGGT bipartite merge path. Pair/group fusion does not
        have this sequential composition and remains mutually exclusive with
        token merging.
        """

        return self.frame_fusion_mode in {
            "temporal-representative",
            "adaptive-temporal-representative",
        } and self._merge_is_enabled(
            getattr(self, "global_merging", False),
            getattr(self, "merging", None),
            getattr(self, "merge_ratio", 0.0),
        )

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)

    def set_merge_ratio(self, merge_ratio: float) -> None:
        if not 0.0 <= merge_ratio <= 1.0:
            raise ValueError(f"merge_ratio must be between 0.0 and 1.0, got {merge_ratio}")
        if self.sparse_attention and merge_ratio > 0.0:
            raise ValueError("Sparse attention and token merging are mutually exclusive")
        if (
            self._frame_fusion_enabled()
            and self.frame_fusion_mode not in {
                "temporal-representative",
                "adaptive-temporal-representative",
            }
            and self._merge_is_enabled(self.global_merging, self.merging, merge_ratio)
        ):
            raise ValueError("Frame fusion requires merge_ratio=0 or disabled token merging")
        if (
            self.use_adaptive_kv_anchor
            and self.adaptive_anchor_layers
            and self._merge_is_enabled(self.global_merging, self.merging, merge_ratio)
        ):
            raise ValueError(
                "Adaptive K/V anchors and token merging are mutually exclusive; "
                "disable global_merging, set merging=None, or use merge_ratio=0.0"
            )
        self.merge_ratio = merge_ratio
        for block in self.inter_frame_blocks:
            block.attn.merge_ratio = merge_ratio

    def set_sparse_attention(
        self,
        enabled: bool,
        sparse_ratio: float | None = None,
        sparse_cdf_threshold: float | None = None,
        sparse_pool_mode: str = "avg",
    ) -> None:
        if enabled and self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio):
            raise ValueError("Sparse attention and token merging are mutually exclusive")
        if enabled and self.use_adaptive_kv_anchor and self.adaptive_anchor_layers:
            raise ValueError("Sparse attention and adaptive K/V anchors are mutually exclusive")
        if enabled and self._frame_fusion_enabled():
            raise ValueError("Sparse attention and frame fusion are mutually exclusive")
        if sparse_pool_mode not in {"avg", "max"}:
            raise ValueError(f"sparse_pool_mode must be 'avg' or 'max', got {sparse_pool_mode!r}")
        self.sparse_attention = enabled
        self.sparse_ratio = sparse_ratio
        self.sparse_cdf_threshold = sparse_cdf_threshold
        self.sparse_pool_mode = sparse_pool_mode

    def set_progressive_attention(
        self,
        config: ProgressiveAttentionConfig | dict | None,
    ) -> None:
        resolved = progressive_config_from_dict(config)
        incompatible = []
        if resolved.enabled and self._merge_is_enabled(
            self.global_merging,
            self.merging,
            self.merge_ratio,
        ):
            incompatible.append("token merging")
        if resolved.enabled and self.sparse_attention:
            incompatible.append("per-layer sparse attention")
        if (
            resolved.enabled
            and self.use_adaptive_kv_anchor
            and self.adaptive_anchor_layers
        ):
            incompatible.append("adaptive K/V anchors")
        if resolved.enabled and self.inter_frame_only_layers:
            incompatible.append("inter-frame-only attention")
        if resolved.enabled and self._frame_fusion_enabled():
            incompatible.append("frame fusion")
        if incompatible:
            raise ValueError(
                "Progressive attention is mutually exclusive with "
                + ", ".join(incompatible)
            )
        self.progressive_attention_config = resolved
        if (
            resolved.enabled
            and resolved.algorithm == "adaptive_pair_scope"
        ):
            adaptive = resolved.adaptive_pair_scope_config
            assert adaptive is not None
            invalid = sorted(
                layer
                for layer in adaptive.enabled_layers
                if layer < 0 or layer >= self.depth
            )
            if invalid:
                raise ValueError(
                    "adaptive pair-scope enabled_layers out of range "
                    f"0..{self.depth - 1}: {invalid}"
                )
            non_global = sorted(
                layer
                for layer in adaptive.enabled_layers
                if self.inter_frame_attention_types[layer] != "global"
            )
            if non_global:
                raise ValueError(
                    "adaptive pair-scope can only run on global inter-frame "
                    f"layers; non-global layers: {non_global}"
                )
        self.progressive_layer_schedule = (
            resolve_progressive_schedule(
                depth=self.depth,
                inter_frame_attention_types=self.inter_frame_attention_types,
                config=resolved,
            )
            if resolved.enabled
            and resolved.algorithm == "legacy_token_scope"
            else {}
        )
        self._progressive_stage_states.clear()
        self.last_progressive_attention_stats.clear()
        self.last_progressive_sample_indices.clear()
        self.last_adaptive_pair_scope_debug.clear()

    def set_frame_fusion(
        self,
        *,
        mode: str = "none",
        num_groups: int | None = None,
        max_group_size: int = 5,
        beta: float = 1.0,
        start_layer: int = -1,
        pair_percent: float = 25.0,
        pool_size: int = 2,
        group_similarity_threshold: float = 0.0,
        target_keep_policy: str = "none",
        target_keep_grid_size: int = 4,
        target_keep_percent: float = 0.0,
        target_keep_threshold: float = 0.0,
        target_keep_seed: int = 33,
        recompute_each_global: bool = False,
        recompute_layers: tuple[int, ...] | list[int] | str | None = None,
        lambda_cost: float = 0.15,
        fixed_compression_ratio: float | None = None,
        merge_top_similarity_percent: float = 100.0,
        layer_lambdas: tuple[float, ...] | list[float] | dict[int, float] | str | None = None,
        spatial_radius: int = 1,
    ) -> None:
        # ``u-m`` is intentionally accepted as a deprecated command-line
        # alias, while all internal state and output metadata use ``selftr``.
        mode = canonical_frame_fusion_mode(mode)
        valid_modes = {
            "none",
            "dp-medoid",
            "pair-top-percent",
            "group-top-percent",
            "sequential-group",
            "sequential-group-average",
            "temporal-representative",
            "adaptive-temporal-representative",
            "adaptive-spatial-representative",
            "h-m",
            "h-r",
            "selftr",
            "u-r",
        }
        if mode not in valid_modes:
            raise ValueError(f"frame_fusion_mode must be one of {sorted(valid_modes)}, got {mode!r}")
        max_group_size = int(max_group_size)
        if max_group_size <= 0:
            raise ValueError(f"frame_fusion_max_group_size must be positive, got {max_group_size}")
        beta = float(beta)
        if beta < 0.0:
            raise ValueError(f"frame_fusion_beta must be non-negative, got {beta}")
        start_layer = int(start_layer)
        if start_layer < -1 or start_layer >= self.depth:
            raise ValueError(
                f"frame_fusion_start_layer must be -1 or in 0..{self.depth - 1}, got {start_layer}"
            )
        pair_percent = float(pair_percent)
        if not 0.0 < pair_percent <= 100.0:
            raise ValueError(
                f"frame_fusion_pair_percent must be in (0, 100], got {pair_percent}"
            )
        pool_size = int(pool_size)
        if pool_size <= 0:
            raise ValueError(f"frame_fusion_pool_size must be positive, got {pool_size}")
        group_similarity_threshold = float(group_similarity_threshold)
        if not -1.0 <= group_similarity_threshold <= 1.0:
            raise ValueError(
                "frame_fusion_group_similarity_threshold must be in [-1, 1], "
                f"got {group_similarity_threshold}"
            )
        target_keep_policy = target_keep_policy.replace("_", "-")
        valid_target_keep_policies = {"none", "random-grid", "least-similar", "similarity-threshold"}
        if target_keep_policy not in valid_target_keep_policies:
            raise ValueError(
                "frame_fusion_target_keep_policy must be one of "
                f"{sorted(valid_target_keep_policies)}, got {target_keep_policy!r}"
            )
        target_keep_grid_size = int(target_keep_grid_size)
        if target_keep_grid_size <= 0:
            raise ValueError(
                f"frame_fusion_target_keep_grid_size must be positive, got {target_keep_grid_size}"
            )
        target_keep_percent = float(target_keep_percent)
        if not 0.0 <= target_keep_percent <= 100.0:
            raise ValueError(
                f"frame_fusion_target_keep_percent must be in [0, 100], got {target_keep_percent}"
            )
        target_keep_threshold = float(target_keep_threshold)
        if not -1.0 <= target_keep_threshold <= 1.0:
            raise ValueError(
                "frame_fusion_target_keep_threshold must be a cosine threshold in "
                f"[-1, 1], got {target_keep_threshold}"
            )
        target_keep_seed = int(target_keep_seed)
        recompute_each_global = bool(recompute_each_global)
        if recompute_layers is not None:
            recompute_layers = self._normalize_frame_fusion_recompute_layers(recompute_layers)
        elif mode == "selftr":
            recompute_layers = (0, 10, 17)
        else:
            recompute_layers = getattr(self, "frame_fusion_recompute_layers", ())
        invalid_recompute_layers = [layer for layer in recompute_layers if layer >= self.depth]
        if invalid_recompute_layers:
            raise ValueError(
                f"frame_fusion_recompute_layers must be in 0..{self.depth - 1}, "
                f"got {invalid_recompute_layers}"
            )
        lambda_cost = float(lambda_cost)
        if lambda_cost < 0.0:
            raise ValueError(f"frame_fusion_lambda_cost must be non-negative, got {lambda_cost}")
        if fixed_compression_ratio is not None:
            fixed_compression_ratio = float(fixed_compression_ratio)
            if not 0.0 <= fixed_compression_ratio < 1.0:
                raise ValueError(
                    "frame_fusion_fixed_compression_ratio must be in [0, 1), "
                    f"got {fixed_compression_ratio}"
                )
        merge_top_similarity_percent = float(merge_top_similarity_percent)
        if not 0.0 < merge_top_similarity_percent <= 100.0:
            raise ValueError(
                "frame_fusion_merge_top_similarity_percent must be in (0, 100], "
                f"got {merge_top_similarity_percent}"
            )
        min_keep_ratio = float(getattr(self, "frame_fusion_min_keep_ratio", 0.05))
        if not 0.0 < min_keep_ratio <= 1.0:
            raise ValueError(
                f"frame_fusion_min_keep_ratio must be in (0, 1], got {min_keep_ratio}"
            )
        temporal_window = int(getattr(self, "frame_fusion_temporal_window", 1))
        if temporal_window <= 0:
            raise ValueError("frame_fusion_temporal_window must be positive")
        spatial_radius = int(spatial_radius)
        if spatial_radius <= 0:
            raise ValueError("frame_fusion_spatial_radius must be positive")
        spatial_neighborhood = str(
            getattr(self, "frame_fusion_spatial_neighborhood", "N8")
        ).upper()
        if spatial_neighborhood not in {"N4", "N8", "N8-R2"}:
            raise ValueError(
                "frame_fusion_spatial_neighborhood must be N4, N8, or N8-R2"
            )
        time_overlap = float(getattr(self, "frame_fusion_time_overlap", 0.5))
        if not 0.0 <= time_overlap <= 1.0:
            raise ValueError("frame_fusion_time_overlap must be in [0, 1]")
        reassignment_candidates = int(
            getattr(self, "frame_fusion_reassignment_candidates", 8)
        )
        if reassignment_candidates <= 0:
            raise ValueError("frame_fusion_reassignment_candidates must be positive")
        representative_update = str(
            getattr(self, "frame_fusion_representative_update", "parent")
        ).replace("_", "-")
        if representative_update not in {"parent", "exact-medoid"}:
            raise ValueError(
                "frame_fusion_representative_update must be parent or exact-medoid"
            )
        if mode == "dp-medoid":
            if num_groups is None:
                raise ValueError("frame_fusion_k is required when frame_fusion_mode='dp-medoid'")
            num_groups = int(num_groups)
            if num_groups <= 0:
                raise ValueError(f"frame_fusion_k must be positive, got {num_groups}")
            if self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio):
                raise ValueError("frame fusion requires merge_ratio=0 or disabled token merging")
            if self.sparse_attention:
                raise ValueError("frame fusion and sparse attention are mutually exclusive")
            if self.progressive_attention_config.enabled:
                raise ValueError("frame fusion and progressive attention are mutually exclusive")
            if self.inter_frame_only_layers:
                raise ValueError("frame fusion and inter-frame-only attention are mutually exclusive")
            if self.use_adaptive_kv_anchor and self.adaptive_anchor_layers:
                raise ValueError("frame fusion and adaptive K/V anchors are mutually exclusive")
            if target_keep_policy != "none":
                raise ValueError("target patch retention is only supported for pair-top-percent frame fusion")
            if recompute_each_global:
                raise ValueError("per-global recomputation is only supported for pair-top-percent frame fusion")
            if recompute_layers:
                raise ValueError("layer-specific recomputation is only supported for spatiotemporal representative fusion")
        elif mode in {
            "pair-top-percent",
            "group-top-percent",
            "sequential-group",
            "sequential-group-average",
        }:
            num_groups = None
            if self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio):
                raise ValueError("frame fusion requires merge_ratio=0 or disabled token merging")
            if self.sparse_attention:
                raise ValueError("frame fusion and sparse attention are mutually exclusive")
            if self.progressive_attention_config.enabled:
                raise ValueError("frame fusion and progressive attention are mutually exclusive")
            if self.inter_frame_only_layers:
                raise ValueError("frame fusion and inter-frame-only attention are mutually exclusive")
            if self.use_adaptive_kv_anchor and self.adaptive_anchor_layers:
                raise ValueError("frame fusion and adaptive K/V anchors are mutually exclusive")
            if target_keep_policy == "least-similar" and target_keep_percent <= 0.0:
                raise ValueError(
                    "frame_fusion_target_keep_percent must be positive for least-similar retention"
                )
            if recompute_each_global and mode != "pair-top-percent":
                raise ValueError(
                    "per-global recomputation is only supported for pair-top-percent frame fusion"
                )
            if recompute_layers:
                raise ValueError("layer-specific recomputation is only supported for spatiotemporal representative fusion")
        elif mode in {
            "temporal-representative",
            "adaptive-temporal-representative",
        }:
            num_groups = None
            if self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio):
                # Temporal representative fusion is applied once, then the
                # restored full token sequence enters FastVGGT on subsequent
                # global blocks.
                pass
            if self.sparse_attention:
                raise ValueError("frame fusion and sparse attention are mutually exclusive")
            if self.progressive_attention_config.enabled:
                raise ValueError("frame fusion and progressive attention are mutually exclusive")
            if self.inter_frame_only_layers:
                raise ValueError("frame fusion and inter-frame-only attention are mutually exclusive")
            if self.use_adaptive_kv_anchor and self.adaptive_anchor_layers:
                raise ValueError("frame fusion and adaptive K/V anchors are mutually exclusive")
            if target_keep_policy == "least-similar" and target_keep_percent <= 0.0:
                raise ValueError(
                    "frame_fusion_target_keep_percent must be positive for least-similar retention"
                )
            if recompute_each_global:
                raise ValueError(
                    "per-global recomputation is not supported for temporal-only representative fusion"
                )
            if recompute_layers:
                raise ValueError("layer-specific recomputation is only supported for spatiotemporal representative fusion")
        elif mode == "adaptive-spatial-representative":
            num_groups = None
            if self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio):
                raise ValueError(
                    "adaptive spatial representative fusion requires merge_ratio=0; "
                    "it is evaluated as a standalone spatial scheme"
                )
            if self.sparse_attention:
                raise ValueError("frame fusion and sparse attention are mutually exclusive")
            if self.progressive_attention_config.enabled:
                raise ValueError("frame fusion and progressive attention are mutually exclusive")
            if self.inter_frame_only_layers:
                raise ValueError("frame fusion and inter-frame-only attention are mutually exclusive")
            if self.use_adaptive_kv_anchor and self.adaptive_anchor_layers:
                raise ValueError("frame fusion and adaptive K/V anchors are mutually exclusive")
            if recompute_each_global:
                raise ValueError(
                    "per-global recomputation is only supported for pair-top-percent frame fusion"
                )
            if recompute_layers:
                raise ValueError("layer-specific recomputation is only supported for spatiotemporal representative fusion")
        elif mode in {"h-m", "h-r", "selftr", "u-r"}:
            num_groups = None
            if self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio):
                raise ValueError(
                    "unified spatiotemporal representative fusion requires merge_ratio=0"
                )
            if self.sparse_attention:
                raise ValueError("frame fusion and sparse attention are mutually exclusive")
            if self.progressive_attention_config.enabled:
                raise ValueError("frame fusion and progressive attention are mutually exclusive")
            if self.inter_frame_only_layers:
                raise ValueError("frame fusion and inter-frame-only attention are mutually exclusive")
            if self.use_adaptive_kv_anchor and self.adaptive_anchor_layers:
                raise ValueError("frame fusion and adaptive K/V anchors are mutually exclusive")
            if recompute_layers and start_layer != -1:
                raise ValueError("layer-specific recomputation requires frame_fusion_start_layer=-1")
        else:
            num_groups = None
            target_keep_policy = "none"
            recompute_each_global = False

        self.frame_fusion_mode = mode
        self.frame_fusion_k = num_groups
        self.frame_fusion_max_group_size = max_group_size
        self.frame_fusion_beta = beta
        self.frame_fusion_start_layer = start_layer
        self.frame_fusion_pair_percent = pair_percent
        self.frame_fusion_pool_size = pool_size
        self.frame_fusion_group_similarity_threshold = group_similarity_threshold
        self.frame_fusion_target_keep_policy = target_keep_policy
        self.frame_fusion_target_keep_grid_size = target_keep_grid_size
        self.frame_fusion_target_keep_percent = target_keep_percent
        self.frame_fusion_target_keep_threshold = target_keep_threshold
        self.frame_fusion_target_keep_seed = target_keep_seed
        self.frame_fusion_recompute_each_global = recompute_each_global
        self.frame_fusion_recompute_layers = tuple(recompute_layers)
        self.frame_fusion_lambda_cost = lambda_cost
        self.frame_fusion_fixed_compression_ratio = fixed_compression_ratio
        self.frame_fusion_merge_top_similarity_percent = merge_top_similarity_percent
        self.frame_fusion_lambda_cost_by_layer = self._normalize_frame_fusion_layer_lambdas(
            layer_lambdas,
            self.frame_fusion_recompute_layers,
        )
        self.frame_fusion_min_keep_ratio = min_keep_ratio
        self.frame_fusion_temporal_window = temporal_window
        self.frame_fusion_spatial_radius = spatial_radius
        self.frame_fusion_spatial_neighborhood = spatial_neighborhood
        self.frame_fusion_time_overlap = time_overlap
        self.frame_fusion_reassignment_candidates = reassignment_candidates
        self.frame_fusion_representative_update = representative_update
        self.last_frame_fusion_debug.clear()
        self._frame_fusion_debug_layers.clear()
        self._frame_fusion_plan_seconds = 0.0
        self._frame_fusion_global_attention_seconds = 0.0
        self._frame_fusion_attention_error_comparisons = 0
        self._frame_fusion_edge_tensor_cache.clear()
        self.last_fastvggt_debug.clear()
        self._fastvggt_merge_debug_layers.clear()

    def progressive_attention_metadata(self) -> dict[str, object]:
        config = self.progressive_attention_config
        if config.algorithm == "adaptive_pair_scope":
            adaptive = config.adaptive_pair_scope_config
            if adaptive is None:
                raise RuntimeError(
                    "adaptive pair-scope metadata lacks configuration"
                )
            return {
                "enabled": config.enabled,
                "algorithm": config.algorithm,
                "semantics": (
                    "within_layer_adaptive_pair_scope_reference_v1"
                ),
                "enabled_layers": list(adaptive.enabled_layers),
                "coarse_num_anchors": adaptive.coarse_num_anchors,
                "coarse_stride": adaptive.coarse_stride,
                "routing_score_mode": adaptive.routing_score_mode,
                "coarse_selection_mode": (
                    adaptive.coarse_selection_mode
                ),
                "coarse_keep_ratio": adaptive.coarse_keep_ratio,
                "fine_selection_mode": adaptive.fine_selection_mode,
                "fine_keep_ratio": adaptive.fine_keep_ratio,
                "refine_factor": adaptive.refine_factor,
                "query_chunk_size": adaptive.query_chunk_size,
                "attention_backend": adaptive.backend_type,
                "efficient_sparse_kernel": False,
                "cross_layer_scope_inheritance": False,
            }
        return {
            "enabled": config.enabled,
            "algorithm": config.algorithm,
            "semantics": "exact_token_pair_reference_v2",
            "stage_ranges": [list(stage) for stage in config.stage_ranges],
            "enabled_stages": list(config.enabled_stages),
            "scope_schedule": list(config.scope_schedule),
            "reset_at_stage_boundary": config.reset_at_stage_boundary,
            "final_scope_mode": config.final_scope_mode,
            "require_stage_final_full": config.require_stage_final_full,
            "mask_enabled": config.mask_enabled,
            "profile_components": config.profile_components,
            "sampling": {
                "type": config.sampling_type,
                "random_seed": config.sampling_random_seed,
                "resample_each_stage": config.sampling_resample_each_stage,
                "sample_tokens_before_qkv": True,
                "patch_tokens_only": True,
            },
            "mask": {
                "head_aggregation": "mean",
                "self_weight": config.self_weight,
                "row_weight": config.row_weight,
                "column_weight": config.column_weight,
                "local_weight": config.local_weight,
                "query_neighbor_radius": config.query_neighbor_radius,
                "key_neighbor_radius": config.key_neighbor_radius,
                "row_keep_ratio": config.row_keep_ratio,
                "column_keep_ratio": config.column_keep_ratio,
                "min_pairs_per_query": config.min_pairs_per_query,
                "dilation_query": config.dilation_query,
                "dilation_key": config.dilation_key,
                "representation": config.mask_representation,
                "query_chunk_size": config.mask_query_chunk_size,
                "max_reference_pair_elements": (
                    config.max_reference_pair_elements
                ),
            },
            "layer_schedule": {
                str(layer): {
                    "stage_index": spec.stage_index,
                    "stage_name": spec.stage_name,
                    "stage_global_position": spec.global_position,
                    "stage_global_count": spec.global_count,
                    "scope": spec.scope,
                }
                for layer, spec in self.progressive_layer_schedule.items()
            },
        }

    def set_merge_random_seed(self, seed: int) -> None:
        self.merge_random_seed = int(seed)
        for block in self.inter_frame_blocks:
            block.attn.merge_random_seed = self.merge_random_seed

    def set_inter_frame_only_layers(self, layers: tuple[int, ...] | list[int]) -> None:
        layer_set = {int(layer) for layer in layers}
        invalid = sorted(layer for layer in layer_set if layer < 0 or layer >= self.depth)
        if invalid:
            raise ValueError(f"inter-frame-only layer indices out of range 0..{self.depth - 1}: {invalid}")
        non_global = sorted(
            layer for layer in layer_set if self.inter_frame_attention_types[layer] != "global"
        )
        if non_global:
            raise ValueError(
                "inter-frame-only attention can only be applied to global inter-frame blocks; "
                f"non-global layers: {non_global}"
            )
        if getattr(self, "use_adaptive_kv_anchor", False):
            overlap = sorted(layer_set & getattr(self, "adaptive_anchor_layers", set()))
            if overlap:
                raise ValueError(
                    "inter-frame-only attention and adaptive K/V anchors are mutually exclusive; "
                    f"overlapping layers: {overlap}"
                )
        if layer_set and self._frame_fusion_enabled():
            raise ValueError("inter-frame-only attention and frame fusion are mutually exclusive")
        self.inter_frame_only_layers = layer_set

    def set_adaptive_kv_anchor(
        self,
        enabled: bool,
        layers: str | int | tuple[int, ...] | list[int] = "none",
        ratio: float = 0.25,
        total: int | None = None,
        min_per_frame: int = 1,
        tau: float = 1.0,
        uniform_mix: float = 0.0,
        strategy: str = "lifting",
        score_alpha_cross: float = 1.0,
        score_beta_intra: float = 0.2,
        score_mode: str = "intra",
        proxy_quota_ratio: float = 0.0,
        intra_source: str = "cached_frame_qk",
        frame_budget_mode: str = "hybrid",
        frame_budget_top_frac: float = 0.1,
        frame_budget_lambda_intra: float = 0.7,
        frame_budget_lambda_reg: float = 0.3,
        frame_budget_reg_topm: int = 4,
        reg_patch_topk_ratio: float = 0.1,
        reg_patch_topk_min: int = 8,
        reg_patch_topk_max: int = 64,
        reg_patch_conf_power: float = 1.0,
        reg_patch_min_conf: float = 0.05,
        query_conditioned_eta: float = 0.1,
        gated_anchor_ratio_per_key_frame: float = 0.1,
        gated_min_per_key_frame: int = 4,
        gated_max_per_key_frame: int = 64,
        always_include_self_frame: bool = True,
        profile: bool = False,
        topm_frames: int | None = 4,
        random_seed: int = 33,
        debug: bool = False,
        debug_dir: str | Path = "outputs/debug_register_mediated_anchor",
    ) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"adaptive_anchor_ratio must be in [0, 1], got {ratio}")
        if total is not None and total < 0:
            raise ValueError(f"adaptive_anchor_total must be non-negative or None, got {total}")
        if min_per_frame < 0:
            raise ValueError(f"adaptive_anchor_min_per_frame must be non-negative, got {min_per_frame}")
        if tau <= 0.0:
            raise ValueError(f"adaptive_anchor_tau must be positive, got {tau}")
        if not 0.0 <= uniform_mix <= 1.0:
            raise ValueError(f"adaptive_anchor_uniform_mix must be in [0, 1], got {uniform_mix}")
        valid_strategies = {
            "all_frame_intra",
            "lifting",
            "frame_pair_gated",
            "hybrid",
            "random_frame_intra",
            "register_gated_intra",
            "register_gated_intra_query",
            "temporal_neighbor_intra",
            "oracle_frame_intra",
            "quota_intra_proxy",
            "register_intra",
            "fixed_grid",
            "intra_only",
            "proxy",
            "proxy_intra",
            "oracle",
            "random",
        }
        strategy = strategy.replace("-", "_").lower()
        if strategy not in valid_strategies:
            raise ValueError(f"adaptive_anchor_strategy must be one of {sorted(valid_strategies)}, got {strategy!r}")
        if topm_frames is not None and topm_frames <= 0:
            topm_frames = None
        if score_mode.replace("-", "_").lower() not in {"intra", "proxy", "linear_fusion", "quota_union"}:
            raise ValueError(
                "adaptive_anchor_score_mode must be one of "
                "['intra', 'proxy', 'linear_fusion', 'quota_union']"
            )
        if not 0.0 <= proxy_quota_ratio <= 1.0:
            raise ValueError(f"adaptive_anchor_proxy_quota_ratio must be in [0, 1], got {proxy_quota_ratio}")
        if intra_source not in {"current_inter_qk", "cached_frame_qk"}:
            raise ValueError(
                "adaptive_anchor_intra_source must be one of "
                "['current_inter_qk', 'cached_frame_qk']"
            )
        if frame_budget_mode not in {"uniform", "intra_concentration", "register_importance", "hybrid"}:
            raise ValueError(
                "adaptive_anchor_frame_budget_mode must be one of "
                "['uniform', 'intra_concentration', 'register_importance', 'hybrid']"
            )
        if not 0.0 < frame_budget_top_frac <= 1.0:
            raise ValueError(
                f"adaptive_anchor_frame_budget_top_frac must be in (0, 1], got {frame_budget_top_frac}"
            )
        if frame_budget_reg_topm <= 0:
            raise ValueError(
                f"adaptive_anchor_frame_budget_reg_topm must be positive, got {frame_budget_reg_topm}"
            )
        if not 0.0 <= reg_patch_topk_ratio <= 1.0:
            raise ValueError(f"adaptive_anchor_reg_patch_topk_ratio must be in [0, 1], got {reg_patch_topk_ratio}")
        if reg_patch_topk_min <= 0:
            raise ValueError(f"adaptive_anchor_reg_patch_topk_min must be positive, got {reg_patch_topk_min}")
        if reg_patch_topk_max <= 0:
            raise ValueError(f"adaptive_anchor_reg_patch_topk_max must be positive, got {reg_patch_topk_max}")
        if reg_patch_topk_max < reg_patch_topk_min:
            raise ValueError(
                "adaptive_anchor_reg_patch_topk_max must be >= adaptive_anchor_reg_patch_topk_min"
            )
        if reg_patch_conf_power < 0.0:
            raise ValueError(
                f"adaptive_anchor_reg_patch_conf_power must be non-negative, got {reg_patch_conf_power}"
            )
        if reg_patch_min_conf < 0.0:
            raise ValueError(
                f"adaptive_anchor_reg_patch_min_conf must be non-negative, got {reg_patch_min_conf}"
            )
        if gated_anchor_ratio_per_key_frame < 0.0:
            raise ValueError(
                "adaptive_anchor_gated_anchor_ratio_per_key_frame must be non-negative, "
                f"got {gated_anchor_ratio_per_key_frame}"
            )
        if gated_min_per_key_frame < 0:
            raise ValueError(
                f"adaptive_anchor_gated_min_per_key_frame must be non-negative, got {gated_min_per_key_frame}"
            )
        if gated_max_per_key_frame <= 0:
            raise ValueError(
                f"adaptive_anchor_gated_max_per_key_frame must be positive, got {gated_max_per_key_frame}"
            )

        layer_set = self._parse_adaptive_anchor_layer_spec(layers)
        if enabled and layer_set:
            layer_set = {
                layer
                for layer in layer_set
                if self.inter_frame_attention_types[layer] == "global"
            }
        if enabled and layer_set:
            overlap = sorted(layer_set & self.inter_frame_only_layers)
            if overlap:
                raise ValueError(
                    "Adaptive K/V anchors and inter-frame-only attention are mutually exclusive; "
                    f"overlapping layers: {overlap}"
                )
            if self.sparse_attention:
                raise ValueError("Adaptive K/V anchors and sparse attention are mutually exclusive")
            if self._frame_fusion_enabled():
                raise ValueError("Adaptive K/V anchors and frame fusion are mutually exclusive")
            if self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio):
                raise ValueError(
                    "Adaptive K/V anchors and token merging are mutually exclusive; "
                    "disable global_merging, set merging=None, or use merge_ratio=0.0"
                )

        self.use_adaptive_kv_anchor = bool(enabled)
        self.adaptive_anchor_layers = layer_set
        self.adaptive_anchor_ratio = ratio
        self.adaptive_anchor_total = total
        self.adaptive_anchor_min_per_frame = int(min_per_frame)
        self.adaptive_anchor_tau = float(tau)
        self.adaptive_anchor_uniform_mix = float(uniform_mix)
        self.adaptive_anchor_strategy = strategy
        self.adaptive_anchor_score_alpha_cross = float(score_alpha_cross)
        self.adaptive_anchor_score_beta_intra = float(score_beta_intra)
        self.adaptive_anchor_score_mode = score_mode.replace("-", "_").lower()
        self.adaptive_anchor_proxy_quota_ratio = float(proxy_quota_ratio)
        self.adaptive_anchor_intra_source = intra_source
        self.adaptive_anchor_frame_budget_mode = frame_budget_mode
        self.adaptive_anchor_frame_budget_top_frac = float(frame_budget_top_frac)
        self.adaptive_anchor_frame_budget_lambda_intra = float(frame_budget_lambda_intra)
        self.adaptive_anchor_frame_budget_lambda_reg = float(frame_budget_lambda_reg)
        self.adaptive_anchor_frame_budget_reg_topm = int(frame_budget_reg_topm)
        self.adaptive_anchor_reg_patch_topk_ratio = float(reg_patch_topk_ratio)
        self.adaptive_anchor_reg_patch_topk_min = int(reg_patch_topk_min)
        self.adaptive_anchor_reg_patch_topk_max = int(reg_patch_topk_max)
        self.adaptive_anchor_reg_patch_conf_power = float(reg_patch_conf_power)
        self.adaptive_anchor_reg_patch_min_conf = float(reg_patch_min_conf)
        self.adaptive_anchor_query_conditioned_eta = float(query_conditioned_eta)
        self.adaptive_anchor_gated_anchor_ratio_per_key_frame = float(gated_anchor_ratio_per_key_frame)
        self.adaptive_anchor_gated_min_per_key_frame = int(gated_min_per_key_frame)
        self.adaptive_anchor_gated_max_per_key_frame = int(gated_max_per_key_frame)
        self.adaptive_anchor_always_include_self_frame = bool(always_include_self_frame)
        self.adaptive_anchor_profile = bool(profile)
        self.adaptive_anchor_topm_frames = None if topm_frames is None else int(topm_frames)
        self.adaptive_anchor_random_seed = int(random_seed)
        self.adaptive_anchor_debug = bool(debug)
        self.adaptive_anchor_debug_dir = Path(debug_dir)

    def _parse_adaptive_anchor_layer_spec(
        self,
        layers: str | int | tuple[int, ...] | list[int] | set[int] | None,
    ) -> set[int]:
        if layers is None:
            return set()
        if isinstance(layers, str):
            spec = layers.strip().lower()
            if spec in {"", "none"}:
                return set()
            if spec == "all":
                return {
                    idx
                    for idx, attention_type in enumerate(self.inter_frame_attention_types)
                    if attention_type == "global"
                }
            parsed: set[int] = set()
            for part in spec.split(","):
                item = part.strip()
                if not item:
                    continue
                if "-" in item:
                    bounds = [bound.strip() for bound in item.split("-", 1)]
                    if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                        raise ValueError(f"Invalid adaptive_anchor_layers range: {part!r}")
                    start, end = int(bounds[0]), int(bounds[1])
                    if start > end:
                        raise ValueError(f"Invalid adaptive_anchor_layers range {part!r}: start > end")
                    parsed.update(range(start, end + 1))
                else:
                    parsed.add(int(item))
            layer_set = parsed
        elif isinstance(layers, int):
            layer_set = {int(layers)}
        else:
            layer_set = {int(layer) for layer in layers}

        invalid = sorted(layer for layer in layer_set if layer < 0 or layer >= self.depth)
        if invalid:
            raise ValueError(f"adaptive_anchor_layers out of range 0..{self.depth - 1}: {invalid}")
        return layer_set

    def set_register_patch_inter_frame(
        self,
        mode: str = "none",
        percent: float = 0.0,
        seed: int | None = None,
    ) -> None:
        valid_modes = {"none", "random", "least-register"}
        if mode not in valid_modes:
            raise ValueError(f"register patch inter-frame mode must be one of {sorted(valid_modes)}, got {mode}")
        if not 0.0 <= percent <= 100.0:
            raise ValueError(f"register patch inter-frame percent must be in [0, 100], got {percent}")
        self.register_patch_inter_frame_mode = mode
        self.register_patch_inter_frame_percent = percent
        if seed is not None:
            self.register_patch_inter_frame_seed = seed

    def set_layer_token_swap(
        self,
        layer: int | None,
        kind: str = "none",
        pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
    ) -> None:
        """Configure an experimental frame-token swap after one aggregator layer."""

        valid_kinds = {"none", "patch", "special", "whole"}
        if kind not in valid_kinds:
            raise ValueError(f"token swap kind must be one of {sorted(valid_kinds)}, got {kind!r}")
        if layer is None or kind == "none":
            self.layer_token_swap_layer = None
            self.layer_token_swap_kind = "none"
            self.layer_token_swap_pairs = ()
            return

        layer = int(layer)
        if layer < 0 or layer >= self.depth:
            raise ValueError(f"token swap layer must be in [0, {self.depth - 1}], got {layer}")

        normalized_pairs: list[tuple[int, int]] = []
        used_frames: set[int] = set()
        for first, second in pairs:
            first = int(first)
            second = int(second)
            if first < 0 or second < 0:
                raise ValueError(f"token swap frame indices must be non-negative, got {(first, second)}")
            if first == second:
                raise ValueError(f"token swap pair cannot use the same frame twice: {(first, second)}")
            if first in used_frames or second in used_frames:
                raise ValueError("token swap pairs must be non-overlapping for a simultaneous swap")
            used_frames.add(first)
            used_frames.add(second)
            normalized_pairs.append((first, second))
        if not normalized_pairs:
            raise ValueError("token swap requires at least one frame pair")

        self.layer_token_swap_layer = layer
        self.layer_token_swap_kind = kind
        self.layer_token_swap_pairs = tuple(normalized_pairs)

    def forward(
        self,
        images: torch.Tensor,
        patch_tokens: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor | None], int]:
        batch_size, num_frames, num_channels, height, width = images.shape
        original_num_frames = num_frames
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        first_frame_token_indices = resolve_first_frame_token_indices(
            self.first_frame_token_indices,
            num_frames,
        )
        camera_token = slice_expand_and_flatten(
            self.camera_token,
            batch_size,
            num_frames,
            first_frame_token_indices=first_frame_token_indices,
        )
        register_token = slice_expand_and_flatten(
            self.register_token,
            batch_size,
            num_frames,
            first_frame_token_indices=first_frame_token_indices,
        )

        if patch_tokens is None:
            normalized_images = (images - self._resnet_mean) / self._resnet_std
            patch_tokens = self.patch_embed(
                normalized_images.view(batch_size * num_frames, num_channels, height, width)
            )
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]
        elif patch_tokens.shape[0] != batch_size * num_frames:
            raise ValueError("Cached patch_tokens must have B*S entries")

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        self._frame_fusion_patch_grid_size = patch_grid_size
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        outputs = []
        self._register_patch_selection.clear()
        self._adaptive_intra_scores.clear()
        self._progressive_stage_states.clear()
        self.last_progressive_attention_stats.clear()
        self.last_progressive_sample_indices.clear()
        self.last_adaptive_pair_scope_debug.clear()
        self.last_frame_fusion_debug.clear()
        self._frame_fusion_debug_layers.clear()
        self._frame_fusion_attention_error_stats.clear()
        self.last_fastvggt_debug.clear()
        self._fastvggt_merge_debug_layers.clear()

        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
        frame_fusion_restore_index: torch.Tensor | None = None
        frame_fusion_pair_plans: list[FrameFusionBatchPlan] | None = None
        temporal_representative_plans: list[TemporalRepresentativeBatchPlan] | None = None
        spatial_representative_plans: list[SpatialRepresentativeBatchPlan] | None = None
        spatiotemporal_representative_plans: list[TemporalRepresentativeBatchPlan] | None = None
        temporal_representative_applied = False
        frame_fusion_then_fastvggt = self._frame_fusion_then_fastvggt_enabled()
        frame_fusion_applied = False
        recompute_pair_plans_each_global = (
            self.frame_fusion_mode in {
                "pair-top-percent",
                "group-top-percent",
                "sequential-group",
                "sequential-group-average",
            }
            and self.frame_fusion_recompute_each_global
        )
        recompute_spatiotemporal_layers = (
            self.frame_fusion_mode in {"h-m", "h-r", "selftr", "u-r"}
            and bool(self.frame_fusion_recompute_layers)
        )
        recompute_spatiotemporal_each_global = (
            self.frame_fusion_mode in {"h-m", "h-r", "selftr", "u-r"}
            and self.frame_fusion_recompute_each_global
        )
        if self.frame_fusion_mode == "dp-medoid" and self.frame_fusion_start_layer == -1:
            tokens, frame_fusion_restore_index = self._apply_frame_fusion(tokens, source_layer=-1)
            num_frames = tokens.shape[1]
            frame_fusion_applied = True
        elif (
            self.frame_fusion_mode in {
                "pair-top-percent",
                "group-top-percent",
                "sequential-group",
                "sequential-group-average",
            }
            and self.frame_fusion_start_layer == -1
            and not recompute_pair_plans_each_global
        ):
            frame_fusion_pair_plans = self._build_frame_fusion_pair_plans(
                tokens,
                patch_grid_size=patch_grid_size,
                source_layer=-1,
            )
        elif (
            self.frame_fusion_mode in {
                "temporal-representative",
                "adaptive-temporal-representative",
            }
            and self.frame_fusion_start_layer == -1
        ):
            if self.frame_fusion_mode == "adaptive-temporal-representative":
                temporal_representative_plans = self._build_adaptive_temporal_representative_plans(
                    tokens,
                    source_layer=-1,
                )
            else:
                temporal_representative_plans = self._build_temporal_representative_plans(
                    tokens,
                    source_layer=-1,
                )
        elif (
            self.frame_fusion_mode == "adaptive-spatial-representative"
            and self.frame_fusion_start_layer == -1
        ):
            spatial_representative_plans = self._build_adaptive_spatial_representative_plans(
                tokens,
                source_layer=-1,
            )
        elif (
            self.frame_fusion_mode in {"h-m", "h-r", "selftr", "u-r"}
            and self.frame_fusion_start_layer == -1
            and not recompute_spatiotemporal_layers
            and not recompute_spatiotemporal_each_global
        ):
            spatiotemporal_representative_plans = (
                self._build_spatiotemporal_representative_plans(
                    tokens,
                    source_layer=-1,
                )
            )
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)

        for block_idx in range(self.depth):
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )
            current_pair_plans = frame_fusion_pair_plans
            current_temporal_plans = temporal_representative_plans
            current_spatial_plans = spatial_representative_plans
            current_spatiotemporal_plans = spatiotemporal_representative_plans
            if frame_fusion_then_fastvggt and temporal_representative_applied:
                current_temporal_plans = None
            fastvggt_enabled = not frame_fusion_then_fastvggt or temporal_representative_applied
            if recompute_pair_plans_each_global:
                current_pair_plans = None
                first_recompute_layer = max(self.frame_fusion_start_layer, 0)
                if (
                    block_idx >= first_recompute_layer
                    and self.inter_frame_attention_types[block_idx] == "global"
                ):
                    current_pair_plans = self._build_frame_fusion_pair_plans(
                        frame_tokens,
                        patch_grid_size=patch_grid_size,
                        source_layer=block_idx,
                    )
            elif (
                self.frame_fusion_mode in {
                    "pair-top-percent",
                    "group-top-percent",
                    "sequential-group",
                    "sequential-group-average",
                }
                and frame_fusion_pair_plans is None
                and self.frame_fusion_start_layer == block_idx
            ):
                frame_fusion_pair_plans = self._build_frame_fusion_pair_plans(
                    frame_tokens,
                    patch_grid_size=patch_grid_size,
                    source_layer=block_idx,
                )
                current_pair_plans = frame_fusion_pair_plans
            elif (
                self.frame_fusion_mode in {
                    "temporal-representative",
                    "adaptive-temporal-representative",
                }
                and temporal_representative_plans is None
                and self.frame_fusion_start_layer == block_idx
            ):
                if self.frame_fusion_mode == "adaptive-temporal-representative":
                    temporal_representative_plans = self._build_adaptive_temporal_representative_plans(
                        frame_tokens,
                        source_layer=block_idx,
                    )
                else:
                    temporal_representative_plans = self._build_temporal_representative_plans(
                        frame_tokens,
                        source_layer=block_idx,
                    )
                current_temporal_plans = temporal_representative_plans
            elif (
                self.frame_fusion_mode == "adaptive-spatial-representative"
                and spatial_representative_plans is None
                and self.frame_fusion_start_layer == block_idx
            ):
                spatial_representative_plans = self._build_adaptive_spatial_representative_plans(
                    frame_tokens,
                    source_layer=block_idx,
                )
                current_spatial_plans = spatial_representative_plans
            elif (
                self.frame_fusion_mode in {"h-m", "h-r", "selftr", "u-r"}
                and spatiotemporal_representative_plans is None
                and self.frame_fusion_start_layer == block_idx
            ):
                spatiotemporal_representative_plans = (
                    self._build_spatiotemporal_representative_plans(
                        frame_tokens,
                        source_layer=block_idx,
                    )
                )
                current_spatiotemporal_plans = spatiotemporal_representative_plans
            if (
                recompute_spatiotemporal_layers
                and block_idx in self.frame_fusion_recompute_layers
            ):
                spatiotemporal_representative_plans = (
                    self._build_spatiotemporal_representative_plans(
                        frame_tokens,
                        source_layer=block_idx,
                    )
                )
                current_spatiotemporal_plans = spatiotemporal_representative_plans
            elif (
                recompute_spatiotemporal_each_global
                and self.inter_frame_attention_types[block_idx] == "global"
            ):
                spatiotemporal_representative_plans = (
                    self._build_spatiotemporal_representative_plans(
                        frame_tokens,
                        source_layer=block_idx,
                    )
                )
                current_spatiotemporal_plans = spatiotemporal_representative_plans
            # Preserve the token sequence but bypass the representative KV route
            # at explicitly requested global layers (e.g. final layer 23).
            if block_idx in getattr(self, "frame_fusion_full_attention_layers", ()):
                current_spatiotemporal_plans = None
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
                patch_grid_size,
                frame_fusion_pair_plans=current_pair_plans,
                temporal_representative_plans=current_temporal_plans,
                spatial_representative_plans=current_spatial_plans,
                spatiotemporal_representative_plans=current_spatiotemporal_plans,
                fastvggt_enabled=fastvggt_enabled,
            )
            if (
                frame_fusion_then_fastvggt
                and current_temporal_plans is not None
                and self.inter_frame_attention_types[block_idx] == "global"
            ):
                temporal_representative_applied = True
            layer_token_swap_active = (
                self.layer_token_swap_layer == block_idx
                and self.layer_token_swap_kind != "none"
            )
            if layer_token_swap_active:
                tokens = self._apply_layer_token_swap(
                    tokens,
                    kind=self.layer_token_swap_kind,
                    pairs=self.layer_token_swap_pairs,
                )
            if (not self.retain_only_cached_intermediates) or block_idx in self.cached_layer_indices:
                if layer_token_swap_active:
                    frame_tokens = self._apply_layer_token_swap(
                        frame_tokens,
                        kind=self.layer_token_swap_kind,
                        pairs=self.layer_token_swap_pairs,
                    )
                cached_tokens = torch.cat([frame_tokens, tokens], dim=-1)
                if frame_fusion_applied:
                    if frame_fusion_restore_index is None:
                        raise RuntimeError("frame fusion restore index is missing")
                    cached_tokens = self._restore_frame_fused_tokens(
                        cached_tokens,
                        frame_fusion_restore_index,
                        original_num_frames,
                    )
                outputs.append(cached_tokens)
            else:
                outputs.append(None)

            if (
                self.frame_fusion_mode == "dp-medoid"
                and not frame_fusion_applied
                and self.frame_fusion_start_layer == block_idx
            ):
                tokens, frame_fusion_restore_index = self._apply_frame_fusion(tokens, source_layer=block_idx)
                num_frames = tokens.shape[1]
                frame_fusion_applied = True

        merge_layers = self._fastvggt_merge_debug_layers
        input_total = sum(int(layer["input_tokens"]) for layer in merge_layers)
        output_total = sum(int(layer["output_tokens"]) for layer in merge_layers)
        merged_total = sum(int(layer["merged_tokens"]) for layer in merge_layers)
        self.last_fastvggt_debug = {
            "enabled": bool(
                self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio)
            ),
            "protect_tokens": bool(self.fastvggt_protect_tokens),
            "requested_merge_ratio": float(self.merge_ratio),
            "merge_start_layer": int(self.merging) if self.merging is not None else None,
            "full_attention_tokens_per_layer": int(batch_size * original_num_frames * num_tokens),
            "num_merge_layers": len(merge_layers),
            "input_tokens_total": input_total,
            "output_tokens_total": output_total,
            "merged_tokens_total": merged_total,
            "retention_vs_fastvggt_input": output_total / max(input_total, 1),
            "merged_fraction_vs_fastvggt_input": merged_total / max(input_total, 1),
            "layers": merge_layers,
        }
        if self.last_frame_fusion_debug:
            self.last_frame_fusion_debug["attention_variant"] = self.frame_fusion_attention_variant
            self.last_frame_fusion_debug["planning_seconds"] = float(
                self._frame_fusion_plan_seconds
            )
            self.last_frame_fusion_debug["global_attention_seconds"] = float(
                self._frame_fusion_global_attention_seconds
            )
            if self.frame_fusion_recompute_layers:
                self.last_frame_fusion_debug["recompute_layers"] = list(
                    self.frame_fusion_recompute_layers
                )
                self.last_frame_fusion_debug["recomputed_source_layers"] = [
                    int(layer["source_layer"])
                    for layer in self._frame_fusion_debug_layers
                ]
                self.last_frame_fusion_debug["num_recomputed_layers"] = len(
                    self._frame_fusion_debug_layers
                )
                self.last_frame_fusion_debug["layers"] = self._frame_fusion_debug_layers
            if self.frame_fusion_recompute_each_global and self.frame_fusion_mode in {
                "h-m",
                "h-r",
                "selftr",
                "u-r",
            }:
                self.last_frame_fusion_debug["recompute_each_global"] = True
                self.last_frame_fusion_debug["recomputed_source_layers"] = [
                    int(layer["source_layer"])
                    for layer in self._frame_fusion_debug_layers
                ]
                self.last_frame_fusion_debug["num_recomputed_layers"] = len(
                    self._frame_fusion_debug_layers
                )
                self.last_frame_fusion_debug["layers"] = self._frame_fusion_debug_layers
            if self._frame_fusion_attention_error_stats:
                self.last_frame_fusion_debug["attention_error_by_token_type"] = (
                    self._finalize_attention_error_stats()
                )
        return outputs, self.patch_token_start

    @torch.inference_mode()
    def forward_dino(self, images: torch.Tensor, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Cache patch embeddings for DA-VGGT without re-running the vision encoder."""
        batch, frames, channels, height, width = images.shape
        if batch != 1 or channels != 3:
            raise ValueError("DA-VGGT token caching supports B=1 RGB videos")
        normalized = (images - self._resnet_mean) / self._resnet_std
        flat = normalized.reshape(batch * frames, channels, height, width)
        patches, pooled = [], []
        for start in range(0, len(flat), batch_size):
            output = self.patch_embed(flat[start:start + batch_size])
            if isinstance(output, dict):
                output = output["x_norm_patchtokens"]
            patches.append(output.cpu())
            pooled.append(output.float().mean(dim=1).cpu())
        return torch.cat(patches), torch.cat(pooled)

    def _apply_layer_token_swap(
        self,
        tokens: torch.Tensor,
        *,
        kind: str,
        pairs: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"token swap expects [batch, frames, tokens, channels], got {tuple(tokens.shape)}")
        if kind not in {"patch", "special", "whole"}:
            raise ValueError(f"unknown token swap kind: {kind!r}")
        if not pairs:
            return tokens

        _, num_frames, num_tokens, _ = tokens.shape
        first_indices = torch.tensor([first for first, _ in pairs], device=tokens.device, dtype=torch.long)
        second_indices = torch.tensor([second for _, second in pairs], device=tokens.device, dtype=torch.long)
        max_pair_index = int(torch.maximum(first_indices.max(), second_indices.max()).item())
        if max_pair_index >= num_frames:
            raise ValueError(
                "token swap pair index is out of range for current frame count: "
                f"num_frames={num_frames}, max_pair_index={max_pair_index}"
            )

        if kind == "patch":
            if self.patch_token_start >= num_tokens:
                raise ValueError("patch token swap requested but no patch tokens are present")
            token_slice = slice(self.patch_token_start, None)
        elif kind == "special":
            token_slice = slice(0, self.patch_token_start)
        else:
            token_slice = slice(None)

        swapped = tokens.clone()
        first_values = swapped[:, first_indices, token_slice].clone()
        second_values = swapped[:, second_indices, token_slice].clone()
        swapped[:, first_indices, token_slice] = second_values
        swapped[:, second_indices, token_slice] = first_values
        return swapped

    def _apply_frame_fusion(
        self,
        tokens: torch.Tensor,
        *,
        source_layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.frame_fusion_mode != "dp-medoid":
            raise RuntimeError(f"Unsupported frame fusion mode: {self.frame_fusion_mode}")
        if self.frame_fusion_k is None:
            raise RuntimeError("frame_fusion_k is required for frame fusion")

        batch_size, num_frames, num_tokens, embed_dim = tokens.shape
        if batch_size != 1:
            raise ValueError("frame fusion currently supports batch_size=1")
        if self.frame_fusion_k > num_frames:
            raise ValueError(
                f"frame_fusion_k ({self.frame_fusion_k}) must be <= num_frames ({num_frames})"
            )
        if num_frames > self.frame_fusion_k * self.frame_fusion_max_group_size:
            raise ValueError(
                "frame fusion has no feasible partition: "
                f"num_frames={num_frames}, K={self.frame_fusion_k}, "
                f"M={self.frame_fusion_max_group_size}"
            )

        partition_started = time.perf_counter()
        frame_representations = tokens.float().mean(dim=2)
        normalized = torch.nn.functional.normalize(frame_representations[0], p=2, dim=-1)
        similarity = torch.matmul(normalized, normalized.T).clamp(-1.0, 1.0)
        distance = (1.0 - similarity).clamp_min(0.0)
        segments = compute_frame_fusion_partition(
            distance.detach().cpu(),
            num_groups=self.frame_fusion_k,
            max_group_size=self.frame_fusion_max_group_size,
            beta=self.frame_fusion_beta,
        )
        partition_seconds = time.perf_counter() - partition_started

        fusion_started = time.perf_counter()
        fused_tokens: list[torch.Tensor] = []
        restore_indices = torch.empty(num_frames, device=tokens.device, dtype=torch.long)
        debug_segments: list[dict[str, object]] = []
        for segment_index, segment in enumerate(segments):
            frame_indices = torch.arange(segment.start, segment.end + 1, device=tokens.device)
            local_similarity = similarity.index_select(0, frame_indices)[:, segment.medoid]
            weights = _normalize_similarity_weights(local_similarity).to(dtype=tokens.dtype)
            local_tokens = tokens.index_select(1, frame_indices)
            fused = (local_tokens * weights.view(1, -1, 1, 1)).sum(dim=1)
            fused_tokens.append(fused)
            restore_indices[segment.start : segment.end + 1] = segment_index
            debug_segments.append(
                {
                    "segment_index": segment_index,
                    "start": segment.start,
                    "end": segment.end,
                    "length": segment.length,
                    "medoid": segment.medoid,
                    "cost": segment.cost,
                    "mean_distance": segment.mean_distance,
                    "max_distance": segment.max_distance,
                    "similarities_to_medoid": [float(value) for value in local_similarity.detach().cpu().tolist()],
                    "normalized_weights": [float(value) for value in weights.detach().float().cpu().tolist()],
                }
            )

        fused = torch.stack(fused_tokens, dim=1).contiguous()
        fusion_seconds = time.perf_counter() - fusion_started
        self.last_frame_fusion_debug = {
            "mode": self.frame_fusion_mode,
            "source_layer": source_layer,
            "fastvggt_after_frame_fusion": self._frame_fusion_then_fastvggt_enabled(),
            "fastvggt_merge_ratio": self.merge_ratio if self._frame_fusion_then_fastvggt_enabled() else 0.0,
            "num_frames": num_frames,
            "num_fused_frames": fused.shape[1],
            "tokens_per_frame": num_tokens,
            "embed_dim": embed_dim,
            "num_groups": self.frame_fusion_k,
            "max_group_size": self.frame_fusion_max_group_size,
            "beta": self.frame_fusion_beta,
            "distance": "1 - cosine_similarity",
            "fusion_weight": "nonnegative_cosine_similarity_normalized_within_segment",
            "partition_seconds": partition_seconds,
            "fusion_seconds": fusion_seconds,
            "segments": debug_segments,
        }
        return fused, restore_indices

    @staticmethod
    def _restore_frame_fused_tokens(
        tokens: torch.Tensor,
        restore_indices: torch.Tensor,
        original_num_frames: int,
    ) -> torch.Tensor:
        if restore_indices.numel() != original_num_frames:
            raise ValueError(
                f"restore index length {restore_indices.numel()} does not match original frames {original_num_frames}"
            )
        return tokens.index_select(1, restore_indices.to(device=tokens.device))

    def _build_frame_fusion_pair_plans(
        self,
        tokens: torch.Tensor,
        *,
        patch_grid_size: tuple[int, int],
        source_layer: int,
    ) -> list[FrameFusionBatchPlan]:
        started = time.perf_counter()
        batch_size, num_frames, num_tokens, embed_dim = tokens.shape
        patch_tokens = tokens[:, :, self.patch_token_start :]
        patch_count = patch_tokens.shape[2]
        frame_representations = pooled_frame_representations(
            patch_tokens,
            patch_grid_size=patch_grid_size,
            pool_size=self.frame_fusion_pool_size,
        )
        normalized = torch.nn.functional.normalize(frame_representations.float(), p=2, dim=-1)

        plans: list[FrameFusionBatchPlan] = []
        debug_batches: list[dict[str, object]] = []
        for batch_index in range(batch_size):
            partition_groups: tuple[FrameFusionGroup, ...]
            comparison_pairs: list[FrameFusionPair] = []
            comparison_groups: list[FrameFusionGroup] = []
            if self.frame_fusion_mode in {"sequential-group", "sequential-group-average"}:
                partition_groups = tuple(
                    _sequential_frame_fusion_groups(
                        normalized[batch_index],
                        similarity_threshold=self.frame_fusion_group_similarity_threshold,
                        max_group_size=self.frame_fusion_max_group_size,
                        first_frame=1,
                    )
                )
                selected_pairs = []
                unique_candidate_count = 0
                requested_pair_count = 0
            else:
                selected_pairs, unique_candidate_count, requested_pair_count = (
                    select_frame_fusion_pairs_from_normalized_representations(
                        normalized[batch_index],
                        pair_percent=self.frame_fusion_pair_percent,
                        exclude_frames=(0,),
                        disjoint=self.frame_fusion_mode != "group-top-percent",
                    )
                )
                if self.frame_fusion_mode == "group-top-percent":
                    partition_groups = tuple(_connected_frame_fusion_groups(selected_pairs))
                else:
                    partition_groups = tuple(
                        FrameFusionGroup(
                            anchor=pair.frame_a,
                            members=(pair.frame_a, pair.frame_b),
                        )
                        for pair in selected_pairs
                    )
            if self.frame_fusion_mode == "group-top-percent":
                comparison_pairs, _, _ = select_frame_fusion_pairs_from_normalized_representations(
                    normalized[batch_index],
                    pair_percent=self.frame_fusion_pair_percent,
                    exclude_frames=(0,),
                    disjoint=True,
                )
                comparison_groups = [
                    FrameFusionGroup(anchor=pair.frame_a, members=(pair.frame_a, pair.frame_b))
                    for pair in comparison_pairs
                ]
            groups = tuple(group for group in partition_groups if len(group.members) > 1)
            fusion_pairs = _anchor_target_frame_fusion_pairs(
                groups,
                normalized[batch_index],
            )
            if self.frame_fusion_mode in {"sequential-group", "sequential-group-average"}:
                selected_pairs = list(fusion_pairs)
                unique_candidate_count = len(fusion_pairs)
                requested_pair_count = len(fusion_pairs)
            source_frames = torch.tensor(
                [pair.frame_a for pair in fusion_pairs], device=tokens.device,
                dtype=torch.long,
            )
            target_frames = torch.tensor(
                [pair.frame_b for pair in fusion_pairs], device=tokens.device,
                dtype=torch.long,
            )
            if self.frame_fusion_mode == "sequential-group-average":
                target_keep_patch_indices = (
                    self._select_frame_fusion_group_shared_keep_patch_indices(
                        patch_tokens[batch_index],
                        groups,
                        threshold=self.frame_fusion_target_keep_threshold,
                    )
                )
            else:
                target_keep_patch_indices = self._select_frame_fusion_target_keep_patch_indices(
                    patch_tokens[batch_index],
                    fusion_pairs,
                    patch_grid_size=patch_grid_size,
                    source_layer=source_layer,
                    batch_index=batch_index,
                )
            target_keep_counts = _frame_fusion_target_keep_patch_counts(
                target_keep_patch_indices,
                num_pairs=len(fusion_pairs),
                patch_count=patch_count,
                device=tokens.device,
            )
            target_keep_total = int(target_keep_counts.sum().item())
            target_keep_mean = (
                float(target_keep_counts.float().mean().item())
                if target_keep_counts.numel() > 0
                else 0.0
            )
            target_keep_min = (
                int(target_keep_counts.min().item())
                if target_keep_counts.numel() > 0
                else 0
            )
            target_keep_max = (
                int(target_keep_counts.max().item())
                if target_keep_counts.numel() > 0
                else 0
            )
            attention_indices = frame_fusion_attention_indices(
                num_frames=num_frames,
                tokens_per_frame=num_tokens,
                num_special_tokens=self.patch_token_start,
                source_frames=source_frames,
                target_frames=target_frames,
                target_keep_patch_indices=target_keep_patch_indices,
                device=tokens.device,
            )
            plans.append(
                FrameFusionBatchPlan(
                    pairs=tuple(selected_pairs),
                    groups=groups,
                    source_frames=source_frames,
                    target_frames=target_frames,
                    attention_indices=attention_indices,
                    unique_candidate_count=unique_candidate_count,
                    requested_pair_count=requested_pair_count,
                    target_keep_patch_indices=target_keep_patch_indices,
                )
            )
            debug_batches.append(
                {
                    "batch_index": batch_index,
                    "unique_candidate_pairs": unique_candidate_count,
                    "requested_pairs": requested_pair_count,
                    "selected_pairs": len(selected_pairs),
                    "effective_anchor_target_relations": len(fusion_pairs),
                    "selected_groups": len(groups),
                    "group_size_min": min((len(group.members) for group in groups), default=0),
                    "group_size_max": max((len(group.members) for group in groups), default=0),
                    "group_size_mean": (
                        sum(len(group.members) for group in groups) / len(groups)
                        if groups else 0.0
                    ),
                    "groups": [
                        {"anchor": group.anchor, "members": list(group.members)}
                        for group in groups
                    ],
                    "full_partition": [
                        {"anchor": group.anchor, "members": list(group.members)}
                        for group in partition_groups
                    ],
                    "full_partition_groups": len(partition_groups),
                    "singleton_partition_groups": sum(
                        len(group.members) == 1 for group in partition_groups
                    ),
                    "original_pair_partition": (
                        _frame_fusion_partition_summary(comparison_pairs, comparison_groups)
                        if self.frame_fusion_mode == "group-top-percent"
                        else None
                    ),
                    "group_partition": _frame_fusion_partition_summary(selected_pairs, list(groups)),
                    "attention_tokens": int(attention_indices.numel()),
                    "target_keep_patch_tokens_per_pair": target_keep_mean,
                    "target_keep_patch_tokens_min": target_keep_min,
                    "target_keep_patch_tokens_max": target_keep_max,
                    "target_keep_patch_tokens_total": target_keep_total,
                    "target_keep_patch_tokens_per_relation": [
                        int(value) for value in target_keep_counts.detach().cpu().tolist()
                    ],
                    "anchor_target_relations": [
                        {
                            "anchor": pair.frame_a,
                            "target": pair.frame_b,
                            "frame_similarity": pair.similarity,
                        }
                        for pair in fusion_pairs
                    ],
                    "pairs": [
                        {
                            "frame_a": pair.frame_a,
                            "frame_b": pair.frame_b,
                            "similarity": pair.similarity,
                        }
                        for pair in selected_pairs
                    ],
                }
            )

        selection_seconds = time.perf_counter() - started
        total_patch_tokens = num_frames * patch_count
        retained_by_plan = []
        for plan in plans:
            target_keep_total = int(
                _frame_fusion_target_keep_patch_counts(
                    plan.target_keep_patch_indices,
                    num_pairs=int(plan.target_frames.numel()),
                    patch_count=patch_count,
                    device=tokens.device,
                )
                .sum()
                .item()
            )
            retained_by_plan.append(
                total_patch_tokens - int(plan.target_frames.numel()) * patch_count + target_keep_total
            )
        retained_patch_tokens = max(retained_by_plan, default=total_patch_tokens)
        recompute_each_global = getattr(self, "frame_fusion_recompute_each_global", False)
        debug = {
            "mode": self.frame_fusion_mode,
            "source_layer": source_layer,
            "num_frames": num_frames,
            "tokens_per_frame": num_tokens,
            "patch_tokens_per_frame": patch_count,
            "embed_dim": embed_dim,
            "pair_percent": self.frame_fusion_pair_percent,
            "pool_size": self.frame_fusion_pool_size,
            "group_similarity_threshold": getattr(
                self, "frame_fusion_group_similarity_threshold", 0.0
            ),
            "target_keep_policy": self.frame_fusion_target_keep_policy,
            "target_keep_grid_size": self.frame_fusion_target_keep_grid_size,
            "target_keep_percent": self.frame_fusion_target_keep_percent,
            "target_keep_threshold": getattr(self, "frame_fusion_target_keep_threshold", 0.0),
            "target_keep_seed": self.frame_fusion_target_keep_seed,
            "recompute_each_global": recompute_each_global,
            "pooling": "avg_pool2d_kernel_stride_pool_size_over_patch_grid",
            "candidate_pairs": "nearest_neighbor_unique_undirected_frame_pairs",
            "overlap_policy": (
                "sequential_all_members_threshold_partition"
                if self.frame_fusion_mode in {
                    "sequential-group",
                    "sequential-group-average",
                }
                else "connected_components_without_frame_overlap_dedup"
                if self.frame_fusion_mode == "group-top-percent"
                else "greedy_similarity_ordered_disjoint_pairs"
            ),
            "excluded_frames": [0],
            "similarity": "cosine_similarity_of_pooled_patch_tokens_nearest_neighbor_dedup",
            "selection_seconds": selection_seconds,
            "full_patch_tokens": total_patch_tokens,
            "retained_patch_tokens": retained_patch_tokens,
            "patch_token_retention_vs_full": retained_patch_tokens / max(total_patch_tokens, 1),
            "batches": debug_batches,
        }
        if recompute_each_global:
            if not hasattr(self, "_frame_fusion_debug_layers"):
                self._frame_fusion_debug_layers = []
            self._frame_fusion_debug_layers.append(debug)
            selected_counts = []
            selected_group_counts = []
            retention_values = []
            attention_token_values = []
            for layer_debug in self._frame_fusion_debug_layers:
                layer_batches = layer_debug.get("batches") or []
                for batch in layer_batches:
                    selected_counts.append(float(batch.get("selected_pairs") or 0.0))
                    selected_group_counts.append(float(batch.get("selected_groups") or 0.0))
                    attention_token_values.append(float(batch.get("attention_tokens") or 0.0))
                retention_values.append(float(layer_debug.get("patch_token_retention_vs_full") or 0.0))
            aggregate_debug = dict(debug)
            aggregate_debug["recomputed_source_layers"] = [
                int(layer_debug["source_layer"])
                for layer_debug in self._frame_fusion_debug_layers
            ]
            aggregate_debug["num_recomputed_layers"] = len(self._frame_fusion_debug_layers)
            aggregate_debug["avg_selected_pairs"] = (
                sum(selected_counts) / len(selected_counts) if selected_counts else 0.0
            )
            aggregate_debug["avg_selected_groups"] = (
                sum(selected_group_counts) / len(selected_group_counts)
                if selected_group_counts
                else 0.0
            )
            aggregate_debug["avg_attention_tokens"] = (
                sum(attention_token_values) / len(attention_token_values)
                if attention_token_values
                else 0.0
            )
            aggregate_debug["avg_patch_token_retention_vs_full"] = (
                sum(retention_values) / len(retention_values) if retention_values else 0.0
            )
            aggregate_debug["layers"] = self._frame_fusion_debug_layers
            self.last_frame_fusion_debug = aggregate_debug
        else:
            self.last_frame_fusion_debug = debug
        return plans

    def _build_temporal_representative_plans(
        self,
        tokens: torch.Tensor,
        *,
        source_layer: int,
    ) -> list[TemporalRepresentativeBatchPlan]:
        """Build a fixed per-position temporal dictionary and its inverse map."""

        batch_size, num_frames, num_tokens, embed_dim = tokens.shape
        patch_tokens = tokens[:, :, self.patch_token_start :]
        patch_count = patch_tokens.shape[2]
        threshold = float(self.frame_fusion_target_keep_threshold)
        plans: list[TemporalRepresentativeBatchPlan] = []
        debug_batches: list[dict[str, object]] = []

        for batch_index in range(batch_size):
            # Frame 0 remains a complete VGGT reference frame.  It contributes
            # standalone attention tokens, but frame 1 initializes the temporal
            # representative dictionary used for subsequent sharing.
            reference_representatives = torch.arange(
                patch_count,
                device=tokens.device,
                dtype=torch.long,
            )
            representative_sources = [
                torch.arange(patch_count, device=tokens.device, dtype=torch.long)
            ]
            representative_weights = [
                torch.ones(patch_count, device=tokens.device, dtype=torch.float32)
            ]
            mapping_rows = [reference_representatives.clone()]

            if num_frames == 1:
                current_memory = patch_tokens[batch_index, 0].float()
                current_memory_norm = torch.nn.functional.normalize(
                    current_memory,
                    p=2,
                    dim=-1,
                    eps=1e-8,
                )
                current_representatives = reference_representatives
            else:
                current_memory = patch_tokens[batch_index, 1].float()
                current_memory_norm = torch.nn.functional.normalize(
                    current_memory,
                    p=2,
                    dim=-1,
                    eps=1e-8,
                )
                current_representatives = torch.arange(
                    patch_count,
                    2 * patch_count,
                    device=tokens.device,
                    dtype=torch.long,
                )
                representative_sources.append(
                    patch_count + torch.arange(
                        patch_count,
                        device=tokens.device,
                        dtype=torch.long,
                    )
                )
                representative_weights.append(
                    torch.ones(patch_count, device=tokens.device, dtype=torch.float32)
                )
                mapping_rows.append(current_representatives.clone())

            for frame_index in range(2, num_frames):
                current_frame = patch_tokens[batch_index, frame_index].float()
                current_frame_norm = torch.nn.functional.normalize(
                    current_frame,
                    p=2,
                    dim=-1,
                    eps=1e-8,
                )
                similarity = (current_frame_norm * current_memory_norm).sum(dim=-1)
                shared = similarity >= threshold

                next_representatives = torch.empty_like(current_representatives)
                next_representatives[shared] = current_representatives[shared]
                if bool((~shared).any().item()):
                    new_count = int((~shared).sum().item())
                    first_new = sum(int(source.numel()) for source in representative_sources)
                    new_ids = torch.arange(
                        first_new,
                        first_new + new_count,
                        device=tokens.device,
                        dtype=torch.long,
                    )
                    next_representatives[~shared] = new_ids
                    representative_sources.append(
                        frame_index * patch_count
                        + torch.nonzero(~shared, as_tuple=False).flatten()
                    )
                    representative_weights.append(
                        torch.ones(new_count, device=tokens.device, dtype=torch.float32)
                    )

                if bool(shared.any().item()):
                    shared_ids = current_representatives[shared]
                    all_weights = torch.cat(representative_weights)
                    all_weights.index_add_(
                        0,
                        shared_ids,
                        torch.ones(shared_ids.numel(), device=tokens.device, dtype=torch.float32),
                    )
                    offset = 0
                    updated_weights = []
                    for weights in representative_weights:
                        length = int(weights.numel())
                        updated_weights.append(all_weights[offset : offset + length])
                        offset += length
                    representative_weights = updated_weights

                current_memory_norm = torch.where(
                    shared.unsqueeze(-1),
                    current_memory_norm,
                    current_frame_norm,
                )
                current_memory = torch.where(
                    shared.unsqueeze(-1),
                    current_memory,
                    current_frame,
                )
                current_representatives = next_representatives
                mapping_rows.append(current_representatives.clone())

            mapping = torch.stack(mapping_rows, dim=0)
            source_indices = torch.cat(representative_sources, dim=0)
            weights = torch.cat(representative_weights, dim=0)
            plans.append(
                TemporalRepresentativeBatchPlan(
                    position_to_representative=mapping,
                    representative_source_indices=source_indices,
                    representative_weights=weights,
                )
            )
            debug_batches.append(
                {
                    "representative_count": int(source_indices.numel()),
                    "full_patch_tokens": int(num_frames * patch_count),
                    "representative_patch_tokens": int(source_indices.numel()),
                    "attention_tokens": int(num_frames * self.patch_token_start + source_indices.numel()),
                    "patch_token_retention_vs_full": float(
                        source_indices.numel() / max(num_frames * patch_count, 1)
                    ),
                    "representative_weight_min": float(weights.min().item()),
                    "representative_weight_max": float(weights.max().item()),
                    "representative_weight_mean": float(weights.mean().item()),
                    "mapping_checksum": int(mapping.long().sum().item()),
                    "mapping_shape": list(mapping.shape),
                }
            )

        self.last_frame_fusion_debug = {
            "mode": self.frame_fusion_mode,
            "source_layer": source_layer,
            "fastvggt_after_frame_fusion": self._frame_fusion_then_fastvggt_enabled(),
            "fastvggt_merge_ratio": self.merge_ratio if self._frame_fusion_then_fastvggt_enabled() else 0.0,
            "num_frames": num_frames,
            "tokens_per_frame": num_tokens,
            "patch_tokens_per_frame": patch_count,
            "embed_dim": embed_dim,
            "target_keep_threshold": threshold,
            "mapping": "position_to_temporal_representative",
            "weighting": "uniform_representative_keys",
            "mapping_preserved": True,
            "full_patch_tokens": int(num_frames * patch_count),
            "retained_patch_tokens": int(debug_batches[0]["representative_patch_tokens"])
            if debug_batches
            else 0,
            "patch_token_retention_vs_full": float(debug_batches[0]["patch_token_retention_vs_full"])
            if debug_batches
            else 1.0,
            "batches": debug_batches,
        }
        return plans

    @staticmethod
    def _spatiotemporal_neighbor_offsets(
        neighborhood: str,
        spatial_radius: int = 1,
    ) -> tuple[tuple[int, int], ...]:
        neighborhood = neighborhood.upper()
        spatial_radius = int(spatial_radius)
        if spatial_radius <= 0:
            raise ValueError("spatial_radius must be positive")
        # Undirected same-frame edges use one half of the spatial cube.  The
        # four offsets cover right, down, and the two downward diagonals;
        # reverse directions are generated implicitly by graph symmetrization.
        if neighborhood == "N4":
            offsets = [
                (delta_row, delta_col)
                for delta_row in range(-spatial_radius, spatial_radius + 1)
                for delta_col in range(-spatial_radius, spatial_radius + 1)
                if (delta_row > 0 or (delta_row == 0 and delta_col > 0))
                and (delta_row != 0 or delta_col != 0)
            ]
        else:
            offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
        if neighborhood == "N8-R2":
            offsets.extend([(0, 2), (2, 0)])
        return tuple(offsets)

    @staticmethod
    def _spatiotemporal_cube_spatial_offsets(
        spatial_radius: int = 1,
    ) -> tuple[tuple[int, int], ...]:
        """Return all spatial offsets in a square cube slice."""

        spatial_radius = int(spatial_radius)
        if spatial_radius <= 0:
            raise ValueError("spatial_radius must be positive")
        return tuple(
            (delta_row, delta_col)
            for delta_row in range(-spatial_radius, spatial_radius + 1)
            for delta_col in range(-spatial_radius, spatial_radius + 1)
        )

    @staticmethod
    def _spatiotemporal_cube_undirected_spatial_offsets(
        spatial_radius: int = 1,
    ) -> tuple[tuple[int, int], ...]:
        """Return one half of a square spatial window for undirected edges."""

        spatial_radius = int(spatial_radius)
        if spatial_radius <= 0:
            raise ValueError("spatial_radius must be positive")
        return tuple(
            (delta_row, delta_col)
            for delta_row in range(-spatial_radius, spatial_radius + 1)
            for delta_col in range(-spatial_radius, spatial_radius + 1)
            if (delta_row > 0 or (delta_row == 0 and delta_col > 0))
        )

    def _build_local_spatiotemporal_edges(
        self,
        num_frames: int,
        patch_count: int,
        *,
        include_temporal_spatial: bool,
        exclude_frame_zero: bool = False,
        use_spatiotemporal_cube: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build local candidate edges without an N-by-N distance matrix.

        Same-frame edges use a half-neighborhood, so each undirected spatial
        edge is emitted once. N4 uses the complete 3x3 spatial cube for every
        positive temporal offset, including when the caller uses the legacy
        ``include_temporal_spatial=False`` path.
        """

        source, target = self._build_local_spatiotemporal_edge_tensors(
            num_frames,
            patch_count,
            include_temporal_spatial=include_temporal_spatial,
            exclude_frame_zero=exclude_frame_zero,
            use_spatiotemporal_cube=use_spatiotemporal_cube,
            device=torch.device("cpu"),
            use_cache=False,
        )
        return source.numpy(), target.numpy()

    def _build_local_spatiotemporal_edge_tensors(
        self,
        num_frames: int,
        patch_count: int,
        *,
        include_temporal_spatial: bool,
        exclude_frame_zero: bool = False,
        use_spatiotemporal_cube: bool = False,
        device: torch.device,
        use_cache: bool = True,
        index_offset: int = 0,
        canonical_order: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build and optionally cache the immutable local topology on-device.

        The topology depends only on the sequence/grid configuration, not on
        token values or the source layer.  Each spatial offset is expanded as
        one tensor operation instead of appending one Python integer per edge.
        """

        device = torch.device(device)
        height, width = getattr(self, "_frame_fusion_patch_grid_size", (1, patch_count))
        if height * width != patch_count:
            height, width = 1, patch_count
        spatial_radius = int(getattr(self, "frame_fusion_spatial_radius", 1))
        neighborhood = str(
            getattr(self, "frame_fusion_spatial_neighborhood", "N8")
        ).upper()
        temporal_window = int(getattr(self, "frame_fusion_temporal_window", 1))
        cache_key = (
            int(num_frames),
            int(patch_count),
            int(height),
            int(width),
            neighborhood,
            spatial_radius,
            temporal_window,
            bool(include_temporal_spatial),
            bool(exclude_frame_zero),
            bool(use_spatiotemporal_cube),
            int(index_offset),
            bool(canonical_order),
            device.type,
            device.index,
        )
        cache = getattr(self, "_frame_fusion_edge_tensor_cache", None)
        if use_cache and cache is not None and cache_key in cache:
            return cache[cache_key]

        if use_spatiotemporal_cube:
            # SelfTR's cube is controlled solely by spatial_radius and
            # temporal_window. Legacy N4/N8 neighborhood labels must not
            # alter this topology.
            offsets = self._spatiotemporal_cube_undirected_spatial_offsets(
                spatial_radius
            )
            temporal_offsets = self._spatiotemporal_cube_spatial_offsets(
                spatial_radius
            )
        else:
            offsets = self._spatiotemporal_neighbor_offsets(
                neighborhood,
                spatial_radius,
            )
            temporal_offsets = (
                self._spatiotemporal_cube_spatial_offsets(spatial_radius)
                if neighborhood == "N4"
                else offsets
                if include_temporal_spatial
                else ((0, 0),)
            )
        first_frame = 1 if exclude_frame_zero else 0
        # (first frame, exclusive last frame, temporal delta, dr, dc)
        edge_specs: list[tuple[int, int, int, int, int]] = []
        for dr, dc in offsets:
            edge_specs.append((first_frame, num_frames, 0, dr, dc))
        for delta in range(1, temporal_window + 1):
            if first_frame < num_frames - delta:
                for dr, dc in temporal_offsets:
                    edge_specs.append(
                        (first_frame, num_frames - delta, delta, dr, dc)
                    )

        spec_sizes: list[int] = []
        for frame_begin, frame_end, _, dr, dc in edge_specs:
            valid_rows = max(height - abs(dr), 0)
            valid_cols = max(width - abs(dc), 0)
            spec_sizes.append(
                max(frame_end - frame_begin, 0) * valid_rows * valid_cols
            )
        edge_count = sum(spec_sizes)
        source = torch.empty(edge_count, device=device, dtype=torch.long)
        target = torch.empty(edge_count, device=device, dtype=torch.long)
        position_grid = torch.arange(
            patch_count,
            device=device,
            dtype=torch.long,
        ).view(height, width)
        write_offset = 0
        for spec, spec_size in zip(edge_specs, spec_sizes):
            if spec_size == 0:
                continue
            frame_begin, frame_end, delta, dr, dc = spec
            row_begin = max(-dr, 0)
            row_end = min(height - dr, height)
            col_begin = max(-dc, 0)
            col_end = min(width - dc, width)
            source_positions = position_grid[
                row_begin:row_end,
                col_begin:col_end,
            ].reshape(-1)
            target_positions = position_grid[
                row_begin + dr : row_end + dr,
                col_begin + dc : col_end + dc,
            ].reshape(-1)
            frames = torch.arange(
                frame_begin,
                frame_end,
                device=device,
                dtype=torch.long,
            )
            end_offset = write_offset + spec_size
            source[write_offset:end_offset] = (
                frames[:, None] * patch_count
                + source_positions[None, :]
                - int(index_offset)
            ).reshape(-1)
            target[write_offset:end_offset] = (
                (frames[:, None] + delta) * patch_count
                + target_positions[None, :]
                - int(index_offset)
            ).reshape(-1)
            write_offset = end_offset
        if write_offset != edge_count:
            raise RuntimeError(
                f"spatiotemporal edge construction wrote {write_offset} of {edge_count} edges"
            )
        if canonical_order and edge_count:
            # Match canonicalize_edges' lexicographic order once when the
            # immutable topology is created.  The cached tensor then avoids
            # repeating the much more expensive unique/sort at every plan.
            local_count = num_frames * patch_count - int(index_offset)
            keys = source.to(torch.int64) * int(local_count) + target.to(torch.int64)
            order = torch.argsort(keys, stable=True)
            source = source[order]
            target = target[order]
        if use_cache and cache is not None:
            # A model processes one spatial/temporal layout at a time.  Keep
            # only the current topology so changing resolution cannot retain
            # several hundred MiB of stale CUDA edge tensors.
            cache.clear()
            cache[cache_key] = (source, target)
        return source, target

    @staticmethod
    def _spatiotemporal_knee_index(
        active_counts: list[int],
        distortions: list[float],
        *,
        max_token_count: float | None = None,
    ) -> int:
        if len(active_counts) <= 2:
            return 0
        x = np.asarray(active_counts, dtype=np.float64)
        y = np.asarray(distortions, dtype=np.float64)
        # The x axis is always normalized by the maximum possible number of
        # compressible patch tokens, (F - 1) * P. It must not be normalized by
        # the observed curve endpoint because that endpoint depends on the
        # graph and on the stopping rule.
        x = x / max(
            float(max_token_count if max_token_count is not None else x.max()),
            1.0,
        )
        # Distortions are already normalized by the theoretical maximum
        # cosine distance. Only remove the common zero origin here; do not
        # normalize each sequence independently.
        y = y - y[0]
        start = np.asarray([x[0], y[0]])
        end = np.asarray([x[-1], y[-1]])
        direction = end - start
        norm = max(float(np.linalg.norm(direction)), 1e-12)
        points = np.stack([x, y], axis=1)
        distances = np.abs(
            direction[0] * (start[1] - points[:, 1])
            - (start[0] - points[:, 0]) * direction[1]
        ) / norm
        # Index zero is the uncompressed endpoint.  The full merge path is
        # evaluated first; the knee then selects the operating point.
        return int(np.argmax(distances[1:]) + 1)

    def _batch_mutual_nearest_group_merge(
        self,
        normalized_features: torch.Tensor,
        source_indices: np.ndarray,
        edge_source: np.ndarray | torch.Tensor,
        edge_target: np.ndarray | torch.Tensor,
        *,
        protected: np.ndarray,
        initial_weights: np.ndarray | None = None,
        max_group_size: int | None = None,
        min_keep_ratio: float | None = None,
        lambda_cost: float,
        cost_denominator: float | None = None,
        prefer_best_parent: bool = True,
        fixed_compression_ratio: float | None = None,
        merge_top_similarity_percent: float = 100.0,
        initial_edges_are_unique: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Batch SelfTR merges using mutual nearest-neighbor components.

        Each round evaluates the exact whole-group merge increment on every
        current graph edge, selects ``A-best(B)``/``B-best(A)`` pairs, and
        By default, accepts only pairs with ``delta_E < 2 * lambda_cost``.
        When ``fixed_compression_ratio`` is set, it accepts the same mutual
        pairs until the requested active-token target is reached. Mutual pairs
        are disjoint, so all accepted pairs can be merged in one vectorized
        component update. The graph is maintained as a compact edge tensor;
        no Python adjacency sets or full merge curve are materialized.
        """

        count = int(source_indices.size)
        if count == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), {}
        if initial_weights is None:
            initial_weights = np.ones(count, dtype=np.float64)
        initial_weights = np.asarray(initial_weights, dtype=np.float64)
        if initial_weights.shape != (count,):
            raise ValueError(
                "initial_weights must have one value per source group, "
                f"got shape {initial_weights.shape} for {count} groups"
            )
        if lambda_cost < 0.0:
            raise ValueError("lambda_cost must be non-negative")
        if fixed_compression_ratio is not None and not 0.0 <= float(fixed_compression_ratio) < 1.0:
            raise ValueError(
                "fixed_compression_ratio must be in [0, 1), "
                f"got {fixed_compression_ratio}"
            )
        merge_top_similarity_percent = float(merge_top_similarity_percent)
        if not 0.0 < merge_top_similarity_percent <= 100.0:
            raise ValueError(
                "merge_top_similarity_percent must be in (0, 100], "
                f"got {merge_top_similarity_percent}"
            )

        device = normalized_features.device
        features = normalized_features.float()
        feature_count = int(features.shape[1])
        group_weights = torch.as_tensor(
            initial_weights,
            device=device,
            dtype=features.dtype,
        )
        group_sums = features * group_weights[:, None]
        group_representatives = torch.arange(count, device=device, dtype=torch.long)
        group_sizes = torch.ones(count, device=device, dtype=torch.long)
        group_protected = torch.as_tensor(protected, device=device, dtype=torch.bool)
        labels = torch.arange(count, device=device, dtype=torch.long)
        group_errors = torch.zeros(count, device=device, dtype=features.dtype)

        edge_left = torch.as_tensor(edge_source, device=device, dtype=torch.long)
        edge_right = torch.as_tensor(edge_target, device=device, dtype=torch.long)

        profile_cuda = (
            device.type == "cuda"
            and os.environ.get(
                "SELFTR_PLANNER_PROFILE",
                os.environ.get("VGGT_UM_PLANNER_PROFILE", "0"),
            ) == "1"
        )
        profile_events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = {}

        def profile_start() -> torch.cuda.Event | None:
            if not profile_cuda:
                return None
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event

        def profile_end(name: str, start: torch.cuda.Event | None) -> None:
            if start is None:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            profile_events.setdefault(name, []).append((start, end))

        def canonicalize_edges(
            left: torch.Tensor,
            right: torch.Tensor,
            component_count: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            valid = left != right
            left = left[valid]
            right = right[valid]
            if left.numel() == 0:
                return left, right
            low = torch.minimum(left, right)
            high = torch.maximum(left, right)
            keys = low.to(torch.int64) * int(component_count) + high.to(torch.int64)
            keys = torch.unique(keys, sorted=True)
            return keys // int(component_count), keys % int(component_count)

        profile_event = profile_start()
        if not initial_edges_are_unique:
            edge_left, edge_right = canonicalize_edges(edge_left, edge_right, count)
        profile_end("initial_canonicalize", profile_event)
        min_keep = 0
        if min_keep_ratio is not None:
            min_keep = max(
                int(np.ceil(count * float(min_keep_ratio))),
                int(np.count_nonzero(protected)),
            )
        fixed_target_active = None
        if fixed_compression_ratio is not None:
            fixed_target_active = max(
                int(np.ceil(count * (1.0 - float(fixed_compression_ratio)))),
                min_keep,
                int(np.count_nonzero(protected)),
            )
        max_token_count = max(
            float(count if cost_denominator is None else cost_denominator),
            1.0,
        )
        merge_threshold = 2.0 * float(lambda_cost)
        accepted_merges = 0
        mutual_pairs_seen = 0
        similarity_pairs_seen = 0
        similarity_pairs_kept = 0
        similarity_pairs_filtered = 0
        parallel_rounds = 0
        trace_merges: list[dict[str, object]] | None = (
            [] if getattr(self, "_frame_fusion_trace_merges", False) else None
        )
        # Stable IDs are used only by the optional diagnostic trace. The
        # active tensors below are compacted every parallel round, so their
        # transient indices cannot by themselves describe a replayable merge
        # history.
        stable_group_ids = np.arange(count, dtype=np.int64)
        next_stable_group_id = count
        total_error = torch.zeros((), device=device, dtype=features.dtype)
        stop_reason = "graph_exhausted"
        chunk_size = 131_072
        edge_score_backend = "pytorch"

        while edge_left.numel():
            component_count = int(group_weights.numel())
            active_floor = fixed_target_active if fixed_target_active is not None else min_keep
            if component_count <= active_floor:
                stop_reason = (
                    "fixed_compression_target"
                    if fixed_target_active is not None
                    else "minimum_keep_ratio"
                )
                break

            edge_valid = (
                ~group_protected[edge_left]
                & ~group_protected[edge_right]
            )
            if max_group_size is not None:
                edge_valid &= (
                    group_sizes[edge_left] + group_sizes[edge_right]
                    <= int(max_group_size)
                )
            profile_event = profile_start()
            edge_cost = fused_selftr_edge_cost(
                group_sums,
                group_weights,
                group_representatives,
                group_errors,
                features,
                edge_left,
                edge_right,
                edge_valid,
                prefer_best_parent=prefer_best_parent,
            )
            if edge_cost is None:
                edge_cost = torch.full(
                    (edge_left.numel(),),
                    float("inf"),
                    device=device,
                    dtype=features.dtype,
                )
                for start in range(0, int(edge_left.numel()), chunk_size):
                    end = min(start + chunk_size, int(edge_left.numel()))
                    left = edge_left[start:end]
                    right = edge_right[start:end]
                    merged_sum = group_sums[left] + group_sums[right]
                    merged_weight = group_weights[left] + group_weights[right]
                    left_error = merged_weight - (
                        merged_sum * features[group_representatives[left]]
                    ).sum(dim=-1)
                    right_error = merged_weight - (
                        merged_sum * features[group_representatives[right]]
                    ).sum(dim=-1)
                    if prefer_best_parent:
                        merged_error = torch.minimum(left_error, right_error)
                    else:
                        merged_error = left_error
                    edge_cost[start:end] = (
                        merged_error
                        - group_errors[left]
                        - group_errors[right]
                    ).masked_fill(~edge_valid[start:end], float("inf"))
            else:
                edge_score_backend = "triton_fused"
            profile_end("edge_score", profile_event)

            # Find one deterministic minimum-cost neighbor for every current
            # component.  The second scatter reduction resolves equal-cost
            # ties by the smallest neighbor id.
            profile_event = profile_start()
            directed_group = torch.cat((edge_left, edge_right), dim=0)
            directed_neighbor = torch.cat((edge_right, edge_left), dim=0)
            directed_cost = torch.cat((edge_cost, edge_cost), dim=0)
            best_cost = torch.full(
                (component_count,),
                float("inf"),
                device=device,
                dtype=features.dtype,
            )
            best_cost.scatter_reduce_(
                0,
                directed_group,
                directed_cost,
                reduce="amin",
                include_self=True,
            )
            tie_neighbor = torch.where(
                torch.isfinite(directed_cost)
                & (directed_cost == best_cost[directed_group]),
                directed_neighbor,
                torch.full_like(directed_neighbor, component_count),
            )
            best_neighbor = torch.full(
                (component_count,),
                component_count,
                device=device,
                dtype=torch.long,
            )
            best_neighbor.scatter_reduce_(
                0,
                directed_group,
                tie_neighbor,
                reduce="amin",
                include_self=True,
            )
            profile_end("best_neighbor", profile_event)

            profile_event = profile_start()
            ids = torch.arange(component_count, device=device, dtype=torch.long)
            pair_left = ids[(ids < best_neighbor) & (best_neighbor < component_count)]
            pair_right = best_neighbor[pair_left]
            mutual = best_neighbor[pair_right] == pair_left
            pair_left = pair_left[mutual]
            pair_right = pair_right[mutual]
            mutual_pairs_seen += int(pair_left.numel())
            if pair_left.numel() == 0:
                # Preserve the diagnostic distinction without synchronizing
                # edge_valid on every successful round.
                stop_reason = (
                    "no_mergeable_edges"
                    if not bool(edge_valid.any().item())
                    else "no_mutual_nearest_pairs"
                )
                profile_end("mutual_and_filter", profile_event)
                break

            # Limit each parallel round to the most similar mutual pairs.
            # Similarity is measured between the current parent
            # representatives, and stable sorting makes ties deterministic.
            pair_count = int(pair_left.numel())
            similarity_pairs_seen += pair_count
            similarity_keep: torch.Tensor | None = None
            keep_count = pair_count
            if merge_top_similarity_percent < 100.0:
                pair_similarity = (
                    features[group_representatives[pair_left]]
                    * features[group_representatives[pair_right]]
                ).sum(dim=-1)
                keep_count = max(
                    1,
                    int(
                        np.ceil(
                            pair_count
                            * merge_top_similarity_percent
                            / 100.0
                        )
                    ),
                )
                similarity_order = torch.argsort(
                    pair_similarity,
                    descending=True,
                    stable=True,
                )
                similarity_keep = torch.ones_like(pair_similarity, dtype=torch.bool)
                similarity_keep[similarity_order[keep_count:]] = False
            similarity_pairs_kept += keep_count
            similarity_pairs_filtered += pair_count - keep_count

            merged_sum = group_sums[pair_left] + group_sums[pair_right]
            merged_weight = group_weights[pair_left] + group_weights[pair_right]
            left_error = merged_weight - (
                merged_sum * features[group_representatives[pair_left]]
            ).sum(dim=-1)
            right_error = merged_weight - (
                merged_sum * features[group_representatives[pair_right]]
            ).sum(dim=-1)
            choose_right = prefer_best_parent & (right_error < left_error)
            pair_delta = (
                torch.minimum(left_error, right_error)
                if prefer_best_parent
                else left_error
            ) - group_errors[pair_left] - group_errors[pair_right]
            acceptable = (
                torch.ones_like(pair_delta, dtype=torch.bool)
                if fixed_target_active is not None
                else pair_delta < merge_threshold
            )
            if similarity_keep is not None:
                acceptable &= similarity_keep
            if active_floor:
                available = max(component_count - active_floor, 0)
                # Mutual pairs are disjoint.  In the common case their total
                # count already fits above the floor, avoiding a GPU scalar
                # synchronization merely to count the acceptable subset.
                if pair_count > available:
                    accepted_ids = torch.nonzero(acceptable, as_tuple=False).flatten()
                    if int(accepted_ids.numel()) > available:
                        order = torch.argsort(pair_delta[accepted_ids], stable=True)
                        acceptable[accepted_ids[order[available:]]] = False
            pair_left = pair_left[acceptable]
            pair_right = pair_right[acceptable]
            pair_delta = pair_delta[acceptable]
            choose_right = choose_right[acceptable]
            if pair_left.numel() == 0:
                stop_reason = "minimum_delta_threshold"
                profile_end("mutual_and_filter", profile_event)
                break
            profile_end("mutual_and_filter", profile_event)

            trace_pair_data: list[tuple[int, int, int, int, int, int, float, float, float, int]] = []
            if trace_merges is not None:
                pair_left_cpu = pair_left.detach().cpu().numpy()
                pair_right_cpu = pair_right.detach().cpu().numpy()
                pair_delta_cpu = pair_delta.detach().float().cpu().numpy()
                choose_right_cpu = choose_right.detach().cpu().numpy()
                left_error_cpu = left_error[acceptable].detach().float().cpu().numpy()
                right_error_cpu = right_error[acceptable].detach().float().cpu().numpy()
                for index in range(pair_left_cpu.size):
                    left_current = int(pair_left_cpu[index])
                    right_current = int(pair_right_cpu[index])
                    new_stable = next_stable_group_id
                    next_stable_group_id += 1
                    left_rep = int(group_representatives[left_current].item())
                    right_rep = int(group_representatives[right_current].item())
                    selected_rep = right_rep if bool(choose_right_cpu[index]) else left_rep
                    trace_pair_data.append(
                        (
                            int(stable_group_ids[left_current]),
                            int(stable_group_ids[right_current]),
                            new_stable,
                            left_rep,
                            right_rep,
                            selected_rep,
                            float(group_sizes[left_current].item()),
                            float(group_sizes[right_current].item()),
                            float(pair_delta_cpu[index]),
                            int(parallel_rounds),
                        )
                    )

            profile_event = profile_start()
            old_to_new = torch.empty_like(ids)
            pair_member = torch.zeros(component_count, device=device, dtype=torch.bool)
            pair_member[pair_left] = True
            pair_member[pair_right] = True
            leader = ~pair_member
            leader[pair_left] = True
            leader_ids = torch.nonzero(leader, as_tuple=False).flatten()
            new_ids = torch.cumsum(leader.to(torch.long), dim=0) - 1
            old_to_new.copy_(new_ids)
            old_to_new[pair_right] = new_ids[pair_left]
            new_count = int(leader_ids.numel())

            new_weights = torch.zeros(new_count, device=device, dtype=features.dtype)
            new_weights.index_add_(0, old_to_new, group_weights)
            new_sums = torch.zeros(
                (new_count, feature_count),
                device=device,
                dtype=features.dtype,
            )
            new_sums.index_add_(0, old_to_new, group_sums)
            new_sizes = torch.zeros(new_count, device=device, dtype=torch.long)
            new_sizes.index_add_(0, old_to_new, group_sizes)
            new_representatives = group_representatives[leader_ids].clone()
            chosen_representatives = torch.where(
                choose_right,
                group_representatives[pair_right],
                group_representatives[pair_left],
            )
            new_representatives[new_ids[pair_left]] = chosen_representatives
            new_protected = group_protected[leader_ids]
            new_errors = new_weights - (
                new_sums * features[new_representatives]
            ).sum(dim=-1)
            profile_end("group_update", profile_event)

            if trace_merges is not None:
                leader_ids_cpu = leader_ids.detach().cpu().numpy()
                new_ids_cpu = new_ids.detach().cpu().numpy()
                new_stable_ids = stable_group_ids[leader_ids_cpu].copy()
                for pair_index, data in enumerate(trace_pair_data):
                    new_stable = data[2]
                    new_stable_ids[int(new_ids_cpu[int(pair_left_cpu[pair_index])])] = new_stable
                    left_stable, right_stable, _, left_rep, right_rep, selected_rep, left_size, right_size, delta, round_index = data
                    merged_size = left_size + right_size
                    trace_merges.append(
                        {
                            "round": round_index,
                            "left_group": left_stable,
                            "right_group": right_stable,
                            "new_group": new_stable,
                            "left_representative": int(source_indices[left_rep]),
                            "right_representative": int(source_indices[right_rep]),
                            "new_representative": int(source_indices[selected_rep]),
                            "left_size": int(left_size),
                            "right_size": int(right_size),
                            "new_size": int(merged_size),
                            "merge_delta": float(delta),
                        }
                    )
                stable_group_ids = new_stable_ids

            profile_event = profile_start()
            labels = old_to_new[labels]
            edge_left = old_to_new[edge_left]
            edge_right = old_to_new[edge_right]
            edge_left, edge_right = canonicalize_edges(
                edge_left,
                edge_right,
                new_count,
            )
            profile_end("graph_rebuild", profile_event)
            group_weights = new_weights
            group_sums = new_sums
            group_sizes = new_sizes
            group_representatives = new_representatives
            group_protected = new_protected
            group_errors = new_errors
            accepted_count = int(pair_left.numel())
            accepted_merges += accepted_count
            parallel_rounds += 1
            total_error += pair_delta.sum()

        final_mapping = labels.detach().cpu().numpy()
        final_representatives = group_representatives.detach().cpu().numpy()
        selected_sources = source_indices[final_representatives]
        final_active = int(group_weights.numel())
        if fixed_target_active is not None and final_active <= fixed_target_active:
            stop_reason = "fixed_compression_target"
        total_error_value = float(total_error.detach().cpu())
        final_distortion = total_error_value / (
            2.0 * max(float(initial_weights.sum()), 1.0)
        )
        final_ratio = final_active / max_token_count
        final_objective = final_distortion + float(lambda_cost) * final_ratio
        debug = {
            "initial_active_tokens": count,
            "minimum_active_tokens": (
                fixed_target_active if fixed_target_active is not None else min_keep
            ),
            "max_group_size": max_group_size,
            "lambda_cost": float(lambda_cost),
            "fixed_compression_ratio": (
                None
                if fixed_compression_ratio is None
                else float(fixed_compression_ratio)
            ),
            "fixed_target_active_tokens": fixed_target_active,
            "cost_denominator": float(max_token_count),
            "max_token_count": float(max_token_count),
            "token_count_normalization": "active_tokens / ((F - 1) * P)",
            "distortion_normalization": "E / (2 * M0)",
            "accepted_merges": accepted_merges,
            "selected_merges": accepted_merges,
            "knee_active_tokens": final_active,
            "knee_distortion": float(final_distortion),
            "selected_distortion": float(final_distortion),
            "selected_token_ratio": float(final_ratio),
            "selected_compression_ratio": float(1.0 - final_active / max(count, 1)),
            "selected_objective": float(final_objective),
            "selection": (
                "fixed_compression_ratio"
                if fixed_target_active is not None
                else "mutual_nearest_neighbor_delta_E_lt_2_lambda"
            ),
            "stopping_rule": (
                "fixed_compression_ratio"
                if fixed_target_active is not None
                else "delta_E < 2 * lambda_cost"
            ),
            "stop_reason": stop_reason,
            "parallel_rounds": parallel_rounds,
            "mutual_pairs_seen": mutual_pairs_seen,
            "merge_top_similarity_percent": float(merge_top_similarity_percent),
            "similarity_pairs_seen": similarity_pairs_seen,
            "similarity_pairs_kept": similarity_pairs_kept,
            "similarity_pairs_filtered": similarity_pairs_filtered,
            "edge_score_backend": edge_score_backend,
            "group_size_max": int(group_sizes.max().item()) if group_sizes.numel() else 0,
        }
        if profile_cuda and profile_events:
            last_end = list(profile_events.values())[-1][-1][1]
            last_end.synchronize()
            profile_ms = {
                name: float(sum(start.elapsed_time(end) for start, end in pairs))
                for name, pairs in profile_events.items()
            }
            profile_ms["total_profiled"] = float(sum(profile_ms.values()))
            debug["planner_cuda_profile_ms"] = profile_ms
        if trace_merges is not None:
            debug["merge_trace"] = trace_merges
        return final_mapping, selected_sources, debug

    def _greedy_spatiotemporal_group_merge(
        self,
        normalized_features: torch.Tensor,
        source_indices: np.ndarray,
        edge_source: np.ndarray,
        edge_target: np.ndarray,
        *,
        protected: np.ndarray,
        initial_weights: np.ndarray | None = None,
        max_group_size: int | None = None,
        min_keep_ratio: float | None = None,
        lambda_cost: float | None = None,
        cost_denominator: float | None = None,
        prefer_best_parent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Greedy local whole-group merging used by H-M and SelfTR.

        The queue is local in time and space, but its priority is recomputed
        after every group merge.  A group's error is represented exactly for
        cosine distance by its weighted feature sum, so the merge increment is
        ``E(A union B) - E(A) - E(B)`` rather than a fixed singleton-edge
        distance or a Ward-style proxy. Groups are never split.
        """

        count = int(source_indices.size)
        if count == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), {}
        if initial_weights is None:
            initial_weights = np.ones(count, dtype=np.float64)
        initial_weights = np.asarray(initial_weights, dtype=np.float64)
        if initial_weights.shape != (count,):
            raise ValueError(
                "initial_weights must have one value per source group, "
                f"got shape {initial_weights.shape} for {count} groups"
            )

        feature_device = normalized_features.device
        weight_tensor = torch.as_tensor(
            initial_weights,
            device=feature_device,
            dtype=normalized_features.dtype,
        )
        costs: list[np.ndarray] = []
        chunk_size = 250_000
        for start in range(0, edge_source.size, chunk_size):
            end = min(start + chunk_size, edge_source.size)
            left = torch.as_tensor(edge_source[start:end], device=feature_device)
            right = torch.as_tensor(edge_target[start:end], device=feature_device)
            merged_weight = weight_tensor[left] + weight_tensor[right]
            merged_sum = (
                normalized_features[left] * weight_tensor[left, None]
                + normalized_features[right] * weight_tensor[right, None]
            )
            left_error = merged_weight - (merged_sum * normalized_features[left]).sum(dim=-1)
            right_error = merged_weight - (merged_sum * normalized_features[right]).sum(dim=-1)
            if prefer_best_parent or getattr(self, "frame_fusion_representative_update", "parent") == "exact-medoid":
                costs.append(torch.minimum(left_error, right_error).detach().float().cpu().numpy())
            else:
                costs.append(left_error.detach().float().cpu().numpy())
        edge_cost = np.concatenate(costs) if costs else np.empty(0, dtype=np.float32)

        # The merge path is a CPU-side priority-queue operation.  Keep the
        # feature sums on CPU after the one-time edge-cost pass so each
        # dynamic group update does not launch a tiny GPU dot product and
        # synchronize the device.  This is algebraically identical to the
        # weighted cosine calculations above, but avoids hundreds of
        # thousands of GPU/CPU synchronization points for 300-frame inputs.
        feature_array = normalized_features.detach().float().cpu().numpy()
        del normalized_features

        group_size: list[int] = [1] * count
        representative: list[int] = list(range(count))
        group_protected: list[bool] = [bool(value) for value in protected]
        group_weights: list[float] = [float(value) for value in initial_weights]
        # Singleton groups borrow their source feature without allocating a
        # second copy. Merged groups retain only their active weighted sum.
        group_sums: list[np.ndarray | None] = [None] * count
        group_error: list[float] = [0.0] * count
        active: list[bool] = [True] * count
        group_alias: list[int] = list(range(count))
        group_adjacency: list[set[int]] = [set() for _ in range(count)]
        for left, right in zip(edge_source, edge_target):
            left = int(left)
            right = int(right)
            if left == right:
                continue
            group_adjacency[left].add(right)
            group_adjacency[right].add(left)

        def find_group(value: int) -> int:
            while group_alias[value] != value:
                group_alias[value] = group_alias[group_alias[value]]
                value = group_alias[value]
            return value

        def group_sum(group: int) -> np.ndarray:
            cached = group_sums[group]
            if cached is not None:
                return cached
            if group_weights[group] == 1.0:
                return feature_array[group]
            return feature_array[group] * group_weights[group]

        def merge_statistics(
            left_group: int,
            right_group: int,
        ) -> tuple[float, int, float, np.ndarray]:
            merged_weight = group_weights[left_group] + group_weights[right_group]
            merged_sum = group_sum(left_group) + group_sum(right_group)
            left_rep = representative[left_group]
            right_rep = representative[right_group]
            left_error = merged_weight - float(np.dot(merged_sum, feature_array[left_rep]))
            right_error = merged_weight - float(np.dot(merged_sum, feature_array[right_rep]))
            choose_right = (
                (prefer_best_parent or getattr(self, "frame_fusion_representative_update", "parent") == "exact-medoid")
                and right_error < left_error
            )
            if choose_right:
                return right_error - group_error[left_group] - group_error[right_group], right_rep, right_error, merged_sum
            return left_error - group_error[left_group] - group_error[right_group], left_rep, left_error, merged_sum

        heap: list[tuple[float, int, int, int]] = []
        heap_counter = 0
        for edge_index, (left, right) in enumerate(zip(edge_source, edge_target)):
            left = int(left)
            right = int(right)
            if left > right:
                left, right = right, left
            heapq.heappush(heap, (float(edge_cost[edge_index]), heap_counter, left, right))
            heap_counter += 1

        records: list[tuple[int, int, int, int]] = []
        active_counts = [count]
        distortions = [0.0]
        total_error = 0.0
        best_merges = 0
        best_objective = float("inf")
        min_keep = None
        if min_keep_ratio is not None:
            min_keep = max(
                int(np.ceil(count * float(min_keep_ratio))),
                int(np.count_nonzero(protected)),
            )
        max_token_count = max(
            float(count if cost_denominator is None else cost_denominator),
            1.0,
        )
        if lambda_cost is not None:
            best_objective = float(lambda_cost) * (count / max_token_count)

        while heap:
            _, _, queued_left, queued_right = heapq.heappop(heap)
            left_group = find_group(queued_left)
            right_group = find_group(queued_right)
            if left_group == right_group:
                continue
            if left_group > right_group:
                left_group, right_group = right_group, left_group
            if not active[left_group] or not active[right_group]:
                continue
            if right_group not in group_adjacency[left_group]:
                continue
            # Neighboring pairs are refreshed eagerly after each merge. Old
            # entries therefore become invalid as soon as either endpoint is
            # absorbed into a new group.
            if queued_left != left_group or queued_right != right_group:
                continue
            if (
                (
                    max_group_size is not None
                    and group_size[left_group] + group_size[right_group] > max_group_size
                )
                or group_protected[left_group]
                or group_protected[right_group]
            ):
                continue
            if min_keep is not None and active_counts[-1] - 1 < min_keep:
                break
            merge_delta, new_rep, new_error, merged_sum = merge_statistics(left_group, right_group)
            new_group = len(group_size)
            group_size.append(group_size[left_group] + group_size[right_group])
            representative.append(new_rep)
            group_protected.append(group_protected[left_group] or group_protected[right_group])
            group_weights.append(group_weights[left_group] + group_weights[right_group])
            group_sums.append(merged_sum)
            group_error.append(new_error)
            active.append(True)
            group_alias[left_group] = new_group
            group_alias[right_group] = new_group
            group_alias.append(new_group)
            neighbors = (group_adjacency[left_group] | group_adjacency[right_group]) - {
                left_group,
                right_group,
            }
            group_adjacency.append(set(neighbors))
            for neighbor in neighbors:
                group_adjacency[neighbor].discard(left_group)
                group_adjacency[neighbor].discard(right_group)
                group_adjacency[neighbor].add(new_group)
                if group_protected[new_group] or group_protected[neighbor]:
                    continue
                if (
                    max_group_size is not None
                    and group_size[new_group] + group_size[neighbor] > max_group_size
                ):
                    continue
                neighbor_delta, _, _, _ = merge_statistics(new_group, neighbor)
                heapq.heappush(
                    heap,
                    (
                        neighbor_delta,
                        heap_counter,
                        min(new_group, neighbor),
                        max(new_group, neighbor),
                    ),
                )
                heap_counter += 1
            group_adjacency[left_group].clear()
            group_adjacency[right_group].clear()
            group_sums[left_group] = None
            group_sums[right_group] = None
            group_size[left_group] = 0
            group_size[right_group] = 0
            group_weights[left_group] = 0.0
            group_weights[right_group] = 0.0
            group_error[left_group] = 0.0
            group_error[right_group] = 0.0
            active[left_group] = False
            active[right_group] = False
            records.append((left_group, right_group, new_group, new_rep))
            total_error += merge_delta
            active_counts.append(count - len(records))
            normalized_distortion = total_error / (
                _FRAME_FUSION_MAX_COSINE_DISTANCE
                * max(float(initial_weights.sum()), 1.0)
            )
            distortions.append(float(np.clip(normalized_distortion, 0.0, 1.0)))
            if lambda_cost is not None:
                active_token_ratio = active_counts[-1] / max_token_count
                current_objective = distortions[-1] + float(lambda_cost) * active_token_ratio
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_merges = len(records)
                # Group distortion is monotone under merging.  Therefore the
                # best possible future objective is bounded below by the
                # current distortion plus the cost at the minimum active
                # count.  This is an exact early stop for a fixed lambda and
                # avoids materializing the unused tail of the curve.
                future_lower_bound = distortions[-1] + float(lambda_cost) * (
                    (min_keep if min_keep is not None else 0)
                    / max_token_count
                )
                if best_objective <= future_lower_bound:
                    break

        if lambda_cost is not None:
            selected_merges = best_merges
            selection = "min_distortion_plus_lambda_active_ratio"
            selected_objective = best_objective
        else:
            selected_merges = self._spatiotemporal_knee_index(
                active_counts,
                distortions,
                max_token_count=max_token_count,
            )
            selection = "geometric_knee"
            selected_objective = None
        selected_merges = min(selected_merges, len(records))
        replay_parent = np.arange(count + len(records), dtype=np.int64)

        def replay_find(value: int) -> int:
            while replay_parent[value] != value:
                replay_parent[value] = replay_parent[replay_parent[value]]
                value = int(replay_parent[value])
            return value

        for left_group, right_group, new_group, _ in records[:selected_merges]:
            left_root = replay_find(left_group)
            right_root = replay_find(right_group)
            replay_parent[left_root] = new_group
            replay_parent[right_root] = new_group
            replay_parent[new_group] = new_group
        assignment = np.asarray(
            [replay_find(index) for index in range(count)],
            dtype=np.int64,
        )
        unique_groups = sorted(set(int(group) for group in assignment))
        group_to_local = {group: local for local, group in enumerate(unique_groups)}
        mapping = np.asarray([group_to_local[int(group)] for group in assignment], dtype=np.int64)
        selected_sources = np.asarray(
            [source_indices[representative[group]] for group in unique_groups],
            dtype=np.int64,
        )
        debug = {
            "initial_active_tokens": count,
            "minimum_active_tokens": min_keep,
            "max_group_size": max_group_size,
            "lambda_cost": lambda_cost,
            "cost_denominator": (
                float(count if cost_denominator is None else cost_denominator)
                if lambda_cost is not None
                else None
            ),
            "max_token_count": float(max_token_count),
            "token_count_normalization": "active_tokens / ((F - 1) * P)",
            "distortion_normalization": "average_cosine_distance / 2",
            "accepted_merges": len(records),
            "selected_merges": selected_merges,
            "knee_active_tokens": int(selected_sources.size),
            "knee_distortion": float(distortions[selected_merges]),
            "selected_distortion": float(distortions[selected_merges]),
            "selected_token_ratio": float(
                active_counts[selected_merges] / max_token_count
            ),
            "selected_objective": selected_objective,
            "selection": selection,
            "group_size_max": int(max(group_size, default=1)),
        }
        return mapping, selected_sources, debug

    def _build_hybrid_representative_plan(
        self,
        tokens: torch.Tensor,
        temporal_plan: TemporalRepresentativeBatchPlan,
        *,
        reallocate: bool,
    ) -> TemporalRepresentativeBatchPlan:
        """Apply H-M or H-R to the active representatives from the time stage."""

        patch_tokens = tokens[:, self.patch_token_start :].reshape(-1, tokens.shape[-1])
        patch_count = tokens.shape[1] - self.patch_token_start
        num_frames = tokens.shape[0]
        temporal_mapping = temporal_plan.position_to_representative.detach().cpu().numpy()
        temporal_sources = temporal_plan.representative_source_indices.detach().cpu().numpy()
        temporal_count = int(temporal_sources.size)
        if temporal_count == 0:
            return temporal_plan
        normalized = torch.nn.functional.normalize(patch_tokens.float(), p=2, dim=-1, eps=1e-8)
        source_features = normalized.index_select(
            0, torch.as_tensor(temporal_sources, device=normalized.device)
        )
        source_frames = temporal_sources // patch_count
        source_positions = temporal_sources % patch_count
        protected = source_frames == 0

        if not reallocate:
            height, width = getattr(self, "_frame_fusion_patch_grid_size", (1, patch_count))
            offsets = self._spatiotemporal_neighbor_offsets(
                getattr(self, "frame_fusion_spatial_neighborhood", "N8"),
                int(getattr(self, "frame_fusion_spatial_radius", 1)),
            )
            index_by_frame_position: dict[tuple[int, int], int] = {
                (int(frame), int(position)): index
                for index, (frame, position) in enumerate(zip(source_frames, source_positions))
            }
            edges_left: list[int] = []
            edges_right: list[int] = []
            eta = float(getattr(self, "frame_fusion_time_overlap", 0.5))
            support: list[list[int]] = [[] for _ in range(temporal_count)]
            for linear_index, representative_index in enumerate(temporal_mapping.reshape(-1)):
                support[int(representative_index)].append(linear_index // patch_count)
            for index, (frame, position) in enumerate(zip(source_frames, source_positions)):
                if int(frame) == 0:
                    continue
                row, col = divmod(int(position), int(width))
                for dr, dc in offsets:
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    other = index_by_frame_position.get((int(frame), nr * width + nc))
                    if other is None or other <= index:
                        continue
                    overlap = len(set(support[index]) & set(support[other]))
                    denom = max(min(len(support[index]), len(support[other])), 1)
                    if overlap / denom >= eta:
                        edges_left.append(index)
                        edges_right.append(other)
            edge_left = np.asarray(edges_left, dtype=np.int64)
            edge_right = np.asarray(edges_right, dtype=np.int64)
            protected_indices = np.flatnonzero(protected).astype(np.int64)
            search_indices = np.flatnonzero(~protected).astype(np.int64)
            temporal_weights = temporal_plan.representative_weights.detach().cpu().numpy()
            local_index = np.full(temporal_count, -1, dtype=np.int64)
            local_index[search_indices] = np.arange(search_indices.size, dtype=np.int64)
            search_edge_left = local_index[edge_left]
            search_edge_right = local_index[edge_right]
            assignment, selected_sources, merge_debug = self._greedy_spatiotemporal_group_merge(
                source_features.index_select(
                    0,
                    torch.as_tensor(search_indices, device=tokens.device, dtype=torch.long),
                ),
                search_indices,
                search_edge_left,
                search_edge_right,
                protected=np.zeros(search_indices.size, dtype=bool),
                initial_weights=temporal_weights[search_indices],
                max_group_size=None,
                min_keep_ratio=float(getattr(self, "frame_fusion_min_keep_ratio", 0.05)),
                lambda_cost=float(getattr(self, "frame_fusion_lambda_cost", 0.15)),
                cost_denominator=float(max((num_frames - 1) * patch_count, 1)),
                prefer_best_parent=True,
            )
            temporal_to_final = np.full(temporal_count, -1, dtype=np.int64)
            temporal_to_final[protected_indices] = np.arange(protected_indices.size, dtype=np.int64)
            temporal_to_final[search_indices] = protected_indices.size + assignment
            final_mapping = torch.as_tensor(
                temporal_to_final[temporal_mapping.reshape(-1)].reshape(num_frames, patch_count),
                device=tokens.device,
                dtype=torch.long,
            )
            final_sources = torch.as_tensor(
                np.concatenate(
                    [
                        temporal_sources[protected_indices],
                        temporal_sources[selected_sources],
                    ]
                ),
                device=tokens.device,
                dtype=torch.long,
            )
            merge_debug.update(
                {
                    "reference_frame_patch_tokens": int(protected_indices.size),
                    "search_space_patch_tokens": int(search_indices.size),
                    "search_space_excludes_frame_zero": True,
                }
            )
        else:
            candidate_lists: list[list[int]] = []
            height, width = getattr(self, "_frame_fusion_patch_grid_size", (1, patch_count))
            offsets = self._spatiotemporal_neighbor_offsets(
                getattr(self, "frame_fusion_spatial_neighborhood", "N8"),
                int(getattr(self, "frame_fusion_spatial_radius", 1)),
            )
            temporal_offsets = (
                self._spatiotemporal_cube_spatial_offsets(
                    int(getattr(self, "frame_fusion_spatial_radius", 1))
                )
                if str(getattr(self, "frame_fusion_spatial_neighborhood", "N8")).upper()
                == "N4"
                else offsets
            )
            for frame in range(num_frames):
                for position in range(patch_count):
                    if frame == 0:
                        candidate_lists.append([int(temporal_mapping[0, position])])
                        continue
                    row, col = divmod(position, width)
                    candidates: list[int] = []
                    candidates.extend(int(value) for value in temporal_mapping[1:, position])
                    for dr, dc in offsets:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            candidates.append(int(temporal_mapping[frame, nr * width + nc]))
                    for delta in range(1, int(getattr(self, "frame_fusion_temporal_window", 1)) + 1):
                        for other_frame in (frame - delta, frame + delta):
                            if 1 <= other_frame < num_frames:
                                candidates.extend(int(value) for value in temporal_mapping[other_frame, position : position + 1])
                                for dr, dc in temporal_offsets:
                                    nr, nc = row + dr, col + dc
                                    if 0 <= nr < height and 0 <= nc < width:
                                        candidates.append(int(temporal_mapping[other_frame, nr * width + nc]))
                    # Frame-0 representatives are fixed references.  They may
                    # only serve frame 0 itself, never a later source token.
                    unique = list(
                        dict.fromkeys(
                            candidate for candidate in candidates
                            if not protected[candidate]
                        )
                    )
                    current_rep = int(temporal_mapping[frame, position])
                    if not protected[current_rep] and current_rep not in unique:
                        unique.insert(0, current_rep)
                    candidate_lists.append(unique[: max(2, int(getattr(self, "frame_fusion_reassignment_candidates", 8)))])
            active = np.ones(temporal_count, dtype=bool)
            protected_indices = np.flatnonzero(protected)
            removable = np.flatnonzero(~protected)
            removal_scores = np.full(temporal_count, np.inf, dtype=np.float32)
            for rep_index in removable:
                source_position = int(temporal_sources[rep_index])
                candidates = [candidate for candidate in candidate_lists[source_position] if candidate != rep_index]
                candidates = [candidate for candidate in candidates if active[candidate]]
                if candidates:
                    rep = source_features[rep_index]
                    candidate_tensor = source_features[torch.as_tensor(candidates, device=tokens.device)]
                    removal_scores[rep_index] = float(
                        (1.0 - (candidate_tensor * rep).sum(dim=-1)).min().detach().cpu()
                    )
            target_min = max(
                int(np.ceil(temporal_count * float(getattr(self, "frame_fusion_min_keep_ratio", 0.05)))),
                len(protected_indices),
            )
            remove_count, reallocation_objective = self._select_reallocation_prefix(
                removal_scores,
                removable,
                protected_count=len(protected_indices),
                min_keep=target_min,
                lambda_cost=float(getattr(self, "frame_fusion_lambda_cost", 0.15)),
                cost_denominator=float(max((num_frames - 1) * patch_count, 1)),
            )
            remove_order = removable[np.argsort(removal_scores[removable], kind="stable")]
            active[remove_order[:remove_count]] = False
            active[protected_indices] = True
            survivors = np.flatnonzero(active)
            survivor_to_local = {int(value): index for index, value in enumerate(survivors)}
            final_mapping_np = np.empty(num_frames * patch_count, dtype=np.int64)
            source_index_tensor = torch.as_tensor(temporal_sources[active], device=tokens.device)
            survivor_features = source_features.index_select(0, torch.as_tensor(survivors, device=tokens.device))
            normalized_flat = normalized
            max_candidates = max(len(values) for values in candidate_lists)
            candidate_matrix = np.full((num_frames * patch_count, max_candidates), -1, dtype=np.int64)
            for index, values in enumerate(candidate_lists):
                candidate_matrix[index, : len(values)] = values
            active_tensor = torch.as_tensor(active, device=tokens.device)
            for start in range(0, final_mapping_np.size, 4_096):
                end = min(start + 4_096, final_mapping_np.size)
                candidate_ids = torch.as_tensor(candidate_matrix[start:end], device=tokens.device)
                valid = candidate_ids >= 0
                local_ids = torch.zeros_like(candidate_ids)
                for survivor, local in survivor_to_local.items():
                    local_ids[candidate_ids == survivor] = local
                safe_ids = local_ids.clamp_min(0)
                candidate_features = survivor_features[safe_ids]
                scores = (normalized_flat[start:end, None, :] * candidate_features).sum(dim=-1)
                scores = scores.masked_fill(
                    ~valid | ~active_tensor[candidate_ids.clamp_min(0)],
                    -1e9,
                )
                best_column = scores.argmax(dim=-1, keepdim=True)
                final_mapping_np[start:end] = local_ids.gather(
                    1, best_column
                ).squeeze(1).detach().cpu().numpy()
            final_mapping = torch.as_tensor(final_mapping_np.reshape(num_frames, patch_count), device=tokens.device, dtype=torch.long)
            final_sources = source_index_tensor
            merge_debug = {
                "initial_active_tokens": temporal_count,
                "minimum_active_tokens": target_min,
                "selected_merges": remove_count,
                "knee_active_tokens": int(survivors.size),
                "reassignment": True,
                "selected_objective": float(reallocation_objective),
            }
        weights = torch.bincount(final_mapping.reshape(-1), minlength=int(final_sources.numel())).float()
        self._hybrid_debug = getattr(self, "_hybrid_debug", [])
        self._hybrid_debug.append(merge_debug)
        return TemporalRepresentativeBatchPlan(
            position_to_representative=final_mapping,
            representative_source_indices=final_sources,
            representative_weights=weights,
        )

    def _build_unified_representative_plan(
        self,
        tokens: torch.Tensor,
        *,
        reallocate: bool,
        lambda_cost: float | None = None,
    ) -> TemporalRepresentativeBatchPlan:
        """Build SelfTR/U-R plans on the local time/space candidate graph."""

        num_frames, num_tokens, embed_dim = tokens.shape
        patch_count = num_tokens - self.patch_token_start
        patch_tokens = tokens[:, self.patch_token_start :].reshape(-1, embed_dim)
        if lambda_cost is None:
            lambda_cost = self._frame_fusion_lambda_for_layer(-1)
        normalized = torch.nn.functional.normalize(patch_tokens.float(), p=2, dim=-1, eps=1e-8)
        total = num_frames * patch_count
        protected = np.zeros(total, dtype=bool)
        protected[:patch_count] = True
        if reallocate:
            edge_source, edge_target = self._build_local_spatiotemporal_edges(
                num_frames,
                patch_count,
                include_temporal_spatial=True,
                exclude_frame_zero=True,
            )
            candidate_lists = [[] for _ in range(total)]
            for source, target in zip(edge_source.tolist(), edge_target.tolist()):
                candidate_lists[source].append(target)
                candidate_lists[target].append(source)
            for index in range(total):
                candidate_lists[index].insert(0, index)
            max_candidates = max(len(values) for values in candidate_lists)
            candidate_matrix = np.full((total, max_candidates), -1, dtype=np.int64)
            for index, values in enumerate(candidate_lists):
                candidate_matrix[index, : len(values)] = values
            candidate_ids = torch.as_tensor(candidate_matrix, device=tokens.device)
            valid = candidate_ids >= 0
            nearest = torch.empty(total, device=tokens.device, dtype=torch.float32)
            for start in range(0, total, 4_096):
                end = min(start + 4_096, total)
                local_ids = candidate_ids[start:end]
                local_valid = valid[start:end]
                candidate_features = normalized[local_ids.clamp_min(0)]
                local_score = (normalized[start:end, None, :] * candidate_features).sum(dim=-1)
                local_score = local_score.masked_fill(~local_valid, -1e9)
                local_score[:, 0] = -1e9
                nearest[start:end] = local_score.max(dim=-1).values
            # Deleting a singleton representative is cheap when its nearest
            # local candidate is similar.  Frame 0 is never removable.
            removable = np.flatnonzero(~protected)
            keep = np.zeros(total, dtype=bool)
            keep[:patch_count] = True
            nearest_np = nearest.detach().cpu().numpy()
            target_min = max(
                int(np.ceil(total * float(getattr(self, "frame_fusion_min_keep_ratio", 0.05)))),
                patch_count,
            )
            # Larger nearest-neighbor similarity means cheaper deletion, so
            # use cosine distance as the monotone deletion-path error.
            removal_scores = 1.0 - nearest_np
            remove_count, reallocation_objective = self._select_reallocation_prefix(
                removal_scores,
                removable,
                protected_count=patch_count,
                min_keep=target_min,
                lambda_cost=float(lambda_cost),
                cost_denominator=float(max((num_frames - 1) * patch_count, 1)),
            )
            remove_order = removable[np.argsort(removal_scores[removable], kind="stable")]
            keep[remove_order[remove_count:]] = True
            for index in range(patch_count, total):
                if not any(
                    0 <= candidate < total
                    and not protected[candidate]
                    and keep[candidate]
                    for candidate in candidate_lists[index]
                ):
                    keep[index] = True
            keep_ids = np.flatnonzero(keep)
            keep_map = {int(value): index for index, value in enumerate(keep_ids)}
            keep_tensor = torch.as_tensor(keep, device=tokens.device)
            final_mapping = np.empty(total, dtype=np.int64)
            for start in range(0, total, 4_096):
                end = min(start + 4_096, total)
                local_ids = candidate_ids[start:end]
                local_valid = valid[start:end]
                candidate_features = normalized[local_ids.clamp_min(0)]
                local_score = (normalized[start:end, None, :] * candidate_features).sum(dim=-1)
                local_score = local_score.masked_fill(
                    ~local_valid | ~keep_tensor[local_ids.clamp_min(0)],
                    -1e9,
                )
                selected = local_ids.gather(1, local_score.argmax(dim=-1, keepdim=True)).squeeze(1)
                final_mapping[start:end] = selected.detach().cpu().numpy()
            final_mapping = np.asarray([keep_map[int(value)] for value in final_mapping], dtype=np.int64)
            selected_sources = keep_ids
            debug = {
                "initial_active_tokens": total,
                "minimum_active_tokens": target_min,
                "knee_active_tokens": int(selected_sources.size),
                "selected_merges": int(total - selected_sources.size),
                "reassignment": True,
                "selected_objective": float(reallocation_objective),
            }
        else:
            edge_source, edge_target = self._build_local_spatiotemporal_edge_tensors(
                num_frames,
                patch_count,
                include_temporal_spatial=False,
                exclude_frame_zero=True,
                use_spatiotemporal_cube=True,
                device=tokens.device,
                index_offset=patch_count,
                canonical_order=True,
            )
            # Frame zero is a contiguous protected prefix.  Removing it from
            # global edge ids is therefore an on-device scalar subtraction;
            # no NumPy lookup table or host-to-device edge copy is needed.
            search_indices = np.arange(patch_count, total, dtype=np.int64)
            search_edge_source = edge_source
            search_edge_target = edge_target
            search_features = normalized[patch_count:]
            search_mapping, search_sources, debug = self._batch_mutual_nearest_group_merge(
                search_features,
                search_indices,
                search_edge_source,
                search_edge_target,
                protected=np.zeros(search_indices.size, dtype=bool),
                max_group_size=None,
                min_keep_ratio=float(getattr(self, "frame_fusion_min_keep_ratio", 0.05)),
                lambda_cost=float(lambda_cost),
                cost_denominator=float(max((num_frames - 1) * patch_count, 1)),
                prefer_best_parent=True,
                fixed_compression_ratio=getattr(
                    self, "frame_fusion_fixed_compression_ratio", None
                ),
                merge_top_similarity_percent=float(
                    getattr(self, "frame_fusion_merge_top_similarity_percent", 100.0)
                ),
                initial_edges_are_unique=True,
            )
            final_mapping = np.empty(total, dtype=np.int64)
            final_mapping[:patch_count] = np.arange(patch_count, dtype=np.int64)
            final_mapping[search_indices] = patch_count + search_mapping
            selected_sources = np.concatenate(
                [
                    np.arange(patch_count, dtype=np.int64),
                    search_sources,
                ]
            )
            debug.update(
                {
                    "reference_frame_patch_tokens": patch_count,
                    "search_space_patch_tokens": int(search_indices.size),
                    "search_space_excludes_frame_zero": True,
                }
            )
        mapping = torch.as_tensor(final_mapping.reshape(num_frames, patch_count), device=tokens.device, dtype=torch.long)
        source_indices = torch.as_tensor(selected_sources, device=tokens.device, dtype=torch.long)
        weights = torch.bincount(mapping.reshape(-1), minlength=int(source_indices.numel())).float()
        debug.update(
            {
                "representative_count": int(source_indices.numel()),
                "full_patch_tokens": total,
                "representative_patch_tokens": int(source_indices.numel()),
                "attention_tokens": int(num_frames * self.patch_token_start + source_indices.numel()),
                "representative_update": "best-of-parents" if not reallocate else "reassignment",
                "patch_token_retention_vs_full": float(source_indices.numel() / max(total, 1)),
                "representative_weight_min": float(weights.min().item()),
                "representative_weight_max": float(weights.max().item()),
                "representative_weight_mean": float(weights.mean().item()),
                "mapping_checksum": int(mapping.long().sum().item()),
                "mapping_shape": list(mapping.shape),
            }
        )
        self._hybrid_debug = getattr(self, "_hybrid_debug", [])
        self._hybrid_debug.append(debug.copy())
        return TemporalRepresentativeBatchPlan(
            position_to_representative=mapping,
            representative_source_indices=source_indices,
            representative_weights=weights,
        )

    def _build_spatiotemporal_representative_plans(
        self,
        tokens: torch.Tensor,
        *,
        source_layer: int,
    ) -> list[TemporalRepresentativeBatchPlan]:
        """Build one of H-M, H-R, SelfTR, or U-R using a shared plan format."""

        started = time.perf_counter()
        self._hybrid_debug = []
        plans: list[TemporalRepresentativeBatchPlan] = []
        mode = self.frame_fusion_mode
        layer_lambda = self._frame_fusion_lambda_for_layer(source_layer)
        if mode in {"h-m", "h-r"}:
            temporal_plans = self._build_adaptive_temporal_representative_plans(
                tokens, source_layer=source_layer
            )
            for batch_index, temporal_plan in enumerate(temporal_plans):
                plans.append(
                    self._build_hybrid_representative_plan(
                        tokens[batch_index],
                        temporal_plan,
                        reallocate=mode == "h-r",
                    )
                )
        else:
            for batch_index in range(tokens.shape[0]):
                plans.append(
                    self._build_unified_representative_plan(
                        tokens[batch_index],
                        reallocate=mode == "u-r",
                        lambda_cost=layer_lambda,
                    )
                )
        batch_debug = list(self._hybrid_debug)
        first = batch_debug[0] if batch_debug else {}
        uses_lambda = mode in {"h-m", "h-r", "selftr", "u-r"}
        is_reallocation = mode in {"h-r", "u-r"}
        self.last_frame_fusion_debug = {
            "mode": mode,
            "source_layer": source_layer,
            "mapping": "original_token_to_spatiotemporal_representative",
            "cost_model": (
                "local_spatiotemporal_normalized_distortion_plus_lambda"
                if not is_reallocation
                else "local_spatiotemporal_normalized_reallocation_distortion_plus_lambda"
            ),
            "candidate_topology": (
                "spatiotemporal_cube_by_radius_and_window"
                if mode == "selftr"
                else "legacy_spatial_neighborhood"
            ),
            "spatial_neighborhood": (
                None
                if mode == "selftr"
                else getattr(self, "frame_fusion_spatial_neighborhood", "N8")
            ),
            "temporal_window": getattr(self, "frame_fusion_temporal_window", 1),
            "time_overlap": getattr(self, "frame_fusion_time_overlap", 0.5),
            "minimum_keep_ratio": (
                getattr(self, "frame_fusion_min_keep_ratio", 0.05)
            ),
            "max_group_size": (
                None
            ),
            "lambda_cost": (
                layer_lambda
                if uses_lambda
                else None
            ),
            "fixed_compression_ratio": getattr(
                self, "frame_fusion_fixed_compression_ratio", None
            ),
            "merge_top_similarity_percent": float(
                getattr(self, "frame_fusion_merge_top_similarity_percent", 100.0)
            ),
            "lambda_cost_by_layer": {
                str(layer): float(value)
                for layer, value in getattr(self, "frame_fusion_lambda_cost_by_layer", {}).items()
            },
            "selection": (
                "mutual_nearest_neighbor_delta_E_lt_2_lambda"
                if mode == "u-m"
                else "min(D_m_normalized + lambda_cost * M_m_normalized)"
                if uses_lambda
                else "geometric_knee"
            ),
            "stopping_rule": (
                "delta_E < 2 * lambda_cost"
                if mode == "u-m"
                else "full_curve_selection"
                if uses_lambda
                else "geometric_knee"
            ),
            "distortion_normalization": "average_cosine_distance / 2",
            "token_count_normalization": (
                "active_non_reference_tokens / ((F - 1) * P)"
                if uses_lambda
                else None
            ),
            "cost_scope": (
                "non_reference_patch_tokens"
                if uses_lambda
                else "all_candidate_patch_tokens"
            ),
            "cost_denominator": "(F - 1) * P" if uses_lambda else None,
            "max_token_count": (
                float(
                    max(
                        (tokens.shape[1] - 1)
                        * (tokens.shape[2] - self.patch_token_start),
                        1,
                    )
                )
                if uses_lambda
                else None
            ),
            "reassignment_candidates": getattr(self, "frame_fusion_reassignment_candidates", 8),
            "representative_update": (
                "reassignment"
                if is_reallocation
                else "best-of-parents"
                if uses_lambda
                else getattr(self, "frame_fusion_representative_update", "parent")
            ),
            "representative_value_aggregation": (
                "group-mean" if mode == "u-m" else "source-token"
            ),
            "attention_only": True,
            "mlp_scope": "full_original_token_sequence",
            "batches": [
                {
                    **(batch_debug[batch_index] if batch_index < len(batch_debug) else first),
                    "representative_count": int(plan.representative_source_indices.numel()),
                    "full_patch_tokens": int(tokens.shape[2] - self.patch_token_start) * tokens.shape[1],
                    "representative_patch_tokens": int(plan.representative_source_indices.numel()),
                    "attention_tokens": int(tokens.shape[1] * self.patch_token_start + plan.representative_source_indices.numel()),
                    "patch_token_retention_vs_full": float(
                        plan.representative_source_indices.numel()
                        / max(tokens.shape[1] * (tokens.shape[2] - self.patch_token_start), 1)
                    ),
                    "representative_weight_min": float(plan.representative_weights.min().item()),
                    "representative_weight_max": float(plan.representative_weights.max().item()),
                    "representative_weight_mean": float(plan.representative_weights.mean().item()),
                    "mapping_checksum": int(plan.position_to_representative.sum().item()),
                    "mapping_shape": list(plan.position_to_representative.shape),
                }
                for batch_index, plan in enumerate(plans)
            ],
        }
        if batch_debug:
            self.last_frame_fusion_debug.update(
                {
                    "selected_compression_ratio": float(
                        sum(
                            float(item.get("selected_compression_ratio", 0.0))
                            for item in batch_debug
                            if item.get("selected_compression_ratio") is not None
                        )
                        / max(
                            sum(
                                item.get("selected_compression_ratio") is not None
                                for item in batch_debug
                            ),
                            1,
                        )
                    ),
                    "avg_selected_pairs": float(
                        sum(float(item.get("selected_pairs", 0.0)) for item in batch_debug)
                        / max(len(batch_debug), 1)
                    ),
                    "avg_selected_groups": float(
                        sum(float(item.get("selected_groups", 0.0)) for item in batch_debug)
                        / max(len(batch_debug), 1)
                    ),
                }
            )
        self._frame_fusion_plan_seconds = getattr(
            self, "_frame_fusion_plan_seconds", 0.0
        ) + time.perf_counter() - started
        self.last_frame_fusion_debug["planning_seconds"] = float(
            self._frame_fusion_plan_seconds
        )
        if (
            source_layer in getattr(self, "frame_fusion_recompute_layers", ())
            or (
                getattr(self, "frame_fusion_recompute_each_global", False)
                and self.frame_fusion_mode in {"h-m", "h-r", "selftr", "u-r"}
                and (
                    source_layer < 0
                    or self.inter_frame_attention_types[source_layer] == "global"
                )
            )
        ):
            self._frame_fusion_debug_layers.append(self.last_frame_fusion_debug.copy())
        return plans

    def _build_adaptive_spatial_representative_plans(
        self,
        tokens: torch.Tensor,
        *,
        source_layer: int,
    ) -> list[SpatialRepresentativeBatchPlan]:
        """Build independent spatial representative dictionaries per frame.

        Frame 0 is a reference frame and is copied into the compressed
        dictionary one-to-one. For every later frame, the spatial patch tokens
        are normalized and represented with a medoid followed by lazy-greedy
        maximum-error-reduction selection. The selected prefix minimizes
        ``D_S(k) + lambda_cost * k / P``. The resulting assignment is only used
        for global attention; the attention residual is expanded back to every
        original patch position before the per-token MLP runs.
        """

        batch_size, num_frames, num_tokens, embed_dim = tokens.shape
        patch_tokens = tokens[:, :, self.patch_token_start :]
        patch_count = int(patch_tokens.shape[2])
        plans: list[SpatialRepresentativeBatchPlan] = []
        debug_batches: list[dict[str, object]] = []
        selection_started = time.perf_counter()

        for batch_index in range(batch_size):
            mapping = torch.empty(
                (num_frames, patch_count),
                device=tokens.device,
                dtype=torch.long,
            )
            mapping[0] = torch.arange(patch_count, device=tokens.device)
            source_chunks = [
                torch.arange(patch_count, device=tokens.device, dtype=torch.long)
            ]
            frame_debug: list[dict[str, object]] = []

            for frame_index in range(1, num_frames):
                with torch.autocast(device_type=tokens.device.type, enabled=False):
                    normalized = torch.nn.functional.normalize(
                        patch_tokens[batch_index, frame_index].float(),
                        p=2,
                        dim=-1,
                        eps=1e-8,
                    )
                    similarities = (normalized @ normalized.transpose(0, 1)).clamp(-1.0, 1.0)
                distance = (
                    1.0 - similarities
                ).detach().float().cpu().numpy().astype(np.float32)

                medoid = int(np.argmin(distance.sum(axis=1)))
                selected = [medoid]
                current_error = distance[:, medoid].copy()
                initial_distortion = float(current_error.mean())
                best_k = 1
                best_distortion = initial_distortion
                best_objective = initial_distortion + self.frame_fusion_lambda_cost / max(
                    patch_count, 1
                )
                cost_per_token = self.frame_fusion_lambda_cost / max(patch_count, 1)

                # The gain of an unselected candidate is monotone non-increasing
                # as representatives are added. Stored heap values are upper
                # bounds; recomputing only the current heap maximum avoids the
                # O(P^2 K) cost of rebuilding all candidate gains at every step.
                candidates = np.arange(patch_count, dtype=np.int64)
                candidates = candidates[candidates != medoid]
                if candidates.size:
                    initial_gains = np.maximum(
                        current_error[:, None] - distance[:, candidates], 0.0
                    ).mean(axis=0)
                    heap: list[tuple[float, int]] = [
                        (-float(gain), int(candidate))
                        for gain, candidate in zip(initial_gains, candidates)
                    ]
                    heapq.heapify(heap)
                else:
                    heap = []

                while heap and len(selected) < patch_count:
                    _, candidate = heapq.heappop(heap)
                    if candidate in selected:
                        continue
                    exact_gain = float(
                        np.maximum(current_error - distance[:, candidate], 0.0).mean()
                    )
                    if heap and exact_gain + 1e-12 < -heap[0][0]:
                        heapq.heappush(heap, (-exact_gain, candidate))
                        continue
                    if exact_gain <= cost_per_token:
                        break

                    selected.append(candidate)
                    current_error = np.minimum(current_error, distance[:, candidate])
                    current_distortion = float(current_error.mean())
                    objective = current_distortion + cost_per_token * len(selected)
                    if objective < best_objective:
                        best_k = len(selected)
                        best_distortion = current_distortion
                        best_objective = objective

                selected = selected[:best_k]
                assignment = np.argmin(distance[:, selected], axis=1)
                representative_offset = sum(int(chunk.numel()) for chunk in source_chunks)
                representative_ids = np.arange(
                    representative_offset,
                    representative_offset + best_k,
                    dtype=np.int64,
                )
                mapping[frame_index] = torch.as_tensor(
                    representative_ids[assignment],
                    device=tokens.device,
                    dtype=torch.long,
                )
                source_chunks.append(
                    torch.as_tensor(
                        frame_index * patch_count + np.asarray(selected, dtype=np.int64),
                        device=tokens.device,
                        dtype=torch.long,
                    )
                )
                frame_debug.append(
                    {
                        "frame_index": frame_index,
                        "medoid_index": medoid,
                        "representative_count": best_k,
                        "full_patch_tokens": patch_count,
                        "initial_distortion": initial_distortion,
                        "optimal_distortion": best_distortion,
                        "optimal_score": best_objective,
                        "compute_saving_vs_frame": 1.0 - best_k / max(patch_count, 1),
                    }
                )

            source_indices = torch.cat(source_chunks, dim=0)
            weights = torch.bincount(
                mapping.reshape(-1), minlength=int(source_indices.numel())
            ).float()
            optimal_counts = [
                int(frame_info["representative_count"]) for frame_info in frame_debug
            ]
            plans.append(
                SpatialRepresentativeBatchPlan(
                    position_to_representative=mapping,
                    representative_source_indices=source_indices,
                    representative_weights=weights,
                )
            )
            debug_batches.append(
                {
                    "reference_frame_index": 0,
                    "processed_frame_indices": list(range(1, num_frames)),
                    "frame_representative_counts": optimal_counts,
                    "representative_count": int(source_indices.numel()),
                    "full_patch_tokens": int(num_frames * patch_count),
                    "representative_patch_tokens": int(source_indices.numel()),
                    "attention_tokens": int(
                        num_frames * self.patch_token_start + source_indices.numel()
                    ),
                    "patch_token_retention_vs_full": float(
                        source_indices.numel() / max(num_frames * patch_count, 1)
                    ),
                    "representative_weight_min": float(weights.min().item()),
                    "representative_weight_max": float(weights.max().item()),
                    "representative_weight_mean": float(weights.mean().item()),
                    "mapping_checksum": int(mapping.long().sum().item()),
                    "mapping_shape": list(mapping.shape),
                    "frames": frame_debug,
                }
            )

        selection_seconds = time.perf_counter() - selection_started
        self.last_frame_fusion_debug = {
            "mode": self.frame_fusion_mode,
            "source_layer": source_layer,
            "num_frames": num_frames,
            "tokens_per_frame": num_tokens,
            "patch_tokens_per_frame": patch_count,
            "embed_dim": embed_dim,
            "mapping": "frame_position_to_spatial_representative",
            "weighting": "uniform_representative_keys",
            "cost_model": "spatial_distortion_plus_lambda_k_over_P",
            "lambda_cost": self.frame_fusion_lambda_cost,
            "selection": "medoid_then_lazy_greedy_max_distortion_reduction",
            "reference_frame_index": 0,
            "reference_frame_compression": "none",
            "attention_only": True,
            "mlp_scope": "full_original_token_sequence",
            "mapping_preserved": True,
            "selection_seconds": selection_seconds,
            "full_patch_tokens": int(num_frames * patch_count),
            "retained_patch_tokens": int(debug_batches[0]["representative_patch_tokens"])
            if debug_batches
            else 0,
            "patch_token_retention_vs_full": float(
                debug_batches[0]["patch_token_retention_vs_full"]
            )
            if debug_batches
            else 1.0,
            "batches": debug_batches,
        }
        return plans

    def _build_adaptive_temporal_representative_plans(
        self,
        tokens: torch.Tensor,
        *,
        source_layer: int,
    ) -> list[TemporalRepresentativeBatchPlan]:
        """Build the globally optimized per-position temporal dictionary.

        Every spatial position starts with one segment [1, F-1], represented by
        frame 1. Each heap step splits one segment at the position with the
        largest distortion reduction. The knee of the compute-distortion curve
        selects the final prefix of split operations.
        """

        batch_size, num_frames, num_tokens, embed_dim = tokens.shape
        patch_tokens = tokens[:, :, self.patch_token_start :]
        patch_count = int(patch_tokens.shape[2])
        plans: list[TemporalRepresentativeBatchPlan] = []
        debug_batches: list[dict[str, object]] = []
        selection_started = time.perf_counter()

        for batch_index in range(batch_size):
            if num_frames <= 1:
                mapping = torch.arange(
                    patch_count,
                    device=tokens.device,
                    dtype=torch.long,
                ).view(1, patch_count)
                source_indices = torch.arange(
                    patch_count,
                    device=tokens.device,
                    dtype=torch.long,
                )
                weights = torch.ones(patch_count, device=tokens.device, dtype=torch.float32)
                plans.append(
                    TemporalRepresentativeBatchPlan(
                        position_to_representative=mapping,
                        representative_source_indices=source_indices,
                        representative_weights=weights,
                    )
                )
                debug_batches.append(
                    {
                        "representative_count": patch_count,
                        "full_patch_tokens": patch_count,
                        "representative_patch_tokens": patch_count,
                        "attention_tokens": patch_count + self.patch_token_start,
                        "patch_token_retention_vs_full": 1.0,
                        "optimal_split_count": 0,
                        "max_split_count": 0,
                        "initial_distortion": 0.0,
                        "optimal_score": 0.0,
                        "mapping_checksum": int(mapping.sum().item()),
                        "mapping_shape": list(mapping.shape),
                    }
                )
                continue

            with torch.autocast(device_type=tokens.device.type, enabled=False):
                normalized = torch.nn.functional.normalize(
                    patch_tokens[batch_index].float(), p=2, dim=-1, eps=1e-8
                )
                similarities = torch.einsum(
                    "tpd,spd->pts", normalized, normalized
                ).clamp(-1.0, 1.0)
            distance = (1.0 - similarities).detach().float().cpu().numpy().astype(np.float32)
            prefix = np.concatenate(
                [
                    np.zeros((patch_count, num_frames, 1), dtype=np.float32),
                    np.cumsum(distance, axis=2, dtype=np.float32),
                ],
                axis=2,
            )
            denominator = float(max((num_frames - 1) * patch_count, 1))
            initial_error = prefix[:, 1, num_frames] - prefix[:, 1, 1]
            initial_distortion = float(
                initial_error.sum()
                / (_FRAME_FUSION_MAX_COSINE_DISTANCE * denominator)
            )

            # Segment state is keyed by an id. Splitting keeps the old id for
            # the left segment and allocates one new id for the right segment.
            segment_state: dict[int, tuple[int, int, int, int]] = {}
            heap: list[tuple[float, int, int, int, int, int]] = []
            next_segment_id = patch_count
            split_operations: list[tuple[int, int, int, int, int, int]] = []

            def best_split(position: int, start: int, end: int) -> tuple[float, int]:
                if start >= end:
                    return float("-inf"), -1
                candidates = np.arange(start + 1, end + 1, dtype=np.int64)
                old_error = prefix[position, start, end + 1] - prefix[position, start, start]
                left_error = prefix[position, start, candidates] - prefix[position, start, start]
                right_error = (
                    prefix[position, candidates, end + 1]
                    - prefix[position, candidates, candidates]
                )
                gains = old_error - left_error - right_error
                best_index = int(np.argmax(gains))
                return float(gains[best_index]), int(candidates[best_index])

            def push_best(segment_id: int) -> None:
                state = segment_state.get(segment_id)
                if state is None:
                    return
                position, start, end, _ = state
                gain, split = best_split(position, start, end)
                if split >= 0 and np.isfinite(gain):
                    heapq.heappush(
                        heap,
                        (-gain, position, segment_id, start, end, split),
                    )

            for position in range(patch_count):
                segment_state[position] = (position, 1, num_frames - 1, patch_count + position)
                push_best(position)

            max_split_count = patch_count * max(num_frames - 2, 0)
            # The reference frame is fixed.  Lambda therefore charges only
            # active non-reference patch representatives, normalized by the
            # original non-reference patch-token count.
            non_reference_denominator = float(max((num_frames - 1) * patch_count, 1))
            current_distortion = initial_distortion
            best_k = 0
            best_objective = current_distortion + self.frame_fusion_lambda_cost * (
                patch_count / non_reference_denominator
            )
            best_score = best_objective
            best_compute_saving = 1.0 - patch_count / non_reference_denominator
            best_distortion = initial_distortion

            for split_count in range(1, max_split_count + 1):
                while heap:
                    neg_gain, position, segment_id, start, end, split = heapq.heappop(heap)
                    state = segment_state.get(segment_id)
                    if state is not None and state[:3] == (position, start, end):
                        break
                else:
                    break

                _, _, _, old_representative = segment_state.pop(segment_id)
                new_segment_id = next_segment_id
                next_segment_id += 1
                new_representative = 2 * patch_count + len(split_operations)
                segment_state[segment_id] = (
                    position,
                    start,
                    split - 1,
                    old_representative,
                )
                segment_state[new_segment_id] = (
                    position,
                    split,
                    end,
                    new_representative,
                )
                split_operations.append(
                    (position, segment_id, new_segment_id, start, end, split)
                )

                gain = -neg_gain
                current_distortion = max(
                    0.0,
                    current_distortion
                    - gain / (_FRAME_FUSION_MAX_COSINE_DISTANCE * denominator),
                )
                push_best(segment_id)
                push_best(new_segment_id)

                active_non_reference_tokens = patch_count + split_count
                q = float(active_non_reference_tokens / non_reference_denominator)
                objective = current_distortion + self.frame_fusion_lambda_cost * q
                if objective < best_objective:
                    best_k = split_count
                    best_objective = objective
                    best_score = objective
                    best_compute_saving = 1.0 - q
                    best_distortion = current_distortion

            # Reconstruct the selected prefix of splits. This keeps the curve
            # search independent from the final mapping representation.
            final_segments: dict[int, tuple[int, int, int, int]] = {
                position: (position, 1, num_frames - 1, patch_count + position)
                for position in range(patch_count)
            }
            for operation_index, operation in enumerate(split_operations[:best_k]):
                position, segment_id, new_segment_id, start, end, split = operation
                state = final_segments.pop(segment_id)
                if state[:3] != (position, start, end):
                    raise RuntimeError("adaptive temporal split replay diverged from heap state")
                old_representative = state[3]
                final_segments[segment_id] = (
                    position,
                    start,
                    split - 1,
                    old_representative,
                )
                final_segments[new_segment_id] = (
                    position,
                    split,
                    end,
                    2 * patch_count + operation_index,
                )

            mapping = torch.empty(
                (num_frames, patch_count),
                device=tokens.device,
                dtype=torch.long,
            )
            mapping[0] = torch.arange(patch_count, device=tokens.device)
            mapping[1] = patch_count + torch.arange(patch_count, device=tokens.device)
            source_chunks = [
                torch.arange(patch_count, device=tokens.device, dtype=torch.long),
                patch_count + torch.arange(patch_count, device=tokens.device, dtype=torch.long),
            ]
            for operation_index, operation in enumerate(split_operations[:best_k]):
                position, _, _, _, _, split = operation
                source_chunks.append(
                    torch.tensor(
                        [split * patch_count + position],
                        device=tokens.device,
                        dtype=torch.long,
                    )
                )
            for position, start, end, representative in final_segments.values():
                mapping[start : end + 1, position] = representative
            source_indices = torch.cat(source_chunks, dim=0)
            weights = torch.bincount(
                mapping.reshape(-1), minlength=int(source_indices.numel())
            ).float()
            plans.append(
                TemporalRepresentativeBatchPlan(
                    position_to_representative=mapping,
                    representative_source_indices=source_indices,
                    representative_weights=weights,
                )
            )
            debug_batches.append(
                {
                    "representative_count": int(source_indices.numel()),
                    "full_patch_tokens": int(num_frames * patch_count),
                    "representative_patch_tokens": int(source_indices.numel()),
                    "attention_tokens": int(
                        num_frames * self.patch_token_start + source_indices.numel()
                    ),
                    "patch_token_retention_vs_full": float(
                        source_indices.numel() / max(num_frames * patch_count, 1)
                    ),
                    "optimal_split_count": int(best_k),
                    "max_split_count": int(max_split_count),
                    "initial_distortion": initial_distortion,
                    "optimal_distortion": float(best_distortion),
                    "optimal_score": float(best_score),
                    "optimal_compute_saving": float(best_compute_saving),
                    "representative_weight_min": float(weights.min().item()),
                    "representative_weight_max": float(weights.max().item()),
                    "representative_weight_mean": float(weights.mean().item()),
                    "mapping_checksum": int(mapping.long().sum().item()),
                    "mapping_shape": list(mapping.shape),
                }
            )

        selection_seconds = time.perf_counter() - selection_started
        self.last_frame_fusion_debug = {
            "mode": self.frame_fusion_mode,
            "source_layer": source_layer,
            "num_frames": num_frames,
            "tokens_per_frame": num_tokens,
            "patch_tokens_per_frame": patch_count,
            "embed_dim": embed_dim,
            "mapping": "position_to_temporal_segment_representative",
            "weighting": "uniform_representative_keys",
            "cost_model": "normalized_temporal_distortion_plus_lambda",
            "lambda_cost": self.frame_fusion_lambda_cost,
            "selection": "min(D_t_normalized + lambda_cost * M_t_normalized)",
            "distortion_normalization": "average_cosine_distance / 2",
            "token_count_normalization": "active_non_reference_tokens / ((F - 1) * P)",
            "mapping_preserved": True,
            "selection_seconds": selection_seconds,
            "full_patch_tokens": int(num_frames * patch_count),
            "retained_patch_tokens": int(debug_batches[0]["representative_patch_tokens"])
            if debug_batches
            else 0,
            "patch_token_retention_vs_full": float(
                debug_batches[0]["patch_token_retention_vs_full"]
            )
            if debug_batches
            else 1.0,
            "batches": debug_batches,
        }
        return plans

    def _select_frame_fusion_target_keep_patch_indices(
        self,
        patch_tokens: torch.Tensor,
        selected_pairs: list[FrameFusionPair],
        *,
        patch_grid_size: tuple[int, int],
        source_layer: int,
        batch_index: int,
    ) -> torch.Tensor:
        if not selected_pairs or self.frame_fusion_target_keep_policy == "none":
            return torch.empty((len(selected_pairs), 0), device=patch_tokens.device, dtype=torch.long)

        patch_h, patch_w = patch_grid_size
        patch_count = patch_tokens.shape[1]
        if patch_count != patch_h * patch_w:
            raise ValueError(
                "patch token count does not match grid size: "
                f"{patch_count} vs {patch_h}x{patch_w}"
            )

        if self.frame_fusion_target_keep_policy == "random-grid":
            keep_rows: list[torch.Tensor] = []
            block_size = self.frame_fusion_target_keep_grid_size
            block_offsets = []
            for row_start in range(0, patch_h, block_size):
                row_end = min(row_start + block_size, patch_h)
                for col_start in range(0, patch_w, block_size):
                    col_end = min(col_start + block_size, patch_w)
                    block_offsets.append(
                        [
                            row * patch_w + col
                            for row in range(row_start, row_end)
                            for col in range(col_start, col_end)
                        ]
                    )
            for pair_index, _ in enumerate(selected_pairs):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    self.frame_fusion_target_keep_seed
                    + (source_layer + 1) * 1_000_003
                    + batch_index * 10_007
                    + pair_index
                )
                chosen = []
                for offsets in block_offsets:
                    local_index = int(
                        torch.randint(
                            low=0,
                            high=len(offsets),
                            size=(1,),
                            generator=generator,
                        ).item()
                    )
                    chosen.append(offsets[local_index])
                keep_rows.append(torch.tensor(chosen, device=patch_tokens.device, dtype=torch.long))
            return torch.stack(keep_rows, dim=0)

        if self.frame_fusion_target_keep_policy == "least-similar":
            keep_count = int(
                torch.ceil(
                    torch.tensor(
                        patch_count * self.frame_fusion_target_keep_percent / 100.0,
                        dtype=torch.float32,
                    )
                ).item()
            )
            keep_count = min(patch_count, max(1, keep_count))
            keep_rows = []
            for pair in selected_pairs:
                source = patch_tokens[pair.frame_a].float()
                target = patch_tokens[pair.frame_b].float()
                token_similarity = torch.nn.functional.cosine_similarity(
                    source,
                    target,
                    dim=-1,
                    eps=1e-8,
                )
                keep_rows.append(torch.topk(token_similarity, k=keep_count, largest=False).indices)
            return torch.stack(keep_rows, dim=0).to(device=patch_tokens.device, dtype=torch.long)

        if self.frame_fusion_target_keep_policy == "similarity-threshold":
            keep_rows = []
            for pair in selected_pairs:
                source = patch_tokens[pair.frame_a].float()
                target = patch_tokens[pair.frame_b].float()
                token_similarity = torch.nn.functional.cosine_similarity(
                    source,
                    target,
                    dim=-1,
                    eps=1e-8,
                )
                keep_rows.append(token_similarity < self.frame_fusion_target_keep_threshold)
            return torch.stack(keep_rows, dim=0).to(device=patch_tokens.device, dtype=torch.bool)

        raise RuntimeError(f"Unsupported target keep policy: {self.frame_fusion_target_keep_policy}")

    @staticmethod
    def _select_frame_fusion_group_shared_keep_patch_indices(
        patch_tokens: torch.Tensor,
        groups: tuple[FrameFusionGroup, ...],
        *,
        threshold: float,
    ) -> torch.Tensor:
        """Return per-target masks from mean pairwise token similarity in each group."""

        patch_count = int(patch_tokens.shape[1])
        keep_rows: list[torch.Tensor] = []
        for group in groups:
            member_indices = torch.tensor(
                group.members,
                device=patch_tokens.device,
                dtype=torch.long,
            )
            members = patch_tokens.index_select(0, member_indices).float()
            normalized = torch.nn.functional.normalize(members, p=2, dim=-1, eps=1e-8)
            pair_count = len(group.members) * (len(group.members) - 1) // 2
            if pair_count <= 0:
                continue
            pairwise_sum = torch.zeros(
                patch_count,
                device=patch_tokens.device,
                dtype=torch.float32,
            )
            for first in range(len(group.members)):
                for second in range(first + 1, len(group.members)):
                    pairwise_sum += (normalized[first] * normalized[second]).sum(dim=-1)
            mean_similarity = pairwise_sum / float(pair_count)
            keep_mask = mean_similarity < float(threshold)
            keep_rows.extend(keep_mask.clone() for _ in group.members[1:])
        if not keep_rows:
            return torch.empty(
                (0, patch_count),
                device=patch_tokens.device,
                dtype=torch.bool,
            )
        return torch.stack(keep_rows, dim=0)

    @staticmethod
    def _attention_error_query_indices(
        *,
        num_frames: int,
        num_tokens: int,
        patch_token_start: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Select all special queries and ten uniformly spaced patch queries per frame."""

        special_count = num_frames * patch_token_start
        frame_offsets = torch.arange(num_frames, device=device, dtype=torch.long) * patch_token_start
        indices = {
            "camera": frame_offsets,
            "register": (
                frame_offsets[:, None]
                + torch.arange(1, patch_token_start, device=device, dtype=torch.long)[None, :]
            ).reshape(-1),
        }
        patch_count = num_tokens - patch_token_start
        patch_sample_count = min(10, patch_count)
        patch_positions = torch.linspace(
            0,
            max(patch_count - 1, 0),
            patch_sample_count,
            device=device,
        ).round().to(dtype=torch.long).unique()
        indices["patch"] = (
            special_count
            + torch.arange(num_frames, device=device, dtype=torch.long)[:, None] * patch_count
            + patch_positions[None, :]
        ).reshape(-1)
        return indices

    def _record_attention_error(
        self,
        dense_attention: torch.Tensor,
        compressed_attention: torch.Tensor,
        *,
        num_frames: int,
        num_tokens: int,
        patch_token_start: int,
    ) -> None:
        """Accumulate dense-vs-compressed attention errors by query token type."""

        dense_attention = dense_attention.detach().float().squeeze(0)
        compressed_attention = compressed_attention.detach().float().squeeze(0)
        query_indices = self._attention_error_query_indices(
            num_frames=num_frames,
            num_tokens=num_tokens,
            patch_token_start=patch_token_start,
            device=dense_attention.device,
        )
        for token_type, indices in query_indices.items():
            dense = dense_attention.index_select(0, indices)
            compressed = compressed_attention.index_select(0, indices)
            difference = compressed - dense
            stats = self._frame_fusion_attention_error_stats.setdefault(
                token_type,
                {
                    "sum_abs": 0.0,
                    "sum_squared": 0.0,
                    "sum_reference_squared": 0.0,
                    "max_abs": 0.0,
                    "query_tokens": 0.0,
                    "values": 0.0,
                },
            )
            stats["sum_abs"] += float(difference.abs().sum().item())
            stats["sum_squared"] += float(difference.square().sum().item())
            stats["sum_reference_squared"] += float(dense.square().sum().item())
            stats["max_abs"] = max(stats["max_abs"], float(difference.abs().max().item()))
            stats["query_tokens"] += float(indices.numel())
            stats["values"] += float(difference.numel())
        self._frame_fusion_attention_error_comparisons += 1

    def _finalize_attention_error_stats(self) -> dict[str, object]:
        result: dict[str, object] = {
            "comparison": "pre_projection_attention_value",
            "query_sampling": "all_camera_and_register; 10_uniform_patch_queries_per_frame",
            "num_comparisons": int(self._frame_fusion_attention_error_comparisons),
            "token_types": {},
        }
        token_types = result["token_types"]
        assert isinstance(token_types, dict)
        for token_type, stats in self._frame_fusion_attention_error_stats.items():
            values = max(stats["values"], 1.0)
            token_types[token_type] = {
                "query_tokens": int(stats["query_tokens"]),
                "mean_absolute_error": stats["sum_abs"] / values,
                "rmse": (stats["sum_squared"] / values) ** 0.5,
                "relative_l2": (
                    stats["sum_squared"]
                    / max(stats["sum_reference_squared"], 1.0e-12)
                )
                ** 0.5,
                "max_absolute_error": stats["max_abs"],
            }
        return result

    def _run_temporal_representative_global_attention_block(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        plans: list[TemporalRepresentativeBatchPlan],
    ) -> torch.Tensor:
        """Run global attention on representatives and restore the position map."""

        started = time.perf_counter()
        block = self.inter_frame_blocks[block_idx]
        patch_start = self.patch_token_start
        patch_count = num_tokens - patch_start
        outputs: list[torch.Tensor] = []
        for batch_index, plan in enumerate(plans):
            frame_tokens = tokens[batch_index]
            special_tokens = frame_tokens[:, :patch_start].reshape(-1, embed_dim)
            patch_tokens = frame_tokens[:, patch_start:].reshape(-1, embed_dim)
            if self.frame_fusion_mode == "u-m":
                # Keep the final U-M partition fixed, but summarize each
                # group with all of its current tokens instead of one parent.
                representatives = self._mean_group_representatives(
                    patch_tokens,
                    plan.position_to_representative.to(device=tokens.device).reshape(-1),
                    representative_count=plan.representative_source_indices.numel(),
                )
            elif (
                self.frame_fusion_mode == "u-m"
                and self.frame_fusion_attention_variant == "last-representative"
            ):
                representatives = self._last_group_representatives(
                    patch_tokens,
                    plan.representative_source_indices.to(device=tokens.device),
                )
            else:
                representatives = patch_tokens.index_select(
                    0,
                    plan.representative_source_indices.to(device=tokens.device),
                )
            compressed = torch.cat([special_tokens, representatives], dim=0).unsqueeze(0)

            block.attn.merge_random_seed = self.merge_random_seed
            if self.frame_fusion_attention_variant == "kv-only":
                # Match the compressed path's stable layout: all camera/
                # register tokens first, followed by all patch tokens. This
                # keeps the residual restoration map aligned with Q output.
                full_input = torch.cat([special_tokens, patch_tokens], dim=0).unsqueeze(0)
                normalized_full = block.norm1(full_input)
                normalized_compressed = block.norm1(compressed)
                full_q, full_k, full_v = block.attn.project_qkv(normalized_full)
                _, compressed_k, compressed_v = block.attn.project_qkv(normalized_compressed)
                dense_attention = block.attn.attention_from_qkv(full_q, full_k, full_v)
                compressed_attention = block.attn.attention_from_qkv(
                    full_q,
                    compressed_k,
                    compressed_v,
                )
                self._record_attention_error(
                    dense_attention,
                    compressed_attention,
                    num_frames=num_frames,
                    num_tokens=num_tokens,
                    patch_token_start=patch_start,
                )
                # Keep the representative keys uniform. In particular, do
                # not pass an additive key bias: a non-null attn_mask disables
                # FlashAttention in scaled_dot_product_attention.
                attention_output = block.attn.proj(compressed_attention)
                attention_output = block.attn.proj_drop(attention_output)
            elif self.frame_fusion_attention_variant == "representative-log":
                full_input = torch.cat([special_tokens, patch_tokens], dim=0).unsqueeze(0)
                normalized_full = block.norm1(full_input)
                normalized_compressed = block.norm1(compressed)
                full_q, full_k, full_v = block.attn.project_qkv(normalized_full)
                compressed_q, compressed_k, compressed_v = block.attn.project_qkv(
                    normalized_compressed
                )
                dense_attention = block.attn.attention_from_qkv(full_q, full_k, full_v)
                mapping = plan.position_to_representative.to(device=tokens.device).reshape(-1)
                group_sizes = torch.bincount(
                    mapping,
                    minlength=representatives.shape[0],
                ).to(device=tokens.device, dtype=compressed_k.dtype)
                log_key_weights = torch.cat(
                    [
                        torch.zeros(
                            special_tokens.shape[0],
                            device=tokens.device,
                            dtype=compressed_k.dtype,
                        ),
                        group_sizes.clamp_min(1).log(),
                    ]
                ).view(1, 1, 1, -1)
                compressed_attention = block.attn.attention_from_qkv_with_log_key_weights(
                    compressed_q,
                    compressed_k,
                    compressed_v,
                    log_key_weights,
                )
                expanded_attention_for_error = torch.cat(
                    [
                        compressed_attention[:, : special_tokens.shape[0]],
                        compressed_attention[:, special_tokens.shape[0] :].index_select(
                            1,
                            mapping,
                        ),
                    ],
                    dim=1,
                )
                self._record_attention_error(
                    dense_attention,
                    expanded_attention_for_error,
                    num_frames=num_frames,
                    num_tokens=num_tokens,
                    patch_token_start=patch_start,
                )
                attention_output = block.attn.proj(compressed_attention)
                attention_output = block.attn.proj_drop(attention_output)
            elif self.frame_fusion_attention_variant == "replicated":
                expanded_patch = representatives.index_select(
                    0,
                    plan.position_to_representative.to(device=tokens.device).reshape(-1),
                )
                expanded = torch.cat([special_tokens, expanded_patch], dim=0).unsqueeze(0)
                full_input = torch.cat([special_tokens, patch_tokens], dim=0).unsqueeze(0)
                normalized_full = block.norm1(full_input)
                normalized_expanded = block.norm1(expanded)
                full_q, full_k, full_v = block.attn.project_qkv(normalized_full)
                expanded_q, expanded_k, expanded_v = block.attn.project_qkv(normalized_expanded)
                dense_attention = block.attn.attention_from_qkv(full_q, full_k, full_v)
                expanded_attention = block.attn.attention_from_qkv(
                    expanded_q,
                    expanded_k,
                    expanded_v,
                )
                self._record_attention_error(
                    dense_attention,
                    expanded_attention,
                    num_frames=num_frames,
                    num_tokens=num_tokens,
                    patch_token_start=patch_start,
                )
                attention_output = block.attn.proj(expanded_attention)
                attention_output = block.attn.proj_drop(attention_output)
            else:
                normalized = block.norm1(compressed)
                # Keep the representative keys uniform. In particular, do not
                # pass an additive key bias here: a non-null attn_mask disables
                # FlashAttention in scaled_dot_product_attention.
                attention_output = block.attn(normalized)
            attention_update = block.ls1(attention_output).squeeze(0)

            # Restore only the attention residual to the original full token
            # sequence. The MLP must see each frame's original token rather
            # than a representative-expanded approximation.
            restored_special_update = attention_update[: special_tokens.shape[0]].view(
                num_frames,
                patch_start,
                embed_dim,
            )
            restored_representative_update = attention_update[special_tokens.shape[0] :]
            restored_patch_update = restored_representative_update.index_select(
                0,
                plan.position_to_representative.to(device=tokens.device).reshape(-1),
            ).view(num_frames, patch_count, embed_dim)
            restored_attention_update = torch.cat(
                [restored_special_update, restored_patch_update], dim=1
            )
            full_tokens = frame_tokens + restored_attention_update
            full_tokens = full_tokens + block.ls2(block.mlp(block.norm2(full_tokens)))
            outputs.append(full_tokens)

        result = torch.stack(outputs, dim=0)
        self._frame_fusion_global_attention_seconds += time.perf_counter() - started
        return result

    @staticmethod
    def _mean_group_representatives(
        patch_tokens: torch.Tensor,
        position_to_representative: torch.Tensor,
        *,
        representative_count: int,
    ) -> torch.Tensor:
        """Construct one mean token for every final representative group."""

        representative_sums = torch.zeros(
            (representative_count, patch_tokens.shape[-1]),
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        representative_sums.index_add_(
            0,
            position_to_representative,
            patch_tokens,
        )
        counts = torch.bincount(
            position_to_representative,
            minlength=representative_count,
        ).to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        return representative_sums / counts.clamp_min(1).unsqueeze(-1)

    @staticmethod
    def _last_group_representatives(
        patch_tokens: torch.Tensor,
        representative_source_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return the final representative source token selected by the merge."""

        return patch_tokens.index_select(0, representative_source_indices)

    def _run_adaptive_spatial_representative_global_attention_block(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        plans: list[SpatialRepresentativeBatchPlan],
    ) -> torch.Tensor:
        """Run global attention on spatial representatives and restore all frames."""

        block = self.inter_frame_blocks[block_idx]
        patch_start = self.patch_token_start
        patch_count = num_tokens - patch_start
        outputs: list[torch.Tensor] = []
        for batch_index, plan in enumerate(plans):
            frame_tokens = tokens[batch_index]
            special_tokens = frame_tokens[:, :patch_start].reshape(-1, embed_dim)
            patch_tokens = frame_tokens[:, patch_start:].reshape(-1, embed_dim)
            representatives = patch_tokens.index_select(
                0,
                plan.representative_source_indices.to(device=tokens.device),
            )
            compressed = torch.cat([special_tokens, representatives], dim=0).unsqueeze(0)

            block.attn.merge_random_seed = self.merge_random_seed
            normalized = block.norm1(compressed)
            # Keep the representative keys uniform so SDPA can select the
            # FlashAttention kernel instead of falling back to a masked path.
            attention_output = block.attn(normalized)
            attention_update = block.ls1(attention_output).squeeze(0)

            restored_special_update = attention_update[: special_tokens.shape[0]].view(
                num_frames,
                patch_start,
                embed_dim,
            )
            restored_representative_update = attention_update[special_tokens.shape[0] :]
            restored_patch_update = restored_representative_update.index_select(
                0,
                plan.position_to_representative.to(device=tokens.device).reshape(-1),
            ).view(num_frames, patch_count, embed_dim)
            restored_attention_update = torch.cat(
                [restored_special_update, restored_patch_update], dim=1
            )
            full_tokens = frame_tokens + restored_attention_update
            full_tokens = full_tokens + block.ls2(block.mlp(block.norm2(full_tokens)))
            outputs.append(full_tokens)

        return torch.stack(outputs, dim=0)

    def _run_pair_fused_global_attention_block(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        frame_fusion_pair_plans: list[FrameFusionBatchPlan],
    ) -> torch.Tensor:
        tokens = self._fuse_frame_pair_patch_tokens(tokens, frame_fusion_pair_plans)
        outputs: list[torch.Tensor] = []
        block = self.inter_frame_blocks[block_idx]
        for batch_index in range(batch_size):
            plan = frame_fusion_pair_plans[batch_index]
            flat_tokens = tokens[batch_index].reshape(num_frames * num_tokens, embed_dim)
            attention_indices = plan.attention_indices.to(device=tokens.device)
            if attention_indices.numel() == flat_tokens.shape[0]:
                attended = block(flat_tokens.unsqueeze(0), None).squeeze(0)
                outputs.append(attended.view(num_frames, num_tokens, embed_dim))
                continue

            compressed_tokens = flat_tokens.index_select(0, attention_indices).unsqueeze(0)
            compressed_tokens = block(compressed_tokens, None).squeeze(0)
            restored = flat_tokens.clone()
            restored.index_copy_(0, attention_indices, compressed_tokens)
            restored = self._copy_pair_patch_outputs(
                restored,
                plan,
                tokens_per_frame=num_tokens,
                num_special_tokens=self.patch_token_start,
            )
            outputs.append(restored.view(num_frames, num_tokens, embed_dim))
        return torch.stack(outputs, dim=0)

    def _fuse_frame_pair_patch_tokens(
        self,
        tokens: torch.Tensor,
        frame_fusion_pair_plans: list[FrameFusionBatchPlan],
    ) -> torch.Tensor:
        output = tokens.clone()
        patch_start = self.patch_token_start
        patch_slice = slice(patch_start, None)
        for batch_index, plan in enumerate(frame_fusion_pair_plans):
            if not plan.pairs:
                continue
            source_frames = plan.source_frames.to(device=tokens.device, dtype=torch.long)
            target_frames = plan.target_frames.to(device=tokens.device, dtype=torch.long)
            patch_count = tokens.shape[2] - patch_start
            target_keep_mask = _frame_fusion_target_keep_patch_mask(
                plan.target_keep_patch_indices,
                num_pairs=int(source_frames.numel()),
                patch_count=patch_count,
                device=tokens.device,
            )
            all_patch_offsets = torch.arange(patch_count, device=tokens.device, dtype=torch.long)
            if getattr(self, "frame_fusion_mode", "pair-top-percent") == "sequential-group-average":
                relation_offset = 0
                for group in plan.groups:
                    member_frames = torch.tensor(
                        group.members,
                        device=tokens.device,
                        dtype=torch.long,
                    )
                    shared_offsets = all_patch_offsets[
                        ~target_keep_mask[relation_offset]
                    ]
                    relation_offset += len(group.members) - 1
                    if shared_offsets.numel() == 0:
                        continue
                    token_offsets = patch_start + shared_offsets
                    shared_tokens = output[batch_index].index_select(0, member_frames)
                    shared_tokens = shared_tokens.index_select(1, token_offsets).mean(dim=0)
                    for frame in member_frames:
                        output[batch_index, frame, token_offsets] = shared_tokens
                continue
            for pair_index, (source_frame, target_frame) in enumerate(zip(source_frames, target_frames)):
                fuse_offsets = all_patch_offsets[~target_keep_mask[pair_index]]
                if fuse_offsets.numel() == 0:
                    continue
                token_offsets = patch_start + fuse_offsets
                if getattr(self, "frame_fusion_mode", "pair-top-percent") in {
                    "group-top-percent",
                    "sequential-group",
                }:
                    output[batch_index, target_frame, token_offsets] = output[
                        batch_index, source_frame, token_offsets
                    ]
                else:
                    source_patch_tokens = output[batch_index, source_frame, token_offsets]
                    target_patch_tokens = output[batch_index, target_frame, token_offsets]
                    averaged_patch_tokens = (source_patch_tokens + target_patch_tokens) * 0.5
                    output[batch_index, source_frame, token_offsets] = averaged_patch_tokens
                    output[batch_index, target_frame, token_offsets] = averaged_patch_tokens
        return output

    @staticmethod
    def _copy_pair_patch_outputs(
        flat_tokens: torch.Tensor,
        plan: FrameFusionBatchPlan,
        *,
        tokens_per_frame: int,
        num_special_tokens: int,
    ) -> torch.Tensor:
        if not plan.pairs:
            return flat_tokens
        patch_count = tokens_per_frame - num_special_tokens
        if patch_count <= 0:
            return flat_tokens
        device = flat_tokens.device
        offsets = torch.arange(patch_count, device=device, dtype=torch.long)
        source_frames = plan.source_frames.to(device=device)
        target_frames = plan.target_frames.to(device=device)
        target_keep_mask = _frame_fusion_target_keep_patch_mask(
            plan.target_keep_patch_indices,
            num_pairs=int(source_frames.numel()),
            patch_count=patch_count,
            device=device,
        )
        source_index_chunks = []
        target_index_chunks = []
        for pair_index, (source_frame, target_frame) in enumerate(zip(source_frames, target_frames)):
            copy_offsets = offsets[~target_keep_mask[pair_index]]
            if copy_offsets.numel() == 0:
                continue
            source_index_chunks.append(source_frame * tokens_per_frame + num_special_tokens + copy_offsets)
            target_index_chunks.append(target_frame * tokens_per_frame + num_special_tokens + copy_offsets)
        if not source_index_chunks:
            return flat_tokens
        source_indices = torch.cat(source_index_chunks, dim=0)
        target_indices = torch.cat(target_index_chunks, dim=0)
        return flat_tokens.index_copy(0, target_indices, flat_tokens.index_select(0, source_indices))

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        self._maybe_store_adaptive_intra_scores(
            tokens=tokens,
            batch_size=batch_size,
            num_frames=num_frames,
            num_tokens=num_tokens,
            block_idx=block_idx,
            rope_sincos=rope_sincos,
        )
        self._maybe_store_register_patch_selection(
            tokens,
            batch_size,
            num_frames,
            num_tokens,
            block_idx,
            rope_sincos,
        )
        tokens = self.frame_blocks[block_idx](tokens, rope_sincos)
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _maybe_store_adaptive_intra_scores(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        if not self.use_adaptive_kv_anchor:
            return
        if self.adaptive_anchor_intra_source != "cached_frame_qk":
            return
        if block_idx not in self.adaptive_anchor_layers:
            return

        patch_count = num_tokens - self.patch_token_start
        if patch_count <= 0:
            return

        block = self.frame_blocks[block_idx]
        normalized = block.norm1(tokens)
        batch_frames, _, hidden = normalized.shape
        num_heads = block.attn.num_heads
        head_dim = hidden // num_heads
        qkv = block.attn.qkv(normalized).reshape(batch_frames, num_tokens, 3, num_heads, head_dim)
        q, k, _ = torch.unbind(qkv, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if block.attn.use_qk_norm:
            q = block.attn.q_norm(q)
            k = block.attn.k_norm(k)
        q, k = block.attn.apply_rope(q, k, rope_sincos)

        q_patch = q[:, :, self.patch_token_start :].float()
        k_patch = k[:, :, self.patch_token_start :].float()
        col_mass = torch.empty(batch_frames, patch_count, device=tokens.device, dtype=torch.float32)
        denom = max(batch_size * num_frames * num_heads * patch_count * patch_count, 1)
        frame_chunk = max(1, min(batch_frames, 8_000_000 // denom))
        for start in range(0, batch_frames, frame_chunk):
            end = min(start + frame_chunk, batch_frames)
            logits = torch.matmul(q_patch[start:end], k_patch[start:end].transpose(-2, -1)) * block.attn.scale
            intra = logits.softmax(dim=-1)
            col_mass[start:end] = intra.sum(dim=-2).mean(dim=1)
        self._adaptive_intra_scores[block_idx] = col_mass.view(batch_size, num_frames, patch_count)

    def _maybe_store_register_patch_selection(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        percent = self.register_patch_inter_frame_percent
        mode = self.register_patch_inter_frame_mode
        if mode == "none" or percent <= 0.0:
            return
        patch_count = num_tokens - self.patch_token_start
        if patch_count <= 0:
            return
        keep_count = max(1, min(patch_count, round(patch_count * percent / 100.0)))

        if mode == "random":
            generator = torch.Generator(device=tokens.device)
            generator.manual_seed(self.register_patch_inter_frame_seed + block_idx)
            scores = torch.rand(batch_size, num_frames, patch_count, device=tokens.device, generator=generator)
        elif mode == "least-register":
            scores = self._register_to_patch_attention_scores(tokens, block_idx, rope_sincos)
            scores = scores.view(batch_size, num_frames, patch_count)
            scores = -scores
        else:
            raise ValueError(f"Unknown register patch inter-frame mode: {mode}")

        selected = scores.topk(keep_count, dim=-1, largest=True, sorted=False).indices
        self._register_patch_selection[block_idx] = selected

    def _register_to_patch_attention_scores(
        self,
        tokens: torch.Tensor,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        block = self.frame_blocks[block_idx]
        normalized = block.norm1(tokens)
        batch_frames, num_tokens, hidden = normalized.shape
        num_heads = block.attn.num_heads
        head_dim = hidden // num_heads
        qkv = block.attn.qkv(normalized).reshape(batch_frames, num_tokens, 3, num_heads, head_dim)
        q, k, _ = torch.unbind(qkv, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if block.attn.use_qk_norm:
            q = block.attn.q_norm(q)
            k = block.attn.k_norm(k)
        q, k = block.attn.apply_rope(q, k, rope_sincos)

        patch_token_start = self.patch_token_start
        register_q = q[:, :, 1:patch_token_start].float()
        logits = torch.matmul(register_q, k.float().transpose(-2, -1)) * block.attn.scale
        probabilities = logits.softmax(dim=-1)
        patch_scores = probabilities[..., patch_token_start:].mean(dim=(1, 2))
        return patch_scores.to(dtype=tokens.dtype)

    def _run_adaptive_pair_scope_attention_block(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        patch_grid_size: tuple[int, int],
    ) -> torch.Tensor:
        """Run complete two-level routing inside one global attention layer."""

        adaptive = (
            self.progressive_attention_config.adaptive_pair_scope_config
        )
        if adaptive is None:
            raise RuntimeError(
                "adaptive pair-scope layer lacks adaptive configuration"
            )
        flat_tokens = tokens.reshape(
            batch_size,
            num_frames * num_tokens,
            embed_dim,
        )
        result = adaptive_pair_scope_attention_block(
            self.inter_frame_blocks[block_idx],
            flat_tokens,
            num_frames=num_frames,
            tokens_per_frame=num_tokens,
            num_special_tokens=self.patch_token_start,
            patch_grid_size=patch_grid_size,
            layer_index=block_idx,
            config=adaptive,
        )
        full_patch_tokens = num_frames * (
            num_tokens - self.patch_token_start
        )
        logical_patch_pairs = int(
            result.stats.get("final_logical_patch_pairs", 0)
        )
        result.stats["patch_attention_pair_retention_vs_full"] = (
            logical_patch_pairs
            / max(full_patch_tokens * full_patch_tokens, 1)
        )
        self.last_progressive_attention_stats[block_idx] = result.stats
        if result.debug:
            self.last_adaptive_pair_scope_debug[block_idx] = result.debug
        return result.output.view(
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
        )

    def _run_progressive_attention_block(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        patch_grid_size: tuple[int, int],
    ) -> torch.Tensor:
        config = self.progressive_attention_config
        spec = self.progressive_layer_schedule[block_idx]
        if spec.is_stage_first:
            self._progressive_stage_states.pop(spec.stage_index, None)
        previous_state = (
            self._progressive_stage_states.get(spec.stage_index)
            if config.mask_enabled
            else None
        )
        flat_tokens = tokens.reshape(
            batch_size,
            num_frames * num_tokens,
            embed_dim,
        )
        if spec.scope == "full" and config.final_scope_mode == "dense":
            flat_tokens = self.inter_frame_blocks[block_idx](flat_tokens, None)
            full_patch_tokens = num_frames * (
                num_tokens - self.patch_token_start
            )
            total_tokens = num_frames * num_tokens
            self.last_progressive_attention_stats[block_idx] = {
                "stage_index": spec.stage_index,
                "stage_name": spec.stage_name,
                "layer_index": block_idx,
                "stage_global_position": spec.global_position,
                "stage_global_count": spec.global_count,
                "scope": "full",
                "sampled_patch_tokens": full_patch_tokens,
                "special_tokens": num_frames * self.patch_token_start,
                "qkv_projection_tokens": total_tokens,
                "full_patch_tokens": full_patch_tokens,
                "patch_sampling_ratio": 1.0,
                "sample_before_qkv": True,
                "mask_inherited": False,
                "mask_generated": False,
                "attention_backend": "original_dense_sdpa",
                "logical_attention_pairs_per_batch": (
                    total_tokens * total_tokens
                ),
                "evaluated_attention_pairs_per_batch": (
                    total_tokens * total_tokens
                ),
                "patch_attention_pairs_per_batch": (
                    full_patch_tokens * full_patch_tokens
                ),
                "patch_attention_pair_retention_vs_full": 1.0,
            }
            self._progressive_stage_states.pop(spec.stage_index, None)
            return flat_tokens.view(
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
            )

        if (
            spec.scope == "full"
            and config.final_scope_mode == "inherited_sparse"
            and previous_state is None
        ):
            raise RuntimeError(
                f"Progressive full sparse layer {block_idx} has no inherited mask"
            )
        result = progressive_attention_block(
            self.inter_frame_blocks[block_idx],
            flat_tokens,
            num_frames=num_frames,
            tokens_per_frame=num_tokens,
            num_special_tokens=self.patch_token_start,
            patch_grid_size=patch_grid_size,
            layer_spec=spec,
            config=config,
            previous_state=previous_state,
            build_next_mask=bool(
                config.mask_enabled and not spec.is_stage_last
            ),
        )
        if spec.is_stage_last:
            self._progressive_stage_states.pop(spec.stage_index, None)
        elif result.next_state is not None:
            self._progressive_stage_states[spec.stage_index] = result.next_state
        full_patch_tokens = num_frames * (
            num_tokens - self.patch_token_start
        )
        patch_pairs = int(
            result.stats.get("patch_attention_pairs_per_batch", 0)
        )
        result.stats["patch_attention_pair_retention_vs_full"] = (
            patch_pairs / max(full_patch_tokens * full_patch_tokens, 1)
        )
        self.last_progressive_attention_stats[block_idx] = result.stats
        if result.sample_coordinates is not None:
            self.last_progressive_sample_indices[block_idx] = (
                result.sample_coordinates
            )
        return result.output.view(
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
        )

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
        patch_grid_size: tuple[int, int],
        frame_fusion_pair_plans: list[FrameFusionBatchPlan] | None = None,
        temporal_representative_plans: list[TemporalRepresentativeBatchPlan] | None = None,
        spatial_representative_plans: list[SpatialRepresentativeBatchPlan] | None = None,
        spatiotemporal_representative_plans: list[TemporalRepresentativeBatchPlan] | None = None,
        fastvggt_enabled: bool = True,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            progressive_config = self.progressive_attention_config
            if (
                progressive_config.enabled
                and progressive_config.algorithm
                == "adaptive_pair_scope"
            ):
                adaptive = progressive_config.adaptive_pair_scope_config
                assert adaptive is not None
                if block_idx in adaptive.enabled_layers:
                    return self._run_adaptive_pair_scope_attention_block(
                        tokens,
                        batch_size=batch_size,
                        num_frames=num_frames,
                        num_tokens=num_tokens,
                        embed_dim=embed_dim,
                        block_idx=block_idx,
                        patch_grid_size=patch_grid_size,
                    )
            progressive_spec = self.progressive_layer_schedule.get(block_idx)
            if (
                progressive_config.enabled
                and progressive_config.algorithm == "legacy_token_scope"
                and progressive_spec is not None
            ):
                return self._run_progressive_attention_block(
                    tokens,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    num_tokens=num_tokens,
                    embed_dim=embed_dim,
                    block_idx=block_idx,
                    patch_grid_size=patch_grid_size,
                )
            if self.frame_fusion_mode in {
                "pair-top-percent",
                "group-top-percent",
                "sequential-group",
                "sequential-group-average",
            } and frame_fusion_pair_plans is not None:
                return self._run_pair_fused_global_attention_block(
                    tokens,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    num_tokens=num_tokens,
                    embed_dim=embed_dim,
                    block_idx=block_idx,
                    frame_fusion_pair_plans=frame_fusion_pair_plans,
                )
            if (
                self.frame_fusion_mode == "temporal-representative"
                and temporal_representative_plans is not None
            ):
                return self._run_temporal_representative_global_attention_block(
                    tokens,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    num_tokens=num_tokens,
                    embed_dim=embed_dim,
                    block_idx=block_idx,
                    plans=temporal_representative_plans,
                )
            if (
                self.frame_fusion_mode == "adaptive-temporal-representative"
                and temporal_representative_plans is not None
            ):
                return self._run_temporal_representative_global_attention_block(
                    tokens,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    num_tokens=num_tokens,
                    embed_dim=embed_dim,
                    block_idx=block_idx,
                    plans=temporal_representative_plans,
                )
            if (
                self.frame_fusion_mode == "adaptive-spatial-representative"
                and spatial_representative_plans is not None
            ):
                return self._run_adaptive_spatial_representative_global_attention_block(
                    tokens,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    num_tokens=num_tokens,
                    embed_dim=embed_dim,
                    block_idx=block_idx,
                    plans=spatial_representative_plans,
                )
            if (
                self.frame_fusion_mode in {"h-m", "h-r", "selftr", "u-r"}
                and spatiotemporal_representative_plans is not None
            ):
                return self._run_temporal_representative_global_attention_block(
                    tokens,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    num_tokens=num_tokens,
                    embed_dim=embed_dim,
                    block_idx=block_idx,
                    plans=spatiotemporal_representative_plans,
                )
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            self.inter_frame_blocks[block_idx].attn.merge_random_seed = self.merge_random_seed
            self.inter_frame_blocks[block_idx].attn.precomputed_intra_scores = self._adaptive_intra_scores.get(block_idx)
            inter_frame_only_attention = block_idx in self.inter_frame_only_layers
            adaptive_kv_anchor = self.use_adaptive_kv_anchor and block_idx in self.adaptive_anchor_layers
            if adaptive_kv_anchor:
                if inter_frame_only_attention:
                    raise ValueError(
                        "Adaptive K/V anchors and inter-frame-only attention are mutually exclusive; "
                        f"layer {block_idx}"
                    )
                if self.sparse_attention:
                    raise ValueError("Adaptive K/V anchors and sparse attention are mutually exclusive")
                if self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio):
                    raise ValueError(
                        "Adaptive K/V anchors and token merging are mutually exclusive; "
                        "disable global_merging, set merging=None, or use merge_ratio=0.0"
                    )
            global_merging = (
                block_idx
                if self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio)
                and fastvggt_enabled
                and block_idx >= self.merging
                and not inter_frame_only_attention
                and not adaptive_kv_anchor
                else None
            )
            tokens = self.inter_frame_blocks[block_idx](
                tokens,
                None,
                global_merging=global_merging,
                patch_grid_size=patch_grid_size,
                num_special_tokens=self.patch_token_start,
                sparse_attention=self.sparse_attention and not inter_frame_only_attention,
                sparse_ratio=self.sparse_ratio,
                sparse_cdf_threshold=self.sparse_cdf_threshold,
                sparse_pool_mode=self.sparse_pool_mode,
                inter_frame_only_attention=inter_frame_only_attention,
                use_adaptive_kv_anchor=adaptive_kv_anchor,
                adaptive_anchor_ratio=self.adaptive_anchor_ratio,
                adaptive_anchor_total=self.adaptive_anchor_total,
                adaptive_anchor_min_per_frame=self.adaptive_anchor_min_per_frame,
                adaptive_anchor_tau=self.adaptive_anchor_tau,
                adaptive_anchor_uniform_mix=self.adaptive_anchor_uniform_mix,
                adaptive_anchor_strategy=self.adaptive_anchor_strategy,
                adaptive_anchor_score_alpha_cross=self.adaptive_anchor_score_alpha_cross,
                adaptive_anchor_score_beta_intra=self.adaptive_anchor_score_beta_intra,
                adaptive_anchor_score_mode=self.adaptive_anchor_score_mode,
                adaptive_anchor_proxy_quota_ratio=self.adaptive_anchor_proxy_quota_ratio,
                adaptive_anchor_intra_source=self.adaptive_anchor_intra_source,
                adaptive_anchor_frame_budget_mode=self.adaptive_anchor_frame_budget_mode,
                adaptive_anchor_frame_budget_top_frac=self.adaptive_anchor_frame_budget_top_frac,
                adaptive_anchor_frame_budget_lambda_intra=self.adaptive_anchor_frame_budget_lambda_intra,
                adaptive_anchor_frame_budget_lambda_reg=self.adaptive_anchor_frame_budget_lambda_reg,
                adaptive_anchor_frame_budget_reg_topm=self.adaptive_anchor_frame_budget_reg_topm,
                adaptive_anchor_reg_patch_topk_ratio=self.adaptive_anchor_reg_patch_topk_ratio,
                adaptive_anchor_reg_patch_topk_min=self.adaptive_anchor_reg_patch_topk_min,
                adaptive_anchor_reg_patch_topk_max=self.adaptive_anchor_reg_patch_topk_max,
                adaptive_anchor_reg_patch_conf_power=self.adaptive_anchor_reg_patch_conf_power,
                adaptive_anchor_reg_patch_min_conf=self.adaptive_anchor_reg_patch_min_conf,
                adaptive_anchor_query_conditioned_eta=self.adaptive_anchor_query_conditioned_eta,
                adaptive_anchor_gated_anchor_ratio_per_key_frame=self.adaptive_anchor_gated_anchor_ratio_per_key_frame,
                adaptive_anchor_gated_min_per_key_frame=self.adaptive_anchor_gated_min_per_key_frame,
                adaptive_anchor_gated_max_per_key_frame=self.adaptive_anchor_gated_max_per_key_frame,
                adaptive_anchor_always_include_self_frame=self.adaptive_anchor_always_include_self_frame,
                adaptive_anchor_profile=self.adaptive_anchor_profile,
                adaptive_anchor_topm_frames=self.adaptive_anchor_topm_frames,
                adaptive_anchor_random_seed=self.adaptive_anchor_random_seed,
                adaptive_anchor_debug=self.adaptive_anchor_debug,
            )
            if global_merging is not None:
                attention = self.inter_frame_blocks[block_idx].attn
                input_tokens = int(attention.last_merge_input_tokens)
                output_tokens = int(attention.last_merge_output_tokens)
                self._fastvggt_merge_debug_layers.append(
                    {
                        "layer": int(block_idx),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "merged_tokens": int(attention.last_merged_tokens),
                        "retention": output_tokens / max(input_tokens, 1),
                        "merge_applied": bool(attention.last_merge_applied),
                    }
                )
            self.inter_frame_blocks[block_idx].attn.precomputed_intra_scores = None
            if adaptive_kv_anchor:
                self._maybe_save_adaptive_anchor_debug(block_idx)
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:]
        selected_indices = self._register_patch_selection.pop(block_idx, None)

        if selected_indices is None:
            camera_and_register_tokens = self.inter_frame_blocks[block_idx](camera_and_register_tokens, None)
            return torch.cat(
                [
                    camera_and_register_tokens.view(batch_size, num_frames, patch_token_start, embed_dim),
                    patch_tokens,
                ],
                dim=2,
            )

        selected_indices = selected_indices.to(device=tokens.device)
        keep_count = selected_indices.shape[-1]
        gather_indices = selected_indices.unsqueeze(-1).expand(batch_size, num_frames, keep_count, embed_dim)
        selected_patch_tokens = patch_tokens.gather(dim=2, index=gather_indices)
        inter_frame_tokens = torch.cat(
            [
                camera_and_register_tokens,
                selected_patch_tokens.reshape(batch_size, num_frames * keep_count, embed_dim),
            ],
            dim=1,
        )
        inter_frame_tokens = self.inter_frame_blocks[block_idx](inter_frame_tokens, None)

        camera_and_register_tokens = inter_frame_tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        selected_patch_tokens = inter_frame_tokens[:, num_frames * patch_token_start :].view(
            batch_size, num_frames, keep_count, embed_dim
        )
        patch_tokens = patch_tokens.scatter(dim=2, index=gather_indices, src=selected_patch_tokens)
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2)

    def _maybe_save_adaptive_anchor_debug(self, block_idx: int) -> None:
        if not self.adaptive_anchor_debug:
            return
        debug_payload = self.inter_frame_blocks[block_idx].attn.last_adaptive_anchor_debug
        if debug_payload is None:
            return
        debug_dir = Path(self.adaptive_anchor_debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"step_{self._adaptive_anchor_debug_step:06d}_"
            f"layer_{block_idx:02d}_{self.adaptive_anchor_strategy}.pt"
        )
        torch.save(debug_payload, debug_dir / filename)
        self._adaptive_anchor_debug_step += 1


def _build_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(
    token_tensor: torch.Tensor,
    batch_size: int,
    num_frames: int,
    first_frame_token_indices: tuple[int, ...] = (0,),
) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, num_frames, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:2].expand(batch_size, num_frames, *token_tensor.shape[2:])
    mask = torch.zeros(num_frames, device=token_tensor.device, dtype=torch.bool)
    mask[list(first_frame_token_indices)] = True
    view_shape = (1, num_frames) + (1,) * (token_tensor.ndim - 2)
    tokens = torch.where(mask.view(view_shape), first_frame_token, other_frame_tokens)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])


def _normalize_similarity_weights(similarity: torch.Tensor) -> torch.Tensor:
    weights = similarity.clamp_min(0.0)
    total = weights.sum()
    if float(total.detach().cpu()) <= 1e-12:
        return torch.full_like(weights, 1.0 / max(weights.numel(), 1))
    return weights / total


def pooled_frame_representations(
    patch_tokens: torch.Tensor,
    *,
    patch_grid_size: tuple[int, int],
    pool_size: int = 2,
) -> torch.Tensor:
    if patch_tokens.ndim != 4:
        raise ValueError(
            "patch_tokens must have shape [batch, frames, patches, channels], "
            f"got {tuple(patch_tokens.shape)}"
        )
    batch_size, num_frames, patch_count, embed_dim = patch_tokens.shape
    patch_h, patch_w = patch_grid_size
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"patch_grid_size must be positive, got {patch_grid_size}")
    if patch_count != patch_h * patch_w:
        raise ValueError(
            "patch token count does not match patch_grid_size: "
            f"{patch_count} != {patch_h} * {patch_w}"
        )
    pool_size = int(pool_size)
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    kernel_h = min(pool_size, patch_h)
    kernel_w = min(pool_size, patch_w)
    patches = patch_tokens.float().view(
        batch_size,
        num_frames,
        patch_h,
        patch_w,
        embed_dim,
    )
    patches = patches.permute(0, 1, 4, 2, 3).reshape(
        batch_size * num_frames,
        embed_dim,
        patch_h,
        patch_w,
    )
    pooled = torch.nn.functional.avg_pool2d(
        patches,
        kernel_size=(kernel_h, kernel_w),
        stride=(kernel_h, kernel_w),
    )
    return pooled.flatten(1).view(batch_size, num_frames, -1)


def select_frame_fusion_pairs(
    similarity: torch.Tensor,
    *,
    pair_percent: float,
    exclude_frames: tuple[int, ...] | list[int] = (),
    disjoint: bool = True,
) -> tuple[list[FrameFusionPair], int, int]:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(f"similarity must be a square matrix, got shape {tuple(similarity.shape)}")
    num_frames = int(similarity.shape[0])
    excluded = _validate_frame_pair_selection_inputs(
        num_frames,
        pair_percent=pair_percent,
        exclude_frames=exclude_frames,
    )
    if num_frames - len(excluded) < 2:
        return [], 0, 0

    sim = similarity.detach().float().cpu().clone()
    sim.fill_diagonal_(float("-inf"))
    if excluded:
        excluded_index = torch.tensor(sorted(excluded), dtype=torch.long)
        sim[excluded_index, :] = float("-inf")
        sim[:, excluded_index] = float("-inf")
    candidates_by_pair: dict[tuple[int, int], float] = {}
    eligible_frames = [frame for frame in range(num_frames) if frame not in excluded]
    for frame_index in eligible_frames:
        neighbor = int(torch.argmax(sim[frame_index]).item())
        frame_a, frame_b = sorted((frame_index, neighbor))
        score = float(sim[frame_index, neighbor].item())
        previous = candidates_by_pair.get((frame_a, frame_b))
        if previous is None or score > previous:
            candidates_by_pair[(frame_a, frame_b)] = score
    candidates = [
        FrameFusionPair(frame_a=frame_a, frame_b=frame_b, similarity=score)
        for (frame_a, frame_b), score in candidates_by_pair.items()
    ]
    return _select_top_percent_frame_pairs(
        candidates,
        pair_percent=pair_percent,
        disjoint=disjoint,
    )


def select_frame_fusion_pairs_from_normalized_representations(
    normalized_frame_representations: torch.Tensor,
    *,
    pair_percent: float,
    exclude_frames: tuple[int, ...] | list[int] = (),
    disjoint: bool = True,
) -> tuple[list[FrameFusionPair], int, int]:
    if normalized_frame_representations.ndim != 2:
        raise ValueError(
            "normalized_frame_representations must have shape [frames, channels], "
            f"got {tuple(normalized_frame_representations.shape)}"
        )
    num_frames = int(normalized_frame_representations.shape[0])
    excluded = _validate_frame_pair_selection_inputs(
        num_frames,
        pair_percent=pair_percent,
        exclude_frames=exclude_frames,
    )
    if num_frames - len(excluded) < 2:
        return [], 0, 0

    reps = normalized_frame_representations.detach().float()
    candidates_by_pair: dict[tuple[int, int], float] = {}
    eligible_frames = [frame for frame in range(num_frames) if frame not in excluded]
    excluded_index = (
        torch.tensor(sorted(excluded), device=reps.device, dtype=torch.long)
        if excluded
        else None
    )
    for frame_index in eligible_frames:
        scores = torch.matmul(reps, reps[frame_index]).clamp(-1.0, 1.0)
        scores[frame_index] = float("-inf")
        if excluded_index is not None:
            scores[excluded_index] = float("-inf")
        neighbor = int(torch.argmax(scores).item())
        frame_a, frame_b = sorted((frame_index, neighbor))
        score = float(scores[neighbor].detach().cpu().item())
        previous = candidates_by_pair.get((frame_a, frame_b))
        if previous is None or score > previous:
            candidates_by_pair[(frame_a, frame_b)] = score
    candidates = [
        FrameFusionPair(frame_a=frame_a, frame_b=frame_b, similarity=score)
        for (frame_a, frame_b), score in candidates_by_pair.items()
    ]
    return _select_top_percent_frame_pairs(
        candidates,
        pair_percent=pair_percent,
        disjoint=disjoint,
    )


def _validate_frame_pair_selection_inputs(
    num_frames: int,
    *,
    pair_percent: float,
    exclude_frames: tuple[int, ...] | list[int],
) -> set[int]:
    pair_percent = float(pair_percent)
    if not 0.0 < pair_percent <= 100.0:
        raise ValueError(f"pair_percent must be in (0, 100], got {pair_percent}")
    excluded = {int(frame) for frame in exclude_frames}
    invalid_excluded = sorted(frame for frame in excluded if frame < 0 or frame >= num_frames)
    if invalid_excluded:
        raise ValueError(f"exclude_frames contains out-of-range indices: {invalid_excluded}")
    return excluded


def _select_top_percent_frame_pairs(
    candidates: list[FrameFusionPair],
    *,
    pair_percent: float,
    disjoint: bool,
) -> tuple[list[FrameFusionPair], int, int]:
    candidates.sort(key=lambda pair: pair.similarity, reverse=True)
    unique_candidate_count = len(candidates)
    if unique_candidate_count == 0:
        return [], 0, 0
    requested_pair_count = int(torch.ceil(torch.tensor(unique_candidate_count * pair_percent / 100.0)).item())
    requested_pair_count = min(unique_candidate_count, max(requested_pair_count, 1))

    selected: list[FrameFusionPair] = []
    if not disjoint:
        return candidates[:requested_pair_count], unique_candidate_count, requested_pair_count
    used_frames: set[int] = set()
    for pair in candidates:
        if pair.frame_a in used_frames or pair.frame_b in used_frames:
            continue
        selected.append(pair)
        used_frames.add(pair.frame_a)
        used_frames.add(pair.frame_b)
        if len(selected) >= requested_pair_count:
            break
    return selected, unique_candidate_count, requested_pair_count


def _connected_frame_fusion_groups(
    pairs: list[FrameFusionPair],
) -> list[FrameFusionGroup]:
    """Convert overlapping selected edges into sorted connected components."""

    adjacency: dict[int, set[int]] = {}
    for pair in pairs:
        adjacency.setdefault(pair.frame_a, set()).add(pair.frame_b)
        adjacency.setdefault(pair.frame_b, set()).add(pair.frame_a)

    groups: list[FrameFusionGroup] = []
    visited: set[int] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack = [start]
        members: list[int] = []
        visited.add(start)
        while stack:
            frame = stack.pop()
            members.append(frame)
            for neighbor in sorted(adjacency[frame], reverse=True):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        ordered_members = tuple(sorted(members))
        groups.append(FrameFusionGroup(anchor=ordered_members[0], members=ordered_members))
    return groups


def _sequential_frame_fusion_groups(
    normalized_representations: torch.Tensor,
    *,
    similarity_threshold: float,
    max_group_size: int,
    first_frame: int = 1,
) -> list[FrameFusionGroup]:
    """Partition frames in input order using an all-members similarity gate."""

    if normalized_representations.ndim != 2:
        raise ValueError(
            "normalized_representations must have shape [frames, channels], "
            f"got {tuple(normalized_representations.shape)}"
        )
    num_frames = int(normalized_representations.shape[0])
    max_group_size = int(max_group_size)
    similarity_threshold = float(similarity_threshold)
    first_frame = int(first_frame)
    if max_group_size <= 0:
        raise ValueError(f"max_group_size must be positive, got {max_group_size}")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            f"similarity_threshold must be in [-1, 1], got {similarity_threshold}"
        )
    if not 0 <= first_frame <= num_frames:
        raise ValueError(f"first_frame must be in [0, {num_frames}], got {first_frame}")

    groups: list[FrameFusionGroup] = []
    if first_frame == num_frames:
        return groups
    current = [first_frame]
    for frame in range(first_frame + 1, num_frames):
        if len(current) >= max_group_size:
            groups.append(FrameFusionGroup(anchor=current[0], members=tuple(current)))
            current = [frame]
            continue
        member_indices = torch.tensor(current, device=normalized_representations.device)
        similarities = torch.matmul(
            normalized_representations.index_select(0, member_indices),
            normalized_representations[frame],
        )
        if bool(torch.all(similarities >= similarity_threshold).item()):
            current.append(frame)
        else:
            groups.append(FrameFusionGroup(anchor=current[0], members=tuple(current)))
            current = [frame]
    groups.append(FrameFusionGroup(anchor=current[0], members=tuple(current)))
    return groups


def _anchor_target_frame_fusion_pairs(
    groups: tuple[FrameFusionGroup, ...] | list[FrameFusionGroup],
    normalized_representations: torch.Tensor,
) -> list[FrameFusionPair]:
    return [
        FrameFusionPair(
            frame_a=group.anchor,
            frame_b=frame,
            similarity=float(
                torch.dot(
                    normalized_representations[group.anchor],
                    normalized_representations[frame],
                ).clamp(-1.0, 1.0).item()
            ),
        )
        for group in groups
        for frame in group.members
        if frame != group.anchor
    ]


def _frame_fusion_partition_summary(
    pairs: list[FrameFusionPair],
    groups: list[FrameFusionGroup],
) -> dict[str, object]:
    sizes = [len(group.members) for group in groups]
    participating = sorted({frame for group in groups for frame in group.members})
    candidate_membership: dict[int, int] = {}
    for pair in pairs:
        candidate_membership[pair.frame_a] = candidate_membership.get(pair.frame_a, 0) + 1
        candidate_membership[pair.frame_b] = candidate_membership.get(pair.frame_b, 0) + 1
    overlapping_frames = sorted(
        frame
        for frame, count in _frame_fusion_frame_membership_counts(groups).items()
        if count > 1
    )
    return {
        "candidate_edges_used": len(pairs),
        "groups": len(groups),
        "group_sizes": sizes,
        "group_size_histogram": {
            str(size): sizes.count(size) for size in sorted(set(sizes))
        },
        "participating_frames": len(participating),
        "participating_frame_indices": participating,
        "overlapping_frames": overlapping_frames,
        "candidate_frames_with_multiple_edges": sorted(
            frame for frame, count in candidate_membership.items() if count > 1
        ),
        "candidate_max_edge_degree": max(candidate_membership.values(), default=0),
    }


def _frame_fusion_frame_membership_counts(
    groups: list[FrameFusionGroup],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for group in groups:
        for frame in group.members:
            counts[frame] = counts.get(frame, 0) + 1
    return counts


def _frame_fusion_target_keep_patch_mask(
    target_keep_patch_indices: torch.Tensor | None,
    *,
    num_pairs: int,
    patch_count: int,
    device: torch.device,
) -> torch.Tensor:
    keep_mask = torch.zeros(
        num_pairs,
        patch_count,
        device=device,
        dtype=torch.bool,
    )
    if target_keep_patch_indices is None or target_keep_patch_indices.numel() == 0:
        return keep_mask
    if target_keep_patch_indices.dtype == torch.bool:
        if target_keep_patch_indices.ndim != 2:
            raise ValueError(
                "boolean target_keep_patch_indices must be [num_pairs, patch_count], "
                f"got {tuple(target_keep_patch_indices.shape)}"
            )
        if tuple(target_keep_patch_indices.shape) != (num_pairs, patch_count):
            raise ValueError(
                "boolean target_keep_patch_indices shape must match "
                f"({num_pairs}, {patch_count}), got {tuple(target_keep_patch_indices.shape)}"
            )
        return target_keep_patch_indices.to(device=device, dtype=torch.bool)

    keep_indices = target_keep_patch_indices.to(device=device, dtype=torch.long)
    if keep_indices.ndim != 2:
        raise ValueError(
            "target_keep_patch_indices must be [num_pairs, keep_count], "
            f"got {tuple(keep_indices.shape)}"
        )
    if keep_indices.shape[0] != num_pairs:
        raise ValueError(
            "target_keep_patch_indices first dimension must match selected pairs, "
            f"got {keep_indices.shape[0]} and {num_pairs}"
        )
    if keep_indices.shape[1] == 0:
        return keep_mask
    if int(keep_indices.min().item()) < 0 or int(keep_indices.max().item()) >= patch_count:
        raise ValueError("target keep patch index out of range")
    keep_mask.scatter_(1, keep_indices, True)
    return keep_mask


def _frame_fusion_target_keep_patch_counts(
    target_keep_patch_indices: torch.Tensor | None,
    *,
    num_pairs: int,
    patch_count: int,
    device: torch.device,
) -> torch.Tensor:
    return _frame_fusion_target_keep_patch_mask(
        target_keep_patch_indices,
        num_pairs=num_pairs,
        patch_count=patch_count,
        device=device,
    ).sum(dim=1)


def frame_fusion_attention_indices(
    *,
    num_frames: int,
    tokens_per_frame: int,
    num_special_tokens: int,
    source_frames: torch.Tensor,
    target_frames: torch.Tensor,
    target_keep_patch_indices: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if not 0 <= num_special_tokens <= tokens_per_frame:
        raise ValueError(
            "num_special_tokens must be in [0, tokens_per_frame], "
            f"got {num_special_tokens} for {tokens_per_frame}"
        )
    if source_frames.shape != target_frames.shape:
        raise ValueError(
            "source_frames and target_frames must have the same shape, "
            f"got {tuple(source_frames.shape)} and {tuple(target_frames.shape)}"
        )
    device = source_frames.device if device is None else device
    keep_mask = torch.zeros(
        num_frames,
        tokens_per_frame,
        device=device,
        dtype=torch.bool,
    )
    keep_mask[:, :num_special_tokens] = True
    keep_patch_frames = torch.ones(num_frames, device=device, dtype=torch.bool)
    if target_frames.numel() > 0:
        target_frames = target_frames.to(device=device, dtype=torch.long)
        if int(target_frames.min().item()) < 0 or int(target_frames.max().item()) >= num_frames:
            raise ValueError("target frame index out of range")
        keep_patch_frames[target_frames] = False
    keep_mask[keep_patch_frames, num_special_tokens:] = True
    if target_keep_patch_indices is not None and target_keep_patch_indices.numel() > 0:
        patch_count = tokens_per_frame - num_special_tokens
        target_keep_patch_mask = _frame_fusion_target_keep_patch_mask(
            target_keep_patch_indices,
            num_pairs=int(target_frames.numel()),
            patch_count=patch_count,
            device=device,
        )
        pair_offsets, patch_offsets = target_keep_patch_mask.nonzero(as_tuple=True)
        keep_mask[target_frames[pair_offsets], num_special_tokens + patch_offsets] = True
    return keep_mask.flatten().nonzero(as_tuple=False).flatten()


def compute_frame_fusion_partition(
    distance: torch.Tensor,
    *,
    num_groups: int,
    max_group_size: int,
    beta: float = 1.0,
) -> list[FrameFusionSegment]:
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError(f"distance must be a square matrix, got shape {tuple(distance.shape)}")
    num_frames = int(distance.shape[0])
    num_groups = int(num_groups)
    max_group_size = int(max_group_size)
    beta = float(beta)
    if num_frames <= 0:
        raise ValueError("distance matrix must contain at least one frame")
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    if max_group_size <= 0:
        raise ValueError(f"max_group_size must be positive, got {max_group_size}")
    if beta < 0.0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    if num_groups > num_frames:
        raise ValueError(f"num_groups ({num_groups}) must be <= num_frames ({num_frames})")
    if num_frames > num_groups * max_group_size:
        raise ValueError(
            "No feasible frame partition: "
            f"num_frames={num_frames}, num_groups={num_groups}, max_group_size={max_group_size}"
        )

    dist = distance.detach().float().cpu()
    costs = torch.full((num_frames, num_frames), float("inf"), dtype=torch.float64)
    medoids = torch.full((num_frames, num_frames), -1, dtype=torch.long)
    mean_distances = torch.full((num_frames, num_frames), float("nan"), dtype=torch.float64)
    max_distances = torch.full((num_frames, num_frames), float("nan"), dtype=torch.float64)
    for start in range(num_frames):
        for end in range(start, min(num_frames, start + max_group_size)):
            group = dist[start : end + 1, start : end + 1].double()
            medoid_local = int(torch.argmin(group.sum(dim=0)).item())
            medoid = start + medoid_local
            distances_to_medoid = dist[start : end + 1, medoid].double()
            mean_distance = distances_to_medoid.mean()
            max_distance = distances_to_medoid.max()
            cost = mean_distance + beta * max_distance
            costs[start, end] = cost
            medoids[start, end] = medoid
            mean_distances[start, end] = mean_distance
            max_distances[start, end] = max_distance

    dp = torch.full((num_groups + 1, num_frames + 1), float("inf"), dtype=torch.float64)
    back = torch.full((num_groups + 1, num_frames + 1), -1, dtype=torch.long)
    dp[0, 0] = 0.0
    for group_count in range(1, num_groups + 1):
        min_end = group_count
        max_end = min(num_frames, group_count * max_group_size)
        for end_exclusive in range(min_end, max_end + 1):
            start_min = max(group_count - 1, end_exclusive - max_group_size)
            start_max = end_exclusive - 1
            best_cost = float("inf")
            best_start = -1
            for start in range(start_min, start_max + 1):
                previous = float(dp[group_count - 1, start].item())
                if previous == float("inf"):
                    continue
                candidate = previous + float(costs[start, end_exclusive - 1].item())
                if candidate < best_cost:
                    best_cost = candidate
                    best_start = start
            if best_start >= 0:
                dp[group_count, end_exclusive] = best_cost
                back[group_count, end_exclusive] = best_start

    if not torch.isfinite(dp[num_groups, num_frames]):
        raise RuntimeError("No feasible frame partition found")

    segments: list[FrameFusionSegment] = []
    end_exclusive = num_frames
    for group_count in range(num_groups, 0, -1):
        start = int(back[group_count, end_exclusive].item())
        if start < 0:
            raise RuntimeError(f"Missing DP backpointer for group={group_count}, end={end_exclusive}")
        end = end_exclusive - 1
        segments.append(
            FrameFusionSegment(
                start=start,
                end=end,
                medoid=int(medoids[start, end].item()),
                cost=float(costs[start, end].item()),
                mean_distance=float(mean_distances[start, end].item()),
                max_distance=float(max_distances[start, end].item()),
            )
        )
        end_exclusive = start
    segments.reverse()
    return segments
