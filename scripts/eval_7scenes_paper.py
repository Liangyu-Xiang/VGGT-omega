#!/usr/bin/env python3
"""Evaluate SelfTR or its VGGT backbone on 7 Scenes.

The VGGT-Omega paper samples 10 frames per scene/sequence and reports pairwise
relative-pose AUC plus scale-aligned depth metrics.  The paper does not publish
the sampled frame IDs.  This evaluator defaults to first/last-preserving uniform
sampling for long-sequence experiments; pass ``--sampling-strategy random`` to
use the deterministic RandomState(42) convention from public VGGT loaders.

The dataset must contain RGB-registered ``*.depth.proj.png`` maps.  These can be
generated once by FastVGGT's ``scripts/prepare_7scenes.py``; this evaluator only
reads those files and is therefore compatible with both projects.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from selftr import SelfTR
from selftr.checkpoint import load_state_dict, require_selftr_checkpoint
from selftr.evaluation import (
    depth_error_metrics,
    depth_to_world_points,
    evaluate_pi3_geometry,
    scaled_intrinsics,
    trajectory_pose_metrics,
)
from selftr.identity import METHOD_ID, canonical_frame_fusion_mode, resolve_method_name
from vggt_omega.utils.frame_sampling import SAMPLING_STRATEGIES, uniform_first_last_indices
from vggt_omega.utils.gpu_guard import assert_exclusive_gpu
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.reference_frame import (
    resolve_first_frame_token_indices,
    reference_first_order,
    reorder_reference_first,
    resolve_reference_frame_index,
)


DEFAULT_DATA_ROOT = Path(os.environ.get("SELFTR_7SCENES_ROOT", "data/7scenes"))
DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")
DEFAULT_REGISTER_ONLY_GLOBAL_LAYER_SPEC = "9-23"
PAPER_TARGETS = {
    "auc_3_percent": 29.6,
    "auc_30_percent": 83.1,
    "delta_1_25_percent": 94.6,
    "abs_rel": 0.058,
}
DATASET_NAME = "7Scenes"
DATASET_SPLIT = "official TestSplit.txt files"


@dataclass(frozen=True)
class FrameRecord:
    index: int
    rgb_path: Path
    depth_path: Path
    pose_path: Path
    c2w: np.ndarray


def process_token_retention_percent(layer_stats: list[dict], global_layer_count: int = 24) -> float | None:
    """Span-weight refresh retentions over all global layers."""
    points = sorted(
        (int(row["source_layer"]), float(row["patch_token_retention_percent"]))
        for row in layer_stats
        if row.get("source_layer") is not None
    )
    if not points:
        return None
    weighted = 0.0
    for index, (layer, retention) in enumerate(points):
        stop = points[index + 1][0] if index + 1 < len(points) else global_layer_count
        weighted += retention * max(stop - layer, 0)
    return weighted / global_layer_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce SelfTR and VGGT-backbone metrics on 7 Scenes."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/7scenes_paper"))
    parser.add_argument(
        "--acceleration-method",
        choices=("none", "da-vggt", "sparse-vggt", "fastvggt", "selftr", "u-m"),
        default="none",
        help="Unified interface label; DA-VGGT enables anchor-chunk inference.",
    )
    parser.add_argument(
        "--method-name",
        default=None,
        help="Display name stored in metrics.json (default: SELFTR_METHOD_NAME or SelfTR).",
    )
    parser.add_argument(
        "--da-chunk-size", type=int, default=50,
        help="Frames per anchor chunk for the DA-VGGT adapter.",
    )
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument(
        "--attention-mode",
        choices=("default", "register-only-zero-shot"),
        default="default",
        help=(
            "Attention schedule to evaluate. The register-only option changes the released "
            "checkpoint at inference time and is not a separately trained architecture."
        ),
    )
    parser.add_argument(
        "--register-only-global-layers",
        default="",
        help=(
            "Comma-separated layer indices or inclusive ranges to keep as global attention "
            "inside register-only-zero-shot mode, e.g. '9-23' or 'none'. "
            f"Default in register-only-zero-shot mode: {DEFAULT_REGISTER_ONLY_GLOBAL_LAYER_SPEC}. "
            "Indices are 0-based inter-frame block IDs."
        ),
    )
    parser.add_argument(
        "--inter-frame-only-global-layers",
        default="",
        help=(
            "Comma-separated 0-based global inter-frame layer indices or inclusive ranges "
            "where same-frame token attention is disabled, e.g. '15'."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument(
        "--sampling-stride", type=int, default=3,
        help=(
            "For a pool longer than --sampling-stride * --num-frames, select "
            "frames 0, stride, ... from its start. Otherwise preserve first/last "
            "with uniform sampling; pools shorter than --num-frames are skipped."
        ),
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=SAMPLING_STRATEGIES,
        default="uniform",
        help=(
            "Frame selection strategy. 'uniform' is the default and preserves the "
            "first/last pool frames while sampling the middle evenly; 'random' "
            "keeps the paper-style seeded protocol."
        ),
    )
    parser.add_argument("--sampling-pool-frames", type=int, default=0,
                        help="Uniformly form this many source candidates, then use their first --num-frames; short sequences use source first frames.")
    parser.add_argument(
        "--reference-frame-index",
        type=int,
        default=0,
        help=(
            "Index in the sampled frame list to move to model input position 0. "
            "Original VGGT-Omega uses position-0 camera/register tokens as the "
            "reference-frame distinction. Negative values follow Python indexing."
        ),
    )
    parser.add_argument(
        "--first-frame-token-indices",
        default="0",
        help=(
            "Input positions that receive the learned position-0 camera/register "
            "tokens. Use comma-separated indices, ranges, 'uniform:N', or 'all'. "
            "Position 0 must be included. Default '0' is original VGGT-Omega."
        ),
    )
    parser.add_argument(
        "--frame-fusion-mode",
        choices=(
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
            "u-m",  # Deprecated alias, normalized to "selftr" after parsing.
            "u-r",
        ),
        default="none",
        help="Enable frame fusion. Default keeps original VGGT-Omega behavior.",
    )
    parser.add_argument(
        "--frame-fusion-k",
        type=int,
        default=None,
        help="Number of frame groups for DP medoid fusion, e.g. 80 for 300 input frames.",
    )
    parser.add_argument(
        "--frame-fusion-max-group-size",
        type=int,
        default=5,
        help="Maximum number of original frames per frame-fusion group.",
    )
    parser.add_argument(
        "--frame-fusion-beta",
        type=float,
        default=1.0,
        help="Weight on max distance inside the DP medoid group cost.",
    )
    parser.add_argument(
        "--frame-fusion-start-layer",
        type=int,
        default=-1,
        help="-1 fuses before inter-frame block 0; non-negative values fuse after that block.",
    )
    parser.add_argument(
        "--frame-fusion-pair-percent",
        type=float,
        default=25.0,
        help=(
            "Top percent of nearest-neighbor frame-pair candidates. pair-top-percent "
            "uses disjoint pairs; group-top-percent joins overlapping candidates into groups; "
            "sequential-group builds groups in frame order; sequential-group-average "
            "uses group-mean shared tokens; temporal-representative builds a sequential "
            "per-position temporal representative dictionary."
        ),
    )
    parser.add_argument(
        "--frame-fusion-pool-size",
        type=int,
        default=2,
        help="Spatial average-pooling kernel/stride used to build frame representations for pair fusion.",
    )
    parser.add_argument(
        "--frame-fusion-group-similarity-threshold",
        type=float,
        default=0.0,
        help="Frame-level all-members threshold for sequential-group fusion.",
    )
    parser.add_argument(
        "--frame-fusion-target-keep-policy",
        choices=("none", "random-grid", "least-similar", "similarity-threshold"),
        default="none",
        help="Retain selected target-frame patch tokens instead of fusing the entire paired target frame.",
    )
    parser.add_argument(
        "--frame-fusion-target-keep-grid-size",
        type=int,
        default=4,
        help="Patch-grid block size for random-grid target patch retention.",
    )
    parser.add_argument(
        "--frame-fusion-target-keep-percent",
        type=float,
        default=0.0,
        help="Target patch percentage retained for least-similar retention.",
    )
    parser.add_argument(
        "--frame-fusion-target-keep-threshold",
        type=float,
        default=0.0,
        help="Cosine similarity threshold for similarity-threshold target patch retention.",
    )
    parser.add_argument(
        "--frame-fusion-target-keep-seed",
        type=int,
        default=33,
        help="Random seed for random-grid target patch retention.",
    )
    parser.add_argument(
        "--frame-fusion-recompute-each-global",
        action="store_true",
        help=(
            "Recompute the frame-fusion plan after every frame-attention block "
            "immediately before each global inter-frame attention block. "
            "Supported by pair-top-percent and SelfTR/H-M/U-R/H-R representative fusion."
        ),
    )
    parser.add_argument(
        "--frame-fusion-recompute-layers",
        default=None,
        help=(
            "Comma-separated global layer indices at which to rebuild the SelfTR/H-M "
            "representative plan after frame attention. SelfTR defaults to '0,10,17'; "
            "pass 'none' to disable refreshes."
        ),
    )
    parser.add_argument(
        "--frame-fusion-lambda-cost",
        type=float,
        default=0.04,
        help="Lambda for adaptive temporal representative objective D_tilde + lambda*q.",
    )
    parser.add_argument(
        "--frame-fusion-fixed-compression-ratio",
        type=float,
        default=None,
        help=(
            "For U-M, replace the delta_E < 2*lambda stopping rule with a fixed "
            "fraction of non-reference patch tokens to compress, in [0, 1)."
        ),
    )
    parser.add_argument(
        "--frame-fusion-merge-top-similarity-percent",
        type=float,
        default=100.0,
        help=(
            "Per-round percentage of mutual-nearest merge pairs retained by "
            "representative-token cosine similarity."
        ),
    )
    parser.add_argument(
        "--frame-fusion-layer-lambdas",
        default="",
        help=(
            "Optional per-recompute-layer SelfTR lambdas, as '0:0.1,10:0.1,17:0.1' "
            "or three comma-separated values in recompute-layer order."
        ),
    )
    parser.add_argument("--frame-fusion-min-keep-ratio", type=float, default=0.05)
    parser.add_argument("--frame-fusion-temporal-window", type=int, default=1)
    parser.add_argument("--frame-fusion-spatial-radius", type=int, default=1)
    parser.add_argument(
        "--frame-fusion-spatial-neighborhood",
        choices=("N4", "N8", "N8-R2"),
        default="N8",
    )
    parser.add_argument("--frame-fusion-time-overlap", type=float, default=0.5)
    parser.add_argument("--frame-fusion-reassignment-candidates", type=int, default=8)
    parser.add_argument(
        "--max-geometry-points",
        type=int,
        default=100_000,
        help="Maximum corresponding RGB-D points used for point-cloud metrics.",
    )
    parser.add_argument(
        "--frame-fusion-representative-update",
        choices=("parent", "exact-medoid"),
        default="parent",
    )
    parser.add_argument(
        "--frame-fusion-attention-variant",
        choices=(
            "representative",
            "representative-log",
            "kv-only",
            "replicated",
            "last-representative",
        ),
        default="representative",
        help=(
            "SelfTR attention variant. 'representative-log' restores log group-size "
            "weights; 'kv-only' keeps all queries and compresses "
            "only keys/values; 'replicated' copies representatives to a full "
            "length sequence; 'last-representative' uses the final representative "
            "source token selected for each U-M group instead of its mean."
        ),
    )
    parser.add_argument(
        "--sampling-unit",
        choices=("scene", "sequence"),
        default="sequence",
        help="Sampling unit (default matches common 7 Scenes evaluation loaders).",
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument(
        "--merge-ratio",
        type=float,
        default=0.0,
        help="Token merge ratio. The paper baseline is 0 (disabled).",
    )
    parser.add_argument(
        "--fastvggt-disable-protection",
        action="store_true",
        help="Disable FastVGGT's default deterministic 10%% protected-token subset.",
    )
    parser.add_argument(
        "--register-patch-inter-frame-mode",
        choices=("none", "random", "least-register"),
        default="none",
        help=(
            "For register-only attention, additionally let selected patch tokens participate "
            "in inter-frame attention. 'least-register' selects patches least attended by "
            "same-frame registers in the preceding frame-attention block."
        ),
    )
    parser.add_argument(
        "--register-patch-inter-frame-percent",
        type=float,
        default=0.0,
        help="Per-frame percentage of patch tokens selected for register-only inter-frame attention.",
    )
    parser.add_argument(
        "--depth-alignment",
        choices=("per-frame-median", "per-sequence-median"),
        default="per-frame-median",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.2,
        help="Minimum valid sensor depth in metres (filters wrapped invalid-depth sentinels).",
    )
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional scene/seq-NN names, e.g. chess/seq-03.",
    )
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Run one forward per sequence for metrics and omit CUDA-event latency measurements.",
    )
    parser.add_argument(
        "--require-exclusive-gpu",
        action="store_true",
        help="Abort if the target physical GPU is shared with other compute workloads.",
    )
    parser.add_argument(
        "--exclusive-gpu-index",
        type=int,
        default=None,
        help="Physical GPU index to guard when --require-exclusive-gpu is enabled.",
    )
    parser.add_argument(
        "--exclusive-gpu-max-other-memory-mib",
        type=int,
        default=512,
        help="Allowed non-target residual memory on the guarded physical GPU.",
    )
    parser.add_argument("--sparse-attention", action="store_true", help="Use block-sparse inter-frame global attention.")
    parser.add_argument(
        "--sparse-ratio",
        type=float,
        default=None,
        help="Sparse-VGGT ratio of key blocks to prune, e.g. 0.1.",
    )
    parser.add_argument(
        "--sparse-cdf-threshold",
        type=float,
        default=None,
        help="Sparse-VGGT cumulative attention threshold, e.g. 0.97.",
    )
    parser.add_argument("--sparse-pool-mode", choices=("avg", "max"), default="avg")
    parser.add_argument(
        "--use-adaptive-kv-anchor",
        "--use-register-mediated-anchor",
        "--use_register_mediated_anchor",
        action="store_true",
        help="Enable adaptive K/V anchor token selection in global inter-frame attention.",
    )
    parser.add_argument(
        "--adaptive-anchor-layers",
        "--anchor-layers",
        "--anchor_layers",
        dest="adaptive_anchor_layers",
        default="6-18",
        help="Layer spec passed to VGGTOmega for adaptive K/V anchors, e.g. '6-18'.",
    )
    parser.add_argument(
        "--adaptive-anchor-ratio",
        "--anchor-ratio",
        "--anchor_ratio",
        dest="adaptive_anchor_ratio",
        type=float,
        default=0.2,
        help="Per-frame adaptive anchor ratio passed to VGGTOmega.",
    )
    parser.add_argument(
        "--adaptive-anchor-total",
        "--anchor-total",
        "--anchor_total",
        dest="adaptive_anchor_total",
        type=int,
        default=None,
        help="Optional total adaptive anchor budget passed to VGGTOmega.",
    )
    parser.add_argument(
        "--adaptive-anchor-min-per-frame",
        "--anchor-min-per-frame",
        "--anchor_min_per_frame",
        dest="adaptive_anchor_min_per_frame",
        type=int,
        default=4,
        help="Minimum adaptive anchors retained per frame.",
    )
    parser.add_argument(
        "--adaptive-anchor-tau",
        "--anchor-tau",
        "--anchor_tau",
        dest="adaptive_anchor_tau",
        type=float,
        default=0.5,
        help="Temperature for adaptive anchor score weighting.",
    )
    parser.add_argument(
        "--adaptive-anchor-uniform-mix",
        "--anchor-uniform-mix",
        "--anchor_uniform_mix",
        dest="adaptive_anchor_uniform_mix",
        type=float,
        default=0.05,
        help="Uniform mixing coefficient for adaptive anchor allocation.",
    )
    parser.add_argument(
        "--adaptive-anchor-mode",
        "--adaptive-anchor-strategy",
        "--anchor-mode",
        "--anchor-strategy",
        "--anchor_mode",
        "--anchor_strategy",
        dest="adaptive_anchor_strategy",
        choices=(
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
        ),
        default="register_gated_intra",
        help="Adaptive K/V anchor mode. register-mediated modes are lifting, frame_pair_gated, and hybrid.",
    )
    parser.add_argument(
        "--adaptive-anchor-score-alpha-cross",
        "--anchor-score-alpha-cross",
        "--anchor_score_alpha_cross",
        dest="adaptive_anchor_score_alpha_cross",
        type=float,
        default=1.0,
        help="Weight for register-mediated cross-view patch score.",
    )
    parser.add_argument(
        "--adaptive-anchor-score-beta-intra",
        "--anchor-score-beta-intra",
        "--anchor_score_beta_intra",
        dest="adaptive_anchor_score_beta_intra",
        type=float,
        default=0.2,
        help="Weight for intra-frame structural patch score.",
    )
    parser.add_argument(
        "--adaptive-anchor-score-mode",
        "--anchor-score-mode",
        "--anchor_score_mode",
        dest="adaptive_anchor_score_mode",
        choices=("intra", "proxy", "linear_fusion", "quota_union"),
        default="intra",
        help="Patch-anchor score mode for register-mediated adaptive anchors.",
    )
    parser.add_argument(
        "--adaptive-anchor-proxy-quota-ratio",
        "--anchor-proxy-quota-ratio",
        "--anchor_proxy_quota_ratio",
        dest="adaptive_anchor_proxy_quota_ratio",
        type=float,
        default=0.0,
        help="Proxy quota ratio used by quota_union / quota_intra_proxy.",
    )
    parser.add_argument(
        "--adaptive-anchor-intra-source",
        "--anchor-intra-source",
        "--anchor_intra_source",
        dest="adaptive_anchor_intra_source",
        choices=("current_inter_qk", "cached_frame_qk"),
        default="cached_frame_qk",
        help="Source of intra-frame attention statistics for adaptive anchors.",
    )
    parser.add_argument(
        "--adaptive-anchor-frame-budget-mode",
        "--anchor-frame-budget-mode",
        "--anchor_frame_budget_mode",
        dest="adaptive_anchor_frame_budget_mode",
        choices=("uniform", "intra_concentration", "register_importance", "hybrid"),
        default="hybrid",
        help="Frame-level budget allocation mode for adaptive anchors.",
    )
    parser.add_argument(
        "--adaptive-anchor-frame-budget-top-frac",
        "--anchor-frame-budget-top-frac",
        "--anchor_frame_budget_top_frac",
        dest="adaptive_anchor_frame_budget_top_frac",
        type=float,
        default=0.1,
        help="Top fraction used by intra_concentration frame-budget scoring.",
    )
    parser.add_argument(
        "--adaptive-anchor-frame-budget-lambda-intra",
        "--anchor-frame-budget-lambda-intra",
        "--anchor_frame_budget_lambda_intra",
        dest="adaptive_anchor_frame_budget_lambda_intra",
        type=float,
        default=0.7,
        help="Hybrid frame-budget weight for intra concentration.",
    )
    parser.add_argument(
        "--adaptive-anchor-frame-budget-lambda-reg",
        "--anchor-frame-budget-lambda-reg",
        "--anchor_frame_budget_lambda_reg",
        dest="adaptive_anchor_frame_budget_lambda_reg",
        type=float,
        default=0.3,
        help="Hybrid frame-budget weight for register importance.",
    )
    parser.add_argument(
        "--adaptive-anchor-frame-budget-reg-topm",
        "--anchor-frame-budget-reg-topm",
        "--anchor_frame_budget_reg_topm",
        dest="adaptive_anchor_frame_budget_reg_topm",
        type=int,
        default=4,
        help="Top-M frame-pair strengths used by register_importance frame budgeting.",
    )
    parser.add_argument(
        "--adaptive-anchor-reg-patch-topk-ratio",
        "--anchor-reg-patch-topk-ratio",
        "--anchor_reg_patch_topk_ratio",
        dest="adaptive_anchor_reg_patch_topk_ratio",
        type=float,
        default=0.1,
        help="Top-k sparsification ratio for register-to-patch affinity.",
    )
    parser.add_argument(
        "--adaptive-anchor-reg-patch-topk-min",
        "--anchor-reg-patch-topk-min",
        "--anchor_reg_patch_topk_min",
        dest="adaptive_anchor_reg_patch_topk_min",
        type=int,
        default=8,
        help="Minimum top-k for register-to-patch affinity sparsification.",
    )
    parser.add_argument(
        "--adaptive-anchor-reg-patch-topk-max",
        "--anchor-reg-patch-topk-max",
        "--anchor_reg_patch_topk_max",
        dest="adaptive_anchor_reg_patch_topk_max",
        type=int,
        default=64,
        help="Maximum top-k for register-to-patch affinity sparsification.",
    )
    parser.add_argument(
        "--adaptive-anchor-reg-patch-conf-power",
        "--anchor-reg-patch-conf-power",
        "--anchor_reg_patch_conf_power",
        dest="adaptive_anchor_reg_patch_conf_power",
        type=float,
        default=1.0,
        help="Confidence exponent for register-to-patch affinity reweighting.",
    )
    parser.add_argument(
        "--adaptive-anchor-reg-patch-min-conf",
        "--anchor-reg-patch-min-conf",
        "--anchor_reg_patch_min_conf",
        dest="adaptive_anchor_reg_patch_min_conf",
        type=float,
        default=0.05,
        help="Minimum confidence for register-to-patch affinity contribution.",
    )
    parser.add_argument(
        "--adaptive-anchor-query-conditioned-eta",
        "--anchor-query-conditioned-eta",
        "--anchor_query_conditioned_eta",
        dest="adaptive_anchor_query_conditioned_eta",
        type=float,
        default=0.1,
        help="Query-conditioned modulation weight for register_gated_intra_query.",
    )
    parser.add_argument(
        "--adaptive-anchor-gated-anchor-ratio-per-key-frame",
        "--anchor-gated-anchor-ratio-per-key-frame",
        "--anchor_gated_anchor_ratio_per_key_frame",
        dest="adaptive_anchor_gated_anchor_ratio_per_key_frame",
        type=float,
        default=0.1,
        help="Patch-anchor ratio retained per selected key frame in gated modes.",
    )
    parser.add_argument(
        "--adaptive-anchor-gated-min-per-key-frame",
        "--anchor-gated-min-per-key-frame",
        "--anchor_gated_min_per_key_frame",
        dest="adaptive_anchor_gated_min_per_key_frame",
        type=int,
        default=4,
        help="Minimum number of anchors retained per selected key frame in gated modes.",
    )
    parser.add_argument(
        "--adaptive-anchor-gated-max-per-key-frame",
        "--anchor-gated-max-per-key-frame",
        "--anchor_gated_max_per_key_frame",
        dest="adaptive_anchor_gated_max_per_key_frame",
        type=int,
        default=64,
        help="Maximum number of anchors retained per selected key frame in gated modes.",
    )
    parser.add_argument(
        "--adaptive-anchor-no-self-frame",
        dest="adaptive_anchor_always_include_self_frame",
        action="store_false",
        help="Do not force the query frame itself into gated key-frame neighborhoods.",
    )
    parser.set_defaults(adaptive_anchor_always_include_self_frame=True)
    parser.add_argument(
        "--adaptive-anchor-random-seed",
        "--anchor-random-seed",
        "--anchor_random_seed",
        dest="adaptive_anchor_random_seed",
        type=int,
        default=33,
        help="Random seed for adaptive random anchor/frame baselines.",
    )
    parser.add_argument(
        "--adaptive-anchor-profile",
        "--anchor-profile",
        "--anchor_profile",
        dest="adaptive_anchor_profile",
        action="store_true",
        help="Record runtime profile for adaptive anchor scoring/selection/attention.",
    )
    parser.add_argument(
        "--adaptive-anchor-topm-frames",
        "--anchor-topm-frames",
        "--anchor_topm_frames",
        dest="adaptive_anchor_topm_frames",
        type=int,
        default=4,
        help="Top-M key frames per query frame for frame-pair gated modes. Use 0 to disable gating.",
    )
    parser.add_argument(
        "--adaptive-anchor-debug",
        "--anchor-debug",
        "--anchor_debug",
        dest="adaptive_anchor_debug",
        action="store_true",
        help="Enable adaptive anchor debug output in the model.",
    )
    parser.add_argument(
        "--adaptive-anchor-debug-dir",
        "--anchor-debug-dir",
        "--anchor_debug_dir",
        dest="adaptive_anchor_debug_dir",
        type=Path,
        default=Path("outputs/debug_register_mediated_anchor"),
        help="Directory for per-layer adaptive anchor debug .pt files.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_layer_indices(spec: str, depth: int) -> list[int]:
    normalized = spec.strip().lower()
    if not normalized or normalized == "none":
        return []
    if normalized == "all":
        return list(range(depth))
    layers: set[int] = set()
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid layer range {part!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    invalid = sorted(layer for layer in layers if layer < 0 or layer >= depth)
    if invalid:
        raise ValueError(f"Layer indices out of range 0..{depth - 1}: {invalid}")
    return sorted(layers)


def parse_split_sequence(line: str) -> str:
    digits = "".join(character for character in line if character.isdigit())
    if not digits:
        raise ValueError(f"Invalid 7 Scenes split entry: {line!r}")
    return f"seq-{int(digits):02d}"


def select_sequence_dirs(data_root: Path, requested: Sequence[str] | None) -> list[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    sequence_dirs: list[Path] = []
    for scene in SCENES:
        split_path = data_root / scene / "TestSplit.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing official test split: {split_path}")
        for line in split_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                sequence_dirs.append(data_root / scene / parse_split_sequence(line))

    if requested:
        mapping = {
            f"{path.parent.name}/{path.name}": path
            for path in sequence_dirs
        }
        unknown = sorted(set(requested) - set(mapping))
        if unknown:
            raise ValueError(f"Unknown test sequence(s): {', '.join(unknown)}")
        sequence_dirs = [mapping[name] for name in requested]
    return sequence_dirs


def frame_index(path: Path) -> int:
    return int(path.name.split("-", 1)[1].split(".", 1)[0])


def load_frame_records(sequence_dir: Path) -> list[FrameRecord]:
    rgb_by_index = {frame_index(path): path for path in sequence_dir.glob("frame-*.color.png")}
    pose_by_index = {frame_index(path): path for path in sequence_dir.glob("frame-*.pose.txt")}
    indices = sorted(set(rgb_by_index) & set(pose_by_index))
    if not indices:
        raise FileNotFoundError(f"No extracted RGB/pose frames in {sequence_dir}")

    records: list[FrameRecord] = []
    for index in indices:
        pose = np.loadtxt(pose_by_index[index], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            continue
        projected_depth_path = sequence_dir / f"frame-{index:06d}.depth.proj.png"
        raw_depth_path = sequence_dir / f"frame-{index:06d}.depth.png"
        depth_path = (
            projected_depth_path if projected_depth_path.is_file() else raw_depth_path
        )
        records.append(
            FrameRecord(
                index=index,
                rgb_path=rgb_by_index[index],
                depth_path=depth_path,
                pose_path=pose_by_index[index],
                c2w=pose,
            )
        )
    if not records:
        raise ValueError(f"{sequence_dir}: no frames have finite 4x4 poses")
    return records


def sample_records(
    pools: dict[str, list[FrameRecord]],
    num_frames: int,
    seed: int,
    strategy: str = "uniform",
    sampling_pool_frames: int = 0,
    sampling_stride: int = 3,
) -> tuple[dict[str, list[FrameRecord]], dict[str, list[int]]]:
    del seed, strategy, sampling_pool_frames
    sampled, sampled_indices = {}, {}
    for name, pool in pools.items():
        if len(pool) < num_frames:
            continue
        if len(pool) > sampling_stride * num_frames:
            indices = list(range(0, sampling_stride * num_frames, sampling_stride))
        else:
            indices = uniform_first_last_indices(len(pool), num_frames)
        sampled[name] = [pool[index] for index in indices]
        sampled_indices[name] = indices
    return sampled, sampled_indices


def load_model(
    checkpoint: Path,
    device: torch.device,
    method_name: str,
    merge_ratio: float,
    first_frame_token_indices: tuple[int, ...],
    frame_fusion_mode: str,
    frame_fusion_k: int | None,
    frame_fusion_max_group_size: int,
    frame_fusion_beta: float,
    frame_fusion_start_layer: int,
    frame_fusion_pair_percent: float,
    frame_fusion_pool_size: int,
    frame_fusion_group_similarity_threshold: float,
    frame_fusion_target_keep_policy: str,
    frame_fusion_target_keep_grid_size: int,
    frame_fusion_target_keep_percent: float,
    frame_fusion_target_keep_threshold: float,
    frame_fusion_target_keep_seed: int,
    frame_fusion_recompute_each_global: bool,
    frame_fusion_recompute_layers: str | None,
    frame_fusion_lambda_cost: float,
    frame_fusion_fixed_compression_ratio: float | None,
    frame_fusion_merge_top_similarity_percent: float,
    frame_fusion_layer_lambdas: str,
    frame_fusion_min_keep_ratio: float,
    frame_fusion_temporal_window: int,
    frame_fusion_spatial_radius: int,
    frame_fusion_spatial_neighborhood: str,
    frame_fusion_time_overlap: float,
    frame_fusion_reassignment_candidates: int,
    frame_fusion_representative_update: str,
    frame_fusion_attention_variant: str,
    sparse_attention: bool,
    sparse_ratio: float | None,
    sparse_cdf_threshold: float | None,
    sparse_pool_mode: str,
    use_adaptive_kv_anchor: bool,
    adaptive_anchor_layers: str,
    adaptive_anchor_ratio: float,
    adaptive_anchor_total: int | None,
    adaptive_anchor_min_per_frame: int,
    adaptive_anchor_tau: float,
    adaptive_anchor_uniform_mix: float,
    adaptive_anchor_strategy: str,
    adaptive_anchor_score_alpha_cross: float,
    adaptive_anchor_score_beta_intra: float,
    adaptive_anchor_score_mode: str,
    adaptive_anchor_proxy_quota_ratio: float,
    adaptive_anchor_intra_source: str,
    adaptive_anchor_frame_budget_mode: str,
    adaptive_anchor_frame_budget_top_frac: float,
    adaptive_anchor_frame_budget_lambda_intra: float,
    adaptive_anchor_frame_budget_lambda_reg: float,
    adaptive_anchor_frame_budget_reg_topm: int,
    adaptive_anchor_reg_patch_topk_ratio: float,
    adaptive_anchor_reg_patch_topk_min: int,
    adaptive_anchor_reg_patch_topk_max: int,
    adaptive_anchor_reg_patch_conf_power: float,
    adaptive_anchor_reg_patch_min_conf: float,
    adaptive_anchor_query_conditioned_eta: float,
    adaptive_anchor_gated_anchor_ratio_per_key_frame: float,
    adaptive_anchor_gated_min_per_key_frame: int,
    adaptive_anchor_gated_max_per_key_frame: int,
    adaptive_anchor_always_include_self_frame: bool,
    adaptive_anchor_profile: bool,
    adaptive_anchor_topm_frames: int | None,
    adaptive_anchor_random_seed: int,
    adaptive_anchor_debug: bool,
    adaptive_anchor_debug_dir: Path,
    fastvggt_protect_tokens: bool = True,
) -> SelfTR:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    require_selftr_checkpoint(checkpoint)
    # Pass the ratio explicitly because local FastVGGT experiments may change
    # the model constructor's default.  Zero is the unmodified paper model.
    model_kwargs = {
        "merge_ratio": merge_ratio,
        "fastvggt_protect_tokens": fastvggt_protect_tokens,
        "first_frame_token_indices": first_frame_token_indices,
        "frame_fusion_mode": frame_fusion_mode,
        "frame_fusion_k": frame_fusion_k,
        "frame_fusion_max_group_size": frame_fusion_max_group_size,
        "frame_fusion_beta": frame_fusion_beta,
        "frame_fusion_start_layer": frame_fusion_start_layer,
        "frame_fusion_pair_percent": frame_fusion_pair_percent,
        "frame_fusion_pool_size": frame_fusion_pool_size,
        "frame_fusion_group_similarity_threshold": frame_fusion_group_similarity_threshold,
        "frame_fusion_target_keep_policy": frame_fusion_target_keep_policy,
        "frame_fusion_target_keep_grid_size": frame_fusion_target_keep_grid_size,
        "frame_fusion_target_keep_percent": frame_fusion_target_keep_percent,
        "frame_fusion_target_keep_threshold": frame_fusion_target_keep_threshold,
        "frame_fusion_target_keep_seed": frame_fusion_target_keep_seed,
        "frame_fusion_recompute_each_global": frame_fusion_recompute_each_global,
        "frame_fusion_recompute_layers": frame_fusion_recompute_layers,
        "frame_fusion_lambda_cost": frame_fusion_lambda_cost,
        "frame_fusion_fixed_compression_ratio": frame_fusion_fixed_compression_ratio,
        "frame_fusion_merge_top_similarity_percent": frame_fusion_merge_top_similarity_percent,
        "frame_fusion_layer_lambdas": frame_fusion_layer_lambdas,
        "frame_fusion_min_keep_ratio": frame_fusion_min_keep_ratio,
        "frame_fusion_temporal_window": frame_fusion_temporal_window,
        "frame_fusion_spatial_radius": frame_fusion_spatial_radius,
        "frame_fusion_spatial_neighborhood": frame_fusion_spatial_neighborhood,
        "frame_fusion_time_overlap": frame_fusion_time_overlap,
        "frame_fusion_reassignment_candidates": frame_fusion_reassignment_candidates,
        "frame_fusion_representative_update": frame_fusion_representative_update,
        "frame_fusion_attention_variant": frame_fusion_attention_variant,
        "sparse_attention": sparse_attention,
        "sparse_ratio": sparse_ratio,
        "sparse_cdf_threshold": sparse_cdf_threshold,
        "sparse_pool_mode": sparse_pool_mode,
    }
    adaptive_kwargs = {
        "use_adaptive_kv_anchor": use_adaptive_kv_anchor,
        "adaptive_anchor_layers": adaptive_anchor_layers,
        "adaptive_anchor_ratio": adaptive_anchor_ratio,
        "adaptive_anchor_total": adaptive_anchor_total,
        "adaptive_anchor_min_per_frame": adaptive_anchor_min_per_frame,
        "adaptive_anchor_tau": adaptive_anchor_tau,
        "adaptive_anchor_uniform_mix": adaptive_anchor_uniform_mix,
        "adaptive_anchor_strategy": adaptive_anchor_strategy,
        "adaptive_anchor_score_alpha_cross": adaptive_anchor_score_alpha_cross,
        "adaptive_anchor_score_beta_intra": adaptive_anchor_score_beta_intra,
        "adaptive_anchor_score_mode": adaptive_anchor_score_mode,
        "adaptive_anchor_proxy_quota_ratio": adaptive_anchor_proxy_quota_ratio,
        "adaptive_anchor_intra_source": adaptive_anchor_intra_source,
        "adaptive_anchor_frame_budget_mode": adaptive_anchor_frame_budget_mode,
        "adaptive_anchor_frame_budget_top_frac": adaptive_anchor_frame_budget_top_frac,
        "adaptive_anchor_frame_budget_lambda_intra": adaptive_anchor_frame_budget_lambda_intra,
        "adaptive_anchor_frame_budget_lambda_reg": adaptive_anchor_frame_budget_lambda_reg,
        "adaptive_anchor_frame_budget_reg_topm": adaptive_anchor_frame_budget_reg_topm,
        "adaptive_anchor_reg_patch_topk_ratio": adaptive_anchor_reg_patch_topk_ratio,
        "adaptive_anchor_reg_patch_topk_min": adaptive_anchor_reg_patch_topk_min,
        "adaptive_anchor_reg_patch_topk_max": adaptive_anchor_reg_patch_topk_max,
        "adaptive_anchor_reg_patch_conf_power": adaptive_anchor_reg_patch_conf_power,
        "adaptive_anchor_reg_patch_min_conf": adaptive_anchor_reg_patch_min_conf,
        "adaptive_anchor_query_conditioned_eta": adaptive_anchor_query_conditioned_eta,
        "adaptive_anchor_gated_anchor_ratio_per_key_frame": adaptive_anchor_gated_anchor_ratio_per_key_frame,
        "adaptive_anchor_gated_min_per_key_frame": adaptive_anchor_gated_min_per_key_frame,
        "adaptive_anchor_gated_max_per_key_frame": adaptive_anchor_gated_max_per_key_frame,
        "adaptive_anchor_always_include_self_frame": adaptive_anchor_always_include_self_frame,
        "adaptive_anchor_profile": adaptive_anchor_profile,
        "adaptive_anchor_topm_frames": adaptive_anchor_topm_frames,
        "adaptive_anchor_random_seed": adaptive_anchor_random_seed,
        "adaptive_anchor_debug": adaptive_anchor_debug,
        "adaptive_anchor_debug_dir": adaptive_anchor_debug_dir,
    }
    signature = inspect.signature(SelfTR)
    accepts_extra_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    unsupported_adaptive_kwargs = [
        key for key in adaptive_kwargs if key not in signature.parameters
    ]
    if accepts_extra_kwargs or not unsupported_adaptive_kwargs:
        model_kwargs.update(adaptive_kwargs)
    elif use_adaptive_kv_anchor:
        missing = ", ".join(unsupported_adaptive_kwargs)
        raise RuntimeError(
            "This SelfTR/VGGT build does not support adaptive K/V anchor parameters "
            f"({missing}). Update the bundled VGGT backbone before running with "
            "--use-adaptive-kv-anchor."
        )
    model = SelfTR(method_name=method_name, **model_kwargs)
    state = load_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device).eval()


def da_forward(model: SelfTR, images: torch.Tensor, chunk_size: int) -> dict[str, torch.Tensor]:
    """Full DA-VGGT: cached visual tokens and pose-weighted re-chunking."""
    return model.forward_da_vggt(images, chunk_size=chunk_size)


def to_homogeneous_w2c(extrinsics: torch.Tensor) -> np.ndarray:
    w2c = extrinsics.detach().float().cpu().numpy().astype(np.float64)
    result = np.broadcast_to(np.eye(4), (len(w2c), 4, 4)).copy()
    result[:, :3] = w2c
    return result


def pairwise_pose_errors(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match facebookresearch/vggt's official pairwise angular metric."""
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for first in range(len(pred_w2c)):
        for second in range(first + 1, len(pred_w2c)):
            gt_relative = gt_w2c[first] @ np.linalg.inv(gt_w2c[second])
            pred_relative = pred_w2c[first] @ np.linalg.inv(pred_w2c[second])

            rotation_delta = gt_relative[:3, :3].T @ pred_relative[:3, :3]
            cosine = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
            rotation_errors.append(math.degrees(math.acos(float(cosine))))

            gt_translation = gt_relative[:3, 3]
            pred_translation = pred_relative[:3, 3]
            denominator = np.linalg.norm(gt_translation) * np.linalg.norm(pred_translation)
            if denominator <= 1e-15:
                translation_errors.append(1e6)
            else:
                cosine_t = np.clip(
                    abs(float(np.dot(gt_translation, pred_translation))) / denominator,
                    0.0,
                    1.0,
                )
                translation_errors.append(math.degrees(math.acos(cosine_t)))
    return np.asarray(rotation_errors), np.asarray(translation_errors)


def official_auc(rotation_errors: np.ndarray, translation_errors: np.ndarray, threshold: int) -> float:
    max_errors = np.maximum(rotation_errors, translation_errors)
    histogram, _ = np.histogram(max_errors, bins=np.arange(threshold + 1))
    return float(np.mean(np.cumsum(histogram.astype(np.float64) / len(max_errors))))


def read_resized_depth(path: Path, height: int, width: int) -> np.ndarray:
    with Image.open(path) as image:
        raw = np.asarray(image, dtype=np.uint16)
    resized = Image.fromarray(raw).resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.float32) / 1000.0


def depth_sums(
    predicted: np.ndarray,
    records: Sequence[FrameRecord],
    alignment: str,
    min_depth: float,
    max_depth: float,
) -> tuple[float, int, int, list[float], dict[str, float]]:
    height, width = predicted.shape[1:]
    ground_truth = np.stack(
        [read_resized_depth(record.depth_path, height, width) for record in records]
    )
    valid = np.isfinite(ground_truth) & (ground_truth > min_depth) & (ground_truth < max_depth)
    valid &= np.isfinite(predicted) & (predicted > 0)
    aligned = predicted.astype(np.float64, copy=True)
    scales: list[float] = []
    if alignment == "per-frame-median":
        for index in range(len(predicted)):
            if not np.any(valid[index]):
                scales.append(float("nan"))
                continue
            scale = float(
                np.median(ground_truth[index][valid[index]])
                / np.median(predicted[index][valid[index]])
            )
            aligned[index] *= scale
            scales.append(scale)
    else:
        if not np.any(valid):
            raise ValueError("Sequence has no valid depth pixels")
        scale = float(np.median(ground_truth[valid]) / np.median(predicted[valid]))
        aligned *= scale
        scales = [scale] * len(predicted)

    # Standard monocular depth evaluation clips predictions to the dataset's
    # valid range after resolving scale. Without this, a tiny number of very
    # large edge predictions dominate AbsRel while barely affecting delta1.25.
    aligned = np.clip(aligned, min_depth, max_depth)

    gt_valid = ground_truth[valid].astype(np.float64)
    pred_valid = aligned[valid]
    if len(gt_valid) == 0:
        raise ValueError("Sequence has no valid depth pixels")
    abs_rel_sum = float(np.sum(np.abs(pred_valid - gt_valid) / gt_valid))
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta_count = int(np.count_nonzero(ratio < 1.25))
    return abs_rel_sum, delta_count, len(gt_valid), scales, depth_error_metrics(aligned, ground_truth, valid)


def geometry_from_prediction(
    predicted_depth: np.ndarray,
    pred_w2c: np.ndarray,
    records: Sequence[FrameRecord],
    min_depth: float,
    max_depth: float,
    max_geometry_points: int,
) -> dict[str, float | int]:
    """Pi3-compatible ACC/Comp/NC metrics for Omega's depth+camera outputs.

    Omega predicts depth rather than an explicit global point map.  We therefore
    back-project it using its predicted camera poses; GT is back-projected from
    the RGB-registered depth maps and GT poses on the same evaluation grid.
    Sim(3)+ICP in ``evaluate_pi3_geometry`` is identical to the Pi3 protocol.
    """
    height, width = predicted_depth.shape[1:]
    gt_depth = np.stack([read_resized_depth(record.depth_path, height, width) for record in records])
    intrinsic = np.array([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]])
    # 7Scenes RGB-registered depth maps are 640x480 before model resizing.
    intrinsics = scaled_intrinsics(intrinsic, (480, 640), (height, width), len(records))
    pred_c2w = np.linalg.inv(pred_w2c)
    gt_c2w = np.stack([record.c2w for record in records])
    pred_points = depth_to_world_points(predicted_depth, pred_c2w, intrinsics)
    gt_points = depth_to_world_points(gt_depth, gt_c2w, intrinsics)
    valid = (
        np.isfinite(predicted_depth) & (predicted_depth > 0)
        & np.isfinite(gt_depth) & (gt_depth > min_depth) & (gt_depth < max_depth)
    )
    return evaluate_pi3_geometry(
        pred_points,
        gt_points,
        valid,
        max_points=max_geometry_points,
    )


def main() -> int:
    args = parse_args()
    args.method_name = resolve_method_name(args.method_name)
    args.frame_fusion_mode = canonical_frame_fusion_mode(args.frame_fusion_mode)
    args.acceleration_method = canonical_frame_fusion_mode(args.acceleration_method)
    if args.frame_fusion_recompute_layers is None:
        args.frame_fusion_recompute_layers = (
            "0,10,17" if args.frame_fusion_mode == "selftr" else ""
        )
    if args.num_frames < 2:
        raise ValueError("--num-frames must be at least 2")
    if args.sampling_stride <= 0:
        raise ValueError("--sampling-stride must be positive")
    if args.max_geometry_points < 3:
        raise ValueError("--max-geometry-points must be at least 3")
    resolved_reference_frame_index = resolve_reference_frame_index(
        args.reference_frame_index,
        args.num_frames,
    )
    input_order_from_sample = reference_first_order(
        args.num_frames,
        resolved_reference_frame_index,
    )
    first_frame_token_indices = resolve_first_frame_token_indices(
        args.first_frame_token_indices,
        args.num_frames,
    )
    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be at least 1")
    if args.image_resolution <= 0 or args.image_resolution % 16:
        raise ValueError("--image-resolution must be positive and divisible by 16")
    if not 0 < args.min_depth < args.max_depth:
        raise ValueError("Depth range must satisfy 0 < --min-depth < --max-depth")
    if not 0.0 <= args.merge_ratio <= 1.0:
        raise ValueError("--merge-ratio must be in [0, 1]")
    if args.frame_fusion_lambda_cost < 0.0:
        raise ValueError("--frame-fusion-lambda-cost must be non-negative")
    if args.frame_fusion_fixed_compression_ratio is not None and not (
        0.0 <= args.frame_fusion_fixed_compression_ratio < 1.0
    ):
        raise ValueError(
            "--frame-fusion-fixed-compression-ratio must be in [0, 1)"
        )
    if not 0.0 < args.frame_fusion_merge_top_similarity_percent <= 100.0:
        raise ValueError(
            "--frame-fusion-merge-top-similarity-percent must be in (0, 100]"
        )
    if not 0.0 < args.frame_fusion_min_keep_ratio <= 1.0:
        raise ValueError("--frame-fusion-min-keep-ratio must be in (0, 1]")
    if args.frame_fusion_temporal_window <= 0:
        raise ValueError("--frame-fusion-temporal-window must be positive")
    if args.frame_fusion_spatial_radius <= 0:
        raise ValueError("--frame-fusion-spatial-radius must be positive")
    if not 0.0 <= args.frame_fusion_time_overlap <= 1.0:
        raise ValueError("--frame-fusion-time-overlap must be in [0, 1]")
    if args.frame_fusion_reassignment_candidates <= 0:
        raise ValueError("--frame-fusion-reassignment-candidates must be positive")
    if args.frame_fusion_mode == "dp-medoid":
        if args.frame_fusion_k is None:
            raise ValueError("--frame-fusion-k is required when --frame-fusion-mode dp-medoid")
        if args.frame_fusion_k <= 0:
            raise ValueError("--frame-fusion-k must be positive")
        if args.frame_fusion_max_group_size <= 0:
            raise ValueError("--frame-fusion-max-group-size must be positive")
        if args.frame_fusion_beta < 0.0:
            raise ValueError("--frame-fusion-beta must be non-negative")
        if args.frame_fusion_k > args.num_frames:
            raise ValueError("--frame-fusion-k must be <= --num-frames")
        if args.num_frames > args.frame_fusion_k * args.frame_fusion_max_group_size:
            raise ValueError("--frame-fusion-k * --frame-fusion-max-group-size must cover --num-frames")
    elif args.frame_fusion_mode in {
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
    }:
        if not 0.0 < args.frame_fusion_pair_percent <= 100.0:
            raise ValueError("--frame-fusion-pair-percent must be in (0, 100]")
        if args.frame_fusion_pool_size <= 0:
            raise ValueError("--frame-fusion-pool-size must be positive")
        if args.frame_fusion_target_keep_grid_size <= 0:
            raise ValueError("--frame-fusion-target-keep-grid-size must be positive")
        if not 0.0 <= args.frame_fusion_target_keep_percent <= 100.0:
            raise ValueError("--frame-fusion-target-keep-percent must be in [0, 100]")
        if (
            args.frame_fusion_target_keep_policy == "least-similar"
            and args.frame_fusion_target_keep_percent <= 0.0
        ):
            raise ValueError("--frame-fusion-target-keep-percent must be positive for least-similar")
        if not -1.0 <= args.frame_fusion_target_keep_threshold <= 1.0:
            raise ValueError("--frame-fusion-target-keep-threshold must be in [-1, 1]")
        if not -1.0 <= args.frame_fusion_group_similarity_threshold <= 1.0:
            raise ValueError("--frame-fusion-group-similarity-threshold must be in [-1, 1]")
    if args.frame_fusion_recompute_each_global and args.frame_fusion_mode not in {
        "pair-top-percent",
        "h-m",
        "h-r",
        "selftr",
        "u-r",
    }:
        raise ValueError(
            "--frame-fusion-recompute-each-global requires pair-top-percent or "
            "a spatiotemporal representative mode"
        )
    if args.frame_fusion_mode != "none":
        temporal_modes = {
            "temporal-representative",
            "adaptive-temporal-representative",
        }
        if args.merge_ratio != 0.0 and args.frame_fusion_mode not in temporal_modes:
            raise ValueError(
                "FastVGGT after frame fusion is only supported for temporal representative modes"
            )
    if args.sparse_attention and args.merge_ratio > 0.0:
        raise ValueError("--sparse-attention requires --merge-ratio 0; sparse attention replaces token merging")
    if args.sparse_attention and args.sparse_ratio is None and args.sparse_cdf_threshold is None:
        raise ValueError("--sparse-attention requires --sparse-ratio and/or --sparse-cdf-threshold")
    if not 0.0 <= args.adaptive_anchor_ratio <= 1.0:
        raise ValueError("--adaptive-anchor-ratio must be in [0, 1]")
    if args.adaptive_anchor_total is not None and args.adaptive_anchor_total <= 0:
        raise ValueError("--adaptive-anchor-total must be positive when set")
    if args.adaptive_anchor_min_per_frame < 0:
        raise ValueError("--adaptive-anchor-min-per-frame must be non-negative")
    if args.adaptive_anchor_tau <= 0.0:
        raise ValueError("--adaptive-anchor-tau must be positive")
    if not 0.0 <= args.adaptive_anchor_uniform_mix <= 1.0:
        raise ValueError("--adaptive-anchor-uniform-mix must be in [0, 1]")
    if not 0.0 <= args.adaptive_anchor_proxy_quota_ratio <= 1.0:
        raise ValueError("--adaptive-anchor-proxy-quota-ratio must be in [0, 1]")
    if not 0.0 < args.adaptive_anchor_frame_budget_top_frac <= 1.0:
        raise ValueError("--adaptive-anchor-frame-budget-top-frac must be in (0, 1]")
    if args.adaptive_anchor_frame_budget_reg_topm <= 0:
        raise ValueError("--adaptive-anchor-frame-budget-reg-topm must be positive")
    if not 0.0 <= args.adaptive_anchor_reg_patch_topk_ratio <= 1.0:
        raise ValueError("--adaptive-anchor-reg-patch-topk-ratio must be in [0, 1]")
    if args.adaptive_anchor_reg_patch_topk_min <= 0:
        raise ValueError("--adaptive-anchor-reg-patch-topk-min must be positive")
    if args.adaptive_anchor_reg_patch_topk_max <= 0:
        raise ValueError("--adaptive-anchor-reg-patch-topk-max must be positive")
    if args.adaptive_anchor_reg_patch_topk_max < args.adaptive_anchor_reg_patch_topk_min:
        raise ValueError("--adaptive-anchor-reg-patch-topk-max must be >= --adaptive-anchor-reg-patch-topk-min")
    if args.adaptive_anchor_reg_patch_conf_power < 0.0:
        raise ValueError("--adaptive-anchor-reg-patch-conf-power must be non-negative")
    if args.adaptive_anchor_reg_patch_min_conf < 0.0:
        raise ValueError("--adaptive-anchor-reg-patch-min-conf must be non-negative")
    if args.adaptive_anchor_gated_anchor_ratio_per_key_frame < 0.0:
        raise ValueError("--adaptive-anchor-gated-anchor-ratio-per-key-frame must be non-negative")
    if args.adaptive_anchor_gated_min_per_key_frame < 0:
        raise ValueError("--adaptive-anchor-gated-min-per-key-frame must be non-negative")
    if args.adaptive_anchor_gated_max_per_key_frame <= 0:
        raise ValueError("--adaptive-anchor-gated-max-per-key-frame must be positive")
    if args.adaptive_anchor_topm_frames is not None and args.adaptive_anchor_topm_frames <= 0:
        args.adaptive_anchor_topm_frames = None
    if args.use_adaptive_kv_anchor and args.merge_ratio != 0.0:
        raise ValueError("--use-adaptive-kv-anchor requires --merge-ratio 0")
    if args.use_adaptive_kv_anchor and args.sparse_attention:
        raise ValueError("--use-adaptive-kv-anchor is not compatible with --sparse-attention")

    sequence_dirs = select_sequence_dirs(args.data_root, args.sequences)
    pools: dict[str, list[FrameRecord]] = {}
    for sequence_dir in sequence_dirs:
        sequence_name = f"{sequence_dir.parent.name}/{sequence_dir.name}"
        records = load_frame_records(sequence_dir)
        print(f"{sequence_name}: found {len(records)} frames")
        pool_name = sequence_dir.parent.name if args.sampling_unit == "scene" else sequence_name
        pools.setdefault(pool_name, []).extend(records)
    skipped_pools = {
        name: {"reason": "fewer_than_requested_frames", "available_frames": len(records)}
        for name, records in pools.items()
        if len(records) < args.num_frames
    }
    pools = {name: records for name, records in pools.items() if len(records) >= args.num_frames}
    for pool_name, records in pools.items():
        print(f"{pool_name}: sampling pool has {len(records)} frames")
    for pool_name, details in skipped_pools.items():
        print(f"{pool_name}: skipped ({details['available_frames']} < {args.num_frames} frames)")
    if not pools:
        raise ValueError("No sequence has enough valid frames for the requested --num-frames")

    sampled, sampled_pool_indices = sample_records(
        pools,
        args.num_frames,
        args.seed,
        strategy=args.sampling_strategy,
        sampling_pool_frames=args.sampling_pool_frames,
        sampling_stride=args.sampling_stride,
    )
    sampled_input_pool_indices = {
        name: [sampled_pool_indices[name][index] for index in input_order_from_sample]
        for name in sampled
    }
    sampled = {
        name: reorder_reference_first(records, resolved_reference_frame_index)
        for name, records in sampled.items()
    }
    selection = {
        name: {
            "sampling_strategy": args.sampling_strategy,
            "sampling_mode": (
                f"fixed_stride_{args.sampling_stride}_from_first"
                if len(pools[name]) > args.sampling_stride * args.num_frames
                else "uniform_first_last"
            ),
            "sampling_pool_frames_requested": args.sampling_pool_frames,
            "sampling_pool_size": len(pools[name]),
            "pool_indices": sampled_input_pool_indices[name],
            "original_sample_order_pool_indices": sampled_pool_indices[name],
            "input_order_from_original_sample": input_order_from_sample,
            "reference_frame_original_sample_index": resolved_reference_frame_index,
            "reference_pool_index": sampled_pool_indices[name][resolved_reference_frame_index],
            "reference_frame_index": records[0].index,
            "reference_frame_input_index": 0,
            "first_frame_token_input_indices": list(first_frame_token_indices),
            "frame_indices": [record.index for record in records],
            "rgb_paths": [str(record.rgb_path) for record in records],
            "depth_paths": [str(record.depth_path) for record in records],
            "depth_sources": [
                "projected" if record.depth_path.name.endswith(".depth.proj.png") else "raw"
                for record in records
            ],
        }
        for name, records in sampled.items()
    }
    missing_depth = [
        str(record.depth_path)
        for records in sampled.values()
        for record in records
        if not record.depth_path.is_file()
    ]
    if args.dry_run:
        print(json.dumps({"sampled_frames": selection, "missing_depth": missing_depth}, indent=2))
        return 0 if not missing_depth else 3
    if missing_depth:
        raise FileNotFoundError(
            f"Missing {len(missing_depth)} sampled depth maps; first: {missing_depth[0]}"
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega inference requires CUDA")
    exclusive_gpu_index = args.exclusive_gpu_index
    if args.require_exclusive_gpu and exclusive_gpu_index is None:
        if device.index is None:
            raise ValueError(
                "--exclusive-gpu-index is required with --require-exclusive-gpu when "
                "--device does not encode a physical CUDA index."
            )
        exclusive_gpu_index = int(device.index)
    if args.require_exclusive_gpu:
        assert_exclusive_gpu(
            exclusive_gpu_index,
            allowed_pids={os.getpid()},
            max_other_memory_mib=args.exclusive_gpu_max_other_memory_mib,
        )
        print(
            "Exclusive GPU guard: "
            f"physical_gpu={exclusive_gpu_index}, "
            f"max_other_memory_mib={args.exclusive_gpu_max_other_memory_mib}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sampled_frames.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)
        handle.write("\n")

    print(f"Loading {args.checkpoint}")
    model = load_model(
        args.checkpoint,
        device,
        args.method_name,
        args.merge_ratio,
        first_frame_token_indices,
        args.frame_fusion_mode,
        args.frame_fusion_k,
        args.frame_fusion_max_group_size,
        args.frame_fusion_beta,
        args.frame_fusion_start_layer,
        args.frame_fusion_pair_percent,
        args.frame_fusion_pool_size,
        args.frame_fusion_group_similarity_threshold,
        args.frame_fusion_target_keep_policy,
        args.frame_fusion_target_keep_grid_size,
        args.frame_fusion_target_keep_percent,
        args.frame_fusion_target_keep_threshold,
        args.frame_fusion_target_keep_seed,
        args.frame_fusion_recompute_each_global,
        args.frame_fusion_recompute_layers,
        args.frame_fusion_lambda_cost,
        args.frame_fusion_fixed_compression_ratio,
        args.frame_fusion_merge_top_similarity_percent,
        args.frame_fusion_layer_lambdas,
        args.frame_fusion_min_keep_ratio,
        args.frame_fusion_temporal_window,
        args.frame_fusion_spatial_radius,
        args.frame_fusion_spatial_neighborhood,
        args.frame_fusion_time_overlap,
        args.frame_fusion_reassignment_candidates,
        args.frame_fusion_representative_update,
        args.frame_fusion_attention_variant,
        args.sparse_attention,
        args.sparse_ratio,
        args.sparse_cdf_threshold,
        args.sparse_pool_mode,
        args.use_adaptive_kv_anchor,
        args.adaptive_anchor_layers,
        args.adaptive_anchor_ratio,
        args.adaptive_anchor_total,
        args.adaptive_anchor_min_per_frame,
        args.adaptive_anchor_tau,
        args.adaptive_anchor_uniform_mix,
        args.adaptive_anchor_strategy,
        args.adaptive_anchor_score_alpha_cross,
        args.adaptive_anchor_score_beta_intra,
        args.adaptive_anchor_score_mode,
        args.adaptive_anchor_proxy_quota_ratio,
        args.adaptive_anchor_intra_source,
        args.adaptive_anchor_frame_budget_mode,
        args.adaptive_anchor_frame_budget_top_frac,
        args.adaptive_anchor_frame_budget_lambda_intra,
        args.adaptive_anchor_frame_budget_lambda_reg,
        args.adaptive_anchor_frame_budget_reg_topm,
        args.adaptive_anchor_reg_patch_topk_ratio,
        args.adaptive_anchor_reg_patch_topk_min,
        args.adaptive_anchor_reg_patch_topk_max,
        args.adaptive_anchor_reg_patch_conf_power,
        args.adaptive_anchor_reg_patch_min_conf,
        args.adaptive_anchor_query_conditioned_eta,
        args.adaptive_anchor_gated_anchor_ratio_per_key_frame,
        args.adaptive_anchor_gated_min_per_key_frame,
        args.adaptive_anchor_gated_max_per_key_frame,
        args.adaptive_anchor_always_include_self_frame,
        args.adaptive_anchor_profile,
        args.adaptive_anchor_topm_frames,
        args.adaptive_anchor_random_seed,
        args.adaptive_anchor_debug,
        args.adaptive_anchor_debug_dir,
        fastvggt_protect_tokens=not args.fastvggt_disable_protection,
    )
    if args.attention_mode == "register-only-zero-shot":
        model.aggregator.inter_frame_attention_types = ["register"] * model.aggregator.depth
        register_only_global_layers = parse_layer_indices(
            args.register_only_global_layers or DEFAULT_REGISTER_ONLY_GLOBAL_LAYER_SPEC,
            model.aggregator.depth,
        )
        for layer in register_only_global_layers:
            model.aggregator.inter_frame_attention_types[layer] = "global"
        model.aggregator.set_register_patch_inter_frame(
            mode=args.register_patch_inter_frame_mode,
            percent=args.register_patch_inter_frame_percent,
            seed=args.seed,
        )
    elif (
        args.register_only_global_layers
        or
        args.register_patch_inter_frame_mode != "none"
        or args.register_patch_inter_frame_percent != 0.0
    ):
        raise ValueError(
            "register-only global layers and patch selection are only supported with register-only-zero-shot"
        )
    else:
        register_only_global_layers = []
    inter_frame_only_global_layers = parse_layer_indices(
        args.inter_frame_only_global_layers,
        model.aggregator.depth,
    )
    model.aggregator.set_inter_frame_only_layers(inter_frame_only_global_layers)
    num_register_blocks = model.aggregator.inter_frame_attention_types.count("register")
    global_blocks = [
        index
        for index, attention_type in enumerate(model.aggregator.inter_frame_attention_types)
        if attention_type == "global"
    ]
    print(
        f"Attention schedule: {args.attention_mode} "
        f"({num_register_blocks}/{model.aggregator.depth} inter-frame blocks use register attention; "
        f"global blocks={global_blocks})"
    )
    if inter_frame_only_global_layers:
        print(f"Inter-frame-only global layers: {inter_frame_only_global_layers}")
    if args.sparse_attention:
        print(
            "Sparse global attention: "
            f"sparse_ratio={args.sparse_ratio}, "
            f"cdf_threshold={args.sparse_cdf_threshold}, "
            f"pool={args.sparse_pool_mode}"
        )
    if args.use_adaptive_kv_anchor:
        print(
            "Adaptive K/V anchor: "
            f"layers={args.adaptive_anchor_layers}, "
            f"ratio={args.adaptive_anchor_ratio}, "
            f"total={args.adaptive_anchor_total}, "
            f"min_per_frame={args.adaptive_anchor_min_per_frame}, "
            f"tau={args.adaptive_anchor_tau}, "
            f"uniform_mix={args.adaptive_anchor_uniform_mix}, "
            f"mode={args.adaptive_anchor_strategy}, "
            f"alpha_cross={args.adaptive_anchor_score_alpha_cross}, "
            f"beta_intra={args.adaptive_anchor_score_beta_intra}, "
            f"score_mode={args.adaptive_anchor_score_mode}, "
            f"intra_source={args.adaptive_anchor_intra_source}, "
            f"frame_budget={args.adaptive_anchor_frame_budget_mode}, "
            f"proxy_quota={args.adaptive_anchor_proxy_quota_ratio}, "
            f"topm_frames={args.adaptive_anchor_topm_frames}, "
            f"random_seed={args.adaptive_anchor_random_seed}, "
            f"profile={args.adaptive_anchor_profile}, "
            f"debug={args.adaptive_anchor_debug}, "
            f"debug_dir={args.adaptive_anchor_debug_dir}"
        )
    all_rotation_errors: list[np.ndarray] = []
    all_translation_errors: list[np.ndarray] = []
    total_abs_rel = 0.0
    total_delta = 0
    total_valid = 0
    per_sequence: list[dict[str, object]] = []

    for sequence_name, records in sampled.items():
        images = load_and_preprocess_images(
            [str(record.rgb_path) for record in records],
            mode=args.resize_mode,
            image_resolution=args.image_resolution,
        ).to(device, non_blocking=True)
        if args.require_exclusive_gpu:
            assert_exclusive_gpu(
                exclusive_gpu_index,
                allowed_pids={os.getpid()},
                max_other_memory_mib=args.exclusive_gpu_max_other_memory_mib,
            )
        if not args.skip_timing:
            with torch.inference_mode():
                warmup = da_forward(model, images, args.da_chunk_size) if args.acceleration_method == "da-vggt" else model(images)
            del warmup
            torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings_ms: list[float] = []
        predictions = None
        repeats = 1 if args.skip_timing else args.timing_repeats
        for _ in range(repeats):
            if not args.skip_timing:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            with torch.inference_mode():
                current_predictions = da_forward(model, images, args.da_chunk_size) if args.acceleration_method == "da-vggt" else model(images)
            if not args.skip_timing:
                end_event.record()
                torch.cuda.synchronize(device)
                timings_ms.append(float(start_event.elapsed_time(end_event)))
            else:
                torch.cuda.synchronize(device)
            if predictions is not None:
                del predictions
            predictions = current_predictions
        assert predictions is not None

        with torch.inference_mode():
            extrinsics = predictions["da_w2c"] if "da_w2c" in predictions else encoding_to_camera(
                predictions["pose_enc"], predictions["images"].shape[-2:], build_intrinsics=False
            )[0]
        pred_w2c = to_homogeneous_w2c(extrinsics[0])
        gt_w2c = np.linalg.inv(np.stack([record.c2w for record in records]))
        rotation_errors, translation_errors = pairwise_pose_errors(pred_w2c, gt_w2c)
        predicted_depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
        abs_rel_sum, delta_count, valid_count, scales, depth_metric = depth_sums(
            predicted_depth, records, args.depth_alignment, args.min_depth, args.max_depth
        )
        all_rotation_errors.append(rotation_errors)
        all_translation_errors.append(translation_errors)
        total_abs_rel += abs_rel_sum
        total_delta += delta_count
        total_valid += valid_count

        row: dict[str, object] = {
            "sequence": sequence_name,
            "auc_3_percent": 100 * official_auc(rotation_errors, translation_errors, 3),
            "auc_5_percent": 100 * official_auc(rotation_errors, translation_errors, 5),
            "auc_10_percent": 100 * official_auc(rotation_errors, translation_errors, 10),
            "auc_15_percent": 100 * official_auc(rotation_errors, translation_errors, 15),
            "auc_30_percent": 100 * official_auc(rotation_errors, translation_errors, 30),
            "delta_1_25_percent": 100 * delta_count / valid_count,
            "abs_rel": abs_rel_sum / valid_count,
            "valid_depth_pixels": valid_count,
            "depth_scales": scales,
            "model_latency_ms": None if args.skip_timing else float(np.median(timings_ms)),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        }
        row.update(depth_metric)
        row.update(trajectory_pose_metrics(np.linalg.inv(pred_w2c), np.linalg.inv(gt_w2c)))
        row.update(
            geometry_from_prediction(
                predicted_depth,
                pred_w2c,
                records,
                args.min_depth,
                args.max_depth,
                args.max_geometry_points,
            )
        )
        if model.aggregator.last_frame_fusion_debug:
            debug = model.aggregator.last_frame_fusion_debug
            row["frame_fusion_num_groups"] = debug.get("num_groups")
            row["frame_fusion_num_fused_frames"] = debug.get("num_fused_frames")
            row["frame_fusion_partition_seconds"] = debug.get("partition_seconds")
            row["frame_fusion_fusion_seconds"] = debug.get("fusion_seconds")
            row["frame_fusion_planning_seconds"] = debug.get("planning_seconds")
            row["frame_fusion_global_attention_seconds"] = debug.get(
                "global_attention_seconds"
            )
            row["frame_fusion_attention_variant"] = debug.get(
                "attention_variant", args.frame_fusion_attention_variant
            )
            row["frame_fusion_attention_error_by_token_type"] = debug.get(
                "attention_error_by_token_type"
            )
            batches = debug.get("batches") or []
            first_batch = batches[0] if batches else {}
            row["frame_fusion_selected_pairs"] = debug.get(
                "avg_selected_pairs",
                first_batch.get("selected_pairs"),
            )
            row["frame_fusion_selected_groups"] = debug.get(
                "avg_selected_groups",
                first_batch.get("selected_groups"),
            )
            row["frame_fusion_effective_anchor_target_relations"] = first_batch.get(
                "effective_anchor_target_relations"
            )
            row["frame_fusion_group_size_min"] = first_batch.get("group_size_min")
            row["frame_fusion_group_size_max"] = first_batch.get("group_size_max")
            row["frame_fusion_group_size_mean"] = first_batch.get("group_size_mean")
            row["frame_fusion_planner_cuda_profile_ms"] = first_batch.get(
                "planner_cuda_profile_ms"
            )
            row["frame_fusion_edge_score_backend"] = first_batch.get(
                "edge_score_backend"
            )
            row["frame_fusion_group_partition"] = first_batch.get("group_partition")
            row["frame_fusion_full_partition"] = first_batch.get("full_partition")
            row["frame_fusion_full_partition_groups"] = first_batch.get("full_partition_groups")
            row["frame_fusion_singleton_partition_groups"] = first_batch.get(
                "singleton_partition_groups"
            )
            row["frame_fusion_full_partition"] = first_batch.get("full_partition")
            row["frame_fusion_full_partition_groups"] = first_batch.get("full_partition_groups")
            row["frame_fusion_singleton_partition_groups"] = first_batch.get(
                "singleton_partition_groups"
            )
            row["frame_fusion_original_pair_partition"] = first_batch.get(
                "original_pair_partition"
            )
            row["frame_fusion_attention_tokens"] = debug.get(
                "avg_attention_tokens",
                first_batch.get("attention_tokens"),
            )
            row["frame_fusion_patch_retention_vs_full"] = debug.get(
                "avg_patch_token_retention_vs_full",
                debug.get("patch_token_retention_vs_full"),
            )
            row["frame_fusion_target_keep_policy"] = debug.get("target_keep_policy")
            row["frame_fusion_target_keep_threshold"] = debug.get("target_keep_threshold")
            row["frame_fusion_target_keep_patch_tokens_per_pair"] = first_batch.get(
                "target_keep_patch_tokens_per_pair"
            )
            row["frame_fusion_target_keep_patch_tokens_min"] = first_batch.get(
                "target_keep_patch_tokens_min"
            )
            row["frame_fusion_target_keep_patch_tokens_max"] = first_batch.get(
                "target_keep_patch_tokens_max"
            )
            row["frame_fusion_target_keep_patch_tokens_total"] = first_batch.get(
                "target_keep_patch_tokens_total"
            )
            row["frame_fusion_recompute_each_global"] = debug.get("recompute_each_global")
            row["frame_fusion_recompute_layers"] = debug.get("recompute_layers")
            row["frame_fusion_recomputed_source_layers"] = debug.get(
                "recomputed_source_layers"
            )
            row["frame_fusion_num_recomputed_layers"] = debug.get("num_recomputed_layers")
            row["frame_fusion_cost_model"] = debug.get("cost_model")
            row["frame_fusion_lambda_cost"] = debug.get(
                "lambda_cost", args.frame_fusion_lambda_cost
            )
            row["frame_fusion_fixed_compression_ratio"] = debug.get(
                "fixed_compression_ratio", args.frame_fusion_fixed_compression_ratio
            )
            row["frame_fusion_selected_compression_ratio"] = debug.get(
                "selected_compression_ratio"
            )
            if args.frame_fusion_mode in {
                "temporal-representative",
                "adaptive-temporal-representative",
                "adaptive-spatial-representative",
                "h-m",
                "h-r",
                "selftr",
                "u-r",
            }:
                first_batch = (debug.get("batches") or [{}])[0]
                row["frame_fusion_representative_count"] = first_batch.get("representative_count")
                row["frame_fusion_representative_weight_min"] = first_batch.get(
                    "representative_weight_min"
                )
                row["frame_fusion_representative_weight_max"] = first_batch.get(
                    "representative_weight_max"
                )
                row["frame_fusion_representative_weight_mean"] = first_batch.get(
                    "representative_weight_mean"
                )
                row["frame_fusion_mapping_checksum"] = first_batch.get("mapping_checksum")
                row["frame_fusion_mapping_shape"] = first_batch.get("mapping_shape")
                row["frame_fusion_layer_retention"] = [
                    {
                        "source_layer": layer_debug.get("source_layer"),
                        "representative_count": (layer_batch := (layer_debug.get("batches") or [{}])[0]).get(
                            "representative_count"
                        ),
                        "full_patch_tokens": layer_batch.get("full_patch_tokens"),
                        "patch_token_retention_percent": 100.0
                        * float(layer_batch.get("patch_token_retention_vs_full", 0.0)),
                    }
                    for layer_debug in (debug.get("layers") or [])
                ]
        fastvggt_debug = model.aggregator.last_fastvggt_debug
        row["fastvggt_actual_merge_ratio"] = fastvggt_debug.get("requested_merge_ratio")
        row["fastvggt_actual_merge_layers"] = fastvggt_debug.get("num_merge_layers")
        row["fastvggt_actual_input_tokens"] = fastvggt_debug.get("input_tokens_total")
        row["fastvggt_actual_output_tokens"] = fastvggt_debug.get("output_tokens_total")
        row["fastvggt_actual_merged_tokens"] = fastvggt_debug.get("merged_tokens_total")
        row["fastvggt_actual_retention_vs_input"] = fastvggt_debug.get(
            "retention_vs_fastvggt_input"
        )
        row["fastvggt_actual_layers"] = fastvggt_debug.get("layers")
        if args.frame_fusion_mode == "selftr":
            active_ratio = process_token_retention_percent(row.get("frame_fusion_layer_retention", []))
        elif args.acceleration_method == "fastvggt" or args.merge_ratio > 0:
            raw_ratio = row.get("fastvggt_actual_retention_vs_input")
            active_ratio = None if raw_ratio is None else 100.0 * float(raw_ratio)
        else:
            active_ratio = 100.0
        row["active_token_ratio_percent"] = active_ratio
        row["keep_ratio_percent"] = active_ratio
        per_sequence.append(row)
        latency_text = "skipped" if args.skip_timing else f"{row['model_latency_ms']:.1f}ms"
        print(
            f"[{sequence_name}] AUC@3={row['auc_3_percent']:.2f}, "
            f"AUC@15={row['auc_15_percent']:.2f}, "
            f"AUC@30={row['auc_30_percent']:.2f}, delta1.25={row['delta_1_25_percent']:.2f}, "
            f"AbsRel={row['abs_rel']:.4f}, latency={latency_text}"
        )
        del images, predictions, extrinsics
        torch.cuda.empty_cache()

    rotation_errors = np.concatenate(all_rotation_errors)
    translation_errors = np.concatenate(all_translation_errors)
    overall = {
        "auc_3_percent": 100 * official_auc(rotation_errors, translation_errors, 3),
        "auc_5_percent": 100 * official_auc(rotation_errors, translation_errors, 5),
        "auc_10_percent": 100 * official_auc(rotation_errors, translation_errors, 10),
        "auc_15_percent": 100 * official_auc(rotation_errors, translation_errors, 15),
        "auc_30_percent": 100 * official_auc(rotation_errors, translation_errors, 30),
        "delta_1_25_percent": 100 * total_delta / total_valid,
        "abs_rel": total_abs_rel / total_valid,
        "valid_depth_pixels": total_valid,
        "model_latency_ms_mean": None
        if args.skip_timing
        else float(np.mean([float(row["model_latency_ms"]) for row in per_sequence])),
        "fps": None if args.skip_timing else float(
            args.num_frames / (np.mean([float(row["model_latency_ms"]) for row in per_sequence]) / 1000.0)
        ),
        "peak_allocated_gib_max": float(
            np.max([float(row["peak_allocated_gib"]) for row in per_sequence])
        ),
        "peak_reserved_gib_max": float(
            np.max([float(row["peak_reserved_gib"]) for row in per_sequence])
        ),
        "fastvggt_actual_retention_vs_input_mean": float(
            np.mean(
                [
                    float(row["fastvggt_actual_retention_vs_input"])
                    for row in per_sequence
                    if row.get("fastvggt_actual_retention_vs_input") is not None
                ]
            )
        ),
        "active_token_ratio_percent": float(np.mean([row["active_token_ratio_percent"] for row in per_sequence])),
        "keep_ratio_percent": float(np.mean([row["keep_ratio_percent"] for row in per_sequence])),
    }
    for geometry_key in (
        "acc_mean_m", "acc_median_m", "comp_mean_m", "comp_median_m",
        "nc_mean", "nc_median", "chamfer_mean_m", "chamfer_median_m",
        "f1_at_threshold_percent", "f1_10cm_percent",
    ):
        overall[geometry_key] = float(np.mean([float(row[geometry_key]) for row in per_sequence]))
    result = {
        "method": {
            "display_name": args.method_name,
            "id": METHOD_ID if args.frame_fusion_mode == METHOD_ID else args.acceleration_method,
            "frame_fusion_mode": args.frame_fusion_mode,
        },
        "protocol": {
            "dataset": DATASET_NAME,
            "dataset_split": DATASET_SPLIT,
            "sampling_unit": args.sampling_unit,
            "sampling_strategy": args.sampling_strategy,
            "seed": args.seed,
            "reference_frame_index": args.reference_frame_index,
            "resolved_reference_frame_index": resolved_reference_frame_index,
            "reference_frame_ordering": "stable_front_from_sampled_order",
            "reference_frame_semantics": (
                "sampled frame moved to model input position 0; original "
                "VGGT-Omega assigns position-0 camera/register tokens"
            ),
            "first_frame_token_indices": list(first_frame_token_indices),
            "first_frame_token_index_spec": args.first_frame_token_indices,
            "num_frames_per_sequence": args.num_frames,
            "sampling_stride": args.sampling_stride,
            "skipped_pools": skipped_pools,
            "num_sequences": len(sampled),
            "num_pose_pairs": len(rotation_errors),
            "image_resolution": args.image_resolution,
            "resize_mode": args.resize_mode,
            "depth_alignment": args.depth_alignment,
            "depth_source": (
                "projected"
                if all(
                    record.depth_path.name.endswith(".depth.proj.png")
                    for records in sampled.values()
                    for record in records
                )
                else "official_raw_depth_fallback"
            ),
            "min_depth_m": args.min_depth,
            "max_depth_m": args.max_depth,
            "prediction_clip_m": [args.min_depth, args.max_depth],
            "max_geometry_points": args.max_geometry_points,
            "point_cloud_f1": {
                "metric": "harmonic_mean_precision_recall",
                "threshold_m": 0.1,
                "alignment": "Sim(3)+ICP",
            },
            "merge_ratio": args.merge_ratio,
            "fastvggt_protect_tokens": not args.fastvggt_disable_protection,
            "frame_fusion": {
                "mode": args.frame_fusion_mode,
                "k": args.frame_fusion_k,
                "max_group_size": args.frame_fusion_max_group_size,
                "beta": args.frame_fusion_beta,
                "start_layer": args.frame_fusion_start_layer,
                "pair_percent": args.frame_fusion_pair_percent,
                "pool_size": args.frame_fusion_pool_size,
                "group_similarity_threshold": args.frame_fusion_group_similarity_threshold,
                "target_keep_policy": args.frame_fusion_target_keep_policy,
                "target_keep_grid_size": args.frame_fusion_target_keep_grid_size,
                "target_keep_percent": args.frame_fusion_target_keep_percent,
                "target_keep_threshold": args.frame_fusion_target_keep_threshold,
                "target_keep_seed": args.frame_fusion_target_keep_seed,
                "recompute_each_global": args.frame_fusion_recompute_each_global,
                "recompute_layers": args.frame_fusion_recompute_layers,
                "lambda_cost": args.frame_fusion_lambda_cost,
                "fixed_compression_ratio": args.frame_fusion_fixed_compression_ratio,
                "merge_top_similarity_percent": args.frame_fusion_merge_top_similarity_percent,
                "layer_lambdas": args.frame_fusion_layer_lambdas,
                "min_keep_ratio": args.frame_fusion_min_keep_ratio,
                "temporal_window": args.frame_fusion_temporal_window,
                "spatial_radius": args.frame_fusion_spatial_radius,
                "candidate_topology": (
                    "spatiotemporal_cube_by_radius_and_window"
                    if args.frame_fusion_mode == "selftr"
                    else "legacy_spatial_neighborhood"
                ),
                "spatial_neighborhood": (
                    None
                    if args.frame_fusion_mode == "selftr"
                    else args.frame_fusion_spatial_neighborhood
                ),
                "time_overlap": args.frame_fusion_time_overlap,
                "reassignment_candidates": args.frame_fusion_reassignment_candidates,
                "representative_update": args.frame_fusion_representative_update,
                "attention_variant": args.frame_fusion_attention_variant,
            },
            "sparse_attention": args.sparse_attention,
            "sparse_ratio": args.sparse_ratio,
            "sparse_cdf_threshold": args.sparse_cdf_threshold,
            "sparse_pool_mode": args.sparse_pool_mode,
            "use_adaptive_kv_anchor": args.use_adaptive_kv_anchor,
            "adaptive_anchor_layers": args.adaptive_anchor_layers,
            "adaptive_anchor_active_layers": sorted(model.aggregator.adaptive_anchor_layers),
            "adaptive_anchor_ratio": args.adaptive_anchor_ratio,
            "adaptive_anchor_total": args.adaptive_anchor_total,
            "adaptive_anchor_min_per_frame": args.adaptive_anchor_min_per_frame,
            "adaptive_anchor_tau": args.adaptive_anchor_tau,
            "adaptive_anchor_uniform_mix": args.adaptive_anchor_uniform_mix,
            "adaptive_anchor_strategy": args.adaptive_anchor_strategy,
            "adaptive_anchor_score_alpha_cross": args.adaptive_anchor_score_alpha_cross,
            "adaptive_anchor_score_beta_intra": args.adaptive_anchor_score_beta_intra,
            "adaptive_anchor_score_mode": args.adaptive_anchor_score_mode,
            "adaptive_anchor_proxy_quota_ratio": args.adaptive_anchor_proxy_quota_ratio,
            "adaptive_anchor_intra_source": args.adaptive_anchor_intra_source,
            "adaptive_anchor_frame_budget_mode": args.adaptive_anchor_frame_budget_mode,
            "adaptive_anchor_frame_budget_top_frac": args.adaptive_anchor_frame_budget_top_frac,
            "adaptive_anchor_frame_budget_lambda_intra": args.adaptive_anchor_frame_budget_lambda_intra,
            "adaptive_anchor_frame_budget_lambda_reg": args.adaptive_anchor_frame_budget_lambda_reg,
            "adaptive_anchor_frame_budget_reg_topm": args.adaptive_anchor_frame_budget_reg_topm,
            "adaptive_anchor_reg_patch_topk_ratio": args.adaptive_anchor_reg_patch_topk_ratio,
            "adaptive_anchor_reg_patch_topk_min": args.adaptive_anchor_reg_patch_topk_min,
            "adaptive_anchor_reg_patch_topk_max": args.adaptive_anchor_reg_patch_topk_max,
            "adaptive_anchor_reg_patch_conf_power": args.adaptive_anchor_reg_patch_conf_power,
            "adaptive_anchor_reg_patch_min_conf": args.adaptive_anchor_reg_patch_min_conf,
            "adaptive_anchor_query_conditioned_eta": args.adaptive_anchor_query_conditioned_eta,
            "adaptive_anchor_gated_anchor_ratio_per_key_frame": args.adaptive_anchor_gated_anchor_ratio_per_key_frame,
            "adaptive_anchor_gated_min_per_key_frame": args.adaptive_anchor_gated_min_per_key_frame,
            "adaptive_anchor_gated_max_per_key_frame": args.adaptive_anchor_gated_max_per_key_frame,
            "adaptive_anchor_always_include_self_frame": args.adaptive_anchor_always_include_self_frame,
            "adaptive_anchor_profile": args.adaptive_anchor_profile,
            "adaptive_anchor_topm_frames": args.adaptive_anchor_topm_frames,
            "adaptive_anchor_random_seed": args.adaptive_anchor_random_seed,
            "adaptive_anchor_debug": args.adaptive_anchor_debug,
            "adaptive_anchor_debug_dir": str(args.adaptive_anchor_debug_dir),
            "attention_mode": args.attention_mode,
            "register_only_global_layers": register_only_global_layers,
            "inter_frame_only_global_layers": inter_frame_only_global_layers,
            "attention_schedule": model.aggregator.inter_frame_attention_types,
            "register_patch_inter_frame_mode": args.register_patch_inter_frame_mode,
            "register_patch_inter_frame_percent": args.register_patch_inter_frame_percent,
            "register_attention_blocks": num_register_blocks,
            "total_inter_frame_blocks": model.aggregator.depth,
            "timing_repeats": args.timing_repeats,
            "skip_timing": args.skip_timing,
            "require_exclusive_gpu": args.require_exclusive_gpu,
            "exclusive_gpu_index": exclusive_gpu_index,
            "exclusive_gpu_max_other_memory_mib": args.exclusive_gpu_max_other_memory_mib,
        },
        "paper_targets_1b": PAPER_TARGETS,
        "overall": overall,
        "difference_from_paper": {
            key: float(overall[key]) - target for key, target in PAPER_TARGETS.items()
        },
        "per_sequence": per_sequence,
    }
    for key in ("sq_rel", "rmse_m", "rmse_log", "mae_m", "delta_1_25_sq_percent", "delta_1_25_cu_percent", "ate_rmse_m", "are_deg", "rpe_translation_rmse_m", "rpe_rotation_rmse_deg", "rra_30_percent", "rta_30_percent"):
        result["overall"][key] = float(np.mean([row[key] for row in per_sequence]))
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    np.savez_compressed(
        args.output_dir / "pose_errors.npz",
        rotation_error_deg=rotation_errors,
        translation_error_deg=translation_errors,
    )

    print(f"\n{DATASET_NAME} evaluation result:")
    if PAPER_TARGETS:
        print(f"  AUC@3:     {overall['auc_3_percent']:.2f} ({PAPER_TARGETS['auc_3_percent']:.1f})")
        print(f"  AUC@30:    {overall['auc_30_percent']:.2f} ({PAPER_TARGETS['auc_30_percent']:.1f})")
        print(
            f"  delta1.25: {overall['delta_1_25_percent']:.2f} "
            f"({PAPER_TARGETS['delta_1_25_percent']:.1f})"
        )
        print(f"  AbsRel:    {overall['abs_rel']:.4f} ({PAPER_TARGETS['abs_rel']:.3f})")
    else:
        print(f"  AUC@3:     {overall['auc_3_percent']:.2f}")
        print(f"  AUC@30:    {overall['auc_30_percent']:.2f}")
        print(f"  delta1.25: {overall['delta_1_25_percent']:.2f}")
        print(f"  AbsRel:    {overall['abs_rel']:.4f}")
    print(f"Saved reproducible results to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        started = time.perf_counter()
        exit_code = main()
        print(f"Total elapsed: {time.perf_counter() - started:.1f}s")
        sys.exit(exit_code)
    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
