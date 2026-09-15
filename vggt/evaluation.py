"""Shared original-VGGT inference and Omega-compatible evaluation helpers."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

INTER_FRAME_ATTENTION_MODES = (
    "partial",
    "global",
    "register",
    "alternate1",
    "alternate2",
    "partial_plus17",
    "partial_plus3",
    "partial_pairs",
    "partial_plus3_17",
    "partial_2_10_17_20",
)
AGGREGATOR_DEPTH = 24


SINTEL_SEQUENCES = [
    "alley_2",
    "ambush_4",
    "ambush_5",
    "ambush_6",
    "cave_2",
    "cave_4",
    "market_2",
    "market_5",
    "market_6",
    "shaman_3",
    "sleeping_1",
    "sleeping_2",
    "temple_2",
    "temple_3",
]
SINTEL_TAG_FLOAT = 202021.25


def write_csv(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sequence_names(dataset_root: str | Path, requested: Sequence[str] | None = None) -> list[str]:
    final_root = Path(dataset_root) / "final"
    requested = list(requested) if requested else SINTEL_SEQUENCES
    missing = [seq for seq in requested if not (final_root / seq).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing Sintel sequences below {final_root}: {missing}")
    return requested


def sequence_images(dataset_root: str | Path, seq: str) -> list[str]:
    paths = sorted((Path(dataset_root) / "final" / seq).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No input images found for Sintel sequence {seq}")
    return [str(path) for path in paths]


def sequence_cameras(dataset_root: str | Path, seq: str) -> list[str]:
    paths = sorted((Path(dataset_root) / "camdata_left" / seq).glob("*.cam"))
    if not paths:
        raise FileNotFoundError(f"No camera annotations found for Sintel sequence {seq}")
    return [str(path) for path in paths]


def read_sintel_depth(filename: str | Path) -> np.ndarray:
    with Path(filename).open("rb") as handle:
        tag = np.fromfile(handle, dtype=np.float32, count=1)[0]
        if tag != SINTEL_TAG_FLOAT:
            raise ValueError(f"Invalid Sintel depth tag in {filename}: {tag}")
        width = int(np.fromfile(handle, dtype=np.int32, count=1)[0])
        height = int(np.fromfile(handle, dtype=np.int32, count=1)[0])
        return np.fromfile(handle, dtype=np.float32).reshape(height, width)


def read_bonn_depth(filename: str | Path) -> np.ndarray:
    depth_png = np.asarray(Image.open(filename))
    depth = depth_png.astype(np.float32) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth


def read_sintel_camera(filename: str | Path) -> np.ndarray:
    """Read Sintel world-to-camera annotation and return a 4x4 c2w pose."""
    with Path(filename).open("rb") as handle:
        tag = np.fromfile(handle, dtype=np.float32, count=1)[0]
        if tag != SINTEL_TAG_FLOAT:
            raise ValueError(f"Invalid Sintel camera tag in {filename}: {tag}")
        np.fromfile(handle, dtype=np.float64, count=9)
        w2c = np.fromfile(handle, dtype=np.float64, count=12).reshape(3, 4)
    w2c = np.vstack([w2c, [0.0, 0.0, 0.0, 1.0]])
    return np.linalg.inv(w2c)


def _crop_depth_like_image(depth: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if depth.shape[:2] != (height, width):
        raise ValueError(f"Image shape {(height, width)} does not match depth shape {depth.shape[:2]}")
    aspect_ratio = height / max(width, 1)
    if aspect_ratio < 0.5:
        crop_width = min(width, max(1, int(round(height / 0.5))))
        left = max((width - crop_width) // 2, 0)
        return depth[:, left : left + crop_width]
    if aspect_ratio > 2.0:
        crop_height = min(height, max(1, int(round(width * 2.0))))
        top = max((height - crop_height) // 2, 0)
        return depth[top : top + crop_height]
    return depth


def preprocess_depth(
    image_filename: str | Path,
    depth: np.ndarray,
    resolution: tuple[int, int],
) -> np.ndarray:
    """Apply VGGT-Omega's center crop and resize geometry to a GT depth map."""
    with Image.open(image_filename) as image:
        depth = _crop_depth_like_image(depth, image.size)
    return cv2.resize(depth, resolution, interpolation=cv2.INTER_NEAREST).astype(np.float32)


def preprocess_sintel_depth(
    image_filename: str | Path, depth_filename: str | Path, resolution: tuple[int, int]
) -> np.ndarray:
    return preprocess_depth(image_filename, read_sintel_depth(depth_filename), resolution)


def preprocess_bonn_depth(
    image_filename: str | Path, depth_filename: str | Path, resolution: tuple[int, int]
) -> np.ndarray:
    return preprocess_depth(image_filename, read_bonn_depth(depth_filename), resolution)


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    enable_camera: bool = True,
    enable_depth: bool = True,
    inter_frame_attention: str = "partial",
    register_patch_sample_tokens: int = 0,
    register_patch_sample_ratio: float = 0.0,
    register_patch_sample_mode: str = "uniform",
    register_patch_merge_sources: bool = False,
    register_patch_merge_protect_first_frame: bool = False,
    enable_token_merging: bool = False,
    token_merging_start: int = 0,
    token_merging_ratio: float = 0.9,
    token_merging_layer_ratios: str = "",
    token_merging_method: str = "spatial",
    token_merging_flashvid_alpha: float = 0.7,
    token_merging_flashvid_expansion: float = 1.25,
    token_merging_flashvid_pool_stride: int = 2,
    token_merging_flashvid_tstm_threshold: float = 0.8,
    token_merging_frame_restore_layer: int = 16,
    token_merging_frame_alpha: float = 0.9,
    token_merging_frame_segment_threshold: float = 0.8,
    token_merging_frame_merge_threshold: float = 0.8,
    token_merging_frame_max_window: int = 6,
    token_merging_frame_pool_stride: int = 2,
    token_merging_frame_multi_max_group_size: int = 2,
    token_merging_frame_multi_pair_threshold: float = 0.95,
    token_merging_frame_multi_span_threshold: float = 0.93,
    token_merging_frame_group_strategy: str = "local",
    token_merging_frame_protect_period: int = 0,
    token_merging_frame_protect_prefix: int = 0,
    token_merging_frame_anchor_count: int = 4,
    token_merging_frame_anchor_selection: str = "uniform",
    token_merging_frame_adaptive_boundary_z: float = 2.5,
    token_merging_frame_adaptive_medoid_z: float = 1.5,
    token_merging_frame_patch_fusion_quantile: float = 0.75,
    token_merging_frame_special_cross_attention: bool = False,
    token_merging_frame_special_cross_attention_alpha: float = 0.1,
    token_merging_segment_bank_pair_threshold: float = 0.986,
    token_merging_segment_bank_span_threshold: float = 0.948,
    token_merging_segment_bank_max_group_size: int = 4,
    omega_accelerator: str = "none",
    sparse_vggt_sparse_ratio: float | None = 0.5,
    sparse_vggt_cdf_threshold: float | None = None,
    sparse_vggt_pool_mode: str = "avg",
    da_vggt_max_frames: int = 0,
    da_vggt_sampling_method: str = "fl_maxmin",
    da_vggt_n_anchors: int = 1,
    da_vggt_dino_batch_size: int = 256,
    da_vggt_lambda_div: float = 0.0,
    dynamic_fastvggt_schedule: str = "all",
    skip_global_attention_blocks: str = "",
    skip_inter_frame_attention_blocks: str = "",
    frame_only_inter_frame_blocks: str = "",
    register_only_blocks: str = "",
    um_lambda_cost: float | None = None,
    um_spatial_radius: int = 2,
    um_temporal_window: int = 4,
    um_refresh_layers: str = "0,9,21",
    um_representative_mode: str = "mean",
    um_policy: str = "deltae-adaptive",
    um_target_full_attention_ratio: float | None = None,
    model_bfloat16: bool = True,
) -> VGGT:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if inter_frame_attention != "global":
        raise ValueError("The released VGGT model supports full global attention only.")
    model = VGGT(
            enable_camera=enable_camera,
            enable_depth=enable_depth,
            enable_point=False,
            enable_track=False,
            enable_token_merging=enable_token_merging,
            token_merging_start=token_merging_start,
            token_merging_ratio=token_merging_ratio,
            token_merging_layer_ratios=token_merging_layer_ratios,
            token_merging_method=token_merging_method,
            token_merging_frame_restore_layer=token_merging_frame_restore_layer,
            token_merging_frame_alpha=token_merging_frame_alpha,
            token_merging_frame_segment_threshold=token_merging_frame_segment_threshold,
            token_merging_frame_merge_threshold=token_merging_frame_merge_threshold,
            token_merging_frame_max_window=token_merging_frame_max_window,
            token_merging_frame_pool_stride=token_merging_frame_pool_stride,
            token_merging_frame_multi_max_group_size=token_merging_frame_multi_max_group_size,
            token_merging_frame_multi_pair_threshold=token_merging_frame_multi_pair_threshold,
            token_merging_frame_multi_span_threshold=token_merging_frame_multi_span_threshold,
            skip_global_attention_blocks=skip_global_attention_blocks,
            um_lambda_cost=um_lambda_cost,
            um_spatial_radius=um_spatial_radius,
            um_temporal_window=um_temporal_window,
            um_refresh_layers=um_refresh_layers,
            um_representative_mode=um_representative_mode,
            um_policy=um_policy,
            um_target_full_attention_ratio=um_target_full_attention_ratio,
        ).eval()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if not key.startswith(("point_head.", "track_head."))]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model = model.to(device)
    if model_bfloat16:
        model = model.to(dtype=torch.bfloat16)
        model.explicit_bfloat16_inference = True
    return model


def register_attention_indices(inter_frame_attention: str) -> list[int] | None:
    if inter_frame_attention == "partial":
        return None
    if inter_frame_attention == "global":
        return []
    if inter_frame_attention == "register":
        return list(range(AGGREGATOR_DEPTH))
    if inter_frame_attention == "partial_plus17":
        return [2, 6, 9, 14, 17, 20]
    if inter_frame_attention == "partial_plus3":
        return [2, 3, 6, 9, 14, 20]
    if inter_frame_attention == "partial_pairs":
        return [2, 3, 6, 7, 9, 10, 14, 15, 20, 21]
    if inter_frame_attention == "partial_plus3_17":
        return [2, 3, 6, 9, 14, 17, 20]
    if inter_frame_attention == "partial_2_10_17_20":
        return list(range(2, 11)) + list(range(17, 21))
    if inter_frame_attention == "alternate1":
        return list(range(1, AGGREGATOR_DEPTH, 2))
    if inter_frame_attention == "alternate2":
        return list(range(2, AGGREGATOR_DEPTH, 3))
    raise ValueError(
        f"Unknown inter-frame attention mode {inter_frame_attention!r}; "
        f"expected one of {INTER_FRAME_ATTENTION_MODES}"
    )


def parse_block_indices(value: str) -> list[int]:
    """Parse comma-separated 0-based block indices/ranges such as ``1-7,9``."""
    indices: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending block range {item!r}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(item))
    invalid = sorted(index for index in indices if not 0 <= index < AGGREGATOR_DEPTH)
    if invalid:
        raise ValueError(f"register-only block indices must be in [0, {AGGREGATOR_DEPTH - 1}], got {invalid}")
    return sorted(indices)


def windows(paths: Sequence[str], window_size: int) -> Iterable[Sequence[str]]:
    if window_size <= 0 or window_size >= len(paths):
        yield paths
        return
    for start in range(0, len(paths), window_size):
        yield paths[start : start + window_size]


def infer_sequence(
    image_paths: Sequence[str],
    model: VGGT,
    device: torch.device,
    *,
    window_size: int = 4,
    image_resolution: int = 512,
    input_mode: str = "balanced",
    use_amp: bool = True,
) -> tuple[
    float,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    tuple[int, int],
    dict[str, float | int | str | None],
]:
    all_depths = []
    all_poses = []
    all_confidences = []
    model.aggregator.last_frame_merge_stats = []
    model.aggregator.last_token_merging_stats = []
    frame_merge_stats = []
    frame_special_cross_attention_stats = []
    token_merging_stats = []
    segment_patch_bank_stats = []
    sparse_vggt_stats = []
    da_vggt_stats = []
    output_resolution = None
    elapsed = 0.0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        memory_before_bytes = torch.cuda.memory_allocated(device)
        reserved_before_bytes = torch.cuda.memory_reserved(device)
    else:
        memory_before_bytes = 0
        reserved_before_bytes = 0

    for image_window in windows(image_paths, window_size):
        preprocess_mode = "pad" if input_mode == "balanced" else "crop"
        images = load_and_preprocess_images(list(image_window), mode=preprocess_mode).to(device, non_blocking=True)
        output_resolution = (images.shape[-1], images.shape[-2])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp and device.type == "cuda"):
                predictions = model(images)
        frame_merge_stats.extend(model.aggregator.last_frame_merge_stats)
        token_merging_stats.extend(model.aggregator.last_token_merging_stats)
        model.aggregator.last_frame_merge_stats = []
        model.aggregator.last_token_merging_stats = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - start

        if "depth" in predictions:
            all_depths.append(predictions["depth"][0, ..., 0].detach().cpu())
            all_confidences.append(predictions["depth_conf"][0].detach().cpu())
        if "pose_enc" in predictions:
            w2c, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], predictions["images"].shape[-2:])
            # PyTorch does not implement matrix inversion for bfloat16. The
            # model may emit pose encodings under CUDA autocast, so promote
            # the homogeneous transform before converting w2c to c2w.
            w2c = w2c.float()
            bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=w2c.device, dtype=w2c.dtype)
            bottom = bottom.view(1, 1, 1, 4).expand(w2c.shape[0], w2c.shape[1], -1, -1)
            c2w = torch.linalg.inv(torch.cat([w2c, bottom], dim=-2))
            all_poses.append(c2w[0].detach().cpu())

    if output_resolution is None:
        raise ValueError("At least one input image is required for inference.")
    depths = torch.cat(all_depths) if all_depths else None
    poses = torch.cat(all_poses) if all_poses else None
    confidences = torch.cat(all_confidences) if all_confidences else None
    if device.type == "cuda":
        peak_memory_allocated_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_reserved_bytes = torch.cuda.max_memory_reserved(device)
        memory_after_bytes = torch.cuda.memory_allocated(device)
        reserved_after_bytes = torch.cuda.memory_reserved(device)
    else:
        peak_memory_allocated_bytes = 0
        peak_memory_reserved_bytes = 0
        memory_after_bytes = 0
        reserved_after_bytes = 0
    frames = len(image_paths)
    speed_metrics = {
        "omega_accelerator": getattr(model, "omega_accelerator", "none"),
        "time": elapsed,
        "frames": frames,
        "fps": frames / elapsed if elapsed > 0 else None,
        "seconds_per_frame": elapsed / frames if frames > 0 else None,
        "peak_memory_allocated_bytes": peak_memory_allocated_bytes,
        "peak_memory_allocated_gb": peak_memory_allocated_bytes / 1024**3,
        "peak_memory_reserved_bytes": peak_memory_reserved_bytes,
        "peak_memory_reserved_gb": peak_memory_reserved_bytes / 1024**3,
        "memory_before_allocated_bytes": memory_before_bytes,
        "memory_before_allocated_gb": memory_before_bytes / 1024**3,
        "memory_after_allocated_bytes": memory_after_bytes,
        "memory_after_allocated_gb": memory_after_bytes / 1024**3,
        "reserved_before_bytes": reserved_before_bytes,
        "reserved_before_gb": reserved_before_bytes / 1024**3,
        "reserved_after_bytes": reserved_after_bytes,
        "reserved_after_gb": reserved_after_bytes / 1024**3,
        "device": str(device),
    }
    if frame_merge_stats:
        active_means = [float(stat["active_frames_mean"]) for stat in frame_merge_stats]
        retention_ratios = [float(stat["retention_ratio_mean"]) for stat in frame_merge_stats]
        merge_ratios = [float(stat["merge_ratio_mean"]) for stat in frame_merge_stats]
        anchor_counts = [int(stat.get("anchor_count", 0)) for stat in frame_merge_stats]
        anchor_selections = [stat.get("anchor_selection") for stat in frame_merge_stats if stat.get("anchor_selection")]
        speed_metrics.update(
            {
                "frame_merge_events": len(frame_merge_stats),
                "frame_merge_original_frames": int(frame_merge_stats[0]["original_frames"]),
                "frame_merge_active_frames_min": min(int(stat["active_frames_min"]) for stat in frame_merge_stats),
                "frame_merge_active_frames_mean": sum(active_means) / len(active_means),
                "frame_merge_active_frames_max": max(int(stat["active_frames_max"]) for stat in frame_merge_stats),
                "frame_merge_retention_ratio_mean": sum(retention_ratios) / len(retention_ratios),
                "frame_merge_merge_ratio_mean": sum(merge_ratios) / len(merge_ratios),
                "frame_merge_anchor_count": int(round(sum(anchor_counts) / len(anchor_counts))),
                "frame_merge_anchor_selection": anchor_selections[0] if anchor_selections else None,
                "frame_merge_anchor_frames": frame_merge_stats[0].get("anchor_frames", []),
                "frame_merge_stats": frame_merge_stats,
            }
        )
    if frame_special_cross_attention_stats:
        speed_metrics.update(
            {
                "frame_special_cross_attention_events": len(frame_special_cross_attention_stats),
                "frame_special_cross_attention_alpha": float(
                    frame_special_cross_attention_stats[0]["alpha"]
                ),
                "frame_special_cross_attention_stats": frame_special_cross_attention_stats,
            }
        )
    if token_merging_stats:
        active_tokens = [float(stat["active_tokens"]) for stat in token_merging_stats]
        token_ratios = [float(stat["full_attention_token_ratio"]) for stat in token_merging_stats]
        merged_ratios = [float(stat["merged_away_token_ratio"]) for stat in token_merging_stats]
        active_over_frame_merged_ratios = [
            float(stat.get("active_over_frame_merged_token_ratio", stat["full_attention_token_ratio"]))
            for stat in token_merging_stats
        ]
        active_over_frame_original_ratios = [
            float(stat.get("active_over_frame_original_token_ratio", stat["full_attention_token_ratio"]))
            for stat in token_merging_stats
        ]
        dynamic_patch_tokens = sum(int(stat.get("dynamic_patch_tokens", 0)) for stat in token_merging_stats)
        static_patch_tokens = sum(int(stat.get("static_patch_tokens", 0)) for stat in token_merging_stats)
        dynamic_merged_tokens = sum(
            int(stat.get("dynamic_merged_source_tokens", 0)) for stat in token_merging_stats
        )
        static_merged_tokens = sum(
            int(stat.get("static_merged_source_tokens", 0)) for stat in token_merging_stats
        )
        cross_type_merged_tokens = sum(
            int(stat.get("cross_type_merged_source_tokens", 0)) for stat in token_merging_stats
        )
        speed_metrics.update(
            {
                "token_merging_events": len(token_merging_stats),
                "token_merging_original_tokens": int(token_merging_stats[0]["original_tokens"]),
                "token_merging_active_tokens_mean": sum(active_tokens) / len(active_tokens),
                "token_merging_full_attention_token_ratio_mean": sum(token_ratios) / len(token_ratios),
                "token_merging_merged_away_token_ratio_mean": sum(merged_ratios) / len(merged_ratios),
                "token_merging_active_over_frame_merged_token_ratio_mean": sum(
                    active_over_frame_merged_ratios
                )
                / len(active_over_frame_merged_ratios),
                "token_merging_active_over_frame_original_token_ratio_mean": sum(
                    active_over_frame_original_ratios
                )
                / len(active_over_frame_original_ratios),
                "dynamic_fastvggt_dynamic_patch_tokens": dynamic_patch_tokens,
                "dynamic_fastvggt_static_patch_tokens": static_patch_tokens,
                "dynamic_fastvggt_dynamic_merged_tokens": dynamic_merged_tokens,
                "dynamic_fastvggt_static_merged_tokens": static_merged_tokens,
                "dynamic_fastvggt_dynamic_merge_ratio": float(
                    dynamic_merged_tokens / dynamic_patch_tokens if dynamic_patch_tokens else 0.0
                ),
                "dynamic_fastvggt_static_merge_ratio": float(
                    static_merged_tokens / static_patch_tokens if static_patch_tokens else 0.0
                ),
                "dynamic_fastvggt_patch_merge_ratio": float(
                    (dynamic_merged_tokens + static_merged_tokens)
                    / (dynamic_patch_tokens + static_patch_tokens)
                    if dynamic_patch_tokens + static_patch_tokens
                    else 0.0
                ),
                "dynamic_fastvggt_cross_type_merged_tokens": cross_type_merged_tokens,
                "token_merging_stats": token_merging_stats,
            }
        )
    if segment_patch_bank_stats:
        compressed_frames = [int(stat["compressed_frames"]) for stat in segment_patch_bank_stats]
        compressed_segments = [int(stat["compressed_segments"]) for stat in segment_patch_bank_stats]
        speed_metrics.update(
            {
                "segment_patch_bank_events": len(segment_patch_bank_stats),
                "segment_patch_bank_compressed_frames_mean": sum(compressed_frames) / len(compressed_frames),
                "segment_patch_bank_compressed_segments_mean": sum(compressed_segments) / len(compressed_segments),
                "segment_patch_bank_stats": segment_patch_bank_stats,
            }
        )
    if sparse_vggt_stats:
        sparsities = [float(stat["sparsity"]) for stat in sparse_vggt_stats]
        speed_metrics.update(
            {
                "sparse_vggt_events": len(sparse_vggt_stats),
                "sparse_vggt_sparsity_mean": sum(sparsities) / len(sparsities),
                "sparse_vggt_sparse_ratio": sparse_vggt_stats[0].get("sparse_ratio"),
                "sparse_vggt_cdf_threshold": sparse_vggt_stats[0].get("cdf_threshold"),
                "sparse_vggt_pool_mode": sparse_vggt_stats[0].get("pool_mode"),
                "sparse_vggt_stats": sparse_vggt_stats,
            }
        )
    if da_vggt_stats:
        speed_metrics.update(
            {
                "da_vggt_events": len(da_vggt_stats),
                "da_vggt_num_chunks_mean": sum(float(stat["da_vggt_num_chunks"]) for stat in da_vggt_stats)
                / len(da_vggt_stats),
                "da_vggt_chunk_size": da_vggt_stats[0].get("da_vggt_chunk_size"),
                "da_vggt_sampling_method": da_vggt_stats[0].get("da_vggt_sampling_method"),
                "da_vggt_stats": da_vggt_stats,
            }
        )
    return elapsed, depths, poses, confidences, output_resolution, speed_metrics


def _patch_importance_from_depth_conf(
    depth_conf: torch.Tensor,
    images: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    if depth_conf.ndim != 4:
        raise ValueError(f"Expected depth_conf shape [B, F, H, W], got {tuple(depth_conf.shape)}")
    if images.ndim == 4:
        images = images.unsqueeze(0)
    batch_size, num_frames, _, height, width = images.shape
    patch_h = height // patch_size
    patch_w = width // patch_size
    if depth_conf.shape[:2] != (batch_size, num_frames):
        raise ValueError(
            "depth_conf batch/frame shape does not match images: "
            f"{tuple(depth_conf.shape[:2])} vs {(batch_size, num_frames)}"
        )
    uncertainty = -depth_conf.float()
    uncertainty = uncertainty[:, :, : patch_h * patch_size, : patch_w * patch_size]
    uncertainty = uncertainty.view(batch_size, num_frames, patch_h, patch_size, patch_w, patch_size)
    return uncertainty.mean(dim=(3, 5)).reshape(batch_size, num_frames, patch_h * patch_w)


def save_depth_preview(depth: np.ndarray, filename: str | Path) -> None:
    valid = np.isfinite(depth)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [1, 99])
        scaled = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        scaled = np.zeros_like(depth)
    colored = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(filename), colored)
