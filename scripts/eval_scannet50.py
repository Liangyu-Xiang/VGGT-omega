#!/usr/bin/env python3
"""ScanNet50 evaluation for DenseVGGT, FastVGGT, and SelTR.

The evaluator deliberately keeps the first and last valid camera frames and
uniformly samples the interior.  Geometry follows FastVGGT's ScanNet protocol:
depth is unprojected with the estimated cameras, mapped with the first GT
camera, bbox-scale aligned to the ScanNet mesh, and evaluated after 5 cm voxel
downsampling.  ``cd_m`` is the FastVGGT-compatible bidirectional *sum*;
``overall_m`` is its conventional half, (Acc + Comp) / 2.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


DATA_ROOT = Path("/data_SSD1/mmc_lyxiang/dataset/scannet50_data")
FRAMES_ROOT = DATA_ROOT / "extracted/scannet-frames"
GT_ROOT = DATA_ROOT / "extracted/scannet-dataset"
SCANNET50_SCENES = (
    "scene0000_00", "scene0013_02", "scene0029_01", "scene0042_02", "scene0056_00", "scene0071_00", "scene0084_01", "scene0096_00", "scene0109_00", "scene0121_01",
    "scene0136_01", "scene0150_00", "scene0164_01", "scene0177_01", "scene0194_00", "scene0207_01", "scene0221_01", "scene0238_00", "scene0254_01", "scene0267_00",
    "scene0280_00", "scene0294_02", "scene0309_00", "scene0325_01", "scene0340_01", "scene0353_02", "scene0367_01", "scene0380_02", "scene0395_00", "scene0409_01",
    "scene0421_02", "scene0435_03", "scene0451_01", "scene0466_01", "scene0477_00", "scene0493_01", "scene0509_01", "scene0525_00", "scene0540_02", "scene0555_00",
    "scene0571_00", "scene0582_02", "scene0593_00", "scene0606_01", "scene0619_00", "scene0631_01", "scene0648_00", "scene0663_01", "scene0675_00", "scene0691_00",
)


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def numeric_paths(directory: Path, suffixes: tuple[str, ...]) -> dict[int, Path]:
    result = {}
    for path in directory.iterdir():
        if path.suffix.lower() in suffixes:
            try:
                result[int(path.stem)] = path
            except ValueError:
                pass
    return result


def endpoint_uniform_indices(length: int, requested: int) -> np.ndarray:
    """Endpoint-preserving uniform sampler (unlike stride slicing)."""
    if requested < 2:
        raise ValueError("--num-frames must be at least 2 to retain both endpoints")
    if length < 2:
        raise ValueError("sequence has fewer than two valid RGB/pose/depth frames")
    if requested >= length:
        return np.arange(length, dtype=np.int64)
    indices = np.rint(np.linspace(0, length - 1, requested)).astype(np.int64)
    indices[0], indices[-1] = 0, length - 1
    if len(np.unique(indices)) != requested:
        # This should only occur when requested > length, handled above.
        raise RuntimeError("uniform sampling produced duplicate frame indices")
    return indices


def scene_records(scene_dir: Path, requested: int, require_exact: bool = False) -> tuple[list[dict[str, Any]], np.ndarray]:
    # The official downloaded ScanNet50 frames are flat (00001.jpg/.png/.txt).
    # The fallback preserves compatibility with the raw ScanNet layout.
    if (scene_dir / "color").is_dir():
        images = numeric_paths(scene_dir / "color", (".jpg", ".jpeg", ".png"))
        depths = numeric_paths(scene_dir / "depth", (".png",))
        poses = numeric_paths(scene_dir / "pose", (".txt",))
    else:
        images = numeric_paths(scene_dir, (".jpg", ".jpeg"))
        depths = numeric_paths(scene_dir, (".png",))
        poses = numeric_paths(scene_dir, (".txt",))
    ids, matrices = [], []
    for frame_id in sorted(set(images) & set(depths) & set(poses)):
        pose = np.loadtxt(poses[frame_id], dtype=np.float64)
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            ids.append(frame_id)
            matrices.append(pose)
    if require_exact and len(ids) < requested:
        raise RuntimeError(
            f"{scene_dir.name} has only {len(ids)} valid frames, but {requested} are required. "
            "Point --dataset-root to the fully extracted ScanNet frames; do not use the 300-frame cache."
        )
    selected = endpoint_uniform_indices(len(ids), requested)
    records = [{"id": ids[i], "image": images[ids[i]], "depth": depths[ids[i]]} for i in selected]
    return records, np.stack(matrices, axis=0)[selected]


def load_fastvggt_images(paths: list[Path], target_width: int = 518) -> torch.Tensor:
    """The non-square 518px-width preprocessing used by FastVGGT ScanNet."""
    tensors = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        resized_height = round(height * (target_width / width) / 14) * 14
        image = image.resize((target_width, resized_height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tensors).unsqueeze(0)


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("state_dict", payload))
    return {key.replace("module.", "", 1): value for key, value in state.items()}


def build_model(method: str, checkpoint: Path, device: torch.device) -> VGGT:
    kwargs: dict[str, Any] = {}
    if method == "fastvggt":
        kwargs.update(enable_token_merging=True, token_merging_method="spatial", token_merging_ratio=0.9)
    elif method == "selftr":
        kwargs.update(um_lambda_cost=0.04, um_spatial_radius=2, um_temporal_window=4,
                      um_policy="deltae-adaptive", um_refresh_layers="0,9,21")
    model = VGGT(**kwargs)
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    return model.eval().to(device=device, dtype=torch.bfloat16)


def homogeneous(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[-2:] == (4, 4):
        return matrix
    output = np.zeros((*matrix.shape[:-2], 4, 4), dtype=matrix.dtype)
    output[..., :3, :4] = matrix
    output[..., 3, 3] = 1
    return output


def forward(model: VGGT, images: torch.Tensor, device: torch.device) -> tuple[dict[str, torch.Tensor], float, float, float]:
    if hasattr(model.aggregator, "last_token_merging_stats"):
        model.aggregator.last_token_merging_stats = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    images = images.to(device=device, dtype=torch.bfloat16, non_blocking=True)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(images)
    torch.cuda.synchronize(device)
    latency_s = time.perf_counter() - start
    allocated_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    reserved_gib = torch.cuda.max_memory_reserved(device) / (1024**3)
    return prediction, latency_s, allocated_gib, reserved_gib


def c2w_and_intrinsics(pose_encoding: torch.Tensor, image_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_encoding.float(), image_hw)
    w2c = homogeneous(extrinsic.detach().cpu().numpy()[0])
    return np.linalg.inv(w2c), intrinsic.detach().cpu().numpy()[0]


class Reservoir:
    """Streaming uniform sample, keeping ScanNet-1000 reconstruction bounded."""
    def __init__(self, capacity: int, seed: int = 33):
        self.capacity, self.rng, self.seen = capacity, np.random.RandomState(seed), 0
        self.buffer = np.empty((capacity, 3), dtype=np.float32)
        self.size = 0

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if not len(values):
            return
        start = 0
        if self.size < self.capacity:
            take = min(self.capacity - self.size, len(values))
            self.buffer[self.size:self.size + take] = values[:take]
            self.size += take
            self.seen += take
            start = take
        if start == len(values):
            return
        remaining = values[start:]
        positions = self.seen + np.arange(len(remaining), dtype=np.float64)
        slots = np.floor(self.rng.random_sample(len(remaining)) * (positions + 1)).astype(np.int64)
        accepted = np.flatnonzero(slots < self.capacity)
        if len(accepted):
            # Sequential reservoir updates: latest update to a repeated slot wins.
            reverse_unique = np.unique(slots[accepted][::-1], return_index=True)[1]
            chosen = accepted[::-1][reverse_unique]
            self.buffer[slots[chosen]] = remaining[chosen]
        self.seen += len(remaining)

    def value(self) -> np.ndarray:
        return self.buffer[:self.size].copy()


def predicted_points(depth: np.ndarray, confidence: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray,
                     first_gt_c2w: np.ndarray, threshold: float, sampler: Reservoir) -> None:
    for frame_depth, frame_conf, frame_c2w, frame_k in zip(depth, confidence, c2w, intrinsic):
        valid = np.isfinite(frame_depth) & np.isfinite(frame_conf) & (frame_depth > 0) & (frame_conf >= threshold)
        y, x = np.nonzero(valid)
        if not len(x):
            continue
        z = frame_depth[y, x]
        points_cam = np.stack(((x - frame_k[0, 2]) * z / frame_k[0, 0],
                               (y - frame_k[1, 2]) * z / frame_k[1, 1], z, np.ones_like(z)), axis=1)
        points_local = (frame_c2w @ points_cam.T).T[:, :3]
        points_global = (first_gt_c2w @ np.c_[points_local, np.ones(len(points_local))].T).T[:, :3]
        sampler.add(points_global)


def bbox_scale_align(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_min, pred_max = prediction.min(0), prediction.max(0)
    tgt_min, tgt_max = target.min(0), target.max(0)
    pred_diag = np.linalg.norm(pred_max - pred_min)
    target_diag = np.linalg.norm(tgt_max - tgt_min)
    if pred_diag <= 1e-8 or target_diag <= 1e-8:
        return prediction
    return (prediction - (pred_min + pred_max) / 2) * (target_diag / pred_diag) + (tgt_min + tgt_max) / 2


def normal_consistency(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> np.ndarray:
    source.estimate_normals()
    target.estimate_normals()
    source_normals, target_normals = np.asarray(source.normals), np.asarray(target.normals)
    indices = cKDTree(np.asarray(target.points)).query(np.asarray(source.points), k=1)[1]
    return np.abs(np.sum(source_normals * target_normals[indices], axis=1))


def reconstruction_metrics(prediction: np.ndarray, gt_path: Path, voxel: float, max_distance: float,
                           tau: float) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    target_cloud = o3d.io.read_point_cloud(str(gt_path))
    target = np.asarray(target_cloud.points, dtype=np.float32)
    if len(prediction) < 10 or len(target) < 10:
        raise RuntimeError("too few reconstruction points")
    if len(target) > 100000:
        target = target[np.random.RandomState(33).choice(len(target), 100000, replace=False)]
    prediction = bbox_scale_align(prediction, target)
    pred_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(prediction)).voxel_down_sample(voxel)
    gt_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target)).voxel_down_sample(voxel)
    pred_to_gt = np.clip(np.asarray(pred_cloud.compute_point_cloud_distance(gt_cloud)), 0, max_distance)
    gt_to_pred = np.clip(np.asarray(gt_cloud.compute_point_cloud_distance(pred_cloud)), 0, max_distance)
    precision, recall = float(np.mean(pred_to_gt < tau)), float(np.mean(gt_to_pred < tau))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    nc_a, nc_b = normal_consistency(pred_cloud, gt_cloud), normal_consistency(gt_cloud, pred_cloud)
    acc, comp = float(pred_to_gt.mean()), float(gt_to_pred.mean())
    metrics = {
        "acc_m": acc, "acc_median_m": float(np.median(pred_to_gt)),
        "comp_m": comp, "comp_median_m": float(np.median(gt_to_pred)),
        "nc": float((nc_a.mean() + nc_b.mean()) / 2), "nc_median": float((np.median(nc_a) + np.median(nc_b)) / 2),
        "cd_m": acc + comp, "overall_m": (acc + comp) / 2,
        "f1_at_0.05m": f1, "precision_at_0.05m": precision, "recall_at_0.05m": recall,
        "pred_points_after_voxel": int(len(pred_cloud.points)), "gt_points_after_voxel": int(len(gt_cloud.points)),
    }
    return metrics, np.asarray(pred_cloud.points), np.asarray(gt_cloud.points)


def visual_sample(points: np.ndarray, limit: int = 15000) -> np.ndarray:
    if len(points) <= limit:
        return points
    return points[np.random.RandomState(33).choice(len(points), limit, replace=False)]


def save_reconstruction_visualization(prediction: np.ndarray, target: np.ndarray, output_dir: Path) -> None:
    """Save an inspectable coloured PLY and an overview rendering of metric points."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(prediction))
    gt_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    pred_cloud.paint_uniform_color((0.15, 0.45, 0.95))
    gt_cloud.paint_uniform_color((0.15, 0.75, 0.35))
    o3d.io.write_point_cloud(str(output_dir / "reconstruction_pred_aligned.ply"), pred_cloud)
    o3d.io.write_point_cloud(str(output_dir / "reconstruction_gt.ply"), gt_cloud)
    o3d.io.write_point_cloud(str(output_dir / "reconstruction_overlay.ply"), pred_cloud + gt_cloud)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred, gt = visual_sample(prediction), visual_sample(target)
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=0.25, c="#32b25c", alpha=0.42, label="GT mesh")
    axis.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=0.25, c="#2673e8", alpha=0.42, label="estimated reconstruction")
    axis.set_xlabel("X (m)"); axis.set_ylabel("Y (m)"); axis.set_zlabel("Z (m)")
    axis.set_title("ScanNet reconstruction: predicted vs. GT")
    axis.legend(markerscale=10)
    figure.tight_layout()
    figure.savefig(output_dir / "reconstruction.png", dpi=180)
    plt.close(figure)


def irls_scale_shift(pred: np.ndarray, gt: np.ndarray, iterations: int = 8) -> tuple[float, float]:
    if len(pred) < 2:
        return 1.0, 0.0
    design = np.stack((pred, np.ones_like(pred)), 1)
    weights = np.ones_like(pred)
    solution = np.array([1.0, 0.0])
    for _ in range(iterations):
        solution = np.linalg.lstsq(design * weights[:, None], gt * weights, rcond=None)[0]
        residual = np.abs(design @ solution - gt)
        scale = max(np.median(residual) * 1.4826, 1e-6)
        weights = 1 / np.maximum(1, residual / (1.345 * scale))
    return float(solution[0]), float(solution[1])


def depth_metrics(prediction: np.ndarray, records: list[dict[str, Any]], max_depth: float = 10.0) -> dict[str, float]:
    gt_all, pred_all = [], []
    for depth, record in zip(prediction, records):
        gt = np.asarray(Image.open(record["depth"]), dtype=np.float32) / 1000.0
        resized = F.interpolate(torch.from_numpy(depth)[None, None], size=gt.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
        valid = np.isfinite(resized) & (resized > 0) & np.isfinite(gt) & (gt > 0) & (gt < max_depth)
        gt_all.append(gt[valid]); pred_all.append(resized[valid])
    gt, pred = np.concatenate(gt_all), np.concatenate(pred_all)
    scale, shift = irls_scale_shift(pred, gt)
    pred = np.clip(scale * pred + shift, 1e-6, max_depth)
    abs_error = np.abs(pred - gt)
    ratio = np.maximum(pred / gt, gt / pred)
    return {"absrel": float(np.mean(abs_error / gt)), "delta_lt_1.25": float(np.mean(ratio < 1.25)),
            "l1_mae_m": float(abs_error.mean()), "rmse_m": float(np.sqrt(np.mean((pred - gt) ** 2))),
            "log_rmse": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
            "sqrel": float(np.mean((pred - gt) ** 2 / gt)), "scale": scale, "shift_m": shift,
            "valid_pixels": int(len(gt))}


def rotation_error(rotation: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cosine))


def auc(errors: np.ndarray, threshold: float) -> float:
    # Official VGGT AUC convention: mean CDF over one-degree bins.
    histogram, _ = np.histogram(errors, bins=np.arange(int(threshold) + 1))
    return float(np.mean(np.cumsum(histogram, dtype=np.float64) / len(errors)))


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_mu, tgt_mu = source.mean(0), target.mean(0)
    src, tgt = source - src_mu, target - tgt_mu
    u, singular, vt = np.linalg.svd((tgt.T @ src) / len(source))
    diagonal = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        diagonal[-1] = -1
    rotation = u @ np.diag(diagonal) @ vt
    scale = float(np.sum(singular * diagonal) / max(np.mean(np.sum(src ** 2, axis=1)), 1e-12))
    return scale, rotation, tgt_mu - scale * rotation @ src_mu


def pose_metrics(predicted_c2w: np.ndarray, gt_c2w_world: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    gt = np.linalg.inv(gt_c2w_world[0]) @ gt_c2w_world
    scale, rotation, translation = umeyama(predicted_c2w[:, :3, 3], gt[:, :3, 3])
    aligned = predicted_c2w.copy()
    aligned[:, :3, 3] = (scale * (rotation @ predicted_c2w[:, :3, 3].T)).T + translation
    aligned[:, :3, :3] = rotation @ predicted_c2w[:, :3, :3]
    n = len(gt)
    left, right = np.triu_indices(n, 1)
    pred_w2c, gt_w2c = np.linalg.inv(aligned), np.linalg.inv(gt)
    pred_rel = pred_w2c[left] @ aligned[right]
    gt_rel = gt_w2c[left] @ gt[right]
    rot = rotation_error(gt_rel[:, :3, :3] @ np.swapaxes(pred_rel[:, :3, :3], 1, 2))
    gt_t, pred_t = gt_rel[:, :3, 3], pred_rel[:, :3, 3]
    dots = np.abs(np.sum(gt_t * pred_t, 1) / np.maximum(np.linalg.norm(gt_t, axis=1) * np.linalg.norm(pred_t, axis=1), 1e-12))
    trans = np.degrees(np.arccos(np.clip(dots, -1, 1)))
    pose_error = np.maximum(rot, trans)
    ape = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=1)
    if n > 1:
        rel_p, rel_g = pred_w2c[:-1] @ aligned[1:], gt_w2c[:-1] @ gt[1:]
        rpe_rot = rotation_error(rel_g[:, :3, :3] @ np.swapaxes(rel_p[:, :3, :3], 1, 2))
        rpe_trans = np.linalg.norm(rel_g[:, :3, 3] - rel_p[:, :3, 3], axis=1)
    else:
        rpe_rot, rpe_trans = np.zeros(1), np.zeros(1)
    values = {f"auc_at_{t}_percent": auc(pose_error, t) * 100 for t in (3, 5, 15, 20, 30)}
    values.update({"rra_at_5_percent": float(np.mean(rot < 5) * 100), "rra_at_30_percent": float(np.mean(rot < 30) * 100),
                   "rta_at_5_percent": float(np.mean(trans < 5) * 100), "rta_at_30_percent": float(np.mean(trans < 30) * 100),
                   "ate_rmse_m": float(np.sqrt(np.mean(ape ** 2))), "are_mean_deg": float(rotation_error(gt[:, :3, :3] @ np.swapaxes(aligned[:, :3, :3], 1, 2)).mean()),
                   "rpe_rot_rmse_deg": float(np.sqrt(np.mean(rpe_rot ** 2))), "rpe_trans_rmse_m": float(np.sqrt(np.mean(rpe_trans ** 2))),
                   "ape_mean_m": float(ape.mean()), "ape_median_m": float(np.median(ape)), "pair_count": int(len(pose_error)),
                   "sim3_scale": scale})
    return values, aligned, gt


def save_trajectory_visualization(prediction: np.ndarray, target: np.ndarray, output_path: Path) -> None:
    """FastVGGT-style XZ trajectory view, colour-coding Sim(3)-aligned APE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_xyz, gt_xyz = prediction[:, :3, 3], target[:, :3, 3]
    ape = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(gt_xyz[:, 0], gt_xyz[:, 2], "--", color="0.35", linewidth=1.5, label="GT")
    segments = np.stack((pred_xyz[:-1, (0, 2)], pred_xyz[1:, (0, 2)]), axis=1)
    trace = LineCollection(segments, cmap="viridis", linewidth=2.0)
    trace.set_array((ape[:-1] + ape[1:]) / 2)
    axis.add_collection(trace)
    axis.scatter(pred_xyz[0, 0], pred_xyz[0, 2], marker="o", c="#2673e8", s=28, label="estimated (APE colour)")
    axis.scatter(pred_xyz[-1, 0], pred_xyz[-1, 2], marker="x", c="#2673e8", s=36)
    axis.autoscale(); axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X (m)"); axis.set_ylabel("Z (m)")
    axis.set_title("Camera trajectory after Sim(3) alignment")
    axis.legend(loc="best")
    colorbar = figure.colorbar(trace, ax=axis, pad=0.02)
    colorbar.set_label("absolute position error (m)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def token_retention(model: VGGT, method: str) -> dict[str, Any]:
    if method == "densevggt":
        return {"policy": "fixed_once", "retention_percent": 100.0}
    stats = list(getattr(model.aggregator, "last_token_merging_stats", []))
    if method == "fastvggt":
        value = next((item.get("full_attention_token_ratio") for item in stats if "full_attention_token_ratio" in item), None)
        return {"policy": "fixed_once", "retention_percent": None if value is None else 100 * float(value)}
    refresh = sorted(getattr(model.aggregator, "um_refresh_layers", {0, 9, 21}))
    stages = []
    for block in refresh:
        item = next((row for row in stats if row.get("block") == block and "full_attention_token_ratio" in row), None)
        if item is not None:
            stages.append({"stage": len(stages) + 1, "block": int(block), "retention_percent": 100 * float(item["full_attention_token_ratio"])})
    return {"policy": "selftr_stagewise", "stages": stages,
            "mean_retention_percent": float(np.mean([x["retention_percent"] for x in stages])) if stages else None}


def numeric_mean(items: list[dict[str, Any]]) -> dict[str, float]:
    keys = set().union(*(item.keys() for item in items))
    return {key: float(np.mean([item[key] for item in items if isinstance(item.get(key), (float, int, np.number))]))
            for key in keys if any(isinstance(item.get(key), (float, int, np.number)) for item in items)}


def evaluate_scene(model: VGGT, scene: str, records: list[dict[str, Any]], gt_c2w: np.ndarray, gt_root: Path,
                   args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    images = load_fastvggt_images([record["image"] for record in records])
    prediction, latency_s, allocated_gib, reserved_gib = forward(model, images, device)
    h, w = images.shape[-2:]
    pred_c2w, intrinsic = c2w_and_intrinsics(prediction["pose_enc"], (h, w))
    depth = prediction["depth"].float().detach().cpu().numpy()[0, ..., 0]
    confidence = prediction["depth_conf"].float().detach().cpu().numpy()[0]
    sampler = Reservoir(args.point_sample_limit)
    predicted_points(depth, confidence, pred_c2w, intrinsic, gt_c2w[0], args.depth_confidence_threshold, sampler)
    gt_ply = gt_root / scene / f"{scene}_vh_clean_2.ply"
    if not gt_ply.exists():
        raise FileNotFoundError(gt_ply)
    geometry, pred_cloud, gt_cloud = reconstruction_metrics(sampler.value(), gt_ply, args.voxel_size, args.chamfer_max_distance, args.tau)
    pose, aligned_c2w, local_gt_c2w = pose_metrics(pred_c2w, gt_c2w)
    depth_result = depth_metrics(depth, records)
    visualization = None
    if args.save_visualizations:
        visualization_dir = args.output_dir / scene / "visualization"
        save_reconstruction_visualization(pred_cloud, gt_cloud, visualization_dir)
        save_trajectory_visualization(aligned_c2w, local_gt_c2w, visualization_dir / "trajectory_xz.png")
        visualization = {
            "reconstruction_overlay_ply": str(visualization_dir / "reconstruction_overlay.ply"),
            "reconstruction_png": str(visualization_dir / "reconstruction.png"),
            "trajectory_png": str(visualization_dir / "trajectory_xz.png"),
        }
    result = {"scene": scene, "frames": len(records), "frame_ids": [item["id"] for item in records],
              "pose": pose, "reconstruction": geometry, "depth": depth_result,
              "visualization": visualization,
              "efficiency": {"latency_s": latency_s, "inference_time_s": latency_s, "fps": len(records) / latency_s,
                             "peak_vram_allocated_gib": allocated_gib, "peak_vram_reserved_gib": reserved_gib,
                             "token_retention": token_retention(model, args.method),
                             "sampled_reconstruction_points": int(len(sampler.value()))}}
    del prediction
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", choices=("densevggt", "fastvggt", "selftr"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT,
                        help="ScanNet50 data root; its extracted/scannet-frames directory is used automatically")
    parser.add_argument("--gt-root", type=Path, default=None,
                        help="override GT mesh root; defaults to DATASET_ROOT/extracted/scannet-dataset")
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-exact-frames", action="store_true",
                        help="fail rather than silently evaluate a short sequence with fewer than --num-frames")
    parser.add_argument("--depth-confidence-threshold", type=float, default=1.0)
    parser.add_argument("--point-sample-limit", type=int, default=100000)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--chamfer-max-distance", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--save-visualizations", action="store_true",
                        help="save coloured reconstruction PLY/PNG and FastVGGT-style aligned trajectory PNG")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ScanNet50 evaluation requires CUDA")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    frames_root = args.dataset_root / "extracted/scannet-frames"
    if not frames_root.is_dir():
        # Allow passing the flat frame directory for compatibility with older invocations.
        frames_root = args.dataset_root
    gt_root = args.gt_root or (args.dataset_root / "extracted/scannet-dataset")
    if not frames_root.is_dir() or not gt_root.is_dir():
        raise FileNotFoundError(f"expected ScanNet frames at {frames_root} and GT meshes at {gt_root}")
    available = {path.name for path in frames_root.iterdir() if path.is_dir()}
    scenes = [scene for scene in SCANNET50_SCENES if scene in available]
    if len(scenes) != len(SCANNET50_SCENES):
        missing = sorted(set(SCANNET50_SCENES) - available)
        raise FileNotFoundError(f"missing {len(missing)} ScanNet50 frame directories, e.g. {missing[:5]}")
    if args.scenes:
        unknown = set(args.scenes) - set(SCANNET50_SCENES)
        if unknown:
            raise ValueError(f"unknown ScanNet scenes: {sorted(unknown)}")
        scenes = args.scenes
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    device = torch.device(args.device)
    model = build_model(args.method, args.checkpoint, device)
    protocol = {"sampler": "endpoint-preserving uniform: first and last valid RGB/pose/depth frames + uniform interior",
                "image_preprocessing": "FastVGGT ScanNet: width=518, aspect-preserving height rounded to a multiple of 14",
                "geometry": "FastVGGT coordinate recovery + bbox scale alignment; deterministic 100k streaming sample; 0.05m voxel",
                "cd_m": "FastVGGT-compatible clipped bidirectional sum Acc+Comp (clip=0.5m)",
                "overall_m": "(Acc+Comp)/2", "tau_m": args.tau,
                "pose_note": "AUC@330 is reported as AUC@30; RPW-trans is reported as RPE-trans.",
                "visualization": "optional coloured predicted/GT point clouds and FastVGGT-style Sim(3)-aligned XZ trajectory",
                "token_note": "fixed policies log one retention value; SelTR logs all three U-M refresh stages."}
    results, failures = [], []
    for number, scene in enumerate(scenes, 1):
        scene_output = args.output_dir / scene / "metrics.json"
        if args.resume and scene_output.exists():
            results.append(json.loads(scene_output.read_text()))
            print(f"[{number}/{len(scenes)}] {scene}: resumed", flush=True)
            continue
        print(f"[{number}/{len(scenes)}] {scene}: evaluating", flush=True)
        try:
            records, gt_c2w = scene_records(frames_root / scene, args.num_frames, args.require_exact_frames)
            result = evaluate_scene(model, scene, records, gt_c2w, gt_root, args, device)
            write_json(scene_output, result)
            results.append(result)
            print(f"[{number}/{len(scenes)}] {scene}: done ({result['efficiency']['latency_s']:.2f}s)", flush=True)
        except Exception as exc:
            torch.cuda.empty_cache()
            failure = {"scene": scene, "error": repr(exc), "traceback": traceback.format_exc()}
            failures.append(failure)
            write_json(args.output_dir / scene / "failure.json", failure)
            print(f"[{number}/{len(scenes)}] {scene}: FAILED: {exc}", flush=True)
    summary = {"method": args.method, "num_frames_requested": args.num_frames, "scene_count": len(results),
               "failed_scene_count": len(failures), "protocol": protocol, "scenes": results, "failures": failures,
               "mean_pose": numeric_mean([item["pose"] for item in results]),
               "mean_reconstruction": numeric_mean([item["reconstruction"] for item in results]),
               "mean_depth": numeric_mean([item["depth"] for item in results]),
               "mean_efficiency": numeric_mean([item["efficiency"] for item in results])}
    token_values = [item["efficiency"]["token_retention"] for item in results]
    if args.method == "selftr":
        stages = []
        for stage in range(3):
            values = [x["stages"][stage]["retention_percent"] for x in token_values if len(x.get("stages", [])) > stage]
            stages.append({"stage": stage + 1, "mean_retention_percent": float(np.mean(values)) if values else None})
        summary["token_retention"] = {"policy": "selftr_stagewise", "stages": stages}
    else:
        values = [x["retention_percent"] for x in token_values if x.get("retention_percent") is not None]
        summary["token_retention"] = {"policy": "fixed_once", "retention_percent": float(np.mean(values)) if values else None}
    write_json(args.output_dir / "metrics.json", summary)
    print(json.dumps({"output": str(args.output_dir / 'metrics.json'), "completed": len(results), "failed": len(failures)}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
