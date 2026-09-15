#!/usr/bin/env python3
"""Evaluate original VGGT, FastVGGT, or SelfTR on the 7Scenes test split.

One forward pass is shared by two reconstruction protocols:

* ``reference_cd_m`` reproduces the 7Scenes geometry procedure used by
  ``multiframe_merging_vggt_benchmark``: corresponding depth points, Sim(3),
  point-to-point ICP, then bidirectional nearest-neighbour distance.
* ``fastvggt_cd_m`` reproduces the 7Scenes path in FastVGGT's
  ``eval/eval_7andN.py``: central 224px crop, model point maps expressed in
  camera 1, ICP, and FastVGGT accuracy/completion.  CD is their table form
  ``(accuracy + completion) / 2``.

By default every source frame at the requested stride is evaluated. Pass
``--num-frames`` only when a capped protocol is explicitly intended.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


DEFAULT_DATASET_ROOT = Path("/data_SSD1/mmc_lyxiang/3D/benchmark_stride_eval/datasets/7scenes")
SELFTR_KWARGS = {
    "um_lambda_cost": 0.04,
    "um_spatial_radius": 2,
    "um_temporal_window": 4,
    "um_refresh_layers": "0,9,21",
    "um_representative_mode": "mean",
    "um_policy": "deltae-adaptive",
}
INTRINSIC_7SCENES = np.array(
    [[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", choices=("vggt", "fastvggt", "selftr"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--num-frames", type=int, default=None,
                        help="optional cap after stride sampling; omit for all available frames")
    parser.add_argument("--merge-ratio", type=float, default=0.9)
    parser.add_argument("--lambda-cost", type=float, default=0.04,
                        help="SelTR U-M cost coefficient; used only with --method selftr")
    parser.add_argument("--sequences", nargs="*", default=None, metavar="SCENE/SEQ")
    parser.add_argument("--fastvggt-sample-points", type=int, default=999_999)
    parser.add_argument("--fastvggt-random-seed", type=int, default=33)
    return parser.parse_args()


def checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model", state))
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint must contain a model state dictionary")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def build_model(args: argparse.Namespace) -> VGGT:
    kwargs: dict[str, object] = {}
    if args.method == "fastvggt":
        if not 0.0 <= args.merge_ratio <= 1.0:
            raise ValueError("--merge-ratio must be in [0, 1]")
        kwargs = {
            "enable_token_merging": True,
            "token_merging_method": "spatial",
            "token_merging_ratio": args.merge_ratio,
        }
    elif args.method == "selftr":
        kwargs = dict(SELFTR_KWARGS)
        if args.lambda_cost <= 0:
            raise ValueError("--lambda-cost must be positive")
        kwargs["um_lambda_cost"] = args.lambda_cost
    model = VGGT(**kwargs)
    result = model.load_state_dict(checkpoint_state(args.checkpoint), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {result}")
    return model.to(args.device).eval()


def test_sequences(root: Path, requested: list[str] | None) -> list[str]:
    available: list[str] = []
    for scene in sorted(path for path in root.iterdir() if path.is_dir()):
        split = scene / "TestSplit.txt"
        if not split.is_file():
            continue
        for line in split.read_text().splitlines():
            number = "".join(filter(str.isdigit, line))
            seq = f"{scene.name}/seq-{int(number):02d}"
            if (root / seq).is_dir():
                available.append(seq)
    if requested is None:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(f"Not official/readable 7Scenes test sequences: {missing}")
    return requested


def sequence_inputs(root: Path, sequence: str, stride: int, limit: int | None) -> tuple[list[Path], list[int]]:
    frames = sorted((root / sequence).glob("*.color.png"))
    if not frames:
        raise FileNotFoundError(f"No colour frames found for {sequence}")
    stop = len(frames) if limit is None else min(len(frames), stride * limit)
    indices = list(range(0, stop, stride))
    return [frames[index] for index in indices], indices


def gt_depth_paths(images: list[Path]) -> list[Path]:
    paths = [path.with_name(path.name.replace(".color.png", ".depth.proj.png")) for path in images]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing projected depth maps, first: {missing[0]}")
    return paths


def read_gt_depth(path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(path))
    depth = raw.astype(np.float32) / 1000.0
    depth[(raw == 0) | (raw == 65535)] = -1.0
    return depth


def c2w_from_pose_encoding(pose_enc: torch.Tensor, image_hw: tuple[int, int]) -> np.ndarray:
    w2c, _ = pose_encoding_to_extri_intri(pose_enc.float(), image_hw)
    bottom = torch.zeros((*w2c.shape[:-2], 1, 4), dtype=w2c.dtype, device=w2c.device)
    bottom[..., 0, 3] = 1.0
    return torch.linalg.inv(torch.cat((w2c, bottom), dim=-2)[0]).float().cpu().numpy()


def forward(model: VGGT, images: list[Path], device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = load_and_preprocess_images([str(path) for path in images], mode="crop").to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
        prediction = model(batch)
    depth = prediction["depth"][0, ..., 0].float().cpu().numpy()
    points = prediction["world_points"][0].float().cpu().numpy()
    c2w = c2w_from_pose_encoding(prediction["pose_enc"], tuple(batch.shape[-2:]))
    del prediction, batch
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return depth, points, c2w


def scaled_intrinsics(frames: int, target_hw: tuple[int, int]) -> np.ndarray:
    height, width = target_hw
    output = np.repeat(INTRINSIC_7SCENES[None], frames, axis=0)
    output[:, 0, :] *= width / 640.0
    output[:, 1, :] *= height / 480.0
    output[:, 2, 2] = 1.0
    return output


def depth_to_world(depth: np.ndarray, c2w: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    frames, height, width = depth.shape
    ys, xs = np.mgrid[:height, :width]
    output = np.empty((frames, height, width, 3), dtype=np.float32)
    for index in range(frames):
        k, z = intrinsics[index], depth[index]
        camera = np.stack(((xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z), axis=-1)
        output[index] = camera @ c2w[index, :3, :3].T + c2w[index, :3, 3]
    return output


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(axis=1, keepdims=True), target.mean(axis=1, keepdims=True)
    variance = np.square(source - source_mean).sum(axis=0).mean()
    covariance = ((target - target_mean) @ (source - source_mean).T) / source.shape[1]
    u, singular, vh = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vh
    scale = float(np.trace(np.diag(singular) @ sign) / max(variance, 1e-12))
    return scale, rotation, target_mean - scale * rotation @ source_mean


def rotation_error_degrees(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    delta = np.asarray(reference) @ np.swapaxes(np.asarray(prediction), -1, -2)
    cosine = np.clip((np.trace(delta, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def pose_auc(errors: list[float], threshold: int) -> float:
    if not errors:
        return 0.0
    bins = np.arange(threshold + 1)
    hist, _ = np.histogram(np.asarray(errors, dtype=np.float64), bins=bins)
    return float(np.mean(np.cumsum(hist.astype(np.float64) / len(errors))) * 100.0)


def relative_pose_metrics(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict[str, object]:
    """Official VGGT relative-pose AUC: all pairs, 1-degree histogram bins."""
    pred_w2c, gt_w2c = np.linalg.inv(pred_c2w), np.linalg.inv(gt_c2w)
    errors: list[float] = []
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for left in range(len(pred_w2c)):
        for right in range(left + 1, len(pred_w2c)):
            pred_relative = pred_w2c[left] @ np.linalg.inv(pred_w2c[right])
            gt_relative = gt_w2c[left] @ np.linalg.inv(gt_w2c[right])
            rotation_error = float(rotation_error_degrees(gt_relative[:3, :3][None], pred_relative[:3, :3][None])[0])
            pred_translation, gt_translation = pred_relative[:3, 3], gt_relative[:3, 3]
            denominator = np.linalg.norm(pred_translation) * np.linalg.norm(gt_translation)
            if denominator <= 1e-15:
                translation_error = 1e6
            else:
                cosine = np.clip(abs(np.dot(pred_translation, gt_translation)) / denominator, 0.0, 1.0)
                translation_error = float(np.degrees(np.arccos(cosine)))
            errors.append(max(rotation_error, translation_error))
            rotation_errors.append(rotation_error)
            translation_errors.append(translation_error)
    return {
        "auc_3_percent": pose_auc(errors, 3),
        "auc_5_percent": pose_auc(errors, 5),
        "auc_10_percent": pose_auc(errors, 10),
        "auc_15_percent": pose_auc(errors, 15),
        "auc_30_percent": pose_auc(errors, 30),
        "mean_pose_error_deg": float(np.mean(errors)),
        "mean_rotation_error_deg": float(np.mean(rotation_errors)),
        "mean_translation_error_deg": float(np.mean(translation_errors)),
        "_pose_errors": errors,
        "_rotation_errors": rotation_errors,
        "_translation_errors": translation_errors,
    }


def trajectory_metrics(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict[str, float]:
    """Reference repository's Sim(3)-aligned trajectory metrics."""
    pred, gt = pred_c2w.copy(), gt_c2w.copy()
    scale, rotation, translation = umeyama(pred[:, :3, 3].T, gt[:, :3, 3].T)
    pred[:, :3, 3] = scale * (pred[:, :3, 3] @ rotation.T) + translation.reshape(1, 3)
    pred[:, :3, :3] = rotation[None] @ pred[:, :3, :3]
    absolute_rotation = rotation_error_degrees(gt[:, :3, :3], pred[:, :3, :3])
    ate = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=1)
    pred_w2c, gt_w2c = np.linalg.inv(pred), np.linalg.inv(gt)
    pair_rotation: list[float] = []
    pair_translation: list[float] = []
    rpe_rotation: list[float] = []
    rpe_translation: list[float] = []
    for left in range(len(pred_w2c) - 1):
        for right in range(left + 1, len(pred_w2c)):
            pred_relative = pred_w2c[left] @ np.linalg.inv(pred_w2c[right])
            gt_relative = gt_w2c[left] @ np.linalg.inv(gt_w2c[right])
            rotation_error = float(rotation_error_degrees(gt_relative[:3, :3][None], pred_relative[:3, :3][None])[0])
            a, b = gt_relative[:3, 3], pred_relative[:3, 3]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            translation_error = 180.0 if denominator <= 1e-15 else float(
                np.degrees(np.arccos(np.clip(abs(np.dot(a, b)) / denominator, 0.0, 1.0)))
            )
            pair_rotation.append(rotation_error)
            pair_translation.append(translation_error)
            if right == left + 1:
                rpe_rotation.append(rotation_error)
                rpe_translation.append(float(np.linalg.norm(gt_relative[:3, 3] - pred_relative[:3, 3])))
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(np.square(ate)))),
        "are_deg": float(np.mean(absolute_rotation)),
        "rpe_translation_rmse_m": float(np.sqrt(np.mean(np.square(rpe_translation)))),
        "rpe_rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rpe_rotation)))),
        "rra_30_percent": float(100.0 * np.mean(np.asarray(pair_rotation) < 30.0)),
        "rta_30_percent": float(100.0 * np.mean(np.asarray(pair_translation) < 30.0)),
    }


def reference_metrics(pred_depth: np.ndarray, pred_c2w: np.ndarray, gt_depth: np.ndarray, gt_c2w: np.ndarray) -> dict[str, float | int]:
    """Exact geometric core of the referenced VGGT benchmark's 7Scenes metric."""
    frames, height, width = pred_depth.shape
    gt_resized = np.stack([cv2.resize(item, (width, height), interpolation=cv2.INTER_NEAREST) for item in gt_depth])
    intrinsics = scaled_intrinsics(frames, (height, width))
    pred_map = depth_to_world(pred_depth, pred_c2w, intrinsics)
    gt_map = depth_to_world(gt_resized, gt_c2w, intrinsics)
    valid = np.isfinite(pred_depth) & (pred_depth > 0) & np.isfinite(gt_resized) & (gt_resized > 0) & (gt_resized < 10.0)
    valid_ids = np.flatnonzero(valid.reshape(-1))
    if len(valid_ids) < 3:
        raise ValueError("Reference CD needs at least three valid depth points")
    selected = valid_ids[np.linspace(0, len(valid_ids) - 1, min(100_000, len(valid_ids)), dtype=np.int64)]
    pred = pred_map.reshape(-1, 3)[selected].astype(np.float64, copy=False)
    gt = gt_map.reshape(-1, 3)[selected].astype(np.float64, copy=False)
    finite = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    pred, gt = pred[finite], gt[finite]
    scale, rotation, translation = umeyama(pred.T, gt.T)
    pred = scale * (pred @ rotation.T) + translation.reshape(1, 3)
    pred_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    gt_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))
    transform = o3d.pipelines.registration.registration_icp(
        pred_cloud, gt_cloud, 0.1, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint()
    ).transformation
    pred_cloud.transform(transform)
    pred = np.asarray(pred_cloud.points)
    gt = np.asarray(gt_cloud.points)
    acc = cKDTree(gt).query(pred, workers=-1)[0]
    comp = cKDTree(pred).query(gt, workers=-1)[0]
    return {
        "reference_acc_m": float(acc.mean()),
        "reference_comp_m": float(comp.mean()),
        "reference_cd_m": float((acc.mean() + comp.mean()) / 2),
        "reference_points": int(len(pred)),
        "reference_sim3_scale": float(scale),
    }


def fastvggt_metrics(pred_points: np.ndarray, gt_depth: np.ndarray, gt_c2w: np.ndarray, *, sample_points: int, seed: int) -> dict[str, float | int]:
    """FastVGGT eval_7andN's 7Scenes accuracy/completion reconstruction path."""
    frames, height, width, _ = pred_points.shape
    gt_resized = np.stack([cv2.resize(item, (width, height), interpolation=cv2.INTER_NEAREST) for item in gt_depth])
    gt_world = depth_to_world(gt_resized, gt_c2w, scaled_intrinsics(frames, (height, width)))
    first_w2c = np.linalg.inv(gt_c2w[0])
    gt_h = np.concatenate((gt_world, np.ones((*gt_world.shape[:-1], 1), dtype=gt_world.dtype)), axis=-1)
    gt_camera1 = (gt_h @ first_w2c.T)[..., :3]
    top, left = (height - 224) // 2, (width - 224) // 2
    crop = np.s_[..., top : top + 224, left : left + 224, :]
    valid = (gt_resized > 0) & (gt_resized < 10.0) & np.isfinite(pred_points).all(axis=-1)
    valid = valid[:, top : top + 224, left : left + 224]
    pred = pred_points[crop][valid].reshape(-1, 3).astype(np.float64, copy=False)
    gt = gt_camera1[crop][valid].reshape(-1, 3).astype(np.float64, copy=False)
    if len(pred) < 3:
        raise ValueError("FastVGGT CD needs at least three valid depth points")
    rng = np.random.default_rng(seed)
    if len(pred) > sample_points:
        pred = pred[rng.choice(len(pred), sample_points, replace=False)]
    if len(gt) > sample_points:
        gt = gt[rng.choice(len(gt), sample_points, replace=False)]
    pred_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    gt_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))
    transform = o3d.pipelines.registration.registration_icp(
        pred_cloud, gt_cloud, 0.1, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint()
    ).transformation
    pred_cloud.transform(transform)
    pred = np.asarray(pred_cloud.points)
    gt = np.asarray(gt_cloud.points)
    acc = cKDTree(gt).query(pred, workers=-1)[0]
    comp = cKDTree(pred).query(gt, workers=-1)[0]
    return {
        "fastvggt_acc_m": float(acc.mean()),
        "fastvggt_comp_m": float(comp.mean()),
        "fastvggt_cd_m": float((acc.mean() + comp.mean()) / 2),
        "fastvggt_points_pred": int(len(pred)),
        "fastvggt_points_gt": int(len(gt)),
    }


def main() -> int:
    args = parse_args()
    if args.stride <= 0 or (args.num_frames is not None and args.num_frames <= 0):
        raise ValueError("--stride must be positive and --num-frames, when supplied, must be positive")
    if not args.checkpoint.is_file() or not args.dataset_root.is_dir():
        raise FileNotFoundError("Checkpoint or dataset root does not exist")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("This formal VGGT evaluation requires CUDA")
    torch.cuda.set_device(args.device)
    model = build_model(args)
    rows: list[dict[str, object]] = []
    all_pose_errors: list[float] = []
    all_rotation_errors: list[float] = []
    all_translation_errors: list[float] = []
    for sequence in test_sequences(args.dataset_root, args.sequences):
        images, source_indices = sequence_inputs(args.dataset_root, sequence, args.stride, args.num_frames)
        print(f"{sequence}: forwarding {len(images)} frames (source stride {args.stride})", flush=True)
        pred_depth, pred_points, pred_c2w = forward(model, images, args.device)
        gt_depth = np.stack([read_gt_depth(path) for path in gt_depth_paths(images)])
        gt_c2w = np.stack([np.loadtxt(path.with_name(path.name.replace(".color.png", ".pose.txt")), dtype=np.float64) for path in images])
        row: dict[str, object] = {"sequence": sequence, "frames": len(images), "source_frame_indices": source_indices}
        row.update(reference_metrics(pred_depth, pred_c2w, gt_depth, gt_c2w))
        row.update(fastvggt_metrics(pred_points, gt_depth, gt_c2w, sample_points=args.fastvggt_sample_points, seed=args.fastvggt_random_seed))
        pose = relative_pose_metrics(pred_c2w, gt_c2w)
        all_pose_errors.extend(pose.pop("_pose_errors"))
        all_rotation_errors.extend(pose.pop("_rotation_errors"))
        all_translation_errors.extend(pose.pop("_translation_errors"))
        row.update(pose)
        row.update(trajectory_metrics(pred_c2w, gt_c2w))
        rows.append(row)
        print(
            f"{sequence}: reference_CD={row['reference_cd_m']:.6f} m, "
            f"FastVGGT_CD={row['fastvggt_cd_m']:.6f} m, AUC@3={row['auc_3_percent']:.2f}",
            flush=True,
        )
    summary = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in (
            "reference_acc_m", "reference_comp_m", "reference_cd_m",
            "fastvggt_acc_m", "fastvggt_comp_m", "fastvggt_cd_m",
            "ate_rmse_m", "are_deg", "rpe_translation_rmse_m", "rpe_rotation_rmse_deg",
            "rra_30_percent", "rta_30_percent",
        )
    }
    summary.update({
        "auc_3_percent": pose_auc(all_pose_errors, 3),
        "auc_5_percent": pose_auc(all_pose_errors, 5),
        "auc_10_percent": pose_auc(all_pose_errors, 10),
        "auc_15_percent": pose_auc(all_pose_errors, 15),
        "auc_30_percent": pose_auc(all_pose_errors, 30),
        "mean_pose_error_deg": float(np.mean(all_pose_errors)),
        "mean_rotation_error_deg": float(np.mean(all_rotation_errors)),
        "mean_translation_error_deg": float(np.mean(all_translation_errors)),
    })
    payload = {
        "protocol": {
            "dataset": "7Scenes official test split", "method": args.method, "stride": args.stride,
            "max_frames_per_sequence": args.num_frames,
            "selection": "all source indices at stride when uncapped; otherwise [0, stride * num_frames) at stride",
            "lambda_cost": args.lambda_cost if args.method == "selftr" else None,
            "reference_cd": "corresponding points + Sim(3) + ICP(0.1m) + bidirectional KD-tree mean / 2; 100000 deterministic points",
            "fastvggt_cd": "FastVGGT eval_7andN: central 224 crop + ICP(0.1m) + accuracy/completion mean / 2; independent deterministic samples capped at 999999",
            "fastvggt_random_seed": args.fastvggt_random_seed,
            "pose": "official VGGT relative-pose AUC over all frame pairs; Sim(3)-aligned ATE/ARE/RPE/RRA/RTA follow the reference repository",
        },
        "summary": summary,
        "per_sequence": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
