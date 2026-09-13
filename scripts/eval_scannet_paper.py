#!/usr/bin/env python3
"""SelfTR ScanNet evaluation with ScanNet RGB calibration."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_7scenes_paper as base
from selftr.evaluation import depth_to_world_points, evaluate_pi3_geometry, scaled_intrinsics

DEFAULT_ROOT = Path(os.environ.get("SELFTR_SCANNET_ROOT", "data/scannet"))

def select_sequence_dirs(data_root: Path, requested: list[str] | None) -> list[Path]:
    dirs = sorted(
        path for path in data_root.iterdir()
        if (path / "color").is_dir()
        and (path / "depth").is_dir()
        and (path / "pose").is_dir()
        and (path / "intrinsic" / "intrinsic_color.txt").is_file()
    )
    if requested:
        lookup = {path.name: path for path in dirs}
        missing = sorted(set(requested) - set(lookup))
        if missing:
            raise ValueError(f"Unknown ScanNet sequence(s): {', '.join(missing)}")
        return [lookup[name] for name in requested]
    return dirs

def load_frame_records(sequence_dir: Path) -> list[base.FrameRecord]:
    records = []
    for rgb in sorted((sequence_dir / "color").glob("*.jpg"), key=lambda p: int(p.stem)):
        depth = sequence_dir / "depth" / f"{rgb.stem}.png"
        pose_path = sequence_dir / "pose" / f"{rgb.stem}.txt"
        if not depth.is_file() or not pose_path.is_file():
            continue
        pose = np.loadtxt(pose_path, dtype=np.float64)
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            records.append(base.FrameRecord(int(rgb.stem), rgb, depth, pose_path, pose))
    if not records:
        raise FileNotFoundError(f"No valid ScanNet RGB/depth/pose triplets in {sequence_dir}")
    return records

def read_resized_depth(path: Path, height: int, width: int) -> np.ndarray:
    raw = np.asarray(Image.open(path), dtype=np.uint16)
    return np.asarray(Image.fromarray(raw).resize((width, height), Image.Resampling.NEAREST), dtype=np.float32) / 1000.0

def geometry_from_prediction(predicted_depth, pred_w2c, records, min_depth, max_depth):
    height, width = predicted_depth.shape[1:]
    gt_depth = np.stack([read_resized_depth(record.depth_path, height, width) for record in records])
    intrinsic = np.loadtxt(records[0].rgb_path.parent.parent / "intrinsic" / "intrinsic_color.txt", dtype=np.float64)[:3, :3]
    intrinsics = scaled_intrinsics(intrinsic, (968, 1296), (height, width), len(records))
    pred_c2w = np.linalg.inv(pred_w2c)
    gt_c2w = np.stack([record.c2w for record in records])
    valid = np.isfinite(predicted_depth) & (predicted_depth > 0) & np.isfinite(gt_depth) & (gt_depth > min_depth) & (gt_depth < max_depth)
    return evaluate_pi3_geometry(depth_to_world_points(predicted_depth, pred_c2w, intrinsics), depth_to_world_points(gt_depth, gt_c2w, intrinsics), valid)

base.select_sequence_dirs = select_sequence_dirs
base.load_frame_records = load_frame_records
base.read_resized_depth = read_resized_depth
base.geometry_from_prediction = geometry_from_prediction
base.DATASET_NAME = "ScanNet"
base.DATASET_SPLIT = "all valid scene directories under --data-root"
base.PAPER_TARGETS = {}

if __name__ == "__main__":
    if "--data-root" not in sys.argv:
        sys.argv.extend(["--data-root", str(DEFAULT_ROOT)])
    raise SystemExit(base.main())
