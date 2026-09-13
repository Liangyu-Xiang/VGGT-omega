#!/usr/bin/env python3
"""SelfTR NRGBD evaluation using the standard 7 Scenes metric protocol."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import eval_7scenes_paper as base

DEFAULT_ROOT = Path(os.environ.get("SELFTR_NRGBD_ROOT", "data/nrgbd"))

def select_sequence_dirs(data_root: Path, requested: list[str] | None) -> list[Path]:
    dirs = sorted(path for path in data_root.iterdir() if (path / "images").is_dir() and (path / "depth").is_dir() and (path / "poses.txt").is_file())
    if requested:
        lookup = {path.name: path for path in dirs}
        missing = sorted(set(requested) - set(lookup))
        if missing:
            raise ValueError(f"Unknown NRGBD sequence(s): {', '.join(missing)}")
        return [lookup[name] for name in requested]
    return dirs

def load_frame_records(sequence_dir: Path) -> list[base.FrameRecord]:
    poses = np.loadtxt(sequence_dir / "poses.txt", dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 4 or poses.shape[0] % 4:
        raise ValueError(f"Invalid NRGBD poses: {sequence_dir / 'poses.txt'}")
    poses = poses.reshape(-1, 4, 4)
    poses[:, :3, 1:3] *= -1.0  # NRGBD OpenGL -> OpenCV camera convention.
    records = []
    for image in sorted((sequence_dir / "images").glob("img*.png"), key=lambda x: int(x.stem.removeprefix("img"))):
        index = int(image.stem.removeprefix("img"))
        depth = sequence_dir / "depth" / f"depth{index}.png"
        if depth.is_file() and index < len(poses) and np.isfinite(poses[index]).all():
            records.append(base.FrameRecord(index, image, depth, sequence_dir / "poses.txt", poses[index]))
    if not records:
        raise FileNotFoundError(f"No valid NRGBD image/depth/pose triplets in {sequence_dir}")
    return records

def read_resized_depth(path: Path, height: int, width: int) -> np.ndarray:
    raw = np.asarray(Image.open(path), dtype=np.uint16)
    raw = np.asarray(Image.fromarray(raw).resize((width, height), Image.Resampling.NEAREST), dtype=np.float32)
    return raw / 1000.0

base.select_sequence_dirs = select_sequence_dirs
base.load_frame_records = load_frame_records
base.read_resized_depth = read_resized_depth
base.DATASET_NAME = "NRGBD"
base.DATASET_SPLIT = "all valid sequence directories under --data-root"
base.PAPER_TARGETS = {}

if __name__ == "__main__":
    if "--data-root" not in sys.argv:
        sys.argv.extend(["--data-root", str(DEFAULT_ROOT)])
    raise SystemExit(base.main())
