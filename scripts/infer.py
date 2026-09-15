#!/usr/bin/env python3
"""Run original VGGT, FastVGGT, or SelfTR forward inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


SELFTR_DEFAULTS = {
    "um_lambda_cost": 0.04,
    "um_spatial_radius": 2,
    "um_temporal_window": 4,
    "um_refresh_layers": "0,9,21",
    "um_representative_mode": "mean",
    "um_policy": "deltae-adaptive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one forward pass with the original VGGT architecture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", choices=("vggt", "fastvggt", "selftr"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True, metavar="IMAGE")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--merge-ratio", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model", state))
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint must contain a model state dictionary.")
    if state and all(name.startswith("module.") for name in state):
        state = {name.removeprefix("module."): value for name, value in state.items()}
    return state


def model_kwargs(args: argparse.Namespace) -> dict[str, object]:
    if args.method == "vggt":
        return {}
    if args.method == "fastvggt":
        if not 0.0 <= args.merge_ratio <= 1.0:
            raise ValueError("--merge-ratio must be between 0 and 1")
        return {
            "enable_token_merging": True,
            "token_merging_method": "spatial",
            "token_merging_ratio": args.merge_ratio,
        }
    return dict(SELFTR_DEFAULTS)


def load_model(args: argparse.Namespace) -> VGGT:
    if args.device.split(":", 1)[0] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Original VGGT, FastVGGT, and SelfTR inference require CUDA.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    missing = [str(path) for path in args.images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input image files do not exist: {', '.join(missing)}")

    model = VGGT(**model_kwargs(args))
    incompatible = model.load_state_dict(load_state_dict(args.checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys[:5]}, "
            f"unexpected={incompatible.unexpected_keys[:5]}"
        )
    return model.to(args.device).eval()


def cpu_prediction(prediction: dict[str, object]) -> dict[str, object]:
    return {
        name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for name, value in prediction.items()
    }


def main() -> int:
    args = parse_args()
    model = load_model(args)
    images = load_and_preprocess_images([str(path) for path in args.images], mode=args.input_mode).to(args.device)
    with torch.inference_mode():
        prediction = model(images)

    print(f"method={args.method}")
    for name, value in prediction.items():
        if isinstance(value, torch.Tensor):
            print(f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cpu_prediction(prediction), args.output)
        print(f"saved={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
