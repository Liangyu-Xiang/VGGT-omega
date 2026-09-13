#!/usr/bin/env python3
"""Official evaluation entry point for the SelfTR paper implementation.

The dataset adapters deliberately reuse one metric implementation so ScanNet,
7 Scenes, and NRGBD create the same ``metrics.json``, sampled-frame manifest,
and pose-error artifact structure.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASET_MODULES = {
    "7scenes": "eval_7scenes_paper",
    "scannet": "eval_scannet_paper",
    "nrgbd": "eval_nrgbd_paper",
}


def _contains_option(arguments: list[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in arguments)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run the official SelfTR evaluator for one supported dataset.",
        add_help=False,
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_MODULES), required=True)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("-h", "--help", action="store_true")
    args, evaluator_args = parser.parse_known_args()
    return args, evaluator_args


def main() -> int:
    args, evaluator_args = parse_args()
    if args.help:
        evaluator_args.append("--help")
    defaults = {
        "--acceleration-method": "selftr",
        "--frame-fusion-mode": "selftr",
        "--merge-ratio": "0",
        "--num-frames": "300",
        "--sampling-unit": "sequence",
        "--sampling-strategy": "uniform",
        "--seed": "42",
        "--image-resolution": "512",
        "--resize-mode": "max_size",
        "--timing-repeats": "3",
        "--frame-fusion-recompute-layers": "0,10,17",
        "--frame-fusion-lambda-cost": "0.04",
        "--frame-fusion-min-keep-ratio": "0.05",
        "--frame-fusion-temporal-window": "4",
        "--frame-fusion-spatial-radius": "2",
        "--frame-fusion-attention-variant": "representative",
    }
    for option, value in defaults.items():
        if not _contains_option(evaluator_args, option):
            evaluator_args.extend((option, value))
    if args.method_name is not None and not _contains_option(evaluator_args, "--method-name"):
        evaluator_args.extend(("--method-name", args.method_name))

    module = importlib.import_module(DATASET_MODULES[args.dataset])
    previous_argv = sys.argv
    try:
        sys.argv = [str(Path(module.__file__).resolve()), *evaluator_args]
        return int(module.main())
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
