#!/usr/bin/env python3
"""Collect completed 7Scenes SelTR lambda runs into a paper-ready CSV/JSON table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = (
    "reference_cd_m", "fastvggt_cd_m", "auc_3_percent", "auc_5_percent", "auc_15_percent",
    "auc_30_percent", "ate_rmse_m", "are_deg", "rpe_translation_rmse_m", "rpe_rotation_rmse_deg",
    "rra_30_percent", "rta_30_percent",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("lambda_*/metrics.json")):
        payload = json.loads(path.read_text())
        protocol, summary = payload["protocol"], payload["summary"]
        rows.append({"lambda_cost": protocol["lambda_cost"], "stride": protocol["stride"],
                     "max_frames_per_sequence": protocol["max_frames_per_sequence"],
                     **{key: summary.get(key) for key in METRICS}})
    rows.sort(key=lambda row: row["lambda_cost"])
    if not rows:
        raise FileNotFoundError(f"no lambda_*/metrics.json under {args.root}")
    (args.root / "lambda_ablation.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.root / "lambda_ablation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
