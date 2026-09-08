#!/usr/bin/env python3
"""Summarize the full 7Scenes VGGT ablation into CSV and Markdown."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COLUMNS = [
    "experiment",
    "num_sequences",
    "frames_per_sequence",
    "latency_ms",
    "fps",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "speedup_vs_baseline",
    "AUC@3",
    "AUC@30",
    "Acc_m",
    "Comp_m",
    "F1@10cm_percent",
    "CD_m",
    "active_token_ratio_percent",
    "selected_compression_percent",
    "sampling_fallback_sequences",
    "metrics_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def load_row(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    overall = data.get("overall", {})
    protocol = data.get("protocol", {})
    per_sequence = data.get("per_sequence", [])
    fallback_count = sum(
        1
        for row in per_sequence
        if row.get("sequence", "").startswith("stairs/")
    )
    compression_values = [
        float(row["frame_fusion_selected_compression_ratio"]) * 100.0
        for row in per_sequence
        if row.get("frame_fusion_selected_compression_ratio") is not None
    ]
    baseline_key = "baseline"
    row = {
        "experiment": path.parent.name,
        "num_sequences": protocol.get("num_sequences", len(per_sequence)),
        "frames_per_sequence": protocol.get("num_frames_per_sequence"),
        "latency_ms": overall.get("model_latency_ms_mean"),
        "fps": overall.get("fps"),
        "peak_allocated_gib": overall.get("peak_allocated_gib_max"),
        "peak_reserved_gib": overall.get("peak_reserved_gib_max"),
        "speedup_vs_baseline": None,
        "AUC@3": overall.get("auc_3_percent"),
        "AUC@30": overall.get("auc_30_percent"),
        "Acc_m": overall.get("acc_mean_m"),
        "Comp_m": overall.get("comp_mean_m"),
        "F1@10cm_percent": overall.get("f1_10cm_percent", overall.get("f1_at_threshold_percent")),
        "CD_m": overall.get("chamfer_mean_m"),
        "active_token_ratio_percent": overall.get("active_token_ratio_percent"),
        "selected_compression_percent": (
            sum(compression_values) / len(compression_values)
            if compression_values
            else None
        ),
        "sampling_fallback_sequences": fallback_count,
        "metrics_path": str(path),
        "_baseline_key": baseline_key,
    }
    return row


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_latency = next(
        (
            float(row["latency_ms"])
            for row in rows
            if row["experiment"] == "baseline" and row["latency_ms"] is not None
        ),
        None,
    )
    for row in rows:
        if baseline_latency and row["latency_ms"] is not None:
            row["speedup_vs_baseline"] = baseline_latency / float(row["latency_ms"])
        row.pop("_baseline_key", None)

    csv_path = output_dir / "ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    display = [
        "experiment",
        "latency_ms",
        "speedup_vs_baseline",
        "peak_allocated_gib",
        "AUC@3",
        "AUC@30",
        "Acc_m",
        "Comp_m",
        "F1@10cm_percent",
        "CD_m",
        "active_token_ratio_percent",
        "sampling_fallback_sequences",
    ]
    lines = [
        "# 7Scenes VGGT ablation summary",
        "",
        "Point-cloud F1 is the harmonic mean of precision and recall at 0.10 m after the same Sim(3)+ICP alignment used for Acc/Comp/CD.",
        "",
        "| " + " | ".join(display) + " |",
        "| " + " | ".join("---" for _ in display) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                fmt(row.get(column), digits=2 if column in {"speedup_vs_baseline", "F1@10cm_percent"} else 3)
                for column in display
            )
            + " |"
        )
    md_path = output_dir / "ablation_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)
    print("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.result_root / "summary"
    paths = sorted(
        path
        for path in args.result_root.iterdir()
        if path.is_dir() and (path / "metrics.json").is_file()
    )
    if not paths:
        raise SystemExit(f"No completed experiment metrics found under {args.result_root}")
    write_outputs([load_row(path / "metrics.json") for path in paths], output_dir)


if __name__ == "__main__":
    main()
