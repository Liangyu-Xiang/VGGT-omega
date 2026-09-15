# ScanNet50 evaluation

`scripts/eval_scannet50.py` evaluates only the original DenseVGGT, FastVGGT,
and SelTR variants.  It uses the 50-scene list from FastVGGT's
`eval/scannet_50.yaml` and endpoint-preserving uniform sampling: the first and
last valid RGB/pose/depth frames are always kept and only the interior is
uniformly selected.

Run the formal configurations with a VGGT checkpoint:

```bash
bash scripts/run_scannet50_500.sh /path/to/vggt.pt
bash scripts/run_scannet50_1000.sh /path/to/vggt.pt
```

The launchers use GPUs 4, 5, and 6 for DenseVGGT, FastVGGT, and SelTR by
default.  Override them with `DENSE_GPU`, `FAST_GPU`, and `SELFTR_GPU`.  They
also default to the FastVGGT Python environment; set `EVAL_PYTHON` to use a
different environment.

The checked-in cache at
`/data_SSD1/mmc_lyxiang/dataset/scannet50_data/extracted/scannet-frames`
currently has 300 frames per scene.  It is sufficient for the 100-frame smoke
test:

```bash
bash scripts/test_scannet50_100.sh /path/to/vggt.pt
```

It is intentionally rejected by the 500/1000 launchers.  Point
`SCANNET_DATA_ROOT` to a full-frame ScanNet extraction before a formal run;
the `--require-exact-frames` guard prevents an accidental 300-frame result.

Each method writes per-scene `metrics.json` files and one aggregate
`metrics.json`.  Pose output includes AUC@3/5/15/20/30, RRA/RTA@5/30, ATE,
ARE, RPE rotation/translation, and APE.  AUC@330 in the request is interpreted
as AUC@30 and `RPW-trans` is output as `rpe_trans_rmse_m`.

Geometry follows FastVGGT's ScanNet route: estimated depth is unprojected with
estimated cameras, mapped back using the first GT camera, bbox-scale aligned to
the ScanNet mesh, then sampled and voxelized at 0.05 m.  `cd_m` is the
FastVGGT-compatible clipped bidirectional sum `Acc + Comp`; `overall_m` is
`(Acc + Comp) / 2`.  The same output includes the requested medians, normal
consistency, precision/recall/F1 at 0.05 m, depth metrics, latency, FPS, peak
allocated/reserved VRAM, and token retention.  FastVGGT/DenseVGGT retain one
fixed-policy statistic; SelTR records all three U-M refresh stages.

With `--save-visualizations` (enabled by all three launcher scripts), each
scene also receives `visualization/reconstruction_overlay.ply`, separate
predicted/GT PLYs, `reconstruction.png`, and `trajectory_xz.png`.  The
trajectory uses FastVGGT's XZ projection and colours the aligned estimated
trajectory by absolute position error.
