# SelfTR

> **Official Implementation of the SelfTR paper.**

This repository is the official implementation of **SelfTR**, an
inference-time spatiotemporal token-compression method for the bundled
VGGT-Omega geometry backbone. It replaces the former experimental identifier
`u-m` with the stable public method identifier `selftr`, while retaining `u-m`
as a deprecated compatibility alias for existing experiments.

The official release focuses only on the VGGT/Omega path described in the
paper: camera estimation, depth prediction, the SelfTR compressor, and
reproducible evaluation on ScanNet, 7 Scenes, and NRGBD. Historical
sparse-attention, DA-VGGT, PI3, and analysis experiments are not part of the
official SelfTR implementation.

## What SelfTR does

VGGT-Omega alternates frame-local and inter-frame attention. SelfTR acts only
before global inter-frame attention:

1. Camera and register tokens are always retained; frame 0 patch tokens are
   kept one-to-one as reference tokens.
2. For patches in later frames, SelfTR constructs a local candidate graph over
   a spatial radius `r` and the next `t` frames.
3. It scores whole-group merges by cosine reconstruction error. Mutually
   nearest neighboring groups are accepted when their exact error increment
   satisfies `ΔE < 2λ`.
4. Each accepted group is represented by the mean of its current tokens for
   global attention; the attention residual is then restored to every original
   patch position and the original per-token MLP is run unchanged.

The official 300-frame configuration is in
[`configs/selftr/reproduction_300.json`](configs/selftr/reproduction_300.json):
`λ=0.04`, `r=2`, `t=4`, 5% minimum retained patches, and plan refreshes after
layers 0, 10, and 17.

## Repository layout

```text
selftr/                         Public API, identity, checkpoint checks, metrics
  identity.py                    Single source of truth for the method name
  config.py                      Typed SelfTR compression configuration
  model.py                       SelfTR model class and checkpoint loader
  evaluation/geometry.py         Portable depth, pose, and geometry metrics
vggt_omega/                     Bundled VGGT-Omega backbone (checkpoint compatible)
configs/selftr/                 Official immutable reproduction settings
scripts/eval_selftr.py          Single entry point for all three datasets
scripts/eval_{7scenes,scannet,nrgbd}_paper.py
tests/                          Unit tests for compression and public identity
```

## Environment

The validated runtime is Linux, Python 3.10+, an NVIDIA GPU with CUDA support,
and PyTorch 2.3 or newer. A 24 GB GPU can run shorter sequences; the complete
300-frame protocol should be run on a GPU with at least 40 GB free memory. The
historical reference runs used an RTX 4090 with 48 GB.

Create an isolated environment and install PyTorch appropriate for the local
CUDA driver. The example below uses CUDA 12.1 wheels; replace `cu121` when
needed according to the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

```bash
conda create -n selftr python=3.10 -y
conda activate selftr

pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1 torchvision==0.18.1
pip install -r requirements.txt
pip install -e .
```

`triton` is optional. When it is available, SelfTR uses a fused CUDA edge-cost
kernel. The numerically equivalent PyTorch implementation is selected when it
is absent or when `SELFTR_TRITON=0`.

Verify the installation before downloading or processing a dataset:

```bash
python -m compileall -q selftr vggt_omega scripts
python -m pytest -q tests/test_selftr_identity.py tests/test_frame_fusion_partition.py
python scripts/eval_selftr.py --dataset 7scenes --help
```

## Checkpoints

SelfTR changes no learned parameters, so it uses the corresponding
**VGGT-Omega** checkpoint directly. Store the checkpoint outside Git, for
example:

```text
pretrained_ckpts/selftr_vggt_omega_1b_512.pt
```

Inspect it before an expensive evaluation:

```bash
python -c "from selftr import inspect_checkpoint; print(inspect_checkpoint('pretrained_ckpts/selftr_vggt_omega_1b_512.pt'))"
```

### Important: supplied VGGT checkpoint

The supplied file
`/data_SSD1/mmc_lyxiang/3D/vggt/ckpt/model.pt` was inspected during this
cleanup. It is an **official VGGT** checkpoint with
`aggregator.global_blocks`, 14-pixel patches, and four register tokens. This
release applies SelfTR to the bundled **VGGT-Omega** implementation, whose
state dictionary uses `aggregator.inter_frame_blocks`; the two architectures
are not load-compatible. The loader now detects this before allocating a
5 GB model and explains the mismatch.

Therefore, use a matching VGGT-Omega checkpoint to reproduce the SelfTR
results below. The supplied file is still useful for verifying that the
checkpoint diagnostic works:

```bash
python -c "from selftr import inspect_checkpoint; print(inspect_checkpoint('/data_SSD1/mmc_lyxiang/3D/vggt/ckpt/model.pt'))"
```

It should report `family='official-vggt'`. Do not use `strict=False` or rename
state-dict keys to force this checkpoint into SelfTR: that would silently
change the model being evaluated.

## Python inference

```python
import torch

from selftr import SelfTR, SelfTRConfig
from vggt_omega.utils.load_fn import load_and_preprocess_images

config = SelfTRConfig(
    recompute_layers=(0, 10, 17),
    lambda_cost=0.04,
    temporal_window=4,
    spatial_radius=2,
)
model = SelfTR.from_checkpoint(
    "pretrained_ckpts/selftr_vggt_omega_1b_512.pt",
    device="cuda",
    **config.model_kwargs(),
)
images = load_and_preprocess_images(
    ["frame_000.png", "frame_001.png"], image_resolution=512, mode="max_size"
).to("cuda")
with torch.inference_mode():
    prediction = model(images)

depth = prediction["depth"]
pose_encoding = prediction["pose_enc"]
```

## Dataset preparation

Keep data outside the repository. Every command below accepts an explicit
`--data-root`, so no server-specific absolute path is required.

| Dataset | Required layout | Notes |
| --- | --- | --- |
| 7 Scenes | `<root>/<scene>/TestSplit.txt`, `<root>/<scene>/seq-NN/frame-XXXXXX.color.png`, matching `.pose.txt`, and `.depth.png` or `.depth.proj.png` | RGB-registered `*.depth.proj.png` is preferred. Raw depth is supported as a fallback. |
| ScanNet | `<root>/sceneXXXX_YY/color/*.jpg`, `depth/*.png`, `pose/*.txt`, `intrinsic/intrinsic_color.txt` | The evaluator accepts every valid scene directory under the root. Depth uses ScanNet's millimetre scale. |
| NRGBD | `<root>/<sequence>/images/imgN.png`, `depth/depthN.png`, and `poses.txt` | `poses.txt` must contain one 4×4 pose per image. The loader converts the stored OpenGL camera convention to OpenCV. |

The 7 Scenes evaluator uses the official test lists. ScanNet and NRGBD use
all valid sequence directories found under the supplied root; specify
`--sequences ...` to evaluate a fixed subset. All evaluators skip invalid pose
or missing RGB/depth tuples and write the exact selected frame IDs to
`sampled_frames.json`.

Validate file discovery without loading a model:

```bash
python scripts/eval_selftr.py --dataset 7scenes \
  --data-root /datasets/7scenes --dry-run
python scripts/eval_selftr.py --dataset scannet \
  --data-root /datasets/scannet --dry-run
python scripts/eval_selftr.py --dataset nrgbd \
  --data-root /datasets/nrgbd --dry-run
```

## Reproducing the 300-frame protocol

`scripts/eval_selftr.py` is the official entry point. Unless overridden, it
applies the configuration in `configs/selftr/reproduction_300.json`:

- 300 frames per sequence, first/last-preserving uniform sampling, seed 42;
- 512-pixel `max_size` preprocessing and per-frame median depth alignment;
- three CUDA-event timing repeats;
- `--frame-fusion-mode selftr`, no FastVGGT merge, and the SelfTR parameters
  listed above.

Run one dataset at a time on an otherwise idle GPU:

```bash
CHECKPOINT=/checkpoints/selftr_vggt_omega_1b_512.pt

python scripts/eval_selftr.py --dataset 7scenes \
  --data-root /datasets/7scenes --checkpoint "$CHECKPOINT" \
  --device cuda:0 --output-dir outputs/selftr_300/7scenes

python scripts/eval_selftr.py --dataset scannet \
  --data-root /datasets/scannet --checkpoint "$CHECKPOINT" \
  --device cuda:0 --output-dir outputs/selftr_300/scannet

python scripts/eval_selftr.py --dataset nrgbd \
  --data-root /datasets/nrgbd --checkpoint "$CHECKPOINT" \
  --device cuda:0 --output-dir outputs/selftr_300/nrgbd
```

For a short smoke evaluation, override the protocol explicitly; this must not
be compared with the 300-frame reference numbers:

```bash
python scripts/eval_selftr.py --dataset 7scenes \
  --data-root /datasets/7scenes --checkpoint "$CHECKPOINT" \
  --sequences chess/seq-03 --num-frames 10 --timing-repeats 1 \
  --device cuda:0 --output-dir outputs/smoke_7scenes
```

Each output directory contains:

- `sampled_frames.json`: the exact input files and sampled indices;
- `metrics.json`: method name/ID, complete protocol, aggregate metrics, and
  per-sequence metrics;
- `pose_errors.npz`: all pairwise rotation and translation errors.

The geometry metrics are deterministic Pi3-compatible metrics: corresponding
depth point maps are Sim(3)-aligned, refined with point-to-point ICP, then
scored by bidirectional nearest-neighbor accuracy/completeness and normal
consistency. The evaluator records the fixed 100,000-point budget.

## Reference results

The following archived SelfTR runs use the 300-frame protocol above, a
compatible VGGT-Omega checkpoint, and the all-sequence sets visible to the
original runs (18 7 Scenes sequences, 30 ScanNet scenes, and 9 NRGBD
sequences). Values can vary slightly across CUDA, PyTorch, and Open3D
versions; compare the saved protocol and `sampled_frames.json` before treating
a difference as a model change.

| Dataset | AUC@3 ↑ | AUC@30 ↑ | δ1.25 ↑ | AbsRel ↓ | Mean latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 Scenes | 31.07 | 87.52 | 94.49 | 0.0566 | 11,763 |
| ScanNet | 16.64 | 83.70 | 98.44 | 0.0308 | 19,604 |
| NRGBD | 85.51 | 98.07 | 99.59 | 0.0086 | 10,563 |

The row values are reported only for the stated protocol. Changing frame
count, input resize mode, sampled frames, depth alignment, or checkpoint is a
different experiment and must be reported separately.

## Renaming the method

The official display name has one source of truth:
[`selftr/identity.py`](selftr/identity.py).

Change `DEFAULT_METHOD_NAME` to permanently rename the method in future code
and artifacts, while keeping the stable machine identifier `selftr`. For a
single run or a paper variant, use either of the following without editing
code:

```bash
SELFTR_METHOD_NAME="My New Method" python scripts/eval_selftr.py --dataset 7scenes ...
python scripts/eval_selftr.py --dataset 7scenes --method-name "My New Method" ...
```

`u-m`, `u_m`, and `um` remain accepted only as deprecated aliases. New
commands, configuration files, and result tables should use `selftr` and the
chosen display name.

## Reproducibility checklist

Before publishing a comparison, retain all of the following with the output
directory:

1. Git commit ID and `pip freeze` output.
2. Checkpoint filename and cryptographic hash (do not commit the checkpoint).
3. GPU model, driver, CUDA, PyTorch, Triton, and Open3D versions.
4. The unmodified `metrics.json`, `sampled_frames.json`, and `pose_errors.npz`.
5. The complete command line, including any `--sequences` subset.

## License and upstream components

See [`LICENSE`](LICENSE). The bundled VGGT-Omega backbone remains in
`vggt_omega/` to preserve upstream checkpoint compatibility; SelfTR-specific
public code lives under `selftr/`.
