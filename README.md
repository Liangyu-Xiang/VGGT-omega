# SelfTR

Official inference implementation for the **original VGGT** architecture and
the two inference-time compression methods compared with it:

- **VGGT**: the unmodified original VGGT forward path;
- **FastVGGT**: original VGGT with bipartite spatial token merging;
- **SelfTR**: original VGGT with the SelfTR spatiotemporal representative
  planner before global attention.

VGGT uses 14-pixel patches, four register tokens, and checkpoints containing
aggregator.global_blocks. It is not VGGT-Omega; Omega checkpoints are not
compatible with this implementation.

## Install

Use Python 3.10+ and an NVIDIA CUDA environment with PyTorch installed.

~~~bash
pip install -r requirements.txt
pip install -e .
~~~

## Forward inference

`scripts/infer.py` accepts ordered RGB frames, performs one forward inference
pass, prints output tensor shapes, and can save the output dictionary to a
CPU-portable PyTorch file.

~~~bash
CHECKPOINT=pretrained_ckpts/vggt_1b.pt

# Original VGGT baseline.
python scripts/infer.py --method vggt --checkpoint "$CHECKPOINT" \
  --images frame_000.png frame_001.png frame_002.png

# FastVGGT, with its standard 0.9 merge ratio.
python scripts/infer.py --method fastvggt --checkpoint "$CHECKPOINT" \
  --images frame_000.png frame_001.png frame_002.png \
  --merge-ratio 0.9

# SelfTR, using the formal default planner configuration.
python scripts/infer.py --method selftr --checkpoint "$CHECKPOINT" \
  --images frame_000.png frame_001.png frame_002.png \
  --output outputs/selftr_prediction.pt
~~~

Use --input-mode crop for VGGT's official 518-pixel crop preprocessing, or
--input-mode pad to preserve the full image in a padded 518-pixel square.
Input ordering is preserved.

The prediction dictionary contains camera pose encodings, depth/depth
confidence, world points/world-point confidence, and the preprocessed images.
Track predictions are additionally returned when query points are supplied
through the Python API.

## Python API

~~~python
import torch

from vggt.models import VGGT
from vggt.models.selftr import build_selftr_plan
from vggt.utils.load_fn import load_and_preprocess_images

images = load_and_preprocess_images(["frame_000.png", "frame_001.png"]).cuda()

# Original VGGT
vggt = VGGT().cuda().eval()

# FastVGGT
fastvggt = VGGT(
    enable_token_merging=True,
    token_merging_method="spatial",
    token_merging_ratio=0.9,
).cuda().eval()

# SelfTR
selftr = VGGT(
    um_lambda_cost=0.04,
    um_spatial_radius=2,
    um_temporal_window=4,
    um_refresh_layers="0,9,21",
).cuda().eval()

with torch.inference_mode():
    prediction = selftr(images)
~~~

Load the same original-VGGT checkpoint into each model with
model.load_state_dict(torch.load(checkpoint, weights_only=True)).

## Formal evaluation

The 7Scenes evaluator reports both the original benchmark's and FastVGGT's
geometry protocols, plus relative pose and Sim(3)-aligned trajectory metrics.
Leaving `--num-frames` unset evaluates every frame available after stride
sampling.

The formal SelTR λ ablation uses all 7Scenes test frames at stride 3, with no
per-sequence frame cap. It evaluates λ = 0.02, 0.01, 0.03, 0.05, 0.06, 0.07,
0.08, 0.09, and 0.10 and writes a final CSV/JSON table. Specify the dataset
path and one or more GPU IDs explicitly:

~~~bash
bash scripts/run_7scenes_lambda_ablation.sh pretrained_ckpts/vggt_1b.pt \
  /path/to/7scenes 0,1,3
~~~

The GPU list can contain one GPU (sequential execution) or several GPUs
(automatic round-robin partitioning). Set `EVAL_PYTHON` when the intended
Python environment is not the default FastVGGT environment.

## Layout

~~~text
vggt/          Original VGGT backbone plus FastVGGT and SelfTR forward logic
scripts/
  infer.py     Unified three-mode forward-inference entry point
  eval_7scenes.py  Formal 7Scenes evaluator
  run_7scenes_lambda_ablation.sh  Full-sequence SelTR λ ablation
~~~

ScanNet50 evaluation and reconstruction/trajectory visualization scripts are
documented in [docs/scannet50_evaluation.md](docs/scannet50_evaluation.md).

See [LICENSE](LICENSE) for licensing information.
