# Sparse-VGGT on VGGT, VGGT-Ω, and Pi3

This repository vendors the Sparse-VGGT adapter and SpargeAttn CUDA source:

```text
third_party/
├── SpargeAttn/                 # Python package: spas_sage_attn
└── sparse-vggt/
    ├── src/sparse_vggt/        # Sparse-VGGT adapters for VGGT and Pi3
    └── external/
        ├── vggt/vggt/
        └── Pi3/pi3/
```

The files are source only.  Build SpargeAttn on every target machine; do not
copy prebuilt `*.so` files between machines, CUDA versions, or PyTorch builds.

## Requirements

- Python 3.12 for `third_party/sparse-vggt`
- PyTorch with CUDA support (the experiment environment uses CUDA 12.8 builds)
- CUDA Toolkit >= 12.4; CUDA 12.8 is recommended
- NVIDIA GPU with compute capability >= 8.0

For RTX 4090, use compute capability **8.9**.  Do not use an `sm120` build on
this GPU.

## Build SpargeAttn

Activate the same environment that will run the experiment, then run from the
root of the checked-out branch:

```bash
export REPO_ROOT="$(pwd)"
export SPARGEATTN_ROOT="$REPO_ROOT/third_party/SpargeAttn"
export SPARSE_VGGT_ROOT="$REPO_ROOT/third_party/sparse-vggt"

# Change this path if CUDA 12.8 is installed elsewhere.
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# RTX 4090 / Ada Lovelace.
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4

python -m pip install -U pip setuptools wheel packaging ninja
python -m pip install -e "$SPARGEATTN_ROOT" --no-build-isolation
```

For a GPU-less build host, `TORCH_CUDA_ARCH_LIST=8.9` is still required.

## Verify the installed extension

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SPARGEATTN_ROOT:$SPARSE_VGGT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import torch
import spas_sage_attn
import spas_sage_attn._qattn
import spas_sage_attn._fused
from sparse_vggt.models.vggt import sparse_aggregator_from_vggt
from sparse_vggt.models.pi3 import sparse_model_from_pi3

print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("SpargeAttn:", spas_sage_attn.__file__)
print("SpargeAttn extensions: OK")
print("Sparse-VGGT adapters: OK")
PY
```

`spas_sage_attn.__file__` must resolve under the current repository's
`third_party/SpargeAttn/`; otherwise an old machine-wide installation is being
used.

## Runtime environment

Set these variables for **every** Sparse-VGGT job.  This prevents scripts from
falling back to stale external paths such as `SpargeAttn-sm120`.

```bash
export REPO_ROOT="$(pwd)"
export SPARGEATTN_ROOT="$REPO_ROOT/third_party/SpargeAttn"
export SPARSE_VGGT_ROOT="$REPO_ROOT/third_party/sparse-vggt"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SPARGEATTN_ROOT:$SPARSE_VGGT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
```

The method name is `sparse-vggt`.

Recommended experiment arguments:

```text
--acceleration-method sparse-vggt
--sparse-vggt-sparse-ratio 0.5
--sparse-vggt-pool-mode avg
```

For a more conservative accuracy-oriented configuration, use:

```text
--sparse-vggt-sparse-ratio 0.1
--sparse-vggt-cdf-threshold 0.97
```

VGGT-Ω's scripts use the equivalent Omega arguments:

```text
--acceleration-method sparse-vggt
--sparse-attention
--sparse-ratio 0.5
--sparse-pool-mode avg
```

## Typical errors

| Error | Resolution |
| --- | --- |
| `No module named spas_sage_attn` | Set `SPARGEATTN_ROOT` and install the package with the active Python interpreter. |
| `No module named spas_sage_attn._qattn` | The CUDA extension was not built. Re-run the editable installation above. |
| `no kernel image is available for execution` | Rebuild on the target GPU with `TORCH_CUDA_ARCH_LIST=8.9` for RTX 4090. |
| `CUDA_HOME is None` | Set `CUDA_HOME` to the directory containing `bin/nvcc`. |
| Poor Sparse-VGGT accuracy | Verify the imported SpargeAttn path, then ensure the sparse ratio and CDF threshold match the intended experiment configuration. |

Changing PyTorch, CUDA Toolkit, or GPU architecture requires rebuilding
SpargeAttn in the corresponding environment.
