"""Optional fused CUDA kernels for the unified merge planner.

The planner evaluates the same whole-group reconstruction objective for many
graph edges.  Expressing that objective as ordinary indexed PyTorch operators
materializes several ``[edge_chunk, feature_dim]`` tensors.  The kernel below
keeps those intermediates inside one Triton program and writes only one scalar
cost per edge.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _selftr_edge_cost_kernel(
        group_sums_ptr,
        group_weights_ptr,
        group_representatives_ptr,
        group_errors_ptr,
        features_ptr,
        edge_left_ptr,
        edge_right_ptr,
        edge_valid_ptr,
        output_ptr,
        edge_count,
        group_sums_stride,
        features_stride,
        FEATURE_DIM: tl.constexpr,
        BLOCK_FEATURES: tl.constexpr,
        BLOCK_EDGES: tl.constexpr,
        PREFER_BEST_PARENT: tl.constexpr,
    ):
        edge_offsets = (
            tl.program_id(axis=0) * BLOCK_EDGES
            + tl.arange(0, BLOCK_EDGES)
        )
        edge_mask = edge_offsets < edge_count
        left = tl.load(edge_left_ptr + edge_offsets, mask=edge_mask, other=0)
        right = tl.load(edge_right_ptr + edge_offsets, mask=edge_mask, other=0)
        left_rep = tl.load(
            group_representatives_ptr + left,
            mask=edge_mask,
            other=0,
        )
        right_rep = tl.load(
            group_representatives_ptr + right,
            mask=edge_mask,
            other=0,
        )

        feature_offsets = tl.arange(0, BLOCK_FEATURES)
        feature_mask = feature_offsets < FEATURE_DIM
        matrix_mask = edge_mask[:, None] & feature_mask[None, :]
        left_sum = tl.load(
            group_sums_ptr
            + left[:, None] * group_sums_stride
            + feature_offsets[None, :],
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        right_sum = tl.load(
            group_sums_ptr
            + right[:, None] * group_sums_stride
            + feature_offsets[None, :],
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        left_feature = tl.load(
            features_ptr
            + left_rep[:, None] * features_stride
            + feature_offsets[None, :],
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        right_feature = tl.load(
            features_ptr
            + right_rep[:, None] * features_stride
            + feature_offsets[None, :],
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)

        merged_sum = left_sum + right_sum
        left_dot = tl.sum(merged_sum * left_feature, axis=1)
        right_dot = tl.sum(merged_sum * right_feature, axis=1)
        merged_weight = tl.load(
            group_weights_ptr + left,
            mask=edge_mask,
            other=0.0,
        ).to(tl.float32) + tl.load(
            group_weights_ptr + right,
            mask=edge_mask,
            other=0.0,
        ).to(tl.float32)
        left_error = merged_weight - left_dot
        right_error = merged_weight - right_dot
        if PREFER_BEST_PARENT:
            merged_error = tl.minimum(left_error, right_error)
        else:
            merged_error = left_error
        delta = merged_error - tl.load(
            group_errors_ptr + left,
            mask=edge_mask,
            other=0.0,
        ).to(tl.float32) - tl.load(
            group_errors_ptr + right,
            mask=edge_mask,
            other=0.0,
        ).to(tl.float32)
        valid = tl.load(edge_valid_ptr + edge_offsets, mask=edge_mask, other=0)
        delta = tl.where(valid, delta, float("inf"))
        tl.store(output_ptr + edge_offsets, delta, mask=edge_mask)


def fused_selftr_edge_cost(
    group_sums: torch.Tensor,
    group_weights: torch.Tensor,
    group_representatives: torch.Tensor,
    group_errors: torch.Tensor,
    features: torch.Tensor,
    edge_left: torch.Tensor,
    edge_right: torch.Tensor,
    edge_valid: torch.Tensor,
    *,
    prefer_best_parent: bool,
) -> torch.Tensor | None:
    """Return fused per-edge SelfTR costs, or ``None`` for PyTorch fallback.

    Setting ``SELFTR_TRITON=0`` disables the kernel. ``BLOCK_EDGES`` is kept
    configurable for hardware-specific benchmarking without changing model
    semantics; the conservative default avoids register spilling at C=1024.
    """

    if (
        triton is None
        or os.environ.get("SELFTR_TRITON", os.environ.get("VGGT_UM_TRITON", "1")) == "0"
        or group_sums.device.type != "cuda"
        or group_sums.dtype != torch.float32
        or features.dtype != torch.float32
        or not group_sums.is_contiguous()
        or not features.is_contiguous()
    ):
        return None
    if edge_left.numel() == 0:
        return torch.empty_like(edge_left, dtype=group_sums.dtype)

    feature_dim = int(group_sums.shape[1])
    if feature_dim != int(features.shape[1]):
        raise ValueError("group_sums and features must have the same feature dimension")
    block_features = triton.next_power_of_2(feature_dim)
    # Triton reductions with very wide non-power-of-two rows would otherwise
    # consume excessive registers. VGGT uses C=1024, while this guard also
    # covers the small dimensions used in tests.
    if block_features > 2048:
        return None
    block_edges = int(
        os.environ.get(
            "SELFTR_TRITON_BLOCK_EDGES",
            os.environ.get("VGGT_UM_TRITON_BLOCK_EDGES", "1"),
        )
    )
    if block_edges not in (1, 2, 4):
        raise ValueError("SELFTR_TRITON_BLOCK_EDGES must be one of 1, 2, or 4")

    output = torch.empty(edge_left.shape, device=edge_left.device, dtype=torch.float32)
    grid = (triton.cdiv(int(edge_left.numel()), block_edges),)
    _selftr_edge_cost_kernel[grid](
        group_sums,
        group_weights,
        group_representatives,
        group_errors,
        features,
        edge_left,
        edge_right,
        edge_valid,
        output,
        int(edge_left.numel()),
        int(group_sums.stride(0)),
        int(features.stride(0)),
        FEATURE_DIM=feature_dim,
        BLOCK_FEATURES=block_features,
        BLOCK_EDGES=block_edges,
        PREFER_BEST_PARENT=bool(prefer_best_parent),
        num_warps=4 if block_edges == 1 else 8,
    )
    return output


__all__ = ["fused_selftr_edge_cost"]
