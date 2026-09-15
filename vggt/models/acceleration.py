"""Inference-only acceleration primitives shared by the VGGT aggregator.

The implementation intentionally keeps the public VGGT tensor contract intact:
frame merging is restored before cached decoder features are exposed, and spatial
merging is restored immediately after each global-attention operation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FrameMergeState:
    inverse: torch.Tensor
    segments: list[tuple[int, int]]
    merge_groups: list[list[int]]

    @property
    def active_frames(self) -> int:
        return int(self.inverse.max().item()) + 1 if self.inverse.numel() else 0


def parse_layer_ratio_schedule(schedule: str, depth: int) -> dict[int, float]:
    if not schedule:
        return {}
    ratios: dict[int, float] = {}
    for raw in schedule.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid layer ratio item {item!r}; expected layers:ratio")
        layers, ratio_text = item.split(":", 1)
        ratio = float(ratio_text)
        if not 0.0 <= ratio < 1.0:
            raise ValueError("Layer-wise merge ratios must be in [0, 1).")
        if "-" in layers:
            start_text, end_text = layers.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(layers)
        if start < 1 or end > depth or start > end:
            raise ValueError(f"Layer range {layers!r} is outside 1-{depth}")
        for layer in range(start, end + 1):
            ratios[layer - 1] = ratio
    return ratios


def parse_block_indices(value: str, depth: int) -> set[int]:
    if not value.strip():
        return set()
    blocks: set[int] = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending block range {item!r}")
            blocks.update(range(start, end + 1))
        else:
            blocks.add(int(item))
    invalid = sorted(block for block in blocks if not 0 <= block < depth)
    if invalid:
        raise ValueError(f"Block indices must be within [0, {depth - 1}], got {invalid}")
    return blocks


def _pool_tokens(tokens: torch.Tensor, grid_size: tuple[int, int], stride: int) -> torch.Tensor:
    if stride <= 1:
        return tokens
    frames, patches, channels = tokens.shape
    height, width = grid_size
    if patches != height * width:
        raise ValueError("Patch-token count does not match the patch grid")
    pooled = F.avg_pool2d(
        tokens.view(frames, height, width, channels).permute(0, 3, 1, 2).float(),
        kernel_size=stride,
        stride=stride,
        ceil_mode=True,
    )
    return pooled.permute(0, 2, 3, 1).reshape(frames, -1, channels).to(tokens.dtype)


def _pair_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left.float(), dim=-1)
    right = F.normalize(right.float(), dim=-1)
    return (left * right).sum(dim=-1).mean()


def _segments(tokens: torch.Tensor, alpha: float, threshold: float, max_window: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start = 0
    reference = tokens[0]
    ema = tokens.new_tensor(1.0, dtype=torch.float32)
    for index in range(1, tokens.shape[0]):
        similarity = _pair_similarity(reference, tokens[index])
        ema = alpha * similarity.float() + (1.0 - alpha) * ema
        if ema < threshold or (max_window > 0 and index - start + 1 >= max_window):
            result.append((start, index - 1))
            start, reference, ema = index, tokens[index], tokens.new_tensor(1.0, dtype=torch.float32)
    result.append((start, tokens.shape[0] - 1))
    return result


def merge_frames(
    tokens: torch.Tensor,
    *,
    patch_start: int,
    grid_size: tuple[int, int],
    alpha: float,
    segment_threshold: float,
    merge_threshold: float,
    max_window: int,
    pool_stride: int,
    max_group_size: int,
    pair_threshold: float,
    span_threshold: float,
) -> tuple[torch.Tensor, FrameMergeState]:
    """Merge contiguous similar frames using the Omega multiframe policy."""
    frames = tokens.shape[0]
    if frames <= 1:
        return tokens, FrameMergeState(torch.arange(frames, device=tokens.device), [(0, frames - 1)], [])
    pooled = _pool_tokens(tokens[:, patch_start:], grid_size, pool_stride)
    segments = _segments(pooled, alpha, segment_threshold, max_window)
    active: list[torch.Tensor] = []
    inverse = torch.empty(frames, dtype=torch.long, device=tokens.device)
    assigned = [False] * frames
    merge_groups: list[list[int]] = []

    def append_single(index: int) -> None:
        inverse[index] = len(active)
        active.append(tokens[index])
        assigned[index] = True

    def append_group(indices: list[int]) -> None:
        active_index = len(active)
        inverse[torch.tensor(indices, device=tokens.device)] = active_index
        active.append(tokens[indices].float().mean(dim=0).to(tokens.dtype))
        merge_groups.append(indices)
        for index in indices:
            assigned[index] = True

    for start, end in segments:
        if start == end:
            append_single(start)
            continue
        append_single(start)
        cursor = start + 1
        while cursor < end:
            group_size = 0
            for candidate in range(min(max_group_size, 4), 2, -1):
                candidate_end = cursor + candidate - 1
                if candidate_end > end:
                    continue
                pair_ok = all(
                    _pair_similarity(pooled[index], pooled[index + 1]) > pair_threshold
                    for index in range(cursor, candidate_end)
                )
                span_ok = _pair_similarity(pooled[cursor], pooled[candidate_end]) > span_threshold
                if pair_ok and span_ok:
                    group_size = candidate
                    break
            if group_size:
                append_group(list(range(cursor, cursor + group_size)))
                cursor += group_size
                continue
            current = _pair_similarity(pooled[cursor], pooled[cursor + 1])
            following = _pair_similarity(pooled[cursor + 1], pooled[cursor + 2]) if cursor + 1 < end else current
            if current > merge_threshold and current > following:
                append_group([cursor, cursor + 1])
                cursor += 2
            else:
                append_single(cursor)
                cursor += 1
        if not assigned[end]:
            append_single(end)
    return torch.stack(active), FrameMergeState(inverse, segments, merge_groups)


def restore_frames(tokens: torch.Tensor, state: list[FrameMergeState]) -> torch.Tensor:
    return torch.stack([tokens[index, item.inverse] for index, item in enumerate(state)])


@dataclass
class SpatialMergePlan:
    src_indices: torch.Tensor
    dst_indices: torch.Tensor
    merged_sources: torch.Tensor
    merged_destinations: torch.Tensor
    kept_source_positions: torch.Tensor
    protected_indices: torch.Tensor
    original_tokens: int

    @property
    def active_tokens(self) -> int:
        return int(self.src_indices.numel() - self.merged_sources.shape[1] + self.dst_indices.numel() + self.protected_indices.numel())


@dataclass
class UMPlan:
    """One-refresh U-M representative map for a flattened VGGT global block."""
    inverse: torch.Tensor
    representative_indices: torch.Tensor
    original_tokens: int

    @property
    def active_tokens(self) -> int:
        return int(self.representative_indices.numel())


def build_um_plan(metric: torch.Tensor, *, num_frames: int, patch_start: int,
                  grid_size: tuple[int, int], spatial_radius: int = 2,
                  temporal_window: int = 4, lambda_cost: float = 0.04) -> UMPlan:
    """Build a GPU U-M map from norm1 features on local time/space edges.

    The first frame and all per-frame special tokens are protected.  Remaining
    tokens are paired only when they are mutual best local neighbours and their
    cosine distortion is below the U-M lambda threshold.  This is the same
    representative-token contract as Omega: merge before global attention and
    restore every member from its representative afterwards.
    """
    if metric.shape[0] != 1:
        raise ValueError("U-M currently requires batch size 1")
    h, w = grid_size
    per_frame = patch_start + h * w
    total = metric.shape[1]
    if total != num_frames * per_frame:
        raise ValueError("U-M token layout does not match frames and patch grid")
    device = metric.device
    inverse = torch.arange(total, device=device)
    if num_frames <= 1:
        return UMPlan(inverse, inverse, total)
    patches = torch.nn.functional.normalize(
        metric[0].view(num_frames, per_frame, -1)[:, patch_start:].float(), dim=-1
    ).view(num_frames, h, w, -1)
    # The nearest temporal frame at each local spatial offset gives an O(FPR²)
    # candidate graph; it retains the r/t cube topology without materializing a
    # dense N² matrix.
    best_score = torch.full((num_frames, h, w), -float("inf"), device=device)
    best_index = torch.full((num_frames, h, w), -1, device=device, dtype=torch.long)
    rows = torch.arange(h, device=device)[:, None].expand(h, w)
    cols = torch.arange(w, device=device)[None, :].expand(h, w)
    for dt in range(1, min(int(temporal_window), num_frames - 1) + 1):
        for dy in range(-int(spatial_radius), int(spatial_radius) + 1):
            for dx in range(-int(spatial_radius), int(spatial_radius) + 1):
                r0, r1 = max(0, -dy), min(h, h - dy)
                c0, c1 = max(0, -dx), min(w, w - dx)
                if r0 >= r1 or c0 >= c1:
                    continue
                # source at (r,c) matches the earlier-frame neighbour at
                # (r+dy,c+dx); both slices use the same valid extent.
                score = (patches[dt:, r0:r1, c0:c1] * patches[:-dt, r0 + dy:r1 + dy, c0 + dx:c1 + dx]).sum(-1)
                current = best_score[dt:, r0:r1, c0:c1]
                better = score > current
                best_score[dt:, r0:r1, c0:c1] = torch.where(better, score, current)
                target = ((torch.arange(dt, num_frames, device=device) - dt)[:, None, None] * (h * w)
                          + (rows[r0:r1, c0:c1] + dy) * w + (cols[r0:r1, c0:c1] + dx))
                best_index[dt:, r0:r1, c0:c1] = torch.where(better, target, best_index[dt:, r0:r1, c0:c1])
    base = torch.arange(num_frames, device=device)[:, None, None] * (h * w) + rows * w + cols
    flat_src = base[1:].reshape(-1)
    flat_dst = best_index[1:].reshape(-1)
    flat_score = best_score[1:].reshape(-1)
    valid = (flat_dst >= h * w) & (flat_score >= 1.0 - float(lambda_cost))
    src, dst = flat_src[valid], flat_dst[valid]
    # One disjoint mutual-best pass: deterministic, parallel and keeps the
    # representative choice local in the requested r/t cube.
    lookup = torch.full((num_frames * h * w,), -1, device=device, dtype=torch.long)
    lookup[src] = dst
    mutual = lookup[dst] == src
    src, dst = src[mutual], dst[mutual]
    choose = src < dst
    src, dst = src[choose], dst[choose]
    offset = torch.arange(num_frames, device=device)[:, None] * per_frame + patch_start
    src_full = (src // (h * w)) * per_frame + patch_start + src % (h * w)
    dst_full = (dst // (h * w)) * per_frame + patch_start + dst % (h * w)
    inverse[src_full] = dst_full
    representatives = torch.unique(inverse, sorted=True)
    return UMPlan(inverse, representatives, total)


def um_attention(attention, x: torch.Tensor, pos: torch.Tensor | None, plan: UMPlan) -> torch.Tensor:
    """Aggregate Q/K/V per U-M representative, attend, then restore members."""
    batch, _, channels = x.shape
    qkv = attention.qkv(x).reshape(batch, -1, 3, attention.num_heads, attention.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = attention.q_norm(qkv[0]), attention.k_norm(qkv[1]), qkv[2]
    if attention.rope is not None:
        q, k = attention.rope(q, pos), attention.rope(k, pos)
    def reduce(value):
        value = value.transpose(1, 2).reshape(batch, plan.original_tokens, channels)
        sums = torch.zeros(batch, plan.original_tokens, channels, device=x.device, dtype=value.dtype)
        sums.index_add_(1, plan.inverse, value)
        counts = torch.bincount(plan.inverse, minlength=plan.original_tokens).to(value.dtype).clamp_min_(1)
        return (sums / counts[None, :, None]).index_select(1, plan.representative_indices)
    q, k, v = (reduce(q), reduce(k), reduce(v))
    q = q.reshape(batch, -1, attention.num_heads, attention.head_dim).transpose(1, 2)
    k = k.reshape(batch, -1, attention.num_heads, attention.head_dim).transpose(1, 2)
    v = v.reshape(batch, -1, attention.num_heads, attention.head_dim).transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    out = attention.proj_drop(attention.proj(out.transpose(1, 2).reshape(batch, -1, channels)))
    restored_index = torch.searchsorted(plan.representative_indices, plan.inverse)
    return out.index_select(1, restored_index)


def build_fastvggt_plan(
    metric: torch.Tensor,
    *,
    num_frames: int,
    patch_start: int,
    grid_size: tuple[int, int],
    merge_ratio: float,
) -> SpatialMergePlan | None:
    """FastVGGT's 2x2 bipartite spatial merge with protected special tokens."""
    batch, total, _ = metric.shape
    if batch != 1 or merge_ratio <= 0.0:
        return None
    height, width = grid_size
    tokens_per_frame = patch_start + height * width
    if total != num_frames * tokens_per_frame:
        raise ValueError("FastVGGT merge received an invalid frame/token layout")
    device = metric.device
    destination_mask = torch.zeros(total, dtype=torch.bool, device=device)
    destination_mask[:tokens_per_frame] = True
    offsets = torch.arange(num_frames, device=device) * tokens_per_frame
    if num_frames > 1:
        destination_mask[(offsets[1:, None] + torch.arange(patch_start, device=device)).flatten()] = True
        generator = torch.Generator(device=device)
        generator.manual_seed(33)
        # Preserve FastVGGT's deterministic one-destination-per-2x2-block
        # sampling, but form all indices on-device. The former nested Python
        # loop became costly at 300 frames and fed later CPU-side indexing.
        block_rows = torch.arange(0, height - 1, 2, device=device)
        block_cols = torch.arange(0, width - 1, 2, device=device)
        row_grid, col_grid = torch.meshgrid(block_rows, block_cols, indexing="ij")
        base_rows, base_cols = row_grid.reshape(-1), col_grid.reshape(-1)
        selected = torch.randint(
            4, (base_rows.numel(), num_frames - 1), generator=generator, device=device
        )
        frame_ids = torch.arange(num_frames - 1, device=device).expand(base_rows.numel(), -1)
        patch_indices = (
            (base_rows[:, None] + selected // 2) * width
            + base_cols[:, None] + selected % 2
        ).reshape(-1)
        destination_mask[
            offsets[frame_ids.reshape(-1) + 1] + patch_start + patch_indices
        ] = True
    dst_indices = torch.nonzero(destination_mask, as_tuple=False).flatten()
    src_indices = torch.nonzero(~destination_mask, as_tuple=False).flatten()
    if src_indices.numel() == 0:
        return None
    protected_count = int(total * 0.1)
    protected_indices = torch.arange(0, total, max(1, total // max(protected_count, 1)), device=device)[:protected_count]
    src_metric = F.normalize(metric[:, src_indices].float(), dim=-1)
    dst_metric = F.normalize(metric[:, dst_indices].float(), dim=-1)
    scores = torch.empty((src_indices.numel(),), device=device, dtype=torch.float32)
    matches = torch.empty((src_indices.numel(),), device=device, dtype=torch.long)
    for start in range(0, src_indices.numel(), 4096):
        end = min(start + 4096, src_indices.numel())
        values, indices = (src_metric[:, start:end] @ dst_metric.transpose(-1, -2)).max(dim=-1)
        scores[start:end], matches[start:end] = values[0], indices[0]
    protected_source = torch.isin(src_indices, protected_indices)
    candidates = torch.argsort(scores, descending=True)
    candidates = candidates[~protected_source[candidates]]
    count = min(int(total * merge_ratio), candidates.numel())
    merged_sources = candidates[:count].unsqueeze(0)
    merged_mask = torch.zeros(src_indices.numel(), dtype=torch.bool, device=device)
    merged_mask[merged_sources[0]] = True
    kept_source_positions = torch.nonzero(~merged_mask, as_tuple=False).flatten()
    return SpatialMergePlan(
        src_indices=src_indices,
        dst_indices=dst_indices,
        merged_sources=merged_sources,
        merged_destinations=matches[merged_sources],
        kept_source_positions=kept_source_positions,
        protected_indices=protected_indices,
        original_tokens=total,
    )


def _pack(plan: SpatialMergePlan, x: torch.Tensor, *, reduce: bool) -> torch.Tensor:
    batch, _, channels = x.shape
    src = x[:, plan.src_indices]
    dst = x[:, plan.dst_indices]
    merged = src.gather(1, plan.merged_sources.unsqueeze(-1).expand(batch, -1, channels))
    if reduce:
        dst = dst.scatter_reduce(1, plan.merged_destinations.unsqueeze(-1).expand(batch, -1, channels), merged, reduce="mean")
    kept = src.index_select(1, plan.kept_source_positions)
    protected = x[:, plan.protected_indices]
    return torch.cat([kept, dst, protected], dim=1)


def fastvggt_attention(attention, x: torch.Tensor, pos: torch.Tensor | None, plan: SpatialMergePlan | None) -> torch.Tensor:
    """Run one attention module, packing Q/K/V and restoring its output."""
    if plan is None:
        return attention(x, pos=pos)
    batch, _, channels = x.shape
    qkv = attention.qkv(x).reshape(batch, -1, 3, attention.num_heads, attention.head_dim).permute(2, 0, 3, 1, 4)
    query, key, value = attention.q_norm(qkv[0]), attention.k_norm(qkv[1]), qkv[2]
    if attention.rope is not None:
        query, key = attention.rope(query, pos), attention.rope(key, pos)
    heads = attention.num_heads
    query = _pack(plan, query.permute(0, 2, 1, 3).reshape(batch, -1, channels), reduce=True)
    key = _pack(plan, key.permute(0, 2, 1, 3).reshape(batch, -1, channels), reduce=True)
    value = _pack(plan, value.permute(0, 2, 1, 3).reshape(batch, -1, channels), reduce=True)
    query = query.reshape(batch, -1, heads, attention.head_dim).transpose(1, 2)
    key = key.reshape(batch, -1, heads, attention.head_dim).transpose(1, 2)
    value = value.reshape(batch, -1, heads, attention.head_dim).transpose(1, 2)
    output = F.scaled_dot_product_attention(query, key, value, dropout_p=attention.attn_drop.p if attention.training else 0.0)
    output = output.transpose(1, 2).reshape(batch, -1, channels)
    output = attention.proj_drop(attention.proj(output))
    kept_count = plan.src_indices.numel() - plan.merged_sources.shape[1]
    dst_count = plan.dst_indices.numel()
    kept, destinations = output[:, :kept_count], output[:, kept_count : kept_count + dst_count]
    # SDPA returns the autocast attention dtype (typically bf16), while the
    # normalized residual input can remain fp32.  Restore in the attention
    # output dtype so indexed writes are type-safe, then let the surrounding
    # residual addition perform the standard promotion.
    restored = torch.zeros(batch, plan.original_tokens, channels, device=x.device, dtype=output.dtype)
    merged_values = destinations.gather(1, plan.merged_destinations.unsqueeze(-1).expand(batch, -1, channels))
    restored[:, plan.dst_indices] = destinations
    restored[:, plan.src_indices[plan.kept_source_positions]] = kept
    restored[:, plan.src_indices[plan.merged_sources[0]]] = merged_values
    restored[:, plan.protected_indices] = output[:, kept_count + dst_count :]
    return restored
