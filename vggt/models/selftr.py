"""SelfTR planner and representative attention for original VGGT.

This is the same U-M semantics as VGGT-Omega: frame 0 is protected; the
remaining patch tokens are greedily coalesced on an r-by-r, t-frame local
cube by exact whole-group cosine reconstruction cost; every final group is
represented by the mean *token* before Q/K/V projection.  Only the global
attention residual is compressed and expanded; Pi3's MLP still receives every
original token.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

# The reference PyTorch planner is self-contained.  It deliberately has no
# machine-local Omega or Pi3 dependency.
_fused_um_edge_cost = None


@dataclass(frozen=True)
class UMPlan:
    position_to_representative: torch.Tensor  # [F, P]
    representative_source_indices: torch.Tensor  # patch-only global indices
    representative_weights: torch.Tensor
    special_tokens: int
    num_frames: int
    patch_count: int
    edge_score_backend: str


def _cube_edges(num_frames: int, height: int, width: int, radius: int, window: int, device: torch.device):
    """Omega's canonical U-M topology, with protected frame zero removed."""
    if radius <= 0 or window <= 0:
        raise ValueError("U-M spatial radius and temporal window must be positive")
    patch_count = height * width
    pieces_l, pieces_r = [], []
    grid = torch.arange(patch_count, device=device, dtype=torch.long).view(height, width)

    # Same-frame edges use one half of the cube, hence every undirected edge
    # is emitted exactly once.
    spatial = [(dr, dc) for dr in range(-radius, radius + 1)
               for dc in range(-radius, radius + 1) if dr > 0 or (dr == 0 and dc > 0)]
    temporal = [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)]

    def append_edges(frame_begin: int, frame_end: int, delta: int, dr: int, dc: int):
        row_begin, row_end = max(-dr, 0), min(height - dr, height)
        col_begin, col_end = max(-dc, 0), min(width - dc, width)
        if frame_end <= frame_begin or row_begin >= row_end or col_begin >= col_end:
            return
        src = grid[row_begin:row_end, col_begin:col_end].reshape(-1)
        dst = grid[row_begin + dr:row_end + dr, col_begin + dc:col_end + dc].reshape(-1)
        frames = torch.arange(frame_begin, frame_end, device=device, dtype=torch.long)
        # Subtract one protected frame: local planner ids begin at frame 1.
        pieces_l.append((frames[:, None] * patch_count + src[None, :] - patch_count).reshape(-1))
        pieces_r.append(((frames[:, None] + delta) * patch_count + dst[None, :] - patch_count).reshape(-1))

    for dr, dc in spatial:
        append_edges(1, num_frames, 0, dr, dc)
    for delta in range(1, window + 1):
        for dr, dc in temporal:
            append_edges(1, num_frames - delta, delta, dr, dc)
    if not pieces_l:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
    left, right = torch.cat(pieces_l), torch.cat(pieces_r)
    count = (num_frames - 1) * patch_count
    keys = left.to(torch.int64) * count + right
    order = torch.argsort(keys, stable=True)
    return left[order], right[order]


def _canonicalize_edges(left: torch.Tensor, right: torch.Tensor, count: int):
    valid = left != right
    left, right = left[valid], right[valid]
    if left.numel() == 0:
        return left, right
    low, high = torch.minimum(left, right), torch.maximum(left, right)
    keys = torch.unique(low.to(torch.int64) * count + high, sorted=True)
    return keys // count, keys % count


def _merge_groups(
    features: torch.Tensor,
    edge_left: torch.Tensor,
    edge_right: torch.Tensor,
    lambda_cost: float,
    *,
    policy: str = "deltae-adaptive",
    target_count: int | None = None,
    min_keep_ratio: float = 0.05,
):
    """GPU batch mutual-nearest U-M merge.

    ``deltae-adaptive`` is the STAR planner: it accepts a mutual pair only
    when its whole-group marginal distortion is below ``2 * lambda``.
    The two fixed-budget ablations keep the same graph and mutual scheduler,
    but replace the stopping rule with a target group count.  ``cosine-fixed``
    ranks edges by the current parent-token cosine distance, while
    ``deltae-fixed`` ranks them by the exact whole-group marginal distortion.
    """
    if policy not in {"cosine-fixed", "deltae-fixed", "deltae-adaptive"}:
        raise ValueError(f"Unknown U-M policy: {policy!r}")
    is_fixed = policy.endswith("-fixed")
    if is_fixed and target_count is None:
        raise ValueError(f"{policy} requires target_count")
    count, dim = features.shape
    weights = torch.ones(count, device=features.device, dtype=torch.float32)
    sums = features.contiguous()
    representatives = torch.arange(count, device=features.device, dtype=torch.long)
    errors = torch.zeros(count, device=features.device, dtype=torch.float32)
    labels = representatives.clone()
    min_keep = max(int(math.ceil(count * min_keep_ratio)), 1)
    if is_fixed:
        min_keep = max(min(int(target_count), count), 1)
    backend = "pytorch"

    while edge_left.numel() and weights.numel() > min_keep:
        component_count = int(weights.numel())
        valid = torch.ones(edge_left.numel(), dtype=torch.bool, device=features.device)
        edge_cost = None
        if policy == "cosine-fixed":
            # The local cube contains millions of edges for 300 frames.  Do
            # not materialize all [num_edges, dim] indexed representatives at
            # once; that can exceed the model's memory by tens of GiB.
            edge_cost = torch.empty(edge_left.numel(), device=features.device, dtype=torch.float32)
            chunk_size = 32768
            for start in range(0, edge_left.numel(), chunk_size):
                end = min(start + chunk_size, edge_left.numel())
                left_rep = representatives.index_select(0, edge_left[start:end])
                right_rep = representatives.index_select(0, edge_right[start:end])
                edge_cost[start:end] = 1.0 - (
                    features.index_select(0, left_rep) * features.index_select(0, right_rep)
                ).sum(-1)
        elif _fused_um_edge_cost is not None:
            edge_cost = _fused_um_edge_cost(
                sums.contiguous(), weights.contiguous(), representatives.contiguous(), errors.contiguous(),
                features.contiguous(), edge_left.contiguous(), edge_right.contiguous(), valid,
                prefer_best_parent=True,
            )
        if edge_cost is None:
            # ``edge_left`` has tens of millions of entries for a 300-frame
            # VGGT video.  Indexing ``sums[edge_left]`` in one operation
            # creates an [edges, 1024] tensor (over 100 GiB), although only
            # the scalar marginal cost for each edge is needed.  Keep the
            # exact U-M objective but score independent edge chunks.
            edge_cost = torch.empty(edge_left.numel(), device=features.device, dtype=torch.float32)
            chunk_size = 8192
            for start in range(0, edge_left.numel(), chunk_size):
                end = min(start + chunk_size, edge_left.numel())
                left_ids = edge_left[start:end]
                right_ids = edge_right[start:end]
                merged_sum = sums.index_select(0, left_ids) + sums.index_select(0, right_ids)
                merged_weight = weights.index_select(0, left_ids) + weights.index_select(0, right_ids)
                left_error = merged_weight - (
                    merged_sum * features.index_select(0, representatives.index_select(0, left_ids))
                ).sum(-1)
                right_error = merged_weight - (
                    merged_sum * features.index_select(0, representatives.index_select(0, right_ids))
                ).sum(-1)
                edge_cost[start:end] = (
                    torch.minimum(left_error, right_error)
                    - errors.index_select(0, left_ids)
                    - errors.index_select(0, right_ids)
                )
        else:
            backend = "triton_fused"

        directed_group = torch.cat((edge_left, edge_right))
        directed_neighbor = torch.cat((edge_right, edge_left))
        directed_cost = torch.cat((edge_cost, edge_cost))
        best_cost = torch.full((component_count,), float("inf"), device=features.device)
        best_cost.scatter_reduce_(0, directed_group, directed_cost, reduce="amin", include_self=True)
        tie = torch.where(directed_cost == best_cost[directed_group], directed_neighbor, torch.full_like(directed_neighbor, component_count))
        best = torch.full((component_count,), component_count, device=features.device, dtype=torch.long)
        best.scatter_reduce_(0, directed_group, tie, reduce="amin", include_self=True)

        ids = torch.arange(component_count, device=features.device)
        left = ids[(ids < best) & (best < component_count)]
        right = best[left]
        mutual = best[right] == left
        left, right = left[mutual], right[mutual]
        if left.numel() == 0:
            break
        merged_sum = sums[left] + sums[right]
        merged_weight = weights[left] + weights[right]
        left_error = merged_weight - (merged_sum * features[representatives[left]]).sum(-1)
        right_error = merged_weight - (merged_sum * features[representatives[right]]).sum(-1)
        delta = torch.minimum(left_error, right_error) - errors[left] - errors[right]
        choose_right = right_error < left_error
        acceptable = torch.ones_like(delta, dtype=torch.bool) if is_fixed else delta < (2.0 * float(lambda_cost))
        available = max(component_count - min_keep, 0)
        accepted = torch.nonzero(acceptable, as_tuple=False).flatten()
        if accepted.numel() > available:
            order = torch.argsort(delta[accepted], stable=True)
            acceptable[accepted[order[available:]]] = False
        left, right, choose_right = left[acceptable], right[acceptable], choose_right[acceptable]
        if left.numel() == 0:
            break

        paired = torch.zeros(component_count, dtype=torch.bool, device=features.device)
        paired[left], paired[right] = True, True
        leader = ~paired
        leader[left] = True
        leader_ids = torch.nonzero(leader, as_tuple=False).flatten()
        new_ids = torch.cumsum(leader.long(), 0) - 1
        old_to_new = new_ids.clone()
        old_to_new[right] = new_ids[left]
        new_count = int(leader_ids.numel())
        new_weights = torch.zeros(new_count, device=features.device)
        new_weights.index_add_(0, old_to_new, weights)
        new_sums = torch.zeros((new_count, dim), device=features.device)
        new_sums.index_add_(0, old_to_new, sums)
        new_reps = representatives[leader_ids].clone()
        new_reps[new_ids[left]] = torch.where(choose_right, representatives[right], representatives[left])
        labels = old_to_new[labels]
        edge_left, edge_right = _canonicalize_edges(old_to_new[edge_left], old_to_new[edge_right], new_count)
        weights, sums, representatives = new_weights, new_sums, new_reps
        errors = weights - (sums * features[representatives]).sum(-1)
    return labels, representatives, backend


def build_um_plan(
    metric,
    num_frames,
    special_tokens,
    grid_size,
    spatial_radius=2,
    temporal_window=4,
    lambda_cost=0.04,
    policy="deltae-adaptive",
    target_full_attention_ratio=None,
):
    if metric.shape[0] != 1:
        raise ValueError("SelfTR currently requires batch size 1")
    height, width = grid_size
    patch_count = height * width
    per_frame = special_tokens + patch_count
    if metric.shape[1] != num_frames * per_frame:
        raise ValueError("SelfTR token layout mismatch")
    if num_frames < 2:
        mapping = torch.arange(patch_count, device=metric.device).view(1, patch_count)
        return UMPlan(mapping, torch.arange(patch_count, device=metric.device), torch.ones(patch_count, device=metric.device), special_tokens, num_frames, patch_count, "pytorch")
    patch = metric[0].view(num_frames, per_frame, -1)[:, special_tokens:].reshape(-1, metric.shape[-1])
    features = F.normalize(patch.float(), dim=-1, eps=1e-8).contiguous()
    edge_left, edge_right = _cube_edges(num_frames, height, width, spatial_radius, temporal_window, metric.device)
    target_count = None
    if policy.endswith("-fixed"):
        if target_full_attention_ratio is None:
            raise ValueError(f"{policy} requires target_full_attention_ratio")
        mergeable = (num_frames - 1) * patch_count
        protected = num_frames * special_tokens + patch_count
        total_tokens = num_frames * per_frame
        target_active = int(math.ceil(float(target_full_attention_ratio) * total_tokens))
        target_count = max(1, min(mergeable, target_active - protected))
    labels, reps, backend = _merge_groups(
        features[patch_count:], edge_left, edge_right, lambda_cost,
        policy=policy, target_count=target_count,
    )
    mapping = torch.empty(num_frames * patch_count, dtype=torch.long, device=metric.device)
    mapping[:patch_count] = torch.arange(patch_count, device=metric.device)
    mapping[patch_count:] = patch_count + labels
    sources = torch.cat((torch.arange(patch_count, device=metric.device), patch_count + reps))
    weights = torch.bincount(mapping, minlength=sources.numel()).float()
    return UMPlan(mapping.view(num_frames, patch_count), sources, weights, special_tokens, num_frames, patch_count, backend)


def _mean_groups(values: torch.Tensor, mapping: torch.Tensor, count: int):
    sums = torch.zeros((count, values.shape[-1]), device=values.device, dtype=values.dtype)
    sums.index_add_(0, mapping, values)
    counts = torch.bincount(mapping, minlength=count).to(dtype=values.dtype).clamp_min_(1)
    return sums / counts[:, None]


def um_attention(attn, x, plan: UMPlan, *, norm1, xpos=None, representative_mode="mean"):
    """Compress VGGT global attention with SelfTR representatives."""
    B, total, channels = x.shape
    if B != 1 or total != plan.num_frames * (plan.special_tokens + plan.patch_count):
        raise ValueError("SelfTR attention layout mismatch")
    frame_tokens = x.view(plan.num_frames, plan.special_tokens + plan.patch_count, channels)
    special = frame_tokens[:, :plan.special_tokens].reshape(-1, channels)
    patch = frame_tokens[:, plan.special_tokens:].reshape(-1, channels)
    mapping = plan.position_to_representative.reshape(-1)
    if representative_mode == "mean":
        representatives = _mean_groups(patch, mapping, int(plan.representative_source_indices.numel()))
    elif representative_mode == "last":
        # Use the final parent selected by the U-M planner, without changing
        # the U-M partition or its representative selection objective.
        representatives = patch.index_select(0, plan.representative_source_indices)
    else:
        raise ValueError(f"Unknown U-M representative_mode={representative_mode!r}; expected 'mean' or 'last'")
    # This ordering is deliberate: U-M summarizes current *raw* token states,
    # then runs the block's normal norm1 and Q/K/V projection on the compact
    # sequence. Averaging normalized tokens or projected Q/K/V changes the
    # method's objective and is what the previous Pi3 adapter did.
    compressed = norm1(torch.cat((special, representatives)).unsqueeze(0))

    qkv = attn.qkv(compressed).reshape(1, compressed.shape[1], 3, attn.num_heads, channels // attn.num_heads).transpose(1, 3)
    q, k, v = [qkv[:, :, i] for i in range(3)]
    q, k = attn.q_norm(q).to(v.dtype), attn.k_norm(k).to(v.dtype)
    if attn.rope is not None:
        full_pos = xpos.view(plan.num_frames, plan.special_tokens + plan.patch_count, -1)
        compressed_pos = torch.cat((full_pos[:, :plan.special_tokens].reshape(-1, full_pos.shape[-1]), full_pos[:, plan.special_tokens:].reshape(-1, full_pos.shape[-1]).index_select(0, plan.representative_source_indices))).unsqueeze(0)
        q, k = attn.rope(q, compressed_pos), attn.rope(k, compressed_pos)
    if q.dtype == torch.bfloat16:
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v)
    else:
        out = F.scaled_dot_product_attention(q, k, v)
    update = attn.proj_drop(attn.proj(out.transpose(1, 2).reshape(1, -1, channels)))
    restored_patch = update[:, plan.num_frames * plan.special_tokens:].index_select(1, mapping)
    restored_special = update[:, :plan.num_frames * plan.special_tokens].view(1, plan.num_frames, plan.special_tokens, channels)
    return torch.cat((restored_special, restored_patch.view(1, plan.num_frames, plan.patch_count, channels)), dim=2).reshape(1, total, channels)


def build_selftr_plan(
    metric: torch.Tensor,
    *,
    num_frames: int,
    patch_start: int,
    grid_size: tuple[int, int],
    spatial_radius: int = 2,
    temporal_window: int = 4,
    lambda_cost: float = 0.04,
    policy: str = "deltae-adaptive",
    target_full_attention_ratio: float | None = None,
) -> UMPlan:
    """Build the original-VGGT SelfTR plan using the native token layout."""

    return build_um_plan(
        metric,
        num_frames,
        patch_start,
        grid_size,
        spatial_radius,
        temporal_window,
        lambda_cost,
        policy,
        target_full_attention_ratio,
    )


def selftr_attention(
    attention,
    tokens: torch.Tensor,
    pos: torch.Tensor | None,
    plan: UMPlan,
    *,
    norm1,
    representative_mode: str = "mean",
) -> torch.Tensor:
    """Adapt the original-VGGT global-block call convention to SelfTR."""

    return um_attention(
        attention,
        tokens,
        plan,
        norm1=norm1,
        xpos=pos,
        representative_mode=representative_mode,
    )
