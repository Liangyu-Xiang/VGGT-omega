"""Diversity-aware VGGT chunk scheduling, shared by native model adapters.

This is the DA-VGGT pipeline: appearance features -> diversity partition ->
pseudo-pose weighting -> re-partition.  It deliberately operates on cached
encoder tokens; callers provide the model-specific chunk executor.
"""
from __future__ import annotations

import numpy as np


def cosine_similarity(features):
    x = features.float().numpy()
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return np.clip(x @ x.T, -1.0, 1.0)


def _anchors(n: int, count: int) -> list[int]:
    return list(dict.fromkeys(round(i * (n - 1) / max(count - 1, 1)) for i in range(max(1, count))))


def _insert_anchors(groups, anchors):
    anchor_set = set(anchors)
    return [[anchors[0], *[i for i in group if i not in anchor_set]] for group in groups]


def diversity_partition(sim: np.ndarray, chunk_size: int, anchors: list[int], iters: int = 5):
    """Balanced reverse-similarity partition with 2-opt refinement (DA stage 1)."""
    n = len(sim); groups = [[] for _ in range((n + chunk_size - 1) // chunk_size)]
    capacity = max(1, (n + len(groups) - 1) // len(groups))
    utility = 1.0 - sim; np.fill_diagonal(utility, 0.0)
    remaining = set(range(n)) - set(anchors)
    for group in groups:
        while remaining and len(group) < capacity:
            score = [(utility[i, group].sum() if group else utility[i].sum(), i) for i in remaining]
            _, chosen = max(score); group.append(chosen); remaining.remove(chosen)
    for i in remaining: min(groups, key=len).append(i)
    for _ in range(iters):
        changed = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                best = (0.0, None, None)
                for ia, x in enumerate(groups[a]):
                    for ib, y in enumerate(groups[b]):
                        before = utility[x, groups[a]].sum() + utility[y, groups[b]].sum()
                        after = utility[y, groups[a]].sum() + utility[x, groups[b]].sum()
                        if after - before > best[0]: best = (after - before, ia, ib)
                if best[1] is not None:
                    ia, ib = best[1:]; groups[a][ia], groups[b][ib] = groups[b][ib], groups[a][ia]; changed = True
        if not changed: break
    return _insert_anchors(groups, anchors)


def pseudo_positions(sim: np.ndarray, chunk_indices, positions: np.ndarray, gamma: float):
    n = len(sim); out = np.zeros((n, 3), dtype=np.float64); known = np.asarray(chunk_indices)
    out[known] = positions
    for i in range(n):
        if i in set(chunk_indices): continue
        logits = sim[i, known] / max(gamma, 1e-6); logits -= logits.max()
        weight = np.exp(logits); weight /= weight.sum()
        out[i] = weight @ positions
    return out


def pose_weighted_similarity(sim: np.ndarray, pseudo: np.ndarray, tau: float | None = None):
    distance = np.linalg.norm(pseudo[:, None] - pseudo[None, :], axis=-1)
    tau = float(np.median(distance[distance > 0])) if tau is None and np.any(distance > 0) else (tau or 1.0)
    return sim * np.exp(-distance / max(tau, 1e-6))
