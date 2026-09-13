"""Deprecated compatibility import for the former U-M Triton module."""

from .selftr_triton import fused_selftr_edge_cost


fused_um_edge_cost = fused_selftr_edge_cost

__all__ = ["fused_um_edge_cost"]
