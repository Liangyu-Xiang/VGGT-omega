# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from vggt.models.acceleration import (
    FrameMergeState,
    build_fastvggt_plan,
    fastvggt_attention,
    merge_frames,
    parse_block_indices,
    parse_layer_ratio_schedule,
    restore_frames,
)
from vggt.models.selftr import build_selftr_plan, selftr_attention

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        cached_layer_indices: Tuple[int, ...] = (4, 11, 17, 23),
        enable_token_merging: bool = False,
        token_merging_ratio: float = 0.9,
        token_merging_layer_ratios: str = "",
        token_merging_method: str = "spatial",
        token_merging_start: int = 0,
        token_merging_frame_restore_layer: int = 24,
        token_merging_frame_alpha: float = 0.1,
        token_merging_frame_segment_threshold: float = 0.9,
        token_merging_frame_merge_threshold: float = 0.1,
        token_merging_frame_max_window: int = 20,
        token_merging_frame_pool_stride: int = 2,
        token_merging_frame_multi_max_group_size: int = 4,
        token_merging_frame_multi_pair_threshold: float = 0.986,
        token_merging_frame_multi_span_threshold: float = 0.948,
        skip_global_attention_blocks: str = "",
        um_lambda_cost: float | None = None,
        um_spatial_radius: int = 2,
        um_temporal_window: int = 4,
        um_refresh_layers: str = "0,9,21",
        um_representative_mode: str = "mean",
        um_policy: str = "deltae-adaptive",
        um_target_full_attention_ratio: float | None = None,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.cached_layer_indices.add(depth - 1)
        self.enable_token_merging = enable_token_merging
        self.token_merging_ratio = token_merging_ratio
        self.token_merging_layer_ratios = token_merging_layer_ratios
        self.token_merging_method = token_merging_method
        self.token_merging_start = token_merging_start
        self.token_merging_frame_restore_layer = token_merging_frame_restore_layer
        self.token_merging_frame_alpha = token_merging_frame_alpha
        self.token_merging_frame_segment_threshold = token_merging_frame_segment_threshold
        self.token_merging_frame_merge_threshold = token_merging_frame_merge_threshold
        self.token_merging_frame_max_window = token_merging_frame_max_window
        self.token_merging_frame_pool_stride = token_merging_frame_pool_stride
        self.token_merging_frame_multi_max_group_size = token_merging_frame_multi_max_group_size
        self.token_merging_frame_multi_pair_threshold = token_merging_frame_multi_pair_threshold
        self.token_merging_frame_multi_span_threshold = token_merging_frame_multi_span_threshold
        self._layer_merge_ratios = parse_layer_ratio_schedule(token_merging_layer_ratios, depth)
        self.skip_global_attention_blocks = parse_block_indices(skip_global_attention_blocks, depth)
        self.last_frame_merge_stats: list[dict] = []
        self.last_token_merging_stats: list[dict] = []
        self.um_lambda_cost = um_lambda_cost
        self.um_spatial_radius = int(um_spatial_radius)
        self.um_temporal_window = int(um_temporal_window)
        self.um_refresh_layers = parse_block_indices(um_refresh_layers, depth) if um_lambda_cost is not None else set()
        if um_representative_mode not in {"mean", "last"}:
            raise ValueError("um_representative_mode must be 'mean' or 'last'")
        self.um_representative_mode = um_representative_mode
        if um_policy not in {"cosine-fixed", "deltae-fixed", "deltae-adaptive"}:
            raise ValueError("um_policy must be cosine-fixed, deltae-fixed, or deltae-adaptive")
        if um_policy.endswith("-fixed") and um_target_full_attention_ratio is None:
            raise ValueError("fixed U-M policies require um_target_full_attention_ratio")
        self.um_policy = um_policy
        self.um_target_full_attention_ratio = um_target_full_attention_ratio
        self._um_plan = None

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    @torch.inference_mode()
    def forward_dino(self, images: torch.Tensor, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a sequence once and retain DINO patch tokens on CPU for DA-VGGT.

        The returned patch tokens are exactly those consumed by ``forward_from_patch_tokens``;
        pooled tokens are float32 descriptors used only for DA partitioning.
        """
        B, S, channels, height, width = images.shape
        if B != 1 or channels != 3:
            raise ValueError("DA-VGGT token caching currently supports B=1 RGB videos")
        normalized = (images - self._resnet_mean) / self._resnet_std
        flat = normalized.reshape(B * S, channels, height, width)
        patches, pooled = [], []
        for start in range(0, B * S, batch_size):
            encoded = self.patch_embed(flat[start:start + batch_size])
            if isinstance(encoded, dict):
                encoded = encoded["x_norm_patchtokens"]
            patches.append(encoded.cpu())
            pooled.append(encoded.float().mean(dim=1).cpu())
        return torch.cat(patches, dim=0), torch.cat(pooled, dim=0)

    def forward(self, images: torch.Tensor) -> Tuple[List[Optional[torch.Tensor]], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor | None], int):
                The list of cached outputs from the attention blocks. Entries for
                uncached layers are None so layer indices remain stable.
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        normalized = (images - self._resnet_mean) / self._resnet_std
        patch_tokens = self.patch_embed(normalized.view(B * S, C_in, H, W))

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        return self.forward_from_patch_tokens(patch_tokens, B, S, H, W)

    def forward_from_patch_tokens(
        self, patch_tokens: torch.Tensor, B: int, S: int, H: int, W: int
    ) -> Tuple[List[Optional[torch.Tensor]], int]:
        """Run alternating attention from cached DINO patch tokens (DA-VGGT entry)."""
        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=patch_tokens.device
            )

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, self.patch_start_idx, 2, device=patch_tokens.device, dtype=pos.dtype
            )
            pos = torch.cat([pos_special, pos], dim=1)
        full_pos = pos

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        self._um_plan = None
        frame_merge_states: list[FrameMergeState] | None = None
        active_frames = S

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, active_frames, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, active_frames, P, C, global_idx, pos=pos,
                        patch_grid_size=(H // self.patch_size, W // self.patch_size),
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                layer_idx = len(output_list)
                if layer_idx in self.cached_layer_indices:
                    # concat frame and global intermediates, [B x S x P x 2C]
                    frame_output = frame_intermediates[i]
                    global_output = global_intermediates[i]
                    if frame_merge_states is not None:
                        frame_output = restore_frames(frame_output, frame_merge_states)
                        global_output = restore_frames(global_output, frame_merge_states)
                    concat_inter = torch.cat([frame_output, global_output], dim=-1)
                    output_list.append(concat_inter)
                else:
                    output_list.append(None)

            block_idx = len(output_list) - 1
            if (
                self.enable_token_merging
                and self.token_merging_method == "frame_persistent_spatial"
                and frame_merge_states is None
                and block_idx == self.token_merging_start
                and block_idx < self.token_merging_frame_restore_layer
            ):
                merged_batches = []
                frame_merge_states = []
                for batch_idx in range(B):
                    merged, state = merge_frames(
                        tokens.view(B, active_frames, P, C)[batch_idx],
                        patch_start=self.patch_start_idx,
                        grid_size=(H // self.patch_size, W // self.patch_size),
                        alpha=self.token_merging_frame_alpha,
                        segment_threshold=self.token_merging_frame_segment_threshold,
                        merge_threshold=self.token_merging_frame_merge_threshold,
                        max_window=self.token_merging_frame_max_window,
                        pool_stride=self.token_merging_frame_pool_stride,
                        max_group_size=self.token_merging_frame_multi_max_group_size,
                        pair_threshold=self.token_merging_frame_multi_pair_threshold,
                        span_threshold=self.token_merging_frame_multi_span_threshold,
                    )
                    merged_batches.append(merged)
                    frame_merge_states.append(state)
                counts = {state.active_frames for state in frame_merge_states}
                if len(counts) != 1:
                    raise ValueError("Frame merging produced unequal active-frame counts across the batch")
                active_frames = counts.pop()
                tokens = torch.stack(merged_batches).view(B, active_frames, P, C)
                if full_pos is not None:
                    pos = full_pos.view(B, S, P, 2)[:, :active_frames].reshape(B * active_frames, P, 2)
                self.last_frame_merge_stats.append(
                    {
                        "block": block_idx,
                        "original_frames": S,
                        "active_frames_min": int(active_frames),
                        "active_frames_mean": float(active_frames),
                        "active_frames_max": int(active_frames),
                        "retention_ratio_mean": float(active_frames / S),
                        "merge_ratio_mean": float(1.0 - active_frames / S),
                        "segments": [[list(pair) for pair in state.segments] for state in frame_merge_states],
                        "merge_groups": [state.merge_groups for state in frame_merge_states],
                    }
                )

            if frame_merge_states is not None and block_idx + 1 == self.token_merging_frame_restore_layer:
                tokens = restore_frames(tokens.view(B, active_frames, P, C), frame_merge_states)
                frame_merge_states = None
                active_frames = S
                pos = full_pos

        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, patch_grid_size=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            block_idx = global_idx
            merge_ratio = self._layer_merge_ratios.get(block_idx, self.token_merging_ratio)
            use_fast_merge = (
                self.enable_token_merging
                and self.token_merging_method in {"spatial", "frame_persistent_spatial"}
                and merge_ratio > 0.0
            )
            use_um = self.um_lambda_cost is not None
            if block_idx in self.skip_global_attention_blocks:
                tokens = tokens.view(B, S, P, C)
            elif use_um:
                if self._um_plan is None or block_idx in self.um_refresh_layers:
                    self._um_plan = build_selftr_plan(
                        tokens,
                        num_frames=S,
                        patch_start=self.patch_start_idx,
                        grid_size=patch_grid_size,
                        spatial_radius=self.um_spatial_radius,
                        temporal_window=self.um_temporal_window,
                        lambda_cost=self.um_lambda_cost,
                        policy=self.um_policy,
                        target_full_attention_ratio=self.um_target_full_attention_ratio,
                    )
                residual = selftr_attention(
                    self.global_blocks[block_idx].attn,
                    tokens,
                    pos,
                    self._um_plan,
                    norm1=self.global_blocks[block_idx].norm1,
                    representative_mode=self.um_representative_mode,
                )
                tokens = tokens + self.global_blocks[block_idx].ls1(residual)
                tokens = tokens + self.global_blocks[block_idx].ls2(self.global_blocks[block_idx].mlp(self.global_blocks[block_idx].norm2(tokens)))
                self.last_token_merging_stats.append({
                    "block": block_idx, "mode": "u-m",
                    "original_tokens": int(S * P),
                    "active_tokens": int(S * self.patch_start_idx + self._um_plan.representative_source_indices.numel()),
                    "full_attention_token_ratio": float((S * self.patch_start_idx + self._um_plan.representative_source_indices.numel()) / (S * P)),
                    "merged_away_token_ratio": float(1.0 - (S * self.patch_start_idx + self._um_plan.representative_source_indices.numel()) / (S * P)),
                    "um_edge_score_backend": self._um_plan.edge_score_backend,
                })
                tokens = tokens.view(B, S, P, C)
            elif use_fast_merge:
                plan = build_fastvggt_plan(
                    self.global_blocks[block_idx].norm1(tokens),
                    num_frames=S,
                    patch_start=self.patch_start_idx,
                    grid_size=patch_grid_size,
                    merge_ratio=merge_ratio,
                )
                residual = fastvggt_attention(self.global_blocks[block_idx].attn, self.global_blocks[block_idx].norm1(tokens), pos, plan)
                tokens = tokens + self.global_blocks[block_idx].ls1(residual)
                tokens = tokens + self.global_blocks[block_idx].ls2(self.global_blocks[block_idx].mlp(self.global_blocks[block_idx].norm2(tokens)))
                if plan is not None:
                    self.last_token_merging_stats.append(
                        {
                            "block": block_idx,
                            "original_tokens": plan.original_tokens,
                            "active_tokens": plan.active_tokens,
                            "full_attention_token_ratio": plan.active_tokens / plan.original_tokens,
                            "merged_away_token_ratio": 1.0 - plan.active_tokens / plan.original_tokens,
                        }
                    )
                tokens = tokens.view(B, S, P, C)
            elif self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
