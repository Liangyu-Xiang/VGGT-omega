# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.models.da_vggt import cosine_similarity, diversity_partition, pseudo_positions, pose_weighted_similarity
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
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
        self.explicit_bfloat16_inference = False

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_token_merging=enable_token_merging,
            token_merging_ratio=token_merging_ratio,
            token_merging_layer_ratios=token_merging_layer_ratios,
            token_merging_method=token_merging_method,
            token_merging_start=token_merging_start,
            token_merging_frame_restore_layer=token_merging_frame_restore_layer,
            token_merging_frame_alpha=token_merging_frame_alpha,
            token_merging_frame_segment_threshold=token_merging_frame_segment_threshold,
            token_merging_frame_merge_threshold=token_merging_frame_merge_threshold,
            token_merging_frame_max_window=token_merging_frame_max_window,
            token_merging_frame_pool_stride=token_merging_frame_pool_stride,
            token_merging_frame_multi_max_group_size=token_merging_frame_multi_max_group_size,
            token_merging_frame_multi_pair_threshold=token_merging_frame_multi_pair_threshold,
            token_merging_frame_multi_span_threshold=token_merging_frame_multi_span_threshold,
            skip_global_attention_blocks=skip_global_attention_blocks,
            um_lambda_cost=um_lambda_cost,
            um_spatial_radius=um_spatial_radius,
            um_temporal_window=um_temporal_window,
            um_refresh_layers=um_refresh_layers,
            um_representative_mode=um_representative_mode,
            um_policy=um_policy,
            um_target_full_attention_ratio=um_target_full_attention_ratio,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        # The legacy evaluator keeps heads in FP32 after an autocast encoder.
        # FastVGGT's long-video setup instead converts the whole model to bf16;
        # retain bf16 through the heads in that explicit inference mode.
        head_autocast = bool(self.explicit_bfloat16_inference and images.is_cuda)
        with torch.autocast(
            device_type="cuda" if images.is_cuda else "cpu",
            dtype=torch.bfloat16,
            enabled=head_autocast,
        ):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

    @torch.inference_mode()
    def forward_da_vggt(
        self,
        images: torch.Tensor,
        chunk_size: int = 50,
        dino_batch_size: int = 32,
        local_search_iters: int = 5,
        pseudo_pose_gamma: float = 1e-3,
        pose_tau: float | None = None,
    ) -> dict:
        """Native DA-VGGT inference with cached DINO tokens and pose re-chunking.

        This follows the source method's default one-anchor protocol: DINO
        features are extracted once, reverse-similarity chunks are refined by
        2-opt, a first chunk supplies pseudo poses, then all frames are
        re-partitioned using appearance × pose affinity before chunk inference.
        """
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.shape[0] != 1 or self.camera_head is None:
            raise ValueError("DA-VGGT requires B=1 and an enabled camera head")
        _, frame_count, _, height, width = images.shape
        if frame_count <= chunk_size:
            return self(images)

        patch_tokens_cpu, pooled = self.aggregator.forward_dino(images, batch_size=dino_batch_size)
        similarity = cosine_similarity(pooled)
        anchors = [0]
        initial = diversity_partition(similarity, chunk_size, anchors, local_search_iters)

        def run_chunk(indices):
            ids = torch.tensor(indices, device=images.device, dtype=torch.long)
            tokens = patch_tokens_cpu[ids.cpu()].to(images.device, non_blocking=True)
            aggregate, patch_start = self.aggregator.forward_from_patch_tokens(
                tokens, 1, len(indices), height, width
            )
            pose_list = self.camera_head(aggregate)
            pose = pose_list[-1]
            depth, depth_conf = self.depth_head(
                aggregate, images=images[:, ids], patch_start_idx=patch_start
            ) if self.depth_head is not None else (None, None)
            return pose, depth, depth_conf

        pose0, _, _ = run_chunk(initial[0])
        ext0, _ = pose_encoding_to_extri_intri(pose0, (height, width), build_intrinsics=False)
        c2w0 = torch.linalg.inv(_to_homogeneous(ext0[0])).cpu().numpy()
        pseudo = pseudo_positions(similarity, initial[0], c2w0[:, :3, 3], pseudo_pose_gamma)
        refined = diversity_partition(pose_weighted_similarity(similarity, pseudo, pose_tau), chunk_size, anchors, local_search_iters)

        poses, depths, depth_confs = [None] * frame_count, [None] * frame_count, [None] * frame_count
        reference_c2w = None
        for indices in refined:
            pose, depth, depth_conf = run_chunk(indices)
            ext, _ = pose_encoding_to_extri_intri(pose, (height, width), build_intrinsics=False)
            with torch.amp.autocast("cuda", enabled=False):
                c2w = torch.linalg.inv(_to_homogeneous(ext[0]).float())
                if reference_c2w is None:
                    reference_c2w = c2w[0].clone()
                aligned_c2w = reference_c2w @ torch.linalg.inv(c2w[0]) @ c2w
                aligned_w2c = torch.linalg.inv(aligned_c2w)[:, :3]
            for local, original in enumerate(indices):
                if original == 0 and poses[0] is not None:
                    continue
                poses[original] = aligned_w2c[local]
                if depth is not None:
                    depths[original], depth_confs[original] = depth[0, local], depth_conf[0, local]
        result = {
            "da_w2c": torch.stack(poses).unsqueeze(0),
            "chunk_frame_indices": refined,
            "initial_chunk_frame_indices": initial,
            "da_similarity": similarity,
            "images": images,
        }
        if depths[0] is not None:
            result["depth"] = torch.stack(depths).unsqueeze(0)
            result["depth_conf"] = torch.stack(depth_confs).unsqueeze(0)
        return result


def _to_homogeneous(extrinsics: torch.Tensor) -> torch.Tensor:
    bottom = torch.zeros((*extrinsics.shape[:-2], 1, 4), device=extrinsics.device, dtype=extrinsics.dtype)
    bottom[..., 0, 3] = 1
    return torch.cat((extrinsics, bottom), dim=-2)
