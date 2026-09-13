import pytest
import numpy as np
import torch

from vggt_omega.models.aggregator import (
    Aggregator,
    FrameFusionBatchPlan,
    FrameFusionGroup,
    FrameFusionPair,
    _normalize_similarity_weights,
    compute_frame_fusion_partition,
    frame_fusion_attention_indices,
    pooled_frame_representations,
    select_frame_fusion_pairs,
    select_frame_fusion_pairs_from_normalized_representations,
    _connected_frame_fusion_groups,
    _sequential_frame_fusion_groups,
)
from vggt_omega.models.selftr_triton import fused_selftr_edge_cost


def test_frame_fusion_partition_finds_low_cost_contiguous_groups():
    distance = torch.ones(6, 6)
    distance.fill_diagonal_(0.0)
    distance[:3, :3] = 0.1
    distance[3:, 3:] = 0.1
    distance.fill_diagonal_(0.0)

    segments = compute_frame_fusion_partition(
        distance,
        num_groups=2,
        max_group_size=3,
        beta=1.0,
    )

    assert [(segment.start, segment.end) for segment in segments] == [(0, 2), (3, 5)]
    assert [segment.medoid for segment in segments] == [0, 3]


def test_frame_fusion_partition_rejects_infeasible_group_budget():
    distance = torch.zeros(7, 7)

    with pytest.raises(ValueError, match="No feasible frame partition"):
        compute_frame_fusion_partition(
            distance,
            num_groups=2,
            max_group_size=3,
        )


def test_normalize_similarity_weights_uses_nonnegative_similarity_mass():
    weights = _normalize_similarity_weights(torch.tensor([1.0, 2.0, -4.0]))

    assert torch.allclose(weights, torch.tensor([1.0 / 3.0, 2.0 / 3.0, 0.0]))


def test_normalize_similarity_weights_falls_back_to_uniform_when_mass_is_zero():
    weights = _normalize_similarity_weights(torch.tensor([-1.0, 0.0]))

    assert torch.allclose(weights, torch.tensor([0.5, 0.5]))


def test_pooled_frame_representations_use_size_two_average_pooling():
    patch_tokens = torch.tensor(
        [
            [
                [[1.0], [3.0], [5.0], [7.0]],
                [[2.0], [4.0], [6.0], [8.0]],
            ]
        ]
    )

    representations = pooled_frame_representations(
        patch_tokens,
        patch_grid_size=(2, 2),
        pool_size=2,
    )

    assert torch.allclose(representations, torch.tensor([[[4.0], [5.0]]]))


def test_n4_spacetime_cube_edges_cover_all_cross_frame_positions_once():
    aggregator = Aggregator.__new__(Aggregator)
    aggregator._frame_fusion_patch_grid_size = (3, 3)
    aggregator.frame_fusion_spatial_neighborhood = "N4"
    aggregator.frame_fusion_temporal_window = 2

    source, target = aggregator._build_local_spatiotemporal_edges(
        num_frames=3,
        patch_count=9,
        include_temporal_spatial=False,
    )

    edges = list(zip(source.tolist(), target.tolist()))
    undirected = {(min(left, right), max(left, right)) for left, right in edges}
    assert len(edges) == len(undirected)

    same_frame = [
        (left, right)
        for left, right in edges
        if left // 9 == right // 9
    ]
    cross_frame = [
        (left, right)
        for left, right in edges
        if left // 9 != right // 9
    ]
    # A 3x3 frame has 20 undirected edges in the four-direction half of an
    # 8-neighborhood.  The temporal cube contributes 49 valid edges for each
    # frame pair and each positive time delta.  With three frames and a
    # two-frame window, that gives three frame-pair/delta combinations.
    assert len(same_frame) == 3 * 20
    assert len(cross_frame) == 3 * 49

    center_to_next = {
        right % 9
        for left, right in edges
        if left == 4 and right // 9 == 1
    }
    center_to_two_ahead = {
        right % 9
        for left, right in edges
        if left == 4 and right // 9 == 2
    }
    assert center_to_next == set(range(9))
    assert center_to_two_ahead == set(range(9))

    corner_to_next = {
        right % 9
        for left, right in edges
        if left == 0 and right // 9 == 1
    }
    assert corner_to_next == {0, 1, 3, 4}


def test_n4_spacetime_cube_supports_five_by_five_space_and_four_frame_time_radius():
    aggregator = Aggregator.__new__(Aggregator)
    aggregator._frame_fusion_patch_grid_size = (5, 5)
    aggregator.frame_fusion_spatial_neighborhood = "N4"
    aggregator.frame_fusion_spatial_radius = 2
    aggregator.frame_fusion_temporal_window = 4

    source, target = aggregator._build_local_spatiotemporal_edges(
        num_frames=2,
        patch_count=25,
        include_temporal_spatial=False,
    )
    edges = list(zip(source.tolist(), target.tolist()))
    undirected = {(min(left, right), max(left, right)) for left, right in edges}
    assert len(edges) == len(undirected)

    center_to_next = {
        right % 25
        for left, right in edges
        if left == 12 and right // 25 == 1
    }
    corner_to_next = {
        right % 25
        for left, right in edges
        if left == 0 and right // 25 == 1
    }
    assert center_to_next == set(range(25))
    assert corner_to_next == {0, 1, 2, 5, 6, 7, 10, 11, 12}


def test_tensorized_spacetime_cube_matches_numpy_api_and_reuses_cache():
    aggregator = Aggregator.__new__(Aggregator)
    aggregator._frame_fusion_patch_grid_size = (3, 4)
    aggregator.frame_fusion_spatial_neighborhood = "N4"
    aggregator.frame_fusion_spatial_radius = 1
    aggregator.frame_fusion_temporal_window = 2
    aggregator._frame_fusion_edge_tensor_cache = {}

    tensor_source, tensor_target = (
        aggregator._build_local_spatiotemporal_edge_tensors(
            num_frames=4,
            patch_count=12,
            include_temporal_spatial=False,
            exclude_frame_zero=True,
            device=torch.device("cpu"),
        )
    )
    cached_source, cached_target = (
        aggregator._build_local_spatiotemporal_edge_tensors(
            num_frames=4,
            patch_count=12,
            include_temporal_spatial=False,
            exclude_frame_zero=True,
            device=torch.device("cpu"),
        )
    )
    numpy_source, numpy_target = aggregator._build_local_spatiotemporal_edges(
        num_frames=4,
        patch_count=12,
        include_temporal_spatial=False,
        exclude_frame_zero=True,
    )
    local_source, local_target = aggregator._build_local_spatiotemporal_edge_tensors(
        num_frames=4,
        patch_count=12,
        include_temporal_spatial=False,
        exclude_frame_zero=True,
        device=torch.device("cpu"),
        use_cache=False,
        index_offset=12,
    )

    assert cached_source.data_ptr() == tensor_source.data_ptr()
    assert cached_target.data_ptr() == tensor_target.data_ptr()
    assert np.array_equal(tensor_source.numpy(), numpy_source)
    assert np.array_equal(tensor_target.numpy(), numpy_target)
    assert torch.equal(local_source, tensor_source - 12)
    assert torch.equal(local_target, tensor_target - 12)
    assert len(aggregator._frame_fusion_edge_tensor_cache) == 1


def test_tensorized_spacetime_cube_canonical_order_matches_unique_keys():
    aggregator = Aggregator.__new__(Aggregator)
    aggregator._frame_fusion_patch_grid_size = (3, 4)
    aggregator.frame_fusion_spatial_neighborhood = "N4"
    aggregator.frame_fusion_spatial_radius = 1
    aggregator.frame_fusion_temporal_window = 2
    aggregator._frame_fusion_edge_tensor_cache = {}

    source, target = aggregator._build_local_spatiotemporal_edge_tensors(
        num_frames=4,
        patch_count=12,
        include_temporal_spatial=False,
        exclude_frame_zero=True,
        device=torch.device("cpu"),
        index_offset=12,
        canonical_order=True,
    )
    keys = source * 36 + target

    assert torch.equal(keys, torch.unique(keys, sorted=True))


def test_selftr_cube_depends_only_on_radius_and_temporal_window():
    aggregator = Aggregator.__new__(Aggregator)
    aggregator._frame_fusion_patch_grid_size = (5, 5)
    aggregator.frame_fusion_spatial_radius = 2
    aggregator.frame_fusion_temporal_window = 4
    aggregator._frame_fusion_edge_tensor_cache = {}

    topologies = []
    for neighborhood in ("N4", "N8", "N8-R2"):
        aggregator.frame_fusion_spatial_neighborhood = neighborhood
        source, target = aggregator._build_local_spatiotemporal_edge_tensors(
            num_frames=5,
            patch_count=25,
            include_temporal_spatial=False,
            use_spatiotemporal_cube=True,
            device=torch.device("cpu"),
            use_cache=False,
        )
        topologies.append(torch.stack((source, target), dim=1))

    assert torch.equal(topologies[0], topologies[1])
    assert torch.equal(topologies[1], topologies[2])
    center_to_four_future_frames = {
        (int(right) // 25, int(right) % 25)
        for left, right in topologies[0].tolist()
        if left == 12 and right // 25 > 0
    }
    assert center_to_four_future_frames == {
        (frame, position)
        for frame in range(1, 5)
        for position in range(25)
    }


def test_select_frame_fusion_pairs_uses_nearest_neighbor_deduplicated_pairs():
    similarity = torch.tensor(
        [
            [1.00, 0.95, 0.10, 0.20],
            [0.95, 1.00, 0.30, 0.10],
            [0.10, 0.30, 1.00, 0.90],
            [0.20, 0.10, 0.90, 1.00],
        ]
    )

    pairs, unique_count, requested_count = select_frame_fusion_pairs(
        similarity,
        pair_percent=50.0,
    )

    assert unique_count == 2
    assert requested_count == 1
    assert [(pair.frame_a, pair.frame_b) for pair in pairs] == [(0, 1)]


def test_select_frame_fusion_pairs_skips_overlapping_pairs_by_similarity_order():
    similarity = torch.tensor(
        [
            [1.00, 0.99, 0.10, 0.20],
            [0.99, 1.00, 0.98, 0.10],
            [0.10, 0.98, 1.00, 0.97],
            [0.20, 0.10, 0.97, 1.00],
        ]
    )

    pairs, unique_count, requested_count = select_frame_fusion_pairs(
        similarity,
        pair_percent=100.0,
    )

    assert unique_count == 3
    assert requested_count == 3
    assert [(pair.frame_a, pair.frame_b) for pair in pairs] == [(0, 1), (2, 3)]


def test_select_frame_fusion_pairs_excludes_reference_frame_zero():
    similarity = torch.tensor(
        [
            [1.00, 0.99, 0.98, 0.10],
            [0.99, 1.00, 0.20, 0.10],
            [0.98, 0.20, 1.00, 0.90],
            [0.10, 0.10, 0.90, 1.00],
        ]
    )

    pairs, unique_count, requested_count = select_frame_fusion_pairs(
        similarity,
        pair_percent=100.0,
        exclude_frames=(0,),
    )

    assert unique_count == 2
    assert requested_count == 2
    assert [(pair.frame_a, pair.frame_b) for pair in pairs] == [(2, 3)]
    assert all(0 not in (pair.frame_a, pair.frame_b) for pair in pairs)


def test_select_frame_fusion_pairs_from_representations_matches_nearest_dedup_matrix():
    representations = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        ),
        p=2,
        dim=-1,
    )
    similarity = torch.matmul(representations, representations.T)

    from_matrix = select_frame_fusion_pairs(
        similarity,
        pair_percent=100.0,
        exclude_frames=(0,),
    )
    from_representations = select_frame_fusion_pairs_from_normalized_representations(
        representations,
        pair_percent=100.0,
        exclude_frames=(0,),
    )

    assert from_representations[1:] == from_matrix[1:]
    assert [
        (pair.frame_a, pair.frame_b, pair.similarity)
        for pair in from_representations[0]
    ] == [
        (pair.frame_a, pair.frame_b, pair.similarity)
        for pair in from_matrix[0]
    ]


def test_overlapping_top_percent_pairs_form_connected_frame_groups():
    pairs = [
        FrameFusionPair(frame_a=1, frame_b=2, similarity=0.99),
        FrameFusionPair(frame_a=2, frame_b=3, similarity=0.98),
        FrameFusionPair(frame_a=1, frame_b=3, similarity=0.97),
        FrameFusionPair(frame_a=5, frame_b=6, similarity=0.96),
    ]

    groups = _connected_frame_fusion_groups(pairs)

    assert groups == [
        FrameFusionGroup(anchor=1, members=(1, 2, 3)),
        FrameFusionGroup(anchor=5, members=(5, 6)),
    ]


def test_frame_fusion_attention_indices_keep_all_special_tokens_and_one_patch_side():
    source_frames = torch.tensor([1])
    target_frames = torch.tensor([2])

    indices = frame_fusion_attention_indices(
        num_frames=3,
        tokens_per_frame=5,
        num_special_tokens=2,
        source_frames=source_frames,
        target_frames=target_frames,
    )

    assert torch.equal(
        indices,
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    )


def test_frame_fusion_attention_indices_keep_selected_target_patches():
    indices = frame_fusion_attention_indices(
        num_frames=3,
        tokens_per_frame=6,
        num_special_tokens=2,
        source_frames=torch.tensor([1]),
        target_frames=torch.tensor([2]),
        target_keep_patch_indices=torch.tensor([[1, 3]]),
    )

    assert torch.equal(
        indices,
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17]),
    )


def test_frame_fusion_attention_indices_accept_boolean_target_patch_mask():
    indices = frame_fusion_attention_indices(
        num_frames=3,
        tokens_per_frame=6,
        num_special_tokens=2,
        source_frames=torch.tensor([1]),
        target_frames=torch.tensor([2]),
        target_keep_patch_indices=torch.tensor([[False, True, False, True]]),
    )

    assert torch.equal(
        indices,
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17]),
    )


def test_pair_patch_fusion_averages_patches_and_preserves_special_tokens():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 2
    tokens = torch.arange(15, dtype=torch.float32).view(1, 3, 5, 1)
    plan = FrameFusionBatchPlan(
        pairs=(FrameFusionPair(frame_a=0, frame_b=2, similarity=0.9),),
        source_frames=torch.tensor([0]),
        target_frames=torch.tensor([2]),
        attention_indices=torch.arange(12),
        unique_candidate_count=1,
        requested_pair_count=1,
    )

    fused = model._fuse_frame_pair_patch_tokens(tokens, [plan])

    assert torch.equal(fused[0, 0, :2], tokens[0, 0, :2])
    assert torch.equal(fused[0, 2, :2], tokens[0, 2, :2])
    expected_patch_tokens = (tokens[0, 0, 2:] + tokens[0, 2, 2:]) * 0.5
    assert torch.allclose(fused[0, 0, 2:], expected_patch_tokens)
    assert torch.allclose(fused[0, 2, 2:], expected_patch_tokens)


def test_pair_patch_fusion_preserves_selected_target_patch_tokens():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 2
    tokens = torch.arange(18, dtype=torch.float32).view(1, 3, 6, 1)
    plan = FrameFusionBatchPlan(
        pairs=(FrameFusionPair(frame_a=0, frame_b=2, similarity=0.9),),
        source_frames=torch.tensor([0]),
        target_frames=torch.tensor([2]),
        attention_indices=torch.arange(13),
        unique_candidate_count=1,
        requested_pair_count=1,
        target_keep_patch_indices=torch.tensor([[1, 3]]),
    )

    fused = model._fuse_frame_pair_patch_tokens(tokens, [plan])

    assert torch.equal(fused[0, 0, 3], tokens[0, 0, 3])
    assert torch.equal(fused[0, 2, 3], tokens[0, 2, 3])
    assert torch.equal(fused[0, 0, 5], tokens[0, 0, 5])
    assert torch.equal(fused[0, 2, 5], tokens[0, 2, 5])
    for offset in (2, 4):
        expected = (tokens[0, 0, offset] + tokens[0, 2, offset]) * 0.5
        assert torch.allclose(fused[0, 0, offset], expected)
        assert torch.allclose(fused[0, 2, offset], expected)


def test_pair_patch_fusion_preserves_boolean_masked_target_patch_tokens():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 2
    tokens = torch.arange(18, dtype=torch.float32).view(1, 3, 6, 1)
    plan = FrameFusionBatchPlan(
        pairs=(FrameFusionPair(frame_a=0, frame_b=2, similarity=0.9),),
        source_frames=torch.tensor([0]),
        target_frames=torch.tensor([2]),
        attention_indices=torch.arange(13),
        unique_candidate_count=1,
        requested_pair_count=1,
        target_keep_patch_indices=torch.tensor([[False, True, False, True]]),
    )

    fused = model._fuse_frame_pair_patch_tokens(tokens, [plan])

    assert torch.equal(fused[0, 0, 3], tokens[0, 0, 3])
    assert torch.equal(fused[0, 2, 3], tokens[0, 2, 3])
    assert torch.equal(fused[0, 0, 5], tokens[0, 0, 5])
    assert torch.equal(fused[0, 2, 5], tokens[0, 2, 5])
    for offset in (2, 4):
        expected = (tokens[0, 0, offset] + tokens[0, 2, offset]) * 0.5
        assert torch.allclose(fused[0, 0, offset], expected)
        assert torch.allclose(fused[0, 2, offset], expected)


def test_group_patch_fusion_copies_shared_tokens_from_anchor_without_averaging():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 1
    model.frame_fusion_mode = "group-top-percent"
    tokens = torch.tensor(
        [
            [
                [[10.0], [100.0], [200.0], [300.0]],
                [[20.0], [101.0], [201.0], [301.0]],
                [[30.0], [102.0], [202.0], [302.0]],
            ]
        ]
    )
    plan = FrameFusionBatchPlan(
        pairs=(
            FrameFusionPair(frame_a=0, frame_b=1, similarity=0.99),
            FrameFusionPair(frame_a=1, frame_b=2, similarity=0.98),
        ),
        groups=(FrameFusionGroup(anchor=0, members=(0, 1, 2)),),
        source_frames=torch.tensor([0, 0]),
        target_frames=torch.tensor([1, 2]),
        attention_indices=torch.arange(10),
        unique_candidate_count=2,
        requested_pair_count=2,
        target_keep_patch_indices=torch.tensor(
            [[False, True, False], [True, False, True]]
        ),
    )

    fused = model._fuse_frame_pair_patch_tokens(tokens, [plan])

    assert torch.equal(fused[0, 1, 1], tokens[0, 0, 1])
    assert torch.equal(fused[0, 1, 3], tokens[0, 0, 3])
    assert torch.equal(fused[0, 2, 2], tokens[0, 0, 2])
    assert torch.equal(fused[0, 2, 3], tokens[0, 2, 3])


def test_pair_plan_builder_never_pairs_reference_frame_zero():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 1
    model.frame_fusion_pool_size = 2
    model.frame_fusion_pair_percent = 100.0
    model.frame_fusion_mode = "pair-top-percent"
    model.frame_fusion_target_keep_policy = "none"
    model.frame_fusion_target_keep_grid_size = 4
    model.frame_fusion_target_keep_percent = 0.0
    model.frame_fusion_target_keep_seed = 33
    model.last_frame_fusion_debug = {}
    tokens = torch.zeros(1, 4, 5, 2)
    tokens[0, 0, 1:] = torch.tensor([[1.0, 0.0]] * 4)
    tokens[0, 1, 1:] = torch.tensor([[0.99, 0.01]] * 4)
    tokens[0, 2, 1:] = torch.tensor([[0.0, 1.0]] * 4)
    tokens[0, 3, 1:] = torch.tensor([[0.0, 0.95]] * 4)

    plans = model._build_frame_fusion_pair_plans(
        tokens,
        patch_grid_size=(2, 2),
        source_layer=-1,
    )

    assert [(pair.frame_a, pair.frame_b) for pair in plans[0].pairs] == [(2, 3)]
    assert model.last_frame_fusion_debug["excluded_frames"] == [0]


def test_group_plan_builder_expands_overlapping_pairs_from_one_anchor():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 1
    model.frame_fusion_pool_size = 1
    model.frame_fusion_pair_percent = 100.0
    model.frame_fusion_mode = "group-top-percent"
    model.frame_fusion_target_keep_policy = "none"
    model.frame_fusion_target_keep_grid_size = 4
    model.frame_fusion_target_keep_percent = 0.0
    model.frame_fusion_target_keep_threshold = 0.0
    model.frame_fusion_target_keep_seed = 33
    model.frame_fusion_recompute_each_global = False
    model.last_frame_fusion_debug = {}
    tokens = torch.zeros(1, 4, 5, 2)
    tokens[0, 0, 1:] = torch.tensor([[0.0, 1.0]] * 4)
    tokens[0, 1, 1:] = torch.tensor([[0.1, 0.99]] * 4)
    tokens[0, 2, 1:] = torch.tensor([[0.2, 0.98]] * 4)
    tokens[0, 3, 1:] = torch.tensor([[1.0, 0.0]] * 4)

    plans = model._build_frame_fusion_pair_plans(
        tokens,
        patch_grid_size=(2, 2),
        source_layer=-1,
    )

    assert [group.members for group in plans[0].groups] == [(1, 2, 3)]
    assert list(zip(plans[0].source_frames.tolist(), plans[0].target_frames.tolist())) == [
        (1, 2),
        (1, 3),
    ]
    assert model.last_frame_fusion_debug["batches"][0]["selected_groups"] == 1


def test_sequential_groups_use_all_members_threshold_and_exclude_reference_frame():
    representations = torch.nn.functional.normalize(
        torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.99, 0.1],
                [0.98, 0.2],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        p=2,
        dim=-1,
    )

    groups = _sequential_frame_fusion_groups(
        representations,
        similarity_threshold=0.8,
        max_group_size=4,
        first_frame=1,
    )

    assert [group.members for group in groups] == [(1, 2, 3), (4, 5)]
    assert all(0 not in group.members for group in groups)


def test_group_average_fusion_uses_mean_shared_token_and_keeps_distinct_positions():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 1
    model.frame_fusion_mode = "sequential-group-average"
    tokens = torch.tensor(
        [
            [
                [[0.0], [1.0], [10.0], [3.0]],
                [[0.0], [2.0], [20.0], [6.0]],
                [[0.0], [4.0], [30.0], [9.0]],
            ]
        ]
    )
    plan = FrameFusionBatchPlan(
        pairs=(
            FrameFusionPair(frame_a=0, frame_b=1, similarity=0.9),
            FrameFusionPair(frame_a=0, frame_b=2, similarity=0.9),
        ),
        groups=(FrameFusionGroup(anchor=0, members=(0, 1, 2)),),
        source_frames=torch.tensor([0, 0]),
        target_frames=torch.tensor([1, 2]),
        attention_indices=torch.arange(10),
        unique_candidate_count=2,
        requested_pair_count=2,
        target_keep_patch_indices=torch.tensor(
            [[False, True, False], [False, True, False]]
        ),
    )

    fused = model._fuse_frame_pair_patch_tokens(tokens, [plan])

    assert torch.equal(fused[0, 0, 1], torch.tensor([7.0 / 3.0]))
    assert torch.equal(fused[0, 1, 1], torch.tensor([7.0 / 3.0]))
    assert torch.equal(fused[0, 2, 1], torch.tensor([7.0 / 3.0]))
    assert torch.equal(fused[0, 0, 3], torch.tensor([6.0]))
    assert torch.equal(fused[0, 1, 3], torch.tensor([6.0]))
    assert torch.equal(fused[0, 2, 3], torch.tensor([6.0]))
    assert torch.equal(fused[0, 0, 2], tokens[0, 0, 2])
    assert torch.equal(fused[0, 1, 2], tokens[0, 1, 2])
    assert torch.equal(fused[0, 2, 2], tokens[0, 2, 2])


def test_pair_patch_output_copy_reuses_source_patch_tokens():
    flat_tokens = torch.arange(15, dtype=torch.float32).view(15, 1)
    flat_tokens[2:5] = torch.tensor([[100.0], [101.0], [102.0]])
    plan = FrameFusionBatchPlan(
        pairs=(FrameFusionPair(frame_a=0, frame_b=2, similarity=0.9),),
        source_frames=torch.tensor([0]),
        target_frames=torch.tensor([2]),
        attention_indices=torch.arange(12),
        unique_candidate_count=1,
        requested_pair_count=1,
    )

    copied = Aggregator._copy_pair_patch_outputs(
        flat_tokens,
        plan,
        tokens_per_frame=5,
        num_special_tokens=2,
    )

    assert torch.equal(copied[10:12], flat_tokens[10:12])
    assert torch.equal(copied[12:15], flat_tokens[2:5])


def test_pair_patch_output_copy_preserves_selected_target_patch_tokens():
    flat_tokens = torch.arange(18, dtype=torch.float32).view(18, 1)
    flat_tokens[2:6] = torch.tensor([[100.0], [101.0], [102.0], [103.0]])
    plan = FrameFusionBatchPlan(
        pairs=(FrameFusionPair(frame_a=0, frame_b=2, similarity=0.9),),
        source_frames=torch.tensor([0]),
        target_frames=torch.tensor([2]),
        attention_indices=torch.arange(13),
        unique_candidate_count=1,
        requested_pair_count=1,
        target_keep_patch_indices=torch.tensor([[1, 3]]),
    )

    copied = Aggregator._copy_pair_patch_outputs(
        flat_tokens,
        plan,
        tokens_per_frame=6,
        num_special_tokens=2,
    )

    assert torch.equal(copied[14], flat_tokens[2])
    assert torch.equal(copied[16], flat_tokens[4])
    assert torch.equal(copied[15], flat_tokens[15])
    assert torch.equal(copied[17], flat_tokens[17])


def test_pair_patch_output_copy_preserves_boolean_masked_target_patch_tokens():
    flat_tokens = torch.arange(18, dtype=torch.float32).view(18, 1)
    flat_tokens[2:6] = torch.tensor([[100.0], [101.0], [102.0], [103.0]])
    plan = FrameFusionBatchPlan(
        pairs=(FrameFusionPair(frame_a=0, frame_b=2, similarity=0.9),),
        source_frames=torch.tensor([0]),
        target_frames=torch.tensor([2]),
        attention_indices=torch.arange(13),
        unique_candidate_count=1,
        requested_pair_count=1,
        target_keep_patch_indices=torch.tensor([[False, True, False, True]]),
    )

    copied = Aggregator._copy_pair_patch_outputs(
        flat_tokens,
        plan,
        tokens_per_frame=6,
        num_special_tokens=2,
    )

    assert torch.equal(copied[14], flat_tokens[2])
    assert torch.equal(copied[16], flat_tokens[4])
    assert torch.equal(copied[15], flat_tokens[15])
    assert torch.equal(copied[17], flat_tokens[17])


def test_similarity_threshold_target_keep_policy_keeps_tokens_below_threshold():
    model = Aggregator.__new__(Aggregator)
    model.frame_fusion_target_keep_policy = "similarity-threshold"
    model.frame_fusion_target_keep_threshold = 0.5
    patch_tokens = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        ]
    )

    keep_mask = model._select_frame_fusion_target_keep_patch_indices(
        patch_tokens,
        [FrameFusionPair(frame_a=0, frame_b=1, similarity=0.9)],
        patch_grid_size=(1, 3),
        source_layer=-1,
        batch_index=0,
    )

    assert keep_mask.dtype == torch.bool
    assert torch.equal(keep_mask, torch.tensor([[False, True, True]]))


def test_temporal_representative_plan_preserves_mapping_and_occurrence_weights():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 1
    model.frame_fusion_target_keep_threshold = 0.95
    model.frame_fusion_mode = "temporal-representative"
    model.last_frame_fusion_debug = {}

    tokens = torch.tensor(
        [
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.99, 0.01], [1.0, 0.0]],
                [[0.0, 0.0], [0.98, 0.02], [0.99, 0.01]],
            ]
        ]
    )

    plans = model._build_temporal_representative_plans(tokens, source_layer=-1)
    plan = plans[0]

    assert plan.position_to_representative.tolist() == [[0, 1], [2, 3], [2, 3]]
    assert plan.representative_source_indices.tolist() == [0, 1, 2, 3]
    assert torch.equal(plan.representative_weights, torch.tensor([1.0, 1.0, 2.0, 2.0]))
    assert model.last_frame_fusion_debug["mapping_preserved"] is True


def test_adaptive_temporal_plan_keeps_reference_frames_and_positive_weights():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 1
    model.frame_fusion_mode = "adaptive-temporal-representative"
    model.frame_fusion_lambda_cost = 1.0

    tokens = torch.tensor(
        [
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
                [[0.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
            ]
        ]
    )

    plan = model._build_adaptive_temporal_representative_plans(
        tokens, source_layer=-1
    )[0]
    mapping = plan.position_to_representative

    assert mapping[0].tolist() == [0, 1]
    assert mapping[1].tolist() == [2, 3]
    assert torch.equal(
        plan.representative_weights,
        torch.bincount(mapping.reshape(-1), minlength=plan.representative_source_indices.numel()).float(),
    )
    assert bool(torch.all(plan.representative_weights >= 1).item())
    assert float(plan.representative_weights.sum()) == float(mapping.numel())
    assert model.last_frame_fusion_debug["mapping_preserved"] is True
    assert model.last_frame_fusion_debug["cost_model"] == "normalized_temporal_distortion_plus_lambda"


def test_adaptive_temporal_lambda_controls_split_count():
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randn((1, 8, 5, 4), generator=generator)
    split_counts = []
    for lambda_cost in (1.0, 0.75, 0.5, 0.25):
        model = Aggregator.__new__(Aggregator)
        model.patch_token_start = 1
        model.frame_fusion_mode = "adaptive-temporal-representative"
        model.frame_fusion_lambda_cost = lambda_cost
        model._build_adaptive_temporal_representative_plans(tokens, source_layer=-1)
        split_counts.append(model.last_frame_fusion_debug["batches"][0]["optimal_split_count"])

    assert split_counts == sorted(split_counts)


def test_adaptive_spatial_plan_preserves_reference_frame_and_processes_later_frames():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 1
    model.frame_fusion_mode = "adaptive-spatial-representative"
    model.frame_fusion_lambda_cost = 1.0

    tokens = torch.tensor(
        [
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
                [[0.0, 0.0], [-1.0, 0.0], [-0.99, 0.01], [0.0, 1.0]],
            ]
        ]
    )

    plan = model._build_adaptive_spatial_representative_plans(tokens, source_layer=-1)[0]

    assert torch.equal(plan.position_to_representative[0], torch.arange(3))
    assert plan.representative_source_indices[:3].tolist() == [0, 1, 2]
    assert model.last_frame_fusion_debug["reference_frame_index"] == 0
    assert model.last_frame_fusion_debug["reference_frame_compression"] == "none"
    assert model.last_frame_fusion_debug["attention_only"] is True
    assert model.last_frame_fusion_debug["mlp_scope"] == "full_original_token_sequence"
    assert model.last_frame_fusion_debug["batches"][0]["processed_frame_indices"] == [1, 2]
    assert len(model.last_frame_fusion_debug["batches"][0]["frames"]) == 2
    assert torch.equal(
        plan.representative_weights,
        torch.bincount(
            plan.position_to_representative.reshape(-1),
            minlength=plan.representative_source_indices.numel(),
        ).float(),
    )
    assert bool(torch.all(plan.representative_weights >= 1).item())


def test_adaptive_spatial_lambda_controls_representative_count():
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randn((1, 3, 6, 4), generator=generator)
    representative_counts = []
    for lambda_cost in (1.0, 0.75, 0.5, 0.25):
        model = Aggregator.__new__(Aggregator)
        model.patch_token_start = 1
        model.frame_fusion_mode = "adaptive-spatial-representative"
        model.frame_fusion_lambda_cost = lambda_cost
        model._build_adaptive_spatial_representative_plans(tokens, source_layer=-1)
        representative_counts.append(
            model.last_frame_fusion_debug["batches"][0]["frame_representative_counts"]
        )

    assert all(
        lower[0] >= higher[0] and lower[1] >= higher[1]
        for higher, lower in zip(representative_counts, representative_counts[1:])
    )


def test_layer_token_swap_patch_special_and_whole_scopes():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 2
    tokens = torch.arange(15, dtype=torch.float32).view(1, 3, 5, 1)

    patch_swapped = model._apply_layer_token_swap(
        tokens,
        kind="patch",
        pairs=((0, 2),),
    )
    assert torch.equal(patch_swapped[0, 0, :2], tokens[0, 0, :2])
    assert torch.equal(patch_swapped[0, 2, :2], tokens[0, 2, :2])
    assert torch.equal(patch_swapped[0, 0, 2:], tokens[0, 2, 2:])
    assert torch.equal(patch_swapped[0, 2, 2:], tokens[0, 0, 2:])

    special_swapped = model._apply_layer_token_swap(
        tokens,
        kind="special",
        pairs=((0, 2),),
    )
    assert torch.equal(special_swapped[0, 0, :2], tokens[0, 2, :2])
    assert torch.equal(special_swapped[0, 2, :2], tokens[0, 0, :2])
    assert torch.equal(special_swapped[0, 0, 2:], tokens[0, 0, 2:])
    assert torch.equal(special_swapped[0, 2, 2:], tokens[0, 2, 2:])

    whole_swapped = model._apply_layer_token_swap(
        tokens,
        kind="whole",
        pairs=((0, 2),),
    )
    assert torch.equal(whole_swapped[0, 0], tokens[0, 2])
    assert torch.equal(whole_swapped[0, 2], tokens[0, 0])


@pytest.mark.parametrize("mode", ("h-m", "h-r", "selftr", "u-r"))
def test_spatiotemporal_representative_modes_protect_frame_zero(mode):
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 1
    model.frame_fusion_mode = mode
    model.frame_fusion_lambda_cost = 0.15
    model.frame_fusion_min_keep_ratio = 0.4
    model.frame_fusion_temporal_window = 1
    model.frame_fusion_spatial_neighborhood = "N8"
    model.frame_fusion_time_overlap = 0.5
    model.frame_fusion_reassignment_candidates = 8
    model.frame_fusion_max_group_size = 4 if mode.startswith("h") else 8
    model.frame_fusion_representative_update = "parent"
    model._frame_fusion_patch_grid_size = (2, 2)

    tokens = torch.randn((1, 4, 5, 8), generator=torch.Generator().manual_seed(7))
    plans = model._build_spatiotemporal_representative_plans(tokens, source_layer=-1)
    plan = plans[0]
    debug = model.last_frame_fusion_debug

    assert debug["lambda_cost"] == pytest.approx(0.15)
    assert debug["cost_denominator"] == "(F - 1) * P"
    assert debug["selection"] == (
        "mutual_nearest_neighbor_delta_E_lt_2_lambda"
        if mode == "selftr"
        else "min(D_m_normalized + lambda_cost * M_m_normalized)"
    )
    assert debug["representative_update"] == (
        "reassignment" if mode in {"h-r", "u-r"} else "best-of-parents"
    )

    assert tuple(plan.position_to_representative.shape) == (4, 4)
    assert torch.equal(
        plan.representative_weights,
        torch.bincount(
            plan.position_to_representative.reshape(-1),
            minlength=plan.representative_source_indices.numel(),
        ).float(),
    )
    frame_zero_representatives = torch.nonzero(
        plan.representative_source_indices < 4,
        as_tuple=False,
    ).flatten()
    assert frame_zero_representatives.numel() == 4
    assert torch.equal(
        torch.sort(plan.position_to_representative[0]).values,
        torch.sort(frame_zero_representatives).values,
    )
    assert not bool(
        torch.isin(
            plan.position_to_representative[1:].reshape(-1),
            frame_zero_representatives,
        ).any()
    )
    assert int(plan.position_to_representative.min()) >= 0
    assert int(plan.position_to_representative.max()) < plan.representative_source_indices.numel()


@pytest.mark.parametrize("reallocate", (False, True))
def test_unified_spatiotemporal_graph_excludes_frame_zero(reallocate):
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 1
    model.frame_fusion_min_keep_ratio = 0.4
    model.frame_fusion_temporal_window = 1
    model.frame_fusion_spatial_neighborhood = "N8"
    model.frame_fusion_reassignment_candidates = 8
    model.frame_fusion_max_group_size = 8
    model.frame_fusion_representative_update = "parent"
    model._frame_fusion_patch_grid_size = (1, 1)

    # Make the non-reference frames identical so the old implementation
    # would be tempted to absorb them into frame 0 through the temporal edge.
    tokens = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ]
    ).unsqueeze(0)
    model.frame_fusion_mode = "u-r" if reallocate else "selftr"
    plan = model._build_unified_representative_plan(tokens[0], reallocate=reallocate)

    frame_zero_representatives = torch.nonzero(
        plan.representative_source_indices < 1,
        as_tuple=False,
    ).flatten()
    assert frame_zero_representatives.tolist() == [0]
    assert plan.position_to_representative[0, 0].item() == 0
    assert not bool(
        torch.isin(
            plan.position_to_representative[1:].reshape(-1),
            frame_zero_representatives,
        ).any()
    )


def test_unified_merge_path_does_not_use_min_keep_or_group_size_limits():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 1
    model.frame_fusion_min_keep_ratio = 0.05
    model.frame_fusion_max_group_size = 1
    model.frame_fusion_temporal_window = 1
    model.frame_fusion_spatial_neighborhood = "N8"
    model.frame_fusion_time_overlap = 0.5
    model.frame_fusion_representative_update = "parent"
    model._frame_fusion_patch_grid_size = (1, 1)

    # The reference frame is isolated.  The remaining three frames form one
    # identical temporal chain, which must still produce a merge path despite
    # deliberately restrictive legacy parameter values above.
    tokens = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ]
    )
    plan = model._build_unified_representative_plan(tokens, reallocate=False)

    assert plan.representative_source_indices.numel() < 4
    assert plan.position_to_representative[0, 0].item() == 0


def test_unified_merge_path_honors_min_keep_ratio():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 1
    model.frame_fusion_min_keep_ratio = 1.0
    model.frame_fusion_max_group_size = 8
    model.frame_fusion_temporal_window = 1
    model.frame_fusion_spatial_neighborhood = "N8"
    model.frame_fusion_representative_update = "parent"
    model._frame_fusion_patch_grid_size = (1, 1)

    tokens = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ]
    )
    plan = model._build_unified_representative_plan(tokens, reallocate=False)

    assert plan.representative_source_indices.numel() == 4


def test_spatiotemporal_group_error_uses_true_weighted_reconstruction_loss():
    model = Aggregator.__new__(Aggregator)
    model.frame_fusion_representative_update = "parent"
    features = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.8, 0.6]]), dim=-1
    )

    _, selected_sources, debug = model._greedy_spatiotemporal_group_merge(
        features,
        np.arange(2, dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        protected=np.zeros(2, dtype=bool),
        initial_weights=np.asarray([2.0, 1.0]),
        lambda_cost=2.0,
        prefer_best_parent=True,
    )

    # The merged group is represented by token 0. Its exact error is
    # 1 * (1 - 0.8), averaged over total weight 3, rather than the old
    # Ward-style d * 2 * 1 / 3 approximation.
    assert selected_sources.tolist() == [0]
    assert debug["accepted_merges"] == 1
    assert debug["selected_distortion"] == pytest.approx(0.2 / (2.0 * 3.0), abs=1e-6)
    assert debug["distortion_normalization"] == "average_cosine_distance / 2"


def test_unified_batch_merge_accepts_disjoint_mutual_pairs_in_one_round():
    model = Aggregator.__new__(Aggregator)
    features = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )

    mapping, selected_sources, debug = model._batch_mutual_nearest_group_merge(
        features,
        np.arange(4, dtype=np.int64),
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([1, 3], dtype=np.int64),
        protected=np.zeros(4, dtype=bool),
        lambda_cost=0.1,
        cost_denominator=4.0,
    )

    assert mapping.tolist() == [0, 0, 1, 1]
    assert selected_sources.tolist() == [0, 2]
    assert debug["parallel_rounds"] == 1
    assert debug["mutual_pairs_seen"] == 2
    assert debug["accepted_merges"] == 2
    assert debug["selected_merges"] == debug["accepted_merges"]
    assert debug["selection"] == "mutual_nearest_neighbor_delta_E_lt_2_lambda"
    assert debug["stopping_rule"] == "delta_E < 2 * lambda_cost"


def test_fused_selftr_edge_cost_falls_back_on_cpu():
    features = torch.eye(2, dtype=torch.float32)
    result = fused_selftr_edge_cost(
        features,
        torch.ones(2),
        torch.arange(2),
        torch.zeros(2),
        features,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([True]),
        prefer_best_parent=True,
    )

    assert result is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_selftr_edge_cost_matches_pytorch_reference_on_cuda(monkeypatch):
    monkeypatch.setenv("SELFTR_TRITON", "1")
    generator = torch.Generator(device="cuda").manual_seed(7)
    group_count = 37
    feature_dim = 1024
    edge_count = 257
    features = torch.nn.functional.normalize(
        torch.randn(
            group_count,
            feature_dim,
            generator=generator,
            device="cuda",
        ),
        dim=-1,
    )
    group_weights = torch.rand(
        group_count,
        generator=generator,
        device="cuda",
    ) * 4.0 + 1.0
    group_sums = features * group_weights[:, None]
    group_representatives = torch.arange(group_count, device="cuda")
    group_errors = group_weights - (group_sums * features).sum(dim=-1)
    edge_left = torch.randint(
        group_count,
        (edge_count,),
        generator=generator,
        device="cuda",
    )
    edge_right = torch.randint(
        group_count,
        (edge_count,),
        generator=generator,
        device="cuda",
    )
    edge_valid = edge_left != edge_right

    actual = fused_selftr_edge_cost(
        group_sums,
        group_weights,
        group_representatives,
        group_errors,
        features,
        edge_left,
        edge_right,
        edge_valid,
        prefer_best_parent=True,
    )
    assert actual is not None
    merged_sum = group_sums[edge_left] + group_sums[edge_right]
    merged_weight = group_weights[edge_left] + group_weights[edge_right]
    left_error = merged_weight - (
        merged_sum * features[group_representatives[edge_left]]
    ).sum(dim=-1)
    right_error = merged_weight - (
        merged_sum * features[group_representatives[edge_right]]
    ).sum(dim=-1)
    expected = (
        torch.minimum(left_error, right_error)
        - group_errors[edge_left]
        - group_errors[edge_right]
    ).masked_fill(~edge_valid, float("inf"))

    assert torch.allclose(actual, expected, atol=2e-6, rtol=1e-6)


def test_unified_batch_merge_keeps_only_top_similarity_pairs_per_round():
    model = Aggregator.__new__(Aggregator)
    angles = torch.tensor([0.0, 0.2, 0.4, 0.6, 1.4], dtype=torch.float32)
    pair_features = torch.stack((angles.cos(), angles.sin()), dim=1)
    features = pair_features.repeat_interleave(2, dim=0)
    pair_starts = np.arange(0, 10, 2, dtype=np.int64)

    mapping, selected_sources, debug = model._batch_mutual_nearest_group_merge(
        features,
        np.arange(10, dtype=np.int64),
        pair_starts,
        pair_starts + 1,
        protected=np.zeros(10, dtype=bool),
        min_keep_ratio=0.6,
        lambda_cost=2.0,
        cost_denominator=10.0,
        merge_top_similarity_percent=80.0,
    )

    assert mapping[:8].reshape(4, 2).tolist() == [[0, 0], [1, 1], [2, 2], [3, 3]]
    assert mapping[8:].tolist() == [4, 5]
    assert selected_sources.tolist() == [0, 2, 4, 6, 8, 9]
    assert debug["merge_top_similarity_percent"] == pytest.approx(80.0)
    assert debug["similarity_pairs_seen"] == 5
    assert debug["similarity_pairs_kept"] == 4
    assert debug["similarity_pairs_filtered"] == 1


def test_unified_batch_merge_stops_when_delta_exceeds_lambda_threshold():
    model = Aggregator.__new__(Aggregator)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    mapping, selected_sources, debug = model._batch_mutual_nearest_group_merge(
        features,
        np.arange(2, dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        protected=np.zeros(2, dtype=bool),
        lambda_cost=0.1,
        cost_denominator=2.0,
    )

    assert mapping.tolist() == [0, 1]
    assert selected_sources.tolist() == [0, 1]
    assert debug["accepted_merges"] == 0
    assert debug["parallel_rounds"] == 0
    assert debug["stop_reason"] == "minimum_delta_threshold"


def test_unified_batch_merge_can_target_a_fixed_compression_ratio():
    model = Aggregator.__new__(Aggregator)
    features = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )

    mapping, selected_sources, debug = model._batch_mutual_nearest_group_merge(
        features,
        np.arange(4, dtype=np.int64),
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([1, 3], dtype=np.int64),
        protected=np.zeros(4, dtype=bool),
        min_keep_ratio=0.05,
        lambda_cost=0.01,
        cost_denominator=4.0,
        fixed_compression_ratio=0.5,
    )

    assert mapping.tolist() == [0, 0, 1, 1]
    assert selected_sources.tolist() == [0, 2]
    assert debug["selection"] == "fixed_compression_ratio"
    assert debug["stopping_rule"] == "fixed_compression_ratio"
    assert debug["fixed_target_active_tokens"] == 2
    assert debug["selected_compression_ratio"] == pytest.approx(0.5)
    assert debug["stop_reason"] == "fixed_compression_target"


def test_unified_batch_merge_does_not_select_filtered_edges_as_mutual_pairs():
    model = Aggregator.__new__(Aggregator)
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    mapping, selected_sources, debug = model._batch_mutual_nearest_group_merge(
        features,
        np.arange(2, dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        protected=np.asarray([True, False]),
        lambda_cost=0.1,
        cost_denominator=2.0,
    )

    assert mapping.tolist() == [0, 1]
    assert selected_sources.tolist() == [0, 1]
    assert debug["accepted_merges"] == 0
    assert debug["stop_reason"] == "no_mergeable_edges"


def test_unified_debug_uses_frame_count_for_attention_token_statistics():
    model = Aggregator.__new__(Aggregator)
    model.patch_token_start = 1
    model.frame_fusion_mode = "selftr"
    model.frame_fusion_lambda_cost = 0.25
    model.frame_fusion_temporal_window = 1
    model.frame_fusion_spatial_neighborhood = "N8"
    model.frame_fusion_representative_update = "parent"
    model._frame_fusion_patch_grid_size = (1, 1)

    tokens = torch.randn((2, 3, 2, 4), generator=torch.Generator().manual_seed(13))
    plans = model._build_spatiotemporal_representative_plans(tokens, source_layer=-1)
    debug = model.last_frame_fusion_debug

    assert debug["lambda_cost"] == pytest.approx(0.25)
    assert debug["selection"] == "mutual_nearest_neighbor_delta_E_lt_2_lambda"
    assert debug["cost_scope"] == "non_reference_patch_tokens"
    assert debug["cost_denominator"] == "(F - 1) * P"
    assert debug["max_token_count"] == pytest.approx((3 - 1) * 1)
    assert debug["token_count_normalization"] == (
        "active_non_reference_tokens / ((F - 1) * P)"
    )
    assert debug["representative_update"] == "best-of-parents"
    assert debug["representative_value_aggregation"] == "group-mean"
    for batch_debug, plan in zip(debug["batches"], plans):
        assert batch_debug["attention_tokens"] == 3 + plan.representative_source_indices.numel()


def test_unified_mean_representatives_average_every_token_in_final_group():
    patch_tokens = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 2.0],
            [-1.0, 4.0],
            [5.0, 0.0],
        ]
    )
    mapping = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    representatives = Aggregator._mean_group_representatives(
        patch_tokens,
        mapping,
        representative_count=2,
    )

    assert torch.allclose(
        representatives,
        torch.tensor([[2.0, 1.0], [2.0, 2.0]]),
    )


def test_unified_last_representatives_keep_the_final_representative_token():
    patch_tokens = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 2.0],
            [-1.0, 4.0],
            [5.0, 0.0],
        ]
    )
    representative_sources = torch.tensor([1, 3], dtype=torch.long)

    representatives = Aggregator._last_group_representatives(
        patch_tokens,
        representative_sources,
    )

    assert torch.equal(
        representatives,
        torch.tensor([[3.0, 2.0], [5.0, 0.0]]),
    )
