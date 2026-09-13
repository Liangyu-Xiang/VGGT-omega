from selftr import DEFAULT_METHOD_NAME, METHOD_ID, SelfTR, SelfTRConfig
from selftr.identity import canonical_frame_fusion_mode, resolve_method_name


def test_identity_uses_a_stable_id_and_editable_display_name(monkeypatch):
    assert METHOD_ID == "selftr"
    assert resolve_method_name() == DEFAULT_METHOD_NAME
    monkeypatch.setenv("SELFTR_METHOD_NAME", "Renamed Method")
    assert resolve_method_name() == "Renamed Method"


def test_legacy_um_mode_maps_to_selftr():
    assert canonical_frame_fusion_mode("u-m") == METHOD_ID
    assert canonical_frame_fusion_mode("selftr") == METHOD_ID


def test_config_builds_canonical_model_kwargs():
    config = SelfTRConfig(temporal_window=4, spatial_radius=2)
    assert config.model_kwargs()["frame_fusion_mode"] == METHOD_ID
    assert config.model_kwargs()["frame_fusion_temporal_window"] == 4
    assert config.metadata()["method_name"] == DEFAULT_METHOD_NAME


def test_public_model_enables_selftr_without_fastvggt_merging():
    model = SelfTR(
        patch_size=16,
        embed_dim=64,
        enable_camera=False,
        enable_depth=False,
        frame_fusion_recompute_layers=(),
    )
    assert model.aggregator.frame_fusion_mode == METHOD_ID
    assert model.aggregator.merge_ratio == 0.0
