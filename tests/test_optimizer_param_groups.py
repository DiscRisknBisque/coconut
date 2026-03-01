import torch.nn as nn

from run import _build_optimizer_param_groups


class DummyGroupedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_causallm = nn.Linear(4, 4)
        self.kge_projector = nn.Linear(4, 4)
        self.kge_residual_norm = nn.LayerNorm(4)
        self.other_head = nn.Linear(4, 2)


def _group_names(groups):
    return [group["name"] for group in groups]


def test_group_builder_splits_base_and_projection_params():
    model = DummyGroupedModel()

    groups, named, missing = _build_optimizer_param_groups(
        model,
        lr=1e-4,
        lr_base_llm=5e-5,
        lr_projection_mlp=2e-4,
    )

    assert _group_names(groups) == ["base_llm", "projection_mlp", "other"]
    assert len(named["base_llm"]) > 0
    assert len(named["projection_mlp"]) > 0
    assert len(named["other"]) > 0
    assert missing == []


def test_group_builder_skips_frozen_base_group():
    model = DummyGroupedModel()
    for param in model.base_causallm.parameters():
        param.requires_grad = False

    groups, named, missing = _build_optimizer_param_groups(
        model,
        lr=1e-4,
        lr_base_llm=5e-5,
        lr_projection_mlp=2e-4,
    )

    assert "base_llm" not in _group_names(groups)
    assert "projection_mlp" in _group_names(groups)
    assert "other" in _group_names(groups)
    assert "base_llm" in missing
    assert "base_llm" not in named


def test_group_builder_falls_back_to_single_group_without_overrides():
    model = DummyGroupedModel()

    groups, named, missing = _build_optimizer_param_groups(
        model,
        lr=1e-4,
        lr_base_llm=None,
        lr_projection_mlp=None,
    )

    assert _group_names(groups) == ["all"]
    assert groups[0]["lr"] == 1e-4
    assert len(named["all"]) == len(
        [param for _, param in model.named_parameters() if param.requires_grad]
    )
    assert missing == []


def test_group_lr_resolution_uses_overrides_and_fallback_to_lr():
    model = DummyGroupedModel()

    groups, _, _ = _build_optimizer_param_groups(
        model,
        lr=1e-4,
        lr_base_llm=5e-5,
        lr_projection_mlp=None,
    )
    lrs = {group["name"]: group["lr"] for group in groups}

    assert lrs["base_llm"] == 5e-5
    assert lrs["projection_mlp"] == 1e-4
    assert lrs["other"] == 1e-4


def test_no_param_overlap_across_groups():
    model = DummyGroupedModel()

    groups, _, _ = _build_optimizer_param_groups(
        model,
        lr=1e-4,
        lr_base_llm=5e-5,
        lr_projection_mlp=2e-4,
    )

    grouped_param_ids = []
    for group in groups:
        grouped_param_ids.extend([id(param) for param in group["params"]])

    assert len(grouped_param_ids) == len(set(grouped_param_ids))
