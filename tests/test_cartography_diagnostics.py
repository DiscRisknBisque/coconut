import torch

from analysis.cartography_utils import build_diagnostics, decode_trajectory


def test_build_diagnostics_includes_required_sections():
    node_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    trajectory = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
        ]
    )
    labels = ["n0", "n1", "n2"]
    raw_meta = {
        "projection_mode": "raw",
        "explained_variance_ratio": [0.7, 0.2],
        "explained_variance_ratio_sum": 0.9,
        "node_radius_stats": {"min": 0.1, "mean": 0.5, "max": 1.0},
        "trajectory_radius_stats": {"min": 0.2, "mean": 0.8, "max": 1.3},
    }
    norm_meta = {
        "projection_mode": "l2_normalized",
        "explained_variance_ratio": [0.6, 0.3],
        "explained_variance_ratio_sum": 0.9,
        "node_radius_stats": {"min": 0.1, "mean": 0.4, "max": 0.9},
        "trajectory_radius_stats": {"min": 0.2, "mean": 0.5, "max": 0.8},
    }

    diagnostics = build_diagnostics(
        node_embeddings=node_embeddings,
        trajectory=trajectory,
        node_labels=labels,
        raw_projection_meta=raw_meta,
        normalized_projection_meta=norm_meta,
    )

    assert "node_norm_stats" in diagnostics
    assert "trajectory_norm_stats" in diagnostics
    assert "node_centroid_spread_stats" in diagnostics
    assert "trajectory_to_node_centroid_distances" in diagnostics
    assert "nearest_assignments" in diagnostics
    assert "projection" in diagnostics
    assert "radius_ratio_raw_traj_over_nodes_mean" in diagnostics
    assert len(diagnostics["trajectory_to_node_centroid_distances"]) == 2


def test_build_diagnostics_cosine_assignments_match_decode_trajectory():
    node_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    trajectory = torch.tensor(
        [
            [0.8, 0.1],
            [0.1, 0.8],
            [-0.9, 0.2],
        ]
    )
    labels = ["n0", "n1", "n2"]
    raw_meta = {
        "projection_mode": "raw",
        "explained_variance_ratio": [0.6, 0.3],
        "explained_variance_ratio_sum": 0.9,
        "node_radius_stats": {"min": 0.1, "mean": 0.5, "max": 1.0},
        "trajectory_radius_stats": {"min": 0.2, "mean": 0.6, "max": 1.1},
    }
    norm_meta = {
        "projection_mode": "l2_normalized",
        "explained_variance_ratio": [0.55, 0.35],
        "explained_variance_ratio_sum": 0.9,
        "node_radius_stats": {"min": 0.1, "mean": 0.3, "max": 0.8},
        "trajectory_radius_stats": {"min": 0.2, "mean": 0.4, "max": 0.7},
    }

    decoded, _ = decode_trajectory(
        trajectory=trajectory,
        node_embeddings=node_embeddings,
        node_labels=labels,
        metric="cosine",
        stability_threshold=0.7,
    )
    diagnostics = build_diagnostics(
        node_embeddings=node_embeddings,
        trajectory=trajectory,
        node_labels=labels,
        raw_projection_meta=raw_meta,
        normalized_projection_meta=norm_meta,
    )

    expected = [entry["predicted_node_index"] for entry in decoded]
    actual = [
        entry["predicted_node_index"]
        for entry in diagnostics["nearest_assignments"]["cosine"]
    ]
    assert actual == expected
