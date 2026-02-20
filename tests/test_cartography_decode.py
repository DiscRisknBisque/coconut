import pytest
import torch

from analysis.cartography_utils import decode_trajectory, project_embeddings_with_metadata


def test_decode_trajectory_cosine():
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

    decoded, matrix = decode_trajectory(
        trajectory=trajectory,
        node_embeddings=node_embeddings,
        node_labels=labels,
        metric="cosine",
        stability_threshold=0.7,
    )

    assert matrix.shape == (3, 3)
    assert [d["predicted_node_index"] for d in decoded] == [0, 1, 2]
    assert all(0.0 <= d["confidence"] <= 1.0 for d in decoded)


def test_decode_trajectory_euclidean():
    node_embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    trajectory = torch.tensor(
        [
            [0.1, 0.2],
            [1.9, 0.1],
            [0.2, 1.8],
        ]
    )
    labels = ["a", "b", "c"]

    decoded, _ = decode_trajectory(
        trajectory=trajectory,
        node_embeddings=node_embeddings,
        node_labels=labels,
        metric="euclidean",
        stability_threshold=0.1,
    )

    assert [d["predicted_node_index"] for d in decoded] == [0, 1, 2]


def test_project_embeddings_metadata_has_expected_fields():
    pytest.importorskip("sklearn")
    node_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    trajectory = torch.tensor(
        [
            [0.8, 0.1, 0.0],
            [0.1, 0.8, 0.0],
        ]
    )

    nodes_2d, path_2d, metadata = project_embeddings_with_metadata(
        node_embeddings=node_embeddings,
        trajectory=trajectory,
        projection_mode="raw",
    )

    assert nodes_2d.shape == (4, 2)
    assert path_2d.shape == (2, 2)
    assert metadata["projection_mode"] == "raw"
    assert len(metadata["explained_variance_ratio"]) == 2
    assert 0.0 <= metadata["explained_variance_ratio_sum"] <= 1.0
    assert set(metadata["node_radius_stats"].keys()) == {"min", "mean", "max"}
    assert set(metadata["trajectory_radius_stats"].keys()) == {"min", "mean", "max"}


def test_l2_normalized_projection_reduces_scale_mismatch_radius_gap():
    pytest.importorskip("sklearn")
    node_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
        ]
    )
    trajectory = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [9.0, 1.0, 0.0],
        ]
    )

    _, _, raw_meta = project_embeddings_with_metadata(
        node_embeddings=node_embeddings,
        trajectory=trajectory,
        projection_mode="raw",
    )
    _, _, norm_meta = project_embeddings_with_metadata(
        node_embeddings=node_embeddings,
        trajectory=trajectory,
        projection_mode="l2_normalized",
    )

    raw_ratio = raw_meta["trajectory_radius_stats"]["mean"] / max(
        raw_meta["node_radius_stats"]["mean"], 1e-8
    )
    norm_ratio = norm_meta["trajectory_radius_stats"]["mean"] / max(
        norm_meta["node_radius_stats"]["mean"], 1e-8
    )
    assert norm_ratio < raw_ratio
