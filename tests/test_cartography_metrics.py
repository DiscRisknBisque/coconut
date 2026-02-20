import torch

from analysis.cartography_utils import compute_metrics


def test_metrics_with_teleport_and_backtracking():
    decoded = [0, 1, 4, 1, 1]
    edge_index = torch.tensor(
        [
            [0, 1, 1, 4],
            [1, 0, 4, 1],
        ],
        dtype=torch.long,
    )

    metrics = compute_metrics(decoded, edge_index)

    assert metrics["num_steps"] == 5
    assert metrics["num_unique_nodes"] == 3
    assert metrics["teleportation_rate"] == 0.0
    assert metrics["avg_dwell_time"] > 1.0
    assert metrics["backtracking_count"] == 1
    assert metrics["backtracking_frequency"] > 0.0


def test_metrics_with_missing_edge_is_teleport():
    decoded = [0, 2, 3]
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    metrics = compute_metrics(decoded, edge_index)

    assert metrics["teleportation_rate"] == 1.0
    assert metrics["avg_dwell_time"] == 1.0

