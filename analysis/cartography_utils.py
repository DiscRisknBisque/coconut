"""Utilities for decoding and visualizing latent trajectories over graph nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def decode_trajectory(
    trajectory: torch.Tensor,
    node_embeddings: torch.Tensor,
    node_labels: Sequence[str],
    metric: str = "cosine",
    stability_threshold: float = 0.85,
) -> Tuple[List[Dict], torch.Tensor]:
    """Map each trajectory step to its nearest node."""

    if trajectory.ndim != 2:
        raise ValueError(f"Expected trajectory shape [T, D], got {tuple(trajectory.shape)}")
    if node_embeddings.ndim != 2:
        raise ValueError(
            f"Expected node_embeddings shape [N, D], got {tuple(node_embeddings.shape)}"
        )
    if trajectory.shape[1] != node_embeddings.shape[1]:
        raise ValueError(
            "Trajectory and node embeddings must share hidden dimension: "
            f"{trajectory.shape[1]} vs {node_embeddings.shape[1]}"
        )
    if len(node_labels) != node_embeddings.shape[0]:
        raise ValueError(
            f"Expected {node_embeddings.shape[0]} node labels, got {len(node_labels)}"
        )

    metric = metric.lower()
    if metric not in {"cosine", "euclidean"}:
        raise ValueError("metric must be one of: cosine, euclidean")

    if metric == "cosine":
        traj_norm = F.normalize(trajectory, p=2, dim=1)
        nodes_norm = F.normalize(node_embeddings, p=2, dim=1)
        score_matrix = torch.mm(traj_norm, nodes_norm.t())
        best_scores, best_indices = torch.max(score_matrix, dim=1)
        confidence = (best_scores + 1.0) / 2.0
    else:
        dists = torch.cdist(trajectory, node_embeddings, p=2)
        score_matrix = -dists
        best_dists, best_indices = torch.min(dists, dim=1)
        confidence = torch.exp(-best_dists)

    decoded_path: List[Dict] = []
    for step, idx in enumerate(best_indices.tolist()):
        conf = float(confidence[step].item())
        decoded_path.append(
            {
                "step": step,
                "predicted_node_index": idx,
                "predicted_node": node_labels[idx],
                "confidence": conf,
                "is_stable": conf > stability_threshold,
            }
        )

    return decoded_path, score_matrix


def compute_metrics(decoded_indices: Sequence[int], edge_index: torch.Tensor) -> Dict[str, float]:
    """Compute teleportation, dwell-time, and backtracking metrics."""

    if len(decoded_indices) == 0:
        return {
            "num_steps": 0,
            "num_unique_nodes": 0,
            "teleportation_rate": 0.0,
            "avg_dwell_time": 0.0,
            "backtracking_frequency": 0.0,
            "backtracking_count": 0,
        }

    edge_set = set()
    if edge_index.numel() > 0:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
        for src, dst in edge_index.t().tolist():
            edge_set.add((int(src), int(dst)))

    moves = 0
    teleports = 0
    for src, dst in zip(decoded_indices[:-1], decoded_indices[1:]):
        if src == dst:
            continue
        moves += 1
        if (src, dst) not in edge_set:
            teleports += 1
    teleportation_rate = float(teleports / moves) if moves > 0 else 0.0

    dwell_runs: List[int] = []
    run_len = 1
    for prev_node, cur_node in zip(decoded_indices[:-1], decoded_indices[1:]):
        if cur_node == prev_node:
            run_len += 1
        else:
            dwell_runs.append(run_len)
            run_len = 1
    dwell_runs.append(run_len)
    avg_dwell = float(sum(dwell_runs) / len(dwell_runs))

    seen = {decoded_indices[0]}
    backtracking_count = 0
    for prev_node, cur_node in zip(decoded_indices[:-1], decoded_indices[1:]):
        if cur_node in seen and cur_node != prev_node:
            backtracking_count += 1
        seen.add(cur_node)
    backtracking_frequency = float(backtracking_count / max(len(decoded_indices) - 1, 1))

    return {
        "num_steps": int(len(decoded_indices)),
        "num_unique_nodes": int(len(set(decoded_indices))),
        "teleportation_rate": teleportation_rate,
        "avg_dwell_time": avg_dwell,
        "backtracking_frequency": backtracking_frequency,
        "backtracking_count": int(backtracking_count),
    }


def _stat_summary(values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }


def nearest_assignments(
    trajectory: torch.Tensor,
    node_embeddings: torch.Tensor,
    node_labels: Sequence[str],
    metric: str,
) -> List[Dict]:
    metric = metric.lower()
    if metric == "cosine":
        traj = F.normalize(trajectory, p=2, dim=1)
        nodes = F.normalize(node_embeddings, p=2, dim=1)
        scores = torch.mm(traj, nodes.t())
        values, indices = scores.max(dim=1)
        confidence = (values + 1.0) / 2.0
        score_name = "score"
    elif metric == "euclidean":
        dists = torch.cdist(trajectory, node_embeddings, p=2)
        values, indices = dists.min(dim=1)
        confidence = torch.exp(-values)
        score_name = "distance"
    else:
        raise ValueError("metric must be one of: cosine, euclidean")

    assignments: List[Dict] = []
    for step in range(trajectory.shape[0]):
        idx = int(indices[step].item())
        assignments.append(
            {
                "step": step,
                "predicted_node_index": idx,
                "predicted_node": node_labels[idx],
                score_name: float(values[step].item()),
                "confidence": float(confidence[step].item()),
            }
        )
    return assignments


def build_diagnostics(
    node_embeddings: torch.Tensor,
    trajectory: torch.Tensor,
    node_labels: Sequence[str],
    raw_projection_meta: Dict,
    normalized_projection_meta: Dict,
) -> Dict:
    node_norms = torch.norm(node_embeddings, dim=1)
    traj_norms = torch.norm(trajectory, dim=1) if trajectory.numel() > 0 else torch.empty((0,))
    node_centroid = node_embeddings.mean(dim=0)
    node_centroid_spread = torch.norm(node_embeddings - node_centroid, dim=1)
    traj_to_centroid = (
        torch.norm(trajectory - node_centroid, dim=1)
        if trajectory.numel() > 0
        else torch.empty((0,))
    )

    if trajectory.numel() > 0:
        cosine_assignments = nearest_assignments(
            trajectory=trajectory,
            node_embeddings=node_embeddings,
            node_labels=node_labels,
            metric="cosine",
        )
        l2_assignments = nearest_assignments(
            trajectory=trajectory,
            node_embeddings=node_embeddings,
            node_labels=node_labels,
            metric="euclidean",
        )
    else:
        cosine_assignments = []
        l2_assignments = []

    node_radius_mean = float(raw_projection_meta["node_radius_stats"]["mean"])
    traj_radius_mean = float(raw_projection_meta["trajectory_radius_stats"]["mean"])
    radius_ratio = traj_radius_mean / max(node_radius_mean, 1e-8)

    return {
        "node_norm_stats": _stat_summary(node_norms),
        "trajectory_norm_stats": _stat_summary(traj_norms),
        "node_centroid_spread_stats": _stat_summary(node_centroid_spread),
        "trajectory_to_node_centroid_distances": [float(v) for v in traj_to_centroid.tolist()],
        "nearest_assignments": {
            "cosine": cosine_assignments,
            "euclidean": l2_assignments,
        },
        "projection": {
            "raw": raw_projection_meta,
            "l2_normalized": normalized_projection_meta,
        },
        "radius_ratio_raw_traj_over_nodes_mean": radius_ratio,
    }


def _prepare_projection_inputs(
    node_embeddings: torch.Tensor,
    trajectory: torch.Tensor,
    projection_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if projection_mode not in {"raw", "l2_normalized"}:
        raise ValueError("projection_mode must be one of: raw, l2_normalized")
    if projection_mode == "raw":
        return node_embeddings, trajectory
    return F.normalize(node_embeddings, p=2, dim=1), F.normalize(trajectory, p=2, dim=1)


def project_embeddings_with_metadata(
    node_embeddings: torch.Tensor,
    trajectory: torch.Tensor,
    projection_mode: str = "raw",
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Fit PCA on nodes, then project trajectory using the same transform."""

    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for projection. Install with `pip install scikit-learn`."
        ) from exc

    nodes_for_proj, traj_for_proj = _prepare_projection_inputs(
        node_embeddings, trajectory, projection_mode=projection_mode
    )
    node_np = nodes_for_proj.detach().cpu().numpy()
    traj_np = traj_for_proj.detach().cpu().numpy()

    pca = PCA(n_components=2)
    nodes_2d = torch.from_numpy(pca.fit_transform(node_np)).float()
    path_2d = torch.from_numpy(pca.transform(traj_np)).float()

    node_centroid = nodes_for_proj.mean(dim=0)
    node_radius = torch.norm(nodes_for_proj - node_centroid, dim=1)
    traj_radius = torch.norm(traj_for_proj - node_centroid, dim=1)

    metadata = {
        "projection_mode": projection_mode,
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_.tolist()],
        "explained_variance_ratio_sum": float(sum(pca.explained_variance_ratio_.tolist())),
        "node_radius_stats": _stat_summary(node_radius),
        "trajectory_radius_stats": _stat_summary(traj_radius),
    }
    return nodes_2d, path_2d, metadata


def project_embeddings(
    node_embeddings: torch.Tensor, trajectory: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible wrapper for raw-space projection."""

    nodes_2d, path_2d, _ = project_embeddings_with_metadata(
        node_embeddings=node_embeddings,
        trajectory=trajectory,
        projection_mode="raw",
    )
    return nodes_2d, path_2d


def _draw_panel(
    ax,
    *,
    nodes_2d: torch.Tensor,
    path_2d: torch.Tensor,
    node_labels: Sequence[str],
    edge_index: torch.Tensor,
    title: str,
    nearest_node_indices: Sequence[int] | None,
    summary_text: str | None,
):
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    if edge_index.numel() > 0:
        for src, dst in edge_index.t().tolist():
            x0, y0 = nodes_2d[src].tolist()
            x1, y1 = nodes_2d[dst].tolist()
            ax.plot([x0, x1], [y0, y1], color="#b0b0b0", linewidth=0.7, alpha=0.4)

    ax.scatter(nodes_2d[:, 0], nodes_2d[:, 1], color="#1f77b4", s=70, alpha=0.8, label="Nodes")
    for i, label in enumerate(node_labels):
        x, y = nodes_2d[i].tolist()
        ax.annotate(label, (x + 0.02, y + 0.02), fontsize=7, alpha=0.9)

    if path_2d.numel() == 0:
        return None

    t = torch.arange(path_2d.shape[0])
    sc = ax.scatter(
        path_2d[:, 0],
        path_2d[:, 1],
        c=t,
        cmap="viridis",
        s=35,
        alpha=0.9,
        label="Trajectory",
    )
    ax.plot(path_2d[:, 0], path_2d[:, 1], color="#d62728", linewidth=1.2, alpha=0.6)
    for i in range(path_2d.shape[0]):
        ax.annotate(f"{i}", (float(path_2d[i, 0]) + 0.02, float(path_2d[i, 1]) + 0.02), fontsize=8)

    if nearest_node_indices is not None:
        for step, node_idx in enumerate(nearest_node_indices):
            if step >= path_2d.shape[0]:
                break
            if node_idx < 0 or node_idx >= nodes_2d.shape[0]:
                continue
            x0, y0 = path_2d[step].tolist()
            x1, y1 = nodes_2d[node_idx].tolist()
            ax.plot([x0, x1], [y0, y1], linestyle="--", linewidth=0.8, alpha=0.5, color="#ff7f0e")

    if summary_text:
        ax.text(
            0.01,
            0.99,
            summary_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
        )
    return sc


def plot_cartography(
    nodes_2d: torch.Tensor,
    path_2d: torch.Tensor,
    node_labels: Sequence[str],
    edge_index: torch.Tensor,
    out_png: Path,
    out_gif: Path | None = None,
    nearest_node_indices: Sequence[int] | None = None,
    summary_text: str | None = None,
) -> None:
    """Render static plot (and optional GIF) of map + latent trajectory."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install with `pip install matplotlib pillow`."
        ) from exc

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    sc = _draw_panel(
        ax,
        nodes_2d=nodes_2d,
        path_2d=path_2d,
        node_labels=node_labels,
        edge_index=edge_index,
        title="ProsQA Latent Cartography",
        nearest_node_indices=nearest_node_indices,
        summary_text=summary_text,
    )
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="Step")

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    if out_gif is None or path_2d.shape[0] == 0:
        return

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title("ProsQA Latent Cartography (Animated)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.scatter(nodes_2d[:, 0], nodes_2d[:, 1], color="#1f77b4", s=70, alpha=0.8)
    for i, label in enumerate(node_labels):
        x, y = nodes_2d[i].tolist()
        ax.annotate(label, (x + 0.02, y + 0.02), fontsize=7, alpha=0.9)
    if edge_index.numel() > 0:
        for src, dst in edge_index.t().tolist():
            x0, y0 = nodes_2d[src].tolist()
            x1, y1 = nodes_2d[dst].tolist()
            ax.plot([x0, x1], [y0, y1], color="#b0b0b0", linewidth=0.7, alpha=0.4)

    line, = ax.plot([], [], color="#d62728", linewidth=1.4, alpha=0.8)
    point, = ax.plot([], [], "o", color="#ff7f0e", markersize=7)

    def _update(frame: int):
        line.set_data(path_2d[: frame + 1, 0], path_2d[: frame + 1, 1])
        point.set_data([path_2d[frame, 0]], [path_2d[frame, 1]])
        return line, point

    ani = animation.FuncAnimation(fig, _update, frames=path_2d.shape[0], interval=120, blit=True)
    ani.save(str(out_gif), writer="pillow", dpi=120)
    plt.close(fig)


def plot_cartography_dual(
    *,
    raw_nodes_2d: torch.Tensor,
    raw_path_2d: torch.Tensor,
    norm_nodes_2d: torch.Tensor,
    norm_path_2d: torch.Tensor,
    node_labels: Sequence[str],
    edge_index: torch.Tensor,
    out_png: Path,
    nearest_node_indices: Sequence[int] | None = None,
    summary_text: str | None = None,
) -> None:
    """Render side-by-side raw and L2-normalized cartography views."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install with `pip install matplotlib pillow`."
        ) from exc

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sc_left = _draw_panel(
        axes[0],
        nodes_2d=raw_nodes_2d,
        path_2d=raw_path_2d,
        node_labels=node_labels,
        edge_index=edge_index,
        title="Raw PCA Space",
        nearest_node_indices=nearest_node_indices,
        summary_text=summary_text,
    )
    sc_right = _draw_panel(
        axes[1],
        nodes_2d=norm_nodes_2d,
        path_2d=norm_path_2d,
        node_labels=node_labels,
        edge_index=edge_index,
        title="L2-Normalized PCA Space",
        nearest_node_indices=nearest_node_indices,
        summary_text=summary_text,
    )

    if sc_left is not None:
        fig.colorbar(sc_left, ax=axes[0], label="Step")
    if sc_right is not None:
        fig.colorbar(sc_right, ax=axes[1], label="Step")
    fig.suptitle("ProsQA Latent Cartography (Raw vs Normalized)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_json(data: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)
