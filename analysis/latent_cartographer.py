"""ProsQA-only CLI for latent cartography over Coconut+GNN trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coconut import Coconut
from preprocessing.prosqa_nodes import extract_prosqa_nodes

from analysis.cartography_utils import (
    build_diagnostics,
    compute_metrics,
    decode_trajectory,
    plot_cartography,
    plot_cartography_dual,
    project_embeddings_with_metadata,
    save_json,
)


def _normalize_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "_fsdp_wrapped_module.", "_orig_mod.")
    normalized = {}
    for key, value in state.items():
        clean_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    changed = True
        normalized[clean_key] = value
    return normalized


def _load_weights(base_model, coconut_model, checkpoint_path: Path):
    raw_state = torch.load(checkpoint_path, map_location="cpu")
    state = _normalize_state_dict_keys(raw_state)

    has_coconut_keys = any(k.startswith("base_causallm.") for k in state.keys())
    if has_coconut_keys:
        load_result = coconut_model.load_state_dict(state, strict=False)
        print(f"Loaded Coconut weights from {checkpoint_path}: {load_result}")
    else:
        base_result = base_model.load_state_dict(state, strict=False)
        print(f"Loaded base model weights from {checkpoint_path}: {base_result}")


def _resolve_split_path(configs, split: str) -> Path:
    if split == "train":
        return Path(configs.get("train_path", "data/prosqa_train.json"))
    if split in {"valid", "val"}:
        return Path(configs.get("val_path", "data/prosqa_valid.json"))
    if split == "test":
        val_path = configs.get("val_path")
        if val_path and "test" in Path(val_path).name.lower():
            return Path(val_path)
        return Path("data/prosqa_test.json")
    raise ValueError("split must be one of: train, valid, test")


def _build_graph_config(configs: Dict) -> Dict:
    graph_dim = configs.get("graph_dim", 256)
    graph_dropout = configs.get("graph_dropout", 0.1)
    return {
        "use_graph": bool(configs.get("use_graph", True)),
        "graph_encoder": configs.get("graph_encoder", "gcn2"),
        "graph_dim": graph_dim,
        "graph_prefix_len": configs.get("graph_prefix_len", 4),
        "align_loss_weight": configs.get("align_loss_weight", 0.0),
        "latent_injection": configs.get("latent_injection", "residual"),
        "graph_dropout": graph_dropout,
        "graph_projector_hidden": configs.get("graph_projector_hidden", graph_dim),
        "graph_projector_dropout": configs.get("graph_projector_dropout", graph_dropout),
        "graph_prefix_dropout": configs.get("graph_prefix_dropout", graph_dropout),
        "graph_hidden_dim": configs.get("graph_hidden_dim", graph_dim),
        "graph_input_dim": configs.get("graph_input_dim", 259),
    }


def _load_sample(split_path: Path, sample_idx: int) -> Dict:
    with split_path.open() as f:
        data = json.load(f)
    if sample_idx < 0 or sample_idx >= len(data):
        raise IndexError(
            f"sample_idx={sample_idx} out of range for {split_path} with {len(data)} examples"
        )
    sample = data[sample_idx]
    sample["idx"] = sample_idx
    return sample


def _load_graph_sidecar(graph_sidecar_root: Path, split: str, sample_idx: int) -> Dict[str, torch.Tensor]:
    manifest_path = graph_sidecar_root / split / f"manifest_{split}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open() as f:
        manifest = json.load(f)
    graph_path = manifest.get(str(sample_idx))
    if graph_path is None:
        raise KeyError(f"Sample idx={sample_idx} missing from manifest {manifest_path}")
    return torch.load(graph_path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description="ProsQA latent cartography")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--sample-idx", type=int, required=True)
    parser.add_argument("--scheduled-stage", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument(
        "--projection-mode",
        choices=["raw", "l2_normalized"],
        default="l2_normalized",
        help="Projection mode for single-view cartography.png output.",
    )
    parser.add_argument(
        "--plot-layout",
        choices=["single", "dual"],
        default="dual",
        help="Render single cartography view or side-by-side raw/normalized view.",
    )
    parser.add_argument(
        "--distance-metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Metric used to select nearest-node connector lines in plots.",
    )
    parser.add_argument(
        "--diagnostics-json",
        dest="diagnostics_json",
        action="store_true",
        default=True,
        help="Write diagnostics.json with geometry and nearest-neighbor summaries (default: enabled).",
    )
    parser.add_argument(
        "--no-diagnostics-json",
        dest="diagnostics_json",
        action="store_false",
        help="Disable diagnostics.json output.",
    )
    parser.add_argument("--stability-threshold", type=float, default=0.85)
    parser.add_argument("--output-dir", default="analysis_outputs")
    parser.add_argument("--save-gif", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)

    graph_sidecar_root = configs.get("graph_sidecar_root")
    if not graph_sidecar_root:
        raise ValueError("graph_sidecar_root must be set in config for cartography")

    split_path = _resolve_split_path(configs, args.split)
    sample = _load_sample(split_path, args.sample_idx)
    graph_data = _load_graph_sidecar(Path(graph_sidecar_root), args.split, args.sample_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(configs["model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base_model = AutoModelForCausalLM.from_pretrained(configs["model_id"])
    base_model.resize_token_embeddings(len(tokenizer))
    target_id = tokenizer.convert_tokens_to_ids("<<")
    embeddings = base_model.get_input_embeddings()
    for token_id in [latent_id, start_id, end_id]:
        embeddings.weight.data[token_id] = embeddings.weight.data[target_id]
        base_model.lm_head.weight.data[token_id] = base_model.lm_head.weight.data[target_id]

    graph_config = _build_graph_config(configs)
    graph_config["use_graph"] = True
    model = Coconut(
        base_model,
        latent_id,
        start_id,
        end_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        graph_config=graph_config,
    )
    _load_weights(base_model, model, Path(args.checkpoint))
    model = model.to(device)
    model.eval()

    scheduled_stage = (
        args.scheduled_stage if args.scheduled_stage is not None else configs.get("max_latent_stage", 6)
    )
    c_thought = int(configs.get("c_thought", 1))
    max_latent_stage = int(configs.get("max_latent_stage", 6))
    n_steps = len(sample.get("steps", []))
    n_latent_tokens = min(scheduled_stage, min(max_latent_stage, n_steps)) * c_thought

    question_tokens = tokenizer.encode(sample["question"] + "\n", add_special_tokens=True)
    input_tokens = question_tokens + [start_id] + [latent_id] * n_latent_tokens + [end_id]
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    graph_x = graph_data["x"].float().to(device)
    graph_edge_index = graph_data["edge_index"].long().to(device)
    graph_role = graph_data.get("role")
    if graph_role is None:
        graph_role = torch.zeros((graph_x.shape[0],), dtype=torch.long)
    graph_role = graph_role.long().to(device)
    graph_batch = torch.zeros((graph_x.shape[0],), dtype=torch.long, device=device)

    analysis_trace: Dict = {}
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            graph_x=graph_x,
            graph_edge_index=graph_edge_index,
            graph_batch=graph_batch,
            graph_role=graph_role,
            analysis_trace=analysis_trace,
            analysis_batch_index=0,
        )

    node_embeddings = analysis_trace.get("node_embeddings")
    if node_embeddings is None:
        raise RuntimeError("analysis_trace did not contain node_embeddings")

    trajectory_list = analysis_trace.get("trajectory", [])
    if trajectory_list:
        trajectory = torch.stack(trajectory_list, dim=0)
    else:
        trajectory = torch.empty((0, node_embeddings.shape[1]), dtype=node_embeddings.dtype)

    edges_local = analysis_trace.get("edge_index", torch.empty((2, 0), dtype=torch.long))
    entries, _ = extract_prosqa_nodes(sample)
    node_labels = [f"{role}:{text}" for role, text in entries]
    if len(node_labels) != node_embeddings.shape[0]:
        node_labels = [f"node_{i}" for i in range(node_embeddings.shape[0])]

    if trajectory.shape[0] > 0:
        decoded_path, score_matrix = decode_trajectory(
            trajectory=trajectory,
            node_embeddings=node_embeddings,
            node_labels=node_labels,
            metric=args.metric,
            stability_threshold=args.stability_threshold,
        )
    else:
        decoded_path = []
        score_matrix = torch.empty((0, node_embeddings.shape[0]))

    decoded_indices = [entry["predicted_node_index"] for entry in decoded_path]
    metrics = compute_metrics(decoded_indices, edges_local)

    raw_nodes_2d, raw_path_2d, raw_projection_meta = project_embeddings_with_metadata(
        node_embeddings=node_embeddings,
        trajectory=trajectory,
        projection_mode="raw",
    )
    norm_nodes_2d, norm_path_2d, norm_projection_meta = project_embeddings_with_metadata(
        node_embeddings=node_embeddings,
        trajectory=trajectory,
        projection_mode="l2_normalized",
    )

    if args.projection_mode == "raw":
        nodes_2d, path_2d = raw_nodes_2d, raw_path_2d
        selected_projection_meta = raw_projection_meta
    else:
        nodes_2d, path_2d = norm_nodes_2d, norm_path_2d
        selected_projection_meta = norm_projection_meta

    if trajectory.shape[0] > 0:
        metric_decoded_path, _ = decode_trajectory(
            trajectory=trajectory,
            node_embeddings=node_embeddings,
            node_labels=node_labels,
            metric=args.distance_metric,
            stability_threshold=args.stability_threshold,
        )
        nearest_node_indices = [entry["predicted_node_index"] for entry in metric_decoded_path]
    else:
        nearest_node_indices = []

    if trajectory.shape[0] > 0:
        cosine_path, _ = decode_trajectory(
            trajectory=trajectory,
            node_embeddings=node_embeddings,
            node_labels=node_labels,
            metric="cosine",
            stability_threshold=args.stability_threshold,
        )
        mean_cosine_conf = float(
            sum([entry["confidence"] for entry in cosine_path]) / max(len(cosine_path), 1)
        )
    else:
        mean_cosine_conf = 0.0

    raw_node_radius_mean = float(raw_projection_meta["node_radius_stats"]["mean"])
    raw_traj_radius_mean = float(raw_projection_meta["trajectory_radius_stats"]["mean"])
    radius_ratio = raw_traj_radius_mean / max(raw_node_radius_mean, 1e-8)
    summary_text = (
        f"latent_tokens={n_latent_tokens}\n"
        f"mean_cos_conf={mean_cosine_conf:.3f}\n"
        f"raw_radius_ratio={radius_ratio:.2f}"
    )

    output_dir = Path(args.output_dir) / f"{args.split}_{args.sample_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "cartography.png"
    dual_png_path = output_dir / "cartography_dual.png"
    gif_path = output_dir / "cartography.gif" if args.save_gif else None

    # Keep cartography.png as the legacy raw-space rendering for backward compatibility.
    plot_cartography(
        nodes_2d=raw_nodes_2d,
        path_2d=raw_path_2d,
        node_labels=node_labels,
        edge_index=edges_local,
        out_png=png_path,
        out_gif=gif_path,
        nearest_node_indices=nearest_node_indices,
        summary_text=summary_text,
    )

    torch.save(node_embeddings, output_dir / "node_embeddings.pt")
    torch.save(trajectory, output_dir / "reasoning_trajectory.pt")
    torch.save(score_matrix, output_dir / "similarity.pt")

    if args.plot_layout == "dual":
        plot_cartography_dual(
            raw_nodes_2d=raw_nodes_2d,
            raw_path_2d=raw_path_2d,
            norm_nodes_2d=norm_nodes_2d,
            norm_path_2d=norm_path_2d,
            node_labels=node_labels,
            edge_index=edges_local,
            out_png=dual_png_path,
            nearest_node_indices=nearest_node_indices,
            summary_text=summary_text,
        )
    elif args.projection_mode != "raw":
        # In single mode, respect selected projection mode by overwriting cartography.png.
        plot_cartography(
            nodes_2d=nodes_2d,
            path_2d=path_2d,
            node_labels=node_labels,
            edge_index=edges_local,
            out_png=png_path,
            out_gif=gif_path,
            nearest_node_indices=nearest_node_indices,
            summary_text=summary_text,
        )

    save_json(
        {
            "decoded_path": decoded_path,
            "metric": args.metric,
            "stability_threshold": args.stability_threshold,
            "sample_idx": args.sample_idx,
            "split": args.split,
        },
        output_dir / "decoded_path.json",
    )
    save_json(metrics, output_dir / "metrics.json")
    if args.diagnostics_json:
        diagnostics = build_diagnostics(
            node_embeddings=node_embeddings,
            trajectory=trajectory,
            node_labels=node_labels,
            raw_projection_meta=raw_projection_meta,
            normalized_projection_meta=norm_projection_meta,
        )
        diagnostics["selected_projection_mode"] = args.projection_mode
        diagnostics["selected_projection_metadata"] = selected_projection_meta
        diagnostics["distance_metric_for_annotations"] = args.distance_metric
        save_json(diagnostics, output_dir / "diagnostics.json")

        cosine_assignments = diagnostics["nearest_assignments"]["cosine"]
        mean_cosine_score = (
            sum(item["score"] for item in cosine_assignments) / max(len(cosine_assignments), 1)
            if cosine_assignments
            else 0.0
        )
        artifact_hint = (
            "possible projection artifact"
            if mean_cosine_score > 0.8 and radius_ratio > 3.0
            else "trajectory scale appears consistent"
        )
        print(
            "Cartography diagnostic: "
            f"mean_cosine_score={mean_cosine_score:.3f}, "
            f"radius_ratio={radius_ratio:.2f} -> {artifact_hint}"
        )
    save_json(
        {
            "generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True),
            "num_latent_tokens": n_latent_tokens,
        },
        output_dir / "generation.json",
    )

    print(f"Saved latent cartography outputs under: {output_dir}")


if __name__ == "__main__":
    main()
