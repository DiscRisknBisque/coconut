"""ProsQA-only CLI for latent cartography over Coconut+KGE trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cartography_utils import (  # noqa: E402
    build_diagnostics,
    compute_metrics,
    decode_trajectory,
    plot_cartography,
    plot_cartography_dual,
    project_embeddings_with_metadata,
    save_json,
)
from coconut import Coconut  # noqa: E402
from kge_utils import build_anchor_payload, load_entity_to_id  # noqa: E402


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


def _build_kge_config(configs: Dict) -> Dict:
    return {
        "use_kge": bool(configs.get("use_kge", True)),
        "kge_artifact_root": configs.get("kge_artifact_root"),
        "kge_entity_embeddings_file": configs.get(
            "kge_entity_embeddings_file", "entity_embeddings.pt"
        ),
        "kge_relation_embeddings_file": configs.get(
            "kge_relation_embeddings_file", "relation_embeddings.pt"
        ),
        "kge_entity_to_id_file": configs.get("kge_entity_to_id_file", "entity_to_id.json"),
        "kge_metadata_file": configs.get("kge_metadata_file", "metadata.json"),
        "kge_anchor_policy": configs.get("kge_anchor_policy", "query_anchors"),
        "kge_projector_hidden": configs.get("kge_projector_hidden"),
        "kge_projector_activation": configs.get("kge_projector_activation", "gelu"),
        "kge_projector_layernorm": configs.get("kge_projector_layernorm", True),
        "align_loss_weight": configs.get("align_loss_weight", 0.0),
        "latent_injection": configs.get("latent_injection", "residual"),
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


def _project_sample_nodes(
    model: Coconut,
    sample: Dict,
    entity_to_id: Dict[str, int],
    device: torch.device,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    symbols = sample.get("idx_to_symbol") or []
    kept_labels: List[str] = []
    kept_entity_ids: List[int] = []
    old_to_new: Dict[int, int] = {}

    for idx, symbol in enumerate(symbols):
        entity_id = entity_to_id.get(symbol)
        if entity_id is None:
            entity_id = entity_to_id.get(str(symbol).lower())
        if entity_id is None:
            continue
        old_to_new[idx] = len(kept_labels)
        kept_labels.append(str(symbol))
        kept_entity_ids.append(int(entity_id))

    if not kept_entity_ids:
        raise RuntimeError("No sample symbols mapped into entity_to_id; cannot build cartography.")

    entity_ids_tensor = torch.tensor(kept_entity_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        node_embeddings = model.project_entity_ids(entity_ids_tensor).detach().cpu()

    edges = sample.get("edges", []) or []
    remapped_edges = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if src in old_to_new and dst in old_to_new:
            remapped_edges.append((old_to_new[src], old_to_new[dst]))

    if remapped_edges:
        edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return node_embeddings, kept_labels, edge_index


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

    kge_artifact_root = configs.get("kge_artifact_root")
    if not kge_artifact_root:
        raise ValueError("kge_artifact_root must be set in config for cartography")

    entity_to_id_path = Path(kge_artifact_root) / configs.get(
        "kge_entity_to_id_file", "entity_to_id.json"
    )
    entity_to_id = load_entity_to_id(entity_to_id_path)

    split_path = _resolve_split_path(configs, args.split)
    sample = _load_sample(split_path, args.sample_idx)

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

    kge_config = _build_kge_config(configs)
    kge_config["use_kge"] = True
    model = Coconut(
        base_model,
        latent_id,
        start_id,
        end_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        kge_config=kge_config,
    )
    _load_weights(base_model, model, Path(args.checkpoint))
    model = model.to(device)
    model.eval()

    scheduled_stage = (
        args.scheduled_stage
        if args.scheduled_stage is not None
        else configs.get("max_latent_stage", 6)
    )
    c_thought = int(configs.get("c_thought", 1))
    max_latent_stage = int(configs.get("max_latent_stage", 6))
    n_steps = len(sample.get("steps", []))
    n_latent_tokens = min(scheduled_stage, min(max_latent_stage, n_steps)) * c_thought

    question_text = sample["question"] + "\n"
    question_tokens = tokenizer.encode(question_text, add_special_tokens=True)
    input_tokens = question_tokens + [start_id] + [latent_id] * n_latent_tokens + [end_id]
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    anchor_entity_ids, anchor_token_spans, anchor_mask = build_anchor_payload(
        sample=sample,
        tokenizer=tokenizer,
        question_text=question_text,
        question_tokenized=question_tokens,
        entity_to_id=entity_to_id,
    )
    anchor_entity_ids = torch.tensor([anchor_entity_ids], dtype=torch.long, device=device)
    anchor_token_spans = torch.tensor([anchor_token_spans], dtype=torch.long, device=device)
    anchor_mask = torch.tensor([anchor_mask], dtype=torch.bool, device=device)

    analysis_trace: Dict = {}
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            anchor_entity_ids=anchor_entity_ids,
            anchor_token_spans=anchor_token_spans,
            anchor_mask=anchor_mask,
            analysis_trace=analysis_trace,
            analysis_batch_index=0,
        )

    trajectory_list = analysis_trace.get("trajectory", [])
    if trajectory_list:
        trajectory = torch.stack(trajectory_list, dim=0)
    else:
        hidden = model.embedding.embedding_dim
        trajectory = torch.empty((0, hidden), dtype=torch.float32)

    node_embeddings, node_labels, edges_local = _project_sample_nodes(
        model=model,
        sample=sample,
        entity_to_id=entity_to_id,
        device=device,
    )

    if trajectory.shape[0] > 0:
        decoded_path, score_matrix = decode_trajectory(
            trajectory=trajectory,
            node_embeddings=node_embeddings,
            node_labels=node_labels,
            metric=args.metric,
            stability_threshold=args.stability_threshold,
        )
        decoded_indices = [entry["predicted_node_index"] for entry in decoded_path]
        metrics = compute_metrics(decoded_indices, edges_local)
    else:
        decoded_path = []
        score_matrix = torch.empty((0, node_embeddings.shape[0]))
        metrics = compute_metrics([], edges_local)

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

    diagnostics = None
    if args.diagnostics_json:
        diagnostics = build_diagnostics(
            node_embeddings=node_embeddings,
            trajectory=trajectory,
            node_labels=node_labels,
            raw_projection_meta=raw_projection_meta,
            normalized_projection_meta=norm_projection_meta,
        )

    output_dir = Path(args.output_dir) / f"{args.split}_{args.sample_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    title = (
        f"ProsQA KGE Cartography\n"
        f"split={args.split} idx={args.sample_idx} metric={args.metric} "
        f"latent_tokens={n_latent_tokens}\n"
        f"Q: {sample['question'][:140]}"
    )

    png_path = output_dir / "cartography.png"
    dual_png_path = output_dir / "cartography_dual.png"
    gif_path = output_dir / "cartography.gif" if args.save_gif else None

    plot_cartography(
        nodes_2d=raw_nodes_2d,
        path_2d=raw_path_2d,
        node_labels=node_labels,
        decoded_path=decoded_path,
        edge_index=edges_local,
        title=title,
        png_path=png_path,
        gif_path=gif_path,
        distance_metric=args.distance_metric,
    )

    torch.save(node_embeddings, output_dir / "node_embeddings.pt")
    torch.save(trajectory, output_dir / "reasoning_trajectory.pt")
    torch.save(score_matrix, output_dir / "similarity.pt")
    save_json(output_dir / "decoded_path.json", decoded_path)
    save_json(output_dir / "metrics.json", metrics)

    if args.plot_layout == "dual":
        plot_cartography_dual(
            raw_nodes_2d=raw_nodes_2d,
            raw_path_2d=raw_path_2d,
            norm_nodes_2d=norm_nodes_2d,
            norm_path_2d=norm_path_2d,
            node_labels=node_labels,
            decoded_path=decoded_path,
            edge_index=edges_local,
            title=title,
            png_path=dual_png_path,
            distance_metric=args.distance_metric,
        )
    else:
        selected_nodes = raw_nodes_2d if args.projection_mode == "raw" else norm_nodes_2d
        selected_path = raw_path_2d if args.projection_mode == "raw" else norm_path_2d
        plot_cartography(
            nodes_2d=selected_nodes,
            path_2d=selected_path,
            node_labels=node_labels,
            decoded_path=decoded_path,
            edge_index=edges_local,
            title=title,
            png_path=png_path,
            gif_path=gif_path,
            distance_metric=args.distance_metric,
        )

    if diagnostics is not None:
        save_json(output_dir / "diagnostics.json", diagnostics)

    save_json(
        output_dir / "run_config.json",
        {
            "split": args.split,
            "sample_idx": args.sample_idx,
            "scheduled_stage": int(scheduled_stage),
            "n_latent_tokens": n_latent_tokens,
            "metric": args.metric,
            "projection_mode": args.projection_mode,
            "plot_layout": args.plot_layout,
            "distance_metric": args.distance_metric,
            "max_new_tokens": args.max_new_tokens,
            "stability_threshold": args.stability_threshold,
            "generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True),
        },
    )

    print(f"Saved latent cartography outputs under: {output_dir}")


if __name__ == "__main__":
    main()
