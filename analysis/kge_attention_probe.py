"""Standalone KGE anchor attention probe for Coconut + ProsQA."""

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

from analysis.kge_attention_debug import debug_kge_attention_for_batch  # noqa: E402
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
        result = coconut_model.load_state_dict(state, strict=False)
        print(f"Loaded Coconut weights from {checkpoint_path}: {result}")
    else:
        result = base_model.load_state_dict(state, strict=False)
        print(f"Loaded base model weights from {checkpoint_path}: {result}")


def _resolve_split_path(configs: Dict, split: str) -> Path:
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
        "kge_entity_embeddings_file": configs.get("kge_entity_embeddings_file", "entity_embeddings.pt"),
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


def main():
    parser = argparse.ArgumentParser(description="KGE attention probe for Coconut")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--sample-idx", type=int, required=True)
    parser.add_argument("--scheduled-stage", type=int, default=None)
    parser.add_argument("--query-position", type=int, default=-1)
    parser.add_argument(
        "--layer",
        default="last",
        help="Attention layer index (int) or 'last' (default).",
    )
    parser.add_argument("--top-k-tokens", type=int, default=8)
    parser.add_argument(
        "--no-token-text",
        action="store_true",
        help="Skip decoding token ids into text snippets in the report.",
    )
    parser.add_argument("--json-out", default=None, help="Optional JSON file path for diagnostics.")
    args = parser.parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)

    kge_artifact_root = configs.get("kge_artifact_root")
    if not kge_artifact_root:
        raise ValueError("kge_artifact_root must be set in config for KGE attention probe")

    entity_to_id_path = Path(kge_artifact_root) / configs.get("kge_entity_to_id_file", "entity_to_id.json")
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
        args.scheduled_stage if args.scheduled_stage is not None else configs.get("max_latent_stage", 6)
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
    labels = input_ids.clone()
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)

    anchor_entity_ids, anchor_token_spans, anchor_mask = build_anchor_payload(
        sample=sample,
        tokenizer=tokenizer,
        question_text=question_text,
        question_tokenized=question_tokens,
        entity_to_id=entity_to_id,
    )
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "position_ids": position_ids,
        "anchor_entity_ids": torch.tensor([anchor_entity_ids], dtype=torch.long, device=device),
        "anchor_token_spans": torch.tensor([anchor_token_spans], dtype=torch.long, device=device),
        "anchor_mask": torch.tensor([anchor_mask], dtype=torch.bool, device=device),
    }

    layer_arg = args.layer
    if layer_arg != "last":
        try:
            layer_arg = int(layer_arg)
        except ValueError as exc:
            raise ValueError("--layer must be 'last' or an integer") from exc

    diagnostics = debug_kge_attention_for_batch(
        model=model,
        batch=batch,
        tokenizer=None if args.no_token_text else tokenizer,
        batch_index=0,
        layer=layer_arg,
        query_position=args.query_position,
        top_k_tokens=args.top_k_tokens,
        print_report=True,
    )

    diagnostics["sample"] = {
        "split": args.split,
        "sample_idx": int(args.sample_idx),
        "scheduled_stage": int(scheduled_stage),
        "n_latent_tokens": int(n_latent_tokens),
        "question": sample.get("question"),
    }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(diagnostics, f, indent=2)
        print(f"Saved diagnostics JSON to: {out_path}")


if __name__ == "__main__":
    main()
