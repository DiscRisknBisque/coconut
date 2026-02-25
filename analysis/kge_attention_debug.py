"""KGE anchor attention diagnostics for Coconut models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from coconut import Coconut
from kge_utils import ANCHOR_FIELDS


def _resolve_layer_index(attentions: Sequence[torch.Tensor], layer: str | int) -> int:
    if len(attentions) == 0:
        raise ValueError("Model returned no attention tensors.")
    if isinstance(layer, str):
        if layer != "last":
            raise ValueError("layer must be 'last' or an integer index")
        return len(attentions) - 1
    layer_idx = int(layer)
    if layer_idx < 0:
        layer_idx = len(attentions) + layer_idx
    if layer_idx < 0 or layer_idx >= len(attentions):
        raise IndexError(f"layer index {layer} out of range for {len(attentions)} attention layers")
    return layer_idx


def _decode_token(tokenizer, token_id: int) -> Optional[str]:
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return None


def _decode_span(tokenizer, token_ids: Sequence[int]) -> Optional[str]:
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode(list(map(int, token_ids)))
    except Exception:
        return None


def _slice_span_token_ids(input_ids_row: torch.Tensor, start: int, end: int) -> List[int]:
    return [int(v) for v in input_ids_row[start:end].detach().cpu().tolist()]


def _validate_required_batch_keys(batch: Dict[str, Any], required_keys: Sequence[str]):
    missing = [key for key in required_keys if key not in batch]
    if missing:
        raise KeyError(f"Batch missing required keys for KGE attention debug: {missing}")


def debug_kge_attention_for_batch(
    model: Coconut,
    batch: dict,
    tokenizer=None,
    batch_index: int = 0,
    layer: str | int = "last",
    query_position: int = -1,
    warn_threshold_pct: float = 1.0,
    healthy_threshold_pct: float = 15.0,
    top_k_tokens: int = 8,
    print_report: bool = True,
) -> dict:
    """Run a readable KGE-anchor attention diagnostic on one batch item."""

    if not getattr(model, "use_kge", False):
        raise RuntimeError("KGE attention diagnostics require Coconut.use_kge=True")

    required_keys = (
        "input_ids",
        "attention_mask",
        "labels",
        "position_ids",
        "anchor_entity_ids",
        "anchor_token_spans",
        "anchor_mask",
    )
    _validate_required_batch_keys(batch, required_keys)

    batch_size = int(batch["input_ids"].shape[0])
    if batch_index < 0 or batch_index >= batch_size:
        raise IndexError(f"batch_index={batch_index} out of range for batch size {batch_size}")

    was_training = model.training
    base_was_training = model.base_causallm.training
    model.eval()
    model.base_causallm.eval()

    with torch.no_grad():
        coconut_outputs = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
            batch["position_ids"],
            anchor_entity_ids=batch["anchor_entity_ids"],
            anchor_token_spans=batch["anchor_token_spans"],
            anchor_mask=batch["anchor_mask"],
        )

        base_outputs = model.base_causallm(
            inputs_embeds=coconut_outputs.inputs_embeds,
            attention_mask=batch["attention_mask"],
            position_ids=batch.get("position_ids"),
            output_attentions=True,
            return_dict=True,
        )

    if was_training:
        model.train()
    if base_was_training:
        model.base_causallm.train()

    attentions = getattr(base_outputs, "attentions", None)
    if not attentions:
        raise RuntimeError("Base model did not return attentions. Check model support for output_attentions.")

    layer_idx = _resolve_layer_index(attentions, layer)
    layer_attention = attentions[layer_idx]
    if layer_attention.ndim != 4:
        raise ValueError(
            f"Expected attention tensor shape [B, H, S, S], got {tuple(layer_attention.shape)}"
        )

    avg_head_attention = layer_attention.mean(dim=1)
    seq_len = int(avg_head_attention.shape[-1])
    resolved_query_pos = int(query_position if query_position >= 0 else seq_len + query_position)
    if resolved_query_pos < 0 or resolved_query_pos >= seq_len:
        raise IndexError(
            f"query_position={query_position} resolved to {resolved_query_pos}, outside [0, {seq_len})"
        )

    attention_vec = avg_head_attention[batch_index, resolved_query_pos, :].detach().float().cpu()
    total_attention = float(attention_vec.sum().item())

    input_ids_row = batch["input_ids"][batch_index].detach().cpu()
    anchor_spans = batch["anchor_token_spans"][batch_index].detach().cpu()
    anchor_mask = batch["anchor_mask"][batch_index].detach().cpu()
    anchor_entity_ids = batch["anchor_entity_ids"][batch_index].detach().cpu()

    notes: List[str] = []
    skipped_anchors: List[Dict[str, Any]] = []
    anchors: List[Dict[str, Any]] = []
    union_positions: set[int] = set()
    valid_anchor_count = 0

    for anchor_idx in range(anchor_spans.shape[0]):
        slot_name = ANCHOR_FIELDS[anchor_idx] if anchor_idx < len(ANCHOR_FIELDS) else f"anchor_{anchor_idx}"
        entity_id = int(anchor_entity_ids[anchor_idx].item())
        masked_valid = bool(anchor_mask[anchor_idx].item())
        start, end = [int(v) for v in anchor_spans[anchor_idx].tolist()]

        anchor_entry: Dict[str, Any] = {
            "anchor_slot": slot_name,
            "anchor_index": int(anchor_idx),
            "entity_id": entity_id,
            "span": [start, end],
            "is_valid": False,
            "attention_mass": 0.0,
            "attention_pct": 0.0,
        }

        if not masked_valid:
            anchors.append(anchor_entry)
            continue

        if start < 0 or end <= start or end > seq_len:
            reason = f"invalid span [{start}, {end}) for sequence length {seq_len}"
            skipped = dict(anchor_entry)
            skipped["reason"] = reason
            skipped_anchors.append(skipped)
            anchors.append(anchor_entry)
            notes.append(f"Skipped {slot_name}: {reason}")
            continue

        positions = list(range(start, end))
        for pos in positions:
            union_positions.add(pos)

        span_attention_mass = float(attention_vec[positions].sum().item())
        span_attention_pct = (span_attention_mass / max(total_attention, 1e-12)) * 100.0
        token_ids = _slice_span_token_ids(input_ids_row, start, end)

        anchor_entry.update(
            {
                "is_valid": True,
                "attention_mass": span_attention_mass,
                "attention_pct": span_attention_pct,
                "token_ids": token_ids,
            }
        )
        span_text = _decode_span(tokenizer, token_ids)
        if span_text is not None:
            anchor_entry["token_text"] = span_text

        anchors.append(anchor_entry)
        valid_anchor_count += 1

    if valid_anchor_count == 0:
        notes.append("No valid anchors found for this batch item.")

    union_positions_sorted = sorted(union_positions)
    anchor_attention_mass = (
        float(attention_vec[union_positions_sorted].sum().item()) if union_positions_sorted else 0.0
    )
    anchor_attention_pct = (anchor_attention_mass / max(total_attention, 1e-12)) * 100.0

    if anchor_attention_pct < warn_threshold_pct:
        health = "warning"
    elif anchor_attention_pct > healthy_threshold_pct:
        health = "healthy"
    else:
        health = "moderate"

    k = min(max(int(top_k_tokens), 0), seq_len)
    top_values, top_indices = torch.topk(attention_vec, k=k) if k > 0 else (torch.tensor([]), torch.tensor([]))
    top_attention_tokens: List[Dict[str, Any]] = []
    for score, pos in zip(top_values.tolist(), top_indices.tolist()):
        pos_i = int(pos)
        token_id = int(input_ids_row[pos_i].item())
        entry: Dict[str, Any] = {
            "position": pos_i,
            "attention": float(score),
            "token_id": token_id,
        }
        token_text = _decode_token(tokenizer, token_id)
        if token_text is not None:
            entry["token_text"] = token_text
        top_attention_tokens.append(entry)

    anchor_coverage = getattr(coconut_outputs, "anchor_coverage", None)
    anchor_coverage_value = float(anchor_coverage.detach().float().item()) if anchor_coverage is not None else None

    result = {
        "batch_index": int(batch_index),
        "layer_index": int(layer_idx),
        "query_position": int(resolved_query_pos),
        "sequence_length": int(seq_len),
        "total_attention": total_attention,
        "anchor_attention_mass": anchor_attention_mass,
        "anchor_attention_pct": anchor_attention_pct,
        "anchor_positions": union_positions_sorted,
        "anchor_coverage": anchor_coverage_value,
        "valid_anchor_count": int(valid_anchor_count),
        "anchors": anchors,
        "skipped_anchors": skipped_anchors,
        "top_attention_tokens": top_attention_tokens,
        "health": health,
        "notes": notes,
    }

    if print_report:
        print("\n--- KGE ATTENTION DIAGNOSTIC ---")
        print(f"Batch Item: {batch_index}")
        print(f"  Layer: {layer_idx} (requested={layer})")
        print(f"  Query Position: {resolved_query_pos}")
        print(f"  Sequence Length: {seq_len}")
        print(f"  Total Attention Sum: {total_attention:.4f}")
        if anchor_coverage_value is not None:
            print(f"  Anchor Coverage: {anchor_coverage_value:.4f}")
        print(f"  Valid Anchors: {valid_anchor_count}/{anchor_spans.shape[0]}")
        print(
            f"  Total Anchor/KGE Attention: {anchor_attention_mass:.4f} "
            f"({anchor_attention_pct:.2f}%)"
        )

        if health == "warning":
            print("  [WARNING] LLM is likely ignoring the injected KGE anchor spans.")
        elif health == "healthy":
            print("  [HEALTHY] LLM is heavily attending to the injected KGE anchor spans.")
        else:
            print("  [MODERATE] LLM is attending to KGE anchor spans at a moderate level.")

        print("  Per-Anchor Attention:")
        for anchor in anchors:
            span = anchor["span"]
            label = f"{anchor['anchor_slot']} span={span} entity_id={anchor['entity_id']}"
            if not anchor["is_valid"]:
                print(f"    - {label}: skipped/invalid")
                continue
            text_suffix = (
                f" text={anchor.get('token_text')!r}" if "token_text" in anchor else ""
            )
            print(
                f"    - {label}: {anchor['attention_mass']:.4f} "
                f"({anchor['attention_pct']:.2f}%){text_suffix}"
            )

        if top_attention_tokens:
            print(f"  Top-{len(top_attention_tokens)} Attended Tokens:")
            for item in top_attention_tokens:
                text_suffix = f" text={item.get('token_text')!r}" if "token_text" in item else ""
                print(
                    f"    - pos={item['position']} token_id={item['token_id']} "
                    f"attn={item['attention']:.4f}{text_suffix}"
                )

        for note in notes:
            print(f"  Note: {note}")

    return result
