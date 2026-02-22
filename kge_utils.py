from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ANCHOR_FIELDS: Tuple[str, ...] = ("root", "target", "neg_target")
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")
_SUBCLASS_RE = re.compile(r"^Every\s+([A-Za-z0-9_\-]+)\s+is\s+a\s+([A-Za-z0-9_\-]+)\.?$")
_INSTANCE_RE = re.compile(r"^([A-Za-z0-9_\-]+)\s+is\s+a\s+([A-Za-z0-9_\-]+)\.?$")


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    return [segment.strip() for segment in _SENTENCE_SPLIT.split(text) if segment.strip()]


def parse_prosqa_statement(sentence: str) -> Optional[Tuple[str, str, str]]:
    sentence = sentence.strip()
    if not sentence or sentence.endswith("?"):
        return None

    match = _SUBCLASS_RE.match(sentence)
    if match:
        head, tail = match.groups()
        return head, "subclass_of", tail

    if sentence.startswith("Every "):
        return None

    match = _INSTANCE_RE.match(sentence)
    if match:
        head, tail = match.groups()
        return head, "instance_of", tail

    return None


def resolve_anchor_symbols(sample: Dict) -> List[Optional[str]]:
    symbols = sample.get("idx_to_symbol") or []
    anchors: List[Optional[str]] = []
    for field in ANCHOR_FIELDS:
        idx = sample.get(field)
        if isinstance(idx, int) and 0 <= idx < len(symbols):
            anchors.append(symbols[idx])
        else:
            anchors.append(None)
    return anchors


def load_entity_to_id(path: str | Path) -> Dict[str, int]:
    with Path(path).open() as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


def _find_subsequence(sequence: Sequence[int], target: Sequence[int]) -> Optional[Tuple[int, int]]:
    if not target or len(target) > len(sequence):
        return None
    size = len(target)
    for start in range(len(sequence) - size + 1):
        if list(sequence[start : start + size]) == list(target):
            return start, start + size
    return None


def find_symbol_token_span(
    tokenizer,
    question_text: str,
    question_tokenized: Sequence[int],
    symbol: str,
) -> Optional[Tuple[int, int]]:
    if not symbol:
        return None

    span_match = re.search(rf"(?<!\\w){re.escape(symbol)}(?!\\w)", question_text)
    is_fast = bool(getattr(tokenizer, "is_fast", False))
    if span_match and is_fast:
        enc = tokenizer(
            question_text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = enc.get("offset_mapping")
        if offsets:
            start_char, end_char = span_match.span()
            covered = [
                idx
                for idx, (tok_start, tok_end) in enumerate(offsets)
                if tok_end > tok_start and tok_end > start_char and tok_start < end_char
            ]
            if covered:
                return covered[0], covered[-1] + 1

    symbol_tokens = tokenizer.encode(symbol, add_special_tokens=False)
    return _find_subsequence(question_tokenized, symbol_tokens)


def build_anchor_payload(
    sample: Dict,
    tokenizer,
    question_text: str,
    question_tokenized: Sequence[int],
    entity_to_id: Dict[str, int],
) -> Tuple[List[int], List[List[int]], List[bool]]:
    symbols = resolve_anchor_symbols(sample)
    anchor_entity_ids: List[int] = []
    anchor_token_spans: List[List[int]] = []
    anchor_mask: List[bool] = []

    for symbol in symbols:
        if symbol is None:
            anchor_entity_ids.append(-1)
            anchor_token_spans.append([0, 0])
            anchor_mask.append(False)
            continue

        entity_id = entity_to_id.get(symbol)
        if entity_id is None:
            entity_id = entity_to_id.get(symbol.lower(), -1)

        span = find_symbol_token_span(tokenizer, question_text, question_tokenized, symbol)
        valid = entity_id is not None and entity_id >= 0 and span is not None

        anchor_entity_ids.append(int(entity_id) if entity_id is not None else -1)
        if span is None:
            anchor_token_spans.append([0, 0])
        else:
            anchor_token_spans.append([int(span[0]), int(span[1])])
        anchor_mask.append(bool(valid))

    return anchor_entity_ids, anchor_token_spans, anchor_mask
