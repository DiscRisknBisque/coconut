"""Shared ProsQA node extraction utilities for graphification and analysis."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


SENTENCE_PATTERN = re.compile(r"(?<=[.?!])\s+")


def _split_question(text: str) -> tuple[list[str], str]:
    if not text:
        return [], ""
    parts = [segment.strip() for segment in SENTENCE_PATTERN.split(text) if segment.strip()]
    if not parts:
        return [], text
    if parts[-1].endswith("?") or parts[-1].endswith("."):
        question = parts[-1]
        premises = parts[:-1]
    else:
        question = parts[-1]
        premises = parts[:-1]
    return premises, question


def extract_prosqa_nodes(item: Dict) -> Tuple[List[Tuple[str, str]], List[int]]:
    """Return graph node entries and step node indices for one ProsQA sample."""

    entries: List[Tuple[str, str]] = []
    step_indices: List[int] = []

    premises = item.get("premises") or item.get("context") or []
    question_text = item.get("question", "")

    if premises:
        for premise in premises:
            entries.append(("premise", premise))
        question_node = question_text
    else:
        premise_sentences, question_node = _split_question(question_text)
        for premise in premise_sentences:
            entries.append(("premise", premise))

    steps = item.get("steps", []) or []
    for step in steps:
        step_indices.append(len(entries))
        entries.append(("step", step))

    if question_node:
        entries.append(("question", question_node))

    return entries, step_indices

