"""Utilities for constructing graph sidecars for Coconut datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


ROLE_TO_ID = {"premise": 0, "step": 1, "question": 2}


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _simple_lemmatize(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    lowered = text.lower()
    tokens = TOKEN_PATTERN.findall(lowered)
    return [_simple_lemmatize(tok) for tok in tokens if tok not in STOPWORDS]


def hash_bow(tokens: Sequence[str], dim: int = 256, seed: int = 13) -> torch.Tensor:
    vector = torch.zeros(dim, dtype=torch.float32)
    if not tokens:
        return vector

    grams: List[str] = list(tokens)
    grams.extend([f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)])

    for gram in grams:
        digest = hashlib.md5(f"{seed}:{gram}".encode("utf-8"))
        hashed = int(digest.hexdigest(), 16)
        idx = hashed % dim
        sign = -1.0 if (hashed >> 1) & 1 else 1.0
        vector[idx] += sign

    norm = vector.norm(p=2)
    if torch.isfinite(norm) and norm > 0:
        vector /= norm
    return vector


def role_one_hot(role: str) -> torch.Tensor:
    vec = torch.zeros(len(ROLE_TO_ID), dtype=torch.float32)
    vec[ROLE_TO_ID[role]] = 1.0
    return vec


@dataclass
class NodeSpec:
    text: str
    role: str
    tokens: List[str]


def build_node_specs(entries: Iterable[Tuple[str, str]]) -> List[NodeSpec]:
    specs: List[NodeSpec] = []
    for role, text in entries:
        tokens = tokenize(text)
        specs.append(NodeSpec(text=text, role=role, tokens=tokens))
    return specs


def build_edge_index(token_sets: List[List[str]], sequential_pairs: List[Tuple[int, int]]) -> torch.Tensor:
    edges: List[Tuple[int, int]] = []
    vocab_sets = [set(tokens) for tokens in token_sets]

    for i in range(len(vocab_sets)):
        for j in range(i + 1, len(vocab_sets)):
            if vocab_sets[i] and vocab_sets[i].intersection(vocab_sets[j]):
                edges.append((i, j))
                edges.append((j, i))

    for src, dst in sequential_pairs:
        edges.append((src, dst))
        edges.append((dst, src))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return edge_index


def build_feature_matrix(nodes: List[NodeSpec], dim: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    features: List[torch.Tensor] = []
    roles: List[int] = []
    for node in nodes:
        bow = hash_bow(node.tokens, dim=dim)
        role_vec = role_one_hot(node.role)
        features.append(torch.cat([bow, role_vec], dim=0))
        roles.append(ROLE_TO_ID[node.role])
    x = torch.stack(features) if features else torch.zeros((0, dim + len(ROLE_TO_ID)), dtype=torch.float32)
    role_tensor = torch.tensor(roles, dtype=torch.long) if roles else torch.zeros((0,), dtype=torch.long)
    return x, role_tensor


def save_graph_sidecars(
    dataset: List[Dict],
    output_dir: Path,
    split: str,
    node_extractor: Callable[[Dict], Tuple[List[Tuple[str, str]], List[int]]],
    dim: int = 256,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, str] = {}

    for item in dataset:
        idx = item.get("idx")
        if idx is None:
            raise ValueError("Each dataset item must include an 'idx' field before graphification")

        entries, step_indices = node_extractor(item)
        nodes = build_node_specs(entries)
        if not nodes:
            raise ValueError(f"No graph nodes extracted for sample idx={idx}")

        sequential_pairs = list(zip(step_indices, step_indices[1:])) if len(step_indices) > 1 else []
        edge_index = build_edge_index([node.tokens for node in nodes], sequential_pairs)
        x, roles = build_feature_matrix(nodes, dim=dim)

        graph_dict = {
            "edge_index": edge_index,
            "x": x,
            "role": roles,
        }

        graph_path = output_dir / f"{idx}.pt"
        torch.save(graph_dict, graph_path)
        manifest[str(idx)] = str(graph_path)

    manifest_path = output_dir / f"manifest_{split}.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def default_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, help="Input dataset JSON path")
    parser.add_argument("--split", required=True, help="Dataset split name (e.g., train, valid, test)")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for serialized graphs (e.g., data/prontoqa_graphs/train)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=256,
        help="Dimensionality of hashed node features before role concatenation",
    )
    return parser


