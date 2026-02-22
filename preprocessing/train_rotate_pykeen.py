"""Train RotatE with PyKEEN and export embedding artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def to_real_embedding(tensor: torch.Tensor, expected_complex_dim: int) -> torch.Tensor:
    if torch.is_complex(tensor):
        real = torch.view_as_real(tensor)
        return real.reshape(real.shape[0], -1)

    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D embedding tensor, got shape {tuple(tensor.shape)}")

    expected = expected_complex_dim * 2
    if tensor.shape[1] != expected:
        raise ValueError(
            f"Expected RotatE exported dim {expected} for embedding_dim={expected_complex_dim}, "
            f"got {tensor.shape[1]}"
        )
    return tensor


def _read_triples(path: Path) -> np.ndarray:
    triples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            head, relation, tail = line.split("\t")
            triples.append((head, relation, tail))
    if not triples:
        return np.empty((0, 3), dtype=str)
    return np.asarray(triples, dtype=str)


def _extract_metric(metrics: Dict, key: str):
    cur = metrics
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _resolve_pykeen_version() -> str:
    try:
        return pkg_version("pykeen")
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RotatE with PyKEEN")
    parser.add_argument("--triples-dir", default="data/prosqa_rotate", help="Directory with triples_{split}.tsv")
    parser.add_argument("--output-dir", default="data/prosqa_rotate", help="Output artifact directory")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    try:
        import pykeen
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except ImportError as exc:
        raise ImportError("PyKEEN is required. Install with `pip install pykeen`.") from exc

    triples_dir = Path(args.triples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (triples_dir / "entity_to_id.json").open() as f:
        entity_to_id = {str(k): int(v) for k, v in json.load(f).items()}
    with (triples_dir / "relation_to_id.json").open() as f:
        relation_to_id = {str(k): int(v) for k, v in json.load(f).items()}

    train = _read_triples(triples_dir / "triples_train.tsv")
    valid = _read_triples(triples_dir / "triples_valid.tsv")
    test = _read_triples(triples_dir / "triples_test.tsv")

    train_tf = TriplesFactory.from_labeled_triples(
        train,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=True,
    )
    valid_tf = TriplesFactory.from_labeled_triples(
        valid,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=False,
    )
    test_tf = TriplesFactory.from_labeled_triples(
        test,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=False,
    )

    result = pipeline(
        model="RotatE",
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        model_kwargs={"embedding_dim": args.embedding_dim},
        random_seed=args.seed,
        training_kwargs={
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
        },
        optimizer_kwargs={"lr": args.learning_rate},
    )

    model = result.model
    entity = model.entity_representations[0]().detach().cpu()
    relation = model.relation_representations[0]().detach().cpu()

    entity = to_real_embedding(entity, expected_complex_dim=args.embedding_dim).float()
    relation = to_real_embedding(relation, expected_complex_dim=args.embedding_dim).float()

    torch.save(entity, output_dir / "entity_embeddings.pt")
    torch.save(relation, output_dir / "relation_embeddings.pt")

    metrics = result.metric_results.to_dict()

    metadata = {
        "model": "RotatE",
        "embedding_dim": int(args.embedding_dim),
        "projector_in_dim": int(entity.shape[1]),
        "relation_projector_in_dim": int(relation.shape[1]),
        "num_entities": int(entity.shape[0]),
        "num_relations": int(relation.shape[0]),
        "num_epochs": int(args.num_epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "create_inverse_triples": True,
        "pykeen_version": _resolve_pykeen_version(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "mrr": _extract_metric(metrics, "both.realistic.inverse_harmonic_mean_rank"),
            "hits_at_10": _extract_metric(metrics, "both.realistic.hits_at_10"),
        },
        "raw_metrics": metrics,
    }

    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"Saved RotatE artifacts under: {output_dir}")


if __name__ == "__main__":
    main()
