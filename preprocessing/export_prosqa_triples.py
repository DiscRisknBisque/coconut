"""Export ProsQA triples for offline RotatE training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kge_utils import parse_prosqa_statement, split_sentences


def _extract_triples_from_item(item: Dict) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for sentence in split_sentences(item.get("question", "")):
        parsed = parse_prosqa_statement(sentence)
        if parsed is not None:
            triples.append(parsed)
    return triples


def _write_triples(path: Path, triples: Sequence[Tuple[str, str, str]]) -> None:
    with path.open("w") as f:
        for head, relation, tail in triples:
            f.write(f"{head}\t{relation}\t{tail}\n")


def _collect_symbols(dataset: Iterable[Dict]) -> List[str]:
    symbols = set()
    for item in dataset:
        for symbol in item.get("idx_to_symbol", []) or []:
            symbols.add(str(symbol))
    return sorted(symbols)


def _process_split(input_path: Path) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    with input_path.open() as f:
        dataset = json.load(f)

    triples: List[Tuple[str, str, str]] = []
    for item in dataset:
        triples.extend(_extract_triples_from_item(item))
    symbols = _collect_symbols(dataset)
    return triples, symbols


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ProsQA triples for RotatE")
    parser.add_argument("--input-dir", default="data", help="Directory containing prosqa_{split}.json")
    parser.add_argument("--output-dir", default="data/prosqa_rotate", help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    relation_to_id = {"instance_of": 0, "subclass_of": 1}
    all_entities = set()
    inventory = {"train": [], "valid": [], "test": []}

    for split in ("train", "valid", "test"):
        input_path = input_dir / f"prosqa_{split}.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input split: {input_path}")

        triples, symbols = _process_split(input_path)
        inventory[split] = symbols
        for head, _, tail in triples:
            all_entities.add(head)
            all_entities.add(tail)

        _write_triples(output_dir / f"triples_{split}.tsv", triples)

    for split_symbols in inventory.values():
        all_entities.update(split_symbols)

    entity_to_id = {entity: idx for idx, entity in enumerate(sorted(all_entities))}

    with (output_dir / "entity_to_id.json").open("w") as f:
        json.dump(entity_to_id, f, indent=2, sort_keys=True)
    with (output_dir / "relation_to_id.json").open("w") as f:
        json.dump(relation_to_id, f, indent=2, sort_keys=True)
    with (output_dir / "prosqa_symbol_inventory.json").open("w") as f:
        json.dump(inventory, f, indent=2, sort_keys=True)

    print(f"Wrote triples and mappings under: {output_dir}")


if __name__ == "__main__":
    main()
