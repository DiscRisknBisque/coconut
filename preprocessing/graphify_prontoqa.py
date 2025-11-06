"""Build lexical subgraph sidecars for ProntoQA datasets."""

from __future__ import annotations

import json
from pathlib import Path

from graph_utils import default_arg_parser, save_graph_sidecars


def _prontoqa_nodes(item):
    entries = []
    step_indices = []

    for key in ("context", "premises", "facts"):
        context_list = item.get(key, []) or []
        for sentence in context_list:
            entries.append(("premise", sentence))

    steps = item.get("steps", []) or []
    for step in steps:
        step_indices.append(len(entries))
        entries.append(("step", step))

    question = item.get("question")
    if question:
        entries.append(("question", question))

    return entries, step_indices


def main():
    parser = default_arg_parser("Graphify ProntoQA examples with lexical overlap edges")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    with input_path.open() as f:
        data = json.load(f)

    dataset = [{**item, "idx": idx} for idx, item in enumerate(data)]
    save_graph_sidecars(dataset, output_dir, args.split, node_extractor=_prontoqa_nodes, dim=args.dim)


if __name__ == "__main__":
    main()

