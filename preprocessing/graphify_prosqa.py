"""Build lexical subgraph sidecars for ProsQA datasets."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .graph_utils import default_arg_parser, save_graph_sidecars
    from .prosqa_nodes import extract_prosqa_nodes
except ImportError:
    from graph_utils import default_arg_parser, save_graph_sidecars
    from prosqa_nodes import extract_prosqa_nodes


def main():
    parser = default_arg_parser("Graphify ProsQA examples with lexical overlap edges")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    with input_path.open() as f:
        data = json.load(f)

    dataset = [{**item, "idx": idx} for idx, item in enumerate(data)]
    save_graph_sidecars(
        dataset,
        output_dir,
        args.split,
        node_extractor=extract_prosqa_nodes,
        dim=args.dim,
    )


if __name__ == "__main__":
    main()
