"""Top-level pipeline and CLI."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import networkx as nx

from .blocks import Block, extract_blocks
from .evidence import Evidence, collect_all
from .graph import build_graph, structural_importance
from .inference import BlockPosterior, infer, repair_targets
from .report import render_console, render_heatmap_console, to_dict, write_html, write_json


@dataclass
class Analysis:
    source: str
    blocks: list[Block]
    graph: nx.DiGraph
    evidence: list[Evidence]
    posteriors: dict[str, BlockPosterior]

    @property
    def ok(self) -> bool:
        return all(bp.posterior >= 0.5 for bp in self.posteriors.values())


def analyze(
    source: str,
    tests: str | None = None,
    token_logprobs: dict[str, list[float]] | None = None,
    critic_backend: str | None = None,
) -> Analysis:
    """Run the full PCG pipeline over one Python source string."""
    blocks = extract_blocks(source)
    if not blocks:
        raise ValueError("no analysable blocks found in source")
    g = build_graph(blocks)
    ev = collect_all(
        source,
        blocks,
        tests=tests,
        token_logprobs=token_logprobs,
        critic_backend=critic_backend,
    )
    imp = structural_importance(g, blocks)
    post = infer(blocks, g, ev, imp)
    return Analysis(source, blocks, g, ev, post)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pcg",
        description="Probabilistic Constraint Graphs for uncertainty-aware "
        "code verification.",
    )
    ap.add_argument("file", help="Python file to analyse")
    ap.add_argument("-t", "--tests", help="pytest file to execute against it")
    ap.add_argument(
        "--critic",
        choices=["mock", "anthropic"],
        default=None,
        help="evaluate blocks with an LLM code reviewer critic",
    )
    ap.add_argument(
        "--threshold", type=float, default=0.5, help="abstention threshold"
    )
    ap.add_argument("--json", help="write machine-readable report here")
    ap.add_argument("--html", help="write HTML heat map here")
    ap.add_argument("--heatmap", action="store_true", help="print source heat map")
    args = ap.parse_args(argv)

    with open(args.file, encoding="utf-8") as fh:
        source = fh.read()
    tests = None
    if args.tests:
        with open(args.tests, encoding="utf-8") as fh:
            tests = fh.read()

    a = analyze(source, tests=tests, critic_backend=args.critic)
    render_console(a.blocks, a.posteriors, a.evidence, a.graph, args.threshold)
    if args.heatmap:
        render_heatmap_console(source, a.blocks, a.posteriors)

    data = to_dict(a.blocks, a.posteriors, a.evidence, args.threshold)
    if args.json:
        write_json(args.json, data)
        print(f"\nJSON report -> {args.json}")
    if args.html:
        write_html(args.html, source, a.blocks, a.posteriors, data)
        print(f"HTML heat map -> {args.html}")

    return 1 if repair_targets(a.posteriors, args.threshold) else 0


if __name__ == "__main__":
    sys.exit(main())
