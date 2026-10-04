"""Top-level pipeline and CLI."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import networkx as nx

from .blocks import Block, extract_blocks
from .evidence import Evidence, collect_all
from .execution import ExecutionConfig, ExecutionResult, run_tests
from .sensors import SensorModel
from .graph import build_graph, structural_importance
from .inference import BlockPosterior, infer
from .report import (
    render_console,
    render_heatmap_console,
    to_dict,
    write_html,
    write_json,
)


@dataclass
class Analysis:
    source: str
    blocks: list[Block]
    graph: nx.DiGraph
    evidence: list[Evidence]
    posteriors: dict[str, BlockPosterior]
    execution: ExecutionResult | None = None
    threshold: float = 0.5

    @property
    def ok(self) -> bool:
        return (
            bool(self.posteriors)
            and (self.execution is None or self.execution.valid)
            and all(bp.posterior >= self.threshold for bp in self.posteriors.values())
        )


def analyze(
    source: str,
    tests: str | None = None,
    token_logprobs: dict[str, list[float]] | None = None,
    critic_backend: str | None = None,
    *,
    execution: ExecutionConfig | None = None,
    model: SensorModel | None = None,
    reliabilities: dict[str, float] | None = None,
    inference_method: str = "auto",
    seed: int = 7,
) -> Analysis:
    """Run the full PCG pipeline over one Python source string."""
    try:
        blocks = extract_blocks(source)
    except SyntaxError:
        # Preserve a reportable block when the AST cannot be built, so the
        # compiler sensor can still communicate that the whole module is bad.
        blocks = [
            Block(
                bid="module:syntax-error",
                kind="module",
                name="<syntax error>",
                qualname="<syntax error>",
                lineno=1,
                end_lineno=max(1, source.count("\n") + 1),
                source=source,
            )
        ]
    if not blocks:
        raise ValueError("no analysable blocks found in source")
    if len(blocks) > 512:
        raise ValueError("analysis is limited to 512 blocks per candidate module")
    g = build_graph(blocks)
    execution_result = (
        run_tests(source, tests, execution)
        if tests and not any(b.bid == "module:syntax-error" for b in blocks)
        else None
    )
    ev = collect_all(
        source,
        blocks,
        tests=tests,
        token_logprobs=token_logprobs,
        critic_backend=critic_backend,
        execution=execution,
        execution_result=execution_result,
    )
    imp = structural_importance(g, blocks)
    post = infer(
        blocks,
        g,
        ev,
        imp,
        model=model,
        reliabilities=reliabilities,
        method=inference_method,
        seed=seed,
    )
    return Analysis(
        source,
        blocks,
        g,
        ev,
        post,
        execution_result,
        model.posterior_threshold if model else 0.5,
    )


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
        "--threshold", type=float, default=None, help="override model trust threshold"
    )
    ap.add_argument(
        "--execution",
        choices=["local", "docker", "disabled"],
        default="disabled",
        help="local executes trusted code; docker isolates uploaded code",
    )
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--model", help="versioned observation likelihood model JSON")
    ap.add_argument("--inference", choices=["auto", "exact", "gibbs"], default="auto")
    ap.add_argument(
        "--next-tests", help="JSON candidates with name, block IDs, cost_seconds"
    )
    ap.add_argument(
        "--mutation-audit",
        type=int,
        default=0,
        metavar="LIMIT",
        help="audit a passing test suite with up to LIMIT mutations",
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

    model = SensorModel.load(args.model) if args.model else None
    a = analyze(
        source,
        tests=tests,
        critic_backend=args.critic,
        execution=ExecutionConfig(args.execution, args.timeout),
        model=model,
        inference_method=args.inference,
    )
    threshold = args.threshold if args.threshold is not None else a.threshold
    if not 0 < threshold < 1:
        ap.error("threshold must be in (0, 1)")
    render_console(a.blocks, a.posteriors, a.evidence, a.graph, threshold)
    if args.heatmap:
        render_heatmap_console(source, a.blocks, a.posteriors)

    data = to_dict(a.blocks, a.posteriors, a.evidence, threshold)
    data["inference"] = a.graph.graph["inference"]
    data["execution"] = a.execution.to_dict() if a.execution else None
    if args.next_tests:
        import json
        from .probabilistic import rank_next_tests

        with open(args.next_tests, encoding="utf-8") as fh:
            data["next_tests"] = rank_next_tests(
                a.graph.graph["defect_posterior"], json.load(fh)
            )
        print(data["next_tests"])
    if args.mutation_audit:
        if not tests or args.execution == "disabled":
            ap.error("mutation auditing requires tests and enabled execution")
        from .test_quality import mutation_audit

        data["mutation_audit"] = mutation_audit(
            source,
            tests,
            max_mutants=args.mutation_audit,
            execution=ExecutionConfig(args.execution, args.timeout),
        )
        print(data["mutation_audit"])
    if args.json:
        write_json(args.json, data)
        print(f"\nJSON report -> {args.json}")
    if args.html:
        write_html(args.html, source, a.blocks, a.posteriors, data)
        print(f"HTML heat map -> {args.html}")

    if a.execution is not None and not a.execution.valid:
        return 2
    return 1 if any(bp.posterior < threshold for bp in a.posteriors.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
