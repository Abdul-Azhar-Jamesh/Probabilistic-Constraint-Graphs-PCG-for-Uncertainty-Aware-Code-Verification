"""Top-level pipeline and CLI."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

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
    test_plan: dict | None = None
    failure_diagnoses: list[dict] = field(default_factory=list)
    generated_tests: str = ""
    failure_validation: dict = field(default_factory=dict)

    def to_dict(self, threshold: float | None = None) -> dict:
        data = to_dict(
            self.blocks,
            self.posteriors,
            self.evidence,
            self.threshold if threshold is None else threshold,
        )
        data["inference"] = self.graph.graph["inference"]
        data["execution"] = self.execution.to_dict() if self.execution else None
        oracle_runs = (
            [
                t
                for t in self.execution.tests
                if t.oracle != "probe"
                and t.outcome in {"passed", "failed"}
                and t.nodeid
                not in self.failure_validation.get("excluded_from_inference", [])
            ]
            if self.execution and self.execution.valid
            else []
        )
        data["automatic_validation"] = {
            "status": "oracle-failure"
            if any(t.outcome == "failed" for t in oracle_runs)
            else "bounded-oracle-checks"
            if oracle_runs
            else "no-correctness-oracle",
            "oracle_checks": len(oracle_runs),
            "interpretation": "Passing bounded checks is not proof of intended behavior. Unspecified behavior, unsupported interfaces, and dependencies remain unresolved.",
        }
        data["test_plan"] = self.test_plan
        if self.graph.graph.get("test_targets") is not None:
            from .graph_testing import measure_targets

            data["graph_testing"] = measure_targets(
                self.graph.graph["test_targets"], self.execution
            )
            data["graph_testing"]["feedback"] = self.graph.graph.get(
                "testing_feedback", {}
            )
            data["control_flow_graph"] = self.graph.graph["test_targets"].to_dict()
        data["failure_diagnoses"] = self.failure_diagnoses
        data["failure_validation"] = self.failure_validation
        from .findings import summarize_findings

        data["findings"] = summarize_findings(
            self.execution, self.evidence, self.failure_validation,
            calibrated=data["inference"]["calibrated"],
        )
        if self.execution and not self.execution.valid:
            data["automatic_validation"]["status"] = "incomplete"
        from .project import trace_diagnoses

        data["executed_diagnoses"] = trace_diagnoses(self.execution)
        data["graph_edges"] = [
            {"from": a, "to": b, **values}
            for a, b, values in self.graph.edges(data=True)
        ]
        dependence = self.graph.graph.get("dependence")
        if dependence is not None:
            data["dependence"] = {
                "statement_nodes": dependence.graph.number_of_nodes(),
                "statement_edges": dependence.graph.number_of_edges(),
                "unresolved_calls": dependence.graph.graph.get("unresolved_calls", []),
            }
            covered = (
                {ln for t in self.execution.tests for ln in t.lines}
                if self.execution and self.execution.valid
                else set()
            )
            checked = (
                {
                    ln
                    for t in self.execution.tests
                    if t.oracle != "probe" and t.outcome in {"passed", "failed"}
                    for ln in t.lines
                }
                if self.execution and self.execution.valid
                else set()
            )
            data["coverage_gaps"] = [
                {
                    "line": ln,
                    "scope": values["scope"],
                    "status": "oracle-covered"
                    if set(values["lines"]) & checked
                    else "probe-only"
                    if set(values["lines"]) & covered
                    else "unexecuted",
                }
                for ln, values in dependence.graph.nodes(data=True)
            ]
        return data

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
    auto_tests: bool = False,
    contracts: list[dict] | None = None,
    annotation_contracts: bool = False,
    max_generated_cases: int = 40,
    autonomous_testing: bool = False,
    graph_guided: bool = True,
    failure_replays: int = 0,
    minimize_trials: int = 6,
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
    syntax_error = any(b.bid == "module:syntax-error" for b in blocks)
    g = build_graph(blocks, source=None if syntax_error else source)
    plan = None
    original_tests = tests
    if (
        auto_tests
        or autonomous_testing
        or annotation_contracts
        or contracts is not None
    ) and not syntax_error:
        from .testgen import plan_tests

        plan = plan_tests(
            source,
            blocks,
            g,
            contracts=contracts,
            max_cases=max_generated_cases,
            seed=seed,
            annotation_contracts=annotation_contracts,
            autonomous=autonomous_testing,
            graph_guided=graph_guided,
        )
        if tests and "def test_pcg_" in tests:
            raise ValueError("user tests use reserved generated-test names test_pcg_")
        tests = (tests or "") + "\n" + plan.source
    execution_result = (
        run_tests(source, tests, execution)
        if tests and not any(b.bid == "module:syntax-error" for b in blocks)
        else None
    )
    if plan is not None and g.graph.get("test_targets") is not None:
        from .graph_testing import add_uncovered_checks, measure_targets

        feedback = add_uncovered_checks(
            plan,
            g.graph["test_targets"],
            execution_result,
            source,
            max_cases=max_generated_cases,
        )
        if feedback["added_checks"]:
            tests = (original_tests or "") + "\n" + plan.source
            execution_result = run_tests(source, tests, execution)
        feedback["edges_after"] = measure_targets(
            g.graph["test_targets"], execution_result
        )["distinct_branch_edges_seen"]
        g.graph["testing_feedback"] = feedback
    ev = collect_all(
        source,
        blocks,
        tests=tests,
        token_logprobs=token_logprobs,
        critic_backend=critic_backend,
        execution=execution,
        execution_result=execution_result,
    )
    from .failure_validation import validate_failures

    failure_validation = validate_failures(
        {
            "candidate.py": source,
            "test_candidate.py": tests or "",
            **(execution.files if execution else {}),
        },
        ["candidate.py"],
        execution_result or ExecutionResult(status="no_tests"),
        execution or ExecutionConfig(),
        replays=failure_replays,
        minimize_trials=minimize_trials,
    )
    excluded = set(failure_validation["excluded_from_inference"])
    for item in ev:
        observation = item.meta.get("test_observation")
        if observation and observation["name"] in excluded:
            item.meta.pop("test_observation")
            item.meta["execution_status"] = "unstable"
            item.kind, item.polarity, item.weight = "unstable_execution", "neutral", 0.0
    if plan is not None:
        generated = {c["name"]: c for c in plan.cases}
        for item in ev:
            observation = item.meta.get("test_observation")
            if not observation:
                continue
            case = generated.get(observation["name"].rsplit("::", 1)[-1])
            if case and case["oracle"] == "annotation":
                observation["oracle_family"] = (
                    f"generated:annotation:{case['function']}"
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
    from .slicing import failure_slices

    if plan is not None:
        from .testgen import recommend_checks

        plan.recommendations = recommend_checks(
            plan, blocks, g, execution_result, model or SensorModel()
        )

    diagnoses = (
        failure_slices(g.graph["dependence"], execution_result, blocks)
        if "dependence" in g.graph
        else []
    )
    return Analysis(
        source,
        blocks,
        g,
        ev,
        post,
        execution_result,
        model.posterior_threshold if model else 0.5,
        plan.to_dict() if plan else None,
        diagnoses,
        plan.source if plan else "",
        failure_validation,
    )


def main(argv: list[str] | None = None) -> int:
    import sys
    from pathlib import Path

    arguments = sys.argv[1:] if argv is None else argv
    if arguments and Path(arguments[0]).is_dir():
        from .project import main as project_main

        return project_main(arguments)
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
    ap.add_argument(
        "--auto-tests",
        action="store_true",
        help="plan and optionally execute graph-prioritized boundary probes",
    )
    ap.add_argument(
        "--contracts",
        help="JSON cases: expected outputs, expected exceptions or metamorphic relations",
    )
    ap.add_argument(
        "--annotation-contracts",
        action="store_true",
        help="treat supported declared return annotations as test contracts",
    )
    ap.add_argument("--max-generated-cases", type=int, default=40)
    ap.add_argument(
        "--autonomous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="automatically generate shrinking fuzz checks and discover existing doctest/type contracts",
    )
    ap.add_argument(
        "--graph-guided",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use dependency-sliced CFG constraints and coverage feedback for automatic tests",
    )
    ap.add_argument(
        "--write-generated-tests", help="save the generated pytest suite for inspection"
    )
    ap.add_argument("--model", help="versioned observation likelihood model JSON")
    ap.add_argument("--failure-replays", type=int, default=2)
    ap.add_argument("--minimize-trials", type=int, default=6)
    ap.add_argument("--inference", choices=["auto", "exact", "gibbs"], default="auto")
    ap.add_argument(
        "--next-tests", help="JSON candidates with name, block IDs, cost_seconds"
    )
    ap.add_argument(
        "--mutation-audit",
        type=int,
        default=None,
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
    import json

    contracts = None
    if args.contracts:
        with open(args.contracts, encoding="utf-8") as fh:
            contracts = json.load(fh)
    a = analyze(
        source,
        tests=tests,
        critic_backend=args.critic,
        execution=ExecutionConfig(args.execution, args.timeout),
        model=model,
        inference_method=args.inference,
        auto_tests=args.auto_tests or args.annotation_contracts,
        contracts=contracts,
        annotation_contracts=args.annotation_contracts,
        max_generated_cases=args.max_generated_cases,
        autonomous_testing=args.autonomous,
        graph_guided=args.graph_guided,
        failure_replays=args.failure_replays,
        minimize_trials=args.minimize_trials,
    )
    threshold = args.threshold if args.threshold is not None else a.threshold
    if not 0 < threshold < 1:
        ap.error("threshold must be in (0, 1)")
    render_console(a.blocks, a.posteriors, a.evidence, a.graph, threshold)
    if args.heatmap:
        render_heatmap_console(source, a.blocks, a.posteriors)

    data = a.to_dict(threshold)
    if args.write_generated_tests:
        from pathlib import Path

        destination = Path(args.write_generated_tests)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(a.generated_tests, encoding="utf-8")
    if a.test_plan:
        print(
            f"Generated {len(a.test_plan['cases'])} cases; {len(a.test_plan['skipped'])} interfaces/cases need explicit input contracts."
        )
    for row in a.failure_diagnoses:
        print(
            f"{row['test']}: symptom lines {row['symptom_lines']}; possible earlier causes {row['candidate_cause_lines']} ({row['oracle']})"
        )
    if args.next_tests:
        import json
        from .probabilistic import rank_next_tests

        with open(args.next_tests, encoding="utf-8") as fh:
            data["next_tests"] = rank_next_tests(
                a.graph.graph["defect_posterior"], json.load(fh)
            )
        print(data["next_tests"])
    mutation_limit = (
        args.mutation_audit
        if args.mutation_audit is not None
        else (3 if args.autonomous and args.execution != "disabled" else 0)
    )
    if mutation_limit:
        effective_tests = (tests or "") + "\n" + a.generated_tests
        if (
            not effective_tests.strip() or args.execution == "disabled"
        ) and args.mutation_audit is not None:
            ap.error("mutation auditing requires tests and enabled execution")
        from .test_quality import mutation_audit

        eligible = (
            a.execution
            and a.execution.valid
            and any(
                t.oracle != "probe" and t.outcome == "passed" for t in a.execution.tests
            )
            and not any(
                t.oracle != "probe" and t.outcome == "failed" for t in a.execution.tests
            )
        )
        if eligible:
            data["mutation_audit"] = mutation_audit(
                source,
                effective_tests,
                max_mutants=mutation_limit,
                execution=ExecutionConfig(args.execution, args.timeout),
                strengthen_generated=args.autonomous,
            )
        else:
            data["mutation_audit"] = {
                "status": "not-applicable",
                "reason": "requires a complete passing oracle-backed baseline; probes alone cannot measure test strength",
            }
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
