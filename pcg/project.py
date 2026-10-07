"""Project discovery, graph-guided execution, diagnosis and JSON/HTML reports."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
import io
import zipfile
import time

from .blocks import extract_blocks
from .evidence import Evidence, collect_compile, _ast_smells
from .execution import ExecutionConfig, ExecutionResult, TestOutcome, run_project_tests
from .graph import build_graph, structural_importance
from .inference import infer
from .project_graph import ProjectGraph, build_project_graph
from .sensors import SensorModel
from .sensors import test_sensitivity
from .testgen import plan_tests


IGNORED = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    "out",
    "build",
    "dist",
    "node_modules",
    ".pytest-tmp",
}
SUFFIXES = {
    ".py",
    ".pyi",
    ".toml",
    ".ini",
    ".cfg",
    ".txt",
    ".json",
    ".csv",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".sql",
}


@dataclass
class Snapshot:
    files: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    tests: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    import_roots: list[str] = field(default_factory=lambda: [".", "src"])


def unpack_project(payload: bytes, destination: Path) -> Path:
    """Validate all ZIP members before writing a bounded project upload."""
    if len(payload) > 32_000_000:
        raise ValueError("project upload exceeds byte budget")
    root = destination.resolve()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if len(members) > 2000 or sum(m.file_size for m in members) > 32_000_000:
            raise ValueError("project archive exceeds file/byte budget")
        seen = set()
        for member in members:
            path = Path(member.filename)
            target = (root / path).resolve()
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or ":" in member.filename
                or not target.is_relative_to(root)
                or member.file_size > 2_000_000
                or member.filename in seen
                or member.external_attr >> 16 & 0o170000 == 0o120000
            ):
                raise ValueError("invalid project archive member")
            seen.add(member.filename)
        for member in members:
            if member.is_dir():
                continue
            target = root / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    children = list(root.iterdir())
    return children[0] if len(children) == 1 and children[0].is_dir() else root


def discover(root: str | Path) -> Snapshot:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("project path must be a directory")
    result = Snapshot()
    total = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            any(
                part in IGNORED or part.startswith(".test-tmp")
                for part in relative.parts
            )
            or not path.is_file()
        ):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            result.skipped.append(
                {"file": relative.as_posix(), "reason": "symlink/outside project"}
            )
            continue
        if path.suffix not in SUFFIXES or path.name.startswith(".env"):
            continue
        name = relative.as_posix()
        if path.name.startswith("pcg_worker_") or path.name.startswith("_pcg_test_"):
            raise ValueError(
                "project contains reserved worker/generated-test filenames"
            )
        if path.stat().st_size > 2_000_000 or len(result.files) >= 1000:
            raise ValueError("project file/count budget exceeded")
        try:
            source = path.read_text(encoding="utf-8-sig")
        except UnicodeError:
            result.skipped.append({"file": name, "reason": "non-UTF8 resource"})
            continue
        total += len(source.encode())
        if total > 32_000_000:
            raise ValueError("project snapshot byte budget exceeded")
        result.files[name] = source
        is_test = path.stem.startswith("test") or path.stem.endswith("_test")
        if path.suffix == ".py" and is_test:
            result.tests.append(name)
        elif (
            path.suffix == ".py"
            and path.name != "conftest.py"
            and not any(p in {"tests", "test"} for p in relative.parts)
        ):
            result.sources[name] = source
    if not result.sources:
        raise ValueError("project contains no analysable Python source")
    return result


def static_checks(snapshot: Snapshot) -> dict:
    """Use isolated Ruff bug rules without importing any candidate."""
    with tempfile.TemporaryDirectory(prefix="pcg_static_") as folder:
        root = Path(folder)
        for name, source in snapshot.sources.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "--isolated",
                    "--output-format=json",
                    "--select",
                    "E9,F63,F7,F82,B",
                    *snapshot.sources,
                ],
                cwd=root,
                capture_output=True,
                timeout=15,
            )
            if proc.returncode not in (0, 1):
                return {
                    "status": "unavailable",
                    "details": proc.stderr.decode(errors="replace")[-1000:],
                    "findings": [],
                }
            rows = json.loads(proc.stdout)
            for row in rows:
                row["filename"] = Path(row["filename"]).relative_to(root).as_posix()
            return {"status": "complete", "findings": rows}
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return {"status": "unavailable", "details": str(exc), "findings": []}


def trace_diagnoses(execution: ExecutionResult | None) -> list[dict]:
    rows: list[dict] = []
    if not execution or not execution.valid:
        return rows
    for test in execution.tests:
        if test.outcome != "failed":
            continue
        for trace in test.traces:
            events = {e["id"]: e for e in trace["events"]}
            failures = [
                e
                for e in events.values()
                if e["kind"] == "exception" and e["root"] == trace["root"]
            ]
            terminal = (
                failures[-1:]
                or [
                    e
                    for e in events.values()
                    if e["kind"] == "return" and e["root"] == trace["root"]
                ][-1:]
            )
            if not terminal:
                continue
            reached, stack = set(), [e["id"] for e in terminal]
            while stack:
                eid = stack.pop()
                if eid in reached or eid not in events:
                    continue
                reached.add(eid)
                stack.extend(events[eid]["dependencies"])
            locations = sorted(
                {
                    (events[e]["file"], events[e]["line"])
                    for e in reached
                    if events[e]["kind"] not in {"enter", "exit"}
                }
            )
            rows.append(
                {
                    "test": test.nodeid,
                    "oracle": test.oracle,
                    "function": trace["function"],
                    "inputs": trace["inputs"],
                    "symptoms": [
                        {"file": e["file"], "line": e["line"], "kind": e["kind"]}
                        for e in terminal
                    ],
                    "candidate_causes": [
                        {"file": f, "line": ln} for f, ln in locations
                    ],
                    "trace_events": [events[e] for e in sorted(reached)],
                    "truncated": trace["truncated"],
                    "interpretation": "executed dependency slice of a retained invocation; possible causes, not a proof of the unique defect",
                }
            )
    return rows


def project_evidence(
    snapshot: Snapshot,
    bundle: ProjectGraph,
    execution: ExecutionResult,
    static: dict,
    generated: dict,
    excluded_tests: set[str] | None = None,
) -> list[Evidence]:
    evidence = []
    for file, blocks in bundle.by_file.items():
        evidence.extend(collect_compile(snapshot.sources[file], blocks))
        try:
            evidence.extend(_ast_smells(snapshot.sources[file], blocks))
        except SyntaxError:
            pass
    for row in static["findings"]:
        bid = bundle.owners.get((row["filename"], row["location"]["row"]))
        if bid:
            evidence.append(
                Evidence(
                    bid,
                    "static",
                    row["code"],
                    "negative",
                    0.6,
                    row["message"],
                    {"file": row["filename"], "line": row["location"]["row"]},
                )
            )
    if not execution.valid:
        return evidence
    for test in execution.tests:
        if test.outcome not in {"passed", "failed"}:
            continue
        covered = {
            bundle.owners[(file, ln)]
            for file, lines in test.file_lines.items()
            for ln in lines
            if (file, ln) in bundle.owners
        }
        block_map = {b.bid: b for b in bundle.blocks}
        covered = {
            bid
            for bid in covered
            if not block_map[bid].body_lineno
            or any(
                ln >= block_map[bid].body_lineno
                and bundle.owners.get((file, ln)) == bid
                for file, lines in test.file_lines.items()
                for ln in lines
            )
        }
        if not covered:
            continue
        if test.nodeid in (excluded_tests or set()):
            for bid in covered:
                evidence.append(
                    Evidence(
                        bid,
                        "exec",
                        "unstable_execution",
                        "neutral",
                        0.0,
                        test.nodeid,
                        {
                            "execution_status": "unstable",
                            "reason": "failure replay was flaky or inconclusive",
                        },
                    )
                )
            continue
        observation = {
            "name": test.nodeid,
            "blocks": sorted(covered),
            "failed": test.outcome == "failed",
            "oracle_scope": test.oracle_scope,
        }
        case = generated.get(test.nodeid.split("::")[0], {}).get(
            test.nodeid.rsplit("::", 1)[-1]
        )
        if case and test.oracle == "annotation":
            observation["oracle_family"] = (
                f"generated:annotation:{test.nodeid.split('::')[0]}:{case['function']}"
            )
        for bid in covered:
            evidence.append(
                Evidence(
                    bid,
                    "exec",
                    "probe_exception"
                    if test.oracle == "probe" and test.outcome == "failed"
                    else "probe_executed"
                    if test.oracle == "probe"
                    else "test_fail"
                    if observation["failed"]
                    else "test_pass",
                    "neutral"
                    if test.oracle == "probe"
                    else "negative"
                    if observation["failed"]
                    else "positive",
                    0.0 if test.oracle == "probe" else 1.0,
                    test.nodeid,
                    {
                        "execution_status": "probed"
                        if test.oracle == "probe"
                        else "tested",
                        **(
                            {"test_observation": observation}
                            if test.oracle != "probe"
                            else {}
                        ),
                    },
                )
            )
    return evidence


def analyze_project(
    root: str | Path,
    *,
    execution: ExecutionConfig | None = None,
    max_generated_cases: int = 80,
    graph_guided: bool = True,
    model: SensorModel | None = None,
    test_paths: list[str] | None = None,
    cache_dir: str | None = None,
    failure_replays: int = 2,
    minimize_trials: int = 6,
    validation_time_budget: float = 60,
) -> dict:
    if not 1 <= max_generated_cases <= 500:
        raise ValueError("project generated case budget must be 1..500")
    snapshot = discover(root)
    bundle = build_project_graph(snapshot.sources)
    if len(bundle.blocks) > 2048:
        raise ValueError("project graph is limited to 2048 blocks")
    execution = execution or ExecutionConfig("disabled")
    plans, generated = {}, {}
    planners = {}
    targets = snapshot.tests if test_paths is None else test_paths
    targets = list(targets)
    budget = max_generated_cases
    modules = [
        (file, source)
        for file, source in snapshot.sources.items()
        if bundle.modules[file]
    ]
    reserve = min(12, max_generated_cases // 4) if graph_guided else 0
    per_module = max(
        1, min(40, (max_generated_cases - reserve) // max(1, len(modules)))
    )
    for file, source in modules:
        if budget <= reserve:
            snapshot.skipped.append(
                {
                    "file": file,
                    "reason": "initial test allocation exhausted; reserved cases belong to coverage feedback",
                }
            )
            continue
        try:
            blocks = extract_blocks(source)
            graph = build_graph(blocks, source=source)
            plan = plan_tests(
                source,
                blocks,
                graph,
                autonomous=True,
                graph_guided=graph_guided,
                max_cases=min(per_module, budget - reserve),
            )
        except (SyntaxError, ValueError) as exc:
            snapshot.skipped.append({"file": file, "reason": str(exc)})
            continue
        plans[file] = plan.to_dict()
        planners[file] = (plan, graph, source)
        budget -= len(plan.cases)
        if not plan.source:
            continue
        tree = ast.parse(plan.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "candidate":
                node.module = bundle.modules[file]
        test_file = (
            "_pcg_test_" + hashlib.sha256(file.encode()).hexdigest()[:12] + ".py"
        )
        snapshot.files[test_file] = ast.unparse(tree)
        targets.append(test_file)
        generated[test_file] = {c["name"]: c for c in plan.cases}
    started = time.monotonic()
    result, cache_hit = _execute_cached(snapshot, targets, execution, cache_dir)
    initial_seconds = time.monotonic() - started
    static = static_checks(snapshot)
    scheduling = []
    feedback = {}
    if result.valid and graph_guided:
        from .graph_testing import add_uncovered_checks, measure_targets
        from .probabilistic import rank_next_tests

        infer(
            bundle.blocks,
            bundle.graph,
            project_evidence(snapshot, bundle, result, static, generated),
            structural_importance(bundle.graph, bundle.blocks),
            model=model,
        )
        candidates = []
        metrics = {}
        for file, (plan, graph, _) in planners.items():
            owners = {b.bid for b in bundle.by_file[file]}
            if not owners:
                continue
            for bid in list(owners):
                import networkx as nx

                owners.update(nx.ancestors(bundle.graph, bid))
            exercised = [t for t in result.tests if t.file_lines.get(file)]
            cost = max(
                0.05,
                sum(t.duration for t in exercised) / max(1, len(exercised))
                + initial_seconds / max(1, len(result.tests)),
            )
            checked_lines = {ln for t in exercised for ln in t.file_lines.get(file, [])}
            target_graph = graph.graph.get("test_targets")
            predicted_edges = (
                {
                    tuple(edge)
                    for t in target_graph.targets
                    for edge in t["branch_edges"]
                }
                if target_graph
                else set()
            )
            covered_edges = {
                tuple(edge) for t in exercised for edge in t.file_arcs.get(file, [])
            }
            unseen = len(predicted_edges - covered_edges)
            has_oracle = any(c["oracle"] != "probe" for c in plan.cases)
            candidates.append(
                {
                    "name": file,
                    "blocks": sorted(owners),
                    "cost_seconds": cost,
                    "sensitivity": max(
                        (
                            test_sensitivity(
                                model or SensorModel(),
                                c.get("oracle_scope", c["oracle"]),
                            )
                            for c in plan.cases
                            if c["oracle"] != "probe"
                        ),
                        default=0.05,
                    ),
                }
            )
            metrics[file] = {
                "unseen_branch_edges": unseen,
                "covered_lines": len(checked_lines),
                "has_oracle": has_oracle,
            }
        assumed_model = model or SensorModel()
        ranked = rank_next_tests(
            bundle.graph.graph["defect_posterior"],
            candidates,
            sensitivity=assumed_model.likelihoods["exec"]["flag_given_defect"],
            leak=assumed_model.likelihoods["exec"]["flag_given_clean"],
        )
        for row in ranked:
            row.update(metrics[row["name"]])
            if not row["has_oracle"]:
                row["information_gain_bits"] = row["gain_per_second"] = 0.0
            row["priority"] = (
                row["gain_per_second"] * (1 + row["unseen_branch_edges"])
                + 0.001 * row["unseen_branch_edges"]
            )
            row["coverage_basis"] = (
                "static module/ancestor forecast; exact coverage is unknown"
            )
            row["cost_basis"] = (
                "observed test duration plus shared initial execution overhead; cache hits distort overhead"
            )
        scheduling = sorted(ranked, key=lambda row: (-row["priority"], row["name"]))

        added = False
        for allocation in scheduling:
            file = allocation["name"]
            plan, graph, source = planners[file]
            if "test_targets" not in graph.graph:
                continue
            local = replace(
                result,
                tests=[
                    replace(
                        t,
                        lines=t.file_lines.get(file, []),
                        arcs=t.file_arcs.get(file, []),
                    )
                    for t in result.tests
                ],
            )
            row = add_uncovered_checks(
                plan,
                graph.graph["test_targets"],
                local,
                source,
                max_cases=len(plan.cases) + min(6, budget),
            )
            budget -= row["added_checks"]
            feedback[file] = row
            allocation["allocated_checks"] = row["added_checks"]
            if row["added_checks"]:
                tree = ast.parse(plan.source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module == "candidate":
                        node.module = bundle.modules[file]
                test_file = (
                    "_pcg_test_"
                    + hashlib.sha256(file.encode()).hexdigest()[:12]
                    + ".py"
                )
                snapshot.files[test_file] = ast.unparse(tree)
                generated[test_file] = {c["name"]: c for c in plan.cases}
                plans[file] = plan.to_dict()
                added = True
        if added:
            result, cache_hit = _execute_cached(snapshot, targets, execution, cache_dir)
        for file, row in feedback.items():
            local = replace(
                result,
                tests=[
                    replace(t, arcs=t.file_arcs.get(file, [])) for t in result.tests
                ],
            )
            row["edges_after"] = measure_targets(
                planners[file][1].graph["test_targets"], local
            )["distinct_branch_edges_seen"]
    from .failure_validation import validate_failures

    validation = validate_failures(
        snapshot.files,
        list(snapshot.sources),
        result,
        execution,
        import_roots=snapshot.import_roots,
        replays=failure_replays,
        minimize_trials=minimize_trials,
        time_budget=validation_time_budget,
    )
    excluded = set(validation["excluded_from_inference"])
    evidence = project_evidence(snapshot, bundle, result, static, generated, excluded)
    posterior = infer(
        bundle.blocks,
        bundle.graph,
        evidence,
        structural_importance(bundle.graph, bundle.blocks),
        model=model,
    )
    diagnostics = trace_diagnoses(result)
    failures = (
        [
            t
            for t in result.tests
            if t.oracle != "probe"
            and t.outcome == "failed"
            and t.nodeid not in excluded
        ]
        if result.valid
        else []
    )
    checked = (
        [
            t
            for t in result.tests
            if t.oracle != "probe"
            and t.outcome in {"passed", "failed"}
            and t.nodeid not in excluded
        ]
        if result.valid
        else []
    )
    ranking: list[dict] = []
    for block in bundle.blocks:
        probability = posterior[block.bid]
        file = block.bid.split("::", 1)[0]
        ranking.append(
            {
                "block": block.bid,
                "file": file,
                "name": block.qualname,
                "line": block.lineno,
                "end_line": block.end_lineno,
                "defect_probability": probability.culpability,
                "effective_trust": probability.posterior,
                "execution_status": probability.execution_status,
            }
        )
    ranking.sort(key=lambda r: (-r["defect_probability"], r["file"], r["line"]))
    return {
        "schema_version": 2,
        "mode": "project",
        "status": "no-executable-checks"
        if result.status == "no_tests"
        else "incomplete"
        if not result.valid
        else "oracle-failure"
        if failures
        else "unstable-failures"
        if excluded
        else "bounded-checks"
        if checked
        else "no-correctness-oracle",
        "summary": {
            "source_files": len(snapshot.sources),
            "existing_test_files": len(snapshot.tests),
            "generated_checks": sum(len(p["cases"]) for p in plans.values()),
            "oracle_failures": len(failures),
            "probe_failures": sum(
                t.oracle == "probe" and t.outcome == "failed" for t in result.tests
            ),
            "cache_hit": cache_hit,
            "unstable_failures": len(excluded),
        },
        "execution": result.to_dict(),
        "static": static,
        "test_plans": plans,
        "testing_feedback": feedback,
        "adaptive_budget": {
            "reserved_cases": reserve,
            "remaining_cases": budget,
            "allocations": scheduling,
            "interpretation": "expected information gain under assumed sensor probabilities; provisional first-round evidence guides testing only",
        },
        "failure_validation": validation,
        "diagnoses": diagnostics,
        "ranking": ranking,
        "graph": {
            "nodes": [
                {
                    "id": b.bid,
                    "file": b.bid.split("::", 1)[0],
                    "line": b.lineno,
                    "name": b.qualname,
                }
                for b in bundle.blocks
            ],
            "edges": [
                {"from": a, "to": b, **d} for a, b, d in bundle.graph.edges(data=True)
            ],
            "unresolved": bundle.unresolved,
        },
        "control_flow": {
            "nodes": [{"id": n, **d} for n, d in bundle.control_flow.nodes(data=True)],
            "edges": [
                {"from": a, "to": b, **d}
                for a, b, d in bundle.control_flow.edges(data=True)
            ],
        },
        "inference": bundle.graph.graph["inference"],
        "skipped": snapshot.skipped,
        "environment": {
            "backend": execution.backend,
            "python": execution.python_executable or sys.executable
            if execution.backend == "local"
            else None,
            "image": execution.image if execution.backend == "docker" else None,
            "dependency_manifests": [
                n
                for n in snapshot.files
                if n.endswith(
                    ("pyproject.toml", "requirements.txt", "requirements.lock")
                )
            ],
        },
        "limitations": [
            "Probabilities are model assumptions unless an independently evaluated model is supplied.",
            "Existing environment/image must provide project dependencies; no host dependency installation is performed.",
            "Opaque objects, reflection, concurrency and truncated traces leave unresolved dependencies.",
            "No observed failure is not proof of intended behavior.",
        ],
    }


def _execute_cached(
    snapshot: Snapshot,
    targets: list[str],
    execution: ExecutionConfig,
    cache_dir: str | None,
) -> tuple[ExecutionResult, bool]:
    if not cache_dir or execution.backend == "disabled":
        return run_project_tests(
            snapshot.files,
            list(snapshot.sources),
            targets,
            execution,
            import_roots=snapshot.import_roots,
        ), False
    digest = hashlib.sha256()
    for name, content in sorted(snapshot.files.items()):
        digest.update(name.encode() + b"\0" + content.encode() + b"\0")
    digest.update(
        json.dumps([asdict(execution), targets, sys.version], sort_keys=True).encode()
    )
    for name in ("runtime_trace.py", "pytest_plugin.py", "execution.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    # Environment fingerprint prevents reuse after dependency/image changes.
    try:
        command = (
            ["docker", "image", "inspect", "--format", "{{.Id}}", execution.image]
            if execution.backend == "docker"
            else [
                execution.python_executable or sys.executable,
                "-m",
                "pip",
                "freeze",
                "--all",
            ]
        )
        fingerprint = subprocess.run(command, capture_output=True, timeout=5)
        digest.update(fingerprint.stdout)
        if fingerprint.returncode:
            cache_dir = None
    except (OSError, subprocess.TimeoutExpired):
        cache_dir = None
    path = Path(cache_dir) / (digest.hexdigest() + ".json") if cache_dir else None
    if path and path.exists():
        try:
            data = json.loads(path.read_text())
            result = ExecutionResult(
                **{**data, "tests": [TestOutcome(**t) for t in data["tests"]]}
            )
            if result.valid:
                return result, True
        except (ValueError, TypeError, KeyError):
            pass
    result = run_project_tests(
        snapshot.files,
        list(snapshot.sources),
        targets,
        execution,
        import_roots=snapshot.import_roots,
    )
    if path and result.valid:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(result.to_dict()), encoding="utf-8")
        temporary.replace(path)
    return result, False


def write_report(report: dict, json_path: str | None, html_path: str | None) -> None:
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if html_path:
        path = Path(html_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = "".join(
            f"<tr><td>{html.escape(r['file'])}:{r['line']}</td><td>{html.escape(r['name'])}</td><td>{r['defect_probability']:.3f}</td><td>{html.escape(r['execution_status'])}</td></tr>"
            for r in report["ranking"][:100]
        )
        details = "".join(
            f"<details><summary>{html.escape(name)}</summary><pre>{html.escape(json.dumps(report[name], indent=2))}</pre></details>"
            for name in (
                "diagnoses",
                "execution",
                "static",
                "graph",
                "control_flow",
                "test_plans",
                "adaptive_budget",
                "failure_validation",
                "limitations",
            )
        )
        path.write_text(
            f"<!doctype html><meta charset='utf-8'><title>PCG project report</title><style>body{{font:15px system-ui;max-width:1100px;margin:2rem auto}}td,th{{padding:.5rem;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}details{{margin:1rem 0}}</style><h1>PCG project analysis</h1><p>Status: {html.escape(report['status'])}</p><p>Probabilities are estimates; no-failure results are bounded checks.</p><pre>{html.escape(json.dumps(report['summary'], indent=2))}</pre><table><tr><th>Location</th><th>Block</th><th>Estimated defect probability</th><th>Evidence</th></tr>{rows}</table>{details}",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project")
    parser.add_argument(
        "--execution", choices=["local", "docker", "disabled"], default="disabled"
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--python", help="preprovisioned Python environment executable")
    parser.add_argument("--worker-image", default="pcg-worker:latest")
    parser.add_argument("--max-generated-cases", type=int, default=80)
    parser.add_argument(
        "--graph-guided", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--test",
        action="append",
        help="optional existing pytest target including ::nodeid",
    )
    parser.add_argument("--cache")
    parser.add_argument("--failure-replays", type=int, default=2)
    parser.add_argument("--minimize-trials", type=int, default=6)
    parser.add_argument("--validation-time-budget", type=float, default=60)
    parser.add_argument("--model")
    parser.add_argument("--json", default="out/project-report.json")
    parser.add_argument("--html", default="out/project-report.html")
    args = parser.parse_args(argv)
    report = analyze_project(
        args.project,
        execution=ExecutionConfig(
            args.execution,
            args.timeout,
            image=args.worker_image,
            python_executable=args.python,
        ),
        max_generated_cases=args.max_generated_cases,
        graph_guided=args.graph_guided,
        model=SensorModel.load(args.model) if args.model else None,
        test_paths=args.test,
        cache_dir=args.cache,
        failure_replays=args.failure_replays,
        minimize_trials=args.minimize_trials,
        validation_time_budget=args.validation_time_budget,
    )
    write_report(report, args.json, args.html)
    print(json.dumps({"status": report["status"], **report["summary"]}, indent=2))
    print(f"Reports: {args.json}, {args.html}")
    return (
        2
        if report["status"]
        in {"incomplete", "unstable-failures", "no-executable-checks"}
        else 1
        if report["status"] == "oracle-failure"
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
