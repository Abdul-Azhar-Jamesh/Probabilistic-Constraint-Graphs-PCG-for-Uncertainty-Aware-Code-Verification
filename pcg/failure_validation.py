"""Bounded, worker-only replay and graph-anchored failure reduction.

Reduced reproducers are report artifacts, never replacements for the original
suite. Replays validate stability; they are not additional Bayesian observations.
"""

from __future__ import annotations

import ast
import copy
import re
import time
import hashlib
import math
from dataclasses import replace
from typing import Any

from .execution import ExecutionConfig, ExecutionResult, TestOutcome, run_project_tests


def _decode(value: Any) -> Any:
    if type(value) in (int, float, bool, str) or value is None:
        return value
    if (
        isinstance(value, dict)
        and value.get("type") in {"list", "tuple"}
        and value.get("length") == len(value.get("items", []))
    ):
        items = [_decode(v) for v in value["items"]]
        return tuple(items) if value["type"] == "tuple" else items
    raise ValueError("opaque or truncated input")


def trace_reproducer(
    test: TestOutcome, files: dict[str, str]
) -> tuple[str, str] | None:
    """Turn a captured primitive return-contract violation into a concrete test."""
    if test.oracle != "annotation":
        return None
    from .testgen import _type_spec, _HELPERS
    from .project_graph import module_name

    for trace in reversed(test.traces):
        if "." in trace["function"] or trace["truncated"] or trace["file"] not in files:
            continue
        functions = [
            n
            for n in ast.parse(files[trace["file"]]).body
            if isinstance(n, ast.FunctionDef) and n.name == trace["function"]
        ]
        if not functions or functions[0].args.vararg or functions[0].args.kwarg:
            continue
        fn = functions[0]
        spec = _type_spec(fn.returns)
        simple = {
            "int": (int,),
            "float": (int, float),
            "bool": (bool,),
            "str": (str,),
            "None": (type(None),),
        }
        if not isinstance(spec, str) or spec not in simple:
            continue
        values = [
            e.get("value")
            for e in trace["events"]
            if e["kind"] == "return" and e["root"] == trace["root"]
        ]
        if (
            not values
            or isinstance(values[-1], dict)
            or isinstance(values[-1], simple[spec])
        ):
            continue
        try:
            inputs = {k: _decode(v) for k, v in trace["inputs"].items()}
            args = [inputs[p.arg] for p in fn.args.posonlyargs]
            kwargs = {
                p.arg: inputs[p.arg] for p in [*fn.args.args, *fn.args.kwonlyargs]
            }
        except (ValueError, KeyError):
            continue
        file = (
            "_pcg_reproducer_"
            + hashlib.sha256(test.nodeid.encode()).hexdigest()[:12]
            + ".py"
        )
        if file in files:
            continue
        source = (
            _HELPERS
            + f"\n@_pcg_pytest.mark.pcg_oracle('annotation', scope='annotation')\ndef test_reproducer():\n    from {module_name(trace['file'])} import {fn.name} as function\n    args = {args!r}\n    kwargs = {kwargs!r}\n    value = function(*args, **kwargs)\n    assert _pcg_type(value, {spec!r})\n"
        )
        return file, source
    return None


def signature(test: TestOutcome) -> tuple:
    kind = re.match(
        r"(?:E\s+)?([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception|Failure|ExceptionGroup|FailedHealthCheck))(?=:|\b)",
        test.exception or "",
    )
    exception = (
        kind.group(1)
        if kind
        else "AssertionError"
        if (test.exception or "").lstrip().startswith(("assert ", "E assert "))
        else "UnclassifiedFailure"
    )
    if test.file_frames:
        frame = test.file_frames[-1]
        return exception, frame["file"], frame["line"]
    returns = [
        (e["file"], e["line"])
        for t in test.traces
        for e in t["events"]
        if e["kind"] == "return" and e["root"] == t["root"]
    ]
    return (exception, *returns[-1]) if returns else (exception, None, None)


def dependency_locations(test: TestOutcome) -> set[tuple[str, int]]:
    """Follow captured event-graph dependencies backward from the symptom."""
    _, file, line = signature(test)
    for trace in reversed(test.traces):
        events = {e["id"]: e for e in trace["events"]}
        seeds = [
            e["id"]
            for e in events.values()
            if e["kind"] in {"exception", "return"}
            and e["file"] == file
            and e["line"] == line
        ]
        if not seeds:
            continue
        stack, reached = seeds[-1:], set()
        while stack:
            key = stack.pop()
            if key in reached or key not in events:
                continue
            reached.add(key)
            stack.extend(events[key]["dependencies"])
        return {
            (events[key]["file"], events[key]["line"])
            for key in reached
            if events[key]["kind"] not in {"enter", "exit"}
        }
    return set()


def same_failure(original: TestOutcome, trial: TestOutcome) -> bool:
    return signature(original) == signature(trial) and dependency_locations(
        original
    ) <= dependency_locations(trial)


def _size(tree: ast.AST) -> int:
    return len(ast.unparse(tree))


def reductions(source: str, nodeid: str) -> list[str]:
    """Remove setup/actions and shrink literals while retaining assertion syntax."""
    tree = ast.parse(source)
    function = nodeid.rsplit("::", 1)[-1].split("[", 1)[0]
    matches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == function
    ]
    if len(matches) != 1:
        return []
    fn = matches[0]
    candidates = []
    protected_names = set()
    assertions = [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]
    for assertion in assertions:
        for node in ast.walk(assertion):
            operands = (
                [node.left, *node.comparators]
                if isinstance(node, ast.Compare)
                else node.args
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"isinstance", "_pcg_type"}
                else []
            )
            for operand in operands:
                if isinstance(operand, ast.Name):
                    protected_names.add(operand.id)
    protected_assignments = [
        stmt
        for stmt in fn.body
        if isinstance(stmt, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Store)
            and n.id in protected_names
            for n in ast.walk(stmt)
        )
    ]
    protected_nodes = {n for stmt in protected_assignments for n in ast.walk(stmt)}
    # Keep the final action and assertions. Earlier actions may be unnecessary.
    for i, stmt in enumerate(fn.body[:-1]):
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.Expr)):
            continue
        if stmt in protected_assignments:
            continue
        trial = copy.deepcopy(tree)
        target = next(
            n
            for n in ast.walk(trial)
            if isinstance(n, ast.FunctionDef) and n.name == function
        )
        del target.body[i]
        candidates.append(trial)
    originals = list(ast.walk(fn))
    body_nodes = {n for stmt in fn.body for n in ast.walk(stmt)}
    for index, node in enumerate(originals):
        if node not in body_nodes:
            continue
        if node in protected_nodes:
            continue
        if not isinstance(node, (ast.Constant, ast.List, ast.Tuple, ast.Dict)):
            continue
        # Never weaken expected outputs or assertion expressions.
        if any(
            node in ast.walk(assertion)
            for assertion in ast.walk(fn)
            if isinstance(assertion, ast.Assert)
        ):
            continue
        choices: list[Any] = []
        if isinstance(node, ast.Constant) and type(node.value) is int:
            half = node.value // 2 if node.value >= 0 else -((-node.value) // 2)
            choices = [0, 1 if node.value > 0 else -1, half]
        elif isinstance(node, ast.Constant) and type(node.value) is str:
            choices = ["", node.value[: len(node.value) // 2]]
        elif isinstance(node, (ast.List, ast.Tuple)):
            choices = [
                [],
                node.elts[: len(node.elts) // 2],
                node.elts[len(node.elts) // 2 :],
            ]
        elif isinstance(node, ast.Dict):
            choices = [
                ([], []),
                (
                    node.keys[: len(node.keys) // 2],
                    node.values[: len(node.values) // 2],
                ),
            ]
        for value in choices:
            trial = copy.deepcopy(tree)
            target = next(
                n
                for n in ast.walk(trial)
                if isinstance(n, ast.FunctionDef) and n.name == function
            )
            changed = list(ast.walk(target))[index]
            if isinstance(changed, ast.Constant):
                changed.value = value
            elif isinstance(changed, (ast.List, ast.Tuple)):
                changed.elts = copy.deepcopy(value)
            elif isinstance(changed, ast.Dict):
                changed.keys, changed.values = copy.deepcopy(value)
            candidates.append(trial)
    return list(
        dict.fromkeys(
            ast.unparse(t)
            for t in sorted(candidates, key=_size)
            if _size(t) < _size(tree)
        )
    )


def validate_failures(
    files: dict[str, str],
    sources: list[str],
    result: ExecutionResult,
    config: ExecutionConfig,
    *,
    import_roots: list[str] | None = None,
    replays: int = 2,
    max_failures: int = 8,
    minimize_trials: int = 6,
    time_budget: float = 60,
) -> dict:
    if (
        not 0 <= replays <= 5
        or not 0 <= minimize_trials <= 30
        or not 1 <= max_failures <= 50
        or not math.isfinite(time_budget)
        or not 0 < time_budget <= 600
    ):
        raise ValueError("failure validation budgets exceeded")
    report: dict = {
        "replays": replays,
        "failures": [],
        "excluded_from_inference": [],
        "reproducers": [],
        "worker_runs": 0,
        "limitations": [
            "A bounded replay cannot prove determinism.",
            "Reduction preserves exception type and a candidate symptom location, not a proof of identical semantic cause.",
            "Only replayed flaky/inconclusive failures are excluded; unselected failures remain unreplayed.",
            "Beta failure-rate summaries assume exchangeable runs in this environment; they are not defect probabilities.",
        ],
    }
    if not result.valid or config.backend == "disabled" or replays == 0:
        report["status"] = "disabled" if replays == 0 else "execution-unavailable"
        return report
    failed = sorted(
        (t for t in result.tests if t.outcome == "failed"),
        key=lambda t: (t.oracle == "probe", t.nodeid),
    )[:max_failures]
    deadline = time.monotonic() + time_budget
    histories: dict[str, list[TestOutcome | None]] = {t.nodeid: [t] for t in failed}
    for _ in range(replays):
        if not failed or time.monotonic() >= deadline:
            break
        run = run_project_tests(
            files,
            sources,
            [t.nodeid for t in failed],
            replace(
                config,
                timeout=max(1, min(config.timeout, int(deadline - time.monotonic()))),
            ),
            import_roots=import_roots,
        )
        report["worker_runs"] += 1
        by_id = {t.nodeid: t for t in run.tests} if run.valid else {}
        for test in failed:
            histories[test.nodeid].append(by_id.get(test.nodeid))
    stable = []
    for test in failed:
        history = histories[test.nodeid]
        valid = [
            t for t in history if t is not None and t.outcome in {"passed", "failed"}
        ]
        repeats = history[1:]
        if any(
            t is not None
            and t.outcome in {"passed", "failed"}
            and (t.outcome != "failed" or signature(t) != signature(test))
            for t in repeats
        ):
            status = "flaky"
        elif len(history) != replays + 1 or len(valid) != len(history):
            status = "inconclusive"
        else:
            status = "stable-failure"
            stable.append(test)
        failures = sum(t.outcome == "failed" for t in valid)
        report["failures"].append(
            {
                "test": test.nodeid,
                "oracle": test.oracle,
                "status": status,
                "signature": signature(test),
                "outcomes": [
                    t.outcome if t else "harness-unavailable" for t in history
                ],
                "beta_failure_rate": {
                    "alpha": failures + 1,
                    "beta": len(valid) - failures + 1,
                    "mean": (failures + 1) / (len(valid) + 2),
                },
            }
        )
        if status != "stable-failure":
            report["excluded_from_inference"].append(test.nodeid)
    # One bounded minimization job, oracle failures preferred.
    for test in stable[:1]:
        file = test.nodeid.split("::", 1)[0]
        original = files.get(file, "")
        target = test.nodeid
        local_files = files
        best, attempts, reductions_count = original, 0, 0
        concrete = trace_reproducer(test, files)
        if concrete and minimize_trials and time.monotonic() < deadline:
            replay_file, replay_source = concrete
            trial = run_project_tests(
                {**files, replay_file: replay_source},
                sources,
                [replay_file + "::test_reproducer"],
                replace(
                    config,
                    timeout=max(
                        1, min(config.timeout, int(deadline - time.monotonic()))
                    ),
                ),
                import_roots=import_roots,
            )
            attempts += 1
            report["worker_runs"] += 1
            outcome = trial.tests[0] if trial.valid and len(trial.tests) == 1 else None
            if (
                outcome
                and outcome.outcome == "failed"
                and outcome.oracle == test.oracle
                and same_failure(test, outcome)
            ):
                file, best, target = (
                    replay_file,
                    replay_source,
                    replay_file + "::test_reproducer",
                )
                local_files = {**files, file: best}
                reductions_count += 1
        seen = {original}
        while attempts < minimize_trials and time.monotonic() < deadline:
            candidates = [s for s in reductions(best, target) if s not in seen]
            if not candidates:
                break
            accepted = False
            for candidate in candidates:
                if attempts >= minimize_trials or time.monotonic() >= deadline:
                    break
                seen.add(candidate)
                trial = run_project_tests(
                    {**local_files, file: candidate},
                    sources,
                    [target],
                    replace(
                        config,
                        timeout=max(
                            1, min(config.timeout, int(deadline - time.monotonic()))
                        ),
                    ),
                    import_roots=import_roots,
                )
                attempts += 1
                report["worker_runs"] += 1
                outcome = next((t for t in trial.tests if t.nodeid == target), None)
                if (
                    trial.valid
                    and outcome
                    and outcome.outcome == "failed"
                    and outcome.oracle == test.oracle
                    and outcome.oracle_scope == test.oracle_scope
                    and same_failure(test, outcome)
                    and signature(test)[1] is not None
                ):
                    best, accepted = candidate, True
                    reductions_count += 1
                    break
            if not accepted:
                break
        report["reproducers"].append(
            {
                "test": test.nodeid,
                "test_file": file,
                "replay_target": target,
                "source": best,
                "status": "reduced" if reductions_count else "unchanged",
                "attempts": attempts,
                "accepted_reductions": reductions_count,
                "original_bytes": len(original.encode()),
                "reduced_bytes": len(best.encode()),
                "signature": signature(test),
                "preserved_dependency_locations": [
                    {"file": f, "line": line}
                    for f, line in sorted(dependency_locations(test))
                ],
                "dependency_basis": "captured executed event-graph slice"
                if dependency_locations(test)
                else "trace unavailable; symptom anchor only",
                "minimality": "bounded local reduction; global minimality not established",
            }
        )
    report["status"] = (
        "complete"
        if all(len(h) == replays + 1 for h in histories.values())
        else "budget-exhausted"
    )
    return report
