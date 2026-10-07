"""Bounded graph-prioritized boundary probes and declarative contract tests.

No candidate is imported by the planner. Unknown outputs are never copied into
assertions. Probes are neutral; only supplied contracts or opted-in return
annotations provide output oracles.
"""

from __future__ import annotations

import ast
import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from .blocks import Block
from .execution import ExecutionResult
from .sensors import SensorModel, test_sensitivity


@dataclass
class TestPlan:
    source: str = ""
    cases: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    limits: dict = field(default_factory=dict)
    recommendations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cases": self.cases,
            "skipped": self.skipped,
            "limits": self.limits,
            "recommendations": self.recommendations,
        }


def _type_spec(node: ast.expr | None) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _type_spec(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return None
    if isinstance(node, ast.Name) and node.id in {
        "int",
        "float",
        "str",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
    }:
        return node.id
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = _type_spec(node.left), _type_spec(node.right)
        return ["union", left, right] if left and right else None
    if isinstance(node, ast.Subscript):
        name = ast.unparse(node.value).split(".")[-1].lower()
        args = (
            list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        )
        specs = [_type_spec(a) for a in args]
        arity_ok = (
            (name in {"list", "set", "optional"} and len(specs) == 1)
            or (name == "dict" and len(specs) == 2)
            or (name == "union" and len(specs) >= 2)
        )
        if all(specs) and arity_ok:
            return ["union", specs[0], "None"] if name == "optional" else [name, *specs]
    return None


def _values(spec: Any) -> list[Any]:
    if isinstance(spec, list):
        if spec[0] == "union":
            return [v for part in spec[1:] for v in _values(part)]
        if spec[0] in {"list", "set"}:
            vals = _values(spec[1])
            if spec[0] == "set":
                return (
                    [set(), {vals[0]}]
                    if vals and isinstance(vals[0], (int, float, str, bool))
                    else []
                )
            return [[], vals[:1], vals[:2], vals[:2][::-1], vals[:1] * 2, vals[-2:]]
        if spec[0] == "dict":
            keys, vals = _values(spec[1]), _values(spec[2])
            return (
                [{}, {keys[0]: vals[0]}]
                if keys and vals and isinstance(keys[0], (int, float, str, bool))
                else []
            )
        return []
    domains: dict[str, list[Any]] = {
        "int": [0, 1, -1, 2],
        "float": [0.0, 1.0, -1.0, 0.5],
        "str": ["", "a", " ", "0"],
        "bool": [False, True],
        "list": [[], [0], [0, 1]],
        "dict": [{}, {"a": 0}],
        "tuple": [(), (0,)],
        "set": [set(), {0}],
        "None": [None],
    }
    return domains.get(spec, [])


def _accepts_float(spec: Any) -> bool:
    return spec == "float" or (
        isinstance(spec, list)
        and spec[0] == "union"
        and any(_accepts_float(part) for part in spec[1:])
    )


_HELPERS = """
import copy as _pcg_copy
import pytest as _pcg_pytest

def _pcg_type(value, spec):
    if isinstance(spec, str):
        types = {"int": int, "float": (int, float), "str": str, "bool": bool,
                 "list": list, "dict": dict, "tuple": tuple, "set": set}
        return value is None if spec == "None" else isinstance(value, types[spec])
    kind, *parts = spec
    if kind == "union":
        return any(_pcg_type(value, part) for part in parts)
    if kind == "dict":
        return isinstance(value, dict) and all(_pcg_type(k, parts[0]) and _pcg_type(v, parts[1]) for k, v in value.items())
    return isinstance(value, list if kind == "list" else set) and all(_pcg_type(v, parts[0]) for v in value)
"""


def _validate_contracts(contracts: list[dict], names: set[str]) -> None:
    if not isinstance(contracts, list) or len(contracts) > 100:
        raise ValueError("contracts must be a list of at most 100 cases")
    for case in contracts:
        if not isinstance(case, dict) or case.get("function") not in names:
            raise ValueError("contract must name a top-level synchronous function")
        if set(case) - {
            "function",
            "args",
            "kwargs",
            "expected",
            "raises",
            "relation",
            "other_args",
            "other_kwargs",
            "property",
        }:
            raise ValueError("unknown contract field")
        if sum(k in case for k in ("expected", "raises", "relation", "property")) != 1:
            raise ValueError(
                "provide exactly one expected value, exception or relation"
            )
        if "raises" in case and case["raises"] not in {
            "ValueError",
            "TypeError",
            "ZeroDivisionError",
            "IndexError",
            "KeyError",
            "OverflowError",
            "AssertionError",
        }:
            raise ValueError("unsupported expected exception")
        if "relation" in case and case["relation"] not in {
            "equal",
            "nondecreasing",
            "idempotent",
        }:
            raise ValueError("unsupported metamorphic relation")
        if "property" in case:
            if case["property"] not in {
                "nonnegative",
                "length_preserving",
                "sorted",
                "permutation",
                "input_unchanged",
                "idempotent",
            }:
                raise ValueError("unsupported reusable property")
            if set(case) != {"function", "property"}:
                raise ValueError(
                    "a reusable property applies to generated inputs; supply only function and property"
                )
        if (
            not isinstance(case.get("args", []), list)
            or not isinstance(case.get("kwargs", {}), dict)
            or any(not isinstance(k, str) for k in case.get("kwargs", {}))
        ):
            raise ValueError(
                "contract args must be a list and kwargs a string-keyed dict"
            )
        if "relation" in case and (
            not isinstance(case.get("other_args", []), list)
            or not isinstance(case.get("other_kwargs", {}), dict)
        ):
            raise ValueError("invalid second metamorphic input")
        try:
            encoded = json.dumps(case, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("contracts must contain finite JSON values") from exc
        if len(encoded) > 20_000:
            raise ValueError("contract exceeds size limit")


def plan_tests(
    source: str,
    blocks: list[Block],
    graph: nx.DiGraph,
    *,
    contracts: list[dict] | None = None,
    max_cases: int = 40,
    seed: int = 7,
    annotation_contracts: bool = False,
    autonomous: bool = False,
    graph_guided: bool = True,
) -> TestPlan:
    """Generate tests without guessing semantic correctness from function names."""
    if not 1 <= max_cases <= 200:
        raise ValueError("max_cases must be between 1 and 200")
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    contracts = [] if contracts is None else list(contracts)
    if autonomous:
        from .autotest import literal_examples

        for fn in functions.values():
            for row in literal_examples(fn):
                try:
                    encoded = json.dumps(row, allow_nan=False)
                except (TypeError, ValueError):
                    continue
                if len(contracts) < 100 and len(encoded) <= 20_000:
                    contracts.append(row)
    _validate_contracts(contracts, set(functions))
    plan = TestPlan(
        limits={
            "max_cases": max_cases,
            "seed": seed,
            "annotation_contracts": annotation_contracts,
        }
    )
    chunks = [_HELPERS]
    if autonomous:
        plan.skipped.extend(
            {
                "function": node.name,
                "reason": "asynchronous interface is not automatically supported",
            }
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef)
        )
    total_budget = max_cases
    if autonomous:
        max_cases -= min(6, max_cases // 5)
        max_cases = max(
            1,
            max_cases
            - min(
                max_cases // 2,
                2 * len(functions) + sum("property" in c for c in contracts),
            ),
        )

    def emit(
        name: str,
        args: list[Any],
        kwargs: dict,
        oracle: str,
        body: str,
        spec: Any = None,
        scope: str = "example",
    ) -> None:
        index = len(plan.cases)
        scope = oracle if oracle in {"annotation", "probe"} else scope
        plan.cases.append(
            {
                "name": f"test_pcg_{index:03}",
                "function": name,
                "args": repr(args),
                "kwargs": repr(kwargs),
                "oracle": oracle,
                "return_type": spec,
                "oracle_scope": scope,
            }
        )
        chunks.append(
            f"\n@_pcg_pytest.mark.pcg_oracle({oracle!r}, scope={scope!r})\ndef test_pcg_{index:03}():\n    from candidate import {name} as _pcg_fn\n    args = {args!r}\n    kwargs = {kwargs!r}\n"
            + body
        )

    for case in contracts:
        if "property" in case:
            continue
        if len(plan.cases) >= max_cases:
            plan.skipped.append(
                {"function": case["function"], "reason": "contract budget exhausted"}
            )
            continue
        if "expected" in case:
            body = f"    assert _pcg_fn(*args, **kwargs) == {case['expected']!r}\n"
        elif "raises" in case:
            body = f"    with _pcg_pytest.raises({case['raises']}):\n        _pcg_fn(*args, **kwargs)\n"
        else:
            body = "    original = _pcg_copy.deepcopy((args, kwargs))\n    first = _pcg_fn(*args, **kwargs)\n"
            if case["relation"] == "idempotent":
                body += "    assert _pcg_fn(_pcg_copy.deepcopy(first)) == first\n"
            else:
                body += f"    second = _pcg_fn(*{case.get('other_args', [])!r}, **{case.get('other_kwargs', {})!r})\n    assert first {'==' if case['relation'] == 'equal' else '<='} second\n"
        emit(
            case["function"],
            case.get("args", []),
            case.get("kwargs", {}),
            "contract",
            body,
            scope="metamorphic" if "relation" in case else "example",
        )

    by_name = {b.qualname: b for b in blocks if b.kind == "function"}
    # Central functions test more downstream behavior. Stable ties preserve reproducibility.
    ordered = sorted(
        functions,
        key=lambda name: (
            -len(nx.descendants(graph, by_name[name].bid)) if name in by_name else 0,
            name,
        ),
    )
    inputs: dict[str, list[tuple[list, dict]]] = {}
    properties: dict[str, list[str]] = {}
    for case in contracts:
        if "property" in case:
            properties.setdefault(case["function"], []).append(case["property"])
    for name in ordered:
        fn = functions[name]
        if (
            fn.decorator_list
            or fn.args.vararg
            or fn.args.kwarg
            or any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn))
        ):
            plan.skipped.append(
                {
                    "function": name,
                    "reason": "decorated, variadic or generator interface requires explicit cases",
                }
            )
            continue
        params = [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]
        defaults: dict[str, ast.expr] = {}
        positional = [*fn.args.posonlyargs, *fn.args.args]
        defaults.update(
            {
                p.arg: d
                for p, d in zip(
                    positional[len(positional) - len(fn.args.defaults) :],
                    fn.args.defaults,
                )
            }
        )
        defaults.update(
            {
                p.arg: d
                for p, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)
                if d is not None
            }
        )
        domains = []
        for param in params:
            values = _values(_type_spec(param.annotation))
            if not values and param.arg in defaults:
                try:
                    values = [ast.literal_eval(defaults[param.arg])]
                except (ValueError, TypeError):
                    pass
            # Untyped numeric uses provide probe-only input hypotheses.
            if not values and any(
                isinstance(n, ast.BinOp)
                and any(
                    isinstance(a, ast.Name) and a.id == param.arg for a in ast.walk(n)
                )
                for n in ast.walk(fn)
            ):
                values = [0, 1, -1, 2]
            if not values:
                comparisons = [
                    n
                    for n in ast.walk(fn)
                    if isinstance(n, ast.Compare)
                    and any(
                        isinstance(a, ast.Name) and a.id == param.arg
                        for a in ast.walk(n)
                    )
                ]
                if any(
                    isinstance(a, ast.Constant) and type(a.value) in {int, float}
                    for n in comparisons
                    for a in ast.walk(n)
                ):
                    values = [0, 1, -1, 2]
                elif any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == "len"
                    and any(
                        isinstance(a, ast.Name) and a.id == param.arg for a in n.args
                    )
                    for n in ast.walk(fn)
                ):
                    values = [[], [0], [0, 1], "", "a", {}, {"a": 0}]
            for node in ast.walk(fn):
                if isinstance(node, ast.Compare) and any(
                    isinstance(n, ast.Name) and n.id == param.arg
                    for n in ast.walk(node)
                ):
                    for constant in (
                        n.value
                        for n in ast.walk(node)
                        if isinstance(n, ast.Constant) and type(n.value) in {int, float}
                    ):
                        if (
                            isinstance(constant, (int, float))
                            and abs(constant) <= 1_000_000
                            and math.isfinite(constant)
                            and values
                            and type(values[0]) in {int, float}
                        ):
                            spec = _type_spec(param.annotation)
                            if (
                                spec
                                and type(values[0]) is int
                                and not _accepts_float(spec)
                            ):
                                floor, ceil = math.floor(constant), math.ceil(constant)
                                values.extend([floor - 1, floor, ceil, ceil + 1])
                            else:
                                values.extend([constant - 1, constant, constant + 1])
            if not values:
                break
            domains.append(list({repr(v): v for v in values}.values()))
        if len(domains) != len(params):
            plan.skipped.append(
                {
                    "function": name,
                    "reason": "unknown input domain; add annotations or a contract case",
                }
            )
            continue
        combinations = [tuple(d[0] for d in domains)]
        for i, domain in enumerate(domains):
            for value in domain[1:]:
                combo = list(combinations[0])
                combo[i] = value
                combinations.append(tuple(combo))
        # Small pairwise/cartesian exploration exposes interactions between parameters.
        combinations.extend(itertools.islice(itertools.product(*domains), 16))
        unique = {repr(c): c for c in combinations}
        inputs[name] = [
            (
                list(c[: len(positional)]),
                {p.arg: v for p, v in zip(fn.args.kwonlyargs, c[len(positional) :])},
            )
            for c in unique.values()
        ]
    # Round-robin avoids one large function starving all other functions of probes.
    for i in range(max((len(cases) for cases in inputs.values()), default=0)):
        for name in ordered:
            if (
                name not in inputs
                or i >= len(inputs[name])
                or len(plan.cases) >= max_cases
            ):
                continue
            args, kwargs = inputs[name][i]
            emit(name, args, kwargs, "probe", "    _pcg_fn(*args, **kwargs)\n")
            for property_name in properties.get(name, []):
                if len(plan.cases) >= max_cases:
                    break
                check = {
                    "nonnegative": "assert value >= 0",
                    "length_preserving": "assert len(value) == len(input_value)",
                    "sorted": "assert list(value) == sorted(value)",
                    "permutation": "assert sorted(value) == sorted(input_value)",
                    "input_unchanged": "assert (args, kwargs) == original",
                    "idempotent": "assert _pcg_fn(_pcg_copy.deepcopy(value)) == value",
                }[property_name]
                body = (
                    "    original = _pcg_copy.deepcopy((args, kwargs))\n    input_value = original[0][0] if original[0] else next(iter(original[1].values()), None)\n    try:\n        value = _pcg_fn(*args, **kwargs)\n    except Exception:\n        _pcg_pytest.skip('input acceptance is unspecified; inspect the paired probe')\n    "
                    + check
                    + "\n"
                )
                emit(name, args, kwargs, "contract", body, scope="property")
                plan.cases[-1]["property"] = property_name
            spec = _type_spec(functions[name].returns)
            if annotation_contracts and spec and len(plan.cases) < max_cases:
                emit(
                    name,
                    args,
                    kwargs,
                    "annotation",
                    "    try:\n        value = _pcg_fn(*args, **kwargs)\n    except Exception:\n        _pcg_pytest.skip('input acceptance is unspecified; inspect the paired probe')\n"
                    + f"    assert _pcg_type(value, {spec!r}), 'return value violates declared annotation'\n",
                    spec,
                )
    plan.source = "\n".join(chunks) if plan.cases else ""
    for name in properties:
        if name not in inputs:
            plan.skipped.append(
                {
                    "function": name,
                    "reason": "property cannot run without a known generated input domain; supply typed inputs or explicit cases",
                }
            )
    if autonomous:
        from .autotest import add_fuzz_tests, add_sequence_probes
        from .graph_testing import plan_graph_targets

        graph_targets = (
            plan_graph_targets(source, graph.graph["dependence"])
            if graph_guided and "dependence" in graph.graph
            else None
        )
        graph.graph["test_targets"] = graph_targets
        plan.limits["graph_path_solver"] = (
            graph_targets.limits if graph_targets else {"status": "no dependence graph"}
        )

        initial_budget = total_budget - min(6, total_budget // 5)
        sequence_slots = min(3, sum(isinstance(n, ast.ClassDef) for n in tree.body))
        add_fuzz_tests(
            plan,
            {name: functions[name] for name in ordered},
            max_cases=max(1, initial_budget - sequence_slots),
            seed=seed,
            graph_targets=graph_targets,
        )
        add_sequence_probes(
            plan, source, max_cases=initial_budget, seed=seed, graph_guided=graph_guided
        )
        dependence = graph.graph.get("dependence")
        if dependence is not None:
            for case in plan.cases:
                target_fn = functions.get(case["function"])
                if target_fn and case.get("method") == "property-fuzzing":
                    case["branch_dependency_slices"] = [
                        {
                            "condition_line": node.lineno,
                            "earlier_lines": sorted(dependence.backward({node.lineno})),
                        }
                        for node in ast.walk(target_fn)
                        if isinstance(node, (ast.If, ast.While))
                    ][:24]
        if not plan.cases:
            plan.source = ""
    return plan


def recommend_checks(
    plan: TestPlan,
    blocks: list[Block],
    graph: nx.DiGraph,
    execution: ExecutionResult | None,
    model: SensorModel,
) -> list[dict]:
    """Decision support using joint-posterior information gain and graph coverage.

    Coverage forecasts are conservative and labelled as such. Already executed
    cases are not recommended for a redundant rerun. Probe outcomes have no
    correctness oracle, so no defect-information gain is attributed to them.
    """
    from .probabilistic import rank_next_tests

    done = (
        {
            t.nodeid.rsplit("::", 1)[-1]
            for t in execution.tests
            if t.outcome in {"passed", "failed", "skipped", "xfailed", "xpassed"}
        }
        if execution and execution.valid
        else set()
    )
    by_name = {b.qualname: b for b in blocks if b.kind == "function"}
    umbrellas = {b.qualname.split("#seg", 1)[0] for b in blocks if b.kind == "segment"}
    candidates = []
    metadata = {}
    for case in plan.cases:
        if case["name"] in done or case["oracle"] == "probe":
            continue
        target = by_name.get(case["function"])
        if target is None:
            continue
        reachable = nx.ancestors(graph, target.bid) | {target.bid}
        predicted = tuple(
            b.bid for b in blocks if b.bid in reachable and b.qualname not in umbrellas
        )
        if not predicted:
            continue
        candidates.append(
            {
                "name": case["name"],
                "blocks": predicted,
                "cost_seconds": 1.0,
                "sensitivity": test_sensitivity(model, case["oracle_scope"]),
            }
        )
        metadata[case["name"]] = {
            "function": case["function"],
            "oracle": case["oracle"],
            "predicted_blocks": list(predicted),
            "coverage_basis": "static dependencies; actual execution may cover a subset",
            "cost_basis": "unit cost assumption; not measured",
        }
    # rank_next_tests accepts sensitivity/leak explicitly to match this model.
    ranked = rank_next_tests(
        graph.graph["defect_posterior"],
        candidates,
        sensitivity=model.likelihoods["exec"]["flag_given_defect"],
        leak=model.likelihoods["exec"]["flag_given_clean"],
    )
    return [{**row, **metadata[row["name"]]} for row in ranked]
