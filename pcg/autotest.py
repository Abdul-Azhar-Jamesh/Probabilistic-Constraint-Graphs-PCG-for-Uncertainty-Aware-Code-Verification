"""Worker-only property fuzzing, shrinking and conservative automatic oracles.

Planning parses source only. Generated code executes exclusively in the configured
worker. No observed candidate output is promoted to an expected answer.
"""

from __future__ import annotations

import ast
import doctest
from typing import Any, cast


def strategy(spec: Any) -> str | None:
    """Bounded Hypothesis strategy expression for supported declared types."""
    simple = {
        "int": "_st.integers(-10000, 10000)",
        "float": "_st.floats(-10000, 10000, allow_nan=False, allow_infinity=False)",
        "str": "_st.text(max_size=40)",
        "bool": "_st.booleans()",
        "None": "_st.none()",
        "list": "_st.lists(_st.integers(-100,100), max_size=20)",
        "dict": "_st.dictionaries(_st.text(max_size=8), _st.integers(-100,100), max_size=10)",
        "tuple": "_st.lists(_st.integers(-100,100), max_size=20).map(tuple)",
        "set": "_st.sets(_st.integers(-100,100), max_size=20)",
    }
    if isinstance(spec, str):
        return simple.get(spec)
    if not isinstance(spec, list):
        return None
    parts = [strategy(part) for part in spec[1:]]
    if not parts or any(part is None for part in parts):
        return None
    if spec[0] == "union":
        return f"_st.one_of({', '.join(str(p) for p in parts)})"
    if spec[0] in {"list", "set"}:
        method = "lists" if spec[0] == "list" else "sets"
        return f"_st.{method}({parts[0]}, max_size=20)"
    if spec[0] == "dict":
        return f"_st.dictionaries({parts[0]}, {parts[1]}, max_size=10)"
    return None


def literal_examples(fn: ast.FunctionDef) -> list[dict]:
    """Use only a literal call/result subset of existing doctests; never eval."""
    rows = []
    try:
        examples = doctest.DocTestParser().get_examples(ast.get_docstring(fn) or "")
    except ValueError:
        return []
    for example in examples:
        try:
            call = ast.parse(example.source, mode="eval").body
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if call.func.id != fn.name or any(k.arg is None for k in call.keywords):
                continue
            rows.append(
                {
                    "function": fn.name,
                    "args": [ast.literal_eval(a) for a in call.args],
                    "kwargs": {k.arg: ast.literal_eval(k.value) for k in call.keywords},
                    "expected": ast.literal_eval(example.want.strip()),
                }
            )
        except (SyntaxError, ValueError, TypeError):
            continue
    return rows


def branch_inputs(fn: ast.FunctionDef, params: list[str]) -> list[tuple]:
    """Bounded affine branch inversion, including dependencies on earlier locals.

    Handles numeric straight-line aliases and comparisons, without executing AST.
    Other predicates remain covered by ordinary fuzzing; these are input hypotheses.
    """
    env: dict[str, tuple[str, float, float]] = {p: (p, 1, 0) for p in params}

    def affine(node: ast.AST) -> tuple[str, float, float] | None:
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.BinOp):
            left = affine(node.left)
            if (
                left
                and isinstance(node.right, ast.Constant)
                and type(node.right.value) in {int, float}
            ):
                p, a, b = left
                c = cast(float, node.right.value)
                if isinstance(node.op, ast.Add):
                    return p, a, b + c
                if isinstance(node.op, ast.Sub):
                    return p, a, b - c
                if isinstance(node.op, ast.Mult):
                    return p, a * c, b * c
        return None

    results = []
    for statement in fn.body:
        if isinstance(statement, ast.Assign):
            value = affine(statement.value)
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    env.pop(target.id, None)
                    if value:
                        env[target.id] = value
        for compare in (n for n in ast.walk(statement) if isinstance(n, ast.Compare)):
            if len(compare.comparators) != 1:
                continue
            left, right = affine(compare.left), compare.comparators[0]
            if (
                not left
                or not isinstance(right, ast.Constant)
                or type(right.value) not in {int, float}
            ):
                continue
            p, a, b = left
            if not a:
                continue
            boundary = (cast(float, right.value) - b) / a
            if abs(boundary) > 10000:
                continue
            for input_value in (int(boundary) - 1, int(boundary), int(boundary) + 1):
                result = [0] * len(params)
                result[params.index(p)] = input_value
                results.append(tuple(result))
    return list(dict.fromkeys(results))[:24]


FUZZ_IMPORTS = """
from hypothesis import given as _given, settings as _settings, seed as _seed, example as _example, Phase as _Phase, HealthCheck as _HealthCheck
from hypothesis import strategies as _st
"""


def add_fuzz_tests(
    plan: Any, functions: dict, *, max_cases: int, seed: int, graph_targets: Any = None
) -> None:
    """One observation per function/check, not one Bayesian factor per fuzz input."""
    from .testgen import _HELPERS, _type_spec

    chunks = [plan.source or _HELPERS, FUZZ_IMPORTS]
    base_count = len(plan.cases)
    for name, fn in functions.items():
        if (
            fn.decorator_list
            or fn.args.vararg
            or any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn))
        ):
            continue
        params = [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]
        defaults = (
            dict(
                zip(
                    [p.arg for p in [*fn.args.posonlyargs, *fn.args.args]][
                        -len(fn.args.defaults) :
                    ],
                    fn.args.defaults,
                )
            )
            if fn.args.defaults
            else {}
        )
        defaults.update(
            {
                p.arg: d
                for p, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)
                if d is not None
            }
        )
        # Preserve callable/object defaults rather than replace them with random
        # scalar values. Trailing optional positional parameters can be omitted.
        omitted = set()
        for param in reversed([*fn.args.posonlyargs, *fn.args.args]):
            default = defaults.get(param.arg)
            if default is None or _type_spec(param.annotation) is not None:
                break
            try:
                ast.literal_eval(default)
                break
            except (ValueError, TypeError):
                omitted.add(param.arg)
        params = [p for p in params if p.arg not in omitted]
        specs = [_type_spec(p.annotation) for p in params]
        inferred = []
        for i, param in enumerate(params):
            if specs[i] is None and any(
                isinstance(n, ast.BinOp)
                and any(
                    isinstance(a, ast.Name) and a.id == param.arg for a in ast.walk(n)
                )
                and any(
                    isinstance(a, ast.Constant) and type(a.value) in {int, float}
                    for a in ast.walk(n)
                )
                for n in ast.walk(fn)
            ):
                specs[i] = "int"
                inferred.append(param.arg)
        strategies = [strategy(spec) for spec in specs]
        if any(s is None for s in strategies):
            # Usage inference only establishes a hypothesis, never an input contract.
            strategies = [
                s
                or "_st.one_of(_st.integers(-100,100), _st.text(max_size=20), _st.lists(_st.integers(-100,100), max_size=10))"
                for s in strategies
            ]
        positional_count = sum(
            p in fn.args.posonlyargs or p in fn.args.args for p in params
        )
        tuple_strategy = f"_st.tuples({', '.join(str(s) for s in strategies)})"
        return_spec = _type_spec(fn.returns)
        directed = (
            [t for t in graph_targets.targets if t["function"] == name]
            if graph_targets
            else []
        )
        # Reserve separate provenance: probes remain neutral even when they crash.
        property_names = list(
            dict.fromkeys(
                c["property"]
                for c in plan.cases
                if c.get("function") == name and "property" in c
            )
        )
        checks = (
            ([("annotation", None)] if return_spec else [])
            + [("contract", p) for p in property_names]
            + [("probe", None)]
        )
        for oracle, property_name in checks:
            index = len(plan.cases)
            test_name = f"test_pcg_{index:03}"
            scope = "property" if property_name else oracle
            code = f"\n@_pcg_pytest.mark.pcg_oracle({oracle!r}, scope={scope!r})\n@_seed({seed})\n@_settings(max_examples=80, deadline=None, database=None, phases=(_Phase.explicit, _Phase.generate, _Phase.target, _Phase.shrink), suppress_health_check=(_HealthCheck.too_slow,))\n"
            if params and all(spec in ("int", "float") for spec in specs):
                for values in branch_inputs(fn, [p.arg for p in params]):
                    code += f"@_example(values={values!r})\n"
            for target in directed:
                values = tuple(target["args"]) + tuple(
                    target["kwargs"][p.arg] for p in fn.args.kwonlyargs
                )
                code += f"@_example(values={values!r})\n"
            code += f"@_given(values={tuple_strategy})\ndef {test_name}(values):\n    from candidate import {name} as _pcg_fn\n    args = list(values[:{positional_count}])\n    kwargs = dict(zip({[p.arg for p in fn.args.kwonlyargs]!r}, values[{positional_count}:]))\n"
            if oracle != "probe":
                check = f"assert _pcg_type(value, {return_spec!r}), f'return annotation violated: inputs={{values!r}}, output={{value!r}}'"
                if property_name:
                    check = {
                        "nonnegative": "assert value >= 0",
                        "length_preserving": "assert len(value) == len(input_value)",
                        "sorted": "assert list(value) == sorted(value)",
                        "permutation": "assert sorted(value) == sorted(input_value)",
                        "input_unchanged": "assert (args, kwargs) == original",
                        "idempotent": "assert _pcg_fn(_pcg_copy.deepcopy(value)) == value",
                    }[property_name]
                code += (
                    "    original = _pcg_copy.deepcopy((args, kwargs))\n    input_value = original[0][0] if original[0] else next(iter(original[1].values()), None)\n    try:\n        value = _pcg_fn(*args, **kwargs)\n    except Exception:\n        return\n    accepted.append(True)\n    "
                    + check
                    + "\n"
                )
            else:
                code += "    _pcg_fn(*args, **kwargs)\n"
            if oracle != "probe":
                # An entirely rejecting input domain supplies no passing oracle evidence.
                marker, rest = code.split("\n@_seed", 1)
                rest = "@_seed" + rest
                rest = rest.replace(f"def {test_name}(values):", "def check(values):")
                code = (
                    marker
                    + f"\ndef {test_name}():\n    accepted = []\n"
                    + "\n".join("    " + line for line in rest.splitlines())
                    + "\n    check()\n    if not accepted:\n        _pcg_pytest.skip('no generated inputs accepted; no output evidence')\n"
                )
            chunks.append(code)
            plan.cases.append(
                {
                    "name": test_name,
                    "function": name,
                    "oracle": oracle,
                    "oracle_scope": scope,
                    "property": property_name,
                    "method": "property-fuzzing",
                    "max_examples": 80,
                    "shrinking": True,
                    "branch_examples": branch_inputs(fn, [p.arg for p in params])
                    if params and all(spec in ("int", "float") for spec in specs)
                    else [],
                    "input_basis": "annotations or explicitly uncertain usage hypotheses",
                    "inferred_numeric_parameters": inferred,
                    "graph_targets": directed,
                }
            )
    batches: dict[str, list[tuple[str, dict]]] = {}
    for code, case in zip(chunks[2:], plan.cases[base_count:]):
        batches.setdefault(case["function"], []).append((code, case))
    plan.cases = plan.cases[:base_count]
    selected = chunks[:2]
    for i in range(max((len(batch) for batch in batches.values()), default=0)):
        for batch in batches.values():
            if i >= len(batch) or len(plan.cases) >= max_cases:
                continue
            code, case = batch[i]
            new_name = f"test_pcg_{len(plan.cases):03}"
            selected.append(code.replace(case["name"], new_name))
            case["name"] = new_name
            plan.cases.append(case)
    plan.source = "\n".join(selected)
    plan.limits.update(
        {
            "fuzz_examples_per_check": 80,
            "shrinking": "Hypothesis bounded by worker timeout",
            "automatic_oracles": "declared return types and literal doctests; no inferred business intent",
        }
    )


def add_sequence_probes(
    plan: Any, source: str, *, max_cases: int, seed: int, graph_guided: bool = True
) -> None:
    """Explore supported object lifecycles without inventing state invariants."""
    from .testgen import _type_spec

    chunks = [plan.source]
    for cls in (n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)):
        if len(plan.cases) >= max_cases:
            break
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
        init = next((n for n in methods if n.name == "__init__"), None)
        if (
            cls.bases
            or cls.decorator_list
            or (
                init
                and (
                    len(init.args.args) - 1 > len(init.args.defaults)
                    or init.args.posonlyargs
                    or init.args.kwonlyargs
                    or init.args.vararg
                    or init.args.kwarg
                )
            )
        ):
            plan.skipped.append(
                {
                    "function": cls.name,
                    "reason": "object construction requires unsupported inputs or inheritance",
                }
            )
            continue
        alternatives = []
        method_names = []
        for method in methods:
            if (
                method.name.startswith("_")
                or method.decorator_list
                or method.args.posonlyargs
                or method.args.kwonlyargs
                or method.args.vararg
                or method.args.kwarg
            ):
                continue
            if not method.args.args or method.args.args[0].arg != "self":
                continue
            specs = [strategy(_type_spec(p.annotation)) for p in method.args.args[1:]]
            if any(s is None for s in specs):
                continue
            alternatives.append(
                f"_st.tuples(_st.just({method.name!r}), _st.tuples({', '.join(str(s) for s in specs)}))"
            )
            method_names.append(method.name)
        if not alternatives:
            plan.skipped.append(
                {
                    "function": cls.name,
                    "reason": "no supported public synchronous methods",
                }
            )
            continue
        index = len(plan.cases)
        name = f"test_pcg_{index:03}"
        state_metadata = None
        examples = ""
        if graph_guided:
            from .state_testing import sequence_seeds

            seeds, state_metadata = sequence_seeds(
                {m.name: m for m in methods}, method_names
            )
            examples = "".join(f"@_example(steps={steps!r})\n" for steps in seeds)
        chunks.append(
            f"\n@_pcg_pytest.mark.pcg_oracle('probe', scope='probe')\n@_seed({seed})\n@_settings(max_examples=60, deadline=None, database=None, suppress_health_check=(_HealthCheck.too_slow,))\n"
            + examples
            + f"@_given(steps=_st.lists(_st.one_of({', '.join(alternatives)}), min_size=1, max_size=12))\ndef {name}(steps):\n    from candidate import {cls.name} as _pcg_cls\n    obj = _pcg_cls()\n    for method, args in steps:\n        getattr(obj, method)(*args)\n"
        )
        plan.cases.append(
            {
                "name": name,
                "function": cls.name,
                "oracle": "probe",
                "oracle_scope": "probe",
                "method": "stateful-sequence-probe",
                "methods": method_names,
                "max_steps": 12,
                "max_examples": 60,
                "shrinking": True,
                "state_dependency_graph": state_metadata,
            }
        )
    plan.source = "\n".join(chunks)
