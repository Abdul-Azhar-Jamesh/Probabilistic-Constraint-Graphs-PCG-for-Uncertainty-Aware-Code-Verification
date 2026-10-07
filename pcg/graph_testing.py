"""Bounded dependency-sliced CFG path solving for directed test generation.

This planner evaluates supported AST expressions as Z3 terms, never Python code.
Solutions are input hypotheses: only worker execution verifies path reachability.
"""

from __future__ import annotations

import ast
import threading
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import z3

from .slicing import Dependence

_SOLVER_LOCK = threading.Lock()


class Unsupported(ValueError):
    pass


@dataclass
class SequenceLength:
    value: Any


@dataclass
class GraphTargets:
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    targets: list[dict] = field(default_factory=list)
    unresolved: list[dict] = field(default_factory=list)
    limits: dict = field(
        default_factory=lambda: {
            "paths_per_function": 32,
            "path_nodes": 64,
            "solver_timeout_ms": 150,
            "integer_range": [-10000, 10000],
            "container_length": 20,
            "string_length": 40,
            "module_paths": 128,
            "walk_states_per_function": 512,
            "module_walk_states": 2048,
            "coverage_feedback_checks": 6,
        }
    )

    def to_dict(self) -> dict:
        return {
            "nodes": [{"line": n, **d} for n, d in self.graph.nodes(data=True)],
            "edges": [
                {"from": a, "to": b, **d} for a, b, d in self.graph.edges(data=True)
            ],
            "targets": self.targets,
            "unresolved": self.unresolved,
            "limits": self.limits,
        }


def _cfg(
    functions: dict[str, ast.FunctionDef],
) -> tuple[nx.DiGraph, dict[int, ast.stmt]]:
    graph: nx.DiGraph = nx.DiGraph()
    statements = {}
    for name, fn in functions.items():
        exit_node = -fn.lineno
        graph.add_node(exit_node, kind="exit", scope=name)

        def wire(body: list[ast.stmt], following: int) -> int:
            cursor = following
            for stmt in reversed(body):
                ln = stmt.lineno
                statements[ln] = stmt
                graph.add_node(ln, kind=type(stmt).__name__, scope=name)
                if isinstance(stmt, ast.If):
                    yes = wire(stmt.body, cursor)
                    no = wire(stmt.orelse, cursor)
                    # Empty branches reaching the same node cannot represent both
                    # outcomes in DiGraph; retain an explicit unsupported reason.
                    if yes == no:
                        graph.nodes[ln]["unsupported"] = (
                            "empty indistinguishable branches"
                        )
                        graph.add_edge(ln, yes)
                    else:
                        graph.add_edge(ln, yes, branch=True)
                        graph.add_edge(ln, no, branch=False)
                elif isinstance(stmt, (ast.Return, ast.Raise)):
                    graph.add_edge(ln, exit_node)
                elif isinstance(
                    stmt, (ast.For, ast.While, ast.Try, ast.With, ast.Match)
                ):
                    graph.nodes[ln]["unsupported"] = (
                        "compound flow needs bounded dynamic exploration"
                    )
                    graph.add_edge(ln, cursor)
                else:
                    graph.add_edge(ln, cursor)
                cursor = ln
            return cursor

        first = wire(fn.body, exit_node)
        graph.add_node(fn.lineno, kind="entry", scope=name)
        graph.add_edge(fn.lineno, first)
    return graph, statements


def _truth(value: Any) -> Any:
    if isinstance(value, SequenceLength):
        return value.value != 0
    if z3.is_bool(value):
        return value
    if z3.is_int(value):
        return value != 0
    if z3.is_string(value):
        return z3.Length(value) != 0
    raise Unsupported("unsupported predicate type")


def _expr(node: ast.AST, env: dict, functions: dict, depth: int = 0) -> Any:
    if isinstance(node, ast.Constant):
        if type(node.value) is int:
            return z3.IntVal(node.value)
        if type(node.value) is bool:
            return z3.BoolVal(node.value)
        if isinstance(node.value, str):
            return z3.StringVal(node.value)
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _expr(node.operand, env, functions, depth)
        if isinstance(node.op, ast.Not):
            return z3.Not(_truth(value))
        if isinstance(node.op, ast.USub) and z3.is_int(value):
            return -value
        if isinstance(node.op, ast.UAdd) and z3.is_int(value):
            return value
    if isinstance(node, ast.BoolOp):
        values = [_truth(_expr(v, env, functions, depth)) for v in node.values]
        return z3.And(*values) if isinstance(node.op, ast.And) else z3.Or(*values)
    if isinstance(node, ast.BinOp):
        a, b = (
            _expr(node.left, env, functions, depth),
            _expr(node.right, env, functions, depth),
        )
        if z3.is_int(a) and z3.is_int(b):
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if (
                isinstance(node.op, ast.Mod)
                and isinstance(node.right, ast.Constant)
                and type(node.right.value) is int
                and node.right.value > 0
            ):
                return a % b
        if isinstance(node.op, ast.Add) and z3.is_string(a) and z3.is_string(b):
            return z3.Concat(a, b)
    if isinstance(node, ast.Compare):
        values = [
            _expr(n, env, functions, depth) for n in [node.left, *node.comparators]
        ]
        terms = []
        for a, op, b in zip(values, node.ops, values[1:]):
            if isinstance(op, (ast.Eq, ast.NotEq)) and a.sort() == b.sort():
                terms.append(a == b if isinstance(op, ast.Eq) else a != b)
            elif z3.is_int(a) and z3.is_int(b):
                if isinstance(op, ast.Lt):
                    terms.append(a < b)
                elif isinstance(op, ast.LtE):
                    terms.append(a <= b)
                elif isinstance(op, ast.Gt):
                    terms.append(a > b)
                elif isinstance(op, ast.GtE):
                    terms.append(a >= b)
                else:
                    raise Unsupported("unsupported comparison")
            else:
                raise Unsupported("unsupported comparison sorts")
        return z3.And(*terms)
    if isinstance(node, ast.Call) and not node.keywords:
        args = [_expr(n, env, functions, depth) for n in node.args]
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if (
                name == "len"
                and name not in functions
                and name not in env
                and len(args) == 1
            ):
                if isinstance(args[0], SequenceLength):
                    return args[0].value
                if z3.is_string(args[0]):
                    return z3.Length(args[0])
            if (
                name == "abs"
                and name not in functions
                and name not in env
                and len(args) == 1
                and z3.is_int(args[0])
            ):
                return z3.If(args[0] >= 0, args[0], -args[0])
            # Small pure straight-line helper summaries. No import or execution.
            helper = functions.get(name) if name not in env else None
            if (
                helper
                and not helper.decorator_list
                and depth < 2
                and not helper.args.defaults
                and not helper.args.kwonlyargs
                and not helper.args.posonlyargs
                and not helper.args.vararg
                and not helper.args.kwarg
                and len(args) == len(helper.args.args)
            ):
                local = dict(zip([p.arg for p in helper.args.args], args))
                for stmt in helper.body:
                    if (
                        isinstance(stmt, ast.Assign)
                        and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                    ):
                        local[stmt.targets[0].id] = _expr(
                            stmt.value, local, functions, depth + 1
                        )
                    elif isinstance(stmt, ast.Return) and stmt.value:
                        return _expr(stmt.value, local, functions, depth + 1)
                    elif (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)
                    ):
                        continue
                    else:
                        raise Unsupported("helper is not a supported pure expression")
        if isinstance(node.func, ast.Attribute) and len(args) == 1:
            receiver = _expr(node.func.value, env, functions, depth)
            if z3.is_string(receiver) and z3.is_string(args[0]):
                if node.func.attr == "startswith":
                    return z3.PrefixOf(args[0], receiver)
                if node.func.attr == "endswith":
                    return z3.SuffixOf(args[0], receiver)
    raise Unsupported(f"unsupported expression: {type(node).__name__}")


def _input_symbols(fn: ast.FunctionDef) -> tuple[dict, list, dict]:
    from .testgen import _type_spec

    env, bounds, kinds = {}, [], {}
    for param in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]:
        spec = _type_spec(param.annotation)
        # An unannotated scalar is an explicit input hypothesis, not a contract.
        kind = "int" if spec is None else spec
        if kind == "int":
            value = z3.Int(f"{fn.name}:{param.arg}")
            bounds += [value >= -10000, value <= 10000]
        elif kind == "bool":
            value = z3.Bool(f"{fn.name}:{param.arg}")
        elif kind == "str":
            value = z3.String(f"{fn.name}:{param.arg}")
            bounds += [z3.Length(value) <= 40]
        elif kind == "list" or isinstance(kind, list) and kind == ["list", "int"]:
            value = SequenceLength(z3.Int(f"{fn.name}:{param.arg}:length"))
            bounds += [value.value >= 0, value.value <= 20]
            kind = "list"
        else:
            raise Unsupported(f"unsupported input annotation for {param.arg}")
        env[param.arg], kinds[param.arg] = value, kind
    return env, bounds, kinds


def plan_graph_targets(source: str, dependence: Dependence) -> GraphTargets:
    """Serialize default Z3 context use across concurrent dashboard sessions."""
    with _SOLVER_LOCK:
        return _plan_graph_targets(source, dependence)


def _plan_graph_targets(source: str, dependence: Dependence) -> GraphTargets:
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    result = GraphTargets()
    result.graph, statements = _cfg(functions)
    path_budget = 128
    walk_budget = 2048
    for name, fn in functions.items():
        if (
            fn.decorator_list
            or fn.args.vararg
            or fn.args.kwarg
            or any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn))
            or any(s.lineno == fn.lineno for s in fn.body)
            or len([n.lineno for n in ast.walk(fn) if isinstance(n, ast.stmt)])
            != len({n.lineno for n in ast.walk(fn) if isinstance(n, ast.stmt)})
        ):
            result.unresolved.append(
                {
                    "function": name,
                    "reason": "unsupported interface or same-line statements",
                }
            )
            continue
        explored = 0
        walked = 0
        stack = [(fn.lineno, [fn.lineno])]
        while (
            stack
            and explored < 32
            and path_budget > 0
            and walked < 512
            and walk_budget > 0
        ):
            current, path = stack.pop()
            walked += 1
            walk_budget -= 1
            if len(path) > 64:
                result.unresolved.append(
                    {"function": name, "reason": "path length budget"}
                )
                continue
            if current != -fn.lineno:
                for nxt in reversed(list(result.graph.successors(current))):
                    if nxt not in path:
                        stack.append((nxt, path + [nxt]))
                continue
            explored += 1
            path_budget -= 1
            choices = [
                (a, b, result.graph[a][b]["branch"])
                for a, b in zip(path, path[1:])
                if "branch" in result.graph[a][b]
            ]
            if not choices:
                continue
            relevant = set(dependence.backward({a for a, _, _ in choices}))
            try:
                env, bounds, kinds = _input_symbols(fn)
                inputs = dict(env)
                for module_statement in tree.body:
                    if isinstance(module_statement, ast.Assign):
                        for binding in module_statement.targets:
                            if (
                                isinstance(binding, ast.Name)
                                and binding.id not in inputs
                            ):
                                env[binding.id] = None
                                try:
                                    env[binding.id] = _expr(
                                        module_statement.value, env, functions
                                    )
                                except (Unsupported, z3.Z3Exception):
                                    pass
                constraints = list(bounds)
                for ln in path[1:-1]:
                    node = statements[ln]
                    if result.graph.nodes[ln].get("unsupported"):
                        raise Unsupported(result.graph.nodes[ln]["unsupported"])
                    if ln in relevant:
                        if isinstance(node, (ast.Assign, ast.AnnAssign)):
                            targets = (
                                node.targets
                                if isinstance(node, ast.Assign)
                                else [node.target]
                            )
                            if node.value is None or any(
                                not isinstance(t, ast.Name) for t in targets
                            ):
                                raise Unsupported("non-scalar assignment")
                            value = _expr(node.value, env, functions)
                            for target in targets:
                                if isinstance(target, ast.Name):
                                    env[target.id] = value
                        elif isinstance(node, ast.AugAssign) and isinstance(
                            node.target, ast.Name
                        ):
                            expr = ast.BinOp(
                                left=ast.Name(id=node.target.id, ctx=ast.Load()),
                                op=node.op,
                                right=node.value,
                            )
                            env[node.target.id] = _expr(expr, env, functions)
                        elif isinstance(node, ast.Expr) and not isinstance(
                            node.value, ast.Constant
                        ):
                            raise Unsupported(
                                "relevant side effect requires dynamic value tracing"
                            )
                    if isinstance(node, ast.If):
                        decision = next(flag for a, _, flag in choices if a == ln)
                        condition = _truth(_expr(node.test, env, functions))
                        constraints.append(condition if decision else z3.Not(condition))
                solver = z3.Solver()
                solver.set(timeout=150, random_seed=7)
                solver.add(*constraints)
                status = solver.check()
                if status != z3.sat:
                    result.unresolved.append(
                        {
                            "function": name,
                            "path": path,
                            "reason": "infeasible within input bounds"
                            if status == z3.unsat
                            else "solver timeout or unknown",
                        }
                    )
                    continue
                model = solver.model()
                values = {}
                for param, symbol in inputs.items():
                    kind = kinds[param]
                    value = model.eval(
                        symbol.value if isinstance(symbol, SequenceLength) else symbol,
                        model_completion=True,
                    )
                    values[param] = (
                        [0] * value.as_long()
                        if kind == "list"
                        else value.as_string()
                        if kind == "str"
                        else z3.is_true(value)
                        if kind == "bool"
                        else value.as_long()
                    )
                positional = [*fn.args.posonlyargs, *fn.args.args]
                result.targets.append(
                    {
                        "function": name,
                        "args": [values[p.arg] for p in positional],
                        "kwargs": {p.arg: values[p.arg] for p in fn.args.kwonlyargs},
                        "path": path,
                        "branch_edges": [[a, b] for a, b, _ in choices],
                        "decisions": [
                            {"line": a, "outcome": flag} for a, _, flag in choices
                        ],
                        "dependency_slice": sorted(relevant),
                        "method": "dependency-sliced-cfg-smt",
                        "reachability": "predicted; requires execution",
                    }
                )
            except (
                Unsupported,
                z3.Z3Exception,
                AttributeError,
                TypeError,
                ValueError,
            ) as exc:
                result.unresolved.append(
                    {"function": name, "path": path, "reason": str(exc)[:200]}
                )
        if stack:
            result.unresolved.append(
                {"function": name, "reason": "path enumeration budget"}
            )
    return result


def add_uncovered_checks(
    plan: Any, targets: GraphTargets, execution: Any, source: str, *, max_cases: int
) -> dict:
    """Spend reserved slots on unseen graph edges after initial worker coverage.

    Direct replay explores branches hidden when fuzzing stops at its first failure.
    Only the final run supplies Bayesian observations.
    """
    from .testgen import _type_spec

    report: dict = {
        "rounds": 1,
        "added_checks": 0,
        "edges_before": 0,
        "reason": "execution unavailable or no remaining test budget",
    }
    if not execution or not execution.valid or len(plan.cases) >= max_cases:
        return report
    arcs = {tuple(a) for t in execution.tests for a in t.arcs}
    cases = {c["name"]: c for c in plan.cases}
    failed_functions = {
        cases[t.nodeid.rsplit("::", 1)[-1]]["function"]
        for t in execution.tests
        if t.oracle != "probe"
        and t.outcome == "failed"
        and cases.get(t.nodeid.rsplit("::", 1)[-1], {}).get("method")
        == "property-fuzzing"
    }
    wanted = [
        t
        for t in targets.targets
        if any(tuple(e) not in arcs for e in t["branch_edges"])
        or t["function"] in failed_functions
    ]
    edges = {tuple(e) for t in targets.targets for e in t["branch_edges"]}
    report["edges_before"] = len(edges & arcs)
    functions = {
        n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
    }
    seen = set()
    feedback_limit = min(max_cases, len(plan.cases) + 6)
    for target in sorted(wanted, key=lambda t: (-len(t["decisions"]), t["function"])):
        key = repr((target["function"], target["args"], target["kwargs"]))
        if key in seen:
            continue
        seen.add(key)
        name = target["function"]
        spec = _type_spec(functions[name].returns)
        for oracle in ["annotation", "probe"] if spec else ["probe"]:
            if len(plan.cases) >= feedback_limit:
                break
            test_name = f"test_pcg_{len(plan.cases):03}"
            code = f"\n@_pcg_pytest.mark.pcg_oracle({oracle!r}, scope={oracle!r})\ndef {test_name}():\n    from candidate import {name} as _pcg_fn\n    args = {target['args']!r}\n    kwargs = {target['kwargs']!r}\n"
            if oracle == "annotation":
                code += f"    try:\n        value = _pcg_fn(*args, **kwargs)\n    except Exception:\n        _pcg_pytest.skip('input acceptance unspecified; inspect paired probe')\n    assert _pcg_type(value, {spec!r}), f'graph-directed input={{args!r}}, kwargs={{kwargs!r}}, output={{value!r}}'\n"
            else:
                code += "    _pcg_fn(*args, **kwargs)\n"
            plan.source += code
            plan.cases.append(
                {
                    "name": test_name,
                    "function": name,
                    "oracle": oracle,
                    "oracle_scope": oracle,
                    "method": "coverage-directed-cfg-replay",
                    "target": target,
                    "input_basis": "SMT path hypothesis, validated by worker coverage",
                }
            )
            report["added_checks"] += 1
    report["rounds"] = 2 if report["added_checks"] else 1
    report["reason"] = (
        "reserved budget targets unseen CFG branches or isolates failed fuzz paths"
        if report["added_checks"]
        else "all predicted branch edges already covered"
    )
    return report


def measure_targets(targets: GraphTargets, execution: Any) -> dict:
    arcs = (
        {tuple(arc) for t in execution.tests for arc in t.arcs}
        if execution and execution.valid
        else set()
    )
    rows = []
    for target in targets.targets:
        edges = {tuple(edge) for edge in target["branch_edges"]}
        witnesses = (
            [t.nodeid for t in execution.tests if edges <= {tuple(a) for a in t.arcs}]
            if execution and execution.valid
            else []
        )
        rows.append(
            {
                **target,
                "covered_edges": sum(edge in arcs for edge in edges),
                "total_edges": len(edges),
                "all_edges_seen_in_one_test": bool(witnesses),
                "witness_tests": witnesses,
                "interpretation": "coverage is per-test union; a fuzz test may aggregate multiple input paths",
            }
        )
    return {
        "targets": rows,
        "distinct_branch_edges_seen": len(
            {tuple(e) for t in targets.targets for e in t["branch_edges"]} & arcs
        ),
        "unresolved": targets.unresolved,
        "limits": targets.limits,
    }
