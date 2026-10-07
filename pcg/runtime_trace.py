"""Standalone worker instrumentation: bounded execution and value provenance.

Copied to the worker as pcg_worker_trace. No candidate runs in the planner.
AST instrumentation preserves source locations and never evaluates predicates twice.
"""

from __future__ import annotations

import ast
from collections import deque
import importlib.machinery
import math
import sys
from pathlib import Path
from typing import Any

_BaseException = BaseException
_test = "<collection>"
_counter = 0
_call_counter = 0
_frames: dict[int, dict] = {}
_definitions: dict[tuple, int] = {}
_events: dict[int, dict] = {}
_event_order: Any = deque()
_test_events: dict[str, Any] = {}
_roots: dict[str, Any] = {}
_truncated: set[str] = set()


def _safe(value: Any, depth: int = 0) -> Any:
    if value is None or type(value) in (int, bool):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"type": "non-finite-float"}
    if type(value) is str:
        return (
            value
            if len(value) <= 128
            else {"type": "str", "prefix": value[:128], "truncated": True}
        )
    if depth < 2 and type(value) in (list, tuple):
        return {
            "type": type(value).__name__,
            "items": [_safe(v, depth + 1) for v in value[:16]],
            "length": len(value),
        }
    return {"type": "opaque", "object_id": id(value)}


def start_test(name: str) -> None:
    global _test
    _test = name


def _identity(frame: Any, name: str, file: str) -> tuple:
    if frame.f_code.co_name == "<module>" or name not in frame.f_locals:
        return ("global", file, name)
    return ("local", id(frame), name)


def _object(value: Any) -> tuple | None:
    return (
        ("object", id(value))
        if type(value) not in (int, float, bool, str, bytes, tuple, type(None))
        else None
    )


def _record(
    frame: Any, file: str, line: int, kind: str, meta: dict, *, value: Any = None
) -> int:
    global _counter
    if len(_definitions) > 50000:
        _definitions.clear()
        _truncated.add(_test)
    if len(_events) >= 20000:
        _truncated.add(_test)
        _events.pop(_event_order.popleft(), None)
    _counter += 1
    state = _frames.get(id(frame), {})
    deps = set()
    for name in meta.get("reads", []):
        key = _identity(frame, name, file)
        deps.add(_definitions.get(key, 0))
        values = frame.f_locals if name in frame.f_locals else frame.f_globals
        object_key = _object(values[name]) if name in values else None
        if name not in {"self", "cls"} and object_key is not None:
            deps.add(_definitions.get(object_key, 0))
    for base, attr in meta.get("read_fields", []):
        if base in frame.f_locals:
            deps.add(_definitions.get(("field", id(frame.f_locals[base]), attr), 0))
    for control in meta.get("controls", []):
        deps.add(state.get("before", {}).get(control, 0))
    if kind in {"after", "return", "exception", "exit"}:
        deps.add(state.get("before", {}).get(line, 0))
        deps.update(state.get("children", []))
    if kind == "enter":
        caller = state.get("parent")
        if caller in _frames:
            deps.add(_frames[caller].get("current", 0))
    event = {
        "id": _counter,
        "file": file,
        "line": line,
        "scope": meta.get("scope", state.get("scope", "<module>")),
        "call": state.get("call", 0),
        "root": state.get("root", 0),
        "kind": kind,
        "dependencies": sorted(d for d in deps if d)[-32:],
    }
    if kind in {"return", "enter"}:
        event["value"] = _safe(value)
    if kind == "exception":
        event["exception"] = type(value).__name__
    _events[_counter] = event
    _event_order.append(_counter)
    _test_events.setdefault(_test, deque(maxlen=20000)).append(_counter)
    for name in meta.get("writes", []):
        _definitions[_identity(frame, name, file)] = _counter
        object_key = _object(frame.f_locals[name]) if name in frame.f_locals else None
        if kind != "enter" and object_key is not None:
            _definitions[object_key] = _counter
    for base, attr in meta.get("write_fields", []):
        if base in frame.f_locals:
            obj = frame.f_locals[base]
            _definitions[("field", id(obj), attr)] = _counter
            _definitions[("object", id(obj))] = _counter
    if state:
        state["current"] = _counter
    return _counter


def enter(file: str, line: int, scope: str, params: list[str]) -> None:
    global _call_counter
    frame = sys._getframe(1)
    parent_frame = frame.f_back
    while parent_frame and id(parent_frame) not in _frames:
        parent_frame = parent_frame.f_back
    parent = id(parent_frame) if parent_frame else None
    _call_counter += 1
    root = _frames[parent]["root"] if parent in _frames else _call_counter
    _frames[id(frame)] = {
        "call": _call_counter,
        "root": root,
        "parent": parent,
        "scope": scope,
        "before": {},
        "children": [],
    }
    for key in [key for key in _definitions if key[:2] == ("local", id(frame))]:
        del _definitions[key]
    inputs = {p: _safe(frame.f_locals[p]) for p in params if p in frame.f_locals}
    eid = _record(
        frame,
        file,
        line,
        "enter",
        {"scope": scope, "reads": params, "writes": params},
        value=inputs,
    )
    if parent is None:
        _roots.setdefault(_test, deque(maxlen=100)).append(
            {
                "root": root,
                "entry": eid,
                "function": scope,
                "file": file,
                "inputs": inputs,
            }
        )


def before(file: str, line: int, meta: dict) -> None:
    frame = sys._getframe(1)
    if id(frame) not in _frames:
        _frames[id(frame)] = {
            "call": 0,
            "root": 0,
            "scope": "<module>",
            "before": {},
            "children": [],
        }
    state = _frames[id(frame)]
    state["children"] = []
    state["before"][line] = _record(
        frame, file, line, "before", {**meta, "writes": [], "write_fields": []}
    )


def after(file: str, line: int, meta: dict) -> None:
    _record(sys._getframe(1), file, line, "after", meta)


def returned(file: str, line: int, value: Any) -> Any:
    frame = sys._getframe(1)
    eid = _record(frame, file, line, "return", {}, value=value)
    state = _frames.get(id(frame), {})
    parent = state.get("parent")
    if parent in _frames:
        _frames[parent]["children"].append(eid)
    return value


def failed(file: str, line: int, exception: BaseException) -> None:
    frame = sys._getframe(1)
    state = _frames.get(id(frame), {})
    actual = _events.get(state.get("current", 0), {}).get("line", line)
    eid = _record(frame, file, actual, "exception", {}, value=exception)
    state["failure"] = eid
    parent = state.get("parent")
    if parent in _frames:
        _frames[parent]["children"].append(eid)


def exit(file: str, line: int) -> None:
    frame = sys._getframe(1)
    _record(frame, file, line, "exit", {})
    _frames.pop(id(frame), None)


def traces(name: str, budget: int = 600, max_roots: int = 4) -> list[dict]:
    roots = list(_roots.get(name, []))[-max_roots:]
    output = []
    for root in roots:
        own = [
            e
            for e in _test_events.get(name, [])
            if e in _events and _events[e]["root"] == root["root"]
        ]
        seeds = [e for e in own if _events[e]["kind"] in {"exception", "return"}][
            -4:
        ] or own[-1:]
        reached: set[int] = set()
        stack = list(seeds)
        while stack and len(reached) < max(20, budget // max(1, len(roots))):
            eid = stack.pop()
            if eid in reached or eid not in _events:
                continue
            reached.add(eid)
            stack.extend(_events[eid]["dependencies"])
        output.append(
            {
                **root,
                "events": [_events[e] for e in sorted(reached)],
                "truncated": bool(stack)
                or name in _truncated
                or any(
                    d not in _events
                    for e in reached
                    for d in _events[e]["dependencies"]
                ),
                "interpretation": "instrumented executed dependencies; opaque values and object aliases are conservative",
            }
        )
    return output


def _meta(stmt: ast.stmt, scope: str, controls: list[int]) -> dict:
    children: list[ast.AST] = [stmt]
    if isinstance(stmt, (ast.If, ast.While)):
        children = [stmt.test]
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        children = [stmt.target, stmt.iter]
    elif isinstance(
        stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Try)
    ):
        children = []
    atoms = [n for child in children for n in ast.walk(child)]
    reads = {
        n.id for n in atoms if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    writes = {
        n.id
        for n in atoms
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
    }
    read_fields: set[tuple[str, str]] = set()
    write_fields: set[tuple[str, str]] = set()
    for n in atoms:
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            (
                write_fields if isinstance(n.ctx, (ast.Store, ast.Del)) else read_fields
            ).add((n.value.id, n.attr))
        if isinstance(n, ast.AugAssign):
            reads.update(a.id for a in ast.walk(n.target) if isinstance(a, ast.Name))
            if isinstance(n.target, ast.Attribute) and isinstance(
                n.target.value, ast.Name
            ):
                read_fields.add((n.target.value.id, n.target.attr))
        if isinstance(n, ast.Subscript) and isinstance(n.ctx, (ast.Store, ast.Del)):
            writes.update(a.id for a in ast.walk(n.value) if isinstance(a, ast.Name))
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr
            in {
                "append",
                "extend",
                "update",
                "pop",
                "clear",
                "add",
                "remove",
                "sort",
                "reverse",
                "insert",
                "discard",
            }
        ):
            base = n.func.value
            if isinstance(base, ast.Name):
                writes.add(base.id)
            elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                write_fields.add((base.value.id, base.attr))
    return {
        "scope": scope,
        "controls": controls,
        "reads": sorted(reads),
        "writes": sorted(writes),
        "read_fields": sorted(read_fields),
        "write_fields": sorted(write_fields),
    }


def instrument(source: str, file: str) -> ast.Module:
    tree = ast.parse(source, filename=file)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    alias = "_pcg_runtime_trace"
    while alias in names:
        alias += "_"

    def invoke(method: str, line: int, args: list[ast.expr]) -> ast.expr:
        call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=alias, ctx=ast.Load()), attr=method, ctx=ast.Load()
            ),
            args=[ast.Constant(file), ast.Constant(line), *args],
            keywords=[],
        )
        for node in ast.walk(call):
            if isinstance(node, (ast.expr, ast.stmt)) and not hasattr(node, "lineno"):
                node.lineno = node.end_lineno = line
                node.col_offset = node.end_col_offset = 0
        return call

    def literal(value: Any) -> ast.expr:
        expression = ast.parse(repr(value), mode="eval").body
        # Metadata inherits the statement location, preventing fabricated line-1 coverage.
        for node in ast.walk(expression):
            for attribute in ("lineno", "end_lineno", "col_offset", "end_col_offset"):
                if hasattr(node, attribute):
                    delattr(node, attribute)
        return expression

    def body(nodes: list[ast.stmt], scope: str, controls: list[int]) -> list[ast.stmt]:
        output: list[ast.stmt] = []
        for index, stmt in enumerate(nodes):
            if (
                index == 0
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                output.append(stmt)
                continue
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
                output.append(stmt)
                continue
            meta = _meta(stmt, scope, controls)
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = scope + "." + stmt.name if scope != "<module>" else stmt.name
                original = body(stmt.body, qualname, [])
                doc = original[:1] if ast.get_docstring(stmt) else []
                remaining = original[len(doc) :]
                params = [
                    a.arg
                    for a in [
                        *stmt.args.posonlyargs,
                        *stmt.args.args,
                        *stmt.args.kwonlyargs,
                    ]
                ]
                start = ast.Expr(
                    invoke(
                        "enter", stmt.lineno, [ast.Constant(qualname), literal(params)]
                    )
                )
                excname = alias + "_exception"
                handler = ast.ExceptHandler(
                    type=ast.Attribute(
                        value=ast.Name(id=alias, ctx=ast.Load()),
                        attr="_BaseException",
                        ctx=ast.Load(),
                    ),
                    name=excname,
                    body=[
                        ast.Expr(
                            invoke(
                                "failed",
                                stmt.lineno,
                                [ast.Name(id=excname, ctx=ast.Load())],
                            )
                        ),
                        ast.Raise(),
                    ],
                )
                guarded = ast.Try(
                    body=remaining,
                    handlers=[handler],
                    orelse=[],
                    finalbody=[ast.Expr(invoke("exit", stmt.lineno, []))],
                )
                ast.copy_location(start, stmt)
                start.end_lineno = start.lineno
                ast.copy_location(guarded, stmt)
                stmt.body = [*doc, start, guarded]
            elif isinstance(stmt, ast.ClassDef):
                stmt.body = body(stmt.body, stmt.name, [])
                stmt.body.append(
                    ast.copy_location(ast.Expr(invoke("exit", stmt.lineno, [])), stmt)
                )
            else:
                for field in ("body", "orelse", "finalbody"):
                    nested = getattr(stmt, field, None)
                    if isinstance(nested, list):
                        setattr(
                            stmt, field, body(nested, scope, controls + [stmt.lineno])
                        )
                for handler in getattr(stmt, "handlers", []):
                    handler.body = body(handler.body, scope, controls + [stmt.lineno])
                if isinstance(stmt, ast.Return):
                    stmt.value = invoke(
                        "returned", stmt.lineno, [stmt.value or ast.Constant(None)]
                    )
            before_node = ast.Expr(invoke("before", stmt.lineno, [literal(meta)]))
            ast.copy_location(before_node, stmt)
            before_node.end_lineno = before_node.lineno
            output.extend([before_node, stmt])
            if meta["writes"] or meta["write_fields"]:
                after_node = ast.Expr(invoke("after", stmt.lineno, [literal(meta)]))
                ast.copy_location(after_node, stmt)
                after_node.end_lineno = after_node.lineno
                output.append(after_node)
        return output

    tree.body = body(tree.body, "<module>", [])
    tree.body.append(ast.copy_location(ast.Expr(invoke("exit", 1, [])), tree))
    insertion = 1 if ast.get_docstring(tree) else 0
    while (
        insertion < len(tree.body)
        and isinstance((future := tree.body[insertion]), ast.ImportFrom)
        and future.module == "__future__"
    ):
        insertion += 1
    tree.body.insert(
        insertion, ast.Import(names=[ast.alias(name="pcg_worker_trace", asname=alias)])
    )
    return ast.fix_missing_locations(tree)


def install(root: Path, sources: list[str]) -> None:
    allowed = {(root / name).resolve(): name for name in sources}

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname: str) -> Any:
            path = Path(self.path).resolve()
            return compile(
                instrument(path.read_text(encoding="utf-8"), allowed[path]),
                str(path),
                "exec",
            )

    class Finder:
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and Path(spec.origin).resolve() in allowed:
                spec.loader = Loader(fullname, spec.origin)
                return spec
            return None

    sys.meta_path.insert(0, Finder())
