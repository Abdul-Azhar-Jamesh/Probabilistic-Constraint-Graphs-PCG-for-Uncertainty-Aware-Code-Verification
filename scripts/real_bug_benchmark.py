"""Three-repository, pinned BugsInPy compatibility replays (not calibration)."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import importlib.metadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcg.execution import ExecutionConfig  # noqa: E402
from pcg.project import analyze_project, write_report  # noqa: E402


CASES = {
    "tqdm-1": {
        "repo": "tqdm/tqdm",
        "buggy": "8cc777fe8401a05d07f2c97e65d15e4460feab88",
        "fixed": "c0dcf39b046d1b4ff6de14ac99ad9a1b10487512",
        "source": "tqdm/contrib/__init__.py",
        "tests": "tqdm/tests/tests_contrib.py",
    },
    "httpie-1": {
        "repo": "httpie/httpie",
        "buggy": "001bda19450ad85c91345eea3cfa3991e1d492ba",
        "fixed": "5300b0b490b8db48fac30b5e32164be93dc574b7",
        "source": "httpie/downloads.py",
        "tests": "tests/test_downloads.py",
    },
    "luigi-1": {
        "repo": "spotify/luigi",
        "buggy": "1164eb6b85b8a70f596dbb99452bec513e72c12e",
        "fixed": "aec5dc2ed8db53fc282a0bd24aabe59031b6d1ba",
        "source": "luigi/server.py",
        "tests": "test/server_test.py",
    },
}


def fetch(case: dict, commit: str, file: str) -> str:
    with urllib.request.urlopen(
        f"https://raw.githubusercontent.com/{case['repo']}/{commit}/{file}", timeout=30
    ) as response:
        data = response.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError("benchmark download too large")
    return data.decode("utf-8")


def prepare(
    name: str, source: str, upstream: str
) -> tuple[str, str, str, str, list[dict]]:
    tree = ast.parse(source)
    tests = ast.parse(upstream)
    spans = []
    if name == "tqdm-1":
        fn = next(
            n
            for n in tests.body
            if isinstance(n, ast.FunctionDef) and n.name == "test_enumerate"
        )
        fn.decorator_list = []
        regression = (
            "from io import StringIO\nfrom contextlib import closing\nfrom contrib import tenumerate\n"
            + ast.unparse(fn)
        )
        return (
            "contrib.py",
            source,
            regression,
            "Exact changed module; unchanged upstream assertions. Historical Nose setup omitted and IO helpers replaced with stdlib. Installed tqdm supplies dependencies.",
            spans,
        )
    if name == "httpie-1":
        wanted = {
            "get_unique_filename",
            "trim_filename",
            "get_filename_max_length",
            "trim_filename_if_needed",
        }
        nodes = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted
        ]
        segments = []
        next_line = 3
        for node in nodes:
            text = ast.get_source_segment(source, node)
            assert text is not None
            segments.append(text)
            spans.append(
                {
                    "function": node.name,
                    "upstream_start": node.lineno,
                    "snapshot_start": next_line,
                }
            )
            next_line += len(text.splitlines()) + 1
        candidate = "import os\nimport errno\n" + "\n\n".join(segments)
        fn = next(
            n
            for n in ast.walk(tests)
            if isinstance(n, ast.FunctionDef) and n.name == "test_unique_filename"
        )
        fn.args.args = [p for p in fn.args.args if p.arg != "self"]
        for decorator in fn.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "patch"
            ):
                decorator.args[0] = ast.Constant("downloads.get_filename_max_length")
                decorator.keywords.append(
                    ast.keyword(arg="create", value=ast.Constant(True))
                )
        regression = (
            "import pytest\nfrom unittest import mock\nfrom downloads import get_unique_filename\n"
            + ast.unparse(ast.fix_missing_locations(fn))
        )
        return (
            "downloads.py",
            candidate,
            regression,
            "Exact upstream changed function bodies and helpers extracted. Original parametrized assertions preserved. stdlib mock replaces mock package; create=True allows fixed-test patch on buggy revision where helper did not exist. Filesystem length stub is upstream test behavior.",
            spans,
        )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MetricsHandler"
    )
    segment = ast.get_source_segment(source, cls)
    assert segment is not None
    candidate = "import tornado.web\n" + segment
    spans = [{"class": cls.name, "upstream_start": cls.lineno, "snapshot_start": 2}]
    test_cls = next(
        n
        for n in tests.body
        if isinstance(n, ast.ClassDef) and n.name == "MetricsHandlerTest"
    )

    class Rewrite(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            if ast.unparse(node) == "luigi.server.MetricsHandler":
                return ast.copy_location(
                    ast.Name(id="MetricsHandler", ctx=ast.Load()), node
                )
            return self.generic_visit(node)

    regression = (
        "import unittest\nimport tornado.web\nfrom unittest import mock\nfrom metrics import MetricsHandler\n"
        + ast.unparse(Rewrite().visit(test_cls))
    )
    return (
        "metrics.py",
        candidate,
        regression,
        "Exact upstream MetricsHandler class and regression TestCase extracted. Unchanged assertions, stdlib mock and installed Tornado; handler import points to extracted class. Other Luigi services omitted.",
        spans,
    )


def main() -> None:
    output = Path("out/real-bug-benchmark")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, case in CASES.items():
        try:
            upstream = fetch(case, case["fixed"], case["tests"])
            for version in ("buggy", "fixed"):
                source = fetch(case, case[version], case["source"])
                file, candidate, regression, adaptation, spans = prepare(
                    name, source, upstream
                )
                for mode in ("autonomous", "upstream-regression"):
                    folder = output / name / f"{version}-{mode}"
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / file).write_text(candidate, encoding="utf-8")
                    if mode == "upstream-regression":
                        (folder / "test_regression.py").write_text(
                            regression, encoding="utf-8"
                        )
                    started = time.monotonic()
                    report = analyze_project(
                        folder,
                        execution=ExecutionConfig("local", 30),
                        max_generated_cases=8,
                        minimize_trials=4,
                        validation_time_budget=30,
                    )
                    write_report(
                        report, str(folder / "report.json"), str(folder / "report.html")
                    )
                    outcomes = [
                        t
                        for t in report["execution"]["tests"]
                        if t["nodeid"].startswith("test_regression.py")
                    ]
                    rows.append(
                        {
                            "case": name,
                            "repository": case["repo"],
                            "version": version,
                            "mode": mode,
                            "commit": case[version],
                            "source_sha256": hashlib.sha256(
                                source.encode()
                            ).hexdigest(),
                            "regression_sha256": hashlib.sha256(
                                upstream.encode()
                            ).hexdigest(),
                            "adaptation": adaptation,
                            "source_spans": spans,
                            "status": report["status"],
                            "seconds": time.monotonic() - started,
                            **report["summary"],
                            "regression_outcomes": [
                                {"test": t["nodeid"], "outcome": t["outcome"]}
                                for t in outcomes
                            ],
                            "validation": report["failure_validation"],
                            "report": str(folder / "report.json"),
                        }
                    )
                    print(name, version, mode, report["status"], flush=True)
        except Exception as exc:
            rows.append({"case": name, "status": "unavailable", "details": str(exc)})
            print(name, "unavailable:", exc, flush=True)
    paired = []
    for name in CASES:
        versions = {
            r.get("version"): r
            for r in rows
            if r["case"] == name and r.get("mode") == "upstream-regression"
        }
        confirmed = (
            len(versions) == 2
            and any(
                t["outcome"] == "failed"
                for t in versions["buggy"]["regression_outcomes"]
            )
            and bool(versions["fixed"]["regression_outcomes"])
            and all(
                t["outcome"] == "passed"
                for t in versions["fixed"]["regression_outcomes"]
            )
        )
        paired.append({"case": name, "bug_fix_regression_confirmed": confirmed})
    packages = {}
    for name in ("pytest", "coverage", "hypothesis", "tqdm", "tornado"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    summary = {
        "benchmark": "BugsInPy three-repository compatibility replay",
        "rows": rows,
        "pairs": paired,
        "confirmed_pairs": sum(p["bug_fix_regression_confirmed"] for p in paired),
        "environment": {
            "python": sys.version,
            "packages": packages,
        },
        "implementation_sha256": {
            name: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "pcg" / name).read_bytes()
            ).hexdigest()
            for name in (
                "project.py",
                "failure_validation.py",
                "runtime_trace.py",
                "execution.py",
            )
        },
        "limitations": "Small selected compatibility replays, not original benchmark environments, a repository-held-out accuracy study, or calibration. Autonomous misses and probe exceptions are separate from confirmed regression failures. Function extraction narrows environment scope.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"confirmed_pairs": summary["confirmed_pairs"], "pairs": paired}, indent=2
        )
    )


if __name__ == "__main__":
    main()
