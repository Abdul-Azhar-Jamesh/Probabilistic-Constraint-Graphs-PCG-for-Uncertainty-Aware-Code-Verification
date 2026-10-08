"""Pinned BugsInPy compatibility replays with explicit detection metrics."""

from __future__ import annotations

import ast
import argparse
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
    "tqdm-2": {
        "repo": "tqdm/tqdm",
        "buggy": "bef86db56654d271838b145ad77f7040a73a7b4d",
        "fixed": "127af5caf19e7d29c346f5ca8a9c7ef3004b664b",
        "source": "tqdm/utils.py",
        "tests": "tqdm/tests/tests_tqdm.py",
    },
    "httpie-3": {
        "repo": "httpie/httpie",
        "buggy": "8c33e5e3d31d3cd6476c4d9bc963a4c529f883d2",
        "fixed": "589887939507ff26d36ec74bd2c045819cfa3d56",
        "source": "httpie/sessions.py",
        "tests": "tests/test_sessions.py",
    },
}

ROOTS = {
    "tqdm-1": "tenumerate",
    "tqdm-2": "disp_trim",
    "httpie-1": "get_unique_filename",
    "httpie-3": "update_headers",
    "luigi-1": "MetricsHandler.get",
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
    if name in {"tqdm-2", "httpie-3"}:
        wanted = (
            {"disp_trim", "disp_len", "_text_width"}
            if name == "tqdm-2"
            else {"update_headers"}
        )
        nodes = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in wanted
        ]
        if name == "tqdm-2":
            header = "import re\nfrom unicodedata import east_asian_width\nRE_ANSI = re.compile(r'\\x1b\\[[;\\d]*[A-Za-z]')\n"
            header += "_unicode = str\n"
            candidate = header + "\n\n".join(
                ast.get_source_segment(source, n) or "" for n in nodes
            )
            regression = (
                "from display import disp_trim\n"
                "def test_ansi_reset():\n"
                "    text = '\\x1b[31mabc\\x1b[0m'\n"
                "    assert disp_trim(text, 3) == text\n"
                "    assert disp_trim('abcd', 2) == 'ab'\n"
            )
            file = "display.py"
        else:
            # Exact changed method, with a minimal dictionary receiver. This
            # tests header persistence only, not the omitted network workflow.
            method = nodes[0]
            candidate = "SESSION_IGNORED_HEADER_PREFIXES = ['Content-', 'If-']\n" + (
                ast.get_source_segment(source, method) or ""
            )
            regression = (
                "from headers import update_headers\n"
                "def test_explicitly_unset_header():\n"
                "    session = {'headers': {}}\n"
                "    update_headers(session, {'X-Unset': None, 'X-Keep': b'value'})\n"
                "    assert session['headers'] == {'X-Keep': 'value'}\n"
            )
            file = "headers.py"
        parsed = ast.parse(candidate)
        for node in nodes:
            extracted = next(
                n
                for n in ast.walk(parsed)
                if isinstance(n, ast.FunctionDef) and n.name == node.name
            )
            spans.append(
                {
                    "function": node.name,
                    "upstream_start": node.lineno,
                    "snapshot_start": extracted.lineno,
                }
            )
        return (
            file,
            candidate,
            regression,
            (
                "Exact changed function/method bodies extracted; independent regression assertions derived from the published fix, not original upstream tests. Minimal receiver/globals preserve the targeted behavior; full application omitted."
            ),
            spans,
        )
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


def assess_saved_roots(rows: list[dict]) -> dict:
    """Score published changed functions without inventing other block labels."""
    from pcg.inference import brier_score, expected_calibration_error
    from pcg.validation import reliability_bins

    roots = []
    for row in rows:
        if row.get("mode") != "upstream-regression" or row["status"] in {
            "incomplete",
            "no-executable-checks",
        }:
            continue
        report = json.loads(Path(row["report"]).read_text(encoding="utf-8"))
        functions = [
            r
            for r in report["ranking"]
            if "::<module>" not in r["name"] and "#seg" not in r["name"]
        ]
        root = next(
            (
                r
                for r in functions
                if r["name"].split("::", 1)[-1] == ROOTS[row["case"]]
            ),
            None,
        )
        if root:
            row["root_probability"] = root["defect_probability"]
            row["root_rank"] = functions.index(root) + 1
            roots.append(row)
    if not roots:
        return {"status": "no-completed-labelled-roots"}
    probabilities = [r["root_probability"] for r in roots]
    labels = [int(r["version"] == "buggy") for r in roots]
    ranks = [r["root_rank"] for r in roots if r["version"] == "buggy"]
    return {
        "labelled_roots": len(roots),
        "brier": brier_score(probabilities, labels),
        "ece": expected_calibration_error(probabilities, labels),
        "reliability_bins": reliability_bins(probabilities, labels),
        "buggy_root_ranks": ranks,
        "top_1": sum(r == 1 for r in ranks) / len(ranks) if ranks else None,
        "top_3": sum(r <= 3 for r in ranks) / len(ranks) if ranks else None,
        "limitation": "Only published changed functions are labelled; other blocks are not assumed clean. Selected compatibility cases, no fitting or deployment calibration claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="rescore saved reports without executing or fetching candidates",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES),
        help="rerun selected cases and retain other saved results",
    )
    args = parser.parse_args()
    output = Path("out/real-bug-benchmark")
    output.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        from pcg.benchmark_metrics import summarize_benchmark

        path = output / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["targeted_root_assessment"] = assess_saved_roots(summary["rows"])
        summary["quality_metrics"] = summarize_benchmark(summary["rows"])
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary["targeted_root_assessment"], indent=2))
        return
    rows = []
    if args.case and (output / "summary.json").exists():
        previous = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        rows = [r for r in previous["rows"] if r["case"] not in args.case]
    for name, case in CASES.items():
        if args.case and name not in args.case:
            continue
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
                    functions = [
                        r
                        for r in report["ranking"]
                        if "::<module>" not in r["name"] and "#seg" not in r["name"]
                    ]
                    root = next(
                        (
                            r
                            for r in functions
                            if r["name"].split("::", 1)[-1] == ROOTS[name]
                        ),
                        None,
                    )
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
                            "root_probability": root["defect_probability"]
                            if root
                            else None,
                            "root_rank": functions.index(root) + 1 if root else None,
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
        "benchmark": "BugsInPy five-case, three-repository compatibility replay",
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
    from pcg.benchmark_metrics import summarize_benchmark

    summary["quality_metrics"] = summarize_benchmark(rows)
    summary["targeted_root_assessment"] = assess_saved_roots(rows)
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
