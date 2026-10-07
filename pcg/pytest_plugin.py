"""Worker-only pytest plugin: structured outcomes and per-test branch coverage.

Loaded explicitly by execution.py; never imports or executes candidate code itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import coverage
import pcg_worker_trace as runtime


class Recorder:
    def __init__(self) -> None:
        self.root = Path(os.environ["PCG_RESULT_PATH"]).parent.resolve()
        spec = json.loads((self.root / "pcg_worker_spec.json").read_text())
        self.sources = {
            str((self.root / name).resolve()): name for name in spec["sources"]
        }
        runtime.install(self.root, spec["sources"])
        self.cov = coverage.Coverage(
            data_file=None, branch=True, include=list(self.sources)
        )
        self.records: dict[str, dict[str, Any]] = {}
        self.collection_errors: list[str] = []
        self.initialization_lines: list[int] = []
        self.initialization_captured = False
        self.cov.start()

    def pytest_runtest_setup(self, item: Any) -> None:
        runtime.start_test(item.nodeid)
        if not self.initialization_captured:
            data = self.cov.get_data()
            data.set_query_context("")
            self.initialization_lines = sorted(
                {
                    ln
                    for p in data.measured_files()
                    if Path(p).name == "candidate.py"
                    for ln in (data.lines(p) or [])
                }
            )
            data.set_query_contexts(None)
            self.initialization_captured = True
        self.cov.switch_context(item.nodeid)
        marker = item.get_closest_marker("pcg_oracle")
        self.records[item.nodeid] = {
            "nodeid": item.nodeid,
            "outcome": "skipped",
            "phases": {},
            "duration": 0.0,
            "frames": [],
            "exception": None,
            "oracle": marker.args[0] if marker else "user",
            "oracle_scope": marker.kwargs.get("scope", marker.args[0])
            if marker
            else "user",
            "initialization_lines": self.initialization_lines,
        }

    def pytest_runtest_logreport(self, report: Any) -> None:
        record = self.records.setdefault(
            report.nodeid,
            {
                "nodeid": report.nodeid,
                "outcome": "skipped",
                "phases": {},
                "duration": 0.0,
                "frames": [],
                "exception": None,
            },
        )
        record["phases"][report.when] = report.outcome
        record["duration"] += report.duration
        if report.failed:
            record["outcome"] = "failed" if report.when == "call" else "error"
        elif report.when == "call" and record["outcome"] != "error":
            record["outcome"] = report.outcome
        if getattr(report, "wasxfail", None):
            record["outcome"] = "xfailed" if report.skipped else "xpassed"
        if report.failed and hasattr(report.longrepr, "reprtraceback"):
            record["failure_details"] = str(report.longrepr)[-12000:]
            for entry in report.longrepr.reprtraceback.reprentries:
                location = getattr(entry, "reprfileloc", None)
                if location:
                    resolved = str(Path(location.path).resolve())
                    if resolved in self.sources:
                        record.setdefault("file_frames", []).append(
                            {
                                "file": self.sources[resolved],
                                "line": int(location.lineno),
                            }
                        )
                if location and Path(location.path).name == "candidate.py":
                    record["frames"].append(int(location.lineno))
            crash = getattr(report.longrepr, "reprcrash", None)
            record["exception"] = crash.message if crash else "test failed"

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self.collection_errors.append(str(report.longrepr))

    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        self.cov.stop()
        data = self.cov.get_data()
        files = [
            p for p in data.measured_files() if str(Path(p).resolve()) in self.sources
        ]
        remaining = 4000
        ordered = sorted(
            self.records.items(), key=lambda row: row[1]["outcome"] != "failed"
        )
        for nodeid, record in ordered:
            data.set_query_context(nodeid)
            record["file_lines"] = {
                self.sources[str(Path(p).resolve())]: sorted(data.lines(p) or [])
                for p in files
            }
            record["file_arcs"] = {
                self.sources[str(Path(p).resolve())]: sorted(data.arcs(p) or [])
                for p in files
            }
            record["lines"] = sorted(record["file_lines"].get("candidate.py", []))
            record["arcs"] = sorted(record["file_arcs"].get("candidate.py", []))
            if remaining >= 20:
                record["traces"] = runtime.traces(
                    nodeid,
                    min(600 if record["outcome"] == "failed" else 30, remaining),
                    4 if record["outcome"] == "failed" else 1,
                )
                remaining -= sum(len(t["events"]) for t in record["traces"])
        data.set_query_contexts(None)
        payload = {
            "schema_version": 1,
            "exit_code": int(exitstatus),
            "tests": list(self.records.values()),
            "collection_errors": self.collection_errors,
        }
        Path(os.environ["PCG_RESULT_PATH"]).write_text(
            json.dumps(payload), encoding="utf-8"
        )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers", "pcg_oracle(kind): provenance of generated observations"
    )
    config.pluginmanager.register(Recorder(), "pcg-recorder")
