"""Worker-only pytest plugin: structured outcomes and per-test branch coverage.

Loaded explicitly by execution.py; never imports or executes candidate code itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import coverage


class Recorder:
    def __init__(self) -> None:
        self.cov = coverage.Coverage(
            data_file=None, branch=True, include=["*/candidate.py"]
        )
        self.records: dict[str, dict[str, Any]] = {}
        self.collection_errors: list[str] = []
        self.cov.start()

    def pytest_runtest_setup(self, item: Any) -> None:
        self.cov.switch_context(item.nodeid)

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
            for entry in report.longrepr.reprtraceback.reprentries:
                location = getattr(entry, "reprfileloc", None)
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
        files = [p for p in data.measured_files() if Path(p).name == "candidate.py"]
        for nodeid, record in self.records.items():
            data.set_query_context(nodeid)
            record["lines"] = sorted(
                {ln for p in files for ln in (data.lines(p) or [])}
            )
            record["arcs"] = sorted(
                {arc for p in files for arc in (data.arcs(p) or [])}
            )
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
    config.pluginmanager.register(Recorder(), "pcg-recorder")
