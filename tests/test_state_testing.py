"""State dependencies must produce meaningful sequences for unseen classes."""

import ast
import json

from pcg.pipeline import analyze
from pcg.execution import ExecutionConfig
from pcg.state_testing import sequence_seeds, state_graph


SOURCE = """class Session:
    def __init__(self):
        self.opened = False
        self.authorized = False
        self.loaded = False
    def open(self):
        self.opened = True
    def authorize(self):
        if self.opened:
            self.authorized = True
    def load(self):
        if self.authorized:
            self.loaded = True
    def send(self):
        if self.loaded:
            raise RuntimeError('unexpected failure after setup')
    def ping(self):
        return 1
    def status(self):
        return 0
    def version(self):
        return '1'
    def noop(self):
        return None
"""


def methods(source):
    return {
        n.name: n
        for n in ast.parse(source).body[0].body
        if isinstance(n, ast.FunctionDef)
    }


def test_writer_reader_graph_generates_setup_chain_not_unrelated_methods():
    entries = methods(SOURCE)
    seeds, metadata = sequence_seeds(
        entries, [n for n in entries if not n.startswith("_")]
    )
    expected = [("open", ()), ("authorize", ()), ("load", ()), ("send", ())]
    assert expected in seeds
    assert all("ping" not in [step[0] for step in sequence] for sequence in seeds)
    assert {"writer": "load", "reader": "send", "fields": ["loaded"]} in metadata[
        "edges"
    ]
    json.dumps(metadata)


def test_mutation_and_private_helpers_contribute_state_dependencies():
    src = "class Store:\n    def _add(self):\n        self.items.append(1)\n    def update(self):\n        self._add()\n    def read(self):\n        return self.items[0]\n"
    graph = state_graph(methods(src))
    assert graph.has_edge("update", "read")
    assert graph["update"]["read"]["fields"] == ["items"]


def test_read_only_methods_do_not_create_false_writer_edges():
    graph = state_graph(
        methods(
            "class A:\n    def first(self):\n        return self.value\n    def second(self):\n        return self.value\n"
        )
    )
    assert graph.number_of_edges() == 0


def test_graph_sequence_finds_delayed_error_without_user_tests():
    baseline = analyze(
        SOURCE,
        autonomous_testing=True,
        graph_guided=False,
        execution=ExecutionConfig("local"),
    )
    guided = analyze(
        SOURCE,
        autonomous_testing=True,
        graph_guided=True,
        execution=ExecutionConfig("local"),
    )
    assert baseline.execution.valid and guided.execution.valid
    assert not baseline.execution.failed_ids
    assert guided.execution.failed_ids
    assert all(t.oracle == "probe" for t in guided.execution.tests)
    failure = next(t for t in guided.execution.tests if t.outcome == "failed")
    assert "unexpected failure after setup" in failure.failure_details
    case = next(
        c
        for c in guided.test_plan["cases"]
        if c.get("method") == "stateful-sequence-probe"
    )
    assert case["state_dependency_graph"]["seed_sequences"]
    assert guided.to_dict()["automatic_validation"]["status"] == "no-correctness-oracle"


def test_state_sequence_budget_and_counter_repetition():
    entries = methods("class Counter:\n    def step(self):\n        self.n += 1\n")
    seeds, metadata = sequence_seeds(entries, ["step"])
    assert [("step", ())] * 4 in seeds
    assert len(seeds) <= 16 and metadata["explored_states"] <= 256
