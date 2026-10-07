"""Exercise the dashboard entrypoint against the same pipeline as the CLI."""

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")


def test_dashboard_starts_with_execution_disabled():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=30)
    assert not app.exception
    execution = next(
        widget for widget in app.selectbox if widget.label == "Test execution"
    )
    assert execution.value == "Disabled"


def test_public_dashboard_disallows_local_execution(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PCG_PUBLIC_DEPLOYMENT", "1")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(
        timeout=30
    )
    next(w for w in app.selectbox if w.label == "Test execution").set_value(
        "Local trusted code"
    )
    app.run(timeout=30)
    next(w for w in app.checkbox if "I trust this source" in w.label).check()
    app.run(timeout=30)
    assert not app.exception
    assert any("require Docker isolation" in message.value for message in app.error)


def test_dashboard_plans_new_code_checks_without_enabling_execution():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(
        timeout=30
    )
    next(w for w in app.radio if w.label == "Source").set_value("Paste your own")
    app.run(timeout=30)
    next(w for w in app.text_area if w.label == "Python source").set_value(
        "def f(x: int) -> int:\n    return x + 1\n"
    )
    next(
        w for w in app.checkbox if w.label == "Automatically explore and test this code"
    ).check()
    next(
        w
        for w in app.checkbox
        if w.label == "Use declared return types as test contracts"
    ).check()
    app.run(timeout=30)
    assert not app.exception
    assert (
        next(w for w in app.selectbox if w.label == "Test execution").value
        == "Disabled"
    )
    assert any("Generated test plan" in w.label for w in app.expander)
