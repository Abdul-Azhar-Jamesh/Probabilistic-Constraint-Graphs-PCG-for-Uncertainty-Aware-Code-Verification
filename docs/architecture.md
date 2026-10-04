# Implementation architecture

The code dependency graph and the probabilistic observation model serve different, connected purposes.

1. `blocks.py` extracts function, method, module and segment blocks. `graph.py` resolves conservative local dependencies and builds the original directed code graph.
2. `execution.py` starts a trusted local worker or isolated Docker worker. `pytest_plugin.py` writes structured outcomes with per-test line/branch coverage. Support modules may be supplied as validated relative Python files.
3. `evidence.py` attributes observations to covered blocks and retains neutral execution diagnostics. Compiler/static/optional LLM collectors produce independent raw sensor observations.
4. `sensors.py` validates explicit, versioned likelihood models. Default probabilities are assumptions. Legacy files in the output directory cannot silently change behavior.
5. `inference.py` aggregates local observations and creates test factors. `probabilistic.py` conditions latent defect variables exactly or through Gibbs sampling, then evaluates dependency-based trust queries over joint states.
6. `pipeline.py` provides the analysis API and CLI. `app.py` calls this same API with request-local settings. No dashboard request mutates global inference weights.
7. `report.py` renders review scores and observations. JSON reports include execution details and inference diagnostics. `test_quality.py` audits mutation survival; `repair.py` validates fixes against individual test identities.
8. `validation.py` fits on training repositories, selects thresholds on validation repositories, and evaluates untouched test repositories with ablations.

The structured runner is shared by analysis, mutation detection and repair. Harness errors cannot be counted as passing tests. The CLI and dashboard default to disabled test execution; the Python API's default local mode is reserved for trusted caller-supplied code.

Optional real critic calls send snippets to an external API only when explicitly selected. Mock critic results are an experiment, not independent real-world validation.

Generated outputs live under `out/`. Historical calibration artifacts live under `experiments/archive/` and are inactive. There is one dashboard entrypoint and one current runtime calibration/evaluation entrypoint.
