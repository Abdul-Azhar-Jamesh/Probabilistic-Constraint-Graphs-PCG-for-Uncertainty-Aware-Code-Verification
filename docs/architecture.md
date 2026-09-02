# Architecture Overview

This project is organized around a core probabilistic inference engine that analyzes code blocks and propagates uncertainty through dependency graphs.

## Main Components

- `pcg/blocks.py`: AST-based decomposition into blocks.
- `pcg/graph.py`: dependency graph and structural importance calculations.
- `pcg/evidence.py`: evidence collection from compilation, static analysis, execution, and LLM critic signals.
- `pcg/inference.py`: prior estimation, local evidence fusion, and graph propagation.
- `pcg/pipeline.py`: orchestration of analysis and reporting.
- `pcg/fit_weights.py`: calibration and weight fitting.

## Intended Product Layout

The repository is intentionally structured as a simple Python package today to preserve import stability and the project’s existing test suite. The higher-level product layout in the design brief can be layered on top later without disturbing the runtime package.
