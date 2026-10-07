# Evaluation and acceptance checks

## Automated checks

```sh
python -m pip install -e ".[dev,app]"
python -m pytest -q
python -m ruff check pcg tests scripts deploy app.py
```

Tests cover analytical Bayes calculations, explaining away, exact/Gibbs agreement, posterior correlations, duplicate evidence, cyclic and diamond dependencies, class-method resolution, request isolation, actual test coverage, harness failures, skipped and parametrized tests, deadlines, support files and repair regression contracts. Property-based checks verify inference monotonicity and independent contracts for bundled reference programs. Streamlit tests check startup and public-mode execution policy.

`tests/test_graph_diagnosis.py` additionally checks earlier assignments, killed definitions, unexecuted branches, loop-carried values, simple aliases, module initialization, side-effect failures, guarded division, split prior budgets and generated-test provenance. It verifies that probes cannot manufacture defect evidence, validate repairs or inflate mutation scores, and that segment repairs reach validation.

The Docker test is opt-in locally (`PCG_TEST_DOCKER=1`) and enabled in its CI job after building the image. Local tests do not establish Docker execution success.

## Runtime benchmark

```sh
python -m pcg.validation --demo --execution local
python -m pcg.validation --dataset data/cases.json --execution docker
```

The bundled demo uses six small synthetic program families. It demonstrates the evaluation machinery; it is not evidence of production generalization. Surviving and timed-out mutations are recorded as excluded/inconclusive, not mislabelled as equivalent or assigned artificial confidence.

User dataset JSON is a list of cases:

```json
[
  {
    "name": "unique-case-name",
    "repository": "repository-or-program-family",
    "split": "train",
    "source": "def add(a, b):\n    return a + b\n",
    "tests": "from candidate import add\ndef test_add():\n    assert add(1, 2) == 3\n",
    "buggy": [],
    "unreliable": []
  }
]
```

Provide all three splits and multiple representative repositories. All versions of a repository must share a split. Duplicate source content across splits is rejected. Labels refer to candidate-module qualified names, and intrinsic defects must also be labelled unreliable. Ground truth for real defects must come from specifications and confirmed fixes, not PCG's own predictions. The example above illustrates one row, not a complete benchmark.

Training estimates sensor probabilities with Beta(1,1) smoothing and optionally fits logistic structural priors. Validation selects the final runtime trust threshold. The test split is evaluated only after those choices are fixed. Models remain marked uncalibrated: measured ECE on a small dataset does not establish deployment calibration.

Reports compare the full graph model, fusion without dependency edges, tests-only and static-only variants. Each variant uses the same held-out cases; the trust threshold is the full model's fixed validation threshold. Report defect/trust precision, recall, F1, Brier, ECE, false-trust rate and top-three review precision. A separate default-model comparison helps identify regressions from fitting.

Before real-world release, extend this benchmark with independently labelled repositories, bootstrap uncertainty over repositories, realistic test-environment images, flaky-test measurements and operational latency/resource measurements. Do not promote a model because a synthetic result improved.

## Project and published bug checks

Install the optional replay dependency with `python -m pip install -e ".[benchmark]"`.

`python scripts/real_bug_benchmark.py` now evaluates pinned bug/fix compatibility replays from **tqdm, HTTPie and Luigi**. Every case records source/test hashes, upstream commits, adaptations, execution outcomes, stability checks, runtime and report paths under `out/real-bug-benchmark`. HTTPie reuses upstream parametrized filename assertions and exact extracted function bodies; Luigi reuses the upstream handler TestCase with the exact handler class and installed Tornado. These extracted compatibility cases narrow the environment and are not the original full-project benchmark environments. Confirmed bug/fix pairs are counted separately from autonomous misses and unavailable executions. No probability model is fitted to these three selected cases.

`tests/test_project.py` checks package/src imports, cross-module earlier causes, previous object field writes, rare-branch automatic discovery, cache invalidation, missing dependencies and archive traversal rejection. These are correctness regressions, not an accuracy benchmark.

The tqdm-1 case fetches the exact changed module and reuses the fixed version's original regression assertions. Historical Nose setup helpers are replaced with stdlib IO helpers, using installed tqdm dependencies. The original benchmark specifies Python 3.6.9; this is explicitly a compatibility replay, not that original environment. Reports separate autonomous probes from supplied upstream assertions. The buggy revision fails the regression and the fixed revision passes. Both autonomous runs lack a correctness oracle for the enumeration start index: that miss is recorded rather than disguised as success. See the [BugsInPy metadata](https://github.com/soarsmu/BugsInPy/tree/master/projects/tqdm/bugs/1).

Obsolete feature-model training utilities, old fitted artifacts and the disconnected critic experiment have been removed. Current probabilistic model fitting remains in `validation.py`.
