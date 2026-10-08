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

Before real-world release, extend this benchmark with more independently labelled repositories and realistic environments. The measurements added below include bootstrap uncertainty and elapsed time; controlled resource and performance evaluation still remains. Do not promote a model because a synthetic result improved.

## Project and published bug checks

Install the optional replay dependency with `python -m pip install -e ".[benchmark]"`.

`python scripts/real_bug_benchmark.py` evaluates five pinned bug/fix compatibility replays from **tqdm, HTTPie and Luigi**. Every case records source/test hashes, upstream commits, adaptations, execution outcomes, stability checks, runtime and report paths under `out/real-bug-benchmark`. HTTPie reuses upstream parametrized filename assertions and exact extracted function bodies; Luigi reuses the upstream handler TestCase with the exact handler class and installed Tornado. These extracted compatibility cases narrow the environment and are not the original full-project benchmark environments. Confirmed bug/fix pairs are counted separately from autonomous misses and unavailable executions. No probability model is fitted to these five selected cases.

`tests/test_project.py` checks package/src imports, cross-module earlier causes, previous object field writes, rare-branch automatic discovery, cache invalidation, missing dependencies and archive traversal rejection. These are correctness regressions, not an accuracy benchmark.

The tqdm-1 case fetches the exact changed module and reuses the fixed version's original regression assertions. Historical Nose setup helpers are replaced with stdlib IO helpers, using installed tqdm dependencies. The original benchmark specifies Python 3.6.9; this is explicitly a compatibility replay, not that original environment. Reports separate autonomous probes from supplied upstream assertions. The buggy revision fails the regression and the fixed revision passes. Both autonomous runs lack a correctness oracle for the enumeration start index: that miss is recorded rather than disguised as success. See the [BugsInPy metadata](https://github.com/soarsmu/BugsInPy/tree/master/projects/tqdm/bugs/1).

Obsolete feature-model training utilities, old fitted artifacts and the disconnected critic experiment have been removed. Current probabilistic model fitting remains in `validation.py`.

## Broader controlled evaluation and probability assessment

```sh
python -m pcg.validation --curated --execution local --timeout 20 --out out/quality-evaluation
python scripts/evaluate_autonomous.py --execution local
```

The curated dataset contains 12 hand-specified synthetic families and 24 bug/fix versions: arithmetic, empty inputs, boundaries, rare annotation violations, helper normalization, mutable defaults, prefix handling, nested contracts, delayed division, previous object state, list aliasing and helper return types. Four families each belong to train, validation and test. Assertions are independent of the mutation engine and PCG predictions. These are evaluation oracles, not an assumption that random user code already has tests.

The autonomous runner withholds those assertions. Its separate report measures automatic oracle-backed discoveries, misses, probe exceptions and unsupported checks. Crashes on unsupported inputs remain probe observations, including on fixed controls. Do not confuse autonomous discovery with failures found using supplied regression tests.

Probability reports include reliability bins, false-positive rates and repository bootstrap percentile intervals (500 resamples, seed 301). Localization uses intrinsic-defect ranks within each buggy program: top-1/top-3 and mean reciprocal rank. It no longer pools unrelated blocks into one global top-three list. Small repository counts make intervals unstable.

`out/quality-evaluation/report.md` summarizes comparisons; JSON includes outcomes/runtime, held-out predictions, bins and intervals. Fitted candidates remain uncalibrated and do not automatically replace the deployment default. Review Brier and ECE together: better ECE alone need not mean better probabilities, and recall can improve while false alarms increase. Synthetic families do not establish real-world calibration.

The published replay now additionally includes **tqdm-2** (duplicate ANSI reset while trimming) and **HTTPie-3** (explicitly unset headers). Exact changed functions are retained with minimal dependencies; these two cases use independent regression assertions derived from the published fixes, rather than original upstream integration tests. Five selected cases across three repositories remain a small compatibility study. `quality_metrics` records completed/unsupported denominators, detection and false-alarm rates, probe exceptions and median/p95 elapsed time separately for autonomous and regression modes. Timing measured during concurrent local work is descriptive, not a controlled performance benchmark.

## Reporting evidence honestly

JSON, HTML and the dashboard separate syntax errors, oracle failures, probe exceptions, unstable failures, static warnings and incomplete checks. Replay status distinguishes stable from not-replayed failures. Probability bands indicate review priority, not a certificate that code is safe or defective. A root ranking is a hypothesis; a passing bounded suite is not proof of intended behavior.

## AI-assisted repair

AI is useful as a candidate-patch generator after localization: provide the original code, bounded dependency slice, failing input, exception, named oracle and expected behavior when available. Preserve the original tests and contracts; run candidate patches in a fresh worker against the reproducer, existing suite and fresh generated checks. A passing patch is bounded evidence, not proof. Oracle-free probe failures cannot validate a semantic fix, and AI-written assertions must not be treated as independent ground truth merely because the same AI generated the patch. Existing AST repair templates and solver-guided inputs can provide cheaper candidates for simple comparison/arithmetic/guard errors. No new AI integration or automatic source modification is enabled by this evaluation work.
