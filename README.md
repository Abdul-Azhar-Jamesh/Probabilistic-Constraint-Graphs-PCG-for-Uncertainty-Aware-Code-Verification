# Probabilistic Constraint Graphs (PCG) for Uncertainty-Aware Code Verification

PCG is a Bayesian framework designed to assess the correctness of LLM-generated code by fusing heterogeneous evidence sources (compiler diagnostics, static analysis, test executions, and LLM confidences) over a structured constraint graph of code blocks.

Rather than treating code correctness as a binary, program-wide property (e.g., "does it pass all tests?"), PCG isolates errors at the granularity of individual functions/methods and outputs a **Correctness Probability Map**. This allows developers and automated repair systems to distinguish between code that is **inherently buggy** and code that is merely **unreliable due to upstream dependencies**.

---

## ── Pipeline Architecture ──

The verification pipeline processes source code and test suites through five distinct stages:

```
Source Code ──> AST Decomposition ──> Constraint Graph Construction
                                            │
        ┌──── Compiler Diagnostics ─────────┤
        ├──── Static Analysis (Pylint+AST) ─┤
        ├──── Execution (Pytest Outcomes) ──┼──> Bayesian Fusion (Log-Odds)
        └──── LLM Confidence (Logprobs) ────┘            │
                                                         ▼
                                       Noisy-AND Constraint Propagation
                                                         │
                                                         ▼
                                     Correctness Probability Map & Repair Set
```

### 1. AST Block Decomposition (`blocks.py`)
Programs are decomposed into hierarchical **blocks** (functions, methods, and module-level statement groups). Decomposing code at this level aligns with how:
- **Evidence** is reported (e.g., a traceback points to a specific function execution).
- **Repair** is implemented (e.g., regenerating or rewriting a single function instead of the entire file).

### 2. Constraint Graph Construction (`graph.py`)
Edges in the graph represent dependency constraints between blocks. An edge from `B -> A` (or `A` depends on `B`) means that if `B` is buggy, the correctness of `A` is compromised.
Edges carry weights indicating **coupling strengths**:
- `calls` (strength: `0.85`): Direct invocation of helper functions.
- `dataflow` (strength: `0.70`): Passing of variables/objects.
- `sequence` (strength: `0.25`): Temporal or execution order dependencies.

Cycles caused by mutual recursion are resolved by **condensation** (collapsing the cycle into a single representative node to prevent infinite propagation loops).

### 3. Evidence Collection (`evidence.py`)
The pipeline gathers evidence from five distinct sensors:
- **Compile Diagnostics**: Syntax errors, import issues, and compilation warnings.
- **Static Analysis**: Linting errors/warnings (Pylint rules) and AST smells (e.g., deeply nested loops, empty exception handlers).
- **Execution Outcomes**: Test suite results and parsed tracebacks that localize failure points to specific call-stack frames.
- **LLM Confidence**: Token logprobs or structural code properties acting as a proxy for the generator's confidence.
- **LLM Code Review Critic (`critic.py`)**: Direct semantic assessment of code blocks by an LLM reviewer (Claude API or mock heuristic), returning structured bug verdicts with confidence and reasoning.

### 4. Bayesian Fusion & Constraint Propagation (`inference.py`)
1. **Prior Estimation**: Structural features (complexity, length, depth, parameter counts) determine a block's prior probability of being correct.
2. **Local Fusion**: Evidence associated with a block is fused using a Naive Bayes model in log-odds space to calculate its local correctness probability.
3. **Noisy-AND Propagation**: The local probabilities are propagated along the constraint graph to a fixed point. The trust in block $i$, denoted by $P(C_i)$, depends on its local trust $P_{\text{local}}(i)$ and the trustworthiness of all its dependencies:
   $$P(C_i) \leftarrow P_{\text{local}}(i) \cdot \prod_{j \in \text{deps}(i)} [ 1 - s_{ij} \cdot (1 - P(C_j)) ]$$
   where $s_{ij}$ is the coupling strength of the dependency edge from $j$ to $i$.

---

## ── Key Design Concept: Posterior vs. Culpability ──

A naive belief-propagation model creates a "propagation trap." If function `median` is buggy, any function calling it (e.g., `summarize`) will also produce incorrect results. The posterior probability $P(C_{\text{summarize}})$ correctly collapses to near $0.0$. 

However, flagging `summarize` for repair is a **false positive**—its own implementation is correct; it is simply suffering from **inherited doubt**. 

To solve this, PCG reports two separate metrics per block:

1. **`posterior`** ($P(\text{correct} \mid \text{Evidence})$): *"Can I trust this block's output?"* This includes inherited doubt and is used to decide whether to **abstain** from using the code.
2. **`culpability`** ($1 - P_{\text{local}}$): *"Is this block's own implementation defective?"* This isolates evidence originating strictly within the block and is used to identify **repair targets**.

Automated repair systems target blocks with high **culpability**, while users review the final Correctness Map using both metrics.

---

## ── Training & Calibration Engine ──

To move away from hand-tuned constants (reliability coefficients, edge strengths, prior coefficients), PCG features an end-to-end calibration engine.

### 1. Mutation Generation (`mutate.py`)
Correct reference programs (located in the `REFERENCE_PROGRAMS` dictionary in `pcg/corpus.py`) are mutated to synthesize labeled bugs. The mutation operators mirror common LLM bugs:
- **`boundary`**: Flips comparison operators (e.g., `<` to `<=`).
- **`arithmetic`**: Swaps basic operators (e.g., `+` to `-`).
- **`offbyone`**: Adjusts integer constants (e.g., `+1` or `-1`).
- **`dropguard`**: Removes early-return guard clauses.
- **`swapargs`**: Swaps the arguments of non-commutative function calls.

Mutants are ran against the reference test suite. Equivalent mutants (those that still pass all tests) are discarded, ensuring all corpus mutants are **detectable**.

### 2. Feature Extraction (`build_training_set.py`)
Features are extracted for each block across all mutants, compiling both:
- **Structural Features**: `log_cyclomatic`, `log_loc`, `depth`, `n_params`
- **Evidence Features**: Counts and weight sums of compile, static, execution, and LLM evidence.

### 3. Model Calibration (`fit_weights.py` / `calibrate.py`)
Logistic regression models are trained on the training set to replace hand-tuned weights:
1. **Prior Model**: Predicts correctness prior using structural features.
2. **Evidence Model**: Learns the log-odds impact (reliabilities) of different evidence sources.

---

## ── Getting Started ──

### Installation
Ensure you have the required dependencies:
```bash
pip install -r requirements.txt
```

### Regenerating the Corpus Cache
`out/corpus_mutants.pkl` is a shared cache consumed by **both** `python -m pcg.evaluate` and `python scripts/run_calibration.py`. If you add or remove reference programs in `pcg/corpus.py`, delete this file before running either command so the new programs are picked up:
```bash
rm out/corpus_mutants.pkl   # Windows: del out\corpus_mutants.pkl
```
You can also control runtime during iteration by capping the corpus size. Pass `max_mutants=40` to `load_mutant_corpus()` (in `mutate.py`) to use roughly the first 40 detectable mutants instead of the full set.

### Running the Analysis Pipeline
Analyze a target candidate file against its test suite (optionally evaluating each block with an LLM reviewer):
```bash
# Basic run with compiler, static analysis, and test execution evidence:
python -m pcg.pipeline demo/candidate.py -t demo/test_candidate.py --json out/report.json --html out/report.html --heatmap

# Enable the LLM Code Review Critic (mock heuristic or Anthropic Claude API):
python -m pcg.pipeline demo/candidate.py -t demo/test_candidate.py --critic mock
python -m pcg.pipeline demo/candidate.py -t demo/test_candidate.py --critic anthropic
```
This produces:
- A console summary highlighting **Repair Targets** (high culpability) vs. **Downstream Unreliable Blocks** (low culpability but low posterior).
- A breakdown of which sensor triggered doubt (test failures, linter smells, or LLM critic verdicts).
- A machine-readable report in `out/report.json`.
- An interactive HTML heatmap in `out/report.html`.

### Running the Web App (Streamlit)
Launch the interactive Correctness Probability Map:
```bash
streamlit run app.py
```

### Running Tests
Execute the framework test suite:
```bash
pytest
```

### Running Calibration & Weight Fitting
To regenerate the mutant corpus, extract features, and fit the logistic regression weights:
```bash
python scripts/run_calibration.py
# or equivalently:
python -m pcg.calibrate
```

### Running the Evaluation Harness
By default this scores against the **auto-generated mutation corpus** — detectable mutants across all reference programs plus the clean reference programs as all-correct examples — rather than hand-labeled cases. The corpus is built and cached by `load_mutant_corpus()` in `mutate.py`, which is the single source of truth called by both `evaluate.py` and the calibration pipeline.
```bash
python -m pcg.evaluate
```

---

## ── Codebase Layout ──

```
pcg/
  ├── blocks.py             # AST parser and hierarchical block decomposer
  ├── graph.py              # Dependency constraint graph constructor
  ├── evidence.py           # Multi-sensor evidence collection (pylint, pytest, AST, LLM critic)
  ├── critic.py             # LLM code reviewer (Anthropic Claude API + deterministic Mock)
  ├── inference.py          # Prior computation, Naive Bayes fusion, Noisy-AND propagation
  ├── pipeline.py           # Orchestrates the CLI, analysis pipeline, and outputs
  ├── report.py             # Renders console reports, JSON, and HTML heatmaps
  ├── mutate.py             # Mutant generation + load_mutant_corpus() (shared corpus cache loader)
  ├── build_training_set.py # Feature extraction and training set compilation
  ├── fit_weights.py        # Weights calibration via Stratified K-Fold Logistic Regression
  └── calibrate.py          # Shim interface pointing to fit_weights.py
critic_calibration/         # LLM critic calibration harness on real BugsInPy open-source bugs
  ├── build_dataset.py      # Extracts paired (buggy vs fixed) code diffs from BugsInPy
  ├── calibration_dataset.json # 80 real-world labeled examples (40 real bugs, buggy + fixed)
  ├── llm_critic.py         # Standalone critic runner
  ├── run_calibration.py    # Runs the critic over real bug examples and logs predictions
  └── analyze_calibration.py # Computes Accuracy, F1, Brier score, ECE & evidence weights
scripts/                  
  ├── run_calibration.py    # End-to-end training and calibration runner
  └── run_demo_analysis.py  # Simplified script to run and log demo candidate analysis
demo/                     
  ├── candidate.py          # Sample implementation with planted bugs
  ├── test_candidate.py     # Test suite for candidate.py
  └── holdout_case.py       # Separate held-out case for calibration validation
tests/                    
  └── test_pcg.py           # Framework verification tests
```
