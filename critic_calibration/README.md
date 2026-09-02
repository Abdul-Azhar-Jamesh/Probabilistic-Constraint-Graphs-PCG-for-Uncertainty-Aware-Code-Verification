# LLM Critic Calibration Harness

Answers one question: **when the LLM critic says "90% confident this is buggy,"
is it actually right 90% of the time?** If not, don't feed its raw confidence
into a Bayesian posterior — you'll get precise-looking numbers built on an
uncalibrated signal.

## What's real here

- `../BugsInPy/` — a full clone of [BugsInPy](https://github.com/soarsmu/BugsInPy),
  501 real, human-verified bugs across 17 open-source Python projects, each with
  a buggy commit, a fixed commit, and a diff.
- `calibration_dataset.json` — 80 labeled examples (40 real bugs, each split into
  its buggy version and fixed version) extracted directly from those diffs.
  Ground truth is not synthetic; it's what the projects' own maintainers fixed.
- The full pipeline (dataset → critic → metrics → reliability diagram) has been
  run end-to-end and works.

## What's mocked

The actual LLM calls. This sandbox has no `ANTHROPIC_API_KEY`, so I ran the
pipeline with `MockCritic` (a simple, deliberately-flawed heuristic in
`llm_critic.py`) to prove the mechanics work. **Swap in the real critic by
setting your API key and passing `--backend anthropic`** — no other code
changes needed:

```bash
export ANTHROPIC_API_KEY=sk-...
python run_calibration.py --backend anthropic --dataset calibration_dataset.json
python analyze_calibration.py --results calibration_results.json
```

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Parses BugsInPy diffs into labeled (code, is_buggy) examples |
| `llm_critic.py` | The critic itself — `AnthropicCritic` (real) and `MockCritic` (test stand-in) |
| `run_calibration.py` | Runs the critic over the dataset, saves raw verdicts |
| `analyze_calibration.py` | Computes accuracy, Brier score, ECE, reliability diagram, suggested evidence weight |
| `calibration_dataset.json` | The 80 pre-built labeled examples (ready to use) |
| `calibration_results.json` | Mock critic's verdicts on all 80 (example output) |
| `calibration_summary.json` | Computed metrics (example output) |
| `reliability_diagram.png` | Confidence-vs-accuracy plot (example output) |

## Regenerating the dataset

To pull a different sample, more bugs, or different projects:

```bash
python build_dataset.py --n 60 --projects PySnooper cookiecutter httpie sanic tqdm thefuck tornado spacy luigi black
```

Bigger projects (`pandas`, `keras`, `scrapy`, `matplotlib`, `ansible`) are also
in BugsInPy and can be added to `--projects`, but their diffs tend to be
larger and noisier to extract clean snippets from — start small.

## Reading the output

- **Accuracy / Precision / Recall / F1** — is the critic even right, ignoring confidence?
- **Brier score** — accuracy of the *probability*, not just the yes/no verdict. 0 = perfect, 0.25 = coin flip.
- **Expected Calibration Error (ECE)** — the core number. Buckets examples by
  stated confidence and compares to actual accuracy in that bucket. 0 = perfectly calibrated.
- **Reliability diagram** — visual version of the above. If the line drops
  *below* the diagonal at high confidence, the critic is overconfident —
  exactly the failure mode that makes raw LLM confidence dangerous to trust
  in a Bayesian merge gate.
- **Suggested evidence weight** — a starting-point multiplier (0-1) for how
  much to trust this critic's stated confidence relative to a calibrated
  source, when combining it with other PCG evidence. Treat this as a
  starting point to tune against your own PCG posterior outputs, not a
  final answer.

## Honest caveats

- Snippets are diff hunks with a few lines of context, not always full
  functions — the critic sometimes won't have complete context (e.g. missing
  imports or surrounding class state). This mirrors a real limitation:
  your PCG pipeline will face the same partial-context problem on real PRs.
- "Fixed" examples are labeled `is_buggy=False`, which is a simplification —
  a fix for one bug doesn't guarantee the function is bug-free, only that
  *this specific* known bug is gone.
- 40 bugs is a reasonable pilot size, not a statistically bulletproof one.
  Widen `--n` and re-run once you're validating a specific critic prompt
  you intend to ship.
