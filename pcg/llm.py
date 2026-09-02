"""Real LLM confidence estimation via teacher-forced token log-probabilities.

The framework previously used a *structural proxy* for generator confidence
(longer, branchier blocks assumed less reliable). That proxy encodes an
assumption rather than measuring anything. This module replaces it with a
genuine measurement: run a causal language model over the candidate source and
read off, for every token, the log-probability the model assigned to the token
that actually appeared.

Why this is affordable on CPU
-----------------------------
We are *scoring* existing code, not generating it. Scoring is a single
teacher-forced forward pass over the whole file -- one matrix multiply chain,
no autoregressive loop. Generation of the same file would take one forward pass
*per token*. This is the difference between ~1 second and several minutes on a
CPU, and it is why local inference is practical here.

What statistic to use
---------------------
Mean token probability is the obvious choice and a poor one: it is dominated by
trivially predictable tokens (indentation, ``def``, closing parens), so it
mostly measures boilerplate density and correlates with block length rather
than correctness. We therefore compute several statistics per block and let
:mod:`pcg.calibrate` decide empirically which discriminates buggy from correct
blocks on the mutation corpus.

The most useful in practice is *peak surprisal*: a single wrong token (a
flipped comparison, an off-by-one bound) produces one very-low-probability
token surrounded by perfectly ordinary ones. Averaging washes that signal out;
taking the tail keeps it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .blocks import Block

# Small models that fit comfortably in CPU RAM. tiny_starcoder_py is trained on
# Python specifically and gives far sharper surprisal on code than a
# general-purpose model of the same size.
DEFAULT_MODEL = "bigcode/tiny_starcoder_py"
FALLBACK_MODEL = "distilgpt2"

_CACHE: dict[str, tuple[object, object]] = {}


@dataclass
class BlockScore:
    """Per-block statistics derived from token log-probabilities."""

    bid: str
    n_tokens: int
    logprobs: list[float] = field(default_factory=list)

    @property
    def mean_logprob(self) -> float:
        return sum(self.logprobs) / len(self.logprobs) if self.logprobs else 0.0

    @property
    def mean_prob(self) -> float:
        return math.exp(self.mean_logprob)

    @property
    def peak_surprisal(self) -> float:
        """Surprisal (nats) of the single least-expected token in the block.

        This is the statistic that actually localises point defects.
        """
        return -min(self.logprobs) if self.logprobs else 0.0

    @property
    def p10_logprob(self) -> float:
        """10th-percentile logprob -- peak surprisal, but robust to one outlier."""
        if not self.logprobs:
            return 0.0
        s = sorted(self.logprobs)
        return s[max(0, int(0.10 * len(s)) - 1)] if len(s) >= 10 else s[0]

    @property
    def frac_low(self) -> float:
        """Fraction of tokens the model found genuinely surprising (p < 0.05)."""
        if not self.logprobs:
            return 0.0
        thresh = math.log(0.05)
        return sum(1 for lp in self.logprobs if lp < thresh) / len(self.logprobs)

    def as_dict(self) -> dict[str, float]:
        return {
            "n_tokens": self.n_tokens,
            "mean_logprob": round(self.mean_logprob, 4),
            "mean_prob": round(self.mean_prob, 4),
            "peak_surprisal": round(self.peak_surprisal, 4),
            "p10_logprob": round(self.p10_logprob, 4),
            "frac_low": round(self.frac_low, 4),
        }


def load_model(name: str = DEFAULT_MODEL):
    """Load and cache a tokenizer/model pair. Falls back if `name` is unavailable."""
    if name in _CACHE:
        return _CACHE[name]
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "transformers is required for real LLM confidence; "
            "install it or run with --no-llm to use the structural proxy"
        ) from exc

    import torch

    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
    except Exception:
        if name == FALLBACK_MODEL:
            raise
        tok = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForCausalLM.from_pretrained(FALLBACK_MODEL)
        name = FALLBACK_MODEL

    model.eval()
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    _CACHE[name] = (tok, model)
    return tok, model


def _token_logprobs(
    source: str, tok, model, max_len: int = 1024, stride: int = 512
) -> list[tuple[int, int, float]]:
    """Return (char_start, char_end, logprob) for every token in `source`.

    Long files are scored with a sliding window: each window carries `stride`
    tokens of context whose predictions we discard, so every scored token sees
    a full left-context rather than starting cold at a chunk boundary.
    """
    import torch

    enc = tok(source, return_offsets_mapping=True, return_tensors=None)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    if len(ids) < 2:
        return []

    window = min(max_len, getattr(model.config, "n_positions", max_len) or max_len)
    out: list[tuple[int, int, float]] = []
    start = 0
    while start < len(ids) - 1:
        end = min(start + window, len(ids))
        chunk = torch.tensor([ids[start:end]])
        with torch.no_grad():
            logits = model(chunk).logits[0]
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        # logits[i] predicts token i+1, so token j (j>=1) is scored by row j-1.
        # On a continuation window we skip the first `stride` tokens: they were
        # already scored with better context by the previous window.
        first = 1 if start == 0 else stride
        for j in range(first, end - start):
            lp = logprobs[j - 1, chunk[0, j]].item()
            cs, ce = offsets[start + j]
            out.append((cs, ce, lp))

        if end == len(ids):
            break
        start += window - stride
    return out


def score_blocks(
    source: str,
    blocks: list[Block],
    model_name: str = DEFAULT_MODEL,
    tok=None,
    model=None,
) -> dict[str, BlockScore]:
    """Score every block by attributing whole-file token logprobs to blocks.

    The file is scored in one pass so each block is judged *in context* -- a
    call to a helper defined above is predictable, exactly as it should be. We
    then partition tokens by character offset.
    """
    if tok is None or model is None:
        tok, model = load_model(model_name)

    lines = source.splitlines(keepends=True)
    # Character offset of the start of each 1-indexed line.
    line_start: list[int] = [0, 0]
    acc = 0
    for ln in lines:
        acc += len(ln)
        line_start.append(acc)

    def span(b: Block) -> tuple[int, int]:
        lo = line_start[min(b.lineno, len(line_start) - 1)]
        hi_idx = min(b.end_lineno + 1, len(line_start) - 1)
        return lo, line_start[hi_idx]

    # Sort by span width ascending so a nested block claims its tokens before
    # the enclosing one does -- same ownership rule as line_to_block().
    ordered = sorted(blocks, key=lambda b: (b.end_lineno - b.lineno))
    spans = [(b.bid, *span(b)) for b in ordered]

    scores = {b.bid: BlockScore(bid=b.bid, n_tokens=0) for b in blocks}
    for cs, ce, lp in _token_logprobs(source, tok, model):
        if ce <= cs:
            continue
        for bid, lo, hi in spans:
            if lo <= cs < hi:
                scores[bid].logprobs.append(lp)
                scores[bid].n_tokens += 1
                break
    return scores


def token_logprobs_for(
    source: str, blocks: list[Block], model_name: str = DEFAULT_MODEL
) -> dict[str, list[float]]:
    """Convenience wrapper matching the `token_logprobs` hook in `analyze()`."""
    return {bid: s.logprobs for bid, s in score_blocks(source, blocks, model_name).items()}
