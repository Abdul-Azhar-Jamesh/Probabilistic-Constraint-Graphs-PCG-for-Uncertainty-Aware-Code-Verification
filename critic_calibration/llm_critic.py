"""
The "LLM critic" evidence source referenced in the PCG blueprint:
given a code snippet, ask an LLM to judge whether it's buggy and how
confident it is. This module is deliberately isolated from the rest of
the harness so it can be swapped out or upgraded independently.

Two backends:
  - "anthropic": real calls to the Claude API (needs ANTHROPIC_API_KEY)
  - "mock":      a cheap deterministic stand-in so the rest of the
                 pipeline (dataset building, scoring, calibration
                 analysis) can be developed/tested without API cost.
                 Its outputs are intentionally overconfident and
                 slightly biased, so the calibration report has
                 something real to catch.
"""
import json
import random
import re
import time

CRITIC_PROMPT = """You are a code reviewer. Judge whether the following code \
snippet contains a bug (a logic error, edge-case failure, or defect that would \
cause incorrect behavior or a crash under some input).

Respond with ONLY a JSON object, no other text, in this exact form:
{{"is_buggy": true or false, "confidence": <float 0.0-1.0>, "reason": "<one sentence>"}}

confidence is how sure you are in your is_buggy verdict (not the probability \
the code is buggy specifically -- if you're 90% sure it's clean, that's \
is_buggy: false, confidence: 0.9).

Code snippet (file: {file}):
```python
{code}
```"""


def _parse_response(text: str):
    text = text.strip()
    # Strip markdown fences if the model adds them despite instructions.
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return {
            "is_buggy": bool(obj["is_buggy"]),
            "confidence": float(obj["confidence"]),
            "reason": obj.get("reason", ""),
            "raw": text,
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"is_buggy": None, "confidence": None, "reason": "PARSE_ERROR", "raw": text}


class AnthropicCritic:
    def __init__(self, model="claude-sonnet-4-6", max_retries=3):
        from anthropic import Anthropic
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model
        self.max_retries = max_retries

    def judge(self, code: str, file: str = "snippet.py"):
        prompt = CRITIC_PROMPT.format(file=file, code=code)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                return _parse_response(text)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        return {"is_buggy": None, "confidence": None, "reason": f"API_ERROR: {last_err}", "raw": None}


class MockCritic:
    """Deterministic stand-in. Intentionally overconfident and slightly
    biased toward calling things buggy when they look "complex" (long
    lines, nested conditionals) -- a plausible failure mode for a real
    LLM critic, so the calibration report below has something to catch."""

    def __init__(self, seed=7):
        self.rng = random.Random(seed)

    def judge(self, code: str, file: str = "snippet.py"):
        complexity = code.count("if") + code.count("for") + code.count("while") + code.count("except")
        base_p_buggy = min(0.5 + 0.08 * complexity, 0.93)
        noise = self.rng.uniform(-0.15, 0.15)
        p_buggy = max(0.05, min(0.97, base_p_buggy + noise))
        is_buggy = p_buggy > 0.5
        confidence = p_buggy if is_buggy else 1 - p_buggy
        # Mock critic is deliberately overconfident (pushes toward 1.0)
        confidence = min(0.98, confidence ** 0.5)
        return {
            "is_buggy": is_buggy,
            "confidence": round(confidence, 3),
            "reason": "mock heuristic on code complexity",
            "raw": None,
        }


def get_critic(backend: str):
    if backend == "anthropic":
        return AnthropicCritic()
    if backend == "mock":
        return MockCritic()
    raise ValueError(f"unknown backend: {backend}")
