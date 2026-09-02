"""LLM Critic module for PCG.

Provides an LLM code-review judge that evaluates code blocks for logic bugs,
edge-case vulnerabilities, and defects, returning an empirical verdict with
confidence and reasoning.

Two backends:
  - "anthropic": Real calls to Claude API (requires ANTHROPIC_API_KEY).
  - "mock": Deterministic heuristic critic for testing and offline development.
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

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


def _parse_response(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return {
            "is_buggy": bool(obj["is_buggy"]),
            "confidence": float(obj["confidence"]),
            "reason": str(obj.get("reason", "")),
            "raw": text,
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return {
            "is_buggy": None,
            "confidence": None,
            "reason": "PARSE_ERROR",
            "raw": text,
        }


class AnthropicCritic:
    """Judge code snippets using the Anthropic Claude API."""

    def __init__(self, model: str = "claude-sonnet-4-6", max_retries: int = 3) -> None:
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = model
        self.max_retries = max_retries

    def judge(self, code: str, file: str = "snippet.py") -> dict[str, Any]:
        prompt = CRITIC_PROMPT.format(file=file, code=code)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text"
                )
                return _parse_response(text)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2**attempt)
        return {
            "is_buggy": None,
            "confidence": None,
            "reason": f"API_ERROR: {last_err}",
            "raw": None,
        }


class MockCritic:
    """Deterministic stand-in critic based on AST complexity and branchiness."""

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)

    def judge(self, code: str, file: str = "snippet.py") -> dict[str, Any]:
        complexity = (
            code.count("if")
            + code.count("for")
            + code.count("while")
            + code.count("except")
        )
        base_p_buggy = min(0.5 + 0.08 * complexity, 0.93)
        noise = self.rng.uniform(-0.15, 0.15)
        p_buggy = max(0.05, min(0.97, base_p_buggy + noise))
        is_buggy = p_buggy > 0.5
        confidence = p_buggy if is_buggy else (1.0 - p_buggy)
        confidence = min(0.98, confidence**0.5)
        return {
            "is_buggy": is_buggy,
            "confidence": round(confidence, 3),
            "reason": "mock heuristic on code complexity and branching",
            "raw": None,
        }


def get_critic(backend: str | Any) -> Any:
    """Return a critic instance for the specified backend."""
    if hasattr(backend, "judge"):
        return backend
    if backend == "anthropic":
        return AnthropicCritic()
    if backend == "mock":
        return MockCritic()
    if backend in (None, "none", ""):
        return None
    raise ValueError(f"Unknown critic backend: {backend}")
