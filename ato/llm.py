"""The model seam.

The one interface every model call goes through. Nothing outside this module
imports a vendor SDK, which is what keeps the 8-10B secondary condition a second
adapter rather than a refactor.

Shipped here: the Protocol, the response type, and a FakeClient so the whole test
suite runs with no network. The real adapters and the cache are TASKS.md T1.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    parsed: BaseModel | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    cache_key: str = ""
    from_cache: bool = False
    # True when structured parsing failed twice. The caller records a
    # schema-validity failure; it never coerces a malformed verdict into a valid
    # one, because schema validity is a reported metric.
    parse_failed: bool = False


def cache_key(
    messages: Sequence[Message],
    *,
    model: str,
    temperature: float,
    seed: int,
    response_model: type[BaseModel] | None,
) -> str:
    """Content address for a call. A rerun after a harness bug must not re-spend
    the budget and must not move the numbers."""
    payload = {
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "response_model": response_model.__name__ if response_model else None,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
    ) -> LLMResponse: ...


@dataclass
class FakeClient:
    """Canned responses keyed by cache key, or in call order.

    Every test and every `--dry-run` uses this. There is no network in the test
    suite, ever: a test that would make a real call is a test that cannot be run
    in CI and cannot be rerun deterministically.
    """

    by_key: dict[str, str] = field(default_factory=dict)
    in_order: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
    ) -> LLMResponse:
        key = cache_key(
            messages,
            model=model,
            temperature=temperature,
            seed=seed,
            response_model=response_model,
        )
        self.calls.append({"key": key, "model": model, "messages": list(messages)})

        if key in self.by_key:
            text = self.by_key[key]
        elif self.in_order:
            text = self.in_order.pop(0)
        else:
            raise AssertionError(
                f"FakeClient has no response for call {len(self.calls)} (key {key[:12]})"
            )

        parsed, parse_failed = None, False
        if response_model is not None:
            try:
                parsed = response_model.model_validate_json(text)
            except Exception:
                parse_failed = True

        return LLMResponse(
            text=text,
            parsed=parsed,
            cache_key=key,
            parse_failed=parse_failed,
        )
