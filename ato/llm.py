"""The model seam.

The one interface every model call goes through. Nothing outside this module
imports a vendor SDK, which is what keeps the 8-10B secondary condition a second
adapter rather than a refactor.

`LLMClient.complete` is transport. Parse-retry policy lives in
`complete_structured`, because a retry is a different request and schema
validity is a reported metric that needs an attempt count.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError


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


def _request_fingerprint(
    messages: Sequence[Message],
    *,
    model: str,
    temperature: float,
    seed: int,
    response_model: type[BaseModel] | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "response_model": response_model.__name__ if response_model else None,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }


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
    blob = json.dumps(
        _request_fingerprint(
            messages,
            model=model,
            temperature=temperature,
            seed=seed,
            response_model=response_model,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
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


def _parse(text: str, response_model: type[BaseModel] | None) -> tuple[BaseModel | None, bool]:
    if response_model is None:
        return None, False
    try:
        return response_model.model_validate_json(text), False
    except ValidationError:
        return None, True


@dataclass
class StructuredResult:
    parsed: BaseModel | None
    attempts: int
    cache_keys: list[str]
    input_tokens: int
    output_tokens: int
    raw: list[str]


def _validation_error(text: str, response_model: type[BaseModel]) -> str:
    try:
        response_model.model_validate_json(text)
    except ValidationError as exc:
        return str(exc)
    return "response failed validation"


def complete_structured(
    client: LLMClient,
    messages: Sequence[Message],
    *,
    response_model: type[BaseModel],
    model: str,
    temperature: float,
    seed: int,
    max_tokens: int,
) -> StructuredResult:
    """One transport call, and at most one retry carrying the validation error.

    The retry is a different request, so it has its own cache key. A second
    failure returns parsed=None; the caller records the schema-validity failure.
    """

    def once(batch: Sequence[Message]) -> LLMResponse:
        return client.complete(
            batch,
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            response_model=response_model,
        )

    first = once(messages)
    if not first.parse_failed:
        return StructuredResult(
            parsed=first.parsed,
            attempts=1,
            cache_keys=[first.cache_key],
            input_tokens=first.input_tokens,
            output_tokens=first.output_tokens,
            raw=[first.text],
        )

    error = _validation_error(first.text, response_model)
    second = once(
        [
            *messages,
            Message("assistant", first.text),
            Message(
                "user",
                f"That response failed validation: {error}. Return only JSON matching "
                f"the {response_model.__name__} schema. No prose, no code fence.",
            ),
        ]
    )
    return StructuredResult(
        parsed=None if second.parse_failed else second.parsed,
        attempts=2,
        cache_keys=[first.cache_key, second.cache_key],
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        raw=[first.text, second.text],
    )


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / key[:2] / f"{key}.json"


def _load_cached(
    path: Path,
    key: str,
    fingerprint: dict[str, Any],
    response_model: type[BaseModel] | None,
) -> LLMResponse:
    stored = json.loads(path.read_text())
    if stored.get("fingerprint") != fingerprint:
        raise ValueError(f"cache file {path} does not match its request")
    body = stored["response"]
    text = body["text"]
    parse_failed = bool(body["parse_failed"])
    parsed, failed_now = _parse(text, None if parse_failed else response_model)
    if failed_now:
        raise ValueError(f"cache file {path} does not re-parse under {response_model}")
    return LLMResponse(
        text=text,
        parsed=parsed,
        input_tokens=int(body["input_tokens"]),
        output_tokens=int(body["output_tokens"]),
        finish_reason=str(body["finish_reason"]),
        cache_key=key,
        from_cache=True,
        parse_failed=parse_failed,
    )


def _store_cached(path: Path, fingerprint: dict[str, Any], response: LLMResponse) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "response": {
            "text": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "finish_reason": response.finish_reason,
            "parse_failed": response.parse_failed,
        },
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    os.replace(temporary, path)


@dataclass
class CachingClient:
    """Content-addressed wrapper. A hit matches the miss except from_cache=True.

    Parse failures are stored too. A schema-validity failure is a result the
    metrics count; leaving it uncached would make a replay disagree with the
    run it is replaying, and would re-spend budget on calls that already failed.
    """

    inner: LLMClient
    cache_dir: Path = field(default_factory=lambda: Path(".cache/llm"))

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
        fingerprint = _request_fingerprint(
            messages,
            model=model,
            temperature=temperature,
            seed=seed,
            response_model=response_model,
        )
        key = cache_key(
            messages,
            model=model,
            temperature=temperature,
            seed=seed,
            response_model=response_model,
        )
        path = _cache_path(self.cache_dir, key)
        if path.is_file():
            return _load_cached(path, key, fingerprint, response_model)
        response = self.inner.complete(
            messages,
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            response_model=response_model,
        )
        _store_cached(path, fingerprint, response)
        return response


def _split_system(messages: Sequence[Message]) -> tuple[str | None, list[Message]]:
    """The Messages API takes system as a prefix, not as a chat turn.

    A system message after a non-system one is a caller bug. Reordering it
    would change the prompt without the caller being able to tell.
    """
    seen_chat = False
    system_parts: list[str] = []
    chat: list[Message] = []
    for message in messages:
        if message.role == "system":
            if seen_chat:
                raise ValueError("system message after a non-system message")
            system_parts.append(message.content)
            continue
        seen_chat = True
        chat.append(message)
    if not system_parts:
        return None, chat
    return "\n\n".join(system_parts), chat


def _message_text(response: Any) -> str:
    parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _accepts(func: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in parameters)


def temperature_sent(client: LLMClient, temperature: float) -> float | None:
    """The temperature the provider actually received, or None if it received none.

    A fake client and an SDK without the argument both answer None, so the log
    never claims a temperature was pinned when it was not.
    """
    inner: Any = client
    while hasattr(inner, "inner"):
        inner = inner.inner
    if isinstance(inner, AnthropicClient) and inner.sends_temperature:
        return temperature
    return None


class AnthropicClient:
    """The only vendor SDK import in the repo lives in this class.

    `seed` is not sent. The Messages API has no seed parameter; reproducibility
    is the response cache (D10).
    """

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import anthropic  # type: ignore[import-not-found, unused-ignore]

            # Default max_retries is the SDK's own backoff on this same request.
            # The cache key is the request, so a transport retry cannot change it.
            client = anthropic.Anthropic()
        self._client = client
        self.sends_temperature = _accepts(client.messages.create, "temperature")

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
        system, chat = _split_system(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": message.role, "content": message.content} for message in chat],
        }
        # Some SDK versions take no temperature (D13). The requested value still
        # keys the cache; the run log records separately whether it was sent.
        if self.sends_temperature:
            kwargs["temperature"] = temperature
        if system is not None:
            kwargs["system"] = system
        sdk_response = self._client.messages.create(**kwargs)
        text = _message_text(sdk_response)
        parsed, parse_failed = _parse(text, response_model)
        usage = sdk_response.usage
        return LLMResponse(
            text=text,
            parsed=parsed,
            input_tokens=int(usage.input_tokens),
            output_tokens=int(usage.output_tokens),
            finish_reason=str(sdk_response.stop_reason or "stop"),
            cache_key=cache_key(
                messages,
                model=model,
                temperature=temperature,
                seed=seed,
                response_model=response_model,
            ),
            parse_failed=parse_failed,
        )
