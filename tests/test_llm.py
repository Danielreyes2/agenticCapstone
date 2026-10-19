"""The model seam. No test in this repo may make a network call."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ato.llm import (
    AnthropicClient,
    CachingClient,
    FakeClient,
    LLMClient,
    Message,
    cache_key,
    complete_structured,
)
from ato.schema import AgentVerdict

MSGS = [Message("system", "you are the fiduciary agent"), Message("user", "s014")]
CALL = {"model": "test-model", "temperature": 0.0, "seed": 7, "max_tokens": 512}


def test_fake_client_satisfies_the_protocol() -> None:
    assert isinstance(FakeClient(), LLMClient)


def test_cache_key_is_stable_and_discriminating() -> None:
    base = cache_key(MSGS, model="m", temperature=0.0, seed=1, response_model=None)
    assert base == cache_key(MSGS, model="m", temperature=0.0, seed=1, response_model=None)
    assert base != cache_key(MSGS, model="m", temperature=0.7, seed=1, response_model=None)
    assert base != cache_key(MSGS, model="m", temperature=0.0, seed=2, response_model=None)
    assert base != cache_key(MSGS, model="n", temperature=0.0, seed=1, response_model=None)
    assert base != cache_key(MSGS, model="m", temperature=0.0, seed=1, response_model=AgentVerdict)


def test_structured_output_parses_into_the_verdict_model() -> None:
    payload = AgentVerdict(
        agent="fiduciary",
        action_id="a1",
        score=-0.6,
        hard_violation=False,
        confidence=0.8,
        rationale="Tight cycle.",
        evidence=["fiduciary.discretionary_remaining"],
    ).model_dump_json()
    client = FakeClient(in_order=[payload])
    r = client.complete(MSGS, **CALL, response_model=AgentVerdict)
    assert r.parsed is not None and r.parsed.agent == "fiduciary"
    assert r.parse_failed is False


def test_a_malformed_verdict_is_flagged_not_coerced() -> None:
    """Schema validity is a reported metric. A malformed verdict must surface as
    a failure, never be repaired into a valid-looking one."""
    client = FakeClient(in_order=['{"agent": "fiduciary", "score": 9000}'])
    r = client.complete(MSGS, **CALL, response_model=AgentVerdict)
    assert r.parse_failed is True
    assert r.parsed is None


def test_fake_client_refuses_to_improvise() -> None:
    with pytest.raises(AssertionError):
        FakeClient().complete(MSGS, **CALL)


def _verdict_json() -> str:
    return AgentVerdict(
        agent="fiduciary",
        action_id="a1",
        score=-0.6,
        hard_violation=False,
        confidence=0.8,
        rationale="Tight cycle.",
        evidence=["fiduciary.discretionary_remaining"],
    ).model_dump_json()


BAD_VERDICT = '{"agent": "fiduciary", "score": 9000}'


def test_complete_structured_returns_on_the_first_valid_response() -> None:
    client = FakeClient(in_order=[_verdict_json()])
    result = complete_structured(client, MSGS, response_model=AgentVerdict, **CALL)
    assert result.attempts == 1
    assert result.parsed is not None and result.parsed.agent == "fiduciary"
    assert result.raw == [_verdict_json()]
    assert len(result.cache_keys) == 1
    assert len(client.calls) == 1


def test_complete_structured_retries_once_with_the_validation_error() -> None:
    client = FakeClient(in_order=[BAD_VERDICT, _verdict_json()])
    result = complete_structured(client, MSGS, response_model=AgentVerdict, **CALL)
    assert result.attempts == 2
    assert result.parsed is not None and result.parsed.agent == "fiduciary"
    assert result.raw == [BAD_VERDICT, _verdict_json()]
    assert result.cache_keys[0] != result.cache_keys[1]
    retry = client.calls[1]["messages"]
    assert retry[: len(MSGS)] == MSGS
    assert retry[len(MSGS)] == Message("assistant", BAD_VERDICT)
    feedback = retry[len(MSGS) + 1]
    assert feedback.role == "user"
    assert feedback.content.startswith("That response failed validation:")
    assert "AgentVerdict" in feedback.content
    assert feedback.content.endswith("No prose, no code fence.")
    assert len(client.calls) == 2


def test_complete_structured_stops_after_a_second_failure() -> None:
    """A third call would raise: FakeClient has only two canned responses."""
    client = FakeClient(in_order=[BAD_VERDICT, BAD_VERDICT])
    result = complete_structured(client, MSGS, response_model=AgentVerdict, **CALL)
    assert result.parsed is None
    assert result.attempts == 2
    assert result.raw == [BAD_VERDICT, BAD_VERDICT]
    assert len(client.calls) == 2


def test_cache_hit_matches_the_miss_except_from_cache(tmp_path: Path) -> None:
    inner = FakeClient(in_order=[_verdict_json()])
    client = CachingClient(inner, cache_dir=tmp_path)
    miss = client.complete(MSGS, **CALL, response_model=AgentVerdict)
    hit = client.complete(MSGS, **CALL, response_model=AgentVerdict)
    assert miss.from_cache is False
    assert hit.from_cache is True
    assert hit.text == miss.text
    assert hit.parsed == miss.parsed
    assert hit.input_tokens == miss.input_tokens
    assert hit.output_tokens == miss.output_tokens
    assert hit.finish_reason == miss.finish_reason
    assert hit.cache_key == miss.cache_key
    assert hit.parse_failed is miss.parse_failed
    assert len(inner.calls) == 1
    key = miss.cache_key
    stored = json.loads((tmp_path / key[:2] / f"{key}.json").read_text())
    assert set(stored) == {"fingerprint", "response"}
    assert stored["fingerprint"]["model"] == "test-model"
    assert stored["fingerprint"]["seed"] == 7
    assert stored["response"]["text"] == miss.text
    assert stored["response"]["parse_failed"] is False


def test_a_schema_failure_is_cached(tmp_path: Path) -> None:
    inner = FakeClient(in_order=[BAD_VERDICT])
    client = CachingClient(inner, cache_dir=tmp_path)
    miss = client.complete(MSGS, **CALL, response_model=AgentVerdict)
    hit = client.complete(MSGS, **CALL, response_model=AgentVerdict)
    assert miss.parse_failed is True and hit.parse_failed is True
    assert hit.parsed is None and hit.text == miss.text
    assert hit.from_cache is True
    assert len(inner.calls) == 1


def test_vendor_sdk_import_is_confined_to_llm() -> None:
    root = Path(__file__).parents[1] / "ato"
    hits = [path.name for path in root.rglob("*.py") if "anthropic" in path.read_text()]
    assert hits == ["llm.py"]


class _Messages:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.text)],
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            stop_reason="end_turn",
        )


def test_anthropic_client_lifts_a_leading_system_prefix() -> None:
    messages_api = _Messages(_verdict_json())
    client = AnthropicClient(SimpleNamespace(messages=messages_api))
    response = client.complete(
        [Message("system", "one"), Message("system", "two"), *MSGS[1:]],
        **CALL,
        response_model=AgentVerdict,
    )
    sent = messages_api.calls[0]
    assert sent["system"] == "one\n\ntwo"
    assert sent["messages"] == [{"role": "user", "content": "s014"}]
    assert "seed" not in sent
    assert response.parsed is not None and response.parsed.agent == "fiduciary"
    assert response.input_tokens == 11 and response.output_tokens == 7
    assert isinstance(client, LLMClient)


def test_anthropic_client_rejects_a_system_message_after_chat() -> None:
    messages_api = _Messages(_verdict_json())
    client = AnthropicClient(SimpleNamespace(messages=messages_api))
    with pytest.raises(ValueError):
        client.complete(
            [Message("user", "s014"), Message("system", "too late")],
            **CALL,
        )
    assert messages_api.calls == []
