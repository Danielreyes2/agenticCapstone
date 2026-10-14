"""The model seam. No test in this repo may make a network call."""

from __future__ import annotations

import pytest

from ato.llm import FakeClient, LLMClient, Message, cache_key
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
