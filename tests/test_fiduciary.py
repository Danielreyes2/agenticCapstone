"""The fiduciary agent. Tools are pure; reconcile is a constructor argument."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ato.agents.fiduciary import (
    FiduciaryAgent,
    card_recommendation,
    cost_of_deferral,
    reconcile_ledger,
    unused_credits,
)
from ato.faults import LIBRARY, inject_state_fault
from ato.llm import LLMResponse, Message
from ato.provider import FixtureProvider
from ato.schema import AgentVerdict, InputAnomaly, WorldState

CALL = {"model": "test-model", "temperature": 0.0, "seed": 7, "max_tokens": 512}


def _verdict(
    action_id: str,
    *,
    rationale: str = "Prefer the cheaper action.",
    anomaly: dict[str, object] | None = None,
) -> dict[str, object]:
    verdict = AgentVerdict(
        agent="fiduciary",
        action_id=action_id,
        score=0.2,
        hard_violation=False,
        confidence=0.7,
        rationale=rationale,
        evidence=["fiduciary.discretionary_remaining"],
    )
    payload = json.loads(verdict.model_dump_json())
    if anomaly is not None:
        payload["input_anomaly"] = anomaly
    return payload


def _batch(
    *action_ids: str,
    rationale: str = "Prefer the cheaper action.",
    anomaly: dict[str, object] | None = None,
) -> str:
    verdicts = [
        _verdict(action_id, rationale=rationale, anomaly=anomaly) for action_id in action_ids
    ]
    return json.dumps({"verdicts": verdicts})


class _Script:
    def __init__(self, texts: list[str], *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.texts = list(texts)
        self.calls: list[list[Message]] = []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
        response_model: type | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        text = self.texts.pop(0)
        return LLMResponse(
            text=text,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


def test_card_recommendation_falls_through_an_exhausted_cap(fixtures_dir: Path) -> None:
    state = FixtureProvider(fixtures_dir).load("s040")
    rec = card_recommendation(state.fiduciary, "us_supermarket", 40.0)
    assert rec.card_id == "card_b"
    assert rec.effective_rate == 0.02


def test_card_recommendation_is_deterministic(clean: WorldState) -> None:
    first = card_recommendation(clean.fiduciary, "dining", 34.0)
    second = card_recommendation(clean.fiduciary, "dining", 34.0)
    assert first == second
    assert first.card_id == "card_a"
    assert first.effective_rate == 0.04


def test_unused_credits_skip_expired_and_used(fixtures_dir: Path) -> None:
    open_state = FixtureProvider(fixtures_dir).load("s014")
    names = [c.name for c in unused_credits(open_state.fiduciary, open_state.date)]
    assert names == ["dining credit"]

    mixed = FixtureProvider(fixtures_dir).load("s044")
    kept = [c.name for c in unused_credits(mixed.fiduciary, mixed.date)]
    assert kept == ["dining credit"]


def test_reconcile_ledger_matches_the_budget_identity(clean: WorldState) -> None:
    assert reconcile_ledger(clean.fiduciary) is None
    spec = next(s for s in LIBRARY if s.fault_id == "d_fid_balance")
    faulted, _ = inject_state_fault(clean, spec)
    anomaly = reconcile_ledger(faulted.fiduciary)
    assert anomaly is not None
    assert anomaly.field == "fiduciary.discretionary_remaining"
    assert anomaly.expected == 140.0
    assert anomaly.observed == 180.0


def test_cost_of_deferral_is_the_forgone_reward(clean: WorldState) -> None:
    action = clean.candidate_actions[0]
    assert cost_of_deferral(clean.fiduciary, action) == 1.36
    assert cost_of_deferral(clean.fiduciary, action) == cost_of_deferral(clean.fiduciary, action)


def test_fiduciary_refuses_a_foreign_slice(clean: WorldState) -> None:
    agent = FiduciaryAgent(_Script([_batch("a1", "a2")]))
    with pytest.raises(TypeError):
        agent.run(clean.wellness, **CALL)  # type: ignore[arg-type]


def test_prompt_contains_only_the_fiduciary_slice(clean: WorldState) -> None:
    client = _Script([_batch("a1", "a2")])
    FiduciaryAgent(client, reconcile=False).run(clean, **CALL)
    blob = "\n".join(message.content for message in client.calls[0])
    assert "hrv_" not in blob
    assert "longest_free_block" not in blob
    assert "meetings" not in blob
    assert "reconcile" not in blob


def test_system_prompt_states_the_verdict_schema(clean: WorldState) -> None:
    client = _Script([_batch("a1", "a2")])
    FiduciaryAgent(client, reconcile=False).run(clean, **CALL)
    system = client.calls[0][0].content
    for key in AgentVerdict.model_fields:
        assert f'"{key}"' in system
    assert "280" in system
    assert "fiduciary." in system


def test_reconcile_is_a_condition_not_a_fork(clean: WorldState) -> None:
    spec = next(s for s in LIBRARY if s.fault_id == "d_fid_balance")
    faulted, _ = inject_state_fault(clean, spec)
    rationale = "Prefer the cheaper action."
    planted = {"field": "fiduciary.cycle_budget", "expected": 1.0, "observed": 2.0}
    off = FiduciaryAgent(
        _Script([_batch("a1", "a2", rationale=rationale, anomaly=planted)]), reconcile=False
    )
    on = FiduciaryAgent(
        _Script([_batch("a1", "a2", rationale=rationale, anomaly=planted)]), reconcile=True
    )

    without = off.run(faulted, **CALL)
    with_tool = on.run(faulted, **CALL)

    assert all(v.input_anomaly is None for v in without.verdicts)
    expected = InputAnomaly(
        field="fiduciary.discretionary_remaining", expected=140.0, observed=180.0
    )
    assert [v.input_anomaly for v in with_tool.verdicts] == [expected, expected]
    for verdict in with_tool.verdicts:
        assert "140" not in verdict.rationale
        assert "180" not in verdict.rationale
        assert len(verdict.rationale) <= 280
        assert isinstance(verdict, AgentVerdict)


def test_overlong_rationale_is_not_coerced(clean: WorldState) -> None:
    too_long = _verdict("a1")
    too_long["rationale"] = "x" * 281
    bad = json.dumps({"verdicts": [too_long, {**too_long, "action_id": "a2"}]})
    client = _Script([bad, bad])
    result = FiduciaryAgent(client).run(clean, **CALL)
    assert result.parse_failed is True
    assert result.verdicts == []
    assert len(client.calls) == 2


def test_a_bad_turn_is_retried_once(clean: WorldState) -> None:
    client = _Script(["not json", _batch("a1", "a2")])
    result = FiduciaryAgent(client).run(clean, **CALL)
    assert result.parse_failed is False
    assert [v.action_id for v in result.verdicts] == ["a1", "a2"]
    assert len(client.calls) == 2


def test_tool_cap_stops_the_loop(clean: WorldState) -> None:
    client = _Script(['{"tool": "unused_credits", "args": {}}'] * 6)
    result = FiduciaryAgent(client, reconcile=False, tool_cap=0).run(clean, **CALL)
    assert result.truncated is True
    assert result.tool_calls == 0
    assert len(client.calls) == 1


def test_token_budget_flags_truncation_and_keeps_verdicts(clean: WorldState) -> None:
    client = _Script([_batch("a1", "a2")], input_tokens=500, output_tokens=500)
    result = FiduciaryAgent(client, token_budget=10).run(clean, **CALL)
    assert result.truncated is True
    assert [v.action_id for v in result.verdicts] == ["a1", "a2"]
