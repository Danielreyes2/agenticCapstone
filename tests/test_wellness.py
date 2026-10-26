"""The wellness agent. Tools are pure and read the wellness slice only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ato.agents.wellness import (
    WellnessAgent,
    energy_requirement,
    readiness,
    recovery_trend,
)
from ato.faults import LIBRARY, inject_state_fault
from ato.llm import LLMResponse, Message
from ato.provider import FixtureProvider
from ato.schema import AgentVerdict, WorldState

CALL = {"model": "test-model", "temperature": 0.0, "seed": 7, "max_tokens": 512}


def _verdict(action_id: str, *, anomaly: dict[str, object] | None = None) -> dict[str, object]:
    verdict = AgentVerdict(
        agent="wellness",
        action_id=action_id,
        score=0.2,
        hard_violation=False,
        confidence=0.7,
        rationale="The shorter block costs less recovery.",
        evidence=["wellness.readiness_score"],
    )
    payload = json.loads(verdict.model_dump_json())
    if anomaly is not None:
        payload["input_anomaly"] = anomaly
    return payload


def _batch(*action_ids: str, anomaly: dict[str, object] | None = None) -> str:
    return json.dumps(
        {"verdicts": [_verdict(action_id, anomaly=anomaly) for action_id in action_ids]}
    )


class _Script:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls: list[list[Message]] = []

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
        return LLMResponse(text=self.texts.pop(0))


def test_readiness_is_todays_hrv_against_the_stated_baseline(clean: WorldState) -> None:
    report = readiness(clean.wellness)
    assert report.score == 41
    assert report.hrv_delta == -14.0
    assert report == readiness(clean.wellness)


def test_readiness_uses_the_stated_baseline_not_the_recomputed_one(clean: WorldState) -> None:
    spec = next(item for item in LIBRARY if item.fault_id == "d_wel_baseline")
    faulted, _ = inject_state_fault(clean, spec)
    assert readiness(faulted.wellness).hrv_delta == -5.0


def test_readiness_has_no_delta_when_today_was_not_worn(fixtures_dir: Path) -> None:
    state = FixtureProvider(fixtures_dir).load("s021")
    assert readiness(state.wellness).hrv_delta is None
    assert readiness(state.wellness).score == 68


def test_recovery_trend_skips_non_wear_days(clean: WorldState, fixtures_dir: Path) -> None:
    assert recovery_trend(clean.wellness) == -0.167
    unworn = FixtureProvider(fixtures_dir).load("s021")
    assert recovery_trend(unworn.wellness) == 1.0
    empty = FixtureProvider(fixtures_dir).load("s028")
    assert recovery_trend(empty.wellness) is None


def test_recovery_trend_ignores_the_stated_baseline(clean: WorldState) -> None:
    spec = next(item for item in LIBRARY if item.fault_id == "d_wel_baseline")
    faulted, _ = inject_state_fault(clean, spec)
    assert recovery_trend(faulted.wellness) == recovery_trend(clean.wellness)


def test_energy_requirement_is_resting_cost_of_the_duration(clean: WorldState) -> None:
    short, long = clean.candidate_actions
    assert energy_requirement(clean.wellness, short) == 14.62
    assert energy_requirement(clean.wellness, long) == 109.64


def test_wellness_refuses_a_foreign_slice(clean: WorldState) -> None:
    agent = WellnessAgent(_Script([_batch("a1", "a2")]))
    with pytest.raises(TypeError):
        agent.run(clean.fiduciary, **CALL)  # type: ignore[arg-type]


def test_prompt_contains_only_the_wellness_slice(clean: WorldState) -> None:
    client = _Script([_batch("a1", "a2")])
    WellnessAgent(client).run(clean, **CALL)
    blob = "\n".join(message.content for message in client.calls[0])
    assert "discretionary" not in blob
    assert "meetings" not in blob
    assert "longest_free_block" not in blob
    assert "reconcile" not in blob
    assert "hrv_delta" in blob
    system = client.calls[0][0].content
    for key in AgentVerdict.model_fields:
        assert f'"{key}"' in system
    assert "280" in system
    assert "wellness." in system


def test_a_planted_anomaly_is_cleared(clean: WorldState) -> None:
    planted = {"field": "wellness.hrv_baseline_7d", "expected": 1.0, "observed": 2.0}
    result = WellnessAgent(_Script([_batch("a1", "a2", anomaly=planted)])).run(clean, **CALL)
    assert all(verdict.input_anomaly is None for verdict in result.verdicts)
    assert [verdict.action_id for verdict in result.verdicts] == ["a1", "a2"]
