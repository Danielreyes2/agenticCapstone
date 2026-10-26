"""The productivity agent. Tools are pure and read the productivity slice only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ato.agents.productivity import (
    ProductivityAgent,
    deadline_pressure,
    fragmentation,
    reschedule_candidates,
)
from ato.faults import LIBRARY, inject_state_fault
from ato.llm import LLMResponse, Message
from ato.provider import FixtureProvider
from ato.schema import AgentVerdict, WorldState

CALL = {"model": "test-model", "temperature": 0.0, "seed": 7, "max_tokens": 512}


def _verdict(action_id: str, *, anomaly: dict[str, object] | None = None) -> dict[str, object]:
    verdict = AgentVerdict(
        agent="productivity",
        action_id=action_id,
        score=-0.2,
        hard_violation=False,
        confidence=0.7,
        rationale="The longer block fits the open morning.",
        evidence=["productivity.longest_free_block_min"],
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


def test_overlapping_meetings_count_as_one_block(fixtures_dir: Path) -> None:
    state = FixtureProvider(fixtures_dir).load("s033")
    assert fragmentation(state.productivity) == 2
    assert fragmentation(state.productivity) == fragmentation(state.productivity)


def test_fragmentation_ignores_the_stated_longest_block(clean: WorldState) -> None:
    spec = next(item for item in LIBRARY if item.fault_id == "d_pro_block")
    faulted, _ = inject_state_fault(clean, spec)
    assert fragmentation(faulted.productivity) == fragmentation(clean.productivity) == 1


def test_deadline_due_today_contributes_its_weight_once(fixtures_dir: Path) -> None:
    today = FixtureProvider(fixtures_dir).load("s061")
    assert deadline_pressure(today.productivity) == 1.0
    later = FixtureProvider(fixtures_dir).load("s014")
    assert deadline_pressure(later.productivity) == 0.5
    light = FixtureProvider(fixtures_dir).load("s028")
    assert deadline_pressure(light.productivity) == 0.057


def test_only_a_low_priority_meeting_that_lengthens_the_gap_is_movable(fixtures_dir: Path) -> None:
    canonical = FixtureProvider(fixtures_dir).load("s014")
    assert [item.title for item in reschedule_candidates(canonical.productivity)] == [
        "advisor sync"
    ]

    crowded = FixtureProvider(fixtures_dir).load("s021")
    assert reschedule_candidates(crowded.productivity) == []

    overlapped = FixtureProvider(fixtures_dir).load("s033")
    assert [item.title for item in reschedule_candidates(overlapped.productivity)] == ["mail run"]


def test_productivity_refuses_a_foreign_slice(clean: WorldState) -> None:
    agent = ProductivityAgent(_Script([_batch("a1", "a2")]))
    with pytest.raises(TypeError):
        agent.run(clean.wellness, **CALL)  # type: ignore[arg-type]


def test_prompt_contains_only_the_productivity_slice(clean: WorldState) -> None:
    client = _Script([_batch("a1", "a2")])
    ProductivityAgent(client).run(clean, **CALL)
    blob = "\n".join(message.content for message in client.calls[0])
    assert "hrv_" not in blob
    assert "discretionary" not in blob
    assert "readiness_score" not in blob
    assert "reconcile" not in blob
    assert "advisor sync" in blob
    system = client.calls[0][0].content
    for key in AgentVerdict.model_fields:
        assert f'"{key}"' in system
    assert "280" in system
    assert "productivity." in system


def test_a_planted_anomaly_is_cleared(clean: WorldState) -> None:
    planted = {"field": "productivity.longest_free_block_min", "expected": 1.0, "observed": 2.0}
    result = ProductivityAgent(_Script([_batch("a1", "a2", anomaly=planted)])).run(clean, **CALL)
    assert all(verdict.input_anomaly is None for verdict in result.verdicts)
