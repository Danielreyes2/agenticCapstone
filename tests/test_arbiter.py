"""The arbiter. Scoring is a formula; blame is a later call."""

from __future__ import annotations

import json
from pathlib import Path

from ato.arbiter import (
    BlameAttribution,
    VerdictRound,
    arbitrate,
    attribute_blame,
    field_paths,
)
from ato.faults import LIBRARY, applies_to, inject_state_fault
from ato.llm import FakeClient, LLMResponse
from ato.provider import FixtureProvider
from ato.schema import AgentName, AgentVerdict, WorldState, check_state

CALL = {"model": "test-model", "temperature": 0.0, "seed": 7, "max_tokens": 256}
AGENTS: tuple[AgentName, ...] = ("fiduciary", "wellness", "productivity")


def _verdict(
    agent: AgentName,
    action_id: str,
    score: float,
    *,
    hard: bool = False,
) -> AgentVerdict:
    return AgentVerdict(
        agent=agent,
        action_id=action_id,
        score=score,
        hard_violation=hard,
        confidence=0.7,
        rationale="Score from the canned round.",
    )


def _round(
    scores: dict[str, float],
    *,
    hard: set[str] | None = None,
    input_tokens: int = 0,
) -> VerdictRound:
    flagged = hard or set()
    verdicts = [
        _verdict(agent, action_id, score, hard=action_id in flagged)
        for action_id, score in scores.items()
        for agent in AGENTS
    ]
    return VerdictRound(verdicts=verdicts, input_tokens=input_tokens)


def _blame(agent: str, field: str, confidence: float) -> str:
    return json.dumps({"agent": agent, "field": field, "confidence": confidence})


def test_low_readiness_raises_the_wellness_weight(clean: WorldState) -> None:
    # a1 is a fiduciary score only. a2 is a wellness score only.
    verdicts = [
        _verdict("fiduciary", "a1", 1.0),
        _verdict("wellness", "a1", 0.0),
        _verdict("productivity", "a1", 0.0),
        _verdict("fiduciary", "a2", 0.0),
        _verdict("wellness", "a2", 1.0),
        _verdict("productivity", "a2", 0.0),
    ]
    decision = arbitrate(clean, [VerdictRound(verdicts)])
    assert clean.wellness.readiness_score < 50
    assert decision.weights == {"fiduciary": 0.25, "wellness": 0.5, "productivity": 0.25}
    assert decision.action_id == "a2"
    assert decision.utility == 0.5


def test_readiness_at_or_above_50_keeps_equal_weights(fixtures_dir: Path) -> None:
    state = FixtureProvider(fixtures_dir).load("s007")
    verdicts = [
        _verdict("fiduciary", "a1", 1.0),
        _verdict("wellness", "a1", 0.0),
        _verdict("productivity", "a1", 0.0),
        _verdict("fiduciary", "a2", 0.0),
        _verdict("wellness", "a2", 1.0),
        _verdict("productivity", "a2", 0.0),
    ]
    decision = arbitrate(state, [VerdictRound(verdicts)])
    assert state.wellness.readiness_score >= 50
    assert decision.weights["fiduciary"] == decision.weights["wellness"]
    assert decision.action_id == "a1"


def test_a_hard_violation_eliminates_before_scoring(clean: WorldState) -> None:
    verdicts = [
        _verdict("fiduciary", "a1", 1.0, hard=True),
        _verdict("wellness", "a1", 1.0),
        _verdict("productivity", "a1", 1.0),
        _verdict("fiduciary", "a2", -1.0),
        _verdict("wellness", "a2", -1.0),
        _verdict("productivity", "a2", -1.0),
    ]
    decision = arbitrate(clean, [VerdictRound(verdicts)])
    assert decision.action_id == "a2"


def test_stopping_rule_terminates(clean: WorldState) -> None:
    pulled = 0

    def source():
        nonlocal pulled
        while True:
            pulled += 1
            yield _round({"a1": 0.0, "a2": min(1.0, 0.15 * pulled)})

    decision = arbitrate(clean, source())
    assert pulled == 3
    assert decision.rounds == 3
    assert decision.truncated is False
    assert decision.action_id == "a2"


def test_a_stable_round_stops_early(clean: WorldState) -> None:
    pulled: list[float] = []

    def source():
        for score in (0.5, 0.6, 0.99):
            pulled.append(score)
            yield _round({"a1": 0.0, "a2": score})

    decision = arbitrate(clean, source())
    assert pulled == [0.5, 0.6]
    assert decision.rounds == 2


def test_a_new_hard_flag_keeps_the_loop_going(clean: WorldState) -> None:
    pulled = 0

    def source():
        nonlocal pulled
        while True:
            pulled += 1
            hard = {"a1"} if pulled > 1 else set()
            yield _round({"a1": 0.2, "a2": 0.2}, hard=hard)

    decision = arbitrate(clean, source())
    assert pulled == 3
    assert decision.rounds == 3
    assert decision.action_id == "a2"


def test_budget_truncation_keeps_the_choice(clean: WorldState) -> None:
    pulled = 0

    def source():
        nonlocal pulled
        while True:
            pulled += 1
            yield _round({"a1": 0.0, "a2": min(1.0, 0.15 * pulled)}, input_tokens=100)

    decision = arbitrate(clean, source(), token_budget=10)
    assert decision.truncated is True
    assert decision.action_id == "a2"
    assert decision.rounds == 1
    assert pulled == 1


def test_blame_cannot_reshape_the_decision(clean: WorldState) -> None:
    batch = _round({"a1": 0.1, "a2": 0.8})
    untouched = arbitrate(clean, [batch])
    blamed = arbitrate(clean, [batch])
    attribute_blame(
        clean,
        batch.verdicts,
        blamed,
        client=FakeClient(in_order=[_blame("wellness", "wellness.hrv_today", 0.9)]),
        **CALL,
    )
    attribute_blame(
        clean,
        batch.verdicts,
        untouched,
        client=FakeClient(in_order=["not json", "still not json"]),
        **CALL,
    )
    assert blamed == untouched
    assert blamed.action_id == "a2"


def test_blame_parses_one_structured_answer(clean: WorldState) -> None:
    batch = _round({"a1": 0.1, "a2": 0.8})
    decision = arbitrate(clean, [batch])
    client = FakeClient(in_order=[_blame("fiduciary", "fiduciary.discretionary_remaining", 0.4)])
    blame = attribute_blame(clean, batch.verdicts, decision, client=client, **CALL)
    assert blame == BlameAttribution(
        agent="fiduciary",
        field="fiduciary.discretionary_remaining",
        confidence=0.4,
    )
    assert len(client.calls) == 1


def test_field_paths_cover_every_slice_and_ignore_the_fault(clean: WorldState) -> None:
    paths = field_paths(clean)
    for prefix in ("fiduciary.", "wellness.", "productivity."):
        assert any(path.startswith(prefix) for path in paths)
    assert not any(path.startswith("candidate_actions") for path in paths)
    for spec in LIBRARY:
        if spec.surface != "state" or not applies_to(spec, clean):
            continue
        faulted, record = inject_state_fault(clean, spec)
        assert field_paths(faulted) == paths
        assert record.truth_field in paths


def test_blame_outside_the_field_list_is_a_parse_failure(clean: WorldState) -> None:
    batch = _round({"a1": 0.1, "a2": 0.8})
    decision = arbitrate(clean, [batch])
    invented = _blame("fiduciary", "fiduciary.discretionary_budget_remaining", 0.4)
    client = FakeClient(in_order=[invented, invented])
    blame = attribute_blame(clean, batch.verdicts, decision, client=client, **CALL)
    assert blame.parse_failed is True
    assert len(client.calls) == 2


def test_blame_may_abstain(clean: WorldState) -> None:
    batch = _round({"a1": 0.1, "a2": 0.8})
    decision = arbitrate(clean, [batch])
    abstain = json.dumps({"agent": None, "field": None, "confidence": 0.8})
    client = FakeClient(in_order=[abstain])
    blame = attribute_blame(clean, batch.verdicts, decision, client=client, **CALL)
    assert blame.parse_failed is False
    assert blame.agent is None and blame.field is None
    system = client.calls[0]["messages"][0].content
    assert "null" in system


def test_blame_field_must_belong_to_the_named_agent(clean: WorldState) -> None:
    batch = _round({"a1": 0.1, "a2": 0.8})
    decision = arbitrate(clean, [batch])
    crossed = _blame("wellness", "fiduciary.discretionary_remaining", 0.4)
    client = FakeClient(in_order=[crossed, crossed])
    blame = attribute_blame(clean, batch.verdicts, decision, client=client, **CALL)
    assert blame.parse_failed is True


def test_blame_sees_the_state_but_not_the_check(clean: WorldState) -> None:
    spec = next(s for s in LIBRARY if s.fault_id == "d_fid_balance")
    faulted, _ = inject_state_fault(clean, spec)
    batch = _round({"a1": 0.1, "a2": 0.8})
    decision = arbitrate(faulted, [batch])
    client = FakeClient(in_order=[_blame("fiduciary", "fiduciary.cycle_budget", 0.3)])
    blame = attribute_blame(faulted, batch.verdicts, decision, client=client, **CALL)
    assert blame.reconcile_called is False
    sent = "\n".join(message.content for message in client.calls[0]["messages"])
    payload = json.loads(client.calls[0]["messages"][-1].content)
    assert payload["state"]["fiduciary"]["discretionary_remaining"] == 180.0
    assert '"rule"' not in sent
    assert "d_fid_balance" not in sent


def test_reconcile_is_returned_only_when_the_model_asks(clean: WorldState) -> None:
    spec = next(s for s in LIBRARY if s.fault_id == "d_fid_balance")
    faulted, _ = inject_state_fault(clean, spec)
    batch = _round({"a1": 0.1, "a2": 0.8})
    decision = arbitrate(faulted, [batch])
    client = FakeClient(
        in_order=[
            json.dumps({"tool": "reconcile", "args": {}}),
            _blame("fiduciary", "fiduciary.discretionary_remaining", 0.9),
        ]
    )
    blame = attribute_blame(faulted, batch.verdicts, decision, client=client, **CALL)
    assert blame.reconcile_called is True
    assert blame.field == "fiduciary.discretionary_remaining"
    observation = client.calls[1]["messages"][-1].content
    assert json.loads(observation) == [
        violation.model_dump(mode="json") for violation in check_state(faulted)
    ]


def test_a_refusal_is_recorded(clean: WorldState) -> None:
    batch = _round({"a1": 0.1, "a2": 0.8})
    decision = arbitrate(clean, [batch])

    class _Refuses:
        def complete(self, messages: object, **_: object) -> LLMResponse:
            return LLMResponse(text="", finish_reason="refusal")

    blame = attribute_blame(clean, batch.verdicts, decision, client=_Refuses(), **CALL)
    assert blame.refused is True
    assert blame.parse_failed is True
