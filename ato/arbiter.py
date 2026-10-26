"""Arbiter.

Scoring is a pure function of the latest verdict round and the state. Blame is
a second model call that receives a decision already returned, so a different
blame answer cannot reach back into the choice.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from ato.llm import LLMClient, Message, complete_structured
from ato.schema import AgentName, AgentVerdict, WorldState

MAX_ROUNDS = 3
SCORE_MOVE = 0.1
READINESS_FLOOR = 50

_REST = {"fiduciary": 1 / 3, "wellness": 1 / 3, "productivity": 1 / 3}
_LOW_READINESS = {"fiduciary": 0.25, "wellness": 0.5, "productivity": 0.25}


class Decision(BaseModel):
    """A committed choice. Frozen so a later blame call cannot rewrite it."""

    model_config = {"extra": "forbid", "frozen": True}

    action_id: str | None
    utility: float | None
    rationale: str
    truncated: bool
    rounds: int
    weights: dict[str, float]


class BlameAttribution(BaseModel):
    """The second call's answer: an agent, a field, and a confidence.

    `parse_failed` is set here when the model never produced that object. It is
    a property of the call, not of the world.
    """

    model_config = {"extra": "forbid"}

    agent: AgentName | None = None
    field: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    parse_failed: bool = False


@dataclass(frozen=True)
class VerdictRound:
    verdicts: list[AgentVerdict]
    input_tokens: int = 0
    output_tokens: int = 0


def weights_for(state: WorldState) -> dict[str, float]:
    """Wellness weighs more only while readiness is under the floor.

    The numbers are fixed. Fitting them to the confirmation log is out of scope,
    and a missing domain is not folded into the others: the formula has three terms.
    """
    if state.wellness.readiness_score < READINESS_FLOOR:
        return dict(_LOW_READINESS)
    return dict(_REST)


def arbitrate(
    state: WorldState,
    rounds: Iterable[VerdictRound],
    *,
    token_budget: int = 8000,
) -> Decision:
    """Take at most three rounds. Stop earlier when a round settles or the budget breaks.

    The round that crosses the budget is kept. Dropping it would throw away the
    only action the run had found.
    """
    weights = weights_for(state)
    accepted: list[AgentVerdict] | None = None
    previous: list[AgentVerdict] | None = None
    spent = 0
    taken = 0
    truncated = False
    pending = iter(rounds)
    while taken < MAX_ROUNDS:
        nxt = next(pending, None)
        if nxt is None:
            break
        taken += 1
        spent += nxt.input_tokens + nxt.output_tokens
        accepted = list(nxt.verdicts)
        if spent > token_budget:
            truncated = True
            break
        if previous is not None and not _round_moved(previous, accepted):
            break
        previous = accepted
    action_id, utility = _choose(state, accepted or [], weights)
    return Decision(
        action_id=action_id,
        utility=utility,
        rationale=_rationale(action_id, utility, weights),
        truncated=truncated,
        rounds=taken,
        weights=weights,
    )


def attribute_blame(
    verdicts: Sequence[AgentVerdict],
    decision: Decision,
    *,
    client: LLMClient,
    model: str,
    temperature: float,
    seed: int,
    max_tokens: int,
    surface_evidence: bool = False,
) -> BlameAttribution:
    """Ask who was wrong after the choice exists. This function returns no decision.

    `surface_evidence` is condition D. The paths are copied out of the verdicts
    into their own field so provenance is not only buried inside them. The
    decision object is not revised either way.
    """
    payload: dict[str, Any] = {
        "committed_action_id": decision.action_id,
        "verdicts": [verdict.model_dump(mode="json") for verdict in verdicts],
    }
    if surface_evidence:
        payload["evidence"] = [
            {"agent": verdict.agent, "action_id": verdict.action_id, "evidence": verdict.evidence}
            for verdict in verdicts
        ]
    result = complete_structured(
        client,
        [
            Message(
                "system",
                "The action is already committed. If that choice were wrong, name the "
                "agent whose input is most likely responsible and the WorldState field. "
                "Reply with JSON only: agent, field, confidence. No prose, no code fence.",
            ),
            Message("user", json.dumps(payload, sort_keys=True)),
        ],
        response_model=BlameAttribution,
        model=model,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    )
    parsed = result.parsed
    if not isinstance(parsed, BlameAttribution):
        return BlameAttribution(parse_failed=True)
    return parsed


def _choose(
    state: WorldState,
    verdicts: Sequence[AgentVerdict],
    weights: dict[str, float],
) -> tuple[str | None, float | None]:
    flagged = {verdict.action_id for verdict in verdicts if verdict.hard_violation}
    best_id: str | None = None
    best_u: float | None = None
    for action in state.candidate_actions:
        if action.id in flagged:
            continue
        utility = _utility(action.id, verdicts, weights)
        # Strictly greater, so an equal utility keeps the earlier candidate.
        if best_u is None or utility > best_u:
            best_id = action.id
            best_u = utility
    return best_id, best_u


def _utility(action_id: str, verdicts: Sequence[AgentVerdict], weights: dict[str, float]) -> float:
    scores: dict[str, float] = {
        verdict.agent: verdict.score for verdict in verdicts if verdict.action_id == action_id
    }
    return sum(weights[agent] * scores.get(agent, 0.0) for agent in weights)


def _round_moved(previous: Sequence[AgentVerdict], current: Sequence[AgentVerdict]) -> bool:
    """A new elimination, or any score moving by more than 0.1, is a new round's worth."""
    previous_flags = {verdict.action_id for verdict in previous if verdict.hard_violation}
    current_flags = {verdict.action_id for verdict in current if verdict.hard_violation}
    if current_flags - previous_flags:
        return True
    before = {(verdict.agent, verdict.action_id): verdict.score for verdict in previous}
    after = {(verdict.agent, verdict.action_id): verdict.score for verdict in current}
    for key in set(before) | set(after):
        if key not in before or key not in after:
            return True
        if abs(after[key] - before[key]) > SCORE_MOVE:
            return True
    return False


def _rationale(action_id: str | None, utility: float | None, weights: dict[str, float]) -> str:
    names = ("fiduciary", "wellness", "productivity")
    listed = ", ".join(f"{name}={weights[name]:.2f}" for name in names)
    if action_id is None or utility is None:
        text = f"No candidate survived hard violations. Weights {listed}."
    else:
        text = f"Chose {action_id} at utility {utility:.2f}. Weights {listed}."
    return text
