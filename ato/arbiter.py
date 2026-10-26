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

from ato.agents.base import BoundedLoop
from ato.llm import LLMClient, Message
from ato.schema import AgentName, AgentVerdict, WorldState, check_state

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

    `parse_failed`, `reconcile_called` and `refused` describe the call, not the
    world. None of them is derived from whether a fault was injected.
    """

    model_config = {"extra": "forbid"}

    agent: AgentName | None = None
    field: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    parse_failed: bool = False
    reconcile_called: bool = False
    refused: bool = False


class _BlameAnswer(BaseModel):
    """What the model may write. The call-level flags are not its to set."""

    model_config = {"extra": "forbid"}

    agent: AgentName | None
    field: str | None
    confidence: float = Field(ge=0.0, le=1.0)


_SLICES = ("fiduciary", "wellness", "productivity")


def field_paths(state: WorldState) -> list[str]:
    """Every leaf path in the three domain slices, in the syntax `read_path` reads.

    This is the answer space. It is built from the structure, which a fault
    never changes, so a faulted state yields the same list as its clean twin.
    """
    paths: list[str] = []
    for name in _SLICES:
        _leaves(getattr(state, name).model_dump(mode="json"), name, paths)
    return paths


def _leaves(node: Any, prefix: str, out: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _leaves(value, f"{prefix}.{key}", out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _leaves(value, f"{prefix}[{index}]", out)
    else:
        out.append(prefix)


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


_BLAME_SYSTEM = (
    "The action is already committed and will not change. You are given the full "
    "WorldState the agents worked from, their verdicts, and field_paths, the list "
    "of every WorldState field. If this decision were wrong because an input was "
    "wrong, name the agent whose input is most likely responsible and exactly one "
    "path from field_paths that belongs to that agent. If no input looks wrong, "
    "answer with agent null and field null. Before answering you may call one tool: "
    '{"tool": "reconcile", "args": {}} returns a check of each derived value '
    "against the data it is derived from. The final answer is JSON only, with "
    'exactly the keys "agent", "field" and "confidence" (0 to 1). '
    "No prose, no code fence."
)


def attribute_blame(
    state: WorldState,
    verdicts: Sequence[AgentVerdict],
    decision: Decision,
    *,
    client: LLMClient,
    model: str,
    temperature: float,
    seed: int,
    max_tokens: int,
    surface_evidence: bool = False,
    token_budget: int = 20000,
) -> BlameAttribution:
    """Ask who was wrong after the choice exists. This function returns no decision.

    The state is shown because the redundancy a detectable fault breaks lives
    in it. `check_state` is not shown: the model gets it only by calling the
    tool, so noticing stays the model's job. `surface_evidence` is condition D.
    """
    paths = field_paths(state)
    allowed = set(paths)
    payload: dict[str, Any] = {
        "committed_action_id": decision.action_id,
        "state": state.model_dump(mode="json"),
        "verdicts": [verdict.model_dump(mode="json") for verdict in verdicts],
        "field_paths": paths,
    }
    if surface_evidence:
        payload["evidence"] = [
            {"agent": verdict.agent, "action_id": verdict.action_id, "evidence": verdict.evidence}
            for verdict in verdicts
        ]

    def finish(raw: dict[str, Any]) -> _BlameAnswer:
        answer = _BlameAnswer.model_validate(raw)
        if (answer.agent is None) != (answer.field is None):
            raise ValueError("agent and field must both be null or both be set")
        if answer.field is not None:
            if answer.field not in allowed:
                raise ValueError(f"{answer.field} is not one of field_paths")
            if not answer.field.startswith(f"{answer.agent}."):
                raise ValueError(f"{answer.field} does not belong to {answer.agent}")
        return answer

    def reconcile(_: dict[str, Any]) -> str:
        return json.dumps([violation.model_dump(mode="json") for violation in check_state(state)])

    result = BoundedLoop(client, tool_cap=1, token_budget=token_budget).run_until(
        [Message("system", _BLAME_SYSTEM), Message("user", json.dumps(payload, sort_keys=True))],
        {"reconcile": reconcile},
        finish=finish,
        finish_label="an answer with agent, field and confidence",
        model=model,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    )
    called = "reconcile" in result.tools_called
    if result.final is None:
        return BlameAttribution(parse_failed=True, reconcile_called=called, refused=result.refused)
    return BlameAttribution(
        agent=result.final.agent,
        field=result.final.field,
        confidence=result.final.confidence,
        reconcile_called=called,
        refused=result.refused,
    )


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
