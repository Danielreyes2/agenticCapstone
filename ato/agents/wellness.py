"""Wellness agent.

Tools read the wellness slice only. None of them recomputes `hrv_baseline_7d`.
That comparison is W1, and an agent that ran it on every call would catch a
detectable fault before the arbiter was asked. These agents hold no verification
tool, so `input_anomaly` stays empty no matter what the model writes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ato.agents.base import AgentRun, BoundedLoop
from ato.llm import LLMClient, Message
from ato.schema import CandidateAction, Wellness, WorldState


@dataclass(frozen=True)
class Readiness:
    score: int
    hrv_delta: float | None


def readiness(wellness: Wellness) -> Readiness:
    """The stated score, and today's HRV minus the stated baseline.

    The delta uses the baseline on the slice. Recomputing it from `hrv_7d`
    would answer W1 inside the agent.
    """
    baseline = wellness.hrv_baseline_7d
    today = wellness.hrv_today
    if baseline is None or today is None:
        return Readiness(wellness.readiness_score, None)
    return Readiness(wellness.readiness_score, round(today - baseline, 3))


def recovery_trend(wellness: Wellness) -> float | None:
    """HRV change per day from the first observed day in the window to the last.

    A non-wear day is skipped, not counted as zero. None when fewer than two
    days were observed. The stated baseline is not an input.
    """
    observed = [(index, value) for index, value in enumerate(wellness.hrv_7d) if value is not None]
    if len(observed) < 2:
        return None
    (start_index, start_value), (end_index, end_value) = observed[0], observed[-1]
    span = end_index - start_index
    if span == 0:
        return None
    return round((end_value - start_value) / span, 3)


def energy_requirement(wellness: Wellness, action: CandidateAction) -> float:
    """Resting kcal the action's duration costs at this BMR.

    `strain_7d` has no conversion to kcal on the slice, so it is left for the
    model to read rather than folded into a made-up multiplier.
    """
    return round(wellness.bmr_kcal * action.time_min / 1440, 2)


def _payload(state: WorldState) -> dict[str, Any]:
    return {
        "date": state.date.isoformat(),
        "wellness": state.wellness.model_dump(mode="json"),
        "candidate_actions": [action.model_dump(mode="json") for action in state.candidate_actions],
    }


def _system() -> str:
    # The schema is spelled out because a model left to guess wrote its own
    # keys, and a batch that fails validation leaves the arbiter nothing to score.
    return (
        "You are the wellness agent. You see only the wellness slice and the "
        "candidate actions. Reply with JSON only, no prose, no code fence. Either "
        '{"tool": one of readiness, recovery_trend, energy_requirement, "args": {}} or '
        '{"verdicts": [...]} with exactly one verdict per candidate action. '
        "Each verdict is an object with exactly these keys and no others: "
        '"agent": "wellness"; '
        '"action_id": the candidate action id; '
        '"score": a number from -1 to 1, this domain\'s view of the action; '
        '"hard_violation": true only if the action breaks a non-negotiable '
        "recovery constraint; "
        '"confidence": a number from 0 to 1; '
        '"rationale": at most 280 characters; '
        '"evidence": a non-empty list of the dotted paths you relied on, each a '
        "path in the slice you were shown, written like wellness.hrv_today; "
        '"input_anomaly": null. '
        "Do not describe a numeric discrepancy in the rationale."
    )


class WellnessAgent:
    def __init__(
        self,
        client: LLMClient,
        *,
        tool_cap: int = 8,
        token_budget: int = 8000,
    ) -> None:
        self.client = client
        self.tool_cap = tool_cap
        self.token_budget = token_budget

    def run(
        self,
        state: WorldState,
        *,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> AgentRun:
        if not isinstance(state, WorldState):
            raise TypeError("wellness agent cannot read a foreign slice")
        wellness = state.wellness
        actions = list(state.candidate_actions)
        observations, truncated = self._observe(wellness, actions)
        user = json.dumps({"state": _payload(state), "observations": observations}, sort_keys=True)
        loop = BoundedLoop(self.client, tool_cap=self.tool_cap, token_budget=self.token_budget)
        result = loop.run(
            [Message("system", _system()), Message("user", user)],
            self._handlers(wellness, actions),
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            tool_calls=len(observations),
            truncated=truncated,
        )
        if result.parse_failed:
            return result
        stamped = [
            verdict.model_copy(update={"input_anomaly": None}) for verdict in result.verdicts
        ]
        return AgentRun(
            stamped,
            result.truncated,
            result.tool_calls,
            result.input_tokens,
            result.output_tokens,
            result.parse_failed,
        )

    def _observe(
        self, wellness: Wellness, actions: list[CandidateAction]
    ) -> tuple[list[dict[str, Any]], bool]:
        rows: list[dict[str, Any]] = []
        truncated = False

        def room_for(name: str, result: Any) -> None:
            nonlocal truncated
            if len(rows) >= self.tool_cap:
                truncated = True
                return
            rows.append({"tool": name, "result": result})

        report = readiness(wellness)
        room_for("readiness", {"score": report.score, "hrv_delta": report.hrv_delta})
        if len(rows) < self.tool_cap:
            room_for("recovery_trend", recovery_trend(wellness))
        else:
            truncated = True
        for action in actions:
            if len(rows) >= self.tool_cap:
                truncated = True
                break
            room_for(
                "energy_requirement",
                {"action_id": action.id, "kcal": energy_requirement(wellness, action)},
            )
        return rows, truncated

    def _handlers(
        self, wellness: Wellness, actions: list[CandidateAction]
    ) -> dict[str, Callable[[dict[str, Any]], str]]:
        by_id = {action.id: action for action in actions}

        def read(_: dict[str, Any]) -> str:
            report = readiness(wellness)
            return json.dumps({"score": report.score, "hrv_delta": report.hrv_delta})

        def trend(_: dict[str, Any]) -> str:
            return json.dumps(recovery_trend(wellness))

        def energy(args: dict[str, Any]) -> str:
            action = by_id.get(str(args["action_id"]))
            if action is None:
                raise KeyError(f"unknown action {args.get('action_id')}")
            return json.dumps({"kcal": energy_requirement(wellness, action)})

        return {
            "readiness": read,
            "recovery_trend": trend,
            "energy_requirement": energy,
        }
