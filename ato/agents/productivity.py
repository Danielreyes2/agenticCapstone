"""Productivity agent.

Tools read the productivity slice only. The calendar tools recompute from the
meetings, not from `longest_free_block_min`. Using the stated field would make
a drifted P1 value look like the day. These agents hold no verification tool,
so `input_anomaly` stays empty no matter what the model writes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ato.agents.base import AgentRun, BoundedLoop
from ato.llm import LLMClient, Message
from ato.schema import Meeting, Productivity, WorldState


def _mins(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


def _merged_busy(productivity: Productivity) -> list[tuple[int, int]]:
    """Busy intervals after the same overlap merge P1 uses.

    `computed_longest_free_block` returns only the longest gap. The block count
    is the other half of that merge: a meeting that only touches another one is
    not a separate block.
    """
    lo, hi = _mins(productivity.day_start), _mins(productivity.day_end)
    busy: list[tuple[int, int]] = []
    for meeting in productivity.meetings:
        start = max(lo, _mins(meeting.start))
        end = min(hi, _mins(meeting.start) + meeting.duration_min)
        if end > start:
            busy.append((start, end))
    busy.sort()
    merged: list[list[int]] = []
    for start, end in busy:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def fragmentation(productivity: Productivity) -> int:
    """How many separate busy blocks the day contains after merging overlaps."""
    return len(_merged_busy(productivity))


def deadline_pressure(productivity: Productivity) -> float:
    """Sum of weight per day of notice. Due today or overdue contributes its weight once.

    Dividing by zero days would make one due-today deadline infinite.
    """
    total = 0.0
    for deadline in productivity.deadlines:
        if deadline.due_in_days <= 0:
            total += deadline.weight
        else:
            total += deadline.weight / deadline.due_in_days
    return round(total, 3)


def reschedule_candidates(productivity: Productivity) -> list[Meeting]:
    """Low-priority meetings whose removal lengthens the longest free block.

    A low-priority meeting outside that gap is not a candidate. Medium and high
    are not candidates. The comparison uses the calendar, not the stated
    `longest_free_block_min`, so a drifted statement of that field changes nothing.
    """
    current = productivity.computed_longest_free_block()
    found: list[Meeting] = []
    for index, meeting in enumerate(productivity.meetings):
        if meeting.priority != "low":
            continue
        kept = [
            item for item_index, item in enumerate(productivity.meetings) if item_index != index
        ]
        without = productivity.model_copy(update={"meetings": kept})
        if without.computed_longest_free_block() > current:
            found.append(meeting)
    return found


def _payload(state: WorldState) -> dict[str, Any]:
    return {
        "date": state.date.isoformat(),
        "productivity": state.productivity.model_dump(mode="json"),
        "candidate_actions": [action.model_dump(mode="json") for action in state.candidate_actions],
    }


def _system() -> str:
    # The schema is spelled out because a model left to guess wrote its own
    # keys, and a batch that fails validation leaves the arbiter nothing to score.
    return (
        "You are the productivity agent. You see only the productivity slice and "
        "the candidate actions. Reply with JSON only, no prose, no code fence. Either "
        '{"tool": one of fragmentation, deadline_pressure, reschedule_candidates, '
        '"args": {}} or '
        '{"verdicts": [...]} with exactly one verdict per candidate action. '
        "Each verdict is an object with exactly these keys and no others: "
        '"agent": "productivity"; '
        '"action_id": the candidate action id; '
        '"score": a number from -1 to 1, this domain\'s view of the action; '
        '"hard_violation": true only if the action does not fit in the remaining '
        "free time or collides with a deadline that cannot move; "
        '"confidence": a number from 0 to 1; '
        '"rationale": at most 280 characters; '
        '"evidence": a non-empty list of the dotted paths you relied on, each a '
        "path in the slice you were shown, written like "
        "productivity.meetings[0].start; "
        '"input_anomaly": null. '
        "Do not describe a numeric discrepancy in the rationale."
    )


class ProductivityAgent:
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
            raise TypeError("productivity agent cannot read a foreign slice")
        productivity = state.productivity
        observations, truncated = self._observe(productivity)
        user = json.dumps({"state": _payload(state), "observations": observations}, sort_keys=True)
        loop = BoundedLoop(self.client, tool_cap=self.tool_cap, token_budget=self.token_budget)
        result = loop.run(
            [Message("system", _system()), Message("user", user)],
            self._handlers(productivity),
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

    def _observe(self, productivity: Productivity) -> tuple[list[dict[str, Any]], bool]:
        rows: list[dict[str, Any]] = []
        truncated = False

        def room_for(name: str, result: Any) -> None:
            nonlocal truncated
            if len(rows) >= self.tool_cap:
                truncated = True
                return
            rows.append({"tool": name, "result": result})

        room_for("fragmentation", fragmentation(productivity))
        if len(rows) < self.tool_cap:
            room_for("deadline_pressure", deadline_pressure(productivity))
        else:
            truncated = True
        if len(rows) < self.tool_cap:
            room_for(
                "reschedule_candidates",
                [
                    meeting.model_dump(mode="json")
                    for meeting in reschedule_candidates(productivity)
                ],
            )
        else:
            truncated = True
        return rows, truncated

    def _handlers(self, productivity: Productivity) -> dict[str, Callable[[dict[str, Any]], str]]:
        def blocks(_: dict[str, Any]) -> str:
            return json.dumps(fragmentation(productivity))

        def pressure(_: dict[str, Any]) -> str:
            return json.dumps(deadline_pressure(productivity))

        def movable(_: dict[str, Any]) -> str:
            meetings = reschedule_candidates(productivity)
            return json.dumps([meeting.model_dump(mode="json") for meeting in meetings])

        return {
            "fragmentation": blocks,
            "deadline_pressure": pressure,
            "reschedule_candidates": movable,
        }
