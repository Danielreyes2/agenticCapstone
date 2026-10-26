"""Fiduciary agent.

Tools read the fiduciary slice only. `reconcile` is a constructor argument,
default off, so C1a and C1b are the same class. The discrepancy is written to
`input_anomaly` by this code; the model's rationale is not where it goes, or
numeric drift and rhetorical inflation would share a channel.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from ato.agents.base import AgentRun, BoundedLoop
from ato.llm import LLMClient, Message
from ato.schema import (
    EPS,
    CandidateAction,
    Fiduciary,
    InputAnomaly,
    StatementCredit,
    WorldState,
)


@dataclass(frozen=True)
class CardRecommendation:
    card_id: str | None
    effective_rate: float


def _category_spend(fiduciary: Fiduciary, card_id: str, category: str) -> float:
    return round(
        sum(
            t.amount
            for t in fiduciary.transactions
            if t.kind == "purchase" and t.account == card_id and t.reward_category == category
        ),
        2,
    )


def card_recommendation(
    fiduciary: Fiduciary, reward_category: str | None, amount: float
) -> CardRecommendation:
    """Highest rate whose annual cap is not already spent.

    A cap that still has room keeps its rate for the whole amount. The schema
    has no split-tender record, and the exhausted-cap fixture is spend already
    past the cap. There is no quarterly-rotation calendar on `Card`, only
    `caps_annual`, so nothing else adjusts the rate.
    """
    del amount  # signature matches the proposal; exhaustion does not depend on it
    if reward_category is None:
        return CardRecommendation(None, 0.0)
    best_rate = -1.0
    best_id: str | None = None
    for card in fiduciary.cards:
        rate = card.rates.get(reward_category)
        if rate is None:
            continue
        cap = card.caps_annual.get(reward_category)
        if cap is not None and _category_spend(fiduciary, card.id, reward_category) >= cap:
            continue
        if rate > best_rate:
            best_rate = rate
            best_id = card.id
    if best_id is None:
        return CardRecommendation(None, 0.0)
    return CardRecommendation(best_id, best_rate)


def unused_credits(fiduciary: Fiduciary, as_of: date) -> list[StatementCredit]:
    """Unredeemed credits still open on `as_of`. The date is the scenario's, not the clock."""
    return [credit for credit in fiduciary.credits if not credit.used and credit.expires >= as_of]


def reconcile_ledger(fiduciary: Fiduciary) -> InputAnomaly | None:
    """F1, as a tool. The same identity `check_state` uses, returned as structured data."""
    expected = round(fiduciary.cycle_budget - fiduciary.discretionary_spend(), 2)
    observed = fiduciary.discretionary_remaining
    if abs(expected - observed) <= EPS:
        return None
    return InputAnomaly(
        field="fiduciary.discretionary_remaining",
        expected=expected,
        observed=observed,
    )


def cost_of_deferral(fiduciary: Fiduciary, action: CandidateAction) -> float:
    """Reward this cycle would pay and the next one does not.

    That is `cost * effective_rate` on the card `card_recommendation` would
    pick. Expiring credits stay in `unused_credits` so they are not counted twice.
    """
    recommendation = card_recommendation(fiduciary, action.reward_category, action.cost)
    return round(action.cost * recommendation.effective_rate, 2)


def _payload(state: WorldState) -> dict[str, Any]:
    return {
        "date": state.date.isoformat(),
        "fiduciary": state.fiduciary.model_dump(mode="json"),
        "candidate_actions": [action.model_dump(mode="json") for action in state.candidate_actions],
    }


def _system(reconcile: bool) -> str:
    tools = ["card_recommendation", "unused_credits", "cost_of_deferral"]
    if reconcile:
        tools.append("reconcile")
    listed = ", ".join(tools)
    # The schema is spelled out because a model left to guess wrote its own
    # keys, and a batch that fails validation leaves the arbiter nothing to score.
    return (
        "You are the fiduciary agent. You see only the fiduciary slice and the "
        "candidate actions. Reply with JSON only, no prose, no code fence. Either "
        f'{{"tool": one of {listed}, "args": {{}}}} or '
        '{"verdicts": [...]} with exactly one verdict per candidate action. '
        "Each verdict is an object with exactly these keys and no others: "
        '"agent": "fiduciary"; '
        '"action_id": the candidate action id; '
        '"score": a number from -1 to 1, this domain\'s view of the action; '
        '"hard_violation": true only if the action breaks a non-negotiable '
        "money constraint; "
        '"confidence": a number from 0 to 1; '
        '"rationale": at most 280 characters; '
        '"evidence": a non-empty list of the dotted paths you relied on, each a '
        "path in the slice you were shown, written like "
        "fiduciary.cards[0].rates.dining; "
        '"input_anomaly": null. '
        "Do not describe a numeric discrepancy in the rationale."
    )


class FiduciaryAgent:
    def __init__(
        self,
        client: LLMClient,
        *,
        reconcile: bool = False,
        tool_cap: int = 8,
        token_budget: int = 8000,
    ) -> None:
        self.client = client
        self.reconcile = reconcile
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
            raise TypeError("fiduciary agent cannot read a foreign slice")
        fid = state.fiduciary
        actions = list(state.candidate_actions)
        observations, anomaly, truncated = self._observe(state, fid, actions)
        handlers = self._handlers(fid, actions, state.date)
        user = json.dumps({"state": _payload(state), "observations": observations}, sort_keys=True)
        loop = BoundedLoop(self.client, tool_cap=self.tool_cap, token_budget=self.token_budget)
        result = loop.run(
            [Message("system", _system(self.reconcile)), Message("user", user)],
            handlers,
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            tool_calls=len(observations),
            truncated=truncated,
        )
        if result.parse_failed:
            return result
        # The model does not get to invent or suppress the discrepancy.
        stamped = [
            verdict.model_copy(update={"input_anomaly": anomaly}) for verdict in result.verdicts
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
        self,
        state: WorldState,
        fid: Fiduciary,
        actions: list[CandidateAction],
    ) -> tuple[list[dict[str, Any]], InputAnomaly | None, bool]:
        rows: list[dict[str, Any]] = []
        anomaly: InputAnomaly | None = None
        truncated = False

        def room_for(name: str, result: Any) -> None:
            nonlocal truncated
            if len(rows) >= self.tool_cap:
                truncated = True
                return
            rows.append({"tool": name, "result": result})

        if self.reconcile and len(rows) < self.tool_cap:
            anomaly = reconcile_ledger(fid)
            room_for(
                "reconcile",
                anomaly.model_dump(mode="json") if anomaly is not None else None,
            )
        elif self.reconcile:
            truncated = True
        if len(rows) < self.tool_cap:
            room_for(
                "unused_credits",
                [c.model_dump(mode="json") for c in unused_credits(fid, state.date)],
            )
        else:
            truncated = True
        for action in actions:
            if len(rows) >= self.tool_cap:
                truncated = True
                break
            rec = card_recommendation(fid, action.reward_category, action.cost)
            room_for(
                "card_recommendation",
                {
                    "action_id": action.id,
                    "card_id": rec.card_id,
                    "effective_rate": rec.effective_rate,
                },
            )
            if len(rows) >= self.tool_cap:
                truncated = True
                break
            room_for(
                "cost_of_deferral",
                {"action_id": action.id, "cost": cost_of_deferral(fid, action)},
            )
        return rows, anomaly, truncated

    def _handlers(
        self, fid: Fiduciary, actions: list[CandidateAction], as_of: date
    ) -> dict[str, Callable[[dict[str, Any]], str]]:
        by_id = {action.id: action for action in actions}

        def recommend(args: dict[str, Any]) -> str:
            category = args.get("reward_category")
            if category is not None and not isinstance(category, str):
                raise TypeError("reward_category must be a string")
            rec = card_recommendation(fid, category, float(args["amount"]))
            return json.dumps({"card_id": rec.card_id, "effective_rate": rec.effective_rate})

        def credits(_: dict[str, Any]) -> str:
            return json.dumps([c.model_dump(mode="json") for c in unused_credits(fid, as_of)])

        def defer(args: dict[str, Any]) -> str:
            action = by_id.get(str(args["action_id"]))
            if action is None:
                raise KeyError(f"unknown action {args.get('action_id')}")
            return json.dumps({"cost": cost_of_deferral(fid, action)})

        tools: dict[str, Callable[[dict[str, Any]], str]] = {
            "card_recommendation": recommend,
            "unused_credits": credits,
            "cost_of_deferral": defer,
        }
        if self.reconcile:

            def ledger(_: dict[str, Any]) -> str:
                found = reconcile_ledger(fid)
                body = found.model_dump(mode="json") if found is not None else None
                return json.dumps(body)

            tools["reconcile"] = ledger
        return tools
