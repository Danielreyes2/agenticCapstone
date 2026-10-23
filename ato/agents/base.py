"""Bounded reason-act-observe loop.

The loop is hand-rolled because the tool cap and the token budget have to be
visible to the caller. A framework supervisor would hide the truncation the
run log has to record. Truncation stays on the run, not on `AgentVerdict`:
that schema is the domain contract, and a budget flag is run metadata.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ato.llm import LLMClient, LLMResponse, Message
from ato.schema import AgentVerdict


@dataclass
class AgentRun:
    verdicts: list[AgentVerdict]
    truncated: bool
    tool_calls: int
    input_tokens: int
    output_tokens: int
    parse_failed: bool


@dataclass
class _Turn:
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    verdicts: list[AgentVerdict] | None = None
    error: str | None = None


def _interpret(text: str) -> _Turn:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return _Turn(error=str(exc))
    if not isinstance(payload, dict):
        return _Turn(error="response is not a JSON object")
    if "verdicts" in payload:
        raw = payload["verdicts"]
        if not isinstance(raw, list):
            return _Turn(error="verdicts is not a list")
        try:
            return _Turn(verdicts=[AgentVerdict.model_validate(item) for item in raw])
        except ValidationError as exc:
            return _Turn(error=str(exc))
    if "tool" in payload:
        name = payload["tool"]
        args = payload.get("args", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            return _Turn(error="tool call needs a string name and an args object")
        return _Turn(tool_name=name, tool_args=args)
    return _Turn(error="response has neither a tool call nor verdicts")


class BoundedLoop:
    """One model call per turn. A bad turn is retried once, then the run stops."""

    def __init__(self, client: LLMClient, *, tool_cap: int, token_budget: int) -> None:
        self.client = client
        self.tool_cap = tool_cap
        self.token_budget = token_budget

    def run(
        self,
        messages: Sequence[Message],
        tools: Mapping[str, Callable[[dict[str, Any]], str]],
        *,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
        tool_calls: int = 0,
        truncated: bool = False,
    ) -> AgentRun:
        history = list(messages)
        in_tok = 0
        out_tok = 0

        while True:
            _, history, in_tok, out_tok, turn = self._turn(
                history,
                in_tok,
                out_tok,
                model=model,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
            )
            over_budget = in_tok + out_tok > self.token_budget
            if turn.error is not None:
                return AgentRun([], truncated or over_budget, tool_calls, in_tok, out_tok, True)
            if turn.verdicts is not None:
                return AgentRun(
                    turn.verdicts, truncated or over_budget, tool_calls, in_tok, out_tok, False
                )
            if over_budget or tool_calls >= self.tool_cap:
                return AgentRun([], True, tool_calls, in_tok, out_tok, False)
            tool_calls += 1
            handler = tools.get(turn.tool_name or "")
            if handler is None:
                observation = json.dumps({"error": f"unknown tool {turn.tool_name}"})
            else:
                try:
                    observation = handler(turn.tool_args)
                except (KeyError, TypeError, ValueError) as exc:
                    observation = json.dumps({"error": str(exc)})
            history.append(Message("user", observation))

    def _turn(
        self,
        messages: list[Message],
        in_tok: int,
        out_tok: int,
        *,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> tuple[LLMResponse, list[Message], int, int, _Turn]:
        first = self._complete(
            messages, model=model, temperature=temperature, seed=seed, max_tokens=max_tokens
        )
        in_tok += first.input_tokens
        out_tok += first.output_tokens
        turn = _interpret(first.text)
        if turn.error is None:
            return first, [*messages, Message("assistant", first.text)], in_tok, out_tok, turn
        # A retry is a different request. One only; a second failure stops the run.
        retry = [
            *messages,
            Message("assistant", first.text),
            Message(
                "user",
                f"That response failed validation: {turn.error}. Return only JSON, "
                "either a tool call or a verdict batch. No prose, no code fence.",
            ),
        ]
        second = self._complete(
            retry, model=model, temperature=temperature, seed=seed, max_tokens=max_tokens
        )
        in_tok += second.input_tokens
        out_tok += second.output_tokens
        history = [*retry, Message("assistant", second.text)]
        return second, history, in_tok, out_tok, _interpret(second.text)

    def _complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> LLMResponse:
        return self.client.complete(
            messages,
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            response_model=None,
        )
