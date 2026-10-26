"""Bounded reason-act-observe loop.

The loop is hand-rolled because the tool cap and the token budget have to be
visible to the caller. A framework supervisor would hide the truncation the
run log has to record. Truncation stays on the run, not on `AgentVerdict`:
that schema is the domain contract, and a budget flag is run metadata.

The same loop serves the agents and the blame call. Only the shape of the
final answer differs, so the caller passes the parser for it.
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
class LoopResult[T]:
    final: T | None
    truncated: bool
    tool_calls: int
    input_tokens: int
    output_tokens: int
    parse_failed: bool
    tools_called: list[str] = field(default_factory=list)
    refused: bool = False


@dataclass
class _Turn[T]:
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    final: T | None = None
    error: str | None = None


def parse_verdicts(payload: dict[str, Any]) -> list[AgentVerdict]:
    if "verdicts" not in payload:
        raise ValueError("response has neither a tool call nor verdicts")
    raw = payload["verdicts"]
    if not isinstance(raw, list):
        raise ValueError("verdicts is not a list")
    return [AgentVerdict.model_validate(item) for item in raw]


def _interpret[T](text: str, finish: Callable[[dict[str, Any]], T]) -> _Turn[T]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return _Turn(error=str(exc))
    if not isinstance(payload, dict):
        return _Turn(error="response is not a JSON object")
    if "tool" in payload:
        name = payload["tool"]
        args = payload.get("args", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            return _Turn(error="tool call needs a string name and an args object")
        return _Turn(tool_name=name, tool_args=args)
    try:
        return _Turn(final=finish(payload))
    except (ValidationError, ValueError) as exc:
        return _Turn(error=str(exc))


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
        result = self.run_until(
            messages,
            tools,
            finish=parse_verdicts,
            finish_label="a verdict batch",
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            tool_calls=tool_calls,
            truncated=truncated,
        )
        return AgentRun(
            result.final or [],
            result.truncated,
            result.tool_calls,
            result.input_tokens,
            result.output_tokens,
            result.parse_failed,
        )

    def run_until[T](
        self,
        messages: Sequence[Message],
        tools: Mapping[str, Callable[[dict[str, Any]], str]],
        *,
        finish: Callable[[dict[str, Any]], T],
        finish_label: str,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
        tool_calls: int = 0,
        truncated: bool = False,
    ) -> LoopResult[T]:
        history = list(messages)
        in_tok = 0
        out_tok = 0
        called: list[str] = []
        refused = False

        def done(final: T | None, cut: bool, failed: bool) -> LoopResult[T]:
            return LoopResult(
                final, cut, tool_calls, in_tok, out_tok, failed, list(called), refused
            )

        while True:
            responses, history, turn = self._turn(
                history,
                finish,
                finish_label,
                model=model,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
            )
            for response in responses:
                in_tok += response.input_tokens
                out_tok += response.output_tokens
                refused = refused or response.finish_reason == "refusal"
            over_budget = in_tok + out_tok > self.token_budget
            if turn.error is not None:
                return done(None, truncated or over_budget, True)
            if turn.tool_name is None:
                return done(turn.final, truncated or over_budget, False)
            if over_budget or tool_calls >= self.tool_cap:
                return done(None, True, False)
            tool_calls += 1
            called.append(turn.tool_name)
            handler = tools.get(turn.tool_name)
            if handler is None:
                observation = json.dumps({"error": f"unknown tool {turn.tool_name}"})
            else:
                try:
                    observation = handler(turn.tool_args)
                except (KeyError, TypeError, ValueError) as exc:
                    observation = json.dumps({"error": str(exc)})
            history.append(Message("user", observation))

    def _turn[T](
        self,
        messages: list[Message],
        finish: Callable[[dict[str, Any]], T],
        finish_label: str,
        *,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> tuple[list[LLMResponse], list[Message], _Turn[T]]:
        first = self._complete(
            messages, model=model, temperature=temperature, seed=seed, max_tokens=max_tokens
        )
        turn = _interpret(first.text, finish)
        if turn.error is None:
            return [first], [*messages, Message("assistant", first.text)], turn
        # A retry is a different request. One only; a second failure stops the run.
        retry = [
            *messages,
            Message("assistant", first.text),
            Message(
                "user",
                f"That response failed validation: {turn.error}. Return only JSON, "
                f"either a tool call or {finish_label}. No prose, no code fence.",
            ),
        ]
        second = self._complete(
            retry, model=model, temperature=temperature, seed=seed, max_tokens=max_tokens
        )
        history = [*retry, Message("assistant", second.text)]
        return [first, second], history, _interpret(second.text, finish)

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
