"""Agents do not receive each other's verdicts. Each call sees only its own slice."""

from __future__ import annotations

import json

from ato.agents.fiduciary import FiduciaryAgent
from ato.agents.productivity import ProductivityAgent
from ato.agents.wellness import WellnessAgent
from ato.llm import LLMResponse, Message
from ato.schema import AgentName, AgentVerdict, WorldState

CALL = {"model": "test-model", "temperature": 0.0, "seed": 7, "max_tokens": 512}


def _batch(agent: AgentName, marker: str) -> str:
    verdicts = []
    for action_id in ("a1", "a2"):
        verdict = AgentVerdict(
            agent=agent,
            action_id=action_id,
            score=0.1,
            hard_violation=False,
            confidence=0.5,
            rationale=marker,
            evidence=[f"{agent}.marker"],
        )
        verdicts.append(json.loads(verdict.model_dump_json()))
    return json.dumps({"verdicts": verdicts})


class _Script:
    def __init__(self, text: str) -> None:
        self.calls: list[list[Message]] = []
        self.text = text

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
        return LLMResponse(text=self.text)


def test_agents_never_see_each_others_verdicts(clean: WorldState) -> None:
    markers = {
        "fiduciary": "ONLY-FIDUCIARY-VERDICT",
        "wellness": "ONLY-WELLNESS-VERDICT",
        "productivity": "ONLY-PRODUCTIVITY-VERDICT",
    }
    clients = {name: _Script(_batch(name, marker)) for name, marker in markers.items()}
    FiduciaryAgent(clients["fiduciary"]).run(clean, **CALL)
    WellnessAgent(clients["wellness"]).run(clean, **CALL)
    ProductivityAgent(clients["productivity"]).run(clean, **CALL)

    slices = {
        "fiduciary": {"date", "fiduciary", "candidate_actions"},
        "wellness": {"date", "wellness", "candidate_actions"},
        "productivity": {"date", "productivity", "candidate_actions"},
    }
    for name, client in clients.items():
        sent = "\n".join(message.content for message in client.calls[0])
        for other, marker in markers.items():
            if other != name:
                assert marker not in sent
        payload = json.loads(client.calls[0][1].content)
        assert set(payload["state"]) == slices[name]
