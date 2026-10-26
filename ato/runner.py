"""One scenario, one condition, one log row.

The gate in the benchmark build is this write. The row is appended only after
the decision and the blame call have both returned, so a doubt about the
choice cannot be recorded as if it had chosen.

Wellness and productivity agents are not in this runner. A domain with no
verdict contributes 0, which is the scoring rule, so the row does not invent
verdicts those agents did not produce.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from ato.agents.fiduciary import FiduciaryAgent
from ato.arbiter import VerdictRound, arbitrate, attribute_blame
from ato.faults import LIBRARY, FaultRecord, FaultSpec, inject_output_fault, inject_state_fault
from ato.llm import LLMClient, LLMResponse, Message, temperature_sent
from ato.provider import FixtureProvider
from ato.schema import AgentVerdict, WorldState

_DECOMPOSED = frozenset({"B", "C1a", "C1b", "C0", "D"})
_DETECTABLE = frozenset({"C1a", "C1b", "D"})
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

_COLUMNS = (
    "created_at",
    "scenario_id",
    "condition",
    "pair_key",
    "seed",
    "temperature",
    "model",
    "fault_record",
    "chosen_action",
    "blame_agent",
    "blame_field",
    "blame_confidence",
    "blame_parse_failed",
    "prompt_hashes",
    "response_hashes",
    "input_tokens",
    "output_tokens",
    "truncated",
    "agent_parse_failed",
    "agent_verdict_count",
    "blame_refused",
    "reconcile_called",
    "temperature_sent",
)

# Nullable only where null is a true answer: no fault, no choice, no blame, or
# no temperature reached the provider.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    created_at TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    pair_key TEXT NOT NULL,
    seed INTEGER NOT NULL,
    temperature REAL NOT NULL,
    model TEXT NOT NULL,
    fault_record TEXT,
    chosen_action TEXT,
    blame_agent TEXT,
    blame_field TEXT,
    blame_confidence REAL,
    blame_parse_failed INTEGER NOT NULL,
    prompt_hashes TEXT NOT NULL,
    response_hashes TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    truncated INTEGER NOT NULL,
    agent_parse_failed INTEGER NOT NULL,
    agent_verdict_count INTEGER NOT NULL,
    blame_refused INTEGER NOT NULL,
    reconcile_called INTEGER NOT NULL,
    temperature_sent REAL
)
"""


def dry_run_responses(state: WorldState) -> list[str]:
    """Canned fiduciary verdicts and one blame answer. No network, and a unique winner."""
    verdicts = []
    last = len(state.candidate_actions) - 1
    for index, action in enumerate(state.candidate_actions):
        verdict = AgentVerdict(
            agent="fiduciary",
            action_id=action.id,
            score=0.8 if index == last else 0.1,
            hard_violation=False,
            confidence=0.7,
            rationale="Dry-run canned score.",
            evidence=["fiduciary.discretionary_remaining"],
        )
        verdicts.append(json.loads(verdict.model_dump_json()))
    blame = {
        "agent": "fiduciary",
        "field": "fiduciary.discretionary_remaining",
        "confidence": 0.5,
    }
    return [json.dumps({"verdicts": verdicts}), json.dumps(blame)]


def execute(
    *,
    scenario_id: str,
    condition: str,
    seed: int,
    temperature: float,
    model: str,
    db_path: Path,
    cache_dir: Path,
    dry_run: bool = False,
    fault_id: str | None = None,
    client: LLMClient | None = None,
    fixtures_dir: Path | None = None,
    max_tokens: int = 2048,
    token_budget: int = 8000,
) -> None:
    spec = _lookup(fault_id)
    _check_condition(condition, spec)
    state = FixtureProvider(fixtures_dir or _FIXTURES).load(scenario_id)
    record: FaultRecord | None = None
    if spec is not None and spec.surface == "state":
        state, record = inject_state_fault(state, spec)
    inner = _client(client, dry_run, state, cache_dir)
    recording = _RecordingClient(inner)
    agent = FiduciaryAgent(recording, reconcile=condition == "C1b")
    run = agent.run(
        state,
        model=model,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    )
    verdicts = run.verdicts
    if spec is not None and spec.surface == "output":
        verdicts, record = inject_output_fault(verdicts, spec)
    decision = arbitrate(
        state,
        [
            VerdictRound(
                verdicts,
                input_tokens=run.input_tokens,
                output_tokens=run.output_tokens,
            )
        ],
        token_budget=token_budget,
    )
    blame = attribute_blame(
        state,
        verdicts,
        decision,
        client=recording,
        model=model,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        surface_evidence=condition == "D",
    )
    _insert(
        db_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "scenario_id": scenario_id,
            "condition": condition,
            "pair_key": f"{scenario_id}:{seed}",
            "seed": seed,
            "temperature": temperature,
            "model": model,
            "fault_record": record.model_dump_json() if record is not None else None,
            "chosen_action": decision.action_id,
            "blame_agent": blame.agent,
            "blame_field": blame.field,
            "blame_confidence": blame.confidence,
            "blame_parse_failed": int(blame.parse_failed),
            "prompt_hashes": json.dumps(recording.prompt_hashes),
            "response_hashes": json.dumps(recording.response_hashes),
            "input_tokens": recording.input_tokens,
            "output_tokens": recording.output_tokens,
            "truncated": int(run.truncated or decision.truncated),
            "agent_parse_failed": int(run.parse_failed),
            "agent_verdict_count": len(verdicts),
            "blame_refused": int(blame.refused),
            "reconcile_called": int(blame.reconcile_called),
            "temperature_sent": temperature_sent(inner, temperature),
        },
    )


def _client(
    client: LLMClient | None,
    dry_run: bool,
    state: WorldState,
    cache_dir: Path,
) -> LLMClient:
    from ato.llm import CachingClient, FakeClient

    # Imported here so a dry run never constructs the vendor client.
    if client is not None:
        inner: LLMClient = client
    elif dry_run:
        inner = FakeClient(in_order=dry_run_responses(state))
    else:
        from ato.llm import AnthropicClient

        inner = AnthropicClient()
    return CachingClient(inner, cache_dir=cache_dir)


def _lookup(fault_id: str | None) -> FaultSpec | None:
    if fault_id is None:
        return None
    for spec in LIBRARY:
        if spec.fault_id == fault_id:
            return spec
    raise ValueError(f"unknown fault {fault_id}")


def _check_condition(condition: str, spec: FaultSpec | None) -> None:
    if condition == "A":
        raise ValueError("condition A is a single prompt and is not run here")
    if condition not in _DECOMPOSED:
        raise ValueError(f"unknown condition {condition}")
    if condition == "B" and spec is not None:
        raise ValueError("condition B is clean; it does not take a fault")
    if spec is None:
        return
    if condition == "C0" and spec.tier != "undetectable":
        raise ValueError(f"{spec.fault_id} is detectable and cannot be a C0 fault")
    if condition in _DETECTABLE and spec.tier != "detectable":
        raise ValueError(f"{spec.fault_id} is undetectable and cannot be a {condition} fault")


def _insert(db_path: Path, row: dict[str, object]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    names = ", ".join(_COLUMNS)
    marks = ", ".join("?" for _ in _COLUMNS)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_SCHEMA)
        existing = [info[1] for info in conn.execute("PRAGMA table_info(runs)")]
        if existing != list(_COLUMNS):
            # Mixing rows from two schemas would score old runs as if they
            # had the new columns. Start a new log instead.
            raise ValueError(f"{db_path} was written with an older run-log schema; use a new --db")
        conn.execute(
            f"INSERT INTO runs ({names}) VALUES ({marks})",
            tuple(row[name] for name in _COLUMNS),
        )


def _prompt_hash(messages: Sequence[Message]) -> str:
    blob = json.dumps(
        [{"role": message.role, "content": message.content} for message in messages],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class _RecordingClient:
    """Counts tokens and hashes every call. from_cache is not recorded.

    A replay must match the miss. Storing the cache flag would make the second
    row differ from the first by something other than its timestamp.
    """

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.prompt_hashes: list[str] = []
        self.response_hashes: list[str] = []
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
    ) -> LLMResponse:
        self.prompt_hashes.append(_prompt_hash(messages))
        response = self.inner.complete(
            messages,
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            response_model=response_model,
        )
        digest = hashlib.sha256(response.text.encode()).hexdigest()
        self.response_hashes.append(digest)
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        return response
