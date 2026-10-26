"""The runner writes one log row. A replay matches it except the timestamp."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from ato.llm import FakeClient, LLMResponse, Message
from ato.provider import FixtureProvider
from ato.runner import dry_run_responses, execute

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "created_at",
    "scenario_id",
    "condition",
    "pair_key",
    "seed",
    "temperature",
    "model",
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
)


def _rows(db: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM runs ORDER BY rowid"))
    finally:
        conn.close()


def _run(tmp_path: Path, **overrides: object) -> Path:
    db = tmp_path / "runs.sqlite"
    args: dict[str, object] = {
        "scenario_id": "s014",
        "condition": "B",
        "seed": 7,
        "temperature": 0.0,
        "model": "fake",
        "db_path": db,
        "cache_dir": tmp_path / "cache",
        "dry_run": True,
    }
    args.update(overrides)
    execute(**args)  # type: ignore[arg-type]
    return db


def test_clean_s014_writes_one_complete_row(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite"
    cache = tmp_path / "cache"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run.py",
            "--scenario",
            "s014",
            "--condition",
            "B",
            "--dry-run",
            "--seed",
            "7",
            "--temperature",
            "0",
            "--db",
            str(db),
            "--cache-dir",
            str(cache),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    rows = _rows(db)
    assert len(rows) == 1
    row = rows[0]
    for column in REQUIRED:
        assert row[column] is not None
    assert row["fault_record"] is None
    assert row["scenario_id"] == "s014"
    assert row["condition"] == "B"
    assert row["pair_key"] == "s014:7"
    assert row["seed"] == 7
    assert row["temperature"] == 0
    assert row["model"] == "fake"
    assert row["chosen_action"] == "a2"
    assert row["blame_agent"] == "fiduciary"
    assert row["blame_field"] == "fiduciary.discretionary_remaining"
    assert row["truncated"] == 0
    assert json.loads(row["prompt_hashes"])
    assert json.loads(row["response_hashes"])


def test_replay_matches_except_the_timestamp(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite"
    cache = tmp_path / "cache"
    kwargs = {
        "scenario_id": "s014",
        "condition": "B",
        "seed": 7,
        "temperature": 0.0,
        "model": "fake",
        "db_path": db,
        "cache_dir": cache,
    }
    execute(dry_run=True, **kwargs)

    class _Boom:
        def __init__(self) -> None:
            self.calls = 0

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
            self.calls += 1
            raise AssertionError("cache miss")

    boom = _Boom()
    execute(client=boom, dry_run=False, **kwargs)
    assert boom.calls == 0
    rows = _rows(db)
    assert len(rows) == 2
    first = dict(rows[0])
    second = dict(rows[1])
    del first["created_at"]
    del second["created_at"]
    assert first == second


def test_clean_and_faulted_share_a_pair_key(tmp_path: Path) -> None:
    clean = _run(tmp_path, condition="C1a")
    faulted_db = tmp_path / "faulted.sqlite"
    execute(
        scenario_id="s014",
        condition="C1a",
        fault_id="d_fid_balance",
        seed=7,
        temperature=0.0,
        model="fake",
        db_path=faulted_db,
        cache_dir=tmp_path / "fault-cache",
        dry_run=True,
    )
    clean_row = _rows(clean)[0]
    faulted_row = _rows(faulted_db)[0]
    assert clean_row["pair_key"] == faulted_row["pair_key"] == "s014:7"
    assert clean_row["seed"] == faulted_row["seed"]
    assert clean_row["fault_record"] is None
    assert "d_fid_balance" in faulted_row["fault_record"]


def test_condition_d_surfaces_evidence(tmp_path: Path) -> None:
    state = FixtureProvider(ROOT / "fixtures").load("s014")
    spy = FakeClient(in_order=dry_run_responses(state))
    execute(
        scenario_id="s014",
        condition="D",
        seed=7,
        temperature=0.0,
        model="fake",
        db_path=tmp_path / "runs.sqlite",
        cache_dir=tmp_path / "cache",
        client=spy,
    )
    payload = json.loads(spy.calls[1]["messages"][-1].content)
    assert "evidence" in payload
    assert payload["evidence"][0]["evidence"] == ["fiduciary.discretionary_remaining"]


def test_condition_b_rejects_a_fault(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="condition B"):
        _run(tmp_path, fault_id="d_fid_balance")


def test_c0_rejects_a_detectable_fault(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="detectable"):
        _run(tmp_path, condition="C0", fault_id="d_fid_balance")
