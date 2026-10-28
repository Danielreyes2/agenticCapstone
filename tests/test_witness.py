"""A witnessed call is the cached replies in order, not a new model call."""

from __future__ import annotations

import json
import sqlite3

from ato.faults import LIBRARY, inject_state_fault
from ato.schema import WorldState
from viz import board, witness


def test_the_witness_replays_the_fiduciary_and_the_blame_call(tmp_path, monkeypatch) -> None:
    clean = WorldState.model_validate_json((board.FIXTURES / "s014.json").read_text())
    spec = next(item for item in LIBRARY if item.fault_id == "d_fid_balance")
    state, record = inject_state_fault(clean, spec)
    _write_log(tmp_path, record.model_dump_json())
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_call(
        cache / "fid.json",
        speaker="fiduciary",
        user={
            "observations": [{"tool": "unused_credits", "result": []}],
            "state": {
                "date": state.date.isoformat(),
                "fiduciary": state.fiduciary.model_dump(mode="json"),
                "candidate_actions": [
                    action.model_dump(mode="json") for action in state.candidate_actions
                ],
            },
        },
        response=json.dumps(
            {
                "verdicts": [
                    {
                        "action_id": "a2",
                        "score": 0.4,
                        "rationale": "Cooking stays inside the money left.",
                    }
                ]
            }
        ),
    )
    _write_call(
        cache / "blame.json",
        speaker="blame",
        user={
            "committed_action_id": "a2",
            "state": state.model_dump(mode="json"),
            "verdicts": [],
            "field_paths": ["fiduciary.discretionary_remaining"],
        },
        response=json.dumps(
            {"agent": "fiduciary", "field": "fiduciary.discretionary_remaining", "confidence": 0.6}
        ),
    )
    monkeypatch.setattr(board, "RUNS", tmp_path)
    monkeypatch.setattr(witness, "CACHE_DIRS", (cache,))

    viewed = witness.witness_call("s014", model="claude-opus-5-5", condition="C1a", seed=1)
    texts = [beat["text"] for beat in viewed["beats"]]
    speakers = [beat["speaker"] for beat in viewed["beats"]]
    assert "Not called." in texts
    assert any("unused_credits" in text for text in texts)
    assert any("Cooking stays inside" in text for text in texts)
    assert any(text == "I commit cook." for text in texts)
    assert any("discretionary_remaining" in text for text in texts)
    assert speakers.index("fiduciary") < speakers.index("arbiter")
    assert viewed["action"]["id"] == "a2"


def _write_log(directory, fault_record: str) -> None:
    database = directory / "sample.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute(
            """
            CREATE TABLE runs (
                condition TEXT, seed INTEGER, model TEXT, fault_record TEXT,
                chosen_action TEXT, blame_agent TEXT, blame_field TEXT,
                blame_confidence REAL, scenario_id TEXT, temperature_sent REAL
            )
            """
        )
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "C1a",
                1,
                "claude-opus-5-5",
                fault_record,
                "a2",
                "fiduciary",
                "fiduciary.discretionary_remaining",
                0.6,
                "s014",
                None,
            ),
        )


def _write_call(path, *, speaker: str, user: dict, response: str) -> None:
    system = (
        "You are the fiduciary agent. You see only the fiduciary slice."
        if speaker == "fiduciary"
        else "The action is already committed and will not change."
    )
    path.write_text(
        json.dumps(
            {
                "fingerprint": {
                    "model": "claude-opus-5-5",
                    "seed": 1,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(user)},
                    ],
                },
                "response": {"text": response},
            }
        )
    )
