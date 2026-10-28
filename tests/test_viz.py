"""The viewer reads fixtures and may preview a fault. It does not write a run."""

from __future__ import annotations

import json
import sqlite3

from viz import board
from viz.board import fault_index, scenario_index, scenario_view


def test_every_fixture_is_listed_clean() -> None:
    ids = [row["id"] for row in scenario_index()]
    assert ids == [
        "s007",
        "s014",
        "s019",
        "s021",
        "s028",
        "s033",
        "s040",
        "s044",
        "s052",
        "s061",
    ]
    view = scenario_view("s014")
    assert view["fault"] is None
    assert view["violations"] == []
    assert view["fiduciary"]["discretionary_remaining"] == 140.0


def test_a_detectable_preview_breaks_the_invariant_and_an_undetectable_one_does_not() -> None:
    detectable = scenario_view("s014", "d_fid_balance")
    assert detectable["fault"]["before"] == 140.0
    assert detectable["fault"]["after"] == 180.0
    assert detectable["violations"][0]["rule"] == "F1"
    assert detectable["fiduciary"]["discretionary_remaining"] == 180.0

    control = scenario_view("s014", "u_fid_rent")
    assert control["violations"] == []
    assert control["fiduciary"]["liabilities"][0]["amount"] == 1130.0


def test_the_baseline_fault_is_refused_when_no_day_was_observed() -> None:
    view = scenario_view("s028", "d_wel_baseline")
    assert view["applicable"] is False
    assert view["fault"] is None
    assert view["violations"] == []


def test_an_abstain_on_the_control_reads_as_clear(tmp_path, monkeypatch) -> None:
    db = tmp_path / "sample.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE runs (
                condition TEXT, seed INTEGER, model TEXT, fault_record TEXT,
                chosen_action TEXT, blame_agent TEXT, blame_field TEXT,
                blame_confidence REAL, scenario_id TEXT, temperature_sent REAL
            )
            """
        )
        control = json.dumps(
            {
                "truth_agent": "fiduciary",
                "truth_field": "fiduciary.fixed_liabilities_upcoming[0].amount",
                "spec": {"fault_id": "u_fid_rent"},
            }
        )
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("C0", 1, "claude-opus-5-5", control, "a1", None, None, None, "s014", None),
        )
    monkeypatch.setattr(board, "RUNS", tmp_path)
    labels = [row["label"] for row in scenario_view("s014")["runs"]]
    assert labels == ["clear"]


def test_the_picker_offers_state_faults_only() -> None:
    ids = {row["id"] for row in fault_index()}
    assert "d_fid_balance" in ids
    assert "u_fid_rent" in ids
    assert "o_wel_conf" not in ids
