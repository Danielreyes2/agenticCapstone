"""The gate plan is fixed. A dry run never calls the network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ato.gate import plan, render_bar, run_gate, summarize
from ato.llm import FakeClient
from ato.runner import execute


def test_full_plan_is_the_agreed_split() -> None:
    runs = plan(pilot=False)
    assert len(runs) == 35
    clean = [run for run in runs if run.condition == "B"]
    assert len(clean) == 25
    assert {run.scenario_id for run in clean} == {"s014", "s021", "s028", "s040", "s044"}
    assert [run.fault_id for run in runs if run.condition == "C1a"] == ["d_fid_balance"] * 5
    assert [run.fault_id for run in runs if run.condition == "C0"] == ["u_fid_rent"] * 5


def test_progress_bar_fills() -> None:
    assert render_bar(0, 35, "s014") == f"[{'-' * 30}] 0/35  s014"
    assert render_bar(35, 35, "done").startswith(f"[{'#' * 30}] 35/35")


def test_pilot_is_one_clean_call() -> None:
    pilot = plan(pilot=True)
    got = [(run.scenario_id, run.condition, run.seed, run.fault_id) for run in pilot]
    assert got == [("s014", "B", 1, None)]


def test_live_batch_without_confirm_does_not_start(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--confirm"):
        run_gate(
            db_path=tmp_path / "gate.sqlite",
            cache_dir=tmp_path / "cache",
            dry_run=False,
            pilot=False,
            confirm=False,
        )
    assert not (tmp_path / "gate.sqlite").exists()


def test_rescue_is_five_condition_d_runs(tmp_path: Path) -> None:
    report = run_gate(
        db_path=tmp_path / "gate.sqlite",
        cache_dir=tmp_path / "cache",
        dry_run=True,
        pilot=False,
        confirm=False,
        rescue=True,
    )
    assert report["completed"] == 5
    assert report["rescue_localization"] == 1


def test_dry_run_pilot_writes_one_row(tmp_path: Path) -> None:
    report = run_gate(
        db_path=tmp_path / "gate.sqlite",
        cache_dir=tmp_path / "cache",
        dry_run=True,
        pilot=True,
        confirm=False,
    )
    assert report["completed"] == 1
    assert report["rows"] == 1
    assert report["approx_usd"] == 0


def test_dry_run_reports_the_canned_pattern(tmp_path: Path) -> None:
    report = run_gate(
        db_path=tmp_path / "gate.sqlite",
        cache_dir=tmp_path / "cache",
        dry_run=True,
        pilot=False,
        confirm=False,
    )
    assert report["completed"] == 35
    assert report["stability"] == 1
    assert report["false_positive_rate"] == 1
    assert report["detectable_localization"] == 1
    assert report["control_localization"] == 0
    assert report["agent_failures"] == 0
    assert report["stability_excluded"] == 0
    assert report["agent_level"] == {"C1a": 1, "C0": 1, "D": None}
    assert report["abstain_rate"] == {"B": 0, "C1a": 0, "C0": 0}
    assert report["reconcile_rate"] == {"B": 0, "C1a": 0, "C0": 0}
    assert report["blame_refusals"] == 0


def _execute(db: Path, cache: Path, scenario: str, condition: str, seed: int, **kw: object) -> None:
    execute(
        scenario_id=scenario,
        condition=condition,
        seed=seed,
        temperature=0.0,
        model="fake",
        db_path=db,
        cache_dir=cache,
        **kw,  # type: ignore[arg-type]
    )


def test_a_run_without_agent_verdicts_is_a_failure_not_a_decision(tmp_path: Path) -> None:
    db = tmp_path / "gate.sqlite"
    cache = tmp_path / "cache"
    right = json.dumps(
        {"agent": "fiduciary", "field": "fiduciary.discretionary_remaining", "confidence": 0.9}
    )
    abstain = json.dumps({"agent": None, "field": None, "confidence": 0.9})
    _execute(db, cache, "s014", "B", 1, dry_run=True)
    _execute(db, cache, "s014", "B", 2, client=FakeClient(in_order=["no", "no", abstain]))
    _execute(
        db,
        cache,
        "s007",
        "C1a",
        1,
        fault_id="d_fid_balance",
        client=FakeClient(in_order=["no", "no", right]),
    )
    report = summarize(db)
    assert report["agent_failures"] == 2
    assert report["stability_excluded"] == 1
    assert report["stability"] == 1
    assert report["abstain_rate"] == {"B": 0.5, "C1a": 0}
    assert report["detectable_localization"] == 0


def test_full_run_skips_the_pilot_row(tmp_path: Path) -> None:
    db = tmp_path / "gate.sqlite"
    cache = tmp_path / "cache"
    run_gate(db_path=db, cache_dir=cache, dry_run=True, pilot=True, confirm=False)
    report = run_gate(db_path=db, cache_dir=cache, dry_run=True, pilot=False, confirm=False)
    assert report["skipped"] == 1
    assert report["completed"] == 34
    assert report["rows"] == 35


def test_token_cap_stops_after_the_first_row(tmp_path: Path) -> None:
    report = run_gate(
        db_path=tmp_path / "gate.sqlite",
        cache_dir=tmp_path / "cache",
        dry_run=True,
        pilot=False,
        confirm=False,
        token_cap=0,
    )
    assert report["completed"] == 1
    assert report["stopped"]


def test_a_named_model_is_sent_on_a_live_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def spy(**kwargs: object) -> None:
        seen.append(str(kwargs["model"]))

    monkeypatch.setattr("ato.gate.execute", spy)
    run_gate(
        db_path=tmp_path / "gate.sqlite",
        cache_dir=tmp_path / "cache",
        dry_run=False,
        pilot=True,
        confirm=False,
        model="claude-haiku-5-5",
    )
    assert seen == ["claude-haiku-5-5"]
