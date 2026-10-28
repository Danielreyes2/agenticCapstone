"""The condition sweep plans paired runs. A dry run never calls the network."""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from ato.faults import LIBRARY
from ato.generate import generate
from ato.sweep import plan, run_sweep

_FAULTED = ("C1a", "C1b", "C0", "D")


def test_plan_pairs_every_faulted_run_with_a_clean_one() -> None:
    runs = plan(generate(12, seed=0))
    assert len(runs) == 12 * 9
    assert {run.condition for run in runs} == {"B", "C1a", "C1b", "C0", "D"}
    clean = {f"{run.scenario_id}:{run.seed}" for run in runs if run.condition == "B"}
    assert len([run for run in runs if run.condition == "B"]) == 12 * 5
    for run in runs:
        if run.condition == "B":
            assert run.fault_id is None
            assert run.seed in (1, 2, 3, 4, 5)
        else:
            assert run.seed == 1
            assert f"{run.scenario_id}:1" in clean
            assert run.fault_id is not None


def test_detectable_conditions_share_a_fault_and_the_control_does_not() -> None:
    runs = plan(generate(12, seed=0))
    detectable = {spec.fault_id for spec in LIBRARY if spec.tier == "detectable"}
    control = {
        spec.fault_id for spec in LIBRARY if spec.tier == "undetectable" and spec.surface == "state"
    }
    by_scenario: dict[str, set[str]] = {}
    for run in runs:
        if run.condition in ("C1a", "C1b", "D"):
            assert run.fault_id in detectable
            by_scenario.setdefault(run.scenario_id, set()).add(str(run.fault_id))
        if run.condition == "C0":
            assert run.fault_id in control
    assert all(len(faults) == 1 for faults in by_scenario.values())
    c1a = [run.fault_id for run in runs if run.condition == "C1a"]
    counts = Counter(c1a)
    assert max(counts.values()) - min(counts.values()) <= 1
    agents = {spec.target_agent for spec in LIBRARY if spec.fault_id in counts}
    assert agents == {"fiduciary", "wellness", "productivity"}


def test_live_sweep_without_confirm_does_not_start(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--confirm"):
        run_sweep(
            count=2,
            scenario_seed=0,
            db_path=tmp_path / "sweep.sqlite",
            cache_dir=tmp_path / "cache",
            scenarios_dir=tmp_path / "scenarios",
            dry_run=False,
            confirm=False,
        )
    assert not (tmp_path / "sweep.sqlite").exists()
    assert not (tmp_path / "scenarios").exists()


def test_dry_run_writes_a_log_the_harness_can_score(tmp_path: Path) -> None:
    result = run_sweep(
        count=1,
        scenario_seed=0,
        db_path=tmp_path / "sweep.sqlite",
        cache_dir=tmp_path / "cache",
        scenarios_dir=tmp_path / "scenarios",
        dry_run=True,
        confirm=False,
    )
    assert result.planned == 9
    assert result.completed == 9
    assert result.skipped == 0
    report = result.evaluation
    assert report.schema_validity.k == 9
    assert report.stability_mean == 1
    assert report.hard_constraint_violation is None
    assert report.output == {}
    assert report.state["C1a"].localization.k == 1
    assert report.state["C0"].localization.k == 0
    assert set(report.state) == set(_FAULTED)
    with sqlite3.connect(tmp_path / "sweep.sqlite") as conn:
        rows = list(conn.execute("SELECT temperature, temperature_sent, model FROM runs"))
    assert rows
    assert all(
        temperature == 0 and sent is None and model == "fake" for temperature, sent, model in rows
    )

    again = run_sweep(
        count=1,
        scenario_seed=0,
        db_path=tmp_path / "sweep.sqlite",
        cache_dir=tmp_path / "cache",
        scenarios_dir=tmp_path / "scenarios",
        dry_run=True,
        confirm=False,
    )
    assert again.completed == 0
    assert again.skipped == 9


def test_token_cap_stops_after_the_first_row(tmp_path: Path) -> None:
    result = run_sweep(
        count=1,
        scenario_seed=0,
        db_path=tmp_path / "sweep.sqlite",
        cache_dir=tmp_path / "cache",
        scenarios_dir=tmp_path / "scenarios",
        dry_run=True,
        confirm=False,
        token_cap=0,
    )
    assert result.completed == 1
    assert result.stopped
