"""Metrics from a hand-built run log. The expected fractions were computed by hand.

Scenario ids here are not fixtures. A metric that opened the scenario files
would fail while loading them.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ato.eval import Proportion, evaluate

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
    "chosen_violates",
)


def _fault(surface: str, agent: str, field: str) -> str:
    return json.dumps(
        {
            "spec": {
                "fault_id": "hand",
                "surface": surface,
                "fault_class": "numeric_drift",
                "tier": "detectable",
                "target_agent": agent,
            },
            "truth_agent": agent,
            "truth_field": field,
            "before": 0,
            "after": 1,
        }
    )


def _row(
    scenario: str,
    condition: str,
    seed: int,
    action: str | None,
    *,
    fault: str | None = None,
    blame_agent: str | None = None,
    blame_field: str | None = None,
    truncated: int = 0,
    parse_failed: int = 0,
    verdicts: int = 2,
    violates: int = 0,
) -> tuple[object, ...]:
    return (
        "2026-01-01T00:00:00+00:00",
        scenario,
        condition,
        f"{scenario}:{seed}",
        seed,
        0.0,
        "fake",
        fault,
        action,
        blame_agent,
        blame_field,
        0.5 if blame_agent else 0.0,
        0,
        "[]",
        "[]",
        10,
        5,
        truncated,
        parse_failed,
        verdicts,
        0,
        0,
        None,
        violates,
    )


def _write(path: Path, rows: list[tuple[object, ...]], *, with_violations: bool = True) -> None:
    columns = _COLUMNS if with_violations else _COLUMNS[:-1]
    stored = rows if with_violations else [row[:-1] for row in rows]
    names = ", ".join(columns)
    types = " ".join(f"{name} TEXT," for name in columns)
    with sqlite3.connect(path) as conn:
        conn.execute(f"CREATE TABLE runs ({types[:-1]})")
        conn.executemany(
            f"INSERT INTO runs ({names}) VALUES ({', '.join('?' for _ in columns)})", stored
        )


def _check(got: Proportion, k: int, n: int, value: float, low: float, high: float) -> None:
    assert (got.k, got.n) == (k, n)
    assert got.value == pytest.approx(value)
    assert got.low == pytest.approx(low)
    assert got.high == pytest.approx(high)


def test_metrics_match_the_hand_checked_log(tmp_path: Path) -> None:
    state = _fault("state", "fiduciary", "fiduciary.discretionary_remaining")
    control = _fault("state", "fiduciary", "fiduciary.fixed_liabilities_upcoming[0].amount")
    output_hit = _fault("output", "productivity", "productivity.rationale")
    output_miss = _fault("output", "productivity", "productivity.confidence")
    # 9 clean rows with verdicts, 1 clean row that did not parse, 4 state faults,
    # 2 output faults, and 1 truncated state fault that must not enter a rate.
    rows = [
        _row("s1", "B", 1, "a1"),
        _row("s1", "B", 2, "a1"),
        _row("s1", "B", 3, "a1"),
        _row("s1", "B", 4, "a1"),
        _row("s1", "B", 5, "a2"),
        _row("s2", "B", 1, "a1"),
        _row("s2", "B", 2, "a1"),
        _row("s3", "B", 1, "a1"),
        _row("s3", "B", 2, "a1"),
        _row("s4", "B", 1, None, parse_failed=1, verdicts=0),
        _row(
            "s1",
            "C1a",
            1,
            "a2",
            fault=state,
            blame_agent="fiduciary",
            blame_field="fiduciary.discretionary_remaining",
        ),
        _row(
            "s1",
            "C1a",
            2,
            "a1",
            fault=state,
            blame_agent="fiduciary",
            blame_field="fiduciary.discretionary_remaining",
        ),
        _row(
            "s2",
            "C0",
            1,
            "a1",
            fault=control,
            blame_agent="wellness",
            blame_field="wellness.hrv_today",
            violates=1,
        ),
        _row("s2", "C0", 2, "a2", fault=control),
        _row(
            "s3",
            "C1a",
            1,
            "a1",
            fault=output_hit,
            blame_agent="productivity",
            blame_field="productivity.rationale",
        ),
        _row("s3", "C1a", 2, "a1", fault=output_miss, violates=1),
        _row(
            "s1",
            "C1a",
            3,
            "a2",
            fault=state,
            blame_agent="fiduciary",
            blame_field="fiduciary.discretionary_remaining",
            truncated=1,
            violates=1,
        ),
    ]
    db = tmp_path / "hand.sqlite"
    _write(db, rows)

    report = evaluate(db)

    _check(report.stability["s1"], 4, 5, 0.8, 0.3755282641, 0.963776839)
    _check(report.stability["s2"], 2, 2, 1.0, 0.3423719529, 1.0)
    _check(report.stability["s3"], 2, 2, 1.0, 0.3423719529, 1.0)
    assert "s4" not in report.stability
    assert report.stability_excluded == 1
    assert report.stability_mean == pytest.approx((0.8 + 1.0 + 1.0) / 3)
    _check(report.schema_validity, 15, 16, 0.9375, 0.7167073163, 0.9888809427)
    assert report.hard_constraint_violation is not None
    _check(report.hard_constraint_violation, 2, 16, 0.125, 0.0349767493, 0.3602333144)

    state_hit = report.state["C1a"]
    _check(state_hit.localization, 2, 2, 1.0, 0.3423719529, 1.0)
    _check(state_hit.detection, 2, 2, 1.0, 0.3423719529, 1.0)
    _check(state_hit.flip, 1, 2, 0.5, 0.0945286548, 0.9054713452)
    assert state_hit.unpaired == 0

    state_control = report.state["C0"]
    _check(state_control.localization, 0, 2, 0.0, 0.0, 0.6576280471)
    _check(state_control.detection, 1, 2, 0.5, 0.0945286548, 0.9054713452)
    _check(state_control.flip, 1, 2, 0.5, 0.0945286548, 0.9054713452)

    output = report.output["C1a"]
    _check(output.localization, 1, 2, 0.5, 0.0945286548, 0.9054713452)
    _check(output.detection, 1, 2, 0.5, 0.0945286548, 0.9054713452)
    _check(output.flip, 0, 2, 0.0, 0.0, 0.6576280471)
    assert output.unpaired == 0

    assert report.c0_blame == {"none": 1, "wellness": 1}
    assert report.truncated == 1
    assert "B" not in report.state
    assert set(report.state) == {"C1a", "C0"}
    assert set(report.output) == {"C1a"}


def test_a_log_without_the_violation_column_leaves_that_rate_unscored(tmp_path: Path) -> None:
    rows = [
        _row("s1", "B", 1, "a1"),
        _row(
            "s1",
            "C1a",
            1,
            "a2",
            fault=_fault("state", "fiduciary", "fiduciary.discretionary_remaining"),
            blame_agent="fiduciary",
            blame_field="fiduciary.discretionary_remaining",
        ),
    ]
    db = tmp_path / "plain.sqlite"
    _write(db, rows, with_violations=False)
    report = evaluate(db)
    assert report.hard_constraint_violation is None
    _check(report.state["C1a"].localization, 1, 1, 1.0, 0.2065432915, 1.0)


def test_an_empty_log_has_no_rates(tmp_path: Path) -> None:
    db = tmp_path / "empty.sqlite"
    _write(db, [])
    report = evaluate(db)
    assert report.stability == {}
    assert report.stability_mean is None
    assert report.schema_validity.n == 0
    assert report.schema_validity.value is None
    assert report.state == {}
    assert report.output == {}
    assert report.c0_blame == {}
    assert report.truncated == 0
