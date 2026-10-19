"""The invariants, and the fixture discipline that makes them mean anything.

Do not relax anything in this file to make a test pass. A clean fixture that
fails `check_state` is broken data; the fix is the fixture, never the rule.
Clean fixtures are the false-positive baseline for the whole evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ato.provider import DirtyFixtureError, FixtureProvider
from ato.schema import Wellness, WorldState, check_state


def test_every_fixture_reconciles_clean(all_clean: list[WorldState]) -> None:
    assert all_clean, "no fixtures found"
    for state in all_clean:
        assert check_state(state) == [], f"{state.scenario_id} does not reconcile clean"


def test_strict_provider_refuses_a_dirty_fixture(fixtures_dir: Path, tmp_path: Path) -> None:
    raw = json.loads((fixtures_dir / "s014.json").read_text())
    raw["fiduciary"]["discretionary_remaining"] = 260.00
    (tmp_path / "bad.json").write_text(json.dumps(raw))
    with pytest.raises(DirtyFixtureError):
        FixtureProvider(tmp_path).load("bad")


# --- F1: what counts as discretionary spend -------------------------------


def test_transfers_are_not_spend(clean: WorldState) -> None:
    assert clean.fiduciary.discretionary_spend() == 260.00
    assert any(t.kind == "transfer" for t in clean.fiduciary.transactions)


def test_a_non_discretionary_purchase_is_not_spend(fixtures_dir: Path) -> None:
    """Rent is a purchase and is not discretionary. Without the flag, F1 is wrong
    on every real month."""
    s = FixtureProvider(fixtures_dir).load("s021")
    rent = [t for t in s.fiduciary.transactions if not t.discretionary and t.kind == "purchase"]
    assert rent, "s021 should carry a non-discretionary purchase"
    assert s.fiduciary.discretionary_spend() == 412.75
    assert all(t.amount not in {412.75} for t in rent)


def test_issuer_category_is_a_noisy_observation(all_clean: list[WorldState]) -> None:
    """Gas stations and liquor stores arrive labeled as groceries. The reward
    matrix must never key on the issuer field."""
    mislabeled = [
        t
        for s in all_clean
        for t in s.fiduciary.transactions
        if t.statement_category
        and "Groceries" in t.statement_category
        and t.reward_category != "us_supermarket"
    ]
    assert len(mislabeled) >= 2, "fixtures should carry real-world category noise"


# --- W1: absence is not a low reading -------------------------------------


def test_absence_is_expressible_separately_from_a_low_value(fixtures_dir: Path) -> None:
    s = FixtureProvider(fixtures_dir).load("s021")
    assert s.wellness.hrv_observed is False
    assert s.wellness.hrv_today is None
    assert None in s.wellness.hrv_7d
    assert check_state(s) == []


def test_an_empty_window_has_no_baseline(fixtures_dir: Path) -> None:
    s = FixtureProvider(fixtures_dir).load("s028")
    assert s.wellness.hrv_7d == [None] * 7
    assert s.wellness.hrv_baseline_7d is None
    assert s.wellness.observed_baseline() is None
    assert check_state(s) == []


@pytest.mark.parametrize(
    "window,baseline",
    [([None] * 7, 50.0), ([50.0] * 7, None)],
    ids=["number-but-empty", "null-but-observed"],
)
def test_baseline_presence_matches_the_window(
    window: list[float | None], baseline: float | None
) -> None:
    with pytest.raises(ValidationError):
        Wellness(
            hrv_7d=window,
            hrv_baseline_7d=baseline,
            hrv_observed=False,
            hrv_today=None,
            sleep_hours_last=7.0,
            strain_7d=10.0,
            readiness_score=70,
            bmr_kcal=2100,
        )


def test_baseline_ignores_non_wear_days(fixtures_dir: Path) -> None:
    s = FixtureProvider(fixtures_dir).load("s021")
    observed = [v for v in s.wellness.hrv_7d if v is not None]
    assert len(observed) < len(s.wellness.hrv_7d)
    assert s.wellness.observed_baseline() == round(sum(observed) / len(observed), 3)


@pytest.mark.parametrize(
    "observed,today", [(True, None), (False, 44.0)], ids=["claims-but-absent", "absent-but-claims"]
)
def test_absence_shape_rule_is_enforced(observed: bool, today: float | None) -> None:
    with pytest.raises(ValidationError):
        Wellness(
            hrv_7d=[50.0] * 7,
            hrv_baseline_7d=50.0,
            hrv_observed=observed,
            hrv_today=today,
            sleep_hours_last=7.0,
            strain_7d=10.0,
            readiness_score=70,
            bmr_kcal=2100,
        )


# --- P1: the calendar the free block came from ----------------------------


def test_longest_free_block_can_sit_mid_day(fixtures_dir: Path) -> None:
    """Three meetings, and the longest gap is between two of them rather than at
    either end. A naive first-gap or last-gap implementation fails here."""
    s = FixtureProvider(fixtures_dir).load("s021")
    assert s.productivity.computed_longest_free_block() == 210
    assert len(s.productivity.meetings) == 3
