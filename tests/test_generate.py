"""Generated scenarios reconcile by construction and keep the mess the fixtures carry.

The hand-written fixtures are the replay stand-in: there is no separate export
file in the repo. They over-sample edges, because each one pins a case. The
non-wear rate is the one number the schema records from real exports, 4 days
in 139, and the generator is checked against that rather than against the
fixtures' heavier gap rate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ato.generate import NONWEAR_RATE, SWEEP_COUNT, generate
from ato.provider import FixtureProvider
from ato.schema import WorldState, check_state

ROOT = Path(__file__).resolve().parents[1]


def _profile(states: list[WorldState]) -> dict[str, float]:
    nulls = 0
    days = 0
    unobserved = 0
    low_ready = 0
    purchases = 0
    disagree = 0
    overlaps = 0
    zero_head = 0
    exhausted = 0
    readiness: list[int] = []
    budgets: list[float] = []
    for state in states:
        wellness = state.wellness
        days += len(wellness.hrv_7d)
        nulls += sum(value is None for value in wellness.hrv_7d)
        unobserved += not wellness.hrv_observed
        low_ready += wellness.readiness_score < 50
        readiness.append(wellness.readiness_score)
        fiduciary = state.fiduciary
        budgets.append(fiduciary.cycle_budget)
        zero_head += fiduciary.discretionary_remaining == 0
        for txn in fiduciary.transactions:
            if txn.kind != "purchase" or not txn.statement_category or not txn.reward_category:
                continue
            purchases += 1
            groceries = "Groceries" in txn.statement_category
            restaurant = "Restaurant" in txn.statement_category
            mislabeled = (groceries and txn.reward_category != "us_supermarket") or (
                restaurant and txn.reward_category != "dining"
            )
            if mislabeled:
                disagree += 1
        if _overlaps(state):
            overlaps += 1
        if _cap_exhausted(state):
            exhausted += 1
    count = len(states)
    return {
        "nonwear_rate": nulls / days,
        "unobserved_today_rate": unobserved / count,
        "disagree_rate": disagree / purchases,
        "overlap_rate": overlaps / count,
        "low_readiness_rate": low_ready / count,
        "zero_headroom_rate": zero_head / count,
        "exhausted_cap_rate": exhausted / count,
        "readiness_min": float(min(readiness)),
        "readiness_max": float(max(readiness)),
        "budget_span": max(budgets) - min(budgets),
    }


def _mins(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


def _overlaps(state: WorldState) -> bool:
    spans = []
    for meeting in state.productivity.meetings:
        start = _mins(meeting.start)
        spans.append((start, start + meeting.duration_min))
    spans.sort()
    return any(spans[i][0] < spans[i - 1][1] for i in range(1, len(spans)))


def _cap_exhausted(state: WorldState) -> bool:
    fiduciary = state.fiduciary
    for card in fiduciary.cards:
        for category, cap in card.caps_annual.items():
            spend = sum(
                txn.amount
                for txn in fiduciary.transactions
                if txn.kind == "purchase"
                and txn.account == card.id
                and txn.reward_category == category
            )
            if spend >= cap:
                return True
    return False


def test_sweep_count_is_one_hundred() -> None:
    assert SWEEP_COUNT == 100


def test_two_hundred_scenarios_reconcile_by_construction() -> None:
    states = generate(200, seed=0)
    assert len(states) == 200
    ids = [state.scenario_id for state in states]
    assert len(set(ids)) == 200
    fixtures = set(FixtureProvider(ROOT / "fixtures").scenario_ids())
    assert not set(ids) & fixtures
    for state in states:
        assert len(state.wellness.hrv_7d) == 7
        assert check_state(state) == []


def test_generation_does_not_reject_against_check_state(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(state: WorldState) -> list[object]:
        raise AssertionError(f"rejected {state.scenario_id} instead of building it")

    monkeypatch.setattr("ato.schema.check_state", boom)
    import ato.generate as generator

    monkeypatch.setattr(generator, "check_state", boom, raising=False)
    states = generate(30, seed=1)
    assert len(states) == 30


def test_the_same_seed_replays() -> None:
    first = [state.model_dump(mode="json") for state in generate(20, seed=3)]
    second = [state.model_dump(mode="json") for state in generate(20, seed=3)]
    assert first == second
    assert first != [state.model_dump(mode="json") for state in generate(20, seed=4)]


def test_generated_distribution_is_not_tidier_than_the_fixtures() -> None:
    fixtures = [
        FixtureProvider(ROOT / "fixtures").load(scenario_id)
        for scenario_id in FixtureProvider(ROOT / "fixtures").scenario_ids()
    ]
    fix = _profile(fixtures)
    gen = _profile(generate(200, seed=0))
    assert abs(gen["nonwear_rate"] - NONWEAR_RATE) < 0.02
    assert gen["unobserved_today_rate"] > 0
    # Each of these is present in the fixtures. A tidy generator drops them.
    for key in ("disagree_rate", "overlap_rate", "low_readiness_rate", "zero_headroom_rate"):
        assert fix[key] > 0
        assert gen[key] > 0
    assert gen["disagree_rate"] >= 0.05
    assert gen["overlap_rate"] >= 0.05
    assert gen["low_readiness_rate"] >= 0.1
    assert gen["zero_headroom_rate"] >= 0.02
    assert gen["exhausted_cap_rate"] >= 0.05
    assert gen["readiness_min"] < 50 < gen["readiness_max"]
    assert gen["budget_span"] > 200
