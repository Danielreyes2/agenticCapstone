"""Scenario generator.

Derived fields are assigned from the data they are derived from. A state that
fails `check_state` is never drawn and thrown away: F1, W1 and P1 hold because
the values written are the ones those rules recompute.

These scenarios stay out of `fixtures/`. That directory is the hand-written
false-positive baseline, and mixing generated files into it would make the
fault sweep grade the generator.
"""

from __future__ import annotations

import datetime
import random
from typing import Literal

from ato.schema import (
    Account,
    CandidateAction,
    Card,
    Deadline,
    DeepWorkBlock,
    Fiduciary,
    FixedLiability,
    Meeting,
    Productivity,
    StatementCredit,
    Transaction,
    Wellness,
    WorldState,
)

# The sweep size settled in OPEN-2. The generator itself takes a count.
SWEEP_COUNT = 100

# Real exports carry 4 non-wear days in 139. The fixtures are heavier than that
# because each one pins an edge; matching them would over-sample gaps.
NONWEAR_RATE = 4 / 139

# Issuer labels disagree with the reward category often enough that the reward
# matrix cannot key on the issuer field. The fixtures do this on about 5 of 33
# purchases.
_LABEL_NOISE = 0.15
_OVERLAP_RATE = 0.15
_ZERO_HEADROOM = 0.08
_EXHAUSTED_CAP = 0.2

_Priority = Literal["low", "medium", "high"]
_PRIORITIES: tuple[_Priority, ...] = ("low", "medium", "high")

_AGREED: tuple[tuple[str, str], ...] = (
    ("Merchandise & Supplies-Groceries", "us_supermarket"),
    ("Restaurant-Restaurant", "dining"),
    ("Transportation-Fuel", "gas"),
    ("Merchandise & Supplies-Books", "other"),
)
_MISLABELED: tuple[tuple[str, str], ...] = (
    ("Merchandise & Supplies-Groceries", "gas"),
    ("Merchandise & Supplies-Groceries", "other"),
    ("Restaurant-Restaurant", "us_supermarket"),
)
_WINDOWS: tuple[tuple[str, str], ...] = (
    ("08:00", "18:00"),
    ("07:30", "19:00"),
    ("09:00", "17:00"),
)
_MERCHANTS: dict[str, str] = {
    "us_supermarket": "AplPay SAFEWAY  SAN FRANCISCO  CA",
    "dining": "L&J CAFE  EL PASO  TX",
    "gas": "SPEEDWAY 47327  HORIZON CITY  TX",
    "other": "WALGREENS 06112  EL PASO  TX",
}
_ACTIONS: tuple[tuple[str, str, tuple[int, int], tuple[int, int]], ...] = (
    ("order delivery", "dining", (12, 40), (10, 25)),
    ("cook", "us_supermarket", (6, 18), (40, 90)),
    ("groceries", "us_supermarket", (25, 90), (30, 70)),
    ("library afternoon", "other", (0, 0), (60, 150)),
    ("coffee with a friend", "dining", (5, 15), (30, 60)),
)


def generate(count: int, *, seed: int) -> list[WorldState]:
    """`count` clean scenarios from `seed`. The same pair replays byte for byte."""
    rng = random.Random(seed)
    start = datetime.date(2026, 1, 5)
    return [_scenario(rng, index, start + datetime.timedelta(days=index)) for index in range(count)]


def _scenario(rng: random.Random, index: int, day: datetime.date) -> WorldState:
    return WorldState(
        scenario_id=f"g{index + 1:03d}",
        date=day,
        fiduciary=_fiduciary(rng, day),
        wellness=_wellness(rng),
        productivity=_productivity(rng),
        candidate_actions=_actions(rng),
    )


def _fiduciary(rng: random.Random, day: datetime.date) -> Fiduciary:
    transactions = _transactions(rng, day)
    supermarket = round(
        sum(
            txn.amount
            for txn in transactions
            if txn.kind == "purchase"
            and txn.account == "card_a"
            and txn.reward_category == "us_supermarket"
        ),
        2,
    )
    # A cap already below the spend is the exhausted-cap case. It is chosen
    # here, not discovered by retrying a scenario whose lookup fell through.
    if rng.random() < _EXHAUSTED_CAP and supermarket >= 1:
        cap = round(supermarket - 1, 2)
    else:
        cap = 25000.0
    draft = Fiduciary(
        cycle_budget=0.0,
        discretionary_remaining=0.0,
        days_left_in_cycle=rng.randint(5, 20),
        accounts=[
            Account(id="checking", kind="deposit", balance=round(rng.uniform(400, 4000), 2)),
            Account(id="card_a", kind="credit", statement_balance=round(rng.uniform(40, 800), 2)),
            Account(id="card_b", kind="credit", statement_balance=round(rng.uniform(20, 500), 2)),
        ],
        transactions=transactions,
        cards=[
            Card(
                id="card_a",
                rates={"us_supermarket": 0.04, "dining": 0.03, "gas": 0.01, "other": 0.01},
                caps_annual={"us_supermarket": cap},
            ),
            Card(
                id="card_b",
                rates={"dining": 0.03, "us_supermarket": 0.02, "travel": 0.03, "other": 0.01},
            ),
        ],
        credits=_credits(rng, day),
        fixed_liabilities_upcoming=[_liability(rng)],
    )
    spend = draft.discretionary_spend()
    wanted = 0.0 if rng.random() < _ZERO_HEADROOM else round(rng.uniform(15, 400), 2)
    budget = round(spend + wanted, 2)
    # The same expression F1 recomputes, so the remainder cannot drift from the spend.
    remaining = round(budget - spend, 2)
    return draft.model_copy(update={"cycle_budget": budget, "discretionary_remaining": remaining})


def _transactions(rng: random.Random, day: datetime.date) -> list[Transaction]:
    rows = [
        _purchase(rng, day, "card_a", "us_supermarket"),
        _purchase(rng, day, "card_b", "dining"),
        _purchase(rng, day, rng.choice(("card_a", "card_b")), None),
    ]
    if rng.random() < 0.3:
        rows.append(_purchase(rng, day, "checking", "other", discretionary=False))
    if rng.random() < 0.5:
        rows.append(
            Transaction(
                date=day - datetime.timedelta(days=rng.randint(1, 12)),
                account="checking",
                kind="transfer",
                descriptor="CARD A ACH PMT  WEB ID: 2005032111",
                amount=round(-rng.uniform(40, 400), 2),
                discretionary=False,
                counterparty_account="card_a",
            )
        )
    return rows


def _purchase(
    rng: random.Random,
    day: datetime.date,
    account: str,
    reward: str | None,
    *,
    discretionary: bool = True,
) -> Transaction:
    if reward is None:
        statement, reward = rng.choice(_AGREED if rng.random() >= _LABEL_NOISE else _MISLABELED)
    else:
        statement = _statement_for(reward)
        if rng.random() < _LABEL_NOISE:
            statement, reward = rng.choice(_MISLABELED)
    return Transaction(
        date=day - datetime.timedelta(days=rng.randint(1, 20)),
        account=account,
        kind="purchase",
        descriptor=_MERCHANTS.get(reward, _MERCHANTS["other"]),
        amount=round(rng.uniform(8, 120), 2),
        discretionary=discretionary,
        statement_category=statement,
        reward_category=reward,
    )


def _statement_for(reward: str) -> str:
    for statement, category in _AGREED:
        if category == reward:
            return statement
    return "Merchandise & Supplies-Books"


def _credits(rng: random.Random, day: datetime.date) -> list[StatementCredit]:
    roll = rng.random()
    if roll < 0.4:
        return []
    return [
        StatementCredit(
            name="dining credit",
            amount=10.0,
            period="monthly",
            expires=day + datetime.timedelta(days=10 if roll < 0.8 else -10),
            used=roll >= 0.8,
        )
    ]


def _liability(rng: random.Random) -> FixedLiability:
    if rng.random() < 0.7:
        name, amount = "rent", round(rng.uniform(700, 1400), 2)
    else:
        name, amount = "renters insurance", round(rng.uniform(15, 40), 2)
    return FixedLiability(name=name, amount=amount, due_in_days=rng.randint(0, 28))


def _wellness(rng: random.Random) -> Wellness:
    window: list[float | None] = [
        None if rng.random() < NONWEAR_RATE else float(rng.randint(25, 75)) for _ in range(7)
    ]
    worn_today = rng.random() >= NONWEAR_RATE
    today = float(rng.randint(25, 75)) if worn_today else None
    seen = [value for value in window if value is not None]
    # W1's mean, including the empty-window case, where the baseline is null.
    baseline = round(sum(seen) / len(seen), 3) if seen else None
    return Wellness(
        hrv_7d=window,
        hrv_baseline_7d=baseline,
        hrv_observed=today is not None,
        hrv_today=today,
        sleep_hours_last=round(rng.uniform(5.0, 8.5), 1),
        strain_7d=round(rng.uniform(6.0, 18.0), 1),
        readiness_score=rng.randint(35, 85),
        bmr_kcal=rng.randint(1800, 2300),
    )


def _productivity(rng: random.Random) -> Productivity:
    day_start, day_end = rng.choice(_WINDOWS)
    draft = Productivity(
        day_start=day_start,
        day_end=day_end,
        meetings=_meetings(rng, day_start, day_end),
        deep_work_blocks=_deep_work(rng),
        deadlines=[
            Deadline(
                name=rng.choice(("proposal draft", "reading notes", "lab report")),
                due_in_days=rng.randint(0, 14),
                weight=rng.choice((0.4, 0.5, 1.0)),
            )
        ],
        longest_free_block_min=0,
    )
    # P1's own calendar walk, so the stated block is the one the meetings produce.
    return draft.model_copy(update={"longest_free_block_min": draft.computed_longest_free_block()})


def _meetings(rng: random.Random, day_start: str, day_end: str) -> list[Meeting]:
    lo, hi = _mins(day_start), _mins(day_end)
    meetings: list[Meeting] = []
    for _ in range(rng.randint(0, 3)):
        start = lo + rng.randint(0, (hi - lo) // 15 - 1) * 15
        duration = min(rng.choice((15, 30, 45, 60, 90)), hi - start)
        meetings.append(_meeting(rng, start, duration))
    if rng.random() < _OVERLAP_RATE:
        start = lo + 60
        meetings.append(_meeting(rng, start, 60, "high"))
        meetings.append(_meeting(rng, start + 30, 60, "low"))
    return meetings


def _meeting(
    rng: random.Random, start: int, duration: int, priority: _Priority | None = None
) -> Meeting:
    return Meeting(
        title=rng.choice(("advisor sync", "standup", "design review", "studio critique")),
        start=_hhmm(start),
        duration_min=duration,
        priority=priority if priority is not None else rng.choice(_PRIORITIES),
    )


def _deep_work(rng: random.Random) -> list[DeepWorkBlock]:
    if rng.random() < 0.3:
        return []
    return [DeepWorkBlock(start="09:00", duration_min=rng.choice((90, 120, 180)))]


def _actions(rng: random.Random) -> list[CandidateAction]:
    picked = rng.sample(list(_ACTIONS), 2)
    actions = []
    for index, (desc, category, cost, minutes) in enumerate(picked, start=1):
        low, high = cost
        actions.append(
            CandidateAction(
                id=f"a{index}",
                desc=desc,
                cost=float(low if low == high else rng.randint(low, high)),
                time_min=rng.randint(minutes[0], minutes[1]),
                reward_category=category,
            )
        )
    return actions


def _mins(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


def _hhmm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"
