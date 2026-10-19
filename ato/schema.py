"""WorldState and AgentVerdict schemas for ATO.

Design note that matters more than it looks:

    Pydantic enforces SHAPE. A separate function enforces CONSISTENCY.

They are deliberately not the same pass. If a model_validator rejected a state
whose derived scalars disagree with their source data, a detectable fault would
be unconstructible -- the injector could not produce the very thing the
benchmark measures. So `WorldState` will happily hold an inconsistent state, and
`check_state()` reports where it is inconsistent.

The second thing worth noting: `Violation.agent` and `Violation.field` are
exactly the label space of the localization metric. The invariant checker and
the scorer agree by construction rather than by convention.
"""

from __future__ import annotations

import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

AgentName = Literal["fiduciary", "wellness", "productivity"]

# Tolerance for float comparison in the consistency checks. One cent, one
# millisecond of HRV, one minute of calendar. Faults are injected well above it.
EPS = 0.005


# --------------------------------------------------------------------------
# Fiduciary
# --------------------------------------------------------------------------


class Account(BaseModel):
    id: str
    kind: Literal["deposit", "credit"]
    balance: float | None = None
    statement_balance: float | None = None


class Transaction(BaseModel):
    """One line from an issuer data export.

    `account` and `kind` exist because real exports double-count without them:
    a card payment appears once as a checking debit and again as the underlying
    purchases. Only `kind == "purchase"` rows are spend.

    `statement_category` is the issuer's label and is a NOISY OBSERVATION of
    `reward_category`, not a synonym. Real exports disagree often enough that
    the reward matrix cannot key on the issuer field. Both are carried so that
    card-recommendation accuracy can be scored against the hand-labeled one.
    """

    date: datetime.date
    account: str
    kind: Literal["purchase", "transfer", "payment"]
    descriptor: str
    amount: float
    discretionary: bool = True
    statement_category: str | None = None
    reward_category: str | None = None
    counterparty_account: str | None = None


class Card(BaseModel):
    id: str
    rates: dict[str, float] = Field(default_factory=dict)
    caps_annual: dict[str, float] = Field(default_factory=dict)


class StatementCredit(BaseModel):
    name: str
    amount: float
    period: Literal["monthly", "quarterly", "annual"]
    expires: datetime.date
    used: bool = False


class FixedLiability(BaseModel):
    name: str
    amount: float
    due_in_days: int


class Fiduciary(BaseModel):
    cycle_budget: float
    discretionary_remaining: float  # DERIVED -- checked by F1
    days_left_in_cycle: int
    accounts: list[Account] = Field(default_factory=list)
    transactions: list[Transaction] = Field(default_factory=list)
    cards: list[Card] = Field(default_factory=list)
    credits: list[StatementCredit] = Field(default_factory=list)
    fixed_liabilities_upcoming: list[FixedLiability] = Field(default_factory=list)

    def discretionary_spend(self) -> float:
        return round(
            sum(t.amount for t in self.transactions if t.kind == "purchase" and t.discretionary),
            2,
        )


# --------------------------------------------------------------------------
# Wellness
# --------------------------------------------------------------------------


class Wellness(BaseModel):
    """`hrv_7d` is the trailing window BEFORE today; `hrv_today` is today.

    A `None` entry is a legitimate non-wear day, not a zero and not a fault.
    Real exports carry 4 such days in 139. Without an explicit absence
    representation, natural gaps contaminate the false-positive baseline.
    """

    hrv_7d: list[float | None]
    hrv_baseline_7d: float | None  # DERIVED -- checked by W1 when any day was observed
    hrv_observed: bool
    hrv_today: float | None = None
    sleep_hours_last: float
    strain_7d: float
    readiness_score: int
    bmr_kcal: int

    @model_validator(mode="after")
    def _absence_is_coherent(self) -> Wellness:
        # Shape rule, not a consistency rule: observed and present must agree,
        # or "absent" and "low" become indistinguishable downstream.
        if self.hrv_observed and self.hrv_today is None:
            raise ValueError("hrv_observed is true but hrv_today is null")
        if not self.hrv_observed and self.hrv_today is not None:
            raise ValueError("hrv_observed is false but hrv_today carries a value")
        # A placeholder number would read as a real baseline. The mean exists
        # only when some day was observed, so null and "no observations" travel together.
        window_empty = self.observed_baseline() is None
        if window_empty and self.hrv_baseline_7d is not None:
            raise ValueError("hrv_baseline_7d is set but hrv_7d has no observed day")
        if not window_empty and self.hrv_baseline_7d is None:
            raise ValueError("hrv_baseline_7d is null but hrv_7d has an observed day")
        return self

    def observed_baseline(self) -> float | None:
        seen = [v for v in self.hrv_7d if v is not None]
        return round(sum(seen) / len(seen), 3) if seen else None


# --------------------------------------------------------------------------
# Productivity
# --------------------------------------------------------------------------


def _mins(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


class Meeting(BaseModel):
    title: str
    start: str
    duration_min: int
    priority: Literal["low", "medium", "high"] = "medium"


class Deadline(BaseModel):
    name: str
    due_in_days: int
    weight: float = 1.0


class DeepWorkBlock(BaseModel):
    start: str
    duration_min: int


class Productivity(BaseModel):
    day_start: str = "08:00"
    day_end: str = "18:00"
    meetings: list[Meeting] = Field(default_factory=list)
    deep_work_blocks: list[DeepWorkBlock] = Field(default_factory=list)
    deadlines: list[Deadline] = Field(default_factory=list)
    longest_free_block_min: int  # DERIVED -- checked by P1

    def computed_longest_free_block(self) -> int:
        lo, hi = _mins(self.day_start), _mins(self.day_end)
        busy = []
        for m in self.meetings:
            s = max(lo, _mins(m.start))
            e = min(hi, _mins(m.start) + m.duration_min)
            if e > s:
                busy.append((s, e))
        busy.sort()
        merged: list[list[int]] = []
        for s, e in busy:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        best, cursor = 0, lo
        for s, e in merged:
            best = max(best, s - cursor)
            cursor = max(cursor, e)
        return max(best, hi - cursor, 0)


# --------------------------------------------------------------------------
# WorldState
# --------------------------------------------------------------------------


class CandidateAction(BaseModel):
    id: str
    desc: str
    cost: float
    time_min: int
    reward_category: str | None = None


class WorldState(BaseModel):
    scenario_id: str
    date: datetime.date
    fiduciary: Fiduciary
    wellness: Wellness
    productivity: Productivity
    candidate_actions: list[CandidateAction]

    model_config = {"extra": "forbid"}


# --------------------------------------------------------------------------
# Consistency checking -- the detectable/undetectable boundary
# --------------------------------------------------------------------------


class Violation(BaseModel):
    """A broken redundancy.

    `agent` and `field` name the derived scalar whose invariant failed. That is
    NOT necessarily the field that was corrupted: F1 breaks whether the injector
    shifted `discretionary_remaining` or shifted a transaction amount. Choosing
    between the fields an invariant touches is the localization task itself.
    """

    agent: AgentName
    rule: str
    field: str
    expected: float
    observed: float

    @property
    def delta(self) -> float:
        return round(self.observed - self.expected, 4)


def check_state(state: WorldState) -> list[Violation]:
    """Return every broken internal redundancy in `state`.

    A clean fixture must return []. A detectable fault must return >= 1. An
    undetectable fault must return []. Those three facts are enforced by tests
    rather than asserted by the author.
    """
    out: list[Violation] = []

    # F1: budget minus discretionary purchases must equal the stated remainder.
    expected_rem = round(state.fiduciary.cycle_budget - state.fiduciary.discretionary_spend(), 2)
    if abs(expected_rem - state.fiduciary.discretionary_remaining) > EPS:
        out.append(
            Violation(
                agent="fiduciary",
                rule="F1",
                field="fiduciary.discretionary_remaining",
                expected=expected_rem,
                observed=state.fiduciary.discretionary_remaining,
            )
        )

    # W1: the stated baseline must equal the mean of the observed window.
    # An empty window has no mean; the shape rule already requires a null baseline.
    expected_base = state.wellness.observed_baseline()
    stated_base = state.wellness.hrv_baseline_7d
    if (
        expected_base is not None
        and stated_base is not None
        and abs(expected_base - stated_base) > EPS
    ):
        out.append(
            Violation(
                agent="wellness",
                rule="W1",
                field="wellness.hrv_baseline_7d",
                expected=expected_base,
                observed=stated_base,
            )
        )

    # P1: the stated longest free block must match the calendar it came from.
    expected_block = state.productivity.computed_longest_free_block()
    if abs(expected_block - state.productivity.longest_free_block_min) > 1:
        out.append(
            Violation(
                agent="productivity",
                rule="P1",
                field="productivity.longest_free_block_min",
                expected=float(expected_block),
                observed=float(state.productivity.longest_free_block_min),
            )
        )

    return out


# --------------------------------------------------------------------------
# Agent output contract
# --------------------------------------------------------------------------


class InputAnomaly(BaseModel):
    """An agent's structured report that its own inputs do not hang together.

    This field is the resolution to the reconcile short-circuit problem. Without
    it, an agent that calls `reconcile` and finds a discrepancy has nowhere to
    put the finding except `rationale` prose -- which means the arbiter would be
    reading free text, and the numeric-drift and rhetorical-inflation fault
    classes would contaminate each other.

    Populated only in conditions where the agent holds a verification tool.
    """

    field: str
    expected: float
    observed: float


class AgentVerdict(BaseModel):
    agent: AgentName
    action_id: str
    score: float = Field(ge=-1.0, le=1.0)
    hard_violation: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=280)
    evidence: list[str] = Field(default_factory=list)
    input_anomaly: InputAnomaly | None = None

    model_config = {"extra": "forbid"}


# Confidence ceiling when a verdict leans on a field the world did not observe.
UNOBSERVED_CONFIDENCE_CEILING = 0.60


def unobserved_fields(state: WorldState) -> set[str]:
    """WorldState fields that carry no reading today."""
    missing: set[str] = set()
    if not state.wellness.hrv_observed:
        missing.add("wellness.hrv_today")
    return missing


def check_verdicts(verdicts: list[AgentVerdict], state: WorldState) -> list[Violation]:
    """Verdict-surface redundancy: confidence vs. the evidence it cites.

    V1 is what gives confidence miscalibration a DETECTABLE variant. A verdict
    claiming 0.95 confidence while citing a field the world did not observe is
    internally inconsistent in a way the arbiter can check, which means it has a
    localizable ground truth rather than only a flip rate.
    """
    out: list[Violation] = []
    missing = unobserved_fields(state)
    for v in verdicts:
        if missing.intersection(v.evidence) and v.confidence > UNOBSERVED_CONFIDENCE_CEILING + EPS:
            out.append(
                Violation(
                    agent=v.agent,
                    rule="V1",
                    field=f"{v.agent}.confidence",
                    expected=UNOBSERVED_CONFIDENCE_CEILING,
                    observed=v.confidence,
                )
            )
    return out
