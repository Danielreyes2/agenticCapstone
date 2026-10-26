"""Evaluation harness. Every number comes from the run log and nowhere else.

State faults and output faults are separate results. Nothing here averages them:
their answer spaces differ, so a pooled localization accuracy has no chance floor.

Budget-truncated rows are counted and then left out of every rate. A run that
produced no usable agent verdicts is a schema failure, not a decision, so it
is left out of stability and counted as a miss where a fault was injected.

Intervals are Wilson at z = 1.96. A proportion with no rows has no interval.
`chosen_violates` is not a column the runner writes. When the log lacks it, the
hard-constraint rate is absent rather than filled in from the scenario.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

_Z = 1.96


@dataclass(frozen=True)
class Proportion:
    """k successes in n rows, with a 95% Wilson interval."""

    k: int
    n: int
    value: float | None
    low: float | None
    high: float | None


@dataclass(frozen=True)
class Rates:
    localization: Proportion
    detection: Proportion
    flip: Proportion
    unpaired: int


@dataclass(frozen=True)
class Evaluation:
    stability: dict[str, Proportion]
    stability_mean: float | None
    stability_excluded: int
    schema_validity: Proportion
    hard_constraint_violation: Proportion | None
    state: dict[str, Rates]
    output: dict[str, Rates]
    c0_blame: dict[str, int]
    truncated: int


def wilson(k: int, n: int) -> Proportion:
    """95% Wilson interval. The bounds are clamped to [0, 1]."""
    if n == 0:
        return Proportion(0, 0, None, None, None)
    proportion = k / n
    z2 = _Z * _Z
    denom = 1 + z2 / n
    center = (proportion + z2 / (2 * n)) / denom
    margin = _Z * math.sqrt(proportion * (1 - proportion) / n + z2 / (4 * n * n)) / denom
    return Proportion(
        k,
        n,
        proportion,
        max(0.0, center - margin),
        min(1.0, center + margin),
    )


def evaluate(db_path: Path) -> Evaluation:
    rows = _rows(db_path)
    truncated = [row for row in rows if _flag(row, "truncated")]
    scored = [row for row in rows if not _flag(row, "truncated")]
    clean = [row for row in scored if row["fault_record"] is None]
    usable = [row for row in clean if _valid(row)]
    stability = _stability(usable)
    values = [item.value for item in stability.values() if item.value is not None]
    return Evaluation(
        stability=stability,
        stability_mean=sum(values) / len(values) if values else None,
        stability_excluded=len(clean) - len(usable),
        schema_validity=wilson(sum(1 for row in scored if _valid(row)), len(scored)),
        hard_constraint_violation=_violations(scored),
        state=_by_condition(scored, clean, "state"),
        output=_by_condition(scored, clean, "output"),
        c0_blame=_c0_blame(scored),
        truncated=len(truncated),
    )


def _stability(clean: list[sqlite3.Row]) -> dict[str, Proportion]:
    """Per scenario, how often the chosen action matches the mode of that scenario."""
    by_scenario: dict[str, list[str]] = defaultdict(list)
    for row in clean:
        if row["chosen_action"] is None:
            continue
        by_scenario[str(row["scenario_id"])].append(str(row["chosen_action"]))
    out: dict[str, Proportion] = {}
    for scenario, actions in by_scenario.items():
        mode = Counter(actions).most_common(1)[0][1]
        out[scenario] = wilson(mode, len(actions))
    return out


def _by_condition(
    scored: list[sqlite3.Row], clean: list[sqlite3.Row], surface: str
) -> dict[str, Rates]:
    paired = {str(row["pair_key"]): row for row in clean}
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in scored:
        record = _record(row)
        if record is None or _surface(record) != surface:
            continue
        grouped[str(row["condition"])].append(row)
    return {condition: _rates(rows, paired) for condition, rows in grouped.items()}


def _rates(rows: list[sqlite3.Row], paired: dict[str, sqlite3.Row]) -> Rates:
    localized = 0
    detected = 0
    flips = 0
    paired_rows = 0
    for row in rows:
        record = _record(row)
        assert record is not None
        if _hit(row, record):
            localized += 1
        if _detected(row):
            detected += 1
        twin = paired.get(str(row["pair_key"]))
        if twin is None:
            continue
        paired_rows += 1
        if row["chosen_action"] != twin["chosen_action"]:
            flips += 1
    return Rates(
        wilson(localized, len(rows)),
        wilson(detected, len(rows)),
        wilson(flips, paired_rows),
        len(rows) - paired_rows,
    )


def _hit(row: sqlite3.Row, record: dict[str, object]) -> bool:
    if _flag(row, "blame_parse_failed"):
        return False
    same_agent = row["blame_agent"] == record["truth_agent"]
    same_field = row["blame_field"] == record["truth_field"]
    return bool(same_agent and same_field)


def _detected(row: sqlite3.Row) -> bool:
    named = row["blame_agent"] is not None
    return bool(named) and not _flag(row, "blame_parse_failed")


def _c0_blame(scored: list[sqlite3.Row]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in scored:
        if row["condition"] != "C0":
            continue
        if _flag(row, "blame_parse_failed"):
            counts["unparsed"] += 1
        else:
            counts[str(row["blame_agent"] or "none")] += 1
    return dict(counts)


def _violations(scored: list[sqlite3.Row]) -> Proportion | None:
    # `in` on a sqlite row checks values, not column names.
    if not scored or not _has_column(scored[0], "chosen_violates"):
        return None
    known = [row for row in scored if row["chosen_violates"] is not None]
    hits = sum(1 for row in known if _flag(row, "chosen_violates"))
    return wilson(hits, len(known))


def _valid(row: sqlite3.Row) -> bool:
    return not _flag(row, "agent_parse_failed") and int(row["agent_verdict_count"]) > 0


def _has_column(row: sqlite3.Row, name: str) -> bool:
    try:
        row[name]
    except IndexError:
        return False
    return True


def _flag(row: sqlite3.Row, name: str) -> bool:
    return bool(int(row[name] or 0))


def _record(row: sqlite3.Row) -> dict[str, object] | None:
    raw = row["fault_record"]
    if raw is None:
        return None
    parsed = json.loads(str(raw))
    if not isinstance(parsed, dict):
        return None
    return parsed


def _surface(record: dict[str, object]) -> str | None:
    spec = record.get("spec")
    if not isinstance(spec, dict):
        return None
    surface = spec.get("surface")
    return surface if isinstance(surface, str) else None


def _rows(db_path: Path) -> list[sqlite3.Row]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM runs"))
    finally:
        conn.close()
