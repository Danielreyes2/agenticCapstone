"""Read-only view of a fixture, an optional injected fault, and the gate log.

The fault record is shown here because this is the scorer's console. Agents
never import this module, and nothing it returns is written back into a run.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from ato.faults import LIBRARY, applies_to, inject_state_fault
from ato.schema import WorldState, check_state

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
RUNS = ROOT / "runs"
_SCENARIO = re.compile(r"s\d{3}")

_LABELS = {
    "d_fid_balance": "Add 40 to the money left in the cycle",
    "d_fid_txn": "Drop 35 from one purchase",
    "d_wel_baseline": "Drop the heart-rate baseline by 9",
    "d_pro_block": "Shorten the longest free block by 150 minutes",
    "u_fid_rent": "Add 180 to rent, which no invariant checks",
    "u_wel_strain": "Lower the strain score, which no invariant checks",
    "u_pro_deadline": "Push a deadline out by 4 days, which no invariant checks",
}


def scenario_index() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(FIXTURES.glob("s*.json")):
        state = WorldState.model_validate_json(path.read_text())
        rows.append(
            {
                "id": state.scenario_id,
                "date": state.date.isoformat(),
                "actions": [action.desc for action in state.candidate_actions],
            }
        )
    return rows


def fault_index() -> list[dict[str, str]]:
    return [
        {
            "id": spec.fault_id,
            "tier": spec.tier,
            "agent": spec.target_agent,
            "label": _LABELS.get(spec.fault_id, spec.fault_id),
        }
        for spec in LIBRARY
        if spec.surface == "state"
    ]


def scenario_view(scenario_id: str, fault_id: str | None = None) -> dict[str, Any]:
    state = _load(scenario_id)
    fault: dict[str, Any] | None = None
    applicable = True
    reason = ""
    if fault_id:
        spec = next((item for item in LIBRARY if item.fault_id == fault_id), None)
        if spec is None or spec.surface != "state":
            raise KeyError(fault_id)
        if not applies_to(spec, state):
            applicable = False
            reason = (
                "This scenario has no observed heart-rate day, "
                "so the baseline fault does not apply."
            )
        else:
            state, record = inject_state_fault(state, spec)
            fault = {
                "id": spec.fault_id,
                "tier": spec.tier,
                "agent": spec.target_agent,
                "path": record.truth_field,
                "before": record.before,
                "after": record.after,
                "label": _LABELS.get(spec.fault_id, spec.fault_id),
            }
    violations = [
        {
            "rule": item.rule,
            "field": item.field,
            "expected": item.expected,
            "observed": item.observed,
        }
        for item in check_state(state)
    ]
    return {
        "scenario_id": state.scenario_id,
        "date": state.date.isoformat(),
        "applicable": applicable,
        "reason": reason,
        "fault": fault,
        "violations": violations,
        "actions": [action.model_dump() for action in state.candidate_actions],
        "fiduciary": _fiduciary(state),
        "wellness": _wellness(state),
        "productivity": _productivity(state),
        "runs": _runs(scenario_id),
    }


def _load(scenario_id: str) -> WorldState:
    if not _SCENARIO.fullmatch(scenario_id):
        raise KeyError(scenario_id)
    path = FIXTURES / f"{scenario_id}.json"
    if not path.is_file():
        raise KeyError(scenario_id)
    return WorldState.model_validate_json(path.read_text())


def _fiduciary(state: WorldState) -> dict[str, Any]:
    book = state.fiduciary
    return {
        "cycle_budget": book.cycle_budget,
        "discretionary_remaining": book.discretionary_remaining,
        "days_left_in_cycle": book.days_left_in_cycle,
        "spend": book.discretionary_spend(),
        "transactions": [
            {
                "index": index,
                "descriptor": txn.descriptor,
                "amount": txn.amount,
                "kind": txn.kind,
                "discretionary": txn.discretionary,
            }
            for index, txn in enumerate(book.transactions)
        ],
        "liabilities": [
            {"name": item.name, "amount": item.amount, "due_in_days": item.due_in_days}
            for item in book.fixed_liabilities_upcoming
        ],
    }


def _wellness(state: WorldState) -> dict[str, Any]:
    body = state.wellness
    return {
        "readiness_score": body.readiness_score,
        "sleep_hours_last": body.sleep_hours_last,
        "strain_7d": body.strain_7d,
        "hrv_today": body.hrv_today,
        "hrv_baseline_7d": body.hrv_baseline_7d,
        "hrv_7d": body.hrv_7d,
    }


def _productivity(state: WorldState) -> dict[str, Any]:
    day = state.productivity
    start = _minutes(day.day_start)
    span = max(_minutes(day.day_end) - start, 1)
    return {
        "day_start": day.day_start,
        "day_end": day.day_end,
        "longest_free_block_min": day.longest_free_block_min,
        "deadlines": [
            {"name": item.name, "due_in_days": item.due_in_days} for item in day.deadlines
        ],
        "meetings": [
            {
                "title": meeting.title,
                "priority": meeting.priority,
                "left": round(100 * (_minutes(meeting.start) - start) / span, 2),
                "width": round(100 * meeting.duration_min / span, 2),
            }
            for meeting in day.meetings
        ],
    }


def _minutes(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


def _runs(scenario_id: str) -> list[dict[str, Any]]:
    if not RUNS.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for path in sorted(RUNS.glob("*.sqlite")):
        found.extend(_read_log(path, scenario_id))
    return found


def _read_log(path: Path, scenario_id: str) -> list[dict[str, Any]]:
    query = """
        SELECT condition, seed, model, fault_record, chosen_action,
               blame_agent, blame_field, blame_confidence
        FROM runs WHERE scenario_id = ? ORDER BY condition, seed
    """
    try:
        with sqlite3.connect(path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
            # The first gate logged blame before the call could see the state.
            # Those rows are a harness failure, not a result, so leave them out.
            if "temperature_sent" not in columns:
                return []
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (scenario_id,)).fetchall()
    except sqlite3.Error:
        return []
    out = []
    for row in rows:
        truth_agent = None
        truth_field = None
        fault_name = None
        if row["fault_record"]:
            record = json.loads(row["fault_record"])
            truth_agent = record.get("truth_agent")
            truth_field = record.get("truth_field")
            fault_name = record.get("spec", {}).get("fault_id")
        named = row["blame_agent"] is not None
        matched = row["blame_agent"] == truth_agent and row["blame_field"] == truth_field
        # On a clean run or the undetectable control, abstaining is the right call.
        # Naming the injected field is only a hit when the fault was detectable.
        if row["condition"] in {"B", "C0"}:
            label = "clear" if not named else "false alarm"
        else:
            label = "hit" if matched else "miss"
        out.append(
            {
                "log": path.stem,
                "model": row["model"],
                "condition": row["condition"],
                "seed": row["seed"],
                "fault_id": fault_name,
                "chosen_action": row["chosen_action"],
                "blame_agent": row["blame_agent"],
                "blame_field": row["blame_field"],
                "blame_confidence": row["blame_confidence"],
                "truth_field": truth_field,
                "label": label,
            }
        )
    return out
