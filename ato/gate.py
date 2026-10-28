"""Week-1 gate. One planned list of runs, then a report read from the log.

Resamples use different seeds. The seed is part of the cache key and is not
sent to Anthropic, so a second sample is a new call rather than a replay of
the first. The full live list does not start without an explicit confirm.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ato.runner import execute

MODEL = "claude-opus-5-5"
TEMPERATURE = 0.0
# Opus list price, dollars per token. The console spend limit is the real stop.
_INPUT_USD = 4.0 / 1_000_000
_OUTPUT_USD = 20.0 / 1_000_000

CLEAN = ("s014", "s021", "s028", "s040", "s044")
FAULTED = ("s007", "s019", "s033", "s052", "s061")
SEEDS = (1, 2, 3, 4, 5)
# The s014 pilot was about 6k tokens. 35 runs at that size clear 200k,
# so the ceiling sits above one full gate and still stops a runaway.
TOKEN_CAP = 500_000


@dataclass(frozen=True)
class GateRun:
    scenario_id: str
    condition: str
    seed: int
    fault_id: str | None


def plan(*, pilot: bool, rescue: bool = False) -> list[GateRun]:
    """Pilot is one clean s014. Rescue is condition D on the five faulted scenarios."""
    if rescue:
        return [GateRun(scenario, "D", 1, "d_fid_balance") for scenario in FAULTED]
    if pilot:
        return [GateRun("s014", "B", 1, None)]
    runs = [GateRun(scenario, "B", seed, None) for scenario in CLEAN for seed in SEEDS]
    runs += [GateRun(scenario, "C1a", 1, "d_fid_balance") for scenario in FAULTED]
    runs += [GateRun(scenario, "C0", 1, "u_fid_rent") for scenario in FAULTED]
    return runs


def render_bar(done: int, total: int, label: str, width: int = 30) -> str:
    filled = 0 if total == 0 else min(width, round(width * done / total))
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total}  {label}"


def run_gate(
    *,
    db_path: Path,
    cache_dir: Path,
    dry_run: bool,
    pilot: bool,
    confirm: bool,
    rescue: bool = False,
    token_cap: int = TOKEN_CAP,
    show_progress: bool = False,
    model: str = MODEL,
) -> dict[str, object]:
    if not dry_run and not pilot and not confirm and not rescue:
        raise ValueError("the full live gate requires --confirm")
    runs = plan(pilot=pilot, rescue=rescue)
    completed = 0
    skipped = 0
    stopped = ""
    total = len(runs)
    for index, item in enumerate(runs, start=1):
        label = f"{item.scenario_id} {item.condition} seed {item.seed}"
        if _logged(db_path, item):
            skipped += 1
            _show(show_progress, index, total, f"{label} skipped")
            continue
        _show(show_progress, index - 1, total, label)
        execute(
            scenario_id=item.scenario_id,
            condition=item.condition,
            fault_id=item.fault_id,
            seed=item.seed,
            temperature=TEMPERATURE,
            model="fake" if dry_run else model,
            db_path=db_path,
            cache_dir=cache_dir,
            dry_run=dry_run,
            max_tokens=2048,
        )
        completed += 1
        _show(show_progress, index, total, label)
        spent = _spent(db_path)
        if spent >= token_cap:
            stopped = f"stopped at {spent} tokens, cap {token_cap}"
            break
    report = summarize(db_path)
    report["completed"] = completed
    report["skipped"] = skipped
    report["planned"] = len(runs)
    report["stopped"] = stopped
    return report


def summarize(db_path: Path) -> dict[str, object]:
    """Rates per condition. A row with no usable agent verdicts is a failure, not a decision.

    The `*_localization` keys are field-level: agent and field both right. With
    only the fiduciary agent in the run the agent-level rate is nearly free, so
    it is reported beside the field-level rate and never in place of it.
    """
    rows = _rows(db_path)
    by_condition: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    clean = by_condition.get("B", [])
    usable_clean = [row for row in clean if _usable(row)]
    input_tokens = sum(int(row["input_tokens"]) for row in rows)
    output_tokens = sum(int(row["output_tokens"]) for row in rows)
    return {
        "rows": len(rows),
        "agent_failures": sum(1 for row in rows if not _usable(row)),
        "stability": _stability(usable_clean),
        "stability_excluded": len(clean) - len(usable_clean),
        "false_positive_rate": _named(clean),
        "clean_blame": dict(Counter(row["blame_agent"] or "none" for row in clean)),
        "detectable_localization": _localization(by_condition.get("C1a", [])),
        "control_localization": _localization(by_condition.get("C0", [])),
        "rescue_localization": _localization(by_condition.get("D", [])),
        "agent_level": {
            name: _localization(by_condition.get(name, []), field_level=False)
            for name in ("C1a", "C0", "D")
        },
        "abstain_rate": {name: _abstained(group) for name, group in by_condition.items()},
        "reconcile_rate": {
            name: _rate(group, "reconcile_called") for name, group in by_condition.items()
        },
        "blame_refusals": sum(int(str(_column(row, "blame_refused") or 0)) for row in rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "approx_usd": input_tokens * _INPUT_USD + output_tokens * _OUTPUT_USD,
    }


def _column(row: sqlite3.Row, name: str) -> object:
    """None for a column an older log does not have, so the first gate can still be read."""
    try:
        return row[name]
    except IndexError:
        return None


def _usable(row: sqlite3.Row) -> bool:
    count = _column(row, "agent_verdict_count")
    return count is None or int(str(count)) > 0


def _abstained(rows: list[sqlite3.Row]) -> float | None:
    if not rows:
        return None
    quiet = [row for row in rows if row["blame_agent"] is None and not row["blame_parse_failed"]]
    return len(quiet) / len(rows)


def _rate(rows: list[sqlite3.Row], name: str) -> float | None:
    values = [_column(row, name) for row in rows]
    known = [int(str(value)) for value in values if value is not None]
    if not known:
        return None
    return sum(known) / len(known)


def _show(enabled: bool, done: int, total: int, label: str) -> None:
    if enabled:
        print(render_bar(done, total, label), flush=True)


def format_report(report: dict[str, object]) -> str:
    lines = [
        f"completed {report['completed']} of {report['planned']}",
        f"skipped {report.get('skipped', 0)}",
        f"agent_failures {report['agent_failures']}",
        f"stability {report['stability']} (excluded {report['stability_excluded']})",
        f"false_positive_rate {report['false_positive_rate']}",
        f"clean_blame {report['clean_blame']}",
        f"detectable_localization {report['detectable_localization']}",
        f"control_localization {report['control_localization']}",
        f"rescue_localization {report['rescue_localization']}",
        f"agent_level {report['agent_level']}",
        f"abstain_rate {report['abstain_rate']}",
        f"reconcile_rate {report['reconcile_rate']}",
        f"blame_refusals {report['blame_refusals']}",
        f"tokens in {report['input_tokens']} out {report['output_tokens']}",
        f"approx_usd {report['approx_usd']:.4f}",
    ]
    stopped = report.get("stopped") or ""
    if stopped:
        lines.append(str(stopped))
    return "\n".join(lines)


def _stability(clean: list[sqlite3.Row]) -> float | None:
    """Mean, over scenarios, of how often the chosen action matches the mode."""
    by_scenario: dict[str, list[str]] = {}
    for row in clean:
        if row["chosen_action"] is None:
            continue
        by_scenario.setdefault(str(row["scenario_id"]), []).append(str(row["chosen_action"]))
    if not by_scenario:
        return None
    scores = []
    for actions in by_scenario.values():
        mode = Counter(actions).most_common(1)[0][1]
        scores.append(mode / len(actions))
    return sum(scores) / len(scores)


def _named(rows: list[sqlite3.Row]) -> float | None:
    if not rows:
        return None
    named = [
        row for row in rows if row["blame_agent"] is not None and not row["blame_parse_failed"]
    ]
    return len(named) / len(rows)


def _localization(rows: list[sqlite3.Row], *, field_level: bool = True) -> float | None:
    """Hits over every row in the condition, so a run that failed upstream is a miss."""
    if not rows:
        return None
    hits = 0
    for row in rows:
        if row["blame_parse_failed"] or row["fault_record"] is None or not _usable(row):
            continue
        record = json.loads(str(row["fault_record"]))
        agent_ok = row["blame_agent"] == record["truth_agent"]
        field_ok = row["blame_field"] == record["truth_field"]
        if agent_ok and (field_ok or not field_level):
            hits += 1
    return hits / len(rows)


def _logged(db_path: Path, item: GateRun) -> bool:
    """Same scenario, condition, and seed is the same sample. Do not pay for it twice."""
    for row in _rows(db_path):
        same = (
            row["scenario_id"] == item.scenario_id
            and row["condition"] == item.condition
            and int(row["seed"]) == item.seed
        )
        if same:
            return True
    return False


def _rows(db_path: Path) -> list[sqlite3.Row]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM runs ORDER BY rowid"))
    finally:
        conn.close()


def watch(db_path: Path, total: int = 35) -> None:
    """Reprint one line until the log has every planned row. For a run already going."""
    seen = -1
    while True:
        rows = _rows(db_path)
        done = len(rows)
        if done != seen:
            label = ""
            if rows:
                row = rows[-1]
                label = (
                    f"{row['scenario_id']} {row['condition']} seed {row['seed']}{_eta(rows, total)}"
                )
            print("\r" + render_bar(done, total, label) + " " * 8, end="", flush=True)
            seen = done
        if done >= total:
            print(flush=True)
            return
        time.sleep(1)


def _eta(rows: list[sqlite3.Row], total: int) -> str:
    gaps: list[float] = []
    for earlier, later in zip(rows, rows[1:], strict=False):
        start = datetime.fromisoformat(str(earlier["created_at"]))
        end = datetime.fromisoformat(str(later["created_at"]))
        gap = (end - start).total_seconds()
        if 0 < gap < 120:
            gaps.append(gap)
    if not gaps:
        return ""
    left = (sum(gaps) / len(gaps)) * (total - len(rows))
    return f"  ~{int(left // 60)}m {int(left % 60):02d}s left"


def _spent(db_path: Path) -> int:
    rows = _rows(db_path)
    return sum(int(row["input_tokens"]) + int(row["output_tokens"]) for row in rows)
