"""Condition sweep over generated scenarios.

Condition A is cut. It has no agents to attribute blame to, and the spec says
it is the first arm to drop. The open-weight arm is not run: D7 leaves it as a
second adapter, and none is installed. Temperature is not swept: D13, this SDK
accepts no temperature, so varying the requested value would log a sweep that
never reached the model. Every row records the request as 0.0 and leaves
`temperature_sent` null.

Faulted arms use seed 1, the same seed as the first clean resample, so the pair
key matches. The other four clean seeds are the stability floor only.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ato.eval import Evaluation, evaluate
from ato.faults import LIBRARY, FaultSpec, applies_to
from ato.gate import render_bar
from ato.generate import SWEEP_COUNT, generate
from ato.runner import execute
from ato.schema import WorldState

MODEL = "claude-opus-5-5"
TEMPERATURE = 0.0
# One clean resample is about the size of a gate run. 100 scenarios at nine
# runs each sit near 15k tokens apiece, so the cap clears one sweep and stops
# a second accidental one.
TOKEN_CAP = 20_000_000
_STABILITY_SEEDS = (1, 2, 3, 4, 5)
_FAULT_SEED = 1
_DETECTABLE = tuple(
    spec for spec in LIBRARY if spec.surface == "state" and spec.tier == "detectable"
)
_CONTROL = tuple(
    spec for spec in LIBRARY if spec.surface == "state" and spec.tier == "undetectable"
)


@dataclass(frozen=True)
class SweepRun:
    scenario_id: str
    condition: str
    seed: int
    fault_id: str | None


@dataclass(frozen=True)
class SweepResult:
    evaluation: Evaluation
    completed: int
    skipped: int
    planned: int
    stopped: str


def plan(states: Sequence[WorldState]) -> list[SweepRun]:
    """Clean resamples, then the four faulted arms on the paired seed.

    C1a, C1b and D share a fault on a scenario. The only change between them is
    the condition flag. C0 takes the next undetectable fault, so the control is
    not the same number shifted a second time.
    """
    detectable = _rotate(_DETECTABLE, states)
    control = _rotate(_CONTROL, states)
    runs: list[SweepRun] = []
    for state, hit, miss in zip(states, detectable, control, strict=True):
        runs.extend(SweepRun(state.scenario_id, "B", seed, None) for seed in _STABILITY_SEEDS)
        for condition in ("C1a", "C1b", "D"):
            runs.append(SweepRun(state.scenario_id, condition, _FAULT_SEED, hit))
        runs.append(SweepRun(state.scenario_id, "C0", _FAULT_SEED, miss))
    return runs


def run_sweep(
    *,
    count: int = SWEEP_COUNT,
    scenario_seed: int,
    db_path: Path,
    cache_dir: Path,
    scenarios_dir: Path,
    dry_run: bool,
    confirm: bool,
    token_cap: int = TOKEN_CAP,
    show_progress: bool = False,
    model: str = MODEL,
) -> SweepResult:
    states = generate(count, seed=scenario_seed)
    runs = plan(states)
    if not dry_run and not confirm:
        raise ValueError(f"the live sweep is {len(runs)} runs and requires --confirm")
    _write_scenarios(states, scenarios_dir)
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
            fixtures_dir=scenarios_dir,
            dry_run=dry_run,
        )
        completed += 1
        _show(show_progress, index, total, label)
        if _spent(db_path) >= token_cap:
            stopped = f"stopped at {_spent(db_path)} tokens, cap {token_cap}"
            break
    return SweepResult(evaluate(db_path), completed, skipped, total, stopped)


def format_sweep(result: SweepResult) -> str:
    report = result.evaluation
    violation = report.hard_constraint_violation
    lines = [
        f"completed {result.completed} of {result.planned}",
        f"skipped {result.skipped}",
        f"stability_mean {report.stability_mean}",
        f"schema_validity {report.schema_validity.value}",
        f"truncated {report.truncated}",
        f"c0_blame {report.c0_blame}",
        f"hard_constraint_violation {None if violation is None else violation.value}",
    ]
    for surface, groups in (("state", report.state), ("output", report.output)):
        for condition, rates in groups.items():
            lines.append(
                f"{surface} {condition} localization {rates.localization.value} "
                f"detection {rates.detection.value} flip {rates.flip.value}"
            )
    if result.stopped:
        lines.append(result.stopped)
    return "\n".join(lines)


def _rotate(specs: Sequence[FaultSpec], states: Sequence[WorldState]) -> list[str]:
    """One applicable fault per scenario, cycling the library so agents stay balanced."""
    chosen: list[str] = []
    cursor = 0
    for state in states:
        picked: FaultSpec | None = None
        for step in range(len(specs)):
            spec = specs[(cursor + step) % len(specs)]
            if applies_to(spec, state):
                picked = spec
                cursor = (cursor + step + 1) % len(specs)
                break
        if picked is None:
            raise ValueError(f"{state.scenario_id} accepts none of the sweep faults")
        chosen.append(picked.fault_id)
    return chosen


def _write_scenarios(states: Sequence[WorldState], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for state in states:
        path = directory / f"{state.scenario_id}.json"
        path.write_text(state.model_dump_json())


def _logged(db_path: Path, item: SweepRun) -> bool:
    for scenario_id, condition, seed in _keys(db_path):
        if scenario_id == item.scenario_id and condition == item.condition and seed == item.seed:
            return True
    return False


def _keys(db_path: Path) -> list[tuple[str, str, int]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    try:
        return list(conn.execute("SELECT scenario_id, condition, seed FROM runs"))
    finally:
        conn.close()


def _spent(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) FROM runs"
        ).fetchone()
    finally:
        conn.close()
    return int(row[0]) if row is not None else 0


def _show(enabled: bool, done: int, total: int, label: str) -> None:
    if enabled:
        print(render_bar(done, total, label), flush=True)
