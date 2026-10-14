"""The provider seam.

Agents never fetch their own data. They receive a `WorldState` and cannot tell
which provider produced it, which is what lets the benchmark build and the
personal build be the same system.

`FixtureProvider` validates clean fixtures in strict mode by default. That is
not hygiene -- it is load-bearing. If a clean fixture does not reconcile, every
clean run registers an anomaly, the false-positive baseline is garbage, and the
detectable-fault result is uninterpretable. Strict mode catches that at load
time instead of in December.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from .schema import WorldState, check_state


class DirtyFixtureError(ValueError):
    """A fixture that was supposed to be clean does not reconcile."""


@runtime_checkable
class Provider(Protocol):
    def get_state(self, day: datetime.date) -> WorldState: ...


class FixtureProvider:
    """Reads frozen JSON scenarios. The benchmark build always uses this."""

    def __init__(self, root: str | Path, strict: bool = True) -> None:
        self.root = Path(root)
        self.strict = strict

    def load(self, scenario_id: str) -> WorldState:
        path = self.root / f"{scenario_id}.json"
        state = WorldState.model_validate(json.loads(path.read_text()))
        if self.strict:
            violations = check_state(state)
            if violations:
                detail = "; ".join(
                    f"{v.rule} {v.field}: expected {v.expected}, state says {v.observed}"
                    for v in violations
                )
                raise DirtyFixtureError(f"{scenario_id} does not reconcile when clean -- {detail}")
        return state

    def get_state(self, day: datetime.date) -> WorldState:
        for path in sorted(self.root.glob("*.json")):
            if json.loads(path.read_text()).get("date") == day.isoformat():
                return self.load(path.stem)
        raise KeyError(f"no fixture for {day.isoformat()}")

    def scenario_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))


class LiveProvider:
    """Personal build only. Never fault-injected.

    Three adapters produce the same `WorldState` shape:
      - Oura API v2 (personal access token) -> Wellness
      - Google Calendar API (read-only scope) -> Productivity
      - issuer data export, NOT statement PDF -> Fiduciary

    The PDF point is from real exports: spreadsheet and CSV exports parse
    cleanly, statement PDFs lose structure and only some balance lines survive.

    Deliberately unimplemented. Schema freeze is the team's blocking dependency;
    the live adapters are not, and writing them now would invite schema changes
    driven by whichever API was easiest to read this week.
    """

    def get_state(self, day: datetime.date) -> WorldState:
        raise NotImplementedError(
            "LiveProvider lands after the schema freeze. Benchmark build is unaffected."
        )
