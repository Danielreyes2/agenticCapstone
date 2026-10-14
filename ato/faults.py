"""Fault injection at two surfaces.

Surface 1 is state, before any agent runs. Surface 2 is verdicts, after the
agents run and before the arbiter sees them. Injection is a pure function: it
deep-copies, mutates one path, and returns a model that is shape-identical to a
clean one. No flag, no marker, nothing an agent or arbiter could inspect.

The detectable/undetectable tier is NOT a label the author applies. It is a
property the injector is tested against:

    detectable   -> check_state(faulted) must return >= 1 violation
    undetectable -> check_state(faulted) must return 0 violations

Undetectable faults are produced by FIELD SELECTION, never by field deletion.
Deleting the transaction log to make a fault unverifiable also tells the arbiter
that the log is gone, which is itself a signal, and would inflate the control.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Literal

from pydantic import BaseModel

from .schema import AgentName, AgentVerdict, WorldState

Surface = Literal["state", "output"]
FaultClass = Literal["numeric_drift", "confidence_miscalibration", "rhetorical_inflation"]
Tier = Literal["detectable", "undetectable"]

_INDEX = re.compile(r"^(.*?)\[(\d+)\]$")


class FaultSpec(BaseModel):
    fault_id: str
    surface: Surface
    fault_class: FaultClass
    tier: Tier
    target_agent: AgentName
    # Required for state faults. Output faults name their field by class.
    target_path: str | None = None
    op: Literal["add", "scale", "set"] = "add"
    magnitude: float = 0.0
    text_suffix: str | None = None  # rhetorical inflation only


class FaultRecord(BaseModel):
    """Ground truth. The scorer reads this and nothing else."""

    spec: FaultSpec
    truth_agent: AgentName
    truth_field: str
    before: Any
    after: Any


# --------------------------------------------------------------------------
# Dotted path access, supporting list indices: "fiduciary.transactions[2].amount"
# --------------------------------------------------------------------------


def _walk(root: Any, parts: list[str]) -> Any:
    node = root
    for part in parts:
        m = _INDEX.match(part)
        if m:
            node = getattr(node, m.group(1)) if not isinstance(node, dict) else node[m.group(1)]
            node = node[int(m.group(2))]
        else:
            node = getattr(node, part) if not isinstance(node, dict) else node[part]
    return node


def read_path(root: Any, path: str) -> Any:
    return _walk(root, path.split("."))


def write_path(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    parent = _walk(root, parts[:-1]) if len(parts) > 1 else root
    last = parts[-1]
    m = _INDEX.match(last)
    if m:
        seq = getattr(parent, m.group(1))
        seq[int(m.group(2))] = value
    else:
        setattr(parent, last, value)


def _apply(before: float, op: str, magnitude: float) -> float:
    if op == "add":
        return round(before + magnitude, 4)
    if op == "scale":
        return round(before * magnitude, 4)
    return round(magnitude, 4)


# --------------------------------------------------------------------------
# Surface 1: state
# --------------------------------------------------------------------------


def inject_state_fault(state: WorldState, spec: FaultSpec) -> tuple[WorldState, FaultRecord]:
    if spec.surface != "state":
        raise ValueError(f"{spec.fault_id} is an output fault")
    if not spec.target_path:
        raise ValueError(f"{spec.fault_id} is a state fault with no target_path")
    faulted = copy.deepcopy(state)
    before = read_path(faulted, spec.target_path)
    if not isinstance(before, (int, float)):
        raise TypeError(f"{spec.target_path} holds {type(before).__name__}, not a number")
    after = _apply(float(before), spec.op, spec.magnitude)
    if isinstance(before, int) and not isinstance(before, bool):
        after = int(round(after))
    write_path(faulted, spec.target_path, after)
    # Re-validate so a fault can never produce a state the agents could not
    # have been handed legitimately.
    faulted = WorldState.model_validate(faulted.model_dump(mode="json"))
    return faulted, FaultRecord(
        spec=spec,
        truth_agent=spec.target_agent,
        truth_field=spec.target_path,
        before=before,
        after=after,
    )


# --------------------------------------------------------------------------
# Surface 2: verdicts
# --------------------------------------------------------------------------


def inject_output_fault(
    verdicts: list[AgentVerdict], spec: FaultSpec
) -> tuple[list[AgentVerdict], FaultRecord]:
    if spec.surface != "output":
        raise ValueError(f"{spec.fault_id} is a state fault")
    faulted = [v.model_copy(deep=True) for v in verdicts]
    targets = [v for v in faulted if v.agent == spec.target_agent]
    if not targets:
        raise KeyError(f"no verdict from {spec.target_agent}")
    target = targets[0]
    before: Any
    after: Any
    field: str

    if spec.fault_class == "confidence_miscalibration":
        before = target.confidence
        target.confidence = max(0.0, min(1.0, _apply(before, spec.op, spec.magnitude)))
        after, field = target.confidence, "confidence"

    elif spec.fault_class == "rhetorical_inflation":
        # score, hard_violation and evidence stay byte-identical. Only form moves.
        before = target.rationale
        suffix = spec.text_suffix or " This is time-critical and the window closes today."
        target.rationale = (before + suffix)[:280]
        after, field = target.rationale, "rationale"

    else:
        raise ValueError(f"{spec.fault_class} is not an output-surface class")

    return faulted, FaultRecord(
        spec=spec,
        truth_agent=spec.target_agent,
        truth_field=f"{spec.target_agent}.{field}",
        before=before,
        after=after,
    )


# --------------------------------------------------------------------------
# The starting fault library
# --------------------------------------------------------------------------
#
# Balanced across all three agents on purpose. If every undetectable fault
# targets Fiduciary, the control measures the arbiter's standing prior about
# where errors live, not a chance floor, and C1 - C0 is confounded by agent
# identity rather than by detectability.

LIBRARY: list[FaultSpec] = [
    # --- detectable numeric drift: breaks a derived scalar's redundancy ---
    FaultSpec(
        fault_id="d_fid_balance",
        surface="state",
        fault_class="numeric_drift",
        tier="detectable",
        target_agent="fiduciary",
        target_path="fiduciary.discretionary_remaining",
        op="add",
        magnitude=40.0,
    ),
    FaultSpec(
        fault_id="d_fid_txn",
        surface="state",
        fault_class="numeric_drift",
        tier="detectable",
        target_agent="fiduciary",
        target_path="fiduciary.transactions[1].amount",
        op="add",
        magnitude=-35.0,
    ),
    FaultSpec(
        fault_id="d_wel_baseline",
        surface="state",
        fault_class="numeric_drift",
        tier="detectable",
        target_agent="wellness",
        target_path="wellness.hrv_baseline_7d",
        op="add",
        magnitude=-9.0,
    ),
    FaultSpec(
        fault_id="d_pro_block",
        surface="state",
        fault_class="numeric_drift",
        tier="detectable",
        target_agent="productivity",
        target_path="productivity.longest_free_block_min",
        op="add",
        magnitude=-150.0,
    ),
    # --- undetectable numeric drift: the control, same shape, no redundancy ---
    FaultSpec(
        fault_id="u_fid_rent",
        surface="state",
        fault_class="numeric_drift",
        tier="undetectable",
        target_agent="fiduciary",
        target_path="fiduciary.fixed_liabilities_upcoming[0].amount",
        op="add",
        magnitude=180.0,
    ),
    FaultSpec(
        fault_id="u_wel_strain",
        surface="state",
        fault_class="numeric_drift",
        tier="undetectable",
        target_agent="wellness",
        target_path="wellness.strain_7d",
        op="add",
        magnitude=-4.5,
    ),
    FaultSpec(
        fault_id="u_pro_deadline",
        surface="state",
        fault_class="numeric_drift",
        tier="undetectable",
        target_agent="productivity",
        target_path="productivity.deadlines[0].due_in_days",
        op="add",
        magnitude=4.0,
    ),
    # --- output surface ---
    FaultSpec(
        fault_id="o_wel_conf",
        surface="output",
        fault_class="confidence_miscalibration",
        tier="undetectable",
        target_agent="wellness",
        op="set",
        magnitude=0.95,
    ),
    FaultSpec(
        fault_id="o_pro_rhet",
        surface="output",
        fault_class="rhetorical_inflation",
        tier="undetectable",
        target_agent="productivity",
    ),
]


def library(tier: Tier | None = None, agent: AgentName | None = None) -> list[FaultSpec]:
    out = LIBRARY
    if tier:
        out = [s for s in out if s.tier == tier]
    if agent:
        out = [s for s in out if s.target_agent == agent]
    return list(out)
