"""The fault taxonomy, made mechanical.

The detectable/undetectable tier is not a label the author applies. It is a
property the injector is tested against, parameterized over the whole library and
every fixture. A fault mislabeled during scenario authoring fails here rather
than silently contaminating the control.

Do not weaken, skip or xfail anything in this file.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from ato.faults import (
    LIBRARY,
    FaultSpec,
    applies_to,
    inject_output_fault,
    inject_state_fault,
    library,
)
from ato.provider import FixtureProvider
from ato.schema import AgentVerdict, WorldState, check_state, check_verdicts

STATE_SPECS = [s for s in LIBRARY if s.surface == "state"]


def _ids(spec: FaultSpec) -> str:
    return spec.fault_id


# --- the tier distinction --------------------------------------------------


@pytest.mark.parametrize("spec", library(tier="detectable"), ids=_ids)
def test_detectable_faults_break_an_invariant(all_clean: list[WorldState], spec: FaultSpec) -> None:
    applicable = [state for state in all_clean if applies_to(spec, state)]
    assert applicable, f"{spec.fault_id} applies to no fixture"
    for state in applicable:
        faulted, record = inject_state_fault(state, spec)
        violations = check_state(faulted)
        assert violations, (
            f"{spec.fault_id} is labelled detectable but nothing in "
            f"{state.scenario_id} contradicts it"
        )
        assert record.truth_agent in {v.agent for v in violations}


@pytest.mark.parametrize(
    "spec", [s for s in library(tier="undetectable") if s.surface == "state"], ids=_ids
)
def test_undetectable_faults_break_nothing(all_clean: list[WorldState], spec: FaultSpec) -> None:
    for state in all_clean:
        faulted, _ = inject_state_fault(state, spec)
        assert check_state(faulted) == [], (
            f"{spec.fault_id} is localizable in {state.scenario_id}; it cannot serve as the control"
        )


def test_baseline_drift_does_not_apply_to_an_empty_window(fixtures_dir: Path) -> None:
    empty = FixtureProvider(fixtures_dir).load("s028")
    spec = next(s for s in LIBRARY if s.fault_id == "d_wel_baseline")
    assert applies_to(spec, empty) is False
    observed = FixtureProvider(fixtures_dir).load("s014")
    assert applies_to(spec, observed) is True


def test_control_is_balanced_across_agents() -> None:
    """If every undetectable fault targets one agent, C0 measures the arbiter's
    standing prior about where errors live, not a chance floor, and C1 - C0 is
    confounded by agent identity rather than by detectability."""
    assert {s.target_agent for s in library(tier="undetectable")} == {
        "fiduciary",
        "wellness",
        "productivity",
    }


def test_detectable_tier_is_balanced_across_agents() -> None:
    assert {s.target_agent for s in library(tier="detectable")} == {
        "fiduciary",
        "wellness",
        "productivity",
    }


# --- no leakage ------------------------------------------------------------


@pytest.mark.parametrize("spec", STATE_SPECS, ids=_ids)
def test_injection_is_pure_and_shape_preserving(clean: WorldState, spec: FaultSpec) -> None:
    before = clean.model_dump(mode="json")
    faulted, _ = inject_state_fault(clean, spec)
    assert clean.model_dump(mode="json") == before, "injection mutated its input"

    def keys(node: object, prefix: str = "") -> set[str]:
        if isinstance(node, dict):
            return {f"{prefix}.{k}" for k in node} | {
                p for k, v in node.items() for p in keys(v, f"{prefix}.{k}")
            }
        if isinstance(node, list):
            return {p for i, v in enumerate(node) for p in keys(v, f"{prefix}[{i}]")}
        return set()

    assert keys(faulted.model_dump(mode="json")) == keys(before), (
        "a faulted state must be shape-identical to a clean one"
    )


@pytest.mark.parametrize("spec", STATE_SPECS, ids=_ids)
def test_a_fault_actually_changes_something(clean: WorldState, spec: FaultSpec) -> None:
    faulted, record = inject_state_fault(clean, spec)
    assert record.before != record.after
    assert faulted.model_dump(mode="json") != clean.model_dump(mode="json")


def test_fault_ids_are_unique() -> None:
    ids = [s.fault_id for s in LIBRARY]
    assert len(ids) == len(set(ids))


def test_surfaces_and_specs_agree() -> None:
    for spec in LIBRARY:
        if spec.surface == "state":
            assert spec.target_path, f"{spec.fault_id} is a state fault with no target_path"
        else:
            assert spec.fault_class != "numeric_drift"


# --- the output surface ----------------------------------------------------


def verdicts() -> list[AgentVerdict]:
    return [
        AgentVerdict(
            agent="fiduciary",
            action_id="a1",
            score=-0.6,
            hard_violation=False,
            confidence=0.8,
            rationale="Cycle is tight with 8 days left.",
            evidence=["fiduciary.discretionary_remaining"],
        ),
        AgentVerdict(
            agent="wellness",
            action_id="a1",
            score=0.5,
            hard_violation=False,
            confidence=0.55,
            rationale="Recovery deficit is real.",
            evidence=["wellness.hrv_today", "wellness.hrv_baseline_7d"],
        ),
        AgentVerdict(
            agent="productivity",
            action_id="a1",
            score=0.7,
            hard_violation=False,
            confidence=0.7,
            rationale="Protects the morning block.",
            evidence=["productivity.longest_free_block_min"],
        ),
    ]


def test_rhetorical_inflation_leaves_every_scored_field_identical() -> None:
    spec = next(s for s in LIBRARY if s.fault_id == "o_pro_rhet")
    base = verdicts()
    faulted, record = inject_output_fault(base, spec)
    b = next(v for v in base if v.agent == "productivity")
    f = next(v for v in faulted if v.agent == "productivity")
    assert (f.score, f.hard_violation, f.confidence, f.evidence) == (
        b.score,
        b.hard_violation,
        b.confidence,
        b.evidence,
    ), "only form may move; any decision flip must be attributable to form alone"
    assert len(f.rationale) > len(b.rationale)
    assert record.truth_field == "productivity.rationale"


def test_rationale_cap_survives_inflation() -> None:
    spec = next(s for s in LIBRARY if s.fault_id == "o_pro_rhet")
    base = verdicts()
    base[2].rationale = "x" * 275
    faulted, _ = inject_output_fault(base, spec)
    assert len(faulted[2].rationale) <= 280


def test_confidence_fault_is_localizable_only_when_evidence_is_unobserved(
    clean: WorldState,
) -> None:
    """V1 is what gives confidence miscalibration a detectable variant. Same
    injected confidence either way: with HRV observed nothing contradicts it;
    on a non-wear day the verdict cites a field the world never recorded."""
    spec = next(s for s in LIBRARY if s.fault_id == "o_wel_conf")
    faulted, _ = inject_output_fault(verdicts(), spec)
    assert check_verdicts(faulted, clean) == []

    blind = clean.model_copy(deep=True)
    blind.wellness.hrv_observed = False
    blind.wellness.hrv_today = None
    violations = check_verdicts(faulted, blind)
    assert [v.rule for v in violations] == ["V1"]
    assert violations[0].agent == "wellness"


def test_output_injection_does_not_touch_other_agents() -> None:
    for spec in (s for s in LIBRARY if s.surface == "output"):
        base = verdicts()
        faulted, _ = inject_output_fault(base, spec)
        for b, f in zip(base, faulted, strict=True):
            if b.agent == spec.target_agent:
                continue
            assert b.model_dump() == f.model_dump(), f"{spec.fault_id} leaked into {b.agent}"


def test_state_and_output_injectors_reject_each_others_specs(clean: WorldState) -> None:
    state_spec, output_spec = STATE_SPECS[0], next(s for s in LIBRARY if s.surface == "output")
    with pytest.raises(ValueError):
        inject_output_fault(verdicts(), state_spec)
    with pytest.raises(ValueError):
        inject_state_fault(clean, output_spec)


def test_no_two_state_specs_target_the_same_path_and_tier() -> None:
    for a, b in itertools.combinations(STATE_SPECS, 2):
        assert (a.target_path, a.tier) != (b.target_path, b.tier)
