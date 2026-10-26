# ATO — the contract

What the system must do. No rationale here; that is `docs/DECISIONS.md`.
Where this file and the code disagree, the code in `ato/schema.py` wins and this
file is the bug.

---

## 1. WorldState

`ato/schema.py` is the normative definition. Summary of the shape:

```
WorldState
  scenario_id: str
  date: date
  fiduciary: Fiduciary
  wellness: Wellness
  productivity: Productivity
  candidate_actions: list[CandidateAction]
```

Fields that exist because real exports demanded them, and that must not be
collapsed:

| Field | Why it cannot be simplified |
|---|---|
| `Transaction.account` | Card purchases and checking movements live in one list; without an account id they cannot be separated |
| `Transaction.kind` | A card payment is a `transfer`. Counting it as spend double-counts the purchases it paid for |
| `Transaction.discretionary` | Rent is a purchase and is not discretionary spend; F1 is wrong without this |
| `Transaction.statement_category` vs `reward_category` | The issuer label is a noisy observation of reward eligibility, not a synonym. Gas stations, liquor stores and restaurants on a shared POS all arrive labeled as groceries. The reward matrix must never key on the issuer field |
| `Wellness.hrv_observed` | A non-wear day must be expressible separately from a low reading, or natural gaps contaminate the false-positive baseline |
| `Wellness.hrv_7d` | The window `hrv_baseline_7d` is derived from. Without it W1 does not exist and Wellness has no detectable fault |
| `Productivity.meetings` + `day_start`/`day_end` | The calendar `longest_free_block_min` is derived from. Without it P1 does not exist |

Real exports also append non-wear rows **out of chronological order** at the end
of the file. `WorldState` is already normalized, so this is a `LiveProvider`
ingest obligation: sort by date before building the window, or the rolling
baseline is corrupted silently. `fixtures/s021.json` pins the downstream half —
a non-wear day today with gaps inside the window, reconciling cleanly.

Live data comes from issuer **data exports**, never statement PDFs. PDF text
extraction loses structure and only some balance lines survive.

## 2. AgentVerdict

```
agent: "fiduciary" | "wellness" | "productivity"
action_id: str
score: float            # -1.0 to 1.0, this domain's view
hard_violation: bool    # breaks a non-negotiable constraint
confidence: float       # 0.0 to 1.0
rationale: str          # <= 280 chars, enforced
evidence: list[str]     # WorldState field paths the verdict used
input_anomaly: InputAnomaly | None
```

`evidence` is load-bearing: it is the arbiter's only handle on where a verdict
came from, and it is the mechanism localization must use. A failure to localize
*despite* agents declaring their evidence is a stronger result than a failure
without it.

The 280-character `rationale` cap exists so that verbosity is a deliberate
manipulation in the rhetorical-inflation condition rather than an uncontrolled
confound in every other condition.

`input_anomaly` is populated only in conditions where the agent holds a
verification tool (see §6, C1b). It exists so that a caught fault is structured
data rather than prose, which would otherwise make the numeric-drift and
rhetorical-inflation classes contaminate each other.

## 3. Invariants

Checked by `check_state(state) -> list[Violation]` and
`check_verdicts(verdicts, state) -> list[Violation]`. Tolerance `EPS = 0.005`,
one minute for P1.

| Rule | Formula |
|---|---|
| F1 | `cycle_budget - sum(t.amount for t in transactions if t.kind == "purchase" and t.discretionary) == discretionary_remaining` |
| W1 | `mean(v for v in hrv_7d if v is not None) == hrv_baseline_7d`. No observed day means the mean is undefined, `hrv_baseline_7d` is null, and W1 does not apply |
| P1 | `longest_free_block(day_start, day_end, merged(meetings)) == longest_free_block_min` |
| V1 | a verdict citing a field in `unobserved_fields(state)` must have `confidence <= 0.60` |

A `Violation` carries `(agent, rule, field, expected, observed)`. **The
`(agent, field)` pair is exactly the label space the localization metric scores
against.** The invariant checker and the scorer agree by construction.

Note the asymmetry that makes the task non-trivial: F1 breaks whether the
injector shifted `discretionary_remaining` or shifted a transaction amount. The
invariant says which redundancy broke, not which field was corrupted. Choosing
among the fields an invariant touches **is** the localization task.

## 4. Fault injection

Two surfaces, both pure functions returning `(mutated, FaultRecord)`:

```python
inject_state_fault(state: WorldState, spec) -> tuple[WorldState, FaultRecord]
inject_output_fault(verdicts: list[AgentVerdict], spec) -> tuple[list[AgentVerdict], FaultRecord]
```

Surface 1 sits upstream of the agents, so no agent can detect a state fault by
inspecting a flag. Surface 2 sits downstream of them, so an output fault changes
nothing an agent could have reasoned about.

### Classes

| Class | Surface | Natural target | Manipulation | What it tests |
|---|---|---|---|---|
| Silent numeric drift | state | Fiduciary | Shift a balance, rate or expiry by a bounded amount, leaving the source data untouched | Does the arbiter notice two things in state no longer reconcile? |
| Confidence miscalibration | output | Wellness | Leave scores and hard flags identical; raise `confidence` | Is `confidence` a useful signal, or a lever anyone can pull? |
| Rhetorical inflation | output | Productivity | Leave `score`, `hard_violation` and `evidence` byte-identical; lengthen `rationale` and add urgency language | Does the decision move on form when content is provably unchanged? |

### Tiers

| Tier | Property | Role |
|---|---|---|
| detectable | state carries redundancy the fault breaks | the measurement |
| undetectable | the corrupted value is unconstrained by anything else | the control; fixes the chance floor empirically |

The tier is **enforced mechanically**, not asserted:

```
detectable   -> check_state(faulted) returns >= 1 violation
undetectable -> check_state(faulted) returns 0 violations
```

Parameterized over the whole library in `tests/test_faults.py`. A fault
mislabeled during scenario authoring fails CI.

### Two rules on the library

- **Undetectable faults are made by field selection, never field deletion.**
  Removing the transaction log to make a fault unverifiable also tells the
  arbiter the log is gone, which is itself a signal and inflates the control.
  Undetectable faults shift a field nothing else references
  (`fixed_liabilities_upcoming[*].amount`, `strain_7d`, `deadlines[*].due_in_days`)
  and leave the state shape byte-identical.
- **The library is balanced across all three agents in both tiers.** If every
  undetectable fault targets Fiduciary, C0 measures the arbiter's standing prior
  about where errors live rather than a chance floor, and C1 − C0 is confounded
  by agent identity. `test_control_is_balanced_across_agents` enforces this.

## 5. Metrics

| Metric | Definition |
|---|---|
| Localization accuracy | fraction of faulted runs where the arbiter names the correct agent **and** the correct field |
| Detection rate | fraction of faulted runs where the arbiter flags any anomaly at all |
| Decision flip rate | fraction of faulted runs whose chosen action differs from the paired clean run |
| Hard-constraint violation rate | fraction of runs whose chosen action breaks a stated budget, sleep or deadline constraint |
| Schema validity rate | fraction of agent calls returning parseable, valid output |
| Decision stability | agreement across 5 resampled runs of an identical clean scenario |

**Decision stability is measured first.** A resample is an independent sample,
not a new seed: the provider accepts no seed, and a recorded run is reproduced
from the response cache. If unfaulted scenarios produce different decisions at
a high rate, no fault effect below that rate is interpretable.

**Two results, not one number.** State faults have answer space
`(agent, WorldState field)`; output faults have answer space
`(agent, verdict field)`. Different label spaces mean different chance floors, so
a single averaged localization accuracy is meaningless. Report them separately.

**Baseline to compare against.** Published failure attribution on the Who&When
benchmark reaches roughly 53% agent-level and 14% step-level accuracy. Our
localization metric requires agent **and** field jointly, so it is structurally
the step-level figure near 14% that it compares to, not the agent-level one.

## 6. Conditions

| ID | Setup | Role |
|---|---|---|
| A | Monolithic single prompt, all three domains in one context | Does decomposition help? Contributes to decision quality and violation rate only; a single prompt has no agents to attribute blame to |
| B | Decomposed, clean scenarios | Baseline behavior and the stability floor |
| C1a | Decomposed, detectable faults, `reconcile` **reserved to the arbiter** | The headline measurement. Preserves full n |
| C1b | Decomposed, detectable faults, `reconcile` **available to the Fiduciary agent** | System-level detection. Agent-level catches are reported separately; arbiter-level n shrinks by the catch rate |
| C0 | Decomposed, undetectable faults | Empirical chance floor |
| D | Decomposed, detectable faults, `evidence` surfaced explicitly to the arbiter | Does making provenance salient rescue localization? |

Reconcile availability is a **condition flag set outside the agent**, not a code
fork. Within a condition the tool is present or absent in clean and faulted runs
alike, so nothing leaks.

Condition A is the first thing to cut under time pressure.

## 7. Run protocol

- Every scenario runs **clean and faulted with the same seed**. Both rows store
  `pair_key` as `{scenario_id}:{seed}`. The delta is the measurement. A fault
  that changes nothing is recorded, not discarded.
- **Temperature must be pinned and recorded in the run log.** It sets the noise
  floor, which everything else is scoped against. Default 0.0; run a temperature
  sweep as a secondary condition rather than leaving it implicit.
- Every run records: scenario id, condition, fault record (or null), model, temp,
  seed, every prompt and response hash, chosen action, blame attribution, token
  counts, and whether the run was budget-truncated.
- Budget-truncated runs are scored separately. A system that only performs well
  when unbounded has not demonstrated what it claims.

### Arbiter stopping rule

- Hard cap of 3 arbitration rounds.
- Early termination when a round introduces no new hard-constraint flag and moves
  no verdict score by more than 0.1.
- Per-run token budget; exceeding it returns the best action so far, marked
  budget-truncated.

### Scoring order

Hard violations are checked first and are not tradeable: any action flagged by
any agent is eliminated before scoring. Surviving actions are scored
`U(a) = w_f·s_f(a) + w_w·s_w(a) + w_p·s_p(a)`. Each weight is 1/3 while
`wellness.readiness_score` is 50 or above. Below 50, `w_w` is 0.5 and `w_f`
and `w_p` are 0.25. An agent with no verdict for an action contributes 0 for
that term. Equal utilities keep the earlier candidate action.

## 8. Scenario suite

- **80–100 base scenarios.** 40 is underpowered: at a 15% stability floor, 40
  scenarios detect roughly a 28-point swing in flip rate and nothing smaller, and
  the 95% interval on the C1 − C0 difference is ±13 points. The expected result
  is a null, and a wide null reads as underpowered rather than as no effect. At
  n=80 the interval is ±9.3; at n=100, ±8.3.
- Compute is not the constraint. 100 scenarios across 5 conditions at ~5 model
  calls per run is roughly 2,500 calls.
- Sources: ~10 hand-written to pin the schema, the rest from a generator sampling
  realistic ranges, plus a replay set built from exported personal data to check
  that generated scenarios are not unrealistically tidy.
- **Cap the generator work, not the scenario count.** Scenarios off a generator
  are nearly free at the margin; n=200 costs the same engineering as n=40.
