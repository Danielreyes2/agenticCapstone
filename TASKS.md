# Work queue

One at a time, in order. Each ends with `make check` green and a commit. Do not
start a task while the one above it is unfinished. If acceptance criteria are
ambiguous, stop and ask rather than choosing an interpretation.

`[x]` = shipped in the handoff pack.

---

## [x] T0 — Scaffold, schema, provider seam, fault injector, model seam

Already in this repo and green under `make check`: `ato/schema.py`,
`ato/provider.py`, `ato/faults.py`, `ato/llm.py` (Protocol, `LLMResponse`,
`cache_key`, `FakeClient`), two fixtures, 46 tests, ruff clean, mypy strict
clean. Read all of it before T1 — it defines the conventions everything else
follows, and the `LLMClient` Protocol is already pinned, so do not redesign it.

---

## [ ] T1 — `ato/llm.py`: Anthropic adapter and the response cache

The Protocol, `LLMResponse`, `cache_key` and `FakeClient` are shipped. Do not
change their shapes; everything downstream is written against them.

**Build**
- `AnthropicClient` implementing `LLMClient`. The only file in the repo that
  imports a vendor SDK. Retries on transient errors with backoff; a retry must
  not change the cache key.
- `CachingClient` wrapper: content-addressed on
  `sha256(model | temperature | seed | serialized messages | response_model name)`,
  stored under `.cache/llm/`. A cache hit must be byte-identical to the miss.
- Structured output: a verdict is parsed into `AgentVerdict` with Pydantic.
  On a parse failure, retry once with the validation error appended, then record
  a schema-validity failure and return `None`. Never silently coerce.

**Acceptance**
- `tests/test_llm.py` passes with a `FakeClient` returning canned responses; no
  network in the test suite, ever.
- A recorded run replays from cache with zero API calls and produces identical
  output.
- `grep -rn "anthropic" ato/ --include=*.py` matches only `ato/llm.py`.

---

## [ ] T2 — Ten hand-written fixtures

**Build**
`s014` (the canonical worked example) and `s021` (non-wear day, mid-day free
block, non-discretionary rent, three mislabeled merchants) are the pattern.
Extend to ten, each pinning something real-export behavior taught us — not ten
variations of s014. Still missing, at minimum:
- Both candidate actions trip a hard violation.
- A cycle with zero discretionary headroom left.
- A week where every day in `hrv_7d` is a non-wear day (baseline undefined).
- A calendar with back-to-back meetings that must merge into one busy block.
- A card whose category cap is already exhausted, so the reward lookup must fall
  through to a second card.

**Acceptance**
- `test_every_fixture_reconciles_clean` already sweeps `fixtures/*.json`; it must
  stay green as fixtures are added.
- The tier tests in `tests/test_faults.py` already sweep every fault in `LIBRARY`
  against every fixture. If a new fixture breaks a spec (an index-targeted fault
  landing on a different row), fix the **spec** to target by predicate rather
  than index — do not narrow the test.
- No fixture was made by copying an existing one and changing one number.

---

## [ ] T3 — The Fiduciary agent

**Build**
- `ato/agents/base.py`: a bounded reason-act-observe loop. Hard cap on tool
  calls, a token budget, and a truncation flag on the verdict.
- `ato/agents/fiduciary.py` with deterministic tools: `card_recommendation`,
  `unused_credits`, `reconcile`, `cost_of_deferral`. Signatures in the proposal;
  all pure Python, no LLM calls inside a tool.
- `reconcile` availability is a **constructor argument**, not a module-level
  flag — see `docs/DECISIONS.md` OPEN-1. Default off (C1a).
- When reconcile is available and finds a discrepancy, the agent fills
  `input_anomaly` as structured data. It must never describe the discrepancy in
  `rationale` prose instead.

**Acceptance**
- The agent reads only `state.fiduciary` and `state.candidate_actions`. A test
  asserts it raises if handed a state slice it should not see.
- Every tool is deterministic: same input, same output, no clock, no network.
- `test_reconcile_is_a_condition_not_a_fork`: the same agent class with
  `reconcile=False` emits no `input_anomaly` on a faulted state, and with
  `reconcile=True` emits one naming `fiduciary.discretionary_remaining`.
- Verdicts validate against `AgentVerdict` including the 280-char cap.

---

## [ ] T4 — The arbiter

**Build**
- `ato/arbiter.py`. Hard violations eliminate an action before scoring. Weighted
  score over surviving actions, weights hand-set and state-conditioned.
- The stopping rule from `docs/SPEC.md` §7, exactly: 3-round cap, early
  termination on no-new-flag and no-score-move-over-0.1, token budget with a
  truncation marker.
- `attribute_blame(verdicts, decision) -> BlameAttribution` as a **separate model
  call made after the decision is returned**. It must be structurally impossible
  for the blame call to influence the decision; a test should prove this by
  asserting the decision is identical with the blame call stubbed out.

**Acceptance**
- `test_blame_cannot_reshape_the_decision` passes.
- `test_stopping_rule_terminates` over adversarial canned verdicts that would
  otherwise loop.
- Budget-truncated runs are flagged, not dropped.

---

## [ ] T5 — Runner and run log

**Build**
- `ato/runner.py`: given a scenario id, condition, and optional `FaultSpec`,
  execute the full pipeline and write one row.
- SQLite schema per `docs/SPEC.md` §7. Every field in that list, including
  temperature, seed, model, prompt and response hashes, and the truncation flag.
- `scripts/run.py` CLI: `--scenario`, `--condition`, `--fault`, `--seed`,
  `--temperature`, `--dry-run`.
- Clean and faulted runs of a scenario share a seed. The pairing is explicit in
  the schema, not inferred at analysis time.

**Acceptance**
- A full clean run on s014 writes exactly one row with no nulls in the required
  columns.
- `--dry-run` executes the whole pipeline against `FakeClient` with no API calls.
- Re-running an identical invocation hits the cache and produces an identical
  row except its timestamp.

---

## [ ] T6 — **GATE: the week-1 experiment.** Stop here and report.

This is the gate the whole project hangs on. Do not proceed to T7 without a
human reading the numbers.

**Run**
- Ten hand-written scenarios, Fiduciary agent only, arbiter asked to attribute
  blame on five faulted and five clean runs.
- Five reseeded runs of each clean scenario for the stability floor.

**Report**
- Blame attribution: does it discriminate above the C0 chance floor?
- Decision stability across reseeds — target above 80%.
- The per-agent distribution of blame attributions on clean runs (the arbiter's
  standing prior).
- False-positive rate: how often it names someone on a clean run.

**Stop condition.** If the arbiter cannot discriminate at all, and surfacing
`evidence` explicitly does not rescue it, the fault-localization framing is dead.
The pivot is decision stability and form-sensitivity as the primary results. Do
not proceed on the assumption it will work out.

---

## [ ] T7 — Wellness and Productivity agents

Same shape as T3. Tools from the proposal: `readiness`, `energy_requirement`,
`recovery_trend`; `fragmentation`, `deadline_pressure`, `reschedule_candidates`.

**Acceptance**: each agent reads only its own slice; all tools deterministic;
`test_agents_never_see_each_others_verdicts`.

---

## [ ] T8 — Scenario generator

**Build**: sample realistic ranges into valid `WorldState` objects. Every
generated scenario must satisfy `check_state() == []` by construction, not by
rejection sampling.

**Acceptance**: 200 generated scenarios, zero violations, and a distribution
check against the replay set so generated scenarios are not unrealistically tidy.
Resolve `docs/DECISIONS.md` OPEN-2 (count) before running the sweep.

---

## [ ] T9 — Evaluation harness

**Build**: metrics from `docs/SPEC.md` §5, computed from the run log only. State
faults and output faults reported as **two separate results**, never averaged.
Confidence intervals on every proportion. The C0 per-agent blame distribution
reported explicitly.

**Acceptance**: metrics computed from a fixture run log with hand-checked
expected values. No metric reads anything but the run log.

---

## [ ] T10 — Condition sweep

A, B, C1a, C1b, C0, D. Then the open-weight secondary condition. Then the
temperature sweep. Cut A first if time is short.

---

## [ ] T11 — Report

Numbers, intervals, the ablations that make them interpretable, and the outcomes
that were committed to in advance in the proposal.
