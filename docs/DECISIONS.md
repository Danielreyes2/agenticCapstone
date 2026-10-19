# Decisions

Settled choices and their reasoning, plus what is still open. An agent changing
anything in `ato/schema.py` adds an entry here in the same commit.

Status: **settled** = do not relitigate without a new entry. **open** = a real
fork; do not pick one silently.

---

## D1 — Shape validation and consistency checking are separate passes · settled

Pydantic rejects malformed states. `check_state()` reports inconsistent ones.
They cannot be the same pass: if a validator rejected a state whose derived
values disagree with their sources, a detectable fault would be unconstructible.
The thing the benchmark measures has to be representable.

## D2 — All three domains carry a derived value over its source data · settled

An earlier draft held that numeric drift was the only fault class supporting
localization. That was wrong. It was a property of the schema, not of the fault
classes: Fiduciary was the only domain carrying a derived number alongside the
data it came from, so only Fiduciary faults were localizable. Wellness got
`hrv_7d` behind `hrv_baseline_7d`; Productivity got `meetings` plus a day window
behind `longest_free_block_min`; the verdict surface got V1. Localization is now
a three-way question.

## D3 — The detectable/undetectable tier is tested, not asserted · settled

`detectable` must produce at least one violation; `undetectable` must produce
none. Parameterized over the entire fault library. A fault mislabeled during
scenario authoring fails CI instead of silently contaminating the control.

## D4 — Undetectable faults use field selection, never field deletion · settled

The original design made the undetectable variant by removing the transaction
log. Missing data is itself a signal: an arbiter that notices the log is gone can
blame Fiduciary for that reason alone, inflating C0 and shrinking C1 − C0 for a
reason unrelated to detectability. Undetectable faults shift a field nothing else
references and leave the state shape identical.

## D5 — The fault library is balanced across all three agents in both tiers · settled

If every undetectable fault targets Fiduciary, C0 measures the arbiter's standing
prior about where errors live rather than a chance floor. Report the per-agent
distribution of blame attributions in C0 directly; that prior is a finding on its
own.

## D6 — No agent framework. Hand-rolled orchestration · settled

Not LangGraph, not CrewAI, not AutoGen. Three reasons. The fault injector must
sit at two precise seams, and a framework's supervisor abstraction is exactly
what makes intercepting verdicts pre-arbitration awkward. The whole orchestration
is a fixed five-call sequence with a bounded loop — roughly 200 lines. And "we
built the loop" is more defensible against the course brief's thin-wrapper
warning than "we configured a framework."

## D7 — Model access through a thin `LLMClient` Protocol · settled

One interface, an Anthropic adapter now, an open-weight adapter later for the
secondary condition. No vendor SDK is imported outside `ato/llm.py`. This matches
the provider seam already used for world data, and it keeps the 8–10B secondary
condition a second adapter rather than a refactor.

Responses are cached on `(model, temperature, seed, prompt_hash)` so a rerun
after a harness bug does not re-spend the budget and does not change the numbers.

## D8 — Who&When baselines · settled, corrects an earlier error

An earlier draft claimed published agent-level attribution accuracy was "below
10%". That is false; it conflated the step-level figure with the agent-level one.
The original Who&When paper reports 53.5% agent-level and 14.2% step-level for
its best method, with some methods below random; later work reports 51–62% at the
agent level.

Consequence: our prediction that localization collapses is a claim that
*contradicts* the published figures, not one that replicates them. And because
our metric requires agent **and** field jointly, it compares structurally to the
step-level number near 14%, not the agent-level one.

Second consequence: condition D has a predicted direction. Attribution with
complete trace information reaches roughly 66% agent-level and 30% step-level,
substantially above output-only traces. If D does not beat C1 at all, suspect the
harness before believing the finding.

## D9 — Scope exclusions · settled

Preference learning, prompt-injection security, award-travel search. All three
are out, and none gets a stub or a "room for later" abstraction. The confirmation
log is written from day one; nothing is modeled from it.

## D10 — No provider-level seeded determinism · settled

The Anthropic Messages API takes no seed. Reproducibility comes from the
response cache, not from the model. "Five reseeded runs" in SPEC section 5
therefore means five independent samples, not five seeds. That is still a
valid stability floor, arguably a better one since it measures the
nondeterminism the system actually faces, but every mention must say
resampling rather than reseeding.

---

## OPEN-1 — Does the Fiduciary agent hold `reconcile`? · open, leaning

If the agent reconciles on every run, a detectable drift fault is caught at the
agent level and never reaches the arbiter, so blame attribution is never
exercised and the central metric has no data.

This is not a design choice to settle; it is an experimental factor. Run it both
ways: **C1a** with reconcile reserved to the arbiter, **C1b** with the agent
holding it.

The catch that makes this urgent rather than interesting: if the agent catches
70% of drift faults, the arbiter-level metric in C1b runs on 30% of the sample.
At n=40 that is n=12 and a 95% interval of roughly ±28 points. **OPEN-1 is a
power problem, not a design detail.** C1a is the headline condition because it
preserves full n.

Leaning: both conditions, C1a as headline, C1b reported as system-level detection
("the agent caught it X% of the time, the arbiter caught Y% of the rest").

## OPEN-2 — Scenario count · open, leaning 80–100

See `docs/SPEC.md` §8 for the numbers. 40 is underpowered for a null result.
Leaning 80–100. Blocked on nothing; decide before the generator is written, since
the generator makes the count nearly free.

## OPEN-3 — Temperature · open, must be decided before the week-1 gate

Nothing in the original proposal set it, and it is the dial that sets the
decision-stability noise floor — the number the spec calls the one that
determines whether the rest of the evaluation means anything. Default 0.0 and run
a sweep as a secondary condition. Must be recorded in every run log row
regardless.

## OPEN-4 — Does the arbiter produce a usable blame attribution at all? · open, gates everything

If it refuses to name an agent, or names one at random with no discrimination,
the central metric collapses. This is what the week-1 gate (`TASKS.md` T6) tests.

If the gate fails and condition D does not rescue it, the honest pivot is
decision stability and form-sensitivity as the primary results. Both are
measurable regardless and still say something real about supervisor
architectures. Do not spend weeks propping up a metric that does not work.
