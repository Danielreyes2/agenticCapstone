# ATO — agent brief

Read this before touching anything. It is short on purpose. The contract is in
`docs/SPEC.md`; the reasoning behind every settled choice is in
`docs/DECISIONS.md`; the work queue is `TASKS.md`.

## What this is

A fault-injection benchmark for multi-agent supervisor architectures, wearing a
personal-optimization assistant as its environment. Three domain agents
(Fiduciary, Wellness, Productivity) emit verdicts on candidate actions; an
arbiter picks one and is then asked, in a separate call, which agent's input was
most likely corrupted. We inject faults with known ground truth and score
whether the arbiter localizes them.

**The benchmark is the contribution. The assistant is the environment.** When a
change would make the assistant better advice but the measurement worse, the
measurement wins.

## Things that must not happen

1. **Never edit a fixture, an expected value, or an invariant to make a test
   pass.** A clean fixture that fails `check_state` is a broken fixture, and the
   fix is the fixture's data, never the assertion or the rule. Clean fixtures are
   the false-positive baseline for the entire evaluation. If you cannot make a
   test pass without relaxing a check, stop and say so.
2. **Never weaken, delete, or `xfail` a test in `tests/test_invariants.py` or
   `tests/test_faults.py`.** Those are the executable spec, not test code.
3. **Never add a field, flag, marker, or log line to `WorldState` or
   `AgentVerdict` that reveals whether a fault was injected.** The whole
   measurement rests on faulted state being indistinguishable from clean state
   by inspection. Fault metadata lives in `FaultRecord`, which the agents and the
   arbiter never see.
4. **Never let an agent reach outside its `WorldState` slice.** No network calls,
   no file reads, no clock reads inside an agent. All world data arrives through
   the provider.
5. **Do not change `ato/schema.py` without adding an entry to
   `docs/DECISIONS.md`.** The schema is frozen; everything else depends on it.

## Hard scope boundaries

Out of scope this semester. Do not build, scaffold, stub, or "leave room for"
any of these, and do not suggest them in a PR:

- Preference learning from the confirmation log. The log is written from day one;
  nothing is modeled from it.
- Prompt-injection or adversarial-security framing. Faults are upstream data
  corruption, not attacks.
- Award travel and points-redemption search.
- Any web scraping, any live financial API beyond a local CSV/JSON import.
- A UI. This is a benchmark and a CLI.

## The four invariants

Every domain carries a derived number standing over the data it came from. That
redundancy is the entire detectable fault tier.

| Rule | Derived value | Checked against |
|---|---|---|
| F1 | `fiduciary.discretionary_remaining` | `cycle_budget` minus discretionary purchases |
| W1 | `wellness.hrv_baseline_7d` | mean of the observed days in `hrv_7d` |
| P1 | `productivity.longest_free_block_min` | the day window minus merged `meetings` |
| V1 | a verdict's `confidence` | whether the `evidence` it cites was observed at all |

Shape validation (Pydantic) and consistency checking (`check_state`) are
deliberately separate passes. A faulted state must remain constructible.

## Architecture, and the two orderings that carry it

```
Provider -> Fault injector -> 3 agents -> Arbiter -> Blame call -> Gate
```

- **Injection sits upstream of the agents.** A faulted state is indistinguishable
  downstream. Never thread a corruption flag through an agent.
- **Blame is asked after the decision commits**, in a second model call. Asking
  both at once lets the doubt reshape the decision it is meant to evaluate.

## Conventions

- Python 3.12. Pydantic v2 for every boundary. pytest. SQLite for the run log.
- **No agent framework.** No LangGraph, no CrewAI, no AutoGen. The orchestration
  is a fixed five-call sequence with a bounded loop, roughly 200 lines. If you
  find yourself wanting a framework, you are building the wrong thing.
- Model access goes through the `LLMClient` Protocol in `ato/llm.py`. Nothing
  else imports a vendor SDK.
- Agent tools are deterministic Python functions. A tool that is itself an LLM
  call is the thin-wrapper failure the course brief warns against.
- Type hints everywhere. `from __future__ import annotations` at the top.
- Comments explain *why*, never *what*. If a line needs a comment to say what it
  does, rewrite the line.

## Working rules

- One task from `TASKS.md` at a time, in order. Each ends with the full suite
  green and the changes left unstaged for my review. Never run a git write
  command: no commit, no add, no branch, no tag. I handle all version control
  myself.
- Write the test before the implementation where the task has acceptance
  criteria stated as a test.
- Do not refactor files the current task does not name.
- When a task's acceptance criteria are ambiguous, say so and stop. Do not pick
  an interpretation and build on it silently.
- `make check` must pass before you call a task done.

## Where things live

```
ato/schema.py     WorldState, AgentVerdict, check_state, check_verdicts
ato/provider.py   Provider protocol, FixtureProvider, LiveProvider (stub)
ato/faults.py     FaultSpec, FaultRecord, both injection surfaces, LIBRARY
ato/llm.py        LLMClient Protocol, FakeClient, cache_key; adapters land in T1
fixtures/         frozen JSON scenarios; every one must reconcile clean
tests/test_invariants.py  F1/W1/P1 and the fixture discipline
tests/test_faults.py      the tier distinction, leakage, both surfaces
tests/test_llm.py         the model seam; no test here may hit the network
tests/conftest.py         shared fixtures
docs/SPEC.md      the contract
docs/DECISIONS.md why things are the way they are, and what is still open
TASKS.md          the work queue
```
