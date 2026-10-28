# ATO — Autonomous Triad Optimizer

A fault-injection benchmark for multi-agent supervisor architectures, wearing a
personal-optimization assistant as its environment.

Three domain agents (Fiduciary, Wellness, Productivity) emit verdicts on
candidate actions. An arbiter picks one, then is asked in a separate call which
agent's input was most likely corrupted. Faults are injected with known ground
truth and the arbiter's attribution is scored against it.

**The benchmark is the contribution. The assistant is the environment.**

CS 5390 Agentic AI, UTEP. Dr. Jaime Acosta.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
make install
make check
```

## If you are an AI coding agent

Read `AGENTS.md` first, then `docs/SPEC.md`, `docs/DECISIONS.md`, `TASKS.md`.
The kickoff prompt is `docs/KICKOFF_PROMPT.md`.

## Layout

| Path | What |
|---|---|
| `AGENTS.md` | The brief. Non-negotiables, scope boundaries, conventions |
| `docs/SPEC.md` | The contract: schema, invariants, fault taxonomy, metrics, conditions |
| `docs/DECISIONS.md` | Why things are as they are, and the four items still open |
| `docs/REPORT.md` | Week-1 gate: counts, intervals, and the precommitted outcomes |
| `TASKS.md` | Ordered work queue with acceptance criteria |
| `ato/schema.py` | `WorldState`, `AgentVerdict`, `check_state`, `check_verdicts` |
| `ato/provider.py` | The provider seam. `FixtureProvider` (strict), `LiveProvider` (stub) |
| `ato/faults.py` | Both injection surfaces, `FaultSpec`/`FaultRecord`, the fault library |
| `ato/llm.py` | The model seam. Protocol, `cache_key`, `FakeClient` |
| `fixtures/` | Frozen JSON scenarios. Every one reconciles clean, enforced in CI |
| `tests/` | The executable spec |

## The one rule

Never edit a fixture, an expected value, or an invariant to make a test pass.
Clean fixtures are the false-positive baseline for the entire evaluation.
