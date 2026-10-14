# Kickoff prompt

Paste this into Cursor as the first message of a new session on this repo.
Do not paste the proposal. Do not paste this file's surrounding commentary.

---

Read `AGENTS.md`, then `docs/SPEC.md`, then `docs/DECISIONS.md`, then
`TASKS.md`. Then read `ato/schema.py`, `ato/provider.py`, `ato/faults.py` and
`tests/test_ato.py` in full — they are the conventions everything else follows.

Before writing any code, tell me three things:

1. In your own words, what the measurement is and why a faulted `WorldState`
   must be shape-identical to a clean one.
2. The one thing in `AGENTS.md` that you are most likely to violate while
   making a failing test pass.
3. Anything in `TASKS.md` T1 whose acceptance criteria you find ambiguous.

Then stop and wait. Do not start T1 until I answer.

Constraints for this session and every session after it:

- One task at a time, in `TASKS.md` order. `make check` green before you call a
  task done.
- Never edit a fixture, an expected value, or an invariant to make a test pass.
- Never add a framework. Orchestration is hand-rolled.
- If you want to change `ato/schema.py`, stop and tell me why first.
