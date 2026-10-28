# ATO week-1 report

The proposal document is not in this repo. The outcomes this report scores against are the ones written down before the runs: `docs/SPEC.md` §5–6, D8 and D13 in `docs/DECISIONS.md`, and the T6 stop rule in `TASKS.md`.

Intervals are 95% Wilson intervals. A fraction with n = 5 is a signal. OPEN-2 settled the powered sample at 100 scenarios, and that sweep was not run.

## What was committed in advance

- Localization is the fraction of faulted runs that name the correct agent **and** the correct field. It is the step-level number. The Who&When baseline to compare it with is about 14.2% step-level, not the 53.5% agent-level figure (D8).
- The prediction on the table was that this localization collapses.
- Condition D, which shows the verdicts' evidence to the blame call, has a predicted direction: it should beat C1. If it does not beat C1 at all, suspect the harness before believing the finding (D8).
- If the arbiter cannot tell a detectable fault from an undetectable one, and D does not rescue it, drop localization and report stability instead (T6).
- Stability is measured first, on five resamples of each clean scenario, with a target above 80% (T6). The provider accepts no seed (D10). This SDK accepts no temperature, so no result may claim temperature 0 (D13).
- State faults and output faults are two results. They are not averaged (SPEC §5).

## What was run

Two gates, each 40 rows. The runner called the fiduciary agent only. The wellness and productivity agents exist and were not called. The blame call still saw the full state.

| Log | Model | Rows |
|---|---|---|
| `runs/gate2.sqlite` | `claude-opus-5-5` | 40 |
| `runs/gate-haiku.sqlite` | `claude-haiku-5-5` | 40 |

Clean arm B is five scenarios (s014, s021, s028, s040, s044), five resamples each. Detectable arm C1a is `d_fid_balance` on five other scenarios (s007, s019, s033, s052, s061). Control arm C0 is `u_fid_rent` on those same five. Condition D repeats C1a's five faults with evidence shown to the blame call.

Every row requested temperature 0.0. `temperature_sent` is null on all 80 rows. Stability is the model's default sampling.

Not run, and not given a number here: condition A, condition C1b, output-surface faults, an open-weight model, a temperature sweep, and the 100-scenario sweep. The decision flip rate is also absent. The clean scenarios and the faulted scenarios are disjoint, so a faulted row has no clean twin in the log. The hard-constraint rate is absent because the log does not record whether the chosen action broke a constraint.

An earlier Opus log, `runs/gate.sqlite`, scored 0 on every localization. The blame call in that run never saw the state. D8's instruction to suspect the harness applied. The numbers below are from the corrected logs.

## Results

Schema validity is 40/40 on both models, 1.000 with interval [0.912, 1.000]. No run was truncated. No blame call was refused.

Stability is 5/5 on every clean scenario, mean 1.000. Each scenario's interval is [0.566, 1.000]. The point estimate is above the 80% target. The interval includes values below it.

Localization and detection, state faults only. Output faults were not run, so that result is empty rather than filled in from the state rows. On these logs every detection named the correct field and every miss was an abstain, so the detection rate equals the localization rate. Agent-level equals field-level for the same reason, and because the only agent that produced a verdict was the fiduciary.

| Arm | Opus | Haiku |
|---|---|---|
| C1a detectable | 4/5 = 0.800 [0.376, 0.964] | 4/5 = 0.800 [0.376, 0.964] |
| C0 undetectable | 0/5 = 0.000 [0.000, 0.434] | 0/5 = 0.000 [0.000, 0.434] |
| D evidence shown | 5/5 = 1.000 [0.566, 1.000] | 4/5 = 0.800 [0.376, 0.964] |

The C1a miss on both models is s019, an abstain. Opus D names the field on s019 and keeps the other four, so the one-scenario gap between 4/5 and 5/5 is that rescue. Haiku D names the field on s019 and abstains on s033, which C1a had gotten right. The rate does not move.

Clean runs, where the correct answer is to abstain:

| | Opus | Haiku |
|---|---|---|
| Abstain | 24/25 = 0.960 [0.805, 0.993] | 19/25 = 0.760 [0.566, 0.885] |
| Named an agent | 1/25 = 0.040 [0.007, 0.195] | 6/25 = 0.240 [0.115, 0.434] |
| Who was named | productivity, once, on s014 seed 1 | wellness, six times |

All six Haiku false alarms point at a non-wear day. Five are every resample of s014, field `wellness.hrv_7d[2]`, which is null in the fixture. The sixth is s040 seed 3, field `wellness.hrv_7d[5]`, also null. A null day is a legitimate absence, not a fault.

The reconcile tool was not what produced the hits. Opus called it on 7 of 25 clean runs and on none of the faulted runs. Haiku never called it. The 4/5 came from the state being visible and the answer having to be a real field path.

## Reading against the commitments

**The collapse prediction is not supported.** The detectable point estimate is 0.800 on both models, against a control of 0.000 and against a step-level baseline near 0.142. The 95% interval on 4/5 is [0.376, 0.964]. It sits above 0.142 and it also overlaps the control interval [0.000, 0.434]. At n = 5 this is a separation in the point estimates, not a difference the intervals can carry. It is not a claim to have beaten Who&When, and it is not comparable to the 53.5% agent-level figure. D8 forbids that comparison.

**Discrimination is enough that the T6 pivot does not trigger.** The arbiter names the detectable field and abstains on the undetectable one, on both models. The pivot to stability as the only result is not required. Stability is reported anyway, because the spec measures it first.

**Condition D moves in the predicted direction on Opus only, by one scenario.** 5/5 against 4/5, intervals [0.566, 1.000] and [0.376, 0.964], which overlap. Haiku's D rate equals its C1a rate. After the harness fix, "D does not beat C1 at all" is true of Haiku and false of the Opus point estimate. It is not evidence that the corrected harness is still blind.

**The 80% stability target is met as a point estimate and not as an interval.** Every clean scenario agreed with itself on all five resamples. The lower bound is 0.566. Combined with a null `temperature_sent`, this is default-sampling agreement, not a pinned temperature of 0.

**The clean-run prior is "none" for Opus and "none, unless a heart-rate day is missing" for Haiku.** Opus names an agent on 4% of clean runs. Haiku names one on 24%, and every one of those names is a non-wear null. That is the quality difference this gate can support. It is not a difference in detectable localization, which matched exactly, including the shared s019 miss.

## What a later sweep would still have to show

C1b was not run, so OPEN-1 is untouched: these hits are the arbiter's, with reconcile withheld from the agent. Output-surface faults were not run, so the second result SPEC §5 requires does not exist yet. The flip rate needs each faulted scenario run clean under the same pair key. The n = 100 sweep is the sample at which the C1 − C0 interval was supposed to mean something. This gate is the evidence that the measurement can separate the two tiers. It is not that sample.
