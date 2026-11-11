# news_arms — gate 2 audit: discovery report and holdout section, final

Auditor, 2026-11-11, at commit `807bbaa`. Discovery report: PASS on the fifth pass (`62d17f3`,
`preregistration.md` §22–§23.4), unchanged since. Holdout section: FAIL on its first pass (two
statements, §24.1) and second pass (one power figure, §24.2), each corrected in place. The auditor
edited neither `results.md` nor the code.

**In one sentence: nothing failing remains in either the discovery report or the holdout section,
so gate 2 passes.** The study's answer stands: the frozen runners do not earn a detectably
different return on calendar-release days (C1 shares: no difference, an informative null; C2
options: underpowered, no verdict), and the Fed-day loss seen in discovery did not hold at its
discovery size on the holdout.

## Verdict table

| # | Claim | Verdict | Reason |
|---|---|---|---|
| H1 | Power sentence: "at a true 0.25 σ about 50% (C1) and 34% (C2); 80% power at about 0.32 σ (C1) and 0.45 σ (C2); the pre-registered 0.36 σ for C2 (14.8) was an arithmetic error … (§24.2)" | **PASS** | Recomputed: threshold max(0.25, 1.96·SE) plus 0.84·SE gives 0.322 (C1, SE 0.0859) and 0.450 (C2, SE 0.1607); 0.448 at 13.6's planned SE of 0.16. Power at 0.25 σ: 0.50 and 0.34. The text now agrees with C2's MDE of 0.45 σ in the same table. |
| H2 | §24.2 records the 14.8 error with its derivation; 14.8 and 15.8 left as written | **PASS** | Derivation correct (1.96 × 0.16 = 0.31 σ binding; plus 0.84 × SE ≈ 0.45 σ). The correction is made by a dated note, not by rewriting. |
| H3 | Nothing else changed | **PASS** | `git diff b446ee0 807bbaa` touches only that results.md sentence, §24.2 and this file. No code, JSON or table changed. |
| H4 | All other holdout claims: windows (908 / 162,374 shares; 248 / 5,993 options; 0 lost); discovery σ used; C1 +0.024 σ [−0.07, +0.11], pass rule FAIL, informative null; C2 +0.057 σ [−0.17, +0.31], underpowered, no verdict (13.6); Fed row −0.055 σ on 29 sessions, reported only; pre-14:00 $250 raw / $150 excess; within-regime split stated with its sign flip; financial_stress; Q7; first-60; CPI options; own-σ effects; read once | **PASS** (carried) | Reproduced from the raw replay files on the first holdout pass; text corrections verified on the second; unchanged since. |
| D1 | Discovery report, every claim | **PASS** (carried) | Fifth pass. Unchanged. |
| Q1 | Fed days (discovery and holdout) | **QUESTIONABLE** | Caveat below. |
| Q2 | Pre-registered placebo ranks (discovery) | **QUESTIONABLE** | Caveat below. |
| Q3 | Shock days (discovery and holdout) | **QUESTIONABLE** | Caveat below. |

## Caveats to repeat verbatim (not blocking)

- **Fed days:** "Exploratory in discovery (failed the multiple-testing correction, half of it 2022);
  on the holdout −0.055 σ [−0.22, +0.10] on 29 sessions, below the 30-session minimum. The holdout
  rules out the discovery size but cannot say whether discovery was chance or a 2022 effect."
- **C1:** "No difference on release days; the interval excludes ±0.25 σ."
- **C2:** "Underpowered, no verdict: it could detect only about 0.45 σ."
- **Placebo:** "The pre-registered placebo favours 'effect' by construction. Read it beside the
  in-pool rank."
- **Shock days:** "Descriptive; salience-selected list; unstable in discovery, pooled near zero on the
  holdout, sign flipping across regimes."

## Notes (not blocking)

- **The holdout has been read once and is spent.** This line reopens the lock, which is only a
  record of the gate. Any further `analyze --set holdout` run would be a second read and must not
  happen.
- **Recorded exposures:** console exposures of individual holdout daily nets are logged in §19, §21
  and §23. The lock was briefly open during discovery pass three while a mistaken PASS draft stood
  (§23.2). Neither is evidence of an analysis read.

## Not run

- §18 live-vs-replay acceptance ρ: quoted, not recomputed. It is outside gate 2.
- Power is computed analytically (normal approximation on the reported SE), not by bootstrap
  simulation.

GATE 2 VERDICT: PASS
