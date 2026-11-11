# news_arms — gate 2 audit: discovery report (carried) and holdout section, second pass

Auditor, 2026-11-11, at commit `b446ee0`. Discovery report: PASS on the fifth pass (`62d17f3`,
`preregistration.md` §22–§23.4), unchanged since. Holdout section, first pass: FAIL on two
statements (§24.1). The auditor edited neither `results.md` nor the code.

**In one sentence: one number is still wrong, and I supplied it.** The new power sentence says C2
reaches 80% power "at about 0.36 σ". Recomputed under the pre-registered pass rule, the figure is
about **0.45 σ**, the same as C2's MDE in the same table. The 0.36 is an arithmetic error in the
pre-registration (14.8), and I copied it into my first-pass caveat without checking. Every other
correction checks out. The fix is one sentence, no rerun.

## Verdict table

| # | Claim | Verdict | Reason |
|---|---|---|---|
| H1 | "Power as pre-registered (14.8, 15.8): about 50% at a true 0.25 σ for C1, 80% at about 0.31 σ (C1) and 0.36 σ (C2)" (results.md, C1/C2 paragraph) | **FAIL** | Pass rule: CI excludes 0 and estimate ≥ 0.25 σ, so the binding threshold is max(0.25, 1.96·SE). C1: SE 0.0859, threshold 0.25; power at 0.25 σ is 50%; 80% at **0.32 σ**, so "about 0.31" is fine. C2: SE 0.1607, threshold 0.315; power at 0.25 σ is **34%**; 80% at **0.45 σ**, not 0.36. With 13.6's planned SE of 0.16 it is also 0.45 σ, so 14.8's 0.36 was wrong when written. Quoting it beside "MDE 0.45 σ" and "could only have seen 0.45 σ" contradicts the table. |
| H2 | C2 "underpowered, no verdict (13.6)" in the status line, the table and the information-sets paragraph; report.py prints the same unless the observed effect exceeds the MDE | **PASS** | All three places checked. report.py:240-243 renders the pass rule only for shares or when `observed_exceeds_mde` is true. The regenerated results_tables.md line reads "underpowered, no verdict (13.6)". |
| H3 | Holdout shock-day within-regime split: low +0.15 [−0.02, +0.34] (26), mid −0.31 [−0.49, −0.13] (41), high +0.57 [−0.77, +2.95] (10); sign flips; post hoc, descriptive; "the discovery pattern (loss in the high regime) is not there" | **PASS** | Matches the JSON `post_hoc_within_regime`. The high cell (10 sessions) has an interval spanning zero, so "not there" is fair. |
| H4 | Fed row: reported only (29 < 30); interval rules out −0.299 σ but contains −0.156 σ; cannot say chance vs 2022; "did not hold at its discovery size" | **PASS** | `q1.FOMC.tested: false`, n 29. The interval [−0.216, +0.097] excludes −0.299 and contains −0.156. |
| H5 | Own-σ effects (14.Q5): C1 +0.033 σ, C2 +0.069 σ; C1's interval excludes ±0.25 σ, an informative null | **PASS** | My recomputation: +0.0328, +0.0686. C1 CI [−0.071, +0.114]. |
| H6 | "no analysis code read these files before"; console exposures in §19, §21, §23 | **PASS** | Accurate. Those sections log the exposures. |
| H7 | §24.1 records the first holdout pass | **PASS** | Accurate. |
| H8 | Nothing else changed | **PASS** | `git diff e59a9ff b446ee0` touches only results.md (the lines above), preregistration.md (§24.1), report.py (C2 verdict rendering), results_tables.md (that one line) and this file. Analysis JSONs unchanged. |
| H9 | All holdout numbers (windows, σ, C1/C2 statistics, Fed row, pre-14:00 sentence, financial_stress, shock pooled, Q7, first-60, CPI options, read-once) | **PASS** (carried) | Reproduced from raw on the first holdout pass. Unchanged since. |
| D1 | Discovery report | **PASS** (carried) | Fifth pass. Unchanged. |
| Q1 | Fed days (discovery and holdout), placebo, shock days | **QUESTIONABLE** (carried) | Caveats below. |

## FAIL detail and the fix

Replace the power sentence with: "Power under the pass rule, recomputed from the realised SEs
(13.6): at a true 0.25 σ about 50% (C1) and 34% (C2); 80% power at about 0.32 σ (C1) and 0.45 σ
(C2). The pre-registered 0.36 σ for C2 (14.8) was an arithmetic error: with 13.6's own SE of 0.16
the figure is 0.45 σ."

Record the 14.8 error by a dated note in the pre-registration, not by editing 14.8 in place (the
project's rule: wrong numbers are corrected with the reason, never quietly). Nothing else needs
checking on the next pass.

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

## Not run

- §18 acceptance ρ: quoted, not recomputed. It is outside gate 2.
- Power is computed analytically (normal approximation on the reported SE), not by simulating the
  bootstrap. The difference cannot move 0.45 to 0.36.

GATE 2 VERDICT: FAIL
