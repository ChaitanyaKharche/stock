# news_arms — gate 2 audit, fifth pass

Auditor, 2026-11-11, at commit `2f04abc`. Earlier passes are recorded in `preregistration.md` §22,
§23.1, §23.2, §23.3 and in the git history of this file. The auditor edited neither `results.md`
nor the code.

**In one sentence: nothing failing remains, so gate 2 passes, and the holdout may be read once.**
The last false clause is corrected. Every number in results.md was reproduced from the raw replay
files on the second pass, and no code, JSON or table has changed since. The three findings below
must carry their caveats wherever they are quoted.

## Verdict table

| # | Claim | Verdict | Reason |
|---|---|---|---|
| 1 | Fed days, shares (F2): 39 sessions, −0.299 σ [−0.54, −0.03], −$958/day, p 0.030, fails BH (bar 0.025, family of 4); candidate for a forward experiment only | **QUESTIONABLE** | Reproduced exactly. Fragile: half of it is 2022, 2020 goes the other way, and it fails the pre-registered correction. Quote with the caveat below. |
| 2 | Placebo ranks under the pre-registered pool (13.3) | **QUESTIONABLE** | The pool design leans toward "effect". The in-pool rank sits beside every cell; quote both. |
| 3 | Geopolitical shock days, shares (F3): 137 sessions, −0.334 σ | **QUESTIONABLE** | Descriptive by design, unstable by §2, mostly 2020, salience-selected list. |
| 4 | results.md:21-24, what moved with the fill-in: no Q1 or Q3 effect changed sign; three *tested* Q7 cells flipped within ±0.005 σ of zero (Crabel_Stretch and ORB_5min on jobs days, PDH_PDL_Breakout on Fed days, every p > 0.8); four untested Q7 cells flipped within ±0.002 σ; no tested/reported status changed | **PASS** | Matches my full comparison of `f739806` (1,226 sessions) against HEAD (1,249): every cell of q1, q1_single_category, q3, q3_single_category and q7, both arms. Exactly 3 tested flips (p 0.88, 0.83, 0.99) and exactly 4 untested ones, the largest EMA_9_20_Pullback\|energy_shock at +0.0009 → −0.0017 σ. Options: no flips. |
| 5 | §23.3 records the fourth pass | **PASS** | Accurate, including "two of them with no trades at all" (ThreeBarPlay on pandemic_health and cyber_infrastructure days, 0 trades each). |
| 6 | No regression since pass four | **PASS** | `git diff 59a61d0 2f04abc` touches only results.md, preregistration.md and this file. Code, analysis JSON and results_tables.md are unchanged. `pytest trade_analysis/news_arms trade_analysis/live_lab` all pass. No holdout analysis or join file exists. |
| 7 | Every other claim in results.md | **PASS** (carried) | Reproduced from raw on pass two. The fixes from passes three and four were verified then, and nothing numeric has changed since. |

## What was established across the five passes, in brief

- **Baselines:** shares 1,249 of 1,249 regular sessions, 226,678 trades, σ $3,199, B0 −$286/day
  (−1.6 bp per average trade). Options 482 sessions after the 270 no-0DTE exclusions, σ $1,032,
  B0 −$100/day.
- **Release days as a whole:** no difference from an ordinary day (shares −0.039 σ, CI includes
  zero; options −0.091 σ, CI includes zero).
- **Per setup (Q7):** no cell reaches the pre-registered quarter-sigma bar; the shortlist is empty.
- **Integrity:** the tested families equal the pre-committed counts by set equality. Regime uses
  prior sessions only. No conditioning variable comes from the session's own prices.
- **Lock:** it reads only the last non-empty line of this file, and its test asserts that.

## Caveats to repeat verbatim with the findings (not blocking)

- **Fed days:** "Exploratory. It fails the pre-registered multiple-testing correction. Without 2022
  the interval barely excludes zero (−0.156 σ [−0.304, −0.001]), and 2020 goes the other way
  (+0.18 σ against its own year). Four of five years are negative." Only a separately
  pre-registered forward test changes this; there is no Q7 shortlist row to carry it into the
  holdout.
- **Placebo:** "The pre-registered placebo favours 'effect' by construction. Read it beside the
  in-pool rank."
- **Shock days:** "Descriptive. Unstable by §2. Mostly 2020. Salience-selected list."

## Notes (not blocking)

- **Holdout exposure:** one holdout daily net (2024-10-01, +$2,592.69) was seen on the console during
  the fill-in. It is logged in §23. The 2023 options and 2023–2026 shares heartbeat exposures are
  logged in §19 and §21. None was aggregated.
- **Lock window:** the lock was open for minutes during pass three while my mistaken PASS draft
  stood. §23.2 records that nothing read the holdout then; no holdout output file exists.
- **What this PASS means:** the report now says only what its data shows. It does not make the
  three QUESTIONABLE findings any stronger than their caveats.

## Not run

- §18 live-vs-replay acceptance (Spearman ρ 0.932 shares, 0.953 options): quoted, not recomputed.
  It is outside gate 2.

GATE 2 VERDICT: PASS
