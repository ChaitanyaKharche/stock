# news_arms — gate 2 audit, fourth pass

Auditor, 2026-11-11, at commit `59a61d0`. Earlier passes are recorded in `preregistration.md` §22,
§23.1, §23.2 and in the git history of this file. The auditor edited neither `results.md` nor the code.

**In one sentence: one word is missing, and the error is mine. The corrected clause copies the fix
text I wrote in pass three, "three Q7 setup cells … changed sign". I had checked only the *tested*
Q7 cells. Four untested cells also changed sign, so the count is wrong as written. Everything else
passes.** The fix is to insert the word "tested". No rerun is needed. I have now compared every cell
of every table, so this is the last wording item.

## Verdict table

| # | Claim | Verdict | Reason |
|---|---|---|---|
| 0 | results.md:21-23: "three Q7 setup cells within ±0.005 σ of zero did [change sign] … every p > 0.8" | **FAIL (wording)** | The three named are exactly the *tested* cells that flipped. Four untested cells flipped too: EMA_9_20_Pullback on energy_shock days (+0.0009 → −0.0017 σ, p 0.97 → 0.87, 21 sessions) and ThreeBarPlay on NFP, pandemic_health and cyber_infrastructure days (about ±0.00002 σ; two of those cells hold 0 trades, so the sign is just minus the setup's own B0). As an exhaustive count, "three" is false. |
| 1 | Same sentence: "no Q1 or Q3 effect changed sign … and no tested/reported status changed"; release-days placebo verdict "effect" → "chance"; "one more Q7 cell passes BH" | **PASS** | Checked pre-fill-in JSON at `f739806` (same corrected code, 1,226 sessions) against HEAD (1,249), and also `e2da37b` against HEAD. No sign change in any cell of q1, q1_single_category, q3 or q3_single_category, either arm. No change to tested / descriptive / never_tested / BH flags there. Q7 tested set unchanged. BH passes {PDH_PDL_FailedBreak\|FOMC} → {+ VWAP_Reclaim\|FOMC}. Every named p > 0.8 holds (0.882, 0.828, 0.987). |
| 2 | `exclusions()`: any recorded skip of an open full day other than `no_0dte` is lost | **PASS** | analyze.py:581-584. The updated test fails on the `78c27b7` analyze.py (scratch copy) and passes at HEAD. |
| 3 | No regression | **PASS** | Fresh rerun on HEAD equals the committed JSON apart from timestamp, command and sha. 0 lost in both arms (options 270 `no_0dte` recorded). Families set-equal to the counts (Q7 66 / 20; Q1 4 / 2). `pytest trade_analysis/news_arms trade_analysis/live_lab` all pass. Committed JSON changed only `repo_sha` and `generated`. |
| 4 | Lock | **PASS** | Last-line rule unchanged since pass three. Shut on this file. |
| 5 | §23.2 | **PASS** | Records the third pass accurately, including that my PASS draft briefly opened the lock. |
| 6 | Everything else in results.md | **PASS** (carried) | Passes two and three; nothing numeric moved. |
| 7 | Fed days (F2) | **QUESTIONABLE** | Caveat below. |
| 8 | Pre-registered placebo ranks | **QUESTIONABLE** | Caveat below. |
| 9 | Shock days (F3) | **QUESTIONABLE** | Caveat below. |

## FAIL detail and the fix

results.md:21: change "three Q7 setup cells within ±0.005 σ of zero did" to **"three tested Q7 setup
cells within ±0.005 σ of zero did"**. Optionally add "(four untested cells, all within ±0.002 σ, also
flipped)". Either makes the sentence true. A fifth pass need only check that word.

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

- **Holdout exposure:** one holdout daily net (2024-10-01) was seen on the console. It is logged in §23.
- **Lock window:** the lock was open for minutes during pass three while my PASS draft stood.
  §23.2 says nothing read the holdout. I can confirm that no `analysis_*_holdout.json` or holdout
  join file exists.

## Not run

- §18 acceptance ρ: quoted, not recomputed. It is outside gate 2.

GATE 2 VERDICT: FAIL
