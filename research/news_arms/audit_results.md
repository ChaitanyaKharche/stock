# news_arms — gate 2 audit, third pass

Auditor, 2026-11-11, at commit `78c27b7`. Earlier passes: first (27 rows, FAIL) in `preregistration.md`
§22, second (FAIL on three sentences) in §23.1 and the git history of this file. The auditor
edited neither `results.md` nor the code.

**In one sentence: gate 2 fails once more, on one clause. results.md says "no effect changed sign"
after the fill-in, but three Q7 setup cells did, all by under 0.01 σ around zero.** The other two
corrected sentences check out against the data, both guard gaps are closed by tests that fail on
the old code, and nothing regressed. The fix is one clause, with no rerun. I passed this sentence
in the first draft of this report before checking the Q7 cells; that was my error, caught by the
check below.

## Verdict table

| # | Claim | Verdict | Reason |
|---|---|---|---|
| 0 | results.md:18-21: "no effect changed sign" after the fill-in | **FAIL (wording)** | True for every Q1/Q3 cell, false for three tested Q7 cells: Crabel_Stretch\|NFP −0.0045 → +0.0032 σ, ORB_5min\|NFP −0.0038 → +0.0043 σ, PDH_PDL_Breakout\|FOMC +0.0046 → −0.0009 σ (all p > 0.8, none BH). Compared `e2da37b` JSON with HEAD JSON. |
| 1 | Fed-day caveat (F2, shares 39 sessions, −0.299 σ [−0.54, −0.03], p 0.030, BH fail, −$958/day) | **QUESTIONABLE** | Unchanged from pass two. Fails the pre-registered correction; half of it is 2022; 2020 goes the other way. Candidate only, which is how results.md labels it. |
| 2 | Pre-registered placebo ranks | **QUESTIONABLE** | Biased toward "effect" by design (13.3). Quote only beside the in-pool rank, which is in every cell. |
| 3 | F3 shock days (shares −0.334 σ, 137 sessions) | **QUESTIONABLE** | Descriptive, unstable by §2, mostly 2020, salience-selected list. |
| 4 | results.md:137-138: "ORB_5min (72%) sixth of the 14 tested Fed-day cells" | **PASS** | Order by p: PDH_PDL_FailedBreak, VWAP_Reclaim, Crabel_Stretch, VWAP_2sigma_Fade, Six_Lines_NoCap, ORB_5min. 14 cells tested. |
| 5 | results.md:77-79: 39 kept Tue/Thu = 13 from 2022-11-15, 21 next to a weekday market holiday, 5 quarter ends | **PASS** | Recomputed from the replayed options sessions. The 5 are 2020-03-31, 2020-06-30, 2021-09-30, 2022-03-31, 2022-06-30. The first post-cut-over Tue/Thu is 2022-11-15. |
| 6 | results.md:18-21: the release-days placebo verdict went from "effect" to "chance"; no tested/reported status changed | **PASS** | Placebo 0.965 → 0.942, verdict changed as stated. Tested and descriptive flags are identical in every Q1/Q3 cell, and the families are set-equal (row 10). The sign clause in the same sentence is row 0. |
| 7 | `backfill.run`: a day with 0 < bars < `MIN_RTH_BARS` not on the early-close calendar raises `SessionLost` | **PASS** | backfill.py:404-406. The new assertion in `backfill_test.py` **fails on the `50ca53b` backfill.py** (run in a scratch copy) and passes at HEAD. |
| 8 | `holdout_allowed()` reads only the last non-empty line of audit_results.md; the test asserts lock state = that line, and that PASS followed by FAIL stays shut | **PASS** | analyze.py:657-658. Shut before this file was written (run). The new lock test **fails on the `50ca53b` analyze.py** and passes at HEAD. The test no longer breaks when the verdict line changes. |
| 9 | `exclusions()` counts a recorded "half_day" skip on a non-early-close day as lost | **PASS** | analyze.py:581-584. The updated test **fails on the `50ca53b` analyze.py**. |
| 10 | No regression: families, 0 lost, numbers, tests | **PASS** | Details in "Regression checks" below. |
| 11 | All other claims in results.md (pass-two rows 8-21) | **PASS** (carried) | Not re-derived this pass. Nothing numeric changed: a fresh rerun on HEAD equals the committed JSON. |

## FAIL detail and the fix

results.md:20-21 should read: "no Q1 or Q3 effect changed sign (three Q7 setup cells within
±0.005 σ of zero did: Crabel_Stretch and ORB_5min on jobs days, PDH_PDL_Breakout on Fed days)
and no tested/reported status changed". A fourth pass need only check that clause.

## Regression checks run this pass

- **Analysis rerun on HEAD:** both arms equal the committed JSON apart from `generated` /
  `command` / `repo_sha`.
- **Families:** Q7 66 / 20 and Q1 {CPI, FOMC, NFP, PCE} / {NFP, PCE} still equal
  `counts_*_discovery.json` by set equality.
- **Exclusions:** shares 1,249 sessions, 10 half days, 0 recorded skips, 0 lost. Options 482 sessions,
  4 half days, 270 `no_0dte`, 0 lost.
- **Holdout:** no 2023+ chunk in either discovery chunk list.
- **Tests:** `pytest trade_analysis/news_arms trade_analysis/live_lab` all pass.
- **§23.1:** records the second pass accurately.

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

- **`repo_sha`:** the committed JSON records `50ca53b`. HEAD changed only `exclusions()` and the
  lock, and a HEAD rerun is identical, so no number depends on this.
- **Contradictory skip records:** `exclusions()` still accepts a recorded "holiday" skip on a day
  the calendar says was open, as a recorded skip rather than a loss. `backfill.run` cannot produce
  that record (it writes "holiday" only when `is_closed`), so this is unreachable today. Fix if the
  skip format ever gains another writer: count it as lost.
- **Holdout exposure:** one holdout daily net (2024-10-01) was seen on the console. It is logged in
  §23 and results.md. Its effect on C1 cannot be measured.

## Not run

- §18 acceptance ρ: quoted, not recomputed. It is outside gate 2.
- No pass-two PASS row was re-derived from raw this pass. The basis for carrying them is the
  identical HEAD rerun.

GATE 2 VERDICT: FAIL
