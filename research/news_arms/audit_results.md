# news_arms — gate 2 audit, second pass

Auditor, 2026-11-11, at commit `50ca53b`. First pass (2026-11-10, 27 rows, FAIL) is recorded in
`preregistration.md` §22 and in git history of this file. Nothing in `results.md` or the code was
edited by the auditor.

**In one sentence: every number in the report now reproduces and every first-pass defect is fixed,
but three sentences in results.md are still false, so gate 2 fails again. The fix is three text
edits with no rerun.**

All the statistics pass: sessions, σ (one standard deviation of a day's net), B0 (the
all-sessions average day), every effect, CI (95% interval), placebo rank, block CI, dollar
conversion and per-year figure I re-derived from the raw replay files with my own loader. What
fails is wording:

| # | Claim | Verdict | Reason |
|---|---|---|---|
| 1 | "ORB_5min (72%) seventh of fifteen" by p-value on Fed days (results.md:134) | **FAIL (wording)** | After the fill-in it is **6th of 14** tested Fed-day cells (order: PDH_PDL_FailedBreak, VWAP_Reclaim, Crabel_Stretch, VWAP_2sigma_Fade, Six_Lines_NoCap, ORB_5min). "Seventh" was my first-pass figure on 38 sessions and went stale. |
| 2 | "39 Tuesdays and Thursdays that did have an expiry (quarter ends, and every one from 2022-11-15)" (results.md:75-76) | **FAIL (wording)** | 13 are from 2022-11-15 on. Of the 26 before, **21 sit next to a market holiday** (for example 2020-01-21 after MLK Day, 2020-04-09 before Good Friday) and only 5 are month or quarter ends. |
| 3 | "nothing changed sign or status" after the fill-in (results.md:18-19) | **FAIL (wording)** | The shares release-days placebo verdict changed from "effect" (0.965) to "chance" (0.942), as results.md:94 and §23 themselves say. The summary of what moved must not omit it. |
| 4 | F2 Fed days, shares: 39 sessions, −0.299 σ [−0.54, −0.03], p 0.030, BH fail (bar 0.025, family of 4), −$958/day, −$37,400 over five years; candidate only | **QUESTIONABLE** | Reproduced exactly. Without 2022: −0.156 σ [−0.304, −0.001]. 2020 is +0.18 σ against its own year. Fails the pre-registered correction. Labelled exploratory, which is correct. |
| 5 | Placebo ranks, pre-registered pool (13.3) | **QUESTIONABLE** | Reproduced. The design is biased toward "effect". The in-pool sensitivity now sits beside every rank and the text explains the bias correctly (results.md:88-94, 239-241). Quote only together with the in-pool rank. |
| 6 | F3 geopolitical surprise days, shares −0.334 σ, 137 sessions, unstable, 2020-driven, descriptive | **QUESTIONABLE** | Reproduced. The interpretation is now right. It remains salience-selected, descriptive and unstable by §2. |
| 7 | The new replay guard (`backfill.SessionLost`) "cannot lose a session silently any more" (results.md:20-23) | **QUESTIONABLE** | Zero bars or a failed warm-up raise. But a regular weekday served with *some* bars below `MIN_RTH_BARS` is filed as a "half_day" skip (backfill.py, `reason = "holiday" if n_bars == 0 else "half_day"`), and `analyze.exclusions` counts a recorded skip as not lost. Latent: no current chunk has a recorded skip, and the calendar check finds 0 gaps. |
| 8 | Holdout lock shut: whole line in audit_results.md | **PASS** | `holdout_allowed()` returns False and `main --set holdout` returns 2 (run). Tokens inside prose or in results.md do not open it. Two caveats in the detail below, neither blocking. |
| 9 | Shares 1,249 sessions / 226,678 trades / σ $3,199 / B0 −$286 (median −$682) / −$1.57 = −1.6 bp per trade; options 482 / 9,943 / $1,032 / −$100 / −$4.9 per contract | **PASS** | Independent loader: σ $3,198.66, B0 −$285.57, median −$682.31, −$1.574 per trade; options unchanged. |
| 10 | Fill-in: 23 discovery sessions back, 0 lost, not double-counted | **PASS** | 1,259 open weekdays − 10 rule half days = 1,249 replayed. `load_replay` raises on a duplicate (tested by passing fill-a twice). 0 duplicate trade keys. 0 original-chunk trades on fill-in dates. Same hash. Runner code unchanged between `c808dbd` and `e750a4c` (only backfill, autostart and calendar files differ). |
| 11 | NYSE calendar logic | **PASS** | Rule holidays, observed-day rule, Saturday New Year, Juneteenth from 2022, Bush and Carter closures, early closes. Its test checks the rule against every date in the feature file 2018–2025, both directions. |
| 12 | F1 release days: shares 234, −0.039 σ [−0.15, +0.08], block [−0.14, +0.09], −$123 [−$495, +$257], placebo 0.942 / in-pool 0.815, unstable; options 110, −0.091, −$94, 1.000 / 0.925 | **PASS** | All reproduced. My placebo seed gives 0.938 / 0.821 and 0.999 / 0.920. −0.7 bp per trade and 47 sessions a year check. |
| 13 | F2 detail: block CI [−0.56, −0.03]; single-category −0.273 (n 34); by year −0.29 / −0.09 / +0.18 / −0.40 / −0.86; 2022-11-02 −$3,248; win/day 28% vs 40%; −6.4 bp vs −0.5 bp (4× average); 195 vs 180 trades; options stops 41 vs 35%, RoP −5.4 vs −0.2%, twin −7.9 vs +2.1 bp; low-regime −0.21 [−0.41, −0.01]; stable by §2 | **PASS** | All reproduced from raw. |
| 14 | Fed timing: pre-14:00 −$1,145 vs −$272 per session, excess −$874; $8,900 / $6,800 / $7,500 a year; −$7.11 (6,283 trades) vs −$2.87 (1,331); "lost in the morning" | **PASS** | Reproduced: −$8,935, −$6,814, −$7,472 a year. Pre-14:00-entered trades realised −$50.6k before 14:00 and +$6.0k after, so the losses are not booked after the announcement. 09:30–10:30 carries −$646 of the −$958 excess. |
| 15 | F3 numbers: 137, −0.334 [−0.57, −0.12], block [−0.66, −0.07], −$1,069, 1.000 / 1.000; low +0.11, mid −0.26, high −0.67 [−1.08, −0.26]; without 2020 −0.35 [−0.78, +0.10] (35); by year −0.05 / +0.06 / −0.99 / −0.02 / −0.34; financial_stress 2020 −3.40; release-free variant 115 / 51, −0.381 / −0.106; options row; 37 in-session + 19 unknown break times | **PASS** | All reproduced. |
| 16 | F4 (NFP +0.100, 60 sessions, and the rest), F5 (66 / 20 cells; BH passes PDH −0.059 σ / −$189 and VWAP +0.049 σ / +$157; shortlist empty), F6 (−$319 vs −$155, −$164 [−$554, +$234]; options −$88), F7 (no divergence) | **PASS** | All reproduced. |
| 17 | §2 "unstable" rule, by-year and by-regime effects against each stratum's own B0 | **PASS** | `_unstable` and `_by` implement §2 as written. Flags match my hand computation for every cell quoted. |
| 18 | Tested families equal the pre-committed counts | **PASS** | Set equality on the post-fill-in run: Q7 66 / 20; Q1 {CPI, FOMC, NFP, PCE} / {NFP, PCE}. The counts files predate the fill-in (1,226 sessions); the 23 added sessions changed no cell's tested status. |
| 19 | No holdout file read by the discovery code path; 2024 fill-a not in discovery | **PASS** | Discovery chunk lists end at 2022 (50 shares, 36 options); the only fill-in listed is `2022/shares/fill-a`. One holdout daily net (2024-10-01) was seen on the console; it is logged in §23 and results.md:259-261. |
| 20 | No conditioning on the session's own prices (15.4); regime trailing-20, discovery terciles | **PASS** | Join rows (`build_join`) carry no price field, so no condition can read one. Regime re-verified in pass one. |
| 21 | JSON reproducible at its recorded sha | **PASS** | `repo_sha` `ce78ac4`. Code is identical to HEAD `50ca53b` (only JSON and tables differ). The test suite passes. |

## FAIL detail and the fix

Three sentence edits in results.md, no rerun needed:

- **:134**: "ORB_5min (72%) sixth of the fourteen tested Fed-day cells".
- **:75-76**: "39 Tuesdays and Thursdays that did have an expiry (21 next to a market holiday and 5
  month or quarter ends before the cut-over, 13 from 2022-11-15)".
- **:18-19**: add "the shares release-days placebo verdict moved from 'effect' to 'chance'" to the
  list of what moved, and drop "or status".

A third pass need only check these three lines and that no number moved.

## QUESTIONABLE detail: the caveats to repeat verbatim

- **Fed days (row 4):** "Exploratory. It fails the pre-registered multiple-testing correction.
  Without 2022 the interval barely excludes zero, and 2020 goes the other way. Four of five years
  are negative." This changes only with a separately pre-registered forward test. There is no Q7
  shortlist row to carry it into the holdout.
- **Placebo (row 5):** "The pre-registered placebo favours 'effect' by construction. Read it
  beside the in-pool rank."
- **Shock days (row 6):** "Descriptive. Unstable by §2. Mostly 2020. Salience-selected list."
- **Replay guard (row 7):** the fix is to raise `SessionLost` when `n_bars < MIN_RTH_BARS` and the
  calendar says the day is neither closed nor an early close. `exclusions` should also count a
  "half_day" skip on a non-early-close day as lost. This matters for any future replay, not for
  the current numbers.

## Lock caveats (row 8, not blocking)

- `analyze_test.py:174` asserts the lock is shut on the committed files, so it will **fail the
  moment this file carries a pass line**. Fix: assert that the lock state equals this file's final
  verdict line.
- `holdout_allowed()` opens on a pass line *anywhere* in this file, even if a later line says FAIL.
  Fix: read only the last non-empty line.

## Not run

- §18 acceptance ρ (results.md:99): quoted, not recomputed. It is outside gate 2.
- The direction of any bias from the console exposure of one holdout session: it cannot be measured.
  It is logged.
- Holdout coverage was checked by daily-file names only, no contents opened: no regular weekday
  missing in shares 2023-01-03 → 2026-08-27 or options 2023.

GATE 2 VERDICT: FAIL
