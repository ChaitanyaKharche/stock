# 06 - Research record: price levels, breakouts, and the trader's "six lines"

Slice: the studies built around the trader's own method (six horizontal price lines, buy a
naked 0DTE call or put when price breaks one) and the code behind them.
Written 2026-10-13 from a read-only pass. Nothing from the project was run. The only computation was a few read-only
PowerShell counts (not Python) over `live_lab_data/six_lines_QQQ.json` and
`live_lab_data/trades.jsonl`, to check that the documents' headline numbers match the
stored data. They are marked "my recount" where used.

All paths are relative to the repo root
`C:\Users\chaitanyakharche\Documents\stock\.claude\worktrees\blissful-kalam-999916`.
Every code claim carries `path:line`. **(inferred)** marks something I deduced and did not
read. "Reason not recorded" means I looked and found none.

Words used (defined once, here):
- **0DTE**: an option that expires the same day. **ATM**: at-the-money, strike nearest the
  share price. **Naked**: just the call or put, no spread or hedge.
- **bp (basis point)**: one hundredth of a percent. On a $700 share, 1 bp is 7 cents.
- **Close-based break**: a 10-minute bar closes beyond a line. A **touch** is only the wick.
- **MFE**: maximum favourable excursion, the best unrealised move in your favour during a
  trade. **Drift / terminal move**: where price actually ended (at exit).
- **Null**: the test found nothing distinguishable from chance. **CI**: 95% confidence
  interval, the range the true value probably lies in; if it contains 0 the effect is not
  shown. **p**: how often pure chance would give a result this extreme.
- **Pre-registration**: the question, the rules and the pass/fail test written to a file
  BEFORE any number is computed, so the result cannot be bent afterwards.
- **Day-clustered bootstrap**: re-run the average thousands of times, resampling whole days
  rather than single trades, because trades on one day are not independent.
- **Holm**: a correction that makes the bar stricter when many tests are run together.
- **VRP (variance risk premium)**: options are priced a bit richer than the movement that
  actually follows; whoever buys options pays that extra.

---

## 1. In one paragraph

This slice is the research record on one question: does the trader's chart method have an
edge? His method: draw six lines from yesterday's and today's premarket and regular-session
highs and lows, then buy a naked 0DTE call (price closes above a line) or put (below). Each
study is a pre-registration plus a results file, and the code that ran it. The answer, across
every version tried, is that **the break does not predict which way price ends the day**
(null); the one conditional signal that held up, an early break of the first resistance
reaching the second more often, is about how far price travels, not where it finishes; and
**every time a profit cap was measured on a mechanical entry it subtracted value**. The
six-line family is declared closed in `CLAUDE.md`. The same code is now also what the live
lab runs forward as two setups, `Six_Lines` and `Six_Lines_NoCap` (amendment dated
2026-09-25). What stays open is the part nobody has written down: which break he actually
takes, and a time-of-day pattern in his own P&L that is not yet a finding.

## 2. The goal it serves

1. **Original goal** (memory `profitbook-0dte-options-goal.md`, written 2026-07-27): copy the
   trader's manual 0DTE "retest rejection" breakout swings into code on the underlying, then
   build an options picker on top. The first plan was previous-3-days support/resistance
   breaks on 5-minute candles.
2. **First reality check** (`research/PROJECT_REPORT.md:42-48`): measured against his broker
   export, he is not an opening-range retest trader. Median entry 11:39 ET; the prior 15
   minutes moved 0.2029 sigma in his direction 89.3% of the time; only 5.7% of entries fall in
   09:30-09:45. "So the code was testing a strategy nobody was running."
3. **2026-09-17/18**: he describes the method in his own words (previous 3 days' highs/lows,
   premarket and market hours; ATM naked options; cap profit at 25-50% of premium; after
   1-2pm switch to next-day expiry) (`research/breakout_options_preregistration.md:10-24`).
   The pre-registration splits it into four claims that can fail independently:
   A the break predicts continuation; B the move is big enough to pay an ATM option; C the cap
   beats no cap; D the 1DTE switch helps.
4. **2026-09-20**: he corrects the level definition (section 3.1). From then the question is
   "does HIS exact method have an edge": null on 09-21, then dissected for conditional
   structure on 09-23, then added to the live lab on 09-24/25.
5. **Scale of the stakes**: his stated objective is $200-500/week
   (`research/entry_timing_preregistration.md:75-80`). The best mechanical result anywhere in
   this slice is +$4.93/week at one contract (`entry_timing_results.md:141`).

The goal shifted in one important way: early on it was "find a profitable rule", and by
2026-09-24 (`research/PROGRAMME_REVIEW_2026-09-24.md:330-345`) it had become "record what is
closed, and find the one unexamined part of his process", the unwritten selection rule.

---

## 3. How it works, end to end

### 3.1 Q1. The trader's method in plain words

**The six lines.** Each is the raw highest high or lowest low of 1-minute bars
(`trade_analysis/live_lab/six_lines.py:72-80`, `_hi_lo`, window `lo_t <= t < hi_t`):

| source | exact window | code |
|---|---|---|
| yesterday's premarket high and low | yesterday 04:00 to 09:29 ET | `six_lines.py:59, 89` |
| yesterday's market-hours high and low | yesterday 09:30 to 15:59 ET | `six_lines.py:60, 90` |
| today's premarket high and low | today 04:00 to 09:29 ET | `six_lines.py:59, 91` |

Three highs and three lows. "Yesterday" is the previous completed full session. In the
backtest that is the last session with at least 300 regular-hours 1-minute bars (a thinner
day is skipped, `six_lines.py:238-241`, and does not update `prev` at 259). The live loader
does the same, looking back up to 5 prior dates (`levels_live.py:35, 63-76`). Monday uses
Friday (inferred; weekends are not sessions). If any of the six cannot be built (no premarket prints, no yesterday premarket) the
function returns an empty list and the day is not traded; five lines "are not the trader's
six" (`six_lines.py:97-98`; `levels_live.py:22-24`; test
`six_lines_test.py::test_a_day_missing_premarket_yields_no_lines_rather_than_four`).

**R1/R2/R3/S1/S2/S3.** The three highs are sorted ascending: **R1 is the lowest of the three
highs, R3 the highest** (`six_lines.py:100-104`). The three lows sorted descending: **S1 is
the highest of the three lows, S3 the lowest** (`:101, 105`). R lines are "broken" by price
going up, S lines by going down. Caveat (inferred from those lines): R1 is "lowest of the
three highs", NOT "nearest above today's open". When the open is already above R1, R1 is
pre-broken. The research docs' phrase "R1 nearest above" is only true when R1 is live.
Tests pin the ordering (`six_lines_test.py::test_R1_is_the_nearest_resistance_and_S1_the_nearest_support`).

**Pre-broken.** A line the 09:30 open is already beyond (R below the open, S above it) is
flagged `pre_broken` and takes no part in any break count or trade
(`six_lines.py:115-139`). Over the 2,656 stored sessions R1 is pre-broken on 1,167 and S1 on
792 (`live_lab_data/six_lines_QQQ.json`, summary `per_line`).

**What counts as a break.** The 1-minute regular-hours bars are resampled into
**10-minute bars anchored at 09:30** (buckets 09:30, 09:40 ... 15:50;
`trade_analysis/live_lab/orb_veto.py:91-120`). A line is **close-broken** the first time a
10-minute bar CLOSES beyond it (`close > price` for R, `close < price` for S), with **no
buffer** (`six_lines.py:151-154`). **Touch-broken** is a wick (`:147-150`). Both are
recorded; trades use close-based. A bar's label is its START: a "09:40" break means the bar
09:40-09:49 closed beyond the line, known at 09:50. Each line records only its first break
(`:112-113`), otherwise "how many lines broke" would measure "how long the trend lasted".

**The trade** (`first_break_trade`, `six_lines.py:162-217`). One trade per session, on the
earliest-closing break (`min(broken)` on `(time, name)`, `:182`; ties go to the
alphabetically first name, so R1 before R2). R line means long (call), S line means short
(put). Entry price is the **open of the next 10-minute bar** (`:191`). Exits, checked bar by
bar:
1. **Profit cap**: +20 bp from entry, an intrabar limit (`CAP_BP = 20.0`, `:158`, "about +50%
   on an ATM option at delta 0.5"; reasoning in `breakout_sweep.py:18-23`).
2. **Close-back-inside stop** (the "failed" exit): a 10-minute bar CLOSES back on the wrong
   side of the broken line, checked only from the bar AFTER the entry bar (`:204`,
   `b["ts"] > ten[sig_i+1]["ts"]`). The loop `break`s here (`:205-206`).
3. **End of day**: the last 10-minute bar's close (`:207-208`).

**What "retest" means (not part of the six-line method).** In the older ProfitBook strategy:
break a level, pull back INTO a narrow zone around it (+/-0.15 ATR), then a rejection candle
in the breakout direction on above-average volume
(`trade_analysis/backtesting/retest_vs_gapandgo_backtest.py:1-30, 117-165`). That same file
found the older "retest confirmation" code was a tautology that filtered nothing (206 of 255
entries fired on the very first 09:30 bar) and that the code actually ran as a gap-and-go
(`:6-14`). He believes he trades retests; measured, he does not (`PROJECT_REPORT.md:42`).
The six-line family never uses a retest: it enters on the first close beyond a line. The
numbers from that retest-vs-gap-and-go run are not in any research doc I read, and its
output folder is gitignored (`.gitignore:57`).

**The earlier 8-line set and why it was the wrong level set**
(`research/breakout_options_results.md:11-25`; `breakout_levels.py:1-20, 91-118`). The first
pre-registration used the previous THREE completed sessions, each reduced to ONE high and ONE
low over the whole extended session 04:00-20:00 (6 lines), plus today's premarket high/low
(2) = 8. The trader then specified his real set: ONE prior day, with its premarket and
market-hours extremes kept as SEPARATE lines, plus today's premarket = 6. Merging premarket
into market hours deletes yesterday's premarket lines "whenever they sit inside yesterday's
RTH range, which is most days", so the 8-line null "is a null about levels he does not
draw". Other differences in the 8-line code: a 0.05% buffer beyond the line
(`breakout_levels.py:50, 149-150`), a 09:40-15:30 signal window (`:53-54`), a signal on
EVERY level cleared once each, 3.4 per day (8,843 signals / 2,600 sessions), not one per day.
Note the pre-registration says "nearest un-breached level" (`breakout_options_preregistration.md:77`)
but `find_breakouts` loops over all levels (`breakout_levels.py:146-160`); "nearest" exists only
in `nearest_unbroken`, "for reporting, not for the signal" (`:164-176`).

An even earlier formulation exists: `trade_analysis/backtesting/multi_level_orb_backtest.py`
(the "ACTUAL manual strategy": three pairs of levels, the effective line is the max of the 3
highs / min of the 3 lows, i.e. all three must be cleared; plus volume > 20-EMA, MACD 9/17/9 on
OHLC4, DMI/ADX, 5-minute execution with 10/15-minute agreement, entry window 09:30-09:45;
`:1-45, 60-75`). Its result numbers are in gitignored logs, not in a research doc I read.

### 3.2 Q7. How each backtest avoids lookahead (which close is known when)

The project's founding trauma: a 5-minute resample labelled by bar START meant a signal that
was only knowable at bar END was filled five minutes early, which handed a 25% profit target
for free and produced the whole "edge" (memory `0dte-real-data-findings.md`: QQQ +6.44% as
published, -8.34% honest; 78% of trades received the five minutes free). The rules that
descend from it, with where each is enforced:

| risk | how it is closed | where |
|---|---|---|
| Today's lines use information not yet known | Only yesterday's completed session plus today's premarket (complete at 09:30); line building never reads today's regular-hours bars | `six_lines.py:83-106, 241-259`; `breakout_levels.py:28-39` (three named lookahead surfaces) |
| Pre-broken test uses the future | compared with the 09:30 OPEN only | `six_lines.py:135-139` |
| Signal known too early | a break is a 10-minute bar CLOSE; bar label is its START, so it is knowable one bucket later | `six_lines.py:151-154`; `orb_veto.py:91-120` |
| Entry filled at the signal bar's price | entry = OPEN of the NEXT 10-minute bar; signals with no next bar are dropped | `six_lines.py:187-191`; `breakout_levels.py:143-144, 157`; `breakout_sweep.py:88-91`; `early_r1_hold.py:49-50, 64-68` |
| Stop fires on its own entry bar | stop checked only from the bar after the entry bar | `six_lines.py:204`; `early_r1_hold.py:72-77` |
| Cap fills AFTER the position already exited | the loop `break`s unconditionally on "failed"; the earlier version carried on and registered a cap fill later, inflating +0.27 to +4.98 bp | `six_lines.py:205-206`; `breakout_sweep.py:119-130` (comment explains); tests `six_lines_test.py::test_the_cap_cannot_fill_after_the_trade_exited`, `breakout_levels_test.py::test_the_cap_cannot_fill_after_the_position_has_already_exited` |
| Cap and stop in the same bar | cap wins ("a limit order fills intrabar and a close is only known at the bar's end"); stated as a favourable ordering | `breakout_sweep.py:83-86`; `six_lines.py:173-175` |
| Same-day outcome used to gate an earlier trade (veto) | `decide` is evaluated only from bars whose CLOSE is known by T; the whole-session `post` flag is attribution only; only trades entered AFTER T enter the contrast | `orb_veto.py:19-28, 185-189`; `orb_veto_preregistration.md:31-48, 92-108` |
| Benchmark arm selected with future info | arm C is "R1 live and no break by 09:40", known at the 09:50 open, which is also its entry bar | `early_r1_hold.py:89-97`; test `early_r1_hold_test.py::test_a_late_break_is_a_benchmark_row_not_a_treatment_row` |
| Session-end time | the code exits at the last 10-minute bar's close = 16:00; it never uses the docstring's "15:55" (`EOD` branch in `breakout_sweep.py:103` can never fire on a 15:50 bucket, inferred) | `six_lines.py:207-208`; `early_r1_hold.py:18-19`; `early_r1_hold_preregistration.md:40-46` |
| Entry-timing placebo uses a strike chosen with hindsight | VOIDED on 2026-08-26; strike was ATM at HIS entry second, so any earlier placebo second held a contract picked from the future | `entry_timing_preregistration.md:149-183`; `entry_timing_results.md:28-113` |
| Live setup sees future bars in a batch | buffer truncated to `ts <= ctx.bar_ts`; only fires when the 1-minute bar completes a 10-minute bar | `setups.py:653-664, 700`; test `six_lines_setup_test.py::test_a_batch_buffer_cannot_leak_future_bars` |

What the underlying-only studies do NOT include: any price paid. Entry is a bar open, exit a
bar close or the +20 bp limit; no spread, no theta, no commission. The 5 bp "ATM 0DTE cost
floor" (`six_lines.py:159`, from `shares_runner.py`) is applied by comparison, not
subtracted. That is why each results file says "nothing about options".

Statistics used: day-clustered or session bootstrap, 3,000 reps. Seeds: 20260828
(`six_lines.py:329`, `breakout_sweep.py::boot_mean`), 20260923 (`early_r1_hold.py:37`). One
trade per session, so for the six-line and early-R1 studies sessions are already the unit.

### 3.3 Q2. The studies

(numbers copied from the named files)

| # | study (files) | question in plain words | sample | pre-registered rule | result | verdict, what it closed |
|---|---|---|---|---|---|---|
| 1 | 8-line breakout (`breakout_options_preregistration.md`, `breakout_options_results.md`; code `breakout_levels.py`, `breakout_sweep.py`) | Does a 10-minute close beyond one of 8 levels keep going? Does a +20 bp cap (about +50% option gain) help? | QQQ underlying, 2016-01-04 to 2026-09-17; 2,794 days tried, 2,657 usable, 137 thin; 8,843 signals over 2,600 sessions; 7,938 before 13:00 | A survives only if the day-clustered CI on mean move excludes zero on BOTH QQQ and SPY (prereg sec 8); C survives only if capped beats uncapped in dollars | uncapped mean +1.44 bp (median -13.79) CI [-0.97,+3.91] p=0.259; capped +0.27 bp CI [-0.88,+1.46] p=0.640; cap worth -1.17 bp; 64.8% touched the cap first; 37.1% of uncapped signals clear 5 bp | A fails, C fails. Wrong level set (results sec 0). SPY never completed. D (1DTE) permanently unmeasurable: no QQQ option archive, archive is 0DTE only, ThetaData `Options: FREE` since 2026-09-08 |
| 2 | Six lines, his exact set (`six_lines_results.md`; code `six_lines.py`) | Same question with the real six lines, one trade per session on the first break | QQQ, 2016-01-04 to 2026-09-19 run window; 2,795 days tried, 2,656 sessions stored (2016-01-05 to 2026-08-27), 138 thin; 2,438 trades, 218 no-trade (8.2%) | No separate pre-registration file exists (the six-line variant was only listed as secondary S1 in the 8-line prereg, which defined S1 as prior-3-days-only, a different thing). Rule used: CI vs zero and the 5 bp floor. Reason no prereg was written is not recorded | uncapped +0.54 bp (median -13.94) CI [-1.70,+2.87] p=0.611; capped -0.18 bp CI [-1.29,+0.88] p=0.764; cap cost 0.72 bp; both means under 5 bp; 92.0% of sessions break a line; first break median 09:40; 56% long, long +2.30 bp, short -1.74 bp | NULL. Closes the close-based six-line break as a directional entry on QQQ, capped or not. I re-counted the stored JSON: 2,438 trades, mean +0.54, capped -0.182, median -13.94, cap hit 55.1% (matches) |
| 3 | After the break (`after_break_preregistration.md`, `after_break_results.md`; `after_break.py`) | After R1 (S1) breaks in the first 20 minutes (09:30 or 09:40 bar), does price run to R2 (S2)? Do R1/S1 sit on round numbers? | QQQ, 2,656 sessions; R side 317 early breaks, 509 late | Early must beat late and a time-matched control; Wilson CIs, two-proportion test; one pre-specified near/far-from-multiple-of-25 split | R side: reached R2 70.7% (224/317) vs 59.3% (302/509), +11.3 pp, p=0.001 (I re-counted both from the stored JSON); time-matched (R2 breaks within 60 min of R1, breaks by 14:50): 49.2% (156/317) vs 37.9% (181/478), +11.3 pp, p=0.002. S side: 66.8 vs 62.4%, time-matched 48.3 vs 44.1%, p=0.242. Round numbers, ratio to chance: 1.02 / 0.96 / 0.99 / 1.05 for multiples of 1/5/10/25, same in both price eras | R side real as a description of travel; S side null; round numbers null. One metric (`full_reversal`) found defective and flagged |
| 4 | Remove the stop (`early_r1_hold_preregistration.md`, `early_r1_hold_results.md`; `early_r1_hold.py`) | If early R1 breaks are held to the close with no stop, does that turn the travel into money? | QQQ, 2016-01-04 to 2026-09-19; 317 early-break sessions, 1,172 benchmark sessions (1,489 usable, 138 thin) | Survives only if (B-C) > 5 bp AND CI excludes 0 AND sign agrees in 2016-2020 and 2021-2026 | A (with stop) -0.14 bp CI [-6.84,+6.98]; B (no stop) +1.69 bp CI [-10.81,+14.65]; C (benchmark) +4.25 bp CI [-2.10,+10.43]; B-A +1.84 bp CI [-13.02,+16.81] p=0.829; B-C -2.56 bp CI [-17.01,+11.76] p=0.709 (wrong sign); halves +4.87 then -9.97 | FALSIFIED on all 3 gates. Closes "remove the stop"; closes the whole six-line family |
| 5 | His entry timing (`entry_timing_preregistration.md`, `entry_timing_results.md`) | Is HIS entry second better than 20 random seconds on the same ATM contract? Does a mechanical fixed hold on his entries make money? | 339 evaluated of 357 QQQ/SPY round trips, 2025-05-01 to 2026-02-20 (calls 181, puts 158; QQQ 269, SPY 70) | Holm over family m=8 (C-skill x4 horizons 5/10/15/25 min, C-money x4) | C-skill VOIDED (placebo contract chosen with hindsight; raw run showed +27.12% placebo return). Clean post-entry-only control: null at all 4 horizons (exploratory). C-money: 0 of 4 clear; best (10 min) +$0.61/trade = +$4.93/week at 1 contract. His actual tape: +$59.52/week QQQ/SPY, +$85.59/week QQQ, positive at every cut, significant at none | His entry timing shows no measurable skill; any value is in exits. Needs roughly 356 more QQQ days for t=2 |
| 6 | Range-expansion veto (`orb_veto_preregistration.md`; `orb_veto.py`, `orb_veto_backtest.py`) | Would skipping new entries after 11:00 on days with no 10-minute close outside the premarket/opening-range band improve the 13 frozen shares setups? | QQQ shares, about 2,650 sessions from `sharewf_trades.json` | Veto real if CI on mean D(veto) minus D(no veto) excludes zero and is negative, AND filtered total better, survives dropping the best day, sign in 8 of 11 years, tail check (sec 6) | **No result in the record I could see**: no results md; `orb_veto_results.json` (where the script writes, `orb_veto_backtest.py` `OUT`) is not in `live_lab_data/`; no commit message reports one; it is not listed as closed in `CLAUDE.md` | Built, tested (26 tests across `orb_veto_test.py` and `orb_veto_backtest_test.py`), not shown as run. Status unknown |
| 7 | Conditional short straddle (`conditional_0dte_preregistration.md`, `..._armA_results.md`, `..._armB_results.md`) | Is the short 0DTE straddle premium harvestable on a pre-selected subset? Not a levels study; the brief lists it | Arm A: Vilkov SPXW panel, 1,147 usable of 1,397 sessions, 2016-09 to 2024-05, 13,673 trades. Arm B: 2023 only, 73 sessions, 1,113 selected origins of 14,931 | Arm B gates: net mean > 0, session-clustered t > 2.0, paired vs unconditional p<0.05, >= 50 sessions, >= 1,000 trades; 2024 sealed | Arm A failed to reproduce the published rule (my threshold theta=0 made it "buy almost always": bias 2.8% of mid vs 0.6% real premium, 4.6x the signal). Arm B cleared all 5 gates (+$0.01002/trade, t=+2.05, paired t=+3.16, 69.9% of sessions profitable) but is FRAGILE (t falls to 1.83 on dropping the best session, worst 0.1% of trades = 16x the median win, calm year) | The pre-registered "A reproduces AND B passes" branch was not reached because A never reproduced |
| 8 | Live backfill (`six_lines_backfill_results.md`) | What would `Six_Lines` and `Six_Lines_NoCap` have done on the lab's first 19 sessions? | 2026-08-28 to 2026-09-24, QQQ+SPY, real runner code, 26 trades per variant | None. Declared "a BACKFILL, not evidence" | Six_Lines -$156.10 (9 wins of 26; 16 failed, 9 target, 1 eod); Six_Lines_NoCap +$1,533.90 (7 of 26; 17 failed, 9 eod); NoCap's best trade +$1,040.92 is 68% of its total (2026-09-21); median trade about -$36 in both | Shows what the rule does on one trend day. No edge claim; never counted toward any checkpoint. Signals identical to `first_break_trade` on 38 of 38 sessions |

Supporting survey, `research/vendor_and_method_surveys/0dte_expression_survey.md`
(compiled 2026-09-14): the delta-matched break-even hit rate for a ~30-minute ATM 0DTE hold
is 56.9% versus 50.3% for shares (derived by a research pass, "not quoted from a paper",
sec 1). It argues against verticals and against selling premium, and says the three central
academic sources are working papers, not peer reviewed (header). Its sec 9 lists claims not to
build on without re-sourcing.

### 3.4 Q9. How `Six_Lines` and `Six_Lines_NoCap` entered the live lab (amendment 2026-09-25)

**Why they were absent before**: both live runners were regular-hours only and never read a
premarket bar (`levels_live.py:3-5`; `research/forward_test_preregistration.md:154-157`).

**What the amendment says** (`research/forward_test_preregistration.md:143-186`, "Appended,
never substituted. Written before either new setup has traded a live session."):
- Two setups appended to the end of `ALL_SETUPS`: `Six_Lines` (the trader's set, first
  10-minute close beyond a live line, one trade per session, exits +20 bp | 10-minute close
  back inside the line | 15:55) and `Six_Lines_NoCap` (same entry, cap removed). `setups.py:684-744, 752-757`.
- Why now: the only strategy in the programme that is the trader's, not a published one.
- Why an uncapped twin: the cap subtracted value everywhere it was measured (-0.72 bp on this
  exact set); the only way to see it on live fills is to run both.
- Unchanged: the 13 original definitions are byte-identical to hash `1f7247d7839d9950`
  (tested in `six_lines_setup_test.py` lines 265-273); counters continue across the hash
  change; new hash `ccd00ac71f8243e9` added to `live_lab_data/FREEZE.json` (accepted hashes);
  the two new setups start at zero on 2026-09-25; family size 13 to 15 (it only grows, so the
  Holm bar only rises).
- A stated prior before any live trade: expected result is a null; "the one outcome that would
  be new is the uncapped twin clearing the cost floor on live fills".
- Backfill over 2026-08-28 to 2026-09-24 kept in `live_lab_backfill/`, outside
  `live_lab_data/`, never counted toward any checkpoint.

**How it is built so live equals research**:
- `levels_live.py`: at warmup fetches yesterday's extended session; at the first tick at or
  after 09:31 fetches today's premarket and calls the research `six_lines.build_six` itself,
  not a copy (`:21-23, 63-102`). 09:31 not 09:30 because the 09:29 bar is published a moment
  after 09:30:00 (`:17-20`). If lines cannot be built: `sess.levels = ()`, recorded as an
  outage, no trade (`:89-94`).
- `SixLines.evaluate` (`setups.py:696-717`): fires only when a 10-minute bar has just
  completed and that bar holds the session's FIRST close-break of a live line; "a missed
  first break is not replaced by a later one" (`:633-636`, the Crabel_Stretch lesson from
  `incident_2026-09-24_opening_window.md`). `manage` returns "failed" on a completed 10-minute
  bar after the entry bar closing back inside (`:719-735`).
- Stated deviations: the target is fixed from the SIGNAL bar's close, not the next bar's
  open, because the lab must fix it when it decides (`:641-643`); the live flatten is 15:55
  (`runner.py:43`) while research exits at 16:00; no buffer.
- Parity tests: 400 synthetic days against `first_break_trade` itself
  (`six_lines_setup_test.py:75-96`) plus future-bar, no-second-chance, partial-lines and
  freeze-hash tests.
- Hashes: `runner.build_config(["QQQ","SPY"],1500)` must equal `ccd00ac71f8243e9` (test at
  `six_lines_setup_test.py:277-281`).

### 3.5 Q8. The day-charts tool

`trade_analysis/live_lab/day_charts.py` (built 2026-10-08, commit 2156955, "at the user's
request" per memory `day-charts-tool.md`) draws, for each session the lab ran and for QQQ and
SPY, a dark Robinhood-style chart at 2, 5, 10 and 15 minutes, 07:00-16:00 ET with premarket
shaded, the six lines from 09:30, EMA 9/20, VWAP, volume, MACD 12/26/9 and RSI 14
(`day_charts.py:1-45`). What it is for:
- **Looking at the days rather than the numbers.** The lines are the ones the runner logged at
  09:31 (events.jsonl `levels`, from 2026-09-25) or rebuilt by `six_lines.build_six`. If both
  exist they are compared and any mismatch is printed on the chart ("vendor revised since
  09:31"), because "a number that changes when nothing changed is a bug until shown otherwise"
  (`:23-28`, `:558-559`).
- **Each line is labelled with the minute that set it** (`set_at`, `:219-232`), so a line
  resting on one stray premarket print is visible as such. That is how the stray-print issue
  (section 8, open item 3) was found.
- **Matching what the trader sees.** MACD uses 12/26/9 because that is the setting on his own
  Robinhood chart; on 2026-10-08 a 12:10 bar read +0.02 there and 12/26/9 reproduces it while
  the repo's 9/17/9 does not (`:19-24`).
How the owner uses it (memory `day-charts-tool.md`; the doc says nothing more): run from the
gitignored `venv_charts` interpreter (the lab's interpreter must not get matplotlib); output
goes to `live_lab_charts/<day>/<SYM>_<NN>m.png` plus `index.html` and a per-day
`run_summary.md`, all gitignored; `--offline` draws from the bar cache; a network run is
refused 09:00-16:05 ET on weekdays so it cannot compete with the runners for the data
terminal unless `--force` (`:612-615`). Whether he reviews it daily is **reason not recorded**.

---

## 4. Map

Research documents (`research/`):
- `breakout_options_preregistration.md` / `_results.md`: the 8-line study, claims A-D, correction sec 0.
- `six_lines_results.md`: his exact set, the null, the break-rate-by-distance table.
- `six_lines_backfill_results.md`: replay of the two live setups over the first 19 sessions.
- `after_break_preregistration.md` / `_results.md`: early vs late R1 break, round numbers, S side.
- `early_r1_hold_preregistration.md` / `_results.md`: remove the stop; arms A/B/C; falsified.
- `orb_veto_preregistration.md`: the day-level veto; no results doc found.
- `conditional_0dte_preregistration.md`, `_armA_results.md`, `_armB_results.md`: conditional short straddle.
- `entry_timing_preregistration.md` / `_results.md`: his entry vs random seconds; includes the voided control.
- `vendor_and_method_surveys/0dte_expression_survey.md`: why the options arm loses; what not to do.
- Related, read for cross-checks: `construction_results.md` (strike and exit tests),
  `journal_reverse_engineering_results.md` (422-trade exit study), `PROJECT_REPORT.md`,
  `PROGRAMME_REVIEW_2026-09-24.md`, `forward_test_preregistration.md` (amendment).

Code (`trade_analysis/live_lab/` unless stated):
- `six_lines.py` (400 lines): builds the six lines, tallies breaks, runs the one-trade-per-session sweep and writes `live_lab_data/six_lines_<SYM>.json`.
- `breakout_levels.py` (188): the 8-line level builder and signal finder (kept as the record of the wrong set).
- `breakout_sweep.py` (382): runs the 8-line study; `evaluate_signal` holds the cap/stop logic with the lookahead fix commented in place.
- `after_break.py` (280): offline analysis of the stored six-line JSON; early vs late; round numbers.
- `early_r1_hold.py` (280): arms A/B/C; needs the live feed.
- `levels_live.py` (102): builds the six lines live for the runners.
- `orb_veto.py` (233) and `orb_veto_backtest.py` (337): the veto classifier and its contrast; `resample_10m` lives in `orb_veto.py:91` and is imported by every other file here.
- `day_charts.py` (654): per-day chart tool.
- `levels_test.py` (not in scope) supplies `get_ext`, the bar fetch/cache that every study uses (`levels_test.py:36-70`; it drops any bar with a non-positive price at `:60`, the zero-fill guard).
- `setups.py:619-757`: the live `SixLines` and `SixLinesNoCap`.
- `trade_analysis/backtesting/entry_histogram.py` (235): descriptive entry-time histogram of his journal; no p-values.
- `multi_level_orb_backtest.py` (372), `multi_level_option_pricing.py`: the earlier 3-pair confluence formulation and its NBBO pricing.
- `retest_vs_gapandgo_backtest.py` (301): real retest vs the gap-and-go the old code actually ran.
- `options_premium_backtest.py` (342): the original synthetic Black-Scholes premium backtest of yesterday's-high/low breakouts; its own header says synthetic premium is weak evidence (`:1-25`).

Tests (names and docstrings read): `six_lines_test.py`, `six_lines_setup_test.py`,
`breakout_levels_test.py`, `after_break_test.py`, `early_r1_hold_test.py`,
`orb_veto_test.py`, `orb_veto_backtest_test.py`, `day_charts_test.py`,
`trade_analysis/backtesting/entry_histogram_test.py`.

Data: `live_lab_data/six_lines_QQQ.json` (4.1 MB, 2,656 sessions, per-line `price`,
`source`, `pre_broken`, `close_break`, `touch_break`, and the day's `trade`),
`breakout_sweep_QQQ.json` (8-line signals), `early_r1_hold_QQQ.json`. No SPY file exists for
any of them (directory listing).

---

## 5. Design decisions and the reason for each, with the findings that drove them

**Close-based, not touch-based.** On 2026-09-17 QQQ printed ~718 on the session's largest
volume bar and closed back inside a 715.0-716.7 band; a touch rule calls that a breakout, a
close rule calls it the failed breakout it was. "A rule a single tick can flip is not a
rule" (`orb_veto.py:30-35`; `breakout_levels.py:20-27`; `six_lines.py:35-42`). It also
fixes the sign of the error: a close rule can be late but not fooled, and being fooled is
the expensive direction for a naked long option (`orb_veto_preregistration.md:50-62`).

**10-minute bars aligned to 09:30.** "The timeframe he actually reads"
(`orb_veto_preregistration.md:72`). Aligned to the open so 09:30-09:39 is always one bucket
(`orb_veto.py:91-102`). Consequence: "first 15 minutes" cannot be expressed; the studies use
the first two bars (09:30 and 09:40 = 20 minutes), recorded openly
(`after_break_preregistration.md:14-17`).

**First break only, one trade per session.** Because 92.0% of sessions break something and the
median first break is 09:40, "take every break" is "the weather, not a strategy"
(`six_lines.py:166-171`; `six_lines_results.md:56-60`). (The comment's journal comparison is
withdrawn, see section 9.)

**+20 bp as the stand-in for "+50% on an ATM option".** At delta ~0.5 the premium P on spot S
is about 20 bp for a 0DTE ATM on these names, so a +50% option gain needs about 20 bp of
underlying; "theta only makes that worse" (`breakout_sweep.py:18-23`). It is a proxy: no
option price appears in any underlying study.

**+25% not the primary cap in the first prereg, +50% instead.** A +25% target "is a
pre-registered expected LOSS, not a candidate", from the journal work (win rate 71.6% vs
break-even 75.3%) (`breakout_options_preregistration.md:95-112`).

**Exit "close back inside" was chosen by the researcher, not stated by the trader**: "he did
not state a stop. It is the exit most faithful to the logic: the thesis was the break, so the
thesis dying is the exit" (`breakout_options_preregistration.md:96-100`, marked [R]).

**Dollars, not returns, as the headline for the cap**: the journal work found the sign flips
between return and dollar space (`breakout_options_preregistration.md:149-152`). See the
contradiction in section 9.

**A control arm for everything.** The early-vs-late comparison, the 60-minute time-matched
window, and benchmark arm C exist for one reason each: an early break has six hours left to
reach R2 and a 15:30 break has twenty minutes; and QQQ rose from ~$110 to ~$740, so any
rule that buys and holds earns drift (`after_break.py:175-182`; `early_r1_hold.py:14-16`).

**Separate files for experiments.** Owner's explicit instruction 2026-08-02: new variants go
in new files that import proven code and can be deleted without risk (memory
`feedback-isolate-experimental-strategy-code.md`). Visible in `multi_level_*`,
`options_premium_backtest_*`, `retest_vs_gapandgo_backtest.py` ("ISOLATION: imports only,
edits nothing... Delete freely", `:46`).

**Preference for the simple rule**: three attempts to add sophistication (ATR targets,
multi-day confluence, GEX regime, scalp exits) all backtested no better than plain "ORB
confirms, naked option, fixed target" on a small sample (memory
`feedback-prefer-simple-orb-strategy.md`). Note that memory predates and is partly at odds
with the later finding that the fixed cap subtracts value.

### Q3. "Magnitude, not sign", for someone who is not a statistician

Think of the six lines as a weather warning that says "wind will be strong today". It tells
you something real about HOW MUCH, nothing about WHICH WAY. The four pieces of evidence:

| # | evidence | plain words | where |
|---|---|---|---|
| 1 | The old momentum score was a size measure forced to output CALL or PUT | The part of the system that measured something real was measuring size | `research/vrp_preregistration.md` sec 0, quoted second-hand in `breakout_options_results.md:101-106` (I did not read that file) |
| 2 | 8-line: median MFE +34.26 bp, share reaching 20 bp 64.8%, terminal drift +1.44 bp with a CI spanning zero | Price goes there and then comes back; ends nowhere in particular | `breakout_options_results.md:89-100` |
| 3 | Six-line, 2,438 trades: median end move -13.94 bp but the best move reached during the trade was far higher (median MFE 25.09 bp, my recount of the stored JSON); the median day's whole range is 125 bp (docs) / 124.8 (my recount) | A typical trade ends about 14 bp under where it started but was up about 25 bp at its best | `six_lines_results.md:52-73`; stored JSON |
| 4 | Early R1 break reaches R2 70.7% of the time vs 59.3%, but held to the close it is -2.56 bp vs the benchmark | The path is more likely to get there; the closing price is not | `early_r1_hold_results.md:64-85` |

What it means for someone who buys calls and puts: you can be right about the travel and
still lose. A call or put is a bet on direction wrapped around a bet on size; you pay a
premium for the size (the VRP, the extra price options carry over the movement that follows),
and the signal gives you no direction to be paid for. An ATM 0DTE needs about 5 bp of
underlying move to cover spread and theta, and both mean results are under 5 bp
(`six_lines_results.md:33-35`). The survey puts the break-even hit rate for an ATM 0DTE held
about 30 minutes at 56.9% (derived, `0dte_expression_survey.md:16-27`).
The project's own wording of the logic: "you can be right about the travel, pay the variance
risk premium for the convexity, and collect nothing for the direction"
(`early_r1_hold_results.md:83-85`; `CLAUDE.md`). Two honest qualifications the files make:
(a) the pre-registration's "B survives only if A survives" was too strong, because a long
option's loss is capped at the premium, so a convex payoff can be positive on a zero-drift
underlying; the null on A removes the reason to expect B, it does not logically kill it
(`breakout_options_results.md:64-87`); (b) the options-expression arm remains unmeasurable
beyond SPY 0DTE. Per `CLAUDE.md`, do not answer this with a different structure; the owner
trades naked calls and puts and wants to be told when the data says they will not work.

### Q4. "The profit cap subtracts value": three measurements, in plain words and money

A profit cap sells the winner early. A loss is not capped, so it removes the big winners that
pay for the losers. The three measurements:

1. **Journal (his real trades).** Two different runs, which point opposite ways (see section 9):
   - `research/construction_results.md:52-82`: on 296 QQQ/SPY round trips (89 days, 2025-05-13
     to 2025-11-26), a fixed +25% target lifts the win rate from 48.3% to 71.6% and the total
     goes to **-$2,347.76 against his actual +$3,458.34** at his sizing. Mean -3.89% vs +5.08%;
     gap -8.97 percentage points, Holm p=0.027, the only one of five comparisons to clear.
     A +25%-win/-100%-loss structure breaks even at 75.3% wins; 71.6% is below it.
     His own structure: average win +48.1%, average loss -37.0%, breaks even at 43.5%, he runs 45.8%.
   - `journal_reverse_engineering_results.md:196-238`: on 422 reconstructed trades in DOLLAR
     space the same +25% target simulates **+$5.00/trade vs his exit's +$0.78, difference
     +$4.22, CI [-7.98,+16.33], p=0.487**; "the sign flips with the weighting and neither version
     is robust. Test B should be treated as fragile".
2. **8-line mechanical, 7,938 signals (QQQ, 2016-2026), in underlying basis points**: cap worth
   **-1.17 bp** per signal (uncapped +1.44 -> capped +0.27). On a $700 share 1 bp is 7 cents,
   so about 8 cents per share (my arithmetic). Cause: the uncapped series is two-tailed
   (worst -628.4 bp, best +479.0 bp); the cap flattens every large winner to +20 and leaves
   every large loser alone (`breakout_options_results.md:39-62`).
3. **Six-line, his exact set, 2,438 trades**: cap worth **-0.72 bp** (+0.54 -> -0.18), about
   5 cents per share on $700 (my arithmetic) (`six_lines_results.md:37-50`).

Plain-English money summary: on the two mechanical samples the cap costs a few cents per share
per trade; both mean results are noise-sized (CIs span zero) and under the 5 bp an ATM 0DTE
needs. On his real option trades, one run says the +25% target costs about $5,800 over 296
trades at his sizing ($3,458 minus -$2,348, my arithmetic, about $19.6 a trade), the other, on
a bigger sample and in dollars, says it would have helped by $4.22 a trade but is
indistinguishable from zero. Live, on 22 `Six_Lines` ATM trades the cap currently helps (section 8).
Instruction the project drew: "stop capping winners" (`CLAUDE.md`). Honest scope: the
mechanical measurements are on the underlying, with 20 bp as a proxy for a +50% option gain,
and none prices an option.

### Q5. The one real conditional signal, why it does not make money, and the path-versus-endpoint mistake

**What was found** (`after_break_results.md:20-40`): on days when R1 (the lowest of the three
highs, still live) close-breaks in the first two 10-minute bars (09:30 or 09:40), price
reaches R2 70.7% of the time (224 of 317) versus 59.3% (302 of 509) when R1 breaks at 09:50
or later; closes beyond R2 65.6% vs 54.0%; closes beyond R3 46.1% vs 32.2%. It survives the
control that matters: giving both groups the same 60-minute window (breaks by 14:50),
R2 breaks within 60 minutes 49.2% vs 37.9%, +11.3 pp, p=0.002. Of sessions reaching R2, 93%
close beyond it and 46% go through R3. I re-counted 317, 509, 224 and 302 from the stored
JSON and got the same.

**Why it is not money**: (i) "reached R2" is a statement about the path. The pre-registered
test of turning it into a trade (hold to the 16:00 close, no stop) gives B minus the no-signal
benchmark of -2.56 bp, wrong sign, and the benchmark (buy 09:50 on days it did NOT fire) earned
more (+4.25 bp) than either treatment arm. (ii) All of it is long-only: the S side is dead
(p=0.242). (iii) The sample is a decade in which QQQ roughly tripled (all of it in sample; SPY,
the only out-of-sample test, was never run and now is "not warranted",
`early_r1_hold_results.md:100-104`). (iv) Even a real +11 pp on how often R2 is reached is
price travel, and a naked option pays the VRP for travel.

**The 2026-09-23 mistake** (`after_break_results.md:42-58`; `early_r1_hold_results.md:36-62`).
The true number: of 219 early R1 breaks that the close-back-inside stop closed out, 119
(54.3%) still closed beyond R2 later that day. The false conclusion: "the stop is throwing
away more than half of its winners; removing it is the highest-value thing left to measure".
Why it failed: "R2 broke later that day" and "the trade would have been profitable at 16:00"
are different statements. R2 can break at 11:20 and price can be back under R1 by the close; a
held trade exits at the session close, not when R2 breaks. A fact about the PATH was read as a
fact about the ENDPOINT. Caught because the claim was turned into a pre-registered experiment
with the rule fixed beforehand (`early_r1_hold_preregistration.md`, same day). Result: the
stop is worth +1.84 bp (CI [-13.02,+16.81]), "two-thirds of a cent on a $740 stock". The
figure 54.3% was left standing and annotated, not deleted (the project's rule: wrong numbers
are corrected in place with the reason). The 54.3% itself was labelled exploratory because it
was computed after seeing the main result (`after_break_results.md:136-137`). (Note: the table
uses 314 trades, those where R1 was also the session's first break, against 317 early breaks;
the 3 gap is not explained in the files, inferred to be ties with another line.)

---

## 6. Records, formats, invariants, frozen things

- **Pre-registrations are frozen as written.** Nothing above an amendment line is edited; the
  entry-timing prereg shows the pattern: "Appended after execution. Nothing above this line has
  been edited" (`entry_timing_preregistration.md:149-153`). Wrong numbers get a dated correction
  in place (e.g. `breakout_options_results.md:114-149`, `after_break_results.md:44-58`).
- **`live_lab_data/` is a frozen forward test.** Do not touch config hashes
  (`CLAUDE.md`; `FREEZE.json`). Accepted hashes: `1f7247d7839d9950`, `1d7c5a9fc71d67fa`,
  `ccd00ac71f8243e9` (from 2026-09-25). The clock never resets; the family only grows (now 15).
  The 13 originals must stay byte-identical (`six_lines_setup_test.py:265-273`).
- **The stored sweep is the input to `after_break.py`**: `live_lab_data/six_lines_QQQ.json`
  with fixed fields. Changing `six_lines.py` changes what regenerating it would give; the
  committed file reproduces the research numbers (checked: 2,438 trades, +0.54 / -0.182 bp,
  median -13.94).
- **Frozen decision constants**: `CAP_BP=20.0`, `COST_FLOOR_BP=5.0` (`six_lines.py:158-159`);
  `INTERVAL_MIN=10`, `BUFFER=0.0005`, `DECIDE_AT=11:00` (`orb_veto.py:50-54`);
  `SEED`, `REPS=3000`, `SPLIT=2021-01-01` (`early_r1_hold.py:36-38`).
- **Prohibited in the prereg**: reporting any 1DTE number or any QQQ option result; mid-price
  fills; dropping post-switch signals silently; capped results without the uncapped control;
  changing the cap, switch time or level set after seeing the verdict
  (`breakout_options_preregistration.md:168-177`).
- **What cannot be measured, stop offering**: QQQ options (no archive), any 1DTE (archive is
  0DTE by construction), anything new from ThetaData (`Options: FREE` since 2026-09-08). The
  only option archive is SPY 0DTE, 769 sessions 2020-2024, and the 2024 block can be spent once.

## 7. Failure modes, incidents, traps

The one bug class, per `CLAUDE.md`: no crash, no error, a confident number that measured
nothing. Concrete instances in this slice:

1. **Wrong level set (8-line)**: a whole result set was a null about lines he does not draw.
   The merge of premarket into market hours deleted lines silently. Pinned by
   `six_lines_test.py::test_yesterdays_premarket_survives_inside_its_rth_range`.
2. **Profit cap not implemented**: the first `evaluate_signal` omitted exit 1 and reported
   -13.79 bp as "the strategy" (`breakout_sweep.py:66-70`). Found because the trader pushed back.
3. **Cap filling after exit**: lookahead in the fix for #2. Median MFE 34.26 -> 40.70 bp, cap
   hit 64.8% -> 74.5%, capped mean +4.98 bp CI [3.82,6.14] p=0.0003, **VOID**. Found because two
   runs on identical data disagreed (`breakout_options_results.md:151-167`). Tests verified to
   fail on the bad version.
4. **The "1-2 trades, around 11:39" claim**: both numbers wrong (the journal is 2.80-2.90 a
   day, and 11:39/11:44 is a median, not a cluster). Corrected 2026-09-22 in two documents
   (`breakout_options_results.md:114-149`, `six_lines_results.md:98-116`).
5. **Path read as endpoint** (section 5, Q5).
6. **Entry-timing placebo lookahead**: a control arm returning +27.12% mean / 74.3% win on a
   25-minute 0DTE hold was "the tell"; the contract's strike came from spot at his entry second
   (`entry_timing_results.md:28-113`). A second reporting error in the same run: placebo win
   rate computed on 20-draw averages (mechanically inflated).
7. **`full_reversal` metric** uses the whole session's extreme, so a late break on an
   already-swinging day looks like a reversal; flagged `[DEFECTIVE]` and unrepairable from the
   stored sweep (`after_break_results.md:118-128`; `after_break.py:148-165`).
8. **Histogram peak detector reported two modes for one peak**: the fixed-ratio valley rule
   split noise; replaced by a Poisson-noise-aware rule on 2026-09-22; on the real journal the
   split clears by only 2.04 sigma (`entry_histogram.py:93-136`; test pinning 1.95-2.15).
   An earlier version of the test used 25-20-25, which BOTH rules reject, so it passed against
   the unfixed code and proved nothing (`entry_histogram_test.py:101-114`).
9. **Conditional 0DTE tooling**: mixed git SHAs, an always-firing dirty-tree gate, a >100%
   concentration ratio (divided by net instead of gross) twice in one day, and a 4.5x
   overstatement from reporting the session-mean under the per-trade label
   (`conditional_0dte_armA_results.md:81-91`; `..._armB_results.md:71-90`).

Traps for whoever touches this next:
- **Zero-filled bars and holidays**: the vendor returns a full grid of zeros for closed days;
  `min()` of a day then becomes 0.00 and "yesterday's high" 0, which fabricated 105 QQQ / 98 SPY
  entries (6.8-7.0%) in the old code (`multi_level_orb_backtest.py:30-40`; memory
  `0dte-real-data-findings.md`). `get_ext` drops non-positive bars (`levels_test.py:60`).
- **Outage looks like a holiday (inferred)**: in `six_lines.run`, a day with fewer than 300 RTH
  bars is counted "thin" and skipped without updating "yesterday" (`:238-241`). A fetch
  exception is caught and skipped too (`:233-236`). So the 138 "thin" days are real holidays,
  half-days and any vendor gaps, indistinguishable in the output. `get_ext` also caches
  whatever came back (`levels_test.py:40-43, 69`).
- **Stored sweep ends 2026-08-27** although the run window was to 2026-09-19 (JSON; reason not
  recorded).
- **SPY was never run** for any of the six-line studies; every verdict is QQQ-only, in-sample.
- **The `EOD`/15:55 wording**: research exits at 16:00, live flatten is 15:55.
- **Tie rule**: at most one line per bar chosen alphabetically (`six_lines.py:182`;
  `setups.py:678`).
- **Premarket stray prints** feed straight into the lines (section 8, open item 3).

---

## 8. Current state

**Done / closed** (per `CLAUDE.md`, "WHAT IS SETTLED", and the documents): the 8-line null,
the six-line null (2,438 trades), the cap measurement (twice mechanically), round numbers, the
S1 early break, removing the stop, entry-timing skill. The 8 earlier direction nulls and
MOMO_CHASE are in other slices.

**Half-done / unresolved:**
- The veto study (`orb_veto_*`): code and 24 tests exist, no recorded result.
- The conditional straddle: Arm B passed but fragile; Arm A never reproduced; the pre-registered
  next step (a concave-utility linear-program bound) is described, not recorded as done
  (`conditional_0dte_armA_results.md:93-99`).
- Live forward record for the two six-line setups, **my snapshot from the worktree's committed
  `live_lab_data/trades.jsonl` (to 2026-10-12, 12 sessions, not corrected via the correction
  files)**: ATM arm, 22 trades each. `Six_Lines` sum of `pnl_net` **+$113.22** (11 "failed", 11
  "target"); `Six_Lines_NoCap` **-$1,370.78** (16 "failed", 6 "eod"). Neighbour strikes: ATM-1
  -$3.78 / -$1,412.78; ATM+1 +$239.22 / -$1,184.78. So far the cap helps on live fills, the
  reverse of the backtests, at n=22 against a 200-trade bar (`forward_test_preregistration.md:65-74`).
  Do not read this as evidence.

**Open (the project's own list, `CLAUDE.md` "STILL OPEN"):**

*Open item 1, the unwritten selection rule.* 92% of sessions break a line, the median first
break is 09:40 and the rule takes ONE trade per session (1,493 of the 2,438 trades signal in
the 09:xx hour, 566 in 10:xx, 165 in 11:xx, and 89/50/46/29 in 12:xx to 15:xx, my count from the
stored JSON), whereas he averages 2.80 trades a day with a long afternoon tail. Whatever picks
WHICH break he takes has never been written down; every measurement says the entry is null and
the exits carry his record. The earlier claim that the rule and the trader "are not doing the
same thing" came from a frequency gap that does not exist and is withdrawn
(`six_lines_results.md:101-116`).

*Open item 2, the entry-time histogram and why the median must not be quoted.*
Run 2026-09-22 on the lab machine (`round_trips.csv` is gitignored,
`entry_histogram.py:34-39`): 443 round trips over 158 days = 2.80 per day. Counts in 15-minute
buckets starting 09:30 (from `entry_histogram_test.py:97-98`, which pins the real run):
`25 41 29 18 20 19 29 20 22 26 18 18 12 14 13 14 15 12 8 12 12 11 11 7 8 8` (442 in market
hours, one trade after the close). Busiest bucket **09:45 (41 trades)**, median entry **11:44**
(the median falls in the 11:30 bucket), nearly two hours later. A median of a skewed or two-humped
distribution describes an hour that can hold few trades; with a long thin afternoon tail the
median sits hours after where he trades most. `peaks()` returns modes at 09:45 and 11:00, but the
split clears by only 2.04 sigma of counting noise, so the honest description is **one peak plus a
noisy declining plateau, not two humps**. This is the defect the script was written to catch
(`entry_histogram.py:1-14`).

*Open item 3 (also asked), the time-of-day P&L table, exploratory.* Whole book **+$1,502 over
442 trades = +$3.40/trade** (a 443rd trade after the close is omitted, memory
`moves-vs-turns-analysis.md`):

| window as labelled in CLAUDE.md | real clock extent (I checked against the 15-minute bucket array) | n | total | per trade |
|---|---|---|---|---|
| 09:30-09:45 | 09:30-09:59 | 66 | +$2,806 | +$42.52 |
| 10:00-11:30 | 10:00-11:44 | 157 | -$3,183 | -$20.28 |
| 11:45-13:30 | 11:45-13:44 | 130 | +$3,514 | +$27.03 |
| 13:45-15:45 | 13:45-15:59 | 89 | -$1,635 | -$18.37 |

The labels are bucket START times, so each window runs 15 minutes longer than it reads; counts
66/157/130/89 sum to 442 and the dollar totals sum to +$1,502. Three reasons it is not yet a
finding and he should not trade it (`CLAUDE.md` lines under "OPEN and exploratory"):
(a) **the window edges were chosen after seeing the histogram**: 26 buckets and a free choice
of contiguous groupings is the garden of forking paths that `setup_sweep_results.md` spent
2,822,400 cells demonstrating; (b) **no day-clustering**: 442 trades over 158 days are not
independent; (c) **P&L is tail-dominated**, so a bucket mean can be one fill. The histogram
script prints median `med $` and `top1 %` per bucket for exactly that (`entry_histogram.py:176-197`):
a bucket whose mean and median disagree in sign is one trade, not an hour. See section 9 on the
73.9% figure CLAUDE.md attaches to (c).

*Open item 4, stray premarket prints.* Memory `premarket-stray-prints.md` (found 2026-10-08 by
the day-chart agents and a reviewer): one premarket minute, most often 08:05 ET (also 08:00,
08:01, 04:00), carries a wick $2-$13 from that minute's own open and close. `build_six`
takes raw premarket highs and lows with no filter (`six_lines.py:72-80`), so the print becomes
"today PM high/low", and the next day "yday PM high/low". Counted: 23 of 120 premarket extremes
set by one print; on 2026-09-25 an 08:05 low of 737.11 on 1,816 shares set live S1 about $4
below every real trade; a scan of the QQQ cache found such extremes on 25-90% of days per
year; a second pattern, premarket bars repeating yesterday's close 7-16 times at about 07:35.
Not known: bad tick or real thin trade (not checked against trade-condition codes); his own
Robinhood chart shows the same wicks, so they are on the tape. I add one observation (inferred,
not verified): the very first stored session, 2016-01-05, has `R3 = 111.9035` as "yday premarket
high" against a regular-hours high of 110.18, a sub-cent price 1.5% above the day, which looks
like the same phenomenon in the historical data. Why it matters: the same `build_six` feeds the
live setups and every six-line study, including the early-R1 finding listed as real. The
family is settled; the memory note says to raise this as a question, not reopen the research:
does he count those 08:00 spikes when he draws his lines?

## 9. Contradictions (doc or memory versus code or other docs)

1. **Profit cap, journal row.** `CLAUDE.md` (and `PROGRAMME_REVIEW_2026-09-24.md:215`,
   `six_lines_results.md:43`) say "journal, 422 discretionary trades, dollars: +25% target ->
   71.6% win rate, still loses (breakeven 75.3%)". The 71.6% / 75.3% / -$2,347.76 figures come
   from the 296-trade return-space test (`construction_results.md:52-82`). The 422-trade
   dollar-space test (`journal_reverse_engineering_results.md:196-238`) shows a +25% target
   GAINS +$4.22/trade over his exit (p=0.487) and calls the earlier finding "fragile". The
   "three independent datasets" claim therefore stands firmly only for the two mechanical
   samples, both on the underlying.
2. **73.9% attached to the journal.** `CLAUDE.md` (open item on time-of-day P&L) says "P&L here
   is tail-dominated (73.9% in the top 1% of trades)". Elsewhere 73.9% is IntradayMomentumBoundary
   on QQQ shares (`PROJECT_REPORT.md:54-61`; `orb_veto_preregistration.md:137-138`). I found no
   journal-specific top-1% share in any file.
3. **"CSV has grown" explanation for 2.90 vs 2.80 trades/day.** `CLAUDE.md` says 357/123 = 2.90 and
   443/158 = 2.80 are "both right for their snapshot, the CSV has grown". But
   `entry_timing_results.md:159-160` already lists "all symbols n=443 days=158" and "QQQ/SPY
   n=357 days=126" on 2026-08-26, the same 443 as the 2026-09-22 histogram. So the CSV did not
   grow between those runs. The 2.90 pairs 357 trades with 123 days (the days of the 339
   evaluated, inferred) , so it is a mismatched ratio (357/126 = 2.83, 339/123 = 2.76). Also the
   histogram covers ALL symbols (default `--symbol None`, `entry_histogram.py:142`), including 86
   single-name trades, whereas the six-line frequency comparison is a QQQ/SPY question.
4. **Time-of-day window labels.** `CLAUDE.md` "09:30-09:45" is really 09:30-09:59 (bucket start
   labels); memory `moves-vs-turns-analysis.md` says the same ("off by one 15-min bucket").
5. **The withdrawn "journal median 1-2 trades" still sits in code.** `six_lines.py:167-168`
   ("a journal median of 1-2") and `six_lines_test.py:200` ("One per day matches the journal's
   1-2") were not corrected when the documents were (commit 0f99213, 2026-09-22).
6. **`six_lines.py` docstring "15:55" vs code 16:00** (`:173` vs `:207-208`); and live flatten
   15:55 (`runner.py:43`).
7. **Prereg "nearest un-breached level" vs code** (section 3.1).
8. **"R1 nearest above"** (docs) vs "lowest of three highs" (code), differs when R1 is pre-broken.
9. **Memory `feedback-prefer-simple-orb-strategy.md`** recommends a fixed 25-50% target as part of
   "the proven rule"; later work (`construction_results.md`, `six_lines_results.md`) measured
   capping as value-subtracting on mechanical entries.
10. **Memory `profitbook-0dte-options-goal.md`** describes `livebreakout.py` and the retest
    scripts as the strategy; `PROJECT_REPORT.md:42-48` concludes he is a midday momentum
    chaser, not a retest trader. Both can be true historically; the memory is stale as a
    description of the current method.
11. **Commit date vs amendment date.** Live setups and backfill commits are dated 2026-09-24
    (`4cc694f`, `e423c3a`); the amendment and the backfill header say 2026-09-25. Likely time
    zone / run-after-close, not material (inferred).
12. **Pre-registration for the six-line run.** `six_lines.py:24-26` calls itself "a TALLY" and
    "characterise before pricing"; the trade function and the 2,438-trade verdict arrived the
    next day with no separate pre-registration file.

## 10. Questions only the owner can answer

1. **Which break do you take?** The rule fires one trade a day on the first break, mostly
   09:40-09:59. You average 2.80 trades a day with a long afternoon tail. What decides which
   line, and whether to take a second or third one?
2. **Do you count the 08:00-08:05 premarket spikes when you draw your lines?** Or do you draw
   on candle bodies or only on sessions you consider "real"? This decides whether the live lines
   and the research lines are your lines.
3. **Do you look at the day-charts output, and how?** Daily, or after a bad day? Do the 10-minute
   and 15-minute charts match what you see?
4. **Is `trades.jsonl` the record you want me to quote?** The sums above are uncorrected. Which
   file carries the corrected figures (`trade_corrections.jsonl`, `stranded_recovery.jsonl`)?
5. **Was the veto study ever run?** `orb_veto_backtest.py` expects `sharewf_trades.json`; if the
   lab machine produced `orb_veto_results.json` it is not in this checkout.
6. **Why does the stored sweep stop at 2026-08-27** when the run window was to 2026-09-19?
7. **Do you actually enter on a close or on a touch?** Research is close-based on 10-minute
   bars with no buffer; the 08:00 prints and your Robinhood chart suggest you may read
   wicks.
8. **Does the +25-50% cap in your spec mean what the code calls +20 bp?** The code treats 20 bp
   as +50% of an ATM premium at delta 0.5; your real premiums and deltas may differ.
9. **The 1DTE switch after 1-2pm**: you may be trading it live (claim D is unmeasurable here).
   Do you want the lab to record when you do it?
