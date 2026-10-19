# 07 — The research programme as a whole, and the studies of the owner's own trading record

Written 2026-10-13 by a read-only agent. Evidence rule: every statement about a file carries a
`path:line` I read; anything inferred is marked **(inferred)**; "reason not recorded" means I
looked and found none. Paths are relative to the repo root unless they start with `memory/`
(= `C:\Users\chaitanyakharche\.claude\projects\C--Users-chaitanyakharche-Documents-stock\memory\`).
I did not run any code and did not open the main checkout.

Quick index to the nine questions in my brief: Q1 story = section 3.1. Q2 master table = section 4.1.
Q3 the eight nulls = 3.5. Q4 pre-registration = 3.2. Q5 his trading record = 3.3 (+3.4 pipeline).
Q6 the verdict = 3.6. Q7 IMB = 3.7. Q8 what remains testable = 3.8. Q9 how he wants to be
talked to = 3.9. Disagreements between CLAUDE.md and the files = section 9.

---

## 1. In one paragraph

This repo exists to answer one question: can the owner's hand-traded 0DTE options trading (0DTE = an
option that expires the same day) be turned into rules a computer can run profitably? About a year
of work, almost all of it written up as pre-registered experiments (the test and the pass/fail rule
are written down before the number is seen), gave this answer: every attempt to predict
*direction* (up or down) came back null; the signals he uses predict *how far* price travels, not
*which way* it ends; his entry rule can be recovered by machine learning (AUC 0.847) but trading
that rule blind earns about nothing; the only thing in his record that ever looked like skill was
his discretionary exit, and that finding later turned out fragile; and the single candidate that
survived heavy testing (IntradayMomentumBoundary, "IMB", on QQQ shares) pays about $18 a week on
$10,000 with three-quarters of its profit in 1% of its trades. The programme's own conclusion is
that more analysis of the existing data cannot settle whether he has an edge — the sample is the
limit, not the method — so the instrument left is the frozen live forward test (a paper-trading
lab that gathers one new session a day). The standing risk, which shaped every rule below, is "no
crash, no error, a confident number that measured nothing" (`CLAUDE.md:78-80`).

## 2. The goal it serves, and how it changed

* **Start (2026-07):** replicate his manual 0DTE "retest rejection" breakout strategy in code on
  the underlying (the stock/ETF price), then add an *options picker* (contract, strike, expiry,
  sizing) to output recommendations. Source: `memory/profitbook-0dte-options-goal.md:11-15`
  (dated 2026-07-26). Validation plan then: test the swing version first, on previous-3-days
  support/resistance breakout levels, 5-minute candles, volume-filtered
  (`memory/profitbook-0dte-options-goal.md:17`). The public demo is the HF Space `ckharche/ProfitBook`.
* **Reframed (2026-08):** from "tune the strategy" to "is there anything there at all, and what
  does he actually do?" `research/PROJECT_REPORT.md:10-12`: "can a discretionary 0DTE options
  strategy be automated profitably?" The 2026-08-16 lookahead discovery (3.1 phase 3) is what turned
  a tuning project into a measurement project.
* **Reframed again (2026-08-28 onward):** because existing data could not settle it, build an
  instrument that makes *new* data: the live lab (`research/PROJECT_REPORT.md:79`, "the only
  instrument that produces new information"). Branches since: HPC cluster (what data already on
  hand can settle), live lab (what only waiting can settle), HF Space (what can be shown)
  (`research/PROJECT_REPORT.md:17-24`).
* **Now:** the stated stopping position is "prediction is closed; stop capping winners; the open
  question is the unwritten rule that picks *which* break he trades" (`CLAUDE.md:114-121`). His
  stated money objective, used as the yardstick in several studies, is **$200–500 per week**
  (`research/entry_timing_preregistration.md:77-79`, `research/package_preregistration.md:19-20`).
  Whether that is still his number: not recorded.

---

## 3. How it works, end to end

### 3.1 The story of the programme, one paragraph per phase (Q1)

Dates are from the files cited. Git history in this repo begins 2026-07-30 (`git log`: `e17df27`
"Checkpoint before dead-code cleanup sweep", `f384d81` "added 3 day confluence layer"), so earlier
work is only visible through the memory notes.

**Phase 0 — the idea (to 2026-07-30).** He trades 0DTE options on SPY/QQQ (and a few single names)
using a "retest rejection" breakout and wanted it coded. The assistant built `live_deploy*.py` style
scripts that computed signals, entries and targets on the *underlying*, not on real option
contracts (`memory/profitbook-0dte-options-goal.md:11`). Nothing from this phase is quoted as
evidence anywhere later.

**Phase 1 — add sophistication, test on synthetic pricing (about 2026-08-02).** ATR-based
targets/stops, multi-day support/resistance confluence scoring, dealer-gamma ("GEX") regime
de-rating and a scalp-exit repricing model were each backtested against a Black-Scholes simulator
on roughly 30-trade samples. Every one showed no benefit or made things worse, so a standing
preference was recorded: keep it simple — ORB confirm, naked long call/put, fixed target; put any
new variant in a separate file (`memory/feedback-prefer-simple-orb-strategy.md:11-15`,
`memory/feedback-isolate-experimental-strategy-code.md`). Lesson drawn: complexity fitted noise on
a tiny sample. (The memory cross-references a note `options-premium-synthetic-backtest` that does
not exist in `memory/MEMORY.md` — missing.)

**Phase 2 — real quotes and an apparent edge (2026-08-14 to 08-15).** With ThetaData (a market-data
vendor; Options Value tier, plus an $80 Stock STANDARD line for history back to 2016) the 0DTE rule
was re-run on real bid/ask quotes: "QQQ works, SPY does not", +7.07% per trade with 6 of 6 positive
years, and a 2016–2026 test on the underlying with t = 3.55 ("the signal is real")
(`memory/0dte-real-data-findings.md:11-18`, `memory/underlying-orb-decade-test.md:39-44`). Then the
2020 extension weakened it — 6/6 years became 6/7, confidence fell from 70% to about 55%
(`memory/0dte-real-data-findings.md:129-146`). Lesson, learned the hard way a day later: "significant
here, not significant there" was being treated as a finding.

**Phase 3 — the lookahead (2026-08-16 to 08-18).** `find_entries` tested a 5-minute bar's *close* but
recorded the bar's *label* as the entry time; `pandas.resample` labels left, so the bar labelled 09:30
does not close until 09:35, and the backtest bought the option five minutes before the signal
existed. Fixing it turned QQQ +6.44% (win 80.3%) into −8.34% (win 68.9%), and the underlying "t = 3.55"
into t = 0.44; the bias was 88% of the measured underlying signal and +11 to +15 points per option
trade (`memory/0dte-real-data-findings.md:31-43`, `memory/underlying-orb-decade-test.md:11-30`). An
audit on 2026-08-18 added three corrections: the 75.3% breakeven win rate belongs to the +25% target
only; the fixed 0.20% "MFE proxy" is optimistic by about 10 points of win rate against real option
pricing; and about 7% of entries were fabrications from zero-filled holiday closes
(`memory/0dte-real-data-findings.md:60-83`). Confidence in a tradeable edge fell from about 55% to
under 15%. This is where pre-registration and the three reviewer agents (3.2, section 4.3) enter:
the first pre-registrations in `research/` are dated 2026-08-18 (`research/percentile_return_preregistration.md:7`,
`research/c1_f1_f2_preregistration.md:7`).

**Phase 4 — study the trader instead of the rule (2026-08-18 to 08-26).** The reviewer-agent prompt
asked whether anyone had checked that he makes money *at all* against broker statements
(`.claude/agents/strategy-contrarian.md:36-40`). The Robinhood activity export gave a verified P&L of
+$2,341.12 with day-level t below 1 (`memory/robinhood-broker-dataset.md:41-47`). Then, in order:
the percentile-of-achievable-return test (2026-08-18, "consistent with random timing once duration
is matched", `research/percentile_return_results.md:98-104`); a scoping inventory of what remains
testable (`research/remaining_testable_inventory.md`); the C1/F1/F2 three-endpoint pass (all null or
scale-only); an API pull of his full order history with execution timestamps (the activity CSV is
date-only; `research/pull_order_history.py:1-14`), which unblocked the timing questions; then his
declared-indicator study, setup cross-check, 24-setup sweep and ten standalone setups (all null);
the construction tests (the one result that cleared Holm: his exit beats a +25% target); the
entry-timing test (his own control arm contained lookahead and was voided); and the "package" test
(0 of 10 filters help weekly dollars). It closed on 2026-08-26 with the verdict in 3.6
(`research/PROJECT_REPORT.md:66-80`). Lesson: dozens of ways to slice 443 trades, none clears;
the arithmetic of the sample size is the real finding.

**Phase 5 — the live lab and the shares candidate (2026-08-27 to 09-05).** The only thing that ever
cleared a multiplicity correction on a large sample was IMB, and on *shares*, not options; it was not
being forward-tested, so a specification was drafted 2026-08-27 and frozen 2026-08-28 with 13 setups
(`research/live_lab_specification.md:3-6`, `research/forward_test_preregistration.md:3-12`). The
same week: a walk-forward showed 95.7% of IMB's apparent *option* edge was a one-minute lookahead
(`research/walkforward_results.md:56-81`); the journal was reverse-engineered by machine learning
(`research/journal_reverse_engineering_results.md`, 2026-08-29); IMB was stress-tested
(`research/imb_stress_results.md`, 2026-08-29); a shares arm was added (2026-08-31,
`research/shares_lab_preregistration.md`); persistence was tested (2026-09-04); the shares arm was
widened to 15 names after a spread survey and a feed-capacity fix (2026-09-02 and 09-05).

**Phase 6 — variance, and the cluster (2026-09-08 to 09-16).** A bug in the HF demo app revealed that
its momentum score was a pure *magnitude* being forced to emit a call/put direction, which suggested
testing variance rather than direction (`research/vrp_preregistration.md:8-30`). Result: the 0DTE
variance risk premium is real (gross t = +7.52) and 1.84 times too small to trade after spread
(net t = −5.38) (`research/vrp_cost_model_results.md:12-24`). A conditional version was fragile
(2023 only) or failed to replicate (`research/conditional_0dte_armA_results.md`,
`..._armB_results.md`). On the cluster, a 2,822,400-cell sweep of the MOMO_CHASE form (the encoding of
his own behaviour) returned SPA p = 0.8227: closed (`research/momo_sweep_results.md:9-24`). That phase
is another agent's slice; I record only the one-line verdicts.

**Phase 7 — the six lines and what happens after a break (2026-09-17 to 09-25).** He described his
levels precisely on 2026-09-20; the first 8-line sweep had tested a level set he does not draw
(`research/breakout_options_results.md:16-28`). The correct 6-line set came back null
(`research/six_lines_results.md:28-34`); one real *conditional* signal was found — an early R1 break
reaches R2 more often, +11.3 percentage points (`research/after_break_results.md:28-41`) — but it did
not turn into money, and a hypothesis built on it (remove the stop) was falsified on all three
gates (`research/early_r1_hold_results.md:8-33`). The 2026-09-24 programme review catalogued 16
defects found in the preceding four weeks (`research/PROGRAMME_REVIEW_2026-09-24.md:243-304`).

**Phase 8 — keep the lab alive, ask smaller questions (2026-09-26 to today, 2026-10-13).** Network
resilience, incident write-ups, 22-session replay, a day-type study (trend pays, chop loses, trend
cannot be forecast: AUC 0.50, 8 of 8 ex-ante tests failed — `memory/day-type-study.md:13-14`), a
moves-vs-turns split (no consistent effect — `memory/moves-vs-turns-analysis.md:10-12`), a day-chart
tool, and the premarket-stray-print question. The live-lab operations belong to other reports.
Where it stands: see section 8.

### 3.2 What "pre-registration" means here, in practice (Q4)

**Plain English:** write the question, the sample, the exact formulas, the number of tests, and the
result that would make you give up *before* running anything; commit that file; then run it and write
a separate results file. If the result is inconvenient, you may not move the goalposts.

**The recurring structure of a pre-registration file** (visible in all of them, e.g.
`research/c1_f1_f2_preregistration.md`, `research/package_preregistration.md`):

1. A dated header ("Written …, before any endpoint was computed") and a rule that changes are
   *appended as dated amendments, never edited in place* (`research/c1_f1_f2_preregistration.md:5-7`;
   `research/journal_reverse_engineering_preregistration.md:195-308` has amendments A1–A4).
2. Frozen sample and a funnel (how many rows dropped and why) reported afterwards.
3. Timing rule (decision time is the bar's *close*, never its label).
4. One primary endpoint, a fixed family size for the multiplicity correction (**Holm**: a stricter p-value
   threshold when many tests are run) that "will not be reduced if a member fails to execute"
   (`research/construction_preregistration.md:130-135`; `research/package_results.md:48-49`).
5. Inference by **day-clustered bootstrap** (resample whole trading days 10,000 times, because trades
   in one day are not independent; typical design effect 1.31–1.43, `research/c1_f1_f2_preregistration.md:123-127`).
6. A numeric **falsifier** and a power statement (the minimum detectable effect, MDE) committed in
   advance; a result smaller than its MDE is labelled underpowered "regardless of its p-value"
   (`research/construction_preregistration.md:79-85`).
7. A leak audit table (what is known before vs after the trade) and a "Prohibited" list.
8. Controls run and reported *before* the primary (percentile C1–C3,
   `research/percentile_return_preregistration.md:186-204`).
9. A stated prior, so a null cannot later be presented as "expected" and a table cannot be presented as a
   discovery (`research/package_preregistration.md:27-39`).
10. Header flips to "PRE-REGISTERED, AND RUN. Results: …" with the design "kept verbatim"; wrong numbers
    are corrected in place with the reason, never deleted (`CLAUDE.md:99-101`).

**Why the owner adopted it — evidence, and a gap.** The reviewer-agent prompt says why in its own
words: "This project has already published a result that was 88% artifact, survived months of review,
and justified a hardware and data purchase before anyone caught it. You exist so that does not happen
twice" (`.claude/agents/statistical-auditor.md:10-13`); the artifact was the underlying-ORB t = 3.55 that
"partly justified the $80 Stock STANDARD purchase" (`memory/underlying-orb-decade-test.md:31-32`). The
forward-test file states the philosophy: "If everything fails, that is a result and the programme ends
with an answer instead of a suspicion" (`research/forward_test_preregistration.md:139-141`). *Who*
first proposed the discipline, him or the assistant: **reason not recorded** (the pre-registrations are
written in the assistant's first person).

**The specific incidents that justified it** (each is one the discipline either caught or would have):

| incident | what happened | where |
|---|---|---|
| Lookahead (label vs close), 2026-08-16 | 88% of a "t = 3.55" signal was five minutes of free drift | `memory/underlying-orb-decade-test.md:18-30` |
| Intra-bar entry credit, 2026-08-29 | a forward-return measured from the prior bar's close handed him 31.9 s of the next bar's +1.905 bp: +2.271 bp, Holm 0.0096 (clears) vs +0.850 bp, Holm 0.9736 once interpolated at his actual entry second | `research/journal_reverse_engineering_results.md:93-107` |
| Garden of forking paths | the best of 23 setups on his trades had raw p = 0.107 (+9.49%); the "3 gates" bucket showed +7.44% in an inverted-U; both are what a menu search reports and both vanished under the fixed family | `research/setup_crosscheck_results.md:59-64`, `research/indicator_state_results.md:72-75` |
| Garden of forking paths, large scale | the best MOMO_CHASE cell scored Sharpe 1.688, what pure noise predicts; SPA p = 0.8227 over 2,822,400 cells | `research/momo_sweep_results.md:9-24`; `memory/MEMORY.md` index line for "Momo sweep" |
| A control arm that printed money | the entry-timing placebo returned +27.12% with a 74.3% win rate; the ATM strike had been chosen from information at *his* entry second | `research/entry_timing_results.md:42-62` |
| A rule that finds a rule in noise | a depth-3 decision tree on pure noise routinely finds a 70% "win" leaf in 418 trades; the real win/loss tree found 59.3% | `research/journal_reverse_engineering_results.md:346-356` |
| Path versus endpoint, 2026-09-23 | "54.3% of stopped-out sessions later closed beyond R2" (true) was read as "the stop throws away winners" (false); turned into an experiment, it was worth +1.84 bp, p = 0.829 | `research/after_break_results.md:44-58`, `research/early_r1_hold_results.md:8-33`, `CLAUDE.md:86-93` |
| Pre-registration that cannot be satisfied | BH-corrected feature importance needed p ≤ 0.0011 but 200 permutations floor p at 1/201 = 0.00498; written down as the author's own error | `research/journal_reverse_engineering_results.md:326-331` |
| Registered but not run | `momo_v2` sat unexecuted nine days; the 50/100-trade checkpoints passed unrecorded; nine pre-registrations still said "NOT YET RUN" after results landed | `research/momo_v2_results.md:6-9`, `research/live_lab_checkpoint_100_diagnostic.md:9-14`, `research/PROGRAMME_REVIEW_2026-09-24.md:283-285` |

The practices the 2026-09-24 review credits with catching most defects: pre-registration with the
decision rule fixed first; two runs disagreeing on identical data; mutation-testing every new test;
and the trader pushing back when a result did not match his experience
(`research/PROGRAMME_REVIEW_2026-09-24.md:306-314`).

### 3.3 The owner's actual trading record (Q5)

**Two data sources, easy to confuse.**

| | activity export (CSV) | order history via API (the "journal") |
|---|---|---|
| file | `2025 complete trades.csv` (project root; the other three CSVs are redundant subsets) | `research/order_history_raw.json` → `research/round_trips.csv` |
| tracked in git? | no (`.gitignore` has `*.csv`) | no (`.gitignore:81-83`; `*.csv`) |
| carries clock time? | **no** — date only | **yes** — per-fill timestamps, order type, limit price, state |
| size | 1,105 option fills (540 buy-to-open, 565 sell-to-close), 2025-04-07 → 2025-12-03, 120 active option dates, 333 eligible round trips | 1,222 executions, 405 contracts, **443 round trips, 158 days, 2024-09-11 → 2026-02-20**, 13 symbols (357 QQQ/SPY, 86 single names) |
| verified P&L | the older 1,051-fill May–Nov file sums to **+$2,341.12** (day-clustered t 0.63–0.91); the full year is **+$2,073.71** across all option rows / +$2,587.07 on eligible round trips, day-level **t = 0.73**, mean +$22.50/day, CI [−$39.41, +$82.84] | execution-derived net **$1,452.82**, "matching the CSV to the cent" |
| source for the line | `memory/robinhood-broker-dataset.md:11-47` | `research/entry_state_preregistration.md:31-37`; `research/indicator_state_preregistration.md:30-31`; `research/entry_timing_results.md:159-168` |

Both are untracked and, per the histogram script, "only exist on the lab machine"
(`trade_analysis/backtesting/entry_histogram.py:5-8, 36-41`). I could not and did not verify they are
present. `research/pull_order_history.py` is the tool that made the API pull: it is interactive-login
only (never takes credentials as arguments), only issues GETs, and leaves a session token at
`~/.tokens/robinhood.pickle` that the docstring says must be deleted (`pull_order_history.py:20-33,
104-106`). It uses `robin_stocks`, an unofficial client whose terms "restrict automated access"
(`:31-33`). Whether the token was deleted: **not recorded**.

**How he really trades (all numbers with source).**

* *Frequency.* 2.80 trades/day (443/158, `CLAUDE.md:134-136`). He traded about 73–74% of sessions (105 of
  ~144 sessions in the older window) and traded both call and put sides on 56–59 of the active days
  (`research/remaining_testable_inventory.md:14, 76`, `memory/robinhood-broker-dataset.md:66-68`).
* *Contract.* 79% of buys are 0DTE; median premium $0.76; modal size 5 contracts; real fees $0.085 per
  contract round trip, not the $1.30 the older models assumed (`memory/robinhood-broker-dataset.md:63-66`).
* *Entry timing.* Busiest 15-minute bucket 09:45 (41 trades); median 11:44; a long thin afternoon tail;
  `peaks()` finds modes at 09:45 and 11:00 but the split clears only 2.04 sigma, so "one peak plus a
  noisy declining plateau" (`CLAUDE.md:134-142`). Only 5.7% of entries fall in 09:30–09:45, the window
  the repo backtested for months (`research/PROJECT_REPORT.md:41-47`). He is a momentum chaser: the prior
  15-minute move was +0.2029 sigma in his direction 89.3% of the time
  (`research/PROJECT_REPORT.md:41-47`); 92.8% of his entry orders are executable on arrival
  (`research/entry_state_preregistration.md:172-176`); late-day extreme breakout 0.0% and "flag"
  0.7% (`research/setup_crosscheck_results.md:36-44`).
* *Declared indicators.* MACD aligned on 69.4% of entries, DMI aligned 78.5%, ADX above 20 on 69.6%,
  but volume is a coin flip (median 0.986 times its 20-bar average, only 48.3% above it)
  (`research/indicator_state_results.md:16-33`).
* *Win rate and shape.* Realised structure: average win +48.1%, average loss −37.0%, breakeven win
  rate 43.5%, actual about 45.8% (`research/construction_results.md:84-86`); Robinhood-memory figure
  48.9% on a different window (`memory/robinhood-broker-dataset.md:43-44`). Mean hold 40.6 minutes in
  the 422-trade reconstruction, median 25.4 minutes (`research/journal_reverse_engineering_results.md:241-250`;
  `research/setup_standalone_preregistration.md:36-37`). Limit orders earn real price improvement: crossing
  the spread both ways would cost $2.65 per trade more than he actually paid
  (`research/journal_reverse_engineering_results.md:206-209`).

**Where the money comes from, and how tail-dominated it is.**

* The tape is positive at every cut and significant at none (day-clustered, 20,000 reps,
  `research/entry_timing_results.md:156-172`):

  | cut | n | days | mean per trade | 95% CI | p |
  |---|---|---|---|---|---|
  | all symbols | 443 | 158 | +$3.28 | −$13.09 to +$19.19 | 0.680 |
  | QQQ/SPY | 357 | 126 | +$7.03 | −$9.69 to +$23.99 | 0.410 |
  | QQQ | 279 | 89 | +$8.72 | −$8.18 to +$25.41 | 0.304 |
  | single names | 86 | 56 | −$12.28 | −$57.35 to +$29.16 | 0.609 |

  Totals: all +$1,452.82 (+$19.30/week); QQQ/SPY +$2,508.54 (+$59.52/week); QQQ +$2,433.22
  (+$85.59/week); single names −$1,055.72. The single-name split was made after seeing outcomes and the
  document itself says to "treat with suspicion" (`research/entry_timing_results.md:174-179`).
* Tail concentration in the *activity* data: removing the top 5 of 333 contracts turns +$2,341 into −$393;
  one contract is 28% of seven months (`memory/robinhood-broker-dataset.md:46-47`).
* Percent versus dollar lies: the same trades give t = +1.77 in percent-return and t = +0.74 in dollars;
  use dollars (`memory/robinhood-broker-dataset.md:49-52`).
* Time-of-day pattern (**exploratory, not a finding**): +$2,806 / −$3,183 / +$3,514 / −$1,635 across four
  windows on a +$1,502 book (`CLAUDE.md:144-160`). The row label "09:30–09:45" really covers 09:30–10:00
  (25 first-quarter-hour trades + 41 in the 09:45 bucket = 66), per `memory/moves-vs-turns-analysis.md:15`.
  Three reasons not to trade it are in `CLAUDE.md:152-160`; see section 9 for a mis-citation inside
  reason (c).

**Why his rule is "recoverable (AUC 0.847) and loses money."** Plain English: a computer can tell his
entry minutes from other minutes of the same day almost as well as a human could, but doing what that
computer learned, blindly, makes nothing.
* *Recoverable.* 418 entries (of 443; 25 dropped for being too early in the session for the indicator
  window, none outcome-related) vs 3,274 same-session "placebo" minutes (8 per trade); 45 market-state
  features; random-forest out-of-fold AUC 0.8469 (AUC: 0.5 = coin flip, 1.0 = perfect), with 500
  within-session label permutations giving null mean 0.5003, p = 0.0020. It is not time of day (clock-only
  features score 0.556; everything except the clock scores 0.846)
  (`research/journal_reverse_engineering_results.md:11-14, 39-54`).
* *What the rule is.* About 1.1 ATR above the 5-minute 9-EMA (AUC 0.809), 1-minute DI+ minus DI− of about
  +16 (0.806), trailing 15-minute move +0.85 sigma (0.791). His levels — round numbers, prior-day
  levels, the gap — are at chance (AUC 0.50–0.503)
  (`research/journal_reverse_engineering_results.md:58-85`).
* *It loses.* The 4-feature zone over 34,534 blind signals 2016–2026 returned −0.15 to −0.22 bp (1 bp = 0.01%),
  significantly negative at 5 minutes (Holm 0.0456); the full 45-feature model scored on 721,887
  out-of-sample QQQ minutes (ending the day before his first trade), top 1%, had a best CI upper bound of
  +2.34 bp against the +5 bp an ATM 0DTE needs to clear spread and theta
  (`:121-168`). His own entries, once the intra-bar bias is removed, run +0.85 to +1.67 bp with every CI
  crossing zero (`:176-196`). Predicting *whether he enters* gives AUC 0.847 (p = 0.002); predicting *whether it
  worked* gives AUC 0.512 (p = 0.284) (`:372-378`).
* Net: "Whatever separates him from [the rule] is not in these 51 observables, is worth at most 1–2 bp, and
  cannot be copied from this data" (`:402-405`).
* The exit side is covered in 3.6.

**Why the trades cannot be reconstructed without hindsight (as recorded 2026-08-20).** From market data
alone: day selection (he trades ~73% of sessions), direction (he is on both sides on 56% of active days)
and entry timing all fail; only contract choice and sizing are reproducible. Inferring entry time from the
fill price is many-to-one (P(unique) about 0% inside the spread, ~149-minute candidate spans) and also pins
moneyness, so any "he buys X% out of the money near level L" claim built that way is circular
(`memory/robinhood-broker-dataset.md:71-78`; `research/remaining_testable_inventory.md:153-158`). The old
`trade_journal/robinhood_trade_analyzer.py` also fails this test in code: its moneyness field is built
from `(day_open + day_close) / 2`, i.e. the outcome (`trade_analysis/trade_journal/robinhood_trade_analyzer.py:428-437`),
and its P&L uses `matched_qty = min(total_buy_qty, total_sell_qty)` (`:491`). **Important update:** that
"timestamps are unavailable" position was overtaken within days — `pull_order_history.py` delivered
execution timestamps by 2026-08-20 (`research/entry_state_preregistration.md:36-37`;
`research/construction_preregistration.md:119-129`), after which the timing studies ran. What remained
true is that the *rule* is describable (AUC 0.847) but not tradeable.

### 3.4 How the journal studies chain together (code flow)

```
pull_order_history.py  --(robin_stocks GET only)-->  research/order_history_raw.json   [gitignored]
        -> (round-trip builder: not in the files I was assigned; output research/round_trips.csv)  [gitignored]
round_trips.csv columns I saw used: date, symbol, right, expiry, strike, entry_ts, exit_ts, hold_min,
        pct, net, qty, dte, entry_ask, entry_bid
   (trade_analysis/backtesting/journal_features.py:479-512; broker_entry_state_study.py:58-80)
journal_features.py main():433  builds journal_features.pkl : 418 entries + 8 same-session placebos each
        (SEED 20260829, PREFIX_SESSIONS 2, HORIZONS 5/10/15/25; :50-53)
   -> journal_ml.py        Q1 selection, Q2a/Q2b outcome, M2 null importance, M3 depth-3 tree
   -> journal_zone_wf.py   Q3  blind walk-forward of the recovered 4-feature zone (QQQ 2016-26, SPY 2022-26)
   -> journal_model_wf.py  Q3b score 721,887 QQQ minutes with the full classifier -> model_wf.pkl
   -> journal_model_report.py  Q3b reporting from the cached scores
   -> journal_exit.py      Q4a 14 mechanical exits vs his, Q4b per-minute exit hazard (real NBBO, FEE 0.0404/side)
   -> journal_kernel.py    M4 Lo-Mamaysky-Wang kernel patterns, 10 patterns, Holm m=10
Separate: broker_entry_state_study.py  helper functions only (no main) for the entry_state pre-registration
Separate: entry_histogram.py  descriptive entry-time histogram (no p-values)
IMB side: imb_stress.py and persistence.py read live_lab_data/sharewf_trades.json (2,905 trades), written by
        trade_analysis/live_lab/sharewf.py
```
Docstring purposes: `journal_features.py:1-25`, `journal_ml.py:1-19`, `journal_zone_wf.py:1-22`,
`journal_model_wf.py:1-17`, `journal_exit.py:1-14`, `journal_kernel.py:1-17`, `imb_stress.py:1-22`,
`persistence.py:1-13`, `broker_entry_state_study.py:1-33`.

### 3.5 "Eight pre-registered nulls" on direction prediction (Q3)

The list as written in `research/PROJECT_REPORT.md:30-37`, with where each lives:

| # | null | headline | file |
|---|---|---|---|
| 1 | strike distance (C1) | −$1.00 per sigma, CI −6.11 to +4.22, Holm 1.0 | `research/c1_f1_f2_results.md:36` |
| 2 | position size (F1) | −$2.03 per extra contract, Holm 0.048 — "exposure/scale effect only": gone in percent (p 0.139) and in total dollars (p 0.126) | `research/c1_f1_f2_results.md:37, 94-100` |
| 3 | within-day ordinal (F2) | +$0.30 per step, Holm 1.0 | `research/c1_f1_f2_results.md:38, 102-106` |
| 4 | entry-timing percentile | unconstrained 0.5522 (bar was lower CI bound > 0.53, got 0.5143: inconclusive); matched-duration d=10: 0.5057, d=20: 0.5151 → (B) random timing | `research/percentile_return_results.md:37-62, 98-104` |
| 5 | 19-feature ceiling model | OOS R2 −0.078, AUC 0.431 | **no results file in `research/`**; second-hand only: `research/indicator_state_preregistration.md:24-25`, `research/PROJECT_REPORT.md:33` |
| 6 | his declared indicator stack | OOS R2 −0.0623 (p 0.835); AUC 0.392 | `research/indicator_state_results.md:39-50` |
| 7 | ten external published setups, standalone | 0 of 10 advance, 1,542 sessions x 2 symbols; `pullback_9ema` = −0.0017 sigma | **no results file**; `research/setup_standalone_preregistration.md:3` still says "NOT YET RUN"; numbers quoted second-hand in `research/forward_test_preregistration.md:130`, `research/package_results.md:87`, `research/PROJECT_REPORT.md:34-37` |
| 8 | entry timing vs a same-contract control | null at 5/10/15/25 min after the pre-registered C-skill test was voided (own lookahead) and redone with a post-entry-only control (exploratory, not confirmatory) | `research/entry_timing_results.md:94-112` |

Notes: (a) the pre-registrations number their own nulls differently — "Sixth consecutive"
(`research/indicator_state_results.md:101-102`), "Seventh" (`research/setup_crosscheck_results.md:80`,
`research/setup_sweep_results.md:80`), "Ninth" for the package test (`research/package_results.md:155`),
"Nine nulls stand. Eight tested direction" (`research/conditional_0dte_preregistration.md:10-11`). (b) Further
nulls from his record that are **not** in the eight: setup cross-check 0 of 7 (coverage saturated at 91.4%),
24-setup sweep 0 of 23 (and zero raw p < 0.05, fewer than the 1.2 expected by chance), package 0 of 10,
journal exit 0 of 14, kernel patterns 0 of 10.

### 3.6 The verdict: "prediction is closed; the only positive is his discretionary exit; more analysis cannot settle it, only more trading can" (Q6)

**In plain words.** Nothing observable predicts which way his trades go. The one comparison that cleared
a multiplicity correction was that his *discretionary exit* beats a mechanical +25% profit target. That is
a statement about the exit policy, not about being able to forecast. And 443 trades are too few to know
whether he has an edge at all.

**The claim and its sources.**
* Closed: 8 nulls (3.5). Source of the verdict wording: `research/PROJECT_REPORT.md:66-80` (closing phase
  one, 2026-08-26) and `memory/research-programme-verdict.md:11-18`.
* The one positive: construction Test B. A +25% target loses to his discretion by 8.97 percentage points
  per trade, Holm p = 0.027, the only one of five comparisons to clear. The +25% target raises his win rate
  to 71.6% and still loses (a +25%/−100% structure breaks even at 75.3%); his realised structure (average
  win +48.1%, average loss −37.0%) breaks even at 43.5% and he runs 45.8%
  (`research/construction_results.md:63-92`). "Capping winners destroys the asymmetry that makes his record
  positive" (`:87-89`).
* **It is fragile.** On 2026-08-29, rebuilding his P&L path from real quotes on 422 trades (entry at the
  ask, exit at the bid, falling through to his own exit time when a rule never fires), 0 of 14 mechanical
  exits differ from him at Holm < 0.05, and the same +25% rule *gains* +$4.22 per trade (p = 0.487) — "the sign
  flips with the weighting and neither version is robust. Test B should be treated as fragile"
  (`research/journal_reverse_engineering_results.md:211-237`). Note the two runs also differ in sample
  (296 vs 422 trades) and in the fall-through rule (15:45 bid vs his own exit); see section 9.
* **Exits are reproducible but not exploitable.** Per trade, mean maximum favourable excursion (best paper
  profit during the hold) is +$81.68 while he realises +$0.78; the peak arrives at minute 19.1 and he holds
  to 40.6; 74.6% of trades exit after the peak; losers are held longer than winners (43.0 vs 37.8 min).
  Yet every rule that tries to harvest this (all time exits, trailing stops, both stops) is *worse*
  (`research/journal_reverse_engineering_results.md:239-264`). His exit timing *is* predictable from
  observables (OOF AUC 0.7036 vs a within-trade null of 0.6760, p = 0.01) — "a reproducible exit policy of no
  measurable value" (`:266-281`).
* **Why more analysis cannot help:** QQQ per-trade standard error is about $8.57 on a mean of $8.72, so
  t is about 1.0; reaching t = 2 needs about 4 times the observations, roughly 356 more QQQ trading days
  (about 1.5 years) (`research/entry_timing_results.md:201-209`, `research/PROJECT_REPORT.md:75-77`). On the
  smaller activity dataset: resolving a true $10-per-contract edge needs about 2,357 round trips against 333
  on hand (about 3.2 years at the current rate) (`research/remaining_testable_inventory.md:314-318`).
  "The sample is the binding constraint, not the method."
* What the programme adds on top, after the verdict: "MAGNITUDE, NOT SIGN" — four independent measurements
  that these levels predict how far price moves, never which way (`CLAUDE.md:44-55`); and "the profit cap
  subtracts value" (`CLAUDE.md:57-66`) — which I could not reproduce for its first leg (section 9).
* Realised-tape context: a mechanical ATM fixed-hold on his entries is worth +$4.93/week at one contract
  against his +$59.52/week (QQQ/SPY) — a 12–17x gap — but NBBO-priced simulations overstate his
  realised cash by **+$6.85 per trade** (same 334 trades, same capital), so +$4.93/week "should not be
  treated as achievable" (`research/entry_timing_results.md:187-199`, `research/package_results.md:117-145`).
  Strict ATM (nearest strike, no offset) is worth about +$1 per trade, not significant, agreed by two
  independent estimates (+$0.98 and +$0.95) (`research/package_results.md:109-113`).

### 3.7 The IMB shares candidate (Q7)

**What it is.** IntradayMomentumBoundary: a published rule (Zarattini, Aziz, Barbon, SSRN 4824172, Concretum
research) with parameters nobody here invented — "the only item needing no invented parameter"
(`trade_analysis/live_lab/setups.py:215-222`). At the :00 and :30 checkpoints between 10:00 and 15:30, go long
if price is above an upper boundary (the larger of today's open and yesterday's close, scaled up by a
per-minute-of-day volatility `sigma`) and above VWAP; short symmetrically; exit when price falls back
through the larger of VWAP and the boundary (a trailing exit); at most 4 per day (`setups.py:215-254`).
Sample: QQQ shares, 2,905 trades, 1,560 sessions, 2016-01-20 → 2026-08-27, $10,000 notional per trade, real
NBBO both sides, next-bar fills (`research/imb_stress_results.md:1-10`). It lives on shares and dies on
options (+16.46% → +0.70%, p 0.79 once fills are lagged a minute) because QQQ's spread is about 0.5 bp round
trip while an ATM 0DTE costs about 1% of premium plus theta (`:19-24`).

**The stress test** (`research/imb_stress_results.md:40-57`): +$3.34 per trade, CI [+1.75, +4.96], p = 0.0001;
first half +$2.92 (p 0.0040), second half +$3.78 (p 0.0004); 10 of 11 years positive; best year removed
+$2.85, worst year removed +$3.80; long +$3.51, short +$3.16; extra slippage of 2 cents per share per side
still p = 0.045 (breakeven 3.88 cents); best day 9.4% of the total; Sharpe 1.39, 62.5% of 128 months positive.
Cross-instrument: SPY 2022–2026 is again the only one of 13 setups to clear Holm (1,316 trades, +$2.60,
Holm 0.0087) but is correlated 0.717 with QQQ and adds nothing independent (the 120 days SPY fired and QQQ
did not lost $1,191) (`:155-201`). Persistence (2026-09-04): no decay (slope −0.038 $/trade/year, p 0.8745);
rolling mean negative 4.1% of the time in the last three years; every third positive; late third +$2.98 with
raw p 0.0280 but **Holm 0.1400** (does not clear); block bootstrap 7% *narrower* than day-clustered; and a
circular volatility-regime test was caught and redone (`research/persistence_results.md:14-137`). A noted
but un-acted observation: the last four annual rolling means fall 5.49 → 3.57 → 2.90 → 1.65
(`:40-43`); the walk-forward shows the same decay on the underlying (2022 +0.0699% → 2026 +0.0111%)
(`research/walkforward_results.md:28`).

**Why it survived.** It could have been killed by split-half, per-year, per-direction, slippage and
day-concentration tests, and was not (`research/imb_stress_results.md:138-142`).

**Why it is still not trusted.**
* Tail: top 1% (29 trades) = **73.9%** of total P&L; excluding them +$0.88 per trade, p = 0.1764; top 5%
  (145 trades) = 216% (the rest lose $11,308); median trade −$6.81; win rate 27.2%; skew 6.08
  (`:62-72`). A positive-skew trend strategy has its edge in the tail by construction, so this is not
  automatically disqualifying, but "29 trades is a thin reed to hang a decade on" (`:74-78`).
* Path: max drawdown $1,236 (12.4% of capital), 93.5% of trades spent below a prior equity peak, **796 days
  (26.2 months)** under water 2018-12 → 2021-03, 405 days 2025-05 → 2026-06 (`:84-95`).
* Size: **$9,717 over 10.60 years = $917/year = 9.2%/year on $10,000 = $18 per week**. The first number
  printed (15.7%/year) annualised on active days instead of calendar days and was wrong
  (`:28-34`). Trading QQQ and SPY both yields 6.2%/year on $20,000 (`:190-197`).
* A forward test cannot settle it: 10.6 years made 2,905 trades; one more year adds about 9% more
  information; the honest reason to run it forward is plumbing validation (clock, fills, restarts,
  outages), "not statistical proof" (`:143-151, 205-215`).
* Do not delete the trailing stop on the evidence that trailing exits lose $26,655 while end-of-day holds
  make $36,372: end-of-day trades are *selected* for never having gone against you (same path-versus-endpoint
  trap), and changing it would break the published, untuned parameters that make it credible (`:99-114`).
* The idea that MOMO_CHASE (the encoding of his own behaviour) could be automated instead: +$0.05 per
  trade over 5,586 trades on QQQ (p 0.889) and +$0.28 on SPY (`:134`, `:170-171`); closed again by the
  cluster sweep. "It trades nothing like him (27% win rate vs his 45.8%)"
  (`memory/imb-shares-stress-test.md:50-51`).

### 3.8 What genuinely remains testable, and what is ruled out by missing data (Q8)

**Source:** `research/remaining_testable_inventory.md`, compiled 2026-08-18 *before* the three-endpoint pass
and before timestamps arrived. Its own framing: 333 round trips can find a large *leak* (a behaviour costing
far more than $27 per trade), never an *edge* (`:24-28`). Its status per item, as best I can reconstruct it:

| item | inventory priority | what happened afterwards |
|---|---|---|
| A1/A2 day selection | LOW (he trades ~74% of sessions) | not run; "first-side"/B2 explicitly discarded before pre-registering (`research/entry_state_preregistration.md:250-257`) |
| B1 call/put better than a coin flip | LOW–MEDIUM (n≈46, MDE ±20 pts) | not run |
| B2 first side on two-sided days | LOW–MEDIUM (n=59) | discarded as subsumed/underpowered (same place) |
| C1 sigma-normalised strike distance | HIGH | **run, null** (`research/c1_f1_f2_results.md:36`) |
| C2 premium vs outcome | LOW, do not run | confounded by geometry |
| C3 his strike vs alternatives on same chain | BLOCKED (needs timestamps) | **run as construction Test A**: ATM +1.37%, p 0.118, Holm 0.355, underpowered (`research/construction_results.md:30-34`) |
| D1/E1 entry/exit timing | BLOCKED and CLOSED | timing re-opened by timestamps; entry-timing and journal-exit studies ran |
| D2 averaging down | DO NOT RUN — leaks | still unquotable (outcome-path grouping) |
| E2 exit vs hold-to-expiry | LOW, already known | skip |
| E3 tranched exits | LOW, do not run | skip |
| F1 position size | MEDIUM–HIGH | **run**: scale effect only |
| F2 within-day ordinal | MEDIUM–HIGH | **run, null** |
| G1 pre-open regime → outcome | MEDIUM (multiplicity hazard: pre-register exactly one variable) | not run on his trades; the day-type study (2026-10-01) tested regime variables on the mechanical record instead (`memory/day-type-study.md:13-14`) |
| G2 does he choose regimes | LOW | not run |

So the inventory's "smallest set" (C1, F2, F1) is exhausted. What is left on *his* record: B1, B2, A1/A2, G1 —
all low-power — plus the two **open** items: the time-of-day P&L pattern (needs pre-registration and
day-clustering, `CLAUDE.md:152-160`) and the **unwritten selection rule** (92% of sessions break a line by
09:40; what picks *which* break he takes has never been written down, `CLAUDE.md:116-120`). The entry-state
pre-registration (duration-matched percentile + one pre-entry state variable, 2026-08-20) has **no results
file**, its header still says "NOT YET RUN" (`research/entry_state_preregistration.md:3`) and its code is
helper functions only (`trade_analysis/backtesting/broker_entry_state_study.py:58-199`, no `main`): status
unfinished or unrecorded.

**Blocked outright or ruled out by missing data:**
* Out-of-sample validation of anything fitted on his record: seven months to ~17 months, one regime, "no
  held-out period by construction" (`research/remaining_testable_inventory.md:279-284`).
* The value of abstention (days he skipped): "unidentified in principle" — the data holds only the treated arm
  (`research/c1_f1_f2_results.md:129-131`).
* Single-name options (no entitlement) (`research/percentile_return_preregistration.md:47-50`).
* QQQ options of any expiry (no archive), any 1DTE on any symbol (archive is 0DTE by construction), anything
  new from ThetaData (lapsed to `Options: FREE` on 2026-09-08); the only option archive is SPY 0DTE, 769
  sessions, 2020–2024, and the 2024 block "can be spent exactly once" (`CLAUDE.md:69-76`;
  `research/PROGRAMME_REVIEW_2026-09-24.md:346-353`). The 2024 held-out block was never touched in the VRP work
  (`research/vrp_cost_model_results.md:4-6`).
* Index data intraday (SPX, VIX, VIX1D): the 403 on the free tier (`.claude/agents/quantitative-researcher.md:116-121`).
* An honest, power-based statement: resolving a $10-per-contract edge needs about 2,357 round trips; only
  collecting trades moves it (`research/remaining_testable_inventory.md:314-318`).

### 3.9 How the owner wants results communicated, and standing corrections (Q9)

From `CLAUDE.md:3-23` (verbatim-level paraphrase):
* Plain English before jargon, always; he has said repeatedly that explanations are harder than needed, which he
  treats as a defect in the explanation, not in him.
* Lead with the conclusion in one sentence. No table or preamble first.
* Everyday comparisons: "0.54 bp on a $700 stock is about 4 cents" beats "+0.54 bp, CI [−1.70, +2.87]". Cents
  first, bp second.
* Define each term the first time in a reply (bp, MFE, drift, CI, day-clustered).
* Short — five lines if it fits in five.
* Do not relitigate: cite the one line and move on.
* **Do not keep proposing new instruments.** He trades naked calls and puts; no straddles, no spreads. If the data says
  calls and puts won't work, say so rather than offering a consolation structure.
* Metric units; no apologies, fluff, or "great question".

**Standing corrections he has given** (each is in `CLAUDE.md`):
1. Do not say "he takes 1–2 trades around 11:39." Both numbers were wrong (journal is ~2.8–2.9 trades/day) and
   11:39 is a median, not a cluster; "never quote a median as a location without the histogram"
   (`CLAUDE.md:122-132`). Measured shape: busiest bucket 09:45, median 11:44, "one peak plus a noisy declining
   plateau", not two humps (2.04 sigma) (`:134-142`). Corrections recorded in
   `research/breakout_options_results.md:114-142` and `research/six_lines_results.md:99-106`.
2. A true number can support a false conclusion; never read a path statement as an endpoint statement
   (`CLAUDE.md:86-93`).
3. A gate that cannot pass in its environment is broken, not strict; platform checks are SKIP, never FAIL
   (`CLAUDE.md:97-98`). Every test must be verified to fail against the broken version (`:94-96`).
4. Wrong numbers are corrected in place with the reason, never deleted (`CLAUDE.md:99-101`).
5. Do not treat the time-of-day P&L pattern as a finding and do not let him trade it (`CLAUDE.md:144-160`).
6. Do not re-run settled work; do not offer measurements that cannot be made (`CLAUDE.md:25-27, 69-76`).
7. The six-line family is closed; do not re-open without a genuinely new hypothesis and fresh data (`CLAUDE.md:41-42`).

---

## 4. Map

### 4.1 Master table of every study/document in `research/` (Q2)

Built from the files themselves. Verdict vocabulary: **null** (pre-registered test, nothing found),
**real** (survived), **void** (invalidated by a defect), **fragile** (cleared a gate but not robust),
**open** (unfinished or unrecorded), **infra** (operational, not an experiment), **backfill**
(computed after the fact, never counted). "Others cover" = detailed by another agent; I give only the
one-line verdict.

**A. Studies of his own trading record**

| # | file(s) | date | question in plain words | result (headline) | verdict |
|---|---|---|---|---|---|
| 1 | `percentile_return_{preregistration,results}.md` | 2026-08-18 | Do his entry/exit minutes beat random minutes in the same contract on the same day? | mean percentile 0.5522 [0.5143, 0.5885] over 266 ETF trades, but only 0.5057 (d=10) / 0.5151 (d=20) when hold time is matched; `d=close` 0.6895 | null (verdict B); the headline was early-exit policy, not timing skill (`:98-108`) |
| 2 | `remaining_testable_inventory.md` | 2026-08-18 | What can still be tested on 333 round trips? | smallest set: C1, F2, F1; "none of them can establish an edge" | scoping doc, not a study (`:314-318`) |
| 3 | `c1_f1_f2_{preregistration,results}.md` | 2026-08-18 | Do strike distance (C1), size (F1) or within-day order (F2) predict his dollar outcome? | C1 −1.00 (Holm 1.0); F1 −2.03 (Holm 0.048, scale-only); F2 +0.30 (Holm 1.0) | null / scale-only |
| 4 | `entry_state_preregistration.md` | 2026-08-20 | Duration-matched entry percentile + one prior-15-min move variable on the 443 timestamped trips | **no results file**; header "NOT YET RUN"; code has helpers only | **open / unrecorded** |
| 5 | `indicator_state_{preregistration,results}.md` | 2026-08-20 | Does his own declared MACD/DMI/ADX/volume stack separate his outcomes? | OOS R2 −0.0623 (p 0.835), AUC 0.392; all-five-gates −8.26% vs +2.69% (p 0.163) | null |
| 6 | `setup_crosscheck_{preregistration,results}.md` | 2026-08-20 | Do ten public setups' matches predict his outcomes? | coverage 91.4% (saturated); 0 of 7 clear Holm; ORB continuation +9.49% raw p 0.107 | null |
| 7 | `setup_sweep_{preregistration,results}.md` | 2026-08-20 | Do 23 simple setups separate his trades? | 0 raw p < 0.05 (1.2 expected by chance), 0 Holm, 0 BH | null |
| 8 | `setup_standalone_preregistration.md` | 2026-08-20 | Do ten setups carry drift on all sessions, independent of him? | **no results file** here; second-hand: 0 of 10, 1,542 sessions x 2 symbols | null (quoted second-hand); header stale "NOT YET RUN" |
| 9 | `construction_{preregistration,results}.md` | 2026-08-20 / run 08-26 | Is his strike or his exit beating mechanical alternatives? | Test B: +25% target −8.97 pp, Holm 0.027 (the one clear); Test A ATM +1.37%, p 0.118 | **real → later fragile** (see #12) |
| 10 | `entry_timing_{preregistration,results}.md` | 2026-08-26 | Is his entry second better than a random second on the same contract? | C-skill VOID (own lookahead); post-entry control CIs straddle 0; C-money best +$4.93/week | void (C-skill) / null (C-money) |
| 11 | `package_{preregistration,results}.md` | 2026-08-26 | Do free public setups used as filters raise his weekly dollars? | 0 of 10; ORB cont. +55% $/trade but −$1.87/week; NBBO overstates his cash by +$6.85/trade | null |
| 12 | `journal_reverse_engineering_{preregistration,results}.md` | 2026-08-29 | Can ML recover his entry rule, does it pay, can his exit be beaten or copied? | AUC 0.847 (p 0.002); blind rule −0.15 bp; full model CI upper +2.34 bp vs +5; exit 0 of 14; kernel 0 of 10 | rule recoverable; rule null; Test B fragile |
| — | `pull_order_history.py` | by 2026-08-20 | tool, not a study | writes `order_history_raw.json` | tool |

**B. Shares candidate and the walk-forward**

| # | file(s) | date | question | result | verdict |
|---|---|---|---|---|---|
| 13 | `walkforward_results.md` | 2026-08-28 | Does any of 13 frozen setups call direction / pay on real 0DTE quotes (2022–26)? | 1 of 13 (IMB) clears on the underlying; on options 95.7% of IMB edge was lookahead: +16.46% → +0.70%, p 0.79 | null on options; void for the lookahead version |
| 14 | `imb_stress_results.md` | 2026-08-29 | Does IMB survive a stress test? | +$3.34/trade; passes structure, fails tail (top 1% = 73.9%); $18/week | **real but untrusted** |
| 15 | `persistence_{preregistration,results}.md` | 2026-09-04 | Is the IMB edge decaying or regime-bound? | no decay (p 0.87); late era raw 0.028 / Holm 0.14 | persistent (qualified) |
| 16 | `shares_lab_preregistration.md` | 2026-08-31 | Why add a shares arm to the lab | specification (others cover) | infra |
| 17 | `spread_survey_results.md`, `runner_capacity_results.md` | 2026-09-02, 09-05 | Which names can carry a 3–4 bp edge; how many symbols can the feed handle | 16 of 51 names untradeable on spread; capacity ~5 → ~29 symbols, 15 names run | infra (others cover) |
| 18 | `momo_v2_results.md` | 2026-09-13 | Is MOMO_CHASE just mis-specified (V1 +ATR gate, V2 1-minute DMI)? | V0 +$0.03, V1 −$0.03, V2 −$0.09 per trade; none beats +$1.00 and Holm | null |
| 19 | `momo_sweep_{preregistration,results}.md`, `momo_sweep_precondition3.md` | 2026-09-13/16 | Does any of 2,822,400 MOMO_CHASE parameter cells beat not trading? | SPA p(consistent) 0.8227; max t +4.0962 | null; form closed (others cover) |

**C. Variance (HPC) — others cover**

| # | file(s) | date | question | result | verdict |
|---|---|---|---|---|---|
| 20 | `vrp_preregistration.md`, `vrp_baseline_results.md` | 2026-09-08/09 | Can intraday variance be forecast well enough to price the 0DTE premium? | HAR-RV QLIKE 0.430642 is the bar; implied variance vs HAR: p 0.5832 | null |
| 21 | `vrp_cost_model_results.md` | 2026-09-09 | Is the short ATM 0DTE straddle premium tradable? | gross +$0.0205 (t +7.52), net −$0.0171 (t −5.38); 1.84x too small | null (decisive) |
| 22 | `vrp_phase0_concentration.md`, `vrp_harp_specification.md` | 2026-09-11 | Is the IV-vs-HAR gap robust; does a time-of-day term help? | t −2.30 was an artefact (missing exp(s²/2)); true t −0.55; diurnal terms buy 1.3–1.6% | superseded premise / null |
| 23 | `conditional_0dte_{preregistration,armA_results,armB_results}.md` | 2026-09-11 | Is the premium harvestable on a pre-selected subset? | Arm A failed to replicate (bias 4.6x the signal); Arm B +$0.01002/trade, session t +2.05, 2023 only, 2024 sealed | A void, B **fragile** |

**D. Levels / breakouts — others cover**

| # | file(s) | date | question | result | verdict |
|---|---|---|---|---|---|
| 24 | `breakout_options_{preregistration,results}.md` | 2026-09-19 | 8-line breakout on QQQ | +1.44 bp [−0.97, +3.91], cap worth −1.17 bp; wrong level set | null, superseded |
| 25 | `six_lines_results.md` | 2026-09-21 | His exact 6-line breakout | 2,438 trades, +0.54 bp [−1.70, +2.87] p 0.611; capped −0.18 | null |
| 26 | `after_break_{preregistration,results}.md` | 2026-09-23 | What happens after R1/S1 breaks early; round numbers | R1 early→R2 70.7% vs 59.3%; time-matched +11.3 pp p 0.002; round numbers flat; S1 null | **real (only conditional signal)**; no money |
| 27 | `early_r1_hold_{preregistration,results}.md` | 2026-09-23 | Does removing the stop help? | +1.84 bp [−13.02, +16.81]; all 3 gates fail | null |
| 28 | `orb_veto_preregistration.md` | 2026-09-17 | Does a close-based range-expansion veto improve the frozen shares family? | code `orb_veto*.py` exists; **no results file**; header "NOT YET RUN" | **open / unrecorded** |
| 29 | `six_lines_backfill_results.md` | 2026-09-25 | The six lines replayed over the lab's first 19 sessions | 26 trades; Six_Lines −$156.10, NoCap +$1,533.90 (one day = 68%) | backfill |

**E. Forward test and operations (others cover)** — `forward_test_preregistration.md` (frozen 2026-08-28), `live_lab_specification.md` (2026-08-27),
`live_lab_checkpoint_100_diagnostic.md` (2026-09-16), `live_lab_coverage_audit.md` (2026-09-16),
`replay_22_sessions.md` (2026-09-29, backfill), `network_resilience.md` (2026-09-28), three
`incident_2026-09-{15,24,25}_*.md`, `runner_capacity_results.md`.

**F. Programme documents** — `PROJECT_REPORT.md` (2026-09-10), `PROGRAMME_REVIEW_2026-09-24.md`,
`vendor_and_method_surveys/` (four web-research reports: `0dte_expression_survey.md`,
`research_dependence_methods.md`, `research_hpc_setup.md`, `research_variance_sota.md`; the first says earlier
surveys had been lost from scratchpads, `0dte_expression_survey.md:5-8`).

### 4.2 Code in my scope (all under `trade_analysis/`)

| file | lines | purpose |
|---|---|---|
| `backtesting/journal_features.py` | 558 | builds the 51-feature entry/placebo table (`main` at :433) |
| `backtesting/journal_ml.py` | 387 | Q1/Q2/M2/M3 with permutation nulls |
| `backtesting/journal_zone_wf.py` | 248 | Q3 blind walk-forward of the recovered zone |
| `backtesting/journal_model_wf.py` / `journal_model_report.py` | 250 / 93 | Q3b scoring of 721,887 QQQ minutes; report from cache |
| `backtesting/journal_exit.py` | 326 | Q4a/Q4b exit study on real NBBO (FEE 0.0404, BOOT 10000) |
| `backtesting/journal_kernel.py` | 298 | M4 Lo-Mamaysky-Wang kernel patterns |
| `backtesting/broker_entry_state_study.py` | 199 | helpers for the unfinished entry_state study |
| `backtesting/entry_histogram.py` (+`_test.py`) | — | entry-time histogram; descriptive only |
| `backtesting/imb_stress.py` / `persistence.py` | 254 / 209 | IMB stress and persistence tests |
| `trade_journal/robinhood_trade_analyzer.py` | 702 | legacy CSV analyzer (Alpaca + yfinance); **do not reuse** (see 3.3) |
| `trade_journal/trade_journal_analyzer.py` | 794 | legacy v1.0 indicator-at-trade-timestamp analyzer; edit a TRADES list then run |
| `trade_journal/trade_input_parser.py` | 224 | converts a pasted plain-text trade log to the analyzer's input |
| `research/pull_order_history.py` | 109 | one-off Robinhood order-history pull |

### 4.3 The three specialist reviewers the owner set up (`.claude/agents/`, committed 2026-08-31 in `b4c33b0`)

| agent | role | tells you about how research was policed |
|---|---|---|
| `quantitative-researcher.md` (212 lines, opus) | independent researcher; **not** a tuner. Audit what was tested, generate structurally different hypotheses, six-part protocol (mechanism, prediction, exact experiment, falsifier, minimum sample, exploratory vs confirmatory), bias-hunt; "a well-supported negative is a successful outcome" (`:5-6, 33-52, 136-166, 209-212`) | forbids parameter tuning ("same mechanism, same target, different constant") and "re-run a failed experiment" without a stated methodological reason (`:13-31, 73-76`) |
| `statistical-auditor.md` (224 lines, opus) | verifies a backtest before it is believed; **never** optimises or proposes strategy. Verdict PASS / FAIL / QUESTIONABLE per claim; "Verify, do not trust": PASS only if the auditor reproduced the headline number itself (`:15-47`); checklist A–E: time integrity, data integrity, sample construction, inference, validity | exists "so that does not happen twice" (`:10-13`); lists the contaminated modules inheriting `find_entries` (`:49-71`) |
| `strategy-contrarian.md` (188 lines, opus) | red team: assume the interpretation of the setup is wrong; nine standing questions (direction vs volatility, MFE vs terminal return, timing vs sign, wrong option expression, avoid-signal, mean reversion...); every alternative needs mechanism, prediction, test, falsifier, sample size, cost-to-kill; "attack the premise": has anyone checked he makes money against broker statements? (`:36-40`) | its question 2/3 (levels may predict range, not sign) is what the programme later measured four times as "magnitude, not sign" |

Together: researcher proposes and pre-registers, contrarian attacks, auditor reproduces. The prompts also
carry tool constraints that contradict later work: "Buy-only. The trader will not sell an option contract
under any circumstance" (`quantitative-researcher.md:113-115`) while the VRP studies *measure* a short
straddle, and the contrarian lists long straddles as "allowed" (`strategy-contrarian.md:117`) while
`CLAUDE.md:20-22` now says he trades only naked calls and puts. Where the day-type study used "5 parallel agents
+ an auditor" the auditor was this pattern (`memory/day-type-study.md:11`); `research/vrp_phase0_concentration.md:109`
is the only research file that names an auditor.

### 4.4 Memory notes in my scope (all in `memory/`, all dated, all possibly stale)

`research-programme-verdict.md` (2026-08-27, 45 days old), `journal-reverse-engineering.md` (08-29),
`robinhood-broker-dataset.md` (08-20, 53 days old, **partly superseded** by the timestamp pull),
`imb-shares-stress-test.md` (08-29), `day-type-study.md` (10-01), `moves-vs-turns-analysis.md` (09-24),
`profitbook-0dte-options-goal.md` (07-26), plus the earlier-phase notes `0dte-real-data-findings.md`
(**VOID headline numbers**, kept for methods), `underlying-orb-decade-test.md` (**VOID**),
`feedback-prefer-simple-orb-strategy.md`, `feedback-isolate-experimental-strategy-code.md`.

---

## 5. Design decisions and the reason for each

| decision | reason (evidence) |
|---|---|
| Day-clustered bootstrap, 10,000 resamples of whole dates | same-day trades share regime and direction; design effect 1.31–1.43; the earlier "9.4x" was retracted (`research/c1_f1_f2_preregistration.md:123-127`; `.claude/agents/quantitative-researcher.md:156-161`) |
| Dollar endpoints for decisions, percent as a robustness cut | "Percent space lies here": t = 1.77 vs 0.74 on the same trades (`memory/robinhood-broker-dataset.md:49-52`); package test uses $/week because his goal is $200–500/week and a filter can raise $/trade while halving income (`research/package_preregistration.md:14-25`) |
| Holm family size never reduced when a member fails to run | lowering m because a sibling failed would make survivors pass more easily (`research/construction_preregistration.md:130-135`) |
| Report the full volume-versus-quality table whatever the p-values | so a null cannot be read as expected and the table cannot be read as a discovery (`research/package_preregistration.md:27-39`) |
| Placebos drawn from the same session with the same direction | so features are signed identically for real and placebo rows; a classifier separating them on direction bookkeeping alone would be an artefact (`research/journal_reverse_engineering_preregistration.md:48-53`) |
| Six trade-only features barred from the selection model | a model could win on NaN-ness alone and score AUC 1.0 (`research/journal_reverse_engineering_results.md:31-35`) |
| Outcome reported twice: exit-free forward move and his money | his realised return embeds the one skill-like effect (exit); a model could mistake exit skill for entry quality (`research/journal_reverse_engineering_preregistration.md:25-31`) |
| Fills lagged one full minute on both legs | the rule that moved IMB from p = 0.0003 to p = 0.79; "not optional" (`trade_analysis/backtesting/journal_zone_wf.py:8-15`) |
| Single names excluded from the strike test, included in size/ordinal | strike distance in daily sigma is not comparable across 0–7 day expiries (23% vs 94% 0DTE), and choosing a time-scaling after seeing the mix is the post-hoc search that is prohibited (`research/c1_f1_f2_preregistration.md:38-45`) |
| Annualise on calendar time | IMB fires on 1,560 of ~2,657 sessions; capital sits idle the rest; 15.7% → 9.2%/year (`research/imb_stress_results.md:28-34`) |
| Interpolate the fill inside the entry bar | he enters 31.9 s into the next bar, which moves +1.905 bp his way (`research/journal_reverse_engineering_results.md:93-107`) |
| Day-level, not trade-level, "n" for power | the sample is 158 days, not 443 independent trades (`research/c1_f1_f2_results.md:129-133`) |
| New experimental code in new files that only import proven code | standing instruction (`memory/feedback-isolate-experimental-strategy-code.md`; `.claude/agents/quantitative-researcher.md:131-133`); the journal modules say "Isolated experimental code" (`journal_features.py:24`) and `broker_entry_state_study.py:3` says "SAFE TO DELETE" |
| Quote dollars of NBBO with a haircut | NBBO simulation is +$6.85 per trade optimistic against his realised cash (`research/package_results.md:117-145`) |

## 6. Records, formats, invariants, frozen things

* **Frozen:** `live_lab_data/` and its config hashes (`CLAUDE.md:108-110`); the forward-test freeze
  `research/forward_test_preregistration.md:3-12`. Nothing in the journal studies touches it: "Nothing here
  modifies the frozen forward test... A discovery here can only propose a new `setup_id` at count zero"
  (`research/journal_reverse_engineering_preregistration.md:185-187`).
* **Frozen samples per study:** e.g. 443 → 418 + 3,274 placebos + 151 date clusters, "no row and no feature may
  be added, dropped, or redefined" (`research/journal_reverse_engineering_preregistration.md:48-49`); seed
  20260829 throughout (`journal_features.py:52`); IMB 2,905 trades frozen for the persistence study
  (`research/persistence_preregistration.md:30-36`).
* **Pre-registration texts are kept verbatim** after the run; amendments appended
  (see 3.2). Do not edit them to match results.
* **Held-out data spent at most once.** The VRP 2024 block and the MOMO 2022–2026 held-out set "was never
  touched and must not be" (`research/momo_sweep_results.md:20-24`).
* **Personal data never committed:** `research/order_history_raw.json/.txt` and `order_history_raw.*`
  gitignored (`.gitignore:81-83`); `*.csv` gitignored. The journal CSV is lab-machine-only.
* **Credentials:** the pull script takes none as arguments (`pull_order_history.py:20-25`); the legacy
  analyzers read `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` from the environment
  (`trade_analysis/trade_journal/robinhood_trade_analyzer.py:39-45`, same in `trade_journal_analyzer.py`) —
  variable names only; I did not read any value.
* **Hard-coded scratchpad paths:** the journal modules read and write pickles under
  `C:\Users\CHAITA~1\AppData\Local\Temp\claude\C--Users-chaitanyakharche-Documents-stock\3ad3e562-...\scratchpad`
  and read `bars_long.pkl` from `...\8e8c7ab5-...\scratchpad` (`journal_features.py:40-48`,
  `journal_ml.py` `SCRATCH`, `journal_exit.py` `SCRATCH`, `broker_entry_state_study.py:48-52`). Temp
  directories are disposable; if they have been purged the pipelines must rebuild their caches (inferred).
* **Run environment:** memory says run the journal modules with `./venv/Scripts/python.exe` because
  "system python has no sklearn" (`memory/journal-reverse-engineering.md:14-15`), while `CLAUDE.md:105-106`
  says the lab runs on pyenv 3.12.10, not the repo venv. The journal modules import scikit-learn and scipy
  (`journal_ml.py:31-45`, `journal_kernel.py:30`).
* **Standing rule on results documents:** a true number may not stand for a false conclusion (`CLAUDE.md:86-93`).

## 7. Failure modes, incidents, traps

The full table of defects is `research/PROGRAMME_REVIEW_2026-09-24.md:243-304`; those touching my slice:

1. **Label-vs-close lookahead** (voided two result sets; `find_entries`, `options_premium_backtest.py:152-158`).
   The agents list modules still inheriting it: `spy_qqq_0dte_real_backtest.py`, `retest_vs_gapandgo_backtest.py`,
   `options_premium_backtest_scalp.py`, `options_premium_backtest_confluence.py`, `underlying_orb_longrun.py`
   (`.claude/agents/quantitative-researcher.md:94-96`; a snapshot of 2026-08-18, may have changed).
2. **Holiday zero-fill**: closed days return a full zero-filled grid and are not `.empty`; about 7% of entries were
   fabrications (`memory/0dte-real-data-findings.md:73-77`). Also `min()` over a session can silently return 0.00,
   and the 09:30 row is all-zero and the 16:00 row a settlement stub
   (`research/percentile_return_preregistration.md:29-35`).
3. **Sign-flipped broker fields.** `client_bid/ask_at_submission` is sign-negated on 289 of 1,037 orders (all
   sells; repair by negate-and-swap); the claim "buys land at full ask" was a tick-quantisation artefact — report
   cents, never the ratio (`research/entry_state_preregistration.md:15-27`). The broker fields also remain
   unusable for execution-quality claims (`research/construction_results.md:107-112`).
4. **Aggregation leaks in the broker CSV.** A prior "+$2,830.92, 249 round trips, t = 1.07" had a $489.80 leak and
   is refuted; identical rows are distinct fills so dedupe by multiset, not unique rows (row-level dedupe cut the
   sample to 274); parse with `csv.reader` dropping `len(row) != 9` (`memory/robinhood-broker-dataset.md:20-31, 41-47`).
5. **Duration policy mistaken for timing skill** (percentile unconstrained 0.5522 vs matched 0.506; `d=close`
   0.6895) (`research/percentile_return_results.md:56-76`).
6. **Control arm with lookahead** and a win-rate computed on 20-draw averages (`research/entry_timing_results.md:42-78, 227-235`).
7. **Leaky groupings**: averaging-down (grouping variable is a function of the outcome path), and entries-per-day
   *total* as a regressor (known only at the close) — the $14.47 → $3.36 gradient is retired
   (`research/remaining_testable_inventory.md:159-172`; `research/c1_f1_f2_results.md:102-106`).
8. **Circular regime test** (IMB T5 bucketed on the trades' own |return|) produced a false "long-volatility bet"
   story; redone on prior-20-day volatility (`research/persistence_results.md:84-115`).
9. **Test B sign flip** between return space and dollar space (3.6).
10. **Under-powered gate wording**: BH null-importance could not pass by construction (`research/journal_reverse_engineering_results.md:326-331`).
11. **The median taken as a location**: besides the research text, the frozen `MOMO_CHASE` setup anchors its
    entry window on the 11:39 median ("window": "his median entry 11:39 ET",
    `trade_analysis/live_lab/setups.py:164`; params at :159 "10:30-14:30"). Since the histogram shows the busiest
    bucket is 09:45, the window encodes the exact assumption the 2026-09-22 correction warned about.
12. **1-minute vs 5-minute DMI mismatch**: the discriminating feature is 1-minute (AUC 0.806) while `MOMO_CHASE`
    gates on 5-minute (0.634) (`research/journal_reverse_engineering_results.md:83-85`); testing it
    (`momo_v2` V2) was null (`research/momo_v2_results.md:24`).
13. **Pre-registrations that sat unrun or left stale** (section 9 item 5).
14. **MFE gap is arithmetic, not opportunity**: a mean $81.68 MFE vs $0.78 realised is "a large, interpretable,
    intuitive number that no implementable rule can convert" (`research/journal_reverse_engineering_results.md:261-264`).
15. **Unrecomputed figure**: the previously recorded "+0.0254%/trade" underlying capture was flagged suspect and
    its recomputation is not recorded (`research/journal_reverse_engineering_results.md:113-116`;
    `memory/journal-reverse-engineering.md:42-46`).

**What would bite whoever touches this next:** (a) quoting any figure from `0dte-real-data-findings.md` or
`underlying-orb-decade-test.md` (void); (b) re-using the legacy `trade_journal/` analyzers; (c) quoting
"11:39" as a location; (d) quoting the averaging-down figures; (e) reading a statement about price path as a
statement about the endpoint; (f) re-running settled endpoints on the larger CSV ("the extra 15 round trips
changed nothing", `memory/robinhood-broker-dataset.md:20-27`); (g) pricing at NBBO without the $6.85 haircut.

## 8. Current state

* **Done:** all eight direction nulls; journal reverse-engineering (Q1, Q3, Q3b, Q4a/b, M4, Q2); IMB stress and
  persistence; package; construction; entry timing; time-of-day histogram (descriptive); day-type study; moves-vs-turns.
* **Half-done / unrecorded:** the entry_state study (no results); orb_veto (no results file); the 19-feature
  ceiling model and the ten-setup standalone result exist only as second-hand numbers; the "+0.0254%" recompute.
* **Open and explicitly "not a finding":** the time-of-day P&L alternation, the unwritten selection rule,
  the qualitative question of whether a 4-point monotone IMB decline is real.
* **The decisive constraint:** about 1.5 more years of trading (356 more QQQ days) to reach t = 2 on his tape;
  the live lab is the only generator of new data; the 50 and 100 diagnostic checkpoints passed unrecorded and the first
  serious read is 200 trades per setup (`research/live_lab_checkpoint_100_diagnostic.md:9-17`).
* **State as of the last review (2026-09-24):** forward test down about $4,415 on paper (options +$2,397, shares
  −$6,812); 185 passing tests; "one real conditional signal and no tradeable edge"
  (`research/PROGRAMME_REVIEW_2026-09-24.md:375-380`). Later figures (10-01: shares −$12,452.77, combined
  −$10,825.27; `memory/day-type-study.md:16-17`) belong to the live-lab reports.
* **Non-code open items he must handle:** an HF token exposed in `~/.bash_history` on the shared cluster filesystem
  (revocation unconfirmed — highest priority, `CLAUDE.md:111-113`); firewall rules exposing Theta Terminal port 25503
  (`research/PROJECT_REPORT.md:392-394`); whether the Robinhood session token was deleted (not recorded).

## 9. Contradictions (documents vs files, and documents vs documents)

1. **`CLAUDE.md:32` points the MOMO_CHASE 2,822,400-cell result at `setup_sweep_results.md`**; the numbers
   (SPA p = 0.8227, 2,822,400 cells) are in `research/momo_sweep_results.md:9-24`. `setup_sweep_results.md` is
   the 2026-08-20 24-setup sweep on his own trades (23 setups, 0 raw p < 0.05). `CLAUDE.md:157` repeats the
   wrong pointer. (A grep of 2,822,400 and 0.8227 over `research/*.md` shows neither string in `setup_sweep_results.md`.)
2. **"73.9% in the top 1% of trades" is attributed to the journal** (`CLAUDE.md:159`). I found 73.9% only as IMB's
   share (`research/imb_stress_results.md:67`, `research/PROJECT_REPORT.md:60`). The journal's own recorded tail
   figure is different: removing the top 5 of 333 activity-CSV contracts turns +$2,341 into −$393
   (`memory/robinhood-broker-dataset.md:46-47`). I cannot compute the journal's top-1% share without the CSV. Likely
   a mis-transcription.
3. **"journal, 422 discretionary trades, dollars: +25% target → 71.6% win rate, still loses (breakeven 75.3%)"**
   (`CLAUDE.md:61`, `research/PROGRAMME_REVIEW_2026-09-24.md:215`, `research/six_lines_results.md:75-79`). The
   71.6% win rate, the −$2,347.76 and the 75.3% breakeven are from the **296-trade** construction Test B
   (`research/construction_results.md:6-13, 53-58`). The **422-trade** dollar reconstruction says the opposite
   at its point estimate: the +25% rule gains **+$4.22 per trade** (p 0.487) over his exit
   (`research/journal_reverse_engineering_results.md:218, 233-237`); the two also differ in fall-through
   (15:45 bid vs his own exit). So the first of "three datasets" for "the cap subtracts" is not a 422-trade
   result and the later 422-trade measurement points the other way. The other two legs (8-line −1.17 bp, 6-line
   −0.72 bp) are in other agents' slices.
4. **2.90 trades/day.** `CLAUDE.md:122-136` and `research/breakout_options_results.md:127` give 357/123 = 2.90,
   and say the CSV "has grown" to explain 2.80. Both counts are in the same file on the same date:
   `research/entry_timing_results.md:159-162` has all symbols n=443, days=158 and QQQ/SPY n=357, **days=126**;
   the 123 days belong to the 339 *evaluated* trips (`:17-21`). Like-for-like is 357/126 = 2.83 (my arithmetic).
   The conclusion (about 2.8–2.9 per day, not 1–2) is unaffected; the reason given is not what the files show.
5. **Stale "NOT YET RUN" headers.** The 2026-09-24 review says nine pre-registrations had this problem and implies
   it was fixed (`research/PROGRAMME_REVIEW_2026-09-24.md:283-285`). Four still carry it: `entry_state_preregistration.md:3`
   (no results file), `momo_sweep_preregistration.md:3` (results exist), `orb_veto_preregistration.md:3` (no results
   file), `setup_standalone_preregistration.md:3` (result quoted second-hand).
6. **"Test B" is stated as established in older notes and CLAUDE-era summaries** (`memory/research-programme-verdict.md:20-22`
   already says FRAGILE) but `CLAUDE.md:114-121` says "the exits carry his record" without the fragility caveat.
7. **Entry timing is "blocked on execution timestamps" in the inventory** (`research/remaining_testable_inventory.md:151-158,
   272-277`) and in `memory/robinhood-broker-dataset.md:71-78`, but timestamps were pulled by 2026-08-20 and used
   from then on (3.3). Those statements are historically true and no longer current.
8. **`PROJECT_REPORT.md:41-47` still says "median entry 11:39 ET" as the headline description** with no histogram
   caveat; the correction is in `CLAUDE.md:122-142` and two result files, not in the report.
9. **Eight nulls vs actual counting**: three different "consecutive" counts in the files (3.5 note a); two of the
   eight have no results file in `research/`.
10. **Book total**: `CLAUDE.md:145` "+$1,502 over 442 trades = +$3.40/trade" vs the research files' "$1,452.82 over 443"
    (`research/entry_state_preregistration.md:31-34`, `research/entry_timing_results.md:159, 166`); `memory/moves-vs-turns-analysis.md:15`
    says the $1,502 omits one post-close trade.
11. **Test counts**: `CLAUDE.md:104` "~123 checks" vs "185 passing" in `research/PROGRAMME_REVIEW_2026-09-24.md:136-137`.
    Not in my slice; noticed only.
12. **Agents' buy-only constraint vs later work**: see 4.3.

## 10. Questions only the owner can answer

1. Where exactly do `research/round_trips.csv`, `research/order_history_raw.json` and `2025 complete trades.csv`
   live, and are they backed up? They are gitignored and "only exist on the lab machine".
2. Did you delete the Robinhood session token (`~/.tokens/robinhood.pickle`) after the pull?
3. Are the old scratchpad directories (`3ad3e562-...`, `8e8c7ab5-...`) still on disk? They hold the feature table,
   `bars_long.pkl` and the option-quote caches the journal modules need.
4. Where are the results of the 19-feature ceiling model and the ten standalone setups (0 of 10, 1,542 sessions)? And was the
   entry_state study (duration-matched percentile + prior-15-minute-move) ever run? And what did the `orb_veto` backtest return?
5. Is the journal still being extended (new Robinhood pulls after 2026-02-20)? The only thing the programme says
   moves the answer is more trades.
6. Is $200–500 per week still the objective the studies are judged against?
7. Did you or the assistant first propose pre-registration, and where is "the user's own DO-NOT list" that
   `research/entry_timing_results.md:176` cites?
8. Which of the section-9 CLAUDE.md numbers (the 73.9% attribution, the 422/296 mix-up, the setup_sweep pointer, the
   2.90 reason) do you want corrected in place?
9. Do you want the `MOMO_CHASE` entry window re-examined in light of the entry-time histogram, given that the setup
   is frozen in the forward test (so any change is a new setup at count zero)?
10. The synthetic-pricing note `options-premium-synthetic-backtest` (cited in a memory note) does not exist in the memory index; was it deleted?
