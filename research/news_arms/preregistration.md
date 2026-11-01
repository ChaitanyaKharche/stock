---
task_id: R-2026-11-01-01
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
parent: research/news_impact/results.md (R-2026-10-30-08)
role: orchestrator
repo_sha: 3d21893
inputs:
  - "owner's brief 2026-10-31: seven questions; eight conditions 2026-11-01"
  - research/news_impact/events_scheduled.csv@550b4a5      (516 rows, column `session`)
  - research/news_impact/events_geopolitical.csv@3facddc    (321 rows, frozen, `break_time_et`)
  - research/replay_22_sessions.md@3d21893                   (live vs replay fidelity, measured)
  - trade_analysis/live_lab/backfill.py@3d21893              (HistoryFeed, SimClock, run)
  - live_lab_data/@3d21893                                   (44 live sessions, exploratory only)
decision_rule: "section 7; two confirmatory tests, everything else exploratory until forward validation"
outputs:
  - research/news_arms/audit_preregistration.md   (gate 1, before any replay)
  - live_lab_backfill/news_arms/{shares,options}/  (replay outputs, provenance BACKFILL)
  - research/news_arms/join.csv, analysis.json, results.md, audit_results.md (gate 2)
assumptions: "section 9, every one listed with whether it can be checked against live"
open_questions:
  - "replay runtime per session on the laptop (timing run requested)"
  - "whether ThetaData history serves 1-minute stock NBBO back to 2016 and option chains back to 2020 on the current tier (probe requested)"
errors: []
status: DRAFT for owner review; nothing runs until he approves and gate 1 passes
verified_by: null
budget: {agents: 2 (auditor gate 1 and 2), laptop: "one shares replay, one options replay, timed first"}
---

# Pre-registration — does news information change what the frozen runners earn?

The question is not "were news days profitable". It is: **given the frozen runners exactly as
they trade today, does knowing the news calendar or the news flow, at the moment the runner
could have known it, change the distribution of what each arm earns, and if so, by how much,
where, and does it survive data it has never seen.**

Direction prediction failing (eight nulls, `CLAUDE.md`) does not answer this. Neither does
the 2026-10-30 finding that release days are wider. Both are inputs here, not conclusions.

## 0. Why the live record cannot carry this alone

44 live sessions, 14 with any dated event, 22 of them PARTIAL (host suspended). Per-category
cells have 0–6 sessions. The live record is used in exactly one way: as a sanity check that
the replay reproduces its totals (§1.4), and as the **forward validation set from
2026-11-02 onward**, which this study never reads.

## 1. The replay, and the proof it is free of lookahead (owner condition 1)

### 1.1 What runs
`backfill.run(start, end, arm=...)` runs the unmodified `shares_runner.SharesLab` and
`runner.LiveLab` over past sessions. Nothing in the runners is changed; only the feed and the
clock are swapped (`backfill.py:311-360`). Config hashes are the live ones
(`7c6414fc549315f3` shares, `ccd00ac71f8243e9` options) and are written into every record.

### 1.2 Information barrier, by mechanism
| what the runner asks for | how history serves it | where the barrier is |
|---|---|---|
| minute bars of the replayed day | `HistoryFeed.minute_bars` returns only bars whose close time ≤ simulated clock (`_closed`, `backfill.py:135`) | a bar is visible only after it has closed, as live |
| prior sessions' bars (warm-up, lines) | `_no_future(day)` asserts `day ≤ replayed day` (`backfill.py:128`) | any request for a later day raises |
| underlying NBBO | last 1-minute history quote at or before the clock (`stock_quote`, bisect) | never a later quote |
| option chain | last 1-minute quote per contract at or before the clock (`chain_quotes`) | same |
| "now" inside the runner and the store | `SimClock` patched into `module.now_et` and `store._now` | records are stamped with simulated time |
| open interest | **not available in history**; returns `{}` | difference from live, see §9 |

**Tests that must exist and pass before the replay runs** (new file
`trade_analysis/live_lab/backfill_barrier_test.py`): (a) a bar whose close is one second
after the clock is not served; (b) a request for day+1 raises; (c) the NBBO served at
10:00:30 is the 10:00:00 quote, not 10:01:00; (d) the chain served at 14:00 excludes any
row stamped 14:01; (e) the runner's `now_et()` inside a replayed tick equals the simulated
clock. Each test is shown to fail when the corresponding guard is removed.

### 1.3 What the replay reproduces, measured (`replay_22_sessions.md`, 2026-09-29)
- **Session and month totals: yes.** Clean-hours shares gap **$1.32 per trade** over 265 round
  trips. Options: the same test on 39 trades, direction of totals consistent.
- **Trade for trade: no.** 76% of live trades have a replay twin; 44% exact (shares), 31%
  (options). Cause: the vendor corrects bars after the close, moving VWAP by a median 1.6 bp,
  which flips VWAP-cross signals. Setups by share of one-sided trades: Crabel 0%, Gap_Fade
  8%, IMB 19%, Six_Lines 25%, MOMO_CHASE 54%, ORB_5min 72%, **VWAP_Reclaim 79%**.
- **Fill timing:** replay fills on the minute's NBBO; live filled a median 1.5 s later.
- **Execution costs:** identical code: shares long buys ask / sells bid, `spread_bp` recorded;
  options buy ask / sell bid, fee `0.0404` per contract per side (`positions.py`).
- **Missing data:** the replay is a *perfect connection*. Live lost minutes to suspends and
  outages on 22 of 44 sessions. The replay therefore has **more** trades than live on bad
  days; this is a known, directional difference and is why live is a check on totals, not a
  twin.

### 1.4 Reproduction check before anything else is read
The first replay output is compared with the live record on the 44 live sessions: per-arm
daily net, trade counts, exit-reason mix. Acceptance: the replay's session-total correlation
with live ≥ 0.8 on COLLECTED sessions and the per-trade gap within the 2026-09-29 band
(≤ $3 shares). Failure stops the study until explained.

### 1.5 Consequence for the analysis
Because VWAP-dependent setups replay poorly, **per-setup results (Q7) are reported with each
setup's one-sided rate next to them**, and VWAP_Reclaim / ORB_5min / MOMO_CHASE findings are
labelled "low replay fidelity" whatever their p-value.

## 2. Data, windows, and the three disjoint sets (owner conditions 2, 8)

| set | arm | sessions | use |
|---|---|---|---|
| **Discovery** | shares | 2016-01-04 → 2022-12-30 | everything in §6 is explored here first |
| **Holdout** | shares | 2023-01-03 → 2026-09-24 | read **once**, after the discovery report is written and committed, to score the shortlist |
| Discovery | options (SPY only) | 2020-01-02 → 2022-12-30 | thin; results labelled so |
| Holdout | options (SPY only) | 2023-01-03 → 2023-12-29 | once |
| **Forward** | both, live | 2026-11-02 → | never read by this study; the validation the owner asked for |
| live record so far | both | 2026-08-28 → 2026-10-30 | §1.4 only |

QQQ options: no archive, not replayable, not claimed. The 2024 SPY block stays sealed
(`vrp_preregistration.md`). Options history before Nov 2022 has SPY 0DTE only on Mon/Wed/Fri;
the replay skips days without a 0DTE expiry and counts them.

**Volatility regimes**, fixed now: terciles of the trailing-20-session median range of SPY
(`med20_range_bp` from `day_features_SPY.csv`), computed on discovery data only and applied
as fixed cut-points to holdout. Every §6 table is reported overall, by calendar year, and by
regime tercile. A result whose sign flips across terciles or across ≥ 3 of the years is
reported as **unstable**, whatever the pooled p-value.

## 3. Events and information timing (owner condition 5)

Three distinct variables per event, never merged:

| variable | knowable when | from |
|---|---|---|
| `scheduled_today` | before the open | `events_scheduled.csv` `session` column: the calendar says a release is due |
| `release_minute` | at the release | `time_et` (08:30 for CPI/NFP/GDP/PCE, 14:00 FOMC) |
| `shock_minute` | at the break | `break_time_et` for geopolitical rows; `unknown` rows are usable only for day-level questions, never intraday |

**Post-release market reaction is not an information variable.** Any feature that uses the
day's range, gap or close is hindsight and is excluded from every conditioning set. The only
market quantity admitted is the trailing-20 regime (§2), known before the open.

A trade is assigned to **pre** or **post** by comparing its `entry_ts` with the event minute
on the same session; a trade entered at 13:58 on an FOMC day is **pre**. Scheduled-event
*awareness* (the calendar) attaches to the whole session; *release information* attaches
only to post-release trades.

## 4. What is measured, per arm, never pooled across arms (owner condition 4)

Shares risk $10,000 notional per trade; the options arm risks one contract per strike arm
(three arms), typically $150–$700 of premium. **Dollar P&L is never compared across arms.**
Each arm's metrics are in its own units; the cross-arm question (Q6) uses **standardised
daily return** (daily net divided by that arm's discovery-period daily standard deviation).

Per arm, per conditioning set, per discovery/holdout, per year, per regime:

| metric | shares | options |
|---|---|---|
| daily net, mean and median | $ | $ (ATM arm; ATM±1 as sensitivity) |
| standardised daily return | net / σ_daily | same |
| win rate | per trade, and per day | same |
| expectancy | $ per trade; **bp of notional** | $ per contract; **return on premium paid** |
| trade frequency | round trips per session | per session |
| average winner / average loser | $ and bp | $ and % of premium |
| max drawdown | on the cumulative daily series restricted to the condition's days, and on the full series | same |
| options-specific | — | median entry spread %, premium paid, share of exits by reason (stop / target / time / eod / trail), underlying twin (bp) so "direction right, structure lost" is visible |

## 5. Baseline: the same runners with no news conditioning (owner condition 6)

Every conditional number is reported **next to the unconditional baseline of the same arm
over the same set**: the frozen runner, every session, no knowledge of news. The question each
row answers is "did conditioning on this information move the number away from baseline,
and by how much, with what CI". Two baselines are computed:

- **B0**: all sessions in the set.
- **B1**: a **placebo** conditioning: the same number of sessions as the condition, drawn at
  random matched by year and weekday, 200 draws. The condition's effect is reported next to
  the placebo 95th percentile, as in `news_impact`. A condition whose placebo 95th percentile
  reaches half its effect is marked "selection / chance", not "effect".

**Incremental value** is the difference condition − B0 in standardised daily return, with a
day-clustered bootstrap CI (3,000 reps, seed 20260828, resampling sessions).

## 6. The questions, as tables (Q1–Q7)

All exploratory on discovery; the shortlist (≤ 5 rows) is scored on holdout once; nothing
leaves exploratory status before forward validation (§8).

- **Q1** daily net and standardised return by category (5 scheduled types; 8 geopolitical,
  financial_stress reported-only), vs B0 and B1, per arm.
- **Q2** the §4 metric block for the same cells.
- **Q3** scheduled (`surprise = no`, all five release types) vs unscheduled (`surprise = yes`
  geopolitical rows), per arm, same block.
- **Q4** Q1 with the day's direction nowhere in the conditioning set, by construction; a
  descriptive line adds the arm's twin (bp) so a positive P&L on news days can be read as
  "direction" or "structure".
- **Q5** pre vs post trades on event sessions: entry before vs after `release_minute` /
  `shock_minute`; and the one rule-shaped question: on `scheduled_today` sessions, the
  runner's P&L from trades entered in the first 60 minutes vs the same window on
  non-release sessions. Nothing here uses post-release prices as a condition.
- **Q6** per category, standardised return of options minus shares, CI; a cell counts as
  divergence only if both arms' own CIs exclude zero with opposite signs.
- **Q7** setup × category × arm (15 × 13 × 2 = 390 cells): standardised return vs that
  setup's own B0, Benjamini–Hochberg at q = 0.10 for **discovery**; survivors with |effect|
  ≥ 0.25 σ become the shortlist; the shortlist is scored on holdout with Holm at 0.05. Each
  row carries the setup's replay one-sided rate (§1.5).

## 7. Decision rules and minimums (owner condition 3)

**Minimum event counts.** A category is *tested* only with ≥ 30 distinct non-thin event
sessions in the set at hand **and** ≥ 50 trades of the arm on those sessions; otherwise it is
*reported* (n, effect, CI) and marked "underpowered". With 2016–2022 shares discovery this
admits FOMC (~55), CPI, NFP, PCE (~84 each), GDP (~28, reported), trade_tariff and
political_shock (likely), and excludes most other geopolitical categories, which is the
honest outcome of a 7-year window.

**Confirmatory tests (two, Holm-corrected as one family, run on holdout only after discovery
is committed):**
- **C1** shares: standardised daily return on `scheduled_today` sessions minus B0 over the
  holdout. PASS if the 95% day-clustered CI excludes 0 and |effect| ≥ 0.25 σ.
- **C2** options (SPY): same, holdout 2023.
A PASS says news *awareness* shifts that arm's outcomes by at least a quarter of a normal
day's variation. It says nothing about which way to trade and does not change the lab.

**Everything else** (Q1–Q7 by category, by setup, by regime) is exploratory: BH at q = 0.10
within each table, effect sizes with CIs always shown, p-values never shown alone.

## 8. From finding to forward validation (owner condition 8)

- Discovery and holdout results are committed, audited (gate 2), and published to the Space as
  **exploratory**.
- A shortlist row that passes holdout becomes its own one-paragraph pre-registration: the
  condition, the arm, the setup, the metric, the forward sample size (sessions), the decision
  rule. It is evaluated on live sessions from its registration date on.
- **No change to `live_lab/` trading logic, config or setup list from this study.** A forward
  pass is the earliest point at which a dated amendment could even be proposed, and that is a
  separate decision for the owner.

## 9. What cannot be validated against live execution (owner's closing ask)

| item | why it cannot be checked against live | how it is handled |
|---|---|---|
| vendor bar corrections | live saw uncorrected bars; history has corrected ones; the uncorrected stream is gone | per-setup one-sided rates attached to every result (§1.5) |
| fill latency | live filled ~1.5 s after the minute; replay at the minute | known small bias; direction not fixed; noted, not modelled |
| open interest | history has none; the options runner's OI check sees `{}` | measured once: run the live-period replay with and without the OI gate and report the trade-count difference |
| outages and suspends | replay is a perfect connection | live-period totals compared in §1.4; replay has more trades on bad live days by construction |
| **2016–2019 market structure** | the shares arm's rules were written in 2026 on 2026 microstructure; spreads, tick regimes and ETF volumes differ in 2016 | every table by year; discovery 2016–2022 vs holdout 2023–2026 is also a structure test |
| **SPY 0DTE before Nov 2022** | expiries only Mon/Wed/Fri; the options arm's universe was different | options discovery labelled "M/W/F only"; day-of-week reported |
| QQQ options | no archive at all | not claimed |
| the geopolitical list | selected by salience (front pages follow markets); unverifiable two-outlet rule | descriptive only + placebo; never in a confirmatory test |
| Finlight text | one month of history | not used; forward only |
| ThetaData history depth | unknown whether 1-minute stock NBBO (2016) and option chains (2020) are served on the current tier | **probe before anything else**; windows shrink to what the feed serves and the shrink is recorded |
| the replay cache | `backfill._cached` pickles feed answers, including empty ones | the same guard as specs/002 is added to `_cached` before the run (R1 of a new spec 003); the cache directory for this study is fresh |

## 10. Audit trail (owner condition 7)

Every output carries: `repo_sha` of the code, the two config hashes, replay start/end, feed
tier as reported by the terminal, the counts of sessions tried / skipped (no 0DTE, thin, no
bars) by year, the events files' SHAs, the exclusion list, the random seeds, and the exact
command lines. Outputs live under `live_lab_backfill/news_arms/` (provenance `BACKFILL`, never
read by the dashboard) and `research/news_arms/`. `analysis.json` holds every number quoted
in `results.md`; gate 2 traces each one.

## 11. Order of work

1. Owner reviews this file; amendments are dated and land before step 3.
2. Probes on the laptop (feed depth, runtime), results pasted into §9/§12.
3. `backfill_barrier_test.py` and spec 003 (cache guard) written and passing; gate 1 audit.
4. Live-period replay (2026-08-28 → 10-30) and the §1.4 reproduction check.
5. Discovery replay, join, discovery report committed.
6. Holdout read once; shortlist scored; results and gate 2.
7. Space: snapshot builder extended; the study published as exploratory.

## 12. Probe results (to be filled from the laptop, before step 3)

| probe | result |
|---|---|
| shares replay, 20 sessions, minutes | |
| `/stock/history/quote` 1m, SPY, 2016-06-14, rows | |
| `/option/history/quote` 1m, SPY 0DTE chain, 2020-06-12, rows | |
| feed tier string from terminal | |
