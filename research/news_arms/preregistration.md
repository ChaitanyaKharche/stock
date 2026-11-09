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
status: conditionally approved by the owner 2026-11-01; amended (section 13, nine items); gate 1 first pass FAILED 7 of 12; amendment 14; second pass closed 5, left 7 one-sentence fixes; amendment 15; third pass left one item (weekend markers), amendment 16 (11 rows) was itself short by two named-holiday rows; amendment 17 (13 rows) missed the post-early-close row; amendment 18; GATE 1 PASSED on the sixth pass (all 12 items; audit_preregistration.md); OWNER APPROVED 2026-11-01 ("start coding spec 003 and the barrier tests; record the three clarifications before discovery; run replay only after the required probes and tests pass"); section 16 records the clarifications; no replay before barrier tests (a)-(i), probe (h), the section 12 probes and specs 003/004 pass
verified_by: R-2026-11-01-02 (gate 1, six passes, PASS on the text only; probes and tests still to run)
budget: {agents: 2 (auditor gate 1 and 2), laptop: "one shares replay, one options replay, timed first"}
---

# Pre-registration — does the frozen runners' outcome distribution differ under news conditions?

(Title reworded 2026-11-01, amendment 14.12: the study measures conditional differences; it
does not measure what news would do if fed to the runner.)

The question is not "were news days profitable". It is: **do the frozen runners' outcomes, exactly as they trade today, differ between sessions with and without news information that was knowable at the time, and if so, by how much, where, and does the difference survive data it has never seen** (reworded 2026-11-01, amendment 15.12).

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
  random matched by year and weekday, **5,000 draws** (amendment 13.3; the draft said 200).
  Each draw computes the **identical** condition − B0 statistic. The condition's effect is
  reported next to the placebo 95th percentile and its placebo rank. A condition whose placebo 95th percentile
  reaches half its effect is marked "selection / chance", not "effect".

**Conditional difference from baseline** (amendment 14.12; the draft said "incremental
value") is condition − B0 in standardised daily return, with a day-clustered bootstrap CI
(3,000 reps, seed 20260828, resampling sessions).

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
A PASS says that arm's outcomes differ on calendar-release sessions by at least a quarter
of a normal day's variation (amendment 14.12). It says nothing about which way to trade,
nothing about what feeding the calendar to the runner would do, and does not change the lab.

**Everything else** (Q1–Q7 by category, by setup, by regime) is exploratory: BH at q = 0.10
within each table, effect sizes with CIs always shown, p-values never shown alone.

## 8. From finding to forward validation (owner condition 8)

- Discovery and holdout results are committed, audited (gate 2), and published to the Space as
  **exploratory**.
- A shortlist row that passes holdout becomes its own one-paragraph pre-registration: the
  condition, the arm, the setup, the metric, the forward sample size (sessions), the decision
  rule. It is evaluated on live sessions from its registration date on.
- **No change to `live_lab/` trading logic, config or setup list from this study.** A forward
  pass is a result the owner may weigh; it does not itself propose an amendment (14.12).

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

## 13. Amendment 2026-11-01, on the owner's conditional approval, before gate 1

Nine items, each replacing or extending the section it names. Nothing below was written
after any replay output existed; no replay has run.

**13.1 (§1.2) Point-in-time options contract universe.** Valid quote stamps do not prove the
*set of contracts the runner could choose from* was point-in-time. Two more barrier tests:
(f) a contract whose first quote of the day is stamped 11:00 is absent from `chain_quotes`
at 10:59 and present at 11:00, and the runner's strike selection at 10:00 cannot pick it;
(g) `zero_dte()` answers only "does this expiry exist", never a strike list, and the runner's
ATM / ATM±1 choice is shown (by test) to be made from `chain_quotes` at the decision minute
alone. In addition the replay logs, per fill, the number of contracts visible at the
decision minute versus at the close; a fill whose chosen strike was not in the
decision-minute set is a defect that stops the study. Open interest remains unavailable in
history (§9).

**13.2 (whole document) Association, not incremental value.** Everything this study can
produce is a **conditional association**: the frozen runner, unchanged, earns a different
distribution on sessions with condition X than on sessions without it. It **cannot**
establish that feeding X into the runner would improve profitability: the runner never saw X,
no decision was changed, and any policy that acts on X would alter the trade set in ways this
data cannot show. §5's "incremental value" is renamed **"conditional difference from
baseline"**; the phrase "incremental trading value" is reserved for a **separate forward
policy experiment** (a pre-registered rule, run live, compared with the unconditioned runner
on the same sessions), which this study may motivate and never substitutes for.

**13.3 (§5) Placebo.** 5,000 draws per condition (200 in the draft), each computing exactly
the condition − B0 statistic used for the real condition, matched by year and weekday, the
condition's own sessions excluded from the pool. Reported: placebo 95th and 99th percentile
and the real effect's rank among the 5,000.

**13.4 (§3, new table) Event overlap.** An **overlap matrix** (13 × 13, sessions carrying
both categories) is produced from the two event files before any P&L is read and committed
with the discovery report. Handling, fixed now: a session with two or more categories counts
in each category's cell (as in `news_impact`), **and** every Q1–Q7 table is re-run on
**single-category sessions only** as a sensitivity line. A category whose effect changes sign
or loses half its size when overlaps are removed is reported as **confounded by overlap**.
Scheduled-with-scheduled overlaps (CPI and FOMC on one day; NFP and PCE) are listed
individually.

**13.5 (§6 Q5) Pre/post trades are descriptive.** The pre-release versus post-release split
is a description of when the runner's trades happened and what they earned. It is **not** a
causal estimate of the news: the same minutes on non-event days are not a matched control,
intraday volatility and the runner's own time-of-day pattern differ, and a release can move
pre-release behaviour through positioning. Q5 is labelled *descriptive, exploratory* in every
table, and no rule is derived from it in this study.

**13.6 (§7) Power for C2, computed before the data.** Standardised daily return has σ = 1 by
construction. SPY 2023 holdout: about 250 sessions, of which about 48 distinct scheduled
sessions (8 FOMC + 12 CPI + 12 NFP + 4 GDP + 12 PCE, overlaps removed). Two-sample SE of a
mean difference ≈ √(1/48 + 1/202) ≈ 0.16 σ; **minimum detectable effect at 80% power,
two-sided 5%: ≈ 0.45 σ**, above the 0.25 σ threshold. Therefore **C2 is pre-declared
underpowered for a 0.25 σ effect and is reported with its effect size, CI and achieved MDE,
never as PASS/FAIL**, unless the observed holdout effect itself exceeds the MDE. C1 (shares
holdout ≈ 930 sessions, ≈ 185 scheduled): SE ≈ 0.082 σ, MDE ≈ 0.23 σ, adequate for 0.25 σ.
The exact MDEs are recomputed from the realised session counts and written next to each.

**13.7 (§1.4) Live-replay acceptance, per arm, independently.**
- Sessions used: live sessions with ledger outcome **COLLECTED** only (22 of 44 at
  `3d21893`); PARTIAL, ABORTED and INTERRUPTED sessions are excluded from acceptance (the
  replay has trades live could not have) and listed with the hours lost.
- Minimum valid sessions: **15 per arm**; below that the arm's acceptance is "not
  established" and that arm's results are labelled so throughout.
- Shares passes if: Spearman correlation of session net ≥ 0.80, median |per-trade gap| on
  twinned trades ≤ $3.00, and trade-count ratio replay/live within 0.9–1.3.
- Options passes if: Spearman ≥ 0.70 (lower fidelity measured in September), median
  |per-trade gap| on twinned ATM trades ≤ $8.00 (about 5% of a typical premium), trade-count
  ratio within 0.8–1.5, and exit-reason mix differing by ≤ 15 percentage points on any
  reason.
- **An arm that fails is reported as failing; the other arm's pass does not rescue it, and a
  failing arm's news results are exploratory-descriptive only, with no confirmatory test.**

**13.8 (§3, §8, conclusions) Three information sets, kept apart to the end.** Every finding
names which set it belongs to: **(i) calendar awareness** (a release is due today; known
before the open), **(ii) release information** (the number and its time; known at the
minute), **(iii) post-release market reaction** (what prices did; hindsight). Only (i) can
become a pre-open rule; (ii) only an intraday rule from the release minute on; (iii) is never
a conditioning variable and appears only as the descriptive "twin" column. The conclusions
section of `results.md` is structured as three sub-sections with these names.

**13.9 (results template) Four verdicts per finding.** Every reported row carries four
separate fields, never collapsed into one word: **statistical** (CI, BH/Holm status),
**effect size** (σ units and the arm's native units), **replay fidelity** (the setup's or
arm's one-sided rate and acceptance status from 13.7), and **practical usefulness** (which
information set, whether a pre-open or intraday rule could use it, expected trades per year
at that frequency, and whether the native-unit effect exceeds round-trip costs). A row that
is significant but low-fidelity, or significant but worth less than its spread, says so in
its own fields. "Actionable" is not a word this study uses; the most it says is "candidate
for a forward policy experiment".

## 14. Amendment 2026-11-01 (second), on the gate-1 audit (`audit_preregistration.md`)

The auditor failed items 1, 2, 3, 4, 7, 8, 12 and found the bug-class scenario (11). Each
fix below names the audit item. No replay has run; no P&L has been read.

**14.1 (audit 1) Information barrier, completed.**
- Tests (a)–(g) use constructed rows and prove the *mechanics*; they cannot prove what a
  vendor 1-minute quote row stamped T *contains*. New **probe (h), mandatory before any
  replay**: for 5 random sessions × 3 symbols (SPY, QQQ, one shares name) and 5 random
  option contracts, fetch the 1-second (or tick) quotes for the minute [T, T+59 s] and
  determine whether the 1-minute row stamped T is the quote **at** T or the **last** quote
  of the minute. If the latter, `HistoryFeed` must serve a row stamped T only once the clock
  reaches T + 60 s (a change under `live_lab/`, spec-first), and the barrier tests are
  re-pointed to that rule. The probe result is recorded in §12.
- New **test (i)**: during a replay the real feed is wrapped so that any direct call to
  `_get_csv`, `_fetch_minute_bars`, `extended_bars` or any method not routed through
  `HistoryFeed`'s clock-guarded methods **raises**. The runner is shown to complete a
  replayed day under that wrapper.
- Test (e) extended: every record the store writes during a replayed day carries a stamp
  ≤ the simulated clock at the moment it was written.
- §1.2 "as live" is replaced by: **same availability time as live; different content, the
  vendor's final bar rather than the first print** (fidelity consequence in §1.3).
- §9 open-interest row rewritten: the options runner **records** OI (`runner.py:476-483`) and
  **no decision depends on it**; the replay records `null`. Nothing to measure.
- Half days and short sessions dropped by `MIN_RTH_BARS` (`backfill.py:348`) are counted by
  year and listed; they are exclusions, not sessions.

**14.2 (audit 2) Acceptance, re-specified from the measured numbers.**
- Acceptance sessions: live sessions with ledger `COLLECTED` **and** both live config hashes
  in force **and** a clean start (first live tick before 09:31). At `3d21893` that is **16**
  sessions, not 22. Each session is compared only over the minutes live was actually trading
  (its scope); 2026-08-28 → 09-04 (blind until 09:39) and 09-25 (20 minutes asleep) are
  trimmed to scope or excluded if the scope is under 5 hours.
- The live-period options replay runs **QQQ and SPY** (the chains for those 29 sessions are
  cached), so it carries the live hash `ccd00ac71f8243e9`. The historical SPY-only replay
  cannot carry that hash (symbols are hashed); it runs under a **derived hash recorded as
  such**, and its results are labelled "SPY-only derived config".
- Thresholds set from September's measured values (Spearman 0.96 shares, 0.94 options,
  recomputed by the auditor): **shares pass at Spearman ≥ 0.90, options ≥ 0.85.** The
  per-trade gap statistic is defined as the **median of |live net − replay net| over twinned
  trades** (September reported only bin totals, so this is a new statistic with the $3 / $8
  bars as provisional; the realised September value is computed first and the bar may only
  be tightened, never loosened).
- Forward sessions (§2) are admitted to validation under the **same COLLECTED + live-hash +
  clean-start rule**; PARTIAL forward sessions are listed, never scored.

**14.3 (audit 3) Disjoint sets, repaired.**
- Shares holdout ends **2026-08-27**. The live period 2026-08-28 → 10-30 is used for §1.4
  acceptance only and is in no discovery or holdout table.
- Volatility-regime cut-points come from **`day_features_QQQ.csv`** (2016–2026) for both arms;
  `day_features_SPY.csv` starts 2020 and is not used for regimes.
- Forward validation is pinned to the two live hashes and to `live_lab/` code at the SHA of
  the final approved pre-registration; the news snapshot (specs/001) writes a file and
  touches no trading path, verified by its tests.

**14.4 (audit 4) Pre/post, made decidable.** A trade is **post** an event only when the event
minute falls inside 09:30–16:00 **on that session date** and `entry_ts` is at or after it.
Pre-open releases (08:30 CPI/NFP/GDP/PCE) and overnight or weekend shocks make every trade
of the session post, by definition. Geopolitical rows with `break_time_et = unknown` are
excluded from Q5 entirely. Break times are recollections rounded to 30–60 minutes and Q5
says so in every table. **13.8 corrected:** information set (ii) released before 09:30 is
available before the open and may inform a pre-open rule; only releases inside the session
(FOMC 14:00) are intraday-only.

**14.7 (audit 7) Multiple testing, pinned.** For each table, **m = the number of cells
meeting the §7 minimums**, computed from session and trade counts and **committed before any
P&L is read**. `financial_stress` is removed from Q7 and from every tested table. If more
than five cells survive BH on discovery, the shortlist is the **top five by |effect|**.
**C1 and C2 are one Holm family of two.** No geopolitical category reaches 30 holdout
sessions; all geopolitical holdout rows are reported, not tested, and the document says so
in advance.

**14.8 (audit 8) Power, corrected.** Shares holdout has about **170** scheduled sessions (not
185): SE ≈ 0.085 σ, MDE ≈ 0.24 σ (figures corrected in 15.8). Because PASS also requires the point estimate ≥ 0.25 σ,
**power at a true 0.25 σ is about 50% for C1 and lower for C2; 80% power needs about
0.31 σ (C1) and 0.36 σ (C2).** Both numbers are stated next to each result. The analytic SE
treats condition and complement as independent samples; the reported CI comes from the
day-clustered bootstrap of condition − B0, which is the estimand, and the analytic figure is
labelled approximate.

**14.11 (audit 11) The bug-class scenario, closed.** `backfill.run` does not skip sessions
without a 0DTE expiry; before Nov 2022 that would have entered Tue/Thu as $0 options days,
shrunk discovery σ by about a quarter, inflated 2023 effects by about 1.3×, and made
GDP_ADV (released on Thursdays) "beat baseline" by construction. Rule: **a session where
`zero_dte()` is None is excluded from every options table, from B0 and from σ**; it is
counted by weekday and year. A Mon/Wed/Fri session with no 0DTE, or a session on which every
symbol was skipped, **halts the run** for inspection. **Spec 003** (`backfill._cached` never
caches an empty or error response; HTTP errors are distinguished from empty data) is
written, tested and landed before the first replay.

**14.12 (audit 12) Language.** Title reworded (above). §5 "Incremental value" → "conditional
difference from baseline". §7 "news awareness shifts that arm's outcomes" → "that arm's
outcomes differ on calendar-release sessions". §8: a forward pass is a *result the owner may
weigh*; it does not itself propose an amendment.

**14.Q (audit 5, 6, 9, 10: questionable items, answered).**
- (5) Standardised return is reported twice: with discovery σ (comparable across sets) and
  with the set's own σ. Replay exit reasons map to §4 as: `failed` → stop (close back inside),
  `bars` → time, `trail_imb` → trail, `shutdown` → excluded and counted.
- (6) Condition − B0 is diluted by (1 − p) where p is the condition's share of sessions; the
  undiluted condition − complement difference is reported beside it. The "half the effect"
  heuristic is replaced: **a condition is marked "selection / chance" when its real effect
  ranks below the 95th percentile of the 5,000 placebo effects of the same sign.**
- (9) The single-category sensitivity line is also computed for C1 and C2.
- (10) Discovery keeps 2016 as its start, and the shortlist is additionally scored with a
  **2018-start discovery** as a sensitivity; a shortlist row that disappears under the 2018
  start is marked "driven by 2016–2017".

## 15. Amendment 2026-11-01 (third), on the gate-1 second pass

**15.1 (audit 1)** Test (i) is restated: during a replay `lab.feed` is a **proxy** exposing
only `HistoryFeed`'s clock-guarded methods plus `calls`, `stats`, `wait_for_upstream` and
`close`; any other attribute access raises. `HistoryFeed` keeps its own reference to the real
feed for its guarded fetches; the runner never holds one. The runner is shown to complete a
replayed day through the proxy.

**15.2 (audit 2)** The cached QQQ chains for the live period were written before spec 003 and
never inspected. Before acceptance they are re-validated under spec 003 (non-empty, no error
payloads, every minute 09:30–16:00 present); if they do not cover **all 16** acceptance
sessions, options acceptance is **SPY only** and says so. The clause admitting 2026-08-28 →
09-04 is **deleted**: those sessions fail the live-hash rule and are out. The 16 sessions all
started before the open in both arms (auditor verified).

**15.4 (audit 4)** A geopolitical row whose headline carries a parenthetical naming
**Saturday, Sunday, weekend, "holiday", "early close", or a named US market holiday** (Presidents Day, MLK /
Martin Luther King Day, Memorial Day, Good Friday, Labor Day, Thanksgiving, Christmas, New
Year, Juneteenth, Independence Day / July 4) counts as **overnight**: every trade that session
is post, whatever `break_time_et` holds. "(half-day session)" is **not** a marker
(2020-12-24 is a trading day). Applied to the frozen file this marks exactly **14 rows**:
2017-04-24, 2018-05-29, 2019-03-25, 2019-05-06, 2020-11-09, 2021-03-08, **2022-02-22
(Presidents Day)**, 2022-08-08, 2023-02-06, 2024-06-10, 2024-07-22, **2025-01-21 (MLK Day)**,
**2025-07-07 (after the July 3 early close)**, 2025-07-28. A test asserts the count of 14,
that 2020-12-24 is not marked, and that no other in-session row carries an unmatched
parenthetical. (Amendments 16, 17, 18: the draft's three markers caught 3; the list of 11
missed two named holidays; the list of 13 missed the post-early-close vote, found by the
auditor on the fifth pass.) Rows with `break_time_et = unknown` carry the label **"time
unverified"** in every day-level table. New test: **no conditioning column is built from the
session's own prices**; the analysis module exposes the list of conditioning columns and a
test asserts none of them is derived from `open`, `high`, `low`, `close`, `gap_bp`,
`range_bp`, `co_bp`, `first60_*`, `rel_*`, `impact`, `day_class` or any P&L field.

**15.5 (audit 5)** All tests, thresholds (0.25 σ, BH, Holm, MDE) use **discovery σ**. The
own-σ standardisation is descriptive only and is marked so in every table.

**15.7 (audit 7)** Q7 cells use the **arm's** discovery σ (not a per-setup σ), and a session
on which the setup did not trade **counts as 0** for that setup, so the cell's day count is
the full session count.

**15.8 (audit 8)** With 170 scheduled holdout sessions: SE ≈ **0.085 σ**, MDE ≈ **0.24 σ**
(the draft's 0.082 / 0.23 were the n = 185 values). Power at a true 0.25 σ under the pass
rule ≈ 50%; 80% needs ≈ 0.31 σ (C1) and ≈ 0.36 σ (C2). Unchanged otherwise.

**15.12 (audit 12)** Opening question reworded (top of document).

## 16. Clarifications recorded on approval, 2026-11-01, before any discovery

1. **A closed-day geopolitical row with no bracket in its headline cannot be caught from the
   file.** The 15.4 marker covers every bracketed in-session row (14). An unbracketed row
   whose news actually broke on a closed day would be classed in-session and its morning
   trades "pre". Known, unmeasured gap; Q5 is descriptive and says so. The forward Finlight
   log carries real timestamps and does not have this problem.
2. **Gate 1 approved the text, not any result.** Six audit passes checked the design against
   the owner's conditions; no number produced after this point inherits that approval.
   Results pass or fail gate 2 on their own.
3. **The replay is gated by artefacts that must exist and pass first:** barrier tests (a)–(i)
   in `backfill_barrier_test.py`, probe (h) on the vendor's minute-stamp content, the §12
   probes (feed depth, runtime), spec 003 (`backfill._cached` never caches an empty or error
   response) and spec 004 (the runner reaches history only through a clock-guarded proxy).
   Probe results are written to `research/news_arms/probes.json` by
   `trade_analysis/live_lab/backfill_probe.py`, run on the laptop; the orchestrator copies the
   numbers into §12 and does not edit them.

## 11. Order of work

1. Owner reviews this file; amendments are dated and land before step 3.
2. Probes on the laptop (feed depth, runtime), results pasted into §9/§12.
3. `backfill_barrier_test.py` and spec 003 (cache guard) written and passing; gate 1 audit.
4. Live-period replay (2026-08-28 → 10-30) and the §1.4 reproduction check.
5. Discovery replay, join, discovery report committed.
6. Holdout read once; shortlist scored; results and gate 2.
7. Space: snapshot builder extended; the study published as exploratory.

## 12. Probe results (laptop, 2026-11-01 23:11 MST, `backfill_probe.py` at `78462ed`, file `probes.json`)

| probe | result | consequence |
|---|---|---|
| minute-stamp content (h), 30 checks: 5 sessions × SPY/QQQ/XLF × 10:00 and 13:30 | **AT_T 30 / 30**: the 1-minute quote stamped T equals the first 1-second quote of that minute, never the last | no 60-second shift; the barrier as built is correct |
| `/stock/history/quote` 1m, SPY 2016-06-14 | **0 rows** | 2016–2017 not served on the current tier |
| `/stock/history/quote` 1m, XLF 2018-06-13 | 391 rows | 2018 is served |
| `/option/history/quote` 1m, SPY 0DTE 2020-06-12 | 117,300 rows | the options replay 2020–2023 is feasible |
| shares replay, 20 sessions (2024-06-03 → 06-28) | **26.7 min**, rc 0, ≈ 80 s per session | 2018-01-02 → 2026-08-27 ≈ 2,170 sessions ≈ **48 h** of laptop time |
| feed | 63 calls, 0 retries, upstream up | |

A first attempt earlier that evening recorded only connection errors: Theta Terminal was not
running. Both files are in git; the probe now refuses to run without the terminal.

## 17. Amendment 2026-11-02, on the probe results (OWNER APPROVED 2026-11-04: "approved 17.1 to 17.3, go")

**17.1 (§2) Shares discovery starts 2018-01-02**, not 2016-01-04: the feed does not serve
2016–2017 minute quotes on the current tier. Discovery = 2018-01-02 → 2022-12-30 (about
1,255 sessions); holdout unchanged (2023-01-03 → 2026-08-27). This also answers the
auditor's item 10 (2016–2017 microstructure) by removing those years; the 14.Q "2018-start
sensitivity" becomes the main run and is dropped as a sensitivity. Regime cut-points still
come from `day_features_QQQ.csv`, now over 2018–2022.

**17.2 (§11) Runtime and scheduling.** About 48 hours of replay on the laptop, run in
calendar-year chunks (`--start`/`--end` per year, separate `--out` folders), only outside
09:00–16:30 ET because the lab needs the terminal during sessions, and only with the host
prevented from sleeping. Each chunk's `BACKFILL.json` records its own window and hash; the
join concatenates chunks and asserts no session appears twice.

**17.3 (§1.4) Live-period acceptance run first:** 2026-08-28 → 2026-10-30, both arms
(≈ 1.5 h shares, options as measured), compared on the 16 qualifying sessions per 14.2.

## 18. Step 4 record: live-period acceptance, 2026-11-03/05

Replay 2026-08-28 → 2026-10-30, both arms, on the laptop 2026-11-03 23:37–23:49 MST at
`8e5b958` (shares 17.5 min for 45 sessions, options 5.7 min). Compared with the live record
by `acceptance.py` on **16 qualifying sessions per arm** (COLLECTED, clean start, live
hash; 6 excluded for late starts, 22 PARTIAL, 1 INTERRUPTED).

| arm | Spearman (bar) | median twinned gap (bar) | count ratio (bar) | exit mix worst pp (bar) | verdict |
|---|---|---|---|---|---|
| options (ATM, 270 twinned trades) | **0.953** (≥ 0.85) | **$6.00** (≤ $8) | 1.022 (0.8–1.5) | 1.7 (≤ 15) | **PASS** |
| shares (1,938 twinned) | **0.932** (≥ 0.90) | **$2.33** (≤ $3) | 1.009 (0.9–1.3) | — | **PASS** |

**Defect found and corrected before the verdict was read as final:** the first run of
`acceptance.py` (commit `e9e2b0d`) admitted 2026-11-03 as a qualifying session although the
replay ended 2026-10-30, so both arms were scored against a day with zero replay trades;
that run printed shares FAIL at 0.8995. The fix restricts qualifying sessions to dates the
replay produced (test added). Nothing else changed; the bars were not moved. The halt rule
also flagged Labor Day (a holiday, no bars) as a missing 0DTE; it now checks only replayed
sessions. Both runs' JSON are in git.

Both arms clear 14.2. The options arm's confirmatory test C2 stays pre-declared underpowered
(13.6); C1 (shares) stands.

## 19. Step 5 record: shares holdout replay, 2026-11-08

Holdout 2023-01-03 → 2026-08-27 replayed on the laptop 2026-11-08 at `7b2cb4a` (2023, 2024,
2025 as monthly chunks with 3 workers, rc 0 for all 36 months; 2026-01-02 → 08-27 as one
chunk, 164 sessions, 221 min). **Not joined, not read.** The join and the analysis code read
discovery only until the discovery report is committed and gate 2 has passed (§2, 14.3).

**Exposure, recorded rather than hidden (CLAUDE.md: wrong things are corrected in place,
never deleted):** the replay heartbeat prints one session's trade count and net P&L every
two minutes, and the owner pasted those consoles into the chat. The assistant therefore saw
roughly one in five holdout sessions' daily nets, unaggregated, before discovery was
analysed. No sum, mean or event join was computed from them, and no event calendar was
beside them. The heartbeat print is a replay diagnostic, not an analysis read, but it is a
read of holdout numbers and is logged here as one. Mitigation for the options holdout (2023):
run with `--workers 3` and do not paste the console; the exit code and `RUN.json` are
enough.
