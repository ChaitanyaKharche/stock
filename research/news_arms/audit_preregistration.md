---
task_id: R-2026-11-01-02
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
parent: R-2026-11-01-01
role: statistical-auditor
repo_sha: 094b848
inputs: [research/news_arms/preregistration.md@094b848, research/replay_22_sessions.md, trade_analysis/live_lab/backfill.py]
decision_rule: "PASS/FAIL/QUESTIONABLE per item; gate 1"
outputs: [research/news_arms/audit_preregistration.md]
evidence: []
assumptions: []
open_questions: []
errors: []
status: done
verified_by: null
budget: {cap: "one pass"}
---

# Gate 1 audit: news_arms pre-registration

**Short version: do not run the replay on this text. 7 of the 12 items FAIL, 4 are QUESTIONABLE and 1 is PASS.**
Six of the FAILs can be fixed by changing words. The seventh (item 1) also needs the two
new tests described there. The three that matter most:

1. **The holdout overlaps the reproduction check.** The live-period replay (2026-08-28 → 10-30) is
   run and read before discovery is committed. Its first four weeks (08-28 → 09-24) are also
   holdout dates. A replay of those exact days was already published on 2026-09-29.
2. **The options arm will count days with no option as days that made $0.** `backfill.run` does
   not skip days without a 0DTE expiry, whatever §2 says. Before Nov 2022 that is about 40% of
   days (no SPY expiry on Tuesdays and Thursdays). The estimated effect: the options yardstick
   (σ) comes out about 23% too small, so every 2023 holdout effect looks about 1.3× bigger.
3. **The volatility-regime variable does not exist for 2016–2019.** `day_features_SPY.csv`
   starts on 2020-01-06.

How I checked: I read the code paths into the runner (`backfill.py`, `runner.py`,
`shares_runner.py`, `levels_live.py`, `catchup.py`, `feed.py`, `store.py`, `clock.py`). I
recomputed the September fidelity numbers from `live_lab_backfill/*/report.json`. I recounted
sessions, weekdays and overlaps from the two event CSVs, and redid the §13.6 power numbers
with my own code. I ran no replay and no ThetaData call.

## Verdict table

| # | Item | Verdict | Reason (section) |
|---|---|---|---|
| 1 | Information barrier | **FAIL** | Tests (a)–(g) use made-up rows, so they cannot tell what a vendor quote stamped T contains. No test pins the runner to the guarded methods (`backfill.py:259-264` pass-throughs are unguarded). §9 tests an OI gate that does not exist (§1.2, 13.1, §9) |
| 2 | Fidelity acceptance | **FAIL** | "COLLECTED" ≠ clean. Only 16 COLLECTED sessions ran the live hashes. A SPY-only options replay cannot carry `ccd00ac71f8243e9`. The thresholds are not the statistics September measured (1.4, 13.7) |
| 3 | Disjoint sets | **FAIL** | The holdout (to 2026-09-24) overlaps the §1.4 live-period replay (08-28 → 09-24). The §2 regime source starts in 2020 (§2, §11 step 4) |
| 4 | Information sets / pre-post | **FAIL** | `break_time_et` has no calendar day. A naive comparison makes every trade on the 100 after-close rows "pre", and those rows really are all post. 08:30 releases are known before the open, yet 13.8 allows only (i) as a pre-open rule (§3, 13.8) |
| 7 | Multiple-testing families | **FAIL** | m is undefined (are underpowered cells counted?). financial_stress is reported-only in Q1 but inside Q7's 390. No tie-break if more than 5 survive. C2's place in the Holm family is left open by 13.6 (§6 Q7, §7, 13.6) |
| 8 | Power | **FAIL** | The arithmetic reproduces. But C1 has 170 holdout scheduled sessions, not 185. A PASS needs the point estimate ≥ 0.25σ, so power at a true 0.25σ is **50%, not 80%**. The SE is for condition vs complement, but the test is condition − B0 (13.6) |
| 12 | Association language | **FAIL** | §7 "news awareness shifts that arm's outcomes", the title, §5 "Incremental value", and §8 "a forward pass … a dated amendment" still read as causal or incremental (13.2) |
| 5 | Metrics | QUESTIONABLE | Discovery σ is wrong for holdout ("σ = 1 by construction" is false there). Options σ includes zero days. Replay exit reasons `failed`/`bars`/`trail_imb` are not in the §4 list (§4) |
| 6 | Baselines / placebo | QUESTIONABLE | Condition − B0 is diluted by (1−p) and p differs by category. The "half the effect" rule needs about 3.3 SE and has no rule for negative effects. No draw-vs-condition leakage (§5, 13.3) |
| 9 | Overlap | QUESTIONABLE | Fine for Q1–Q7. Reproduced: CPI+FOMC 7, GDP+PCE 5, FOMC+GDP 3; 73 of 758 event sessions are multi-category. But C1/C2 get no single-category line (13.4) |
| 10 | 2016–2019 structure | QUESTIONABLE | 2016–2019 is 4 of the 7 discovery years. Tables by year are descriptive only. The pooled shortlist would be driven by the years least like the rules' 2026 market (§9) |
| 11 | One bug class | **FAIL (found)** | "No 0DTE" and "dead feed" days enter the options series as $0 days. The design does not catch this (item 11) |

---

## FAILs

### 1. Information barrier (§1.2, 13.1): FAIL

**What the seven tests prove.** They check that the bisect and the clock work: no future
bar, no future day, no later quote, no later chain row, the runner's clock, and a
point-in-time contract set. That part is fine. The code I read matches the table in §1.2:
- `_closed`: `backfill.py:130-132`
- `_no_future`: `backfill.py:126-128`
- `stock_quote`: `backfill.py:175-184`
- `chain_quotes`: `backfill.py:219-234`

The six-lines premarket bars go through the same `_closed` filter (`backfill.py:145-153`,
`levels_live.py:84`). Zero-price bars are dropped at source (`feed.py:374, 405`), so the
warm-up's "≥ 300 bars" test (`levels_live.py:68-70`) rejects a holiday's zero-filled grid.
`catchup.run` cannot fire inside a replay because `backfill.run` calls `_tick` directly,
not `run()`.

**Paths no test covers:**
- **What a vendor stamp means.** Tests (c)/(d) are built from made-up rows. They cannot tell
  whether ThetaData's 1-minute quote row stamped 10:00:00 is the NBBO *at* 10:00:00 or the
  last NBBO *in* [10:00, 10:01). If it is the second, every fill sees up to 58 s into the
  future. The only evidence is one spot check of the underlying at 15:55 on 09-22
  (`replay_22_sessions.md` §4). There is none for option quotes. Bars have the same gap:
  `_closed` assumes `ts` marks the start of the minute.
- **Unguarded pass-throughs.** `HistoryFeed._get_csv` (`backfill.py:263-264`) serves any
  path for any date. `_fetch_minute_bars` (`:259-261`) checks the day but does not cut the
  replayed day at the clock. I grepped the runner path: neither is called today. But no
  test fails if that changes.
- **Session inclusion uses the whole day.** It reads the full day's bar count at 16:01
  (`backfill.py:348-349`, `MIN_RTH_BARS = 300`). The 210-bar half-days are dropped as "not
  a session", and live trades them. This is sample selection, not a trading decision, and
  the §10 funnel does not count it.
- **Open interest.** There is no OI gate. `runner.py:476-483` only records `oi`
  (`positions.py:194`). So the §9 measurement "replay with and without the OI gate" can only
  ever return 0. It is a check that cannot fail.
- **Store clock.** Test (e) checks the runner's `now_et()`, not the store's record stamps
  (`S._now`, patched at `backfill.py:333`). `today_et` (`runner.py:30`) is not patched. It
  is reached only via `run()`, which the replay never calls, so it is harmless today.
- **Test (c) uses 10:00:30, a time the replay clock never takes.** The clock ticks at
  HH:MM:02.
- **`calls` counter** (`backfill.py:241`): it always goes up, so the link always looks up.
  It does not leak future data. Noted for completeness.

**"A bar is visible only after it closes", is that the same availability as live?**
The timing is the same: about 2 s after the close in both. **The content is not.** The
replay sees the vendor's corrected final bar. Live saw the first print. Corrected bars carry
prints reported after the decision time; on identical decisions the VWAP moved a median
1.6 bp (`replay_22_sessions.md` §4). §9 treats this as a fidelity problem, but it is
information the runner could not have had at that minute.

**Minimal wording change:**
- §1.2 row 1: "as live" → "at the same time as live; the content is the vendor's final bar,
  not the first print (§9)".
- Add **(h)**: for at least 20 sampled minutes per year, the stock and option 1-minute rows
  stamped T equal the last tick or 1-second quote at or before T.
- Add **(i)**: a spy feed shows the runner, across one full replayed session, calls only
  `minute_bars(_many)`, `extended_bars`, `stock_quote(_many)`, `zero_dte`, `chain_quotes`
  and `open_interest`.
- Extend (e) to the store's record stamps.
- §9 OI row: replace with "the runner has no OI gate; OI is recorded only; no trade-count
  effect".
- §10: count half-days dropped by `MIN_RTH_BARS`.

### 2. Fidelity acceptance (§1.4, 13.7): FAIL

**COLLECTED does not mean clean.** I read the ledger: 22 COLLECTED, 22 PARTIAL. Six of the 22
COLLECTED are 08-28 → 09-04. `replay_22_sessions.md` §2 shows those days blind until 09:39,
with the feed down 41 min on 09-01 and 14 min on 09-04. 09-25 is COLLECTED with 20 min
asleep. 08-28 and 08-31 had no shares arm at all.

**The configs differ.** Live trades by hash:
- Options: `1f7247d…` to 09-24, `ccd00ac…` from 09-25.
- Shares: `53229d6f…` to 09-04, `b53ca8a5…` to 09-24, `7c6414fc…` from 09-25.

So only **16** COLLECTED sessions ran the hashes the replay will run, one above the
15-session floor. §13.7 does not require filtering each day to its live scope, which the
September "clean" columns did. The options config hash includes `symbols` (`runner.py:105`).
A SPY-only replay (§2) therefore cannot carry `ccd00ac71f8243e9` (§1.1), and it cannot be
compared with a live record that traded QQQ and SPY unless live is filtered to SPY.

**The thresholds are not the measured numbers.** Recomputed from `report.json`:
- Session-net Spearman (scope-matched): shares **0.96** (20 days), options **0.94** (18 days).
  Options fidelity was *not* lower on totals. It was lower only trade for trade (31% exact
  twins). So 0.70 has no September basis, and 0.80 sits far below what was measured.
- At n = 15, a sample Spearman of 0.80 has a 95% interval of about [0.46, 0.94], so the bar
  barely tells a good replay from a bad one.
- $1.32/trade is a *net* gap per trade on clean hours. "Median |per-trade gap| on twinned
  trades" was never measured. The per-twin data needed to calibrate it is not in
  `audit.json`.
- Even on "same signal, same exit" options twins, replay beats live by $5.65/trade (169
  trades, +$1,264 vs +$310). That is close to the $8 bar.
- Count ratios 1.08 / 1.06 support the 0.9–1.3 / 0.8–1.5 bands. Exit-mix 15 pp has no
  September basis.

**Re-admission of PARTIAL sessions:** §0 and §8 use "live sessions from 2026-11-02" as
forward validation with no outcome filter. PARTIAL sessions come back in there.

**Minimal wording change:**
- 13.7: "COLLECTED sessions run under the live config hashes (09-25 onward; 16 at
  `3d21893`), each day filtered to its live symbol and setup scope; options compared on SPY
  only, live filtered to SPY".
- "per-trade gap" → "net gap per trade (the statistic of `replay_22_sessions.md` §4)", or
  state that the twinned-trade median is uncalibrated.
- §8: "forward sessions use the same COLLECTED rule".

### 3. Disjoint sets (§2): FAIL

**Overlap.** Shares holdout runs 2023-01-03 → **2026-09-24**. The §1.4 live-period replay
(2026-08-28 → 10-30, §11 step 4) is run and read before discovery (step 5). So about 19
holdout sessions are read early. `replay_22_sessions.md` already published replay P&L for
08-28 → 09-29.

**Regime source.** `med20_range_bp` from `day_features_SPY.csv` begins **2020-01-06**. I
checked: 1,715 rows, the first 21 blank. The "terciles computed on discovery only" are
undefined for 2016–2019 shares, or else silently computed on 2020–2022 (COVID-heavy) alone.
`day_features_QQQ.csv` covers 2016-01-04 → 2026-09-24.

**Motivation came partly from holdout years.** The news_impact finding that motivates C1
used 2016–2026-09, holdout years included. That was range, not P&L, so it is a caveat and
not a defect.

**Forward:** the trading logic is untouched. The news snapshot (`autostart.py:1122`) runs
before 09:02 (60 s timeout, START_AT 09:05, 3-minute margin). It writes `live_lab_data/news/`,
not runner state. Acceptable, but §2 does not pin a code SHA or hash for forward.

**Minimal wording change:**
- §2 holdout end "2026-09-24" → "2026-08-27".
- Regime: "`med20_range_bp` from `day_features_QQQ.csv` (2016+), or SPY from 2020 with
  2016–2019 shares reported without regime". Pick one before the run.
- §2 Forward row: add "config hashes `7c6414fc549315f3` / `ccd00ac71f8243e9`".

### 4. Information sets (§3, 13.8): FAIL

**Event times have no calendar day.**
- In `events_geopolitical.csv`, `date` is the session the market could react on, and
  `break_time_et` has no date (README: "20:00 on a row dated 2016-01-06 = 20:00 on
  2016-01-05").
- 100 rows are ≥ 16:00 and 75 are before 09:30, some of them on the prior day or a closed
  day. "Compare entry_ts with the event minute on the same session" read literally makes
  every trade on the 100 after-close rows "pre". All of them are post.
- Break times are "to the nearest hour or half-hour from my own knowledge, not read off the
  source" (README). So for the 83 intraday rows the pre/post split is uncertain by ±30–60
  min.
- 63 rows are `unknown`. For day-level use, a row whose break was really after the close
  attaches a session to news the session could not have known.

**08:30 releases.** Every trade is post, so Q5's pre/post split is empty for CPI, NFP, GDP
and 8 of 9 PCE vintages. Only FOMC (14:00) and 10 PCE-at-10:00 rows can split. Release
information at 08:30 is known before the open. That contradicts 13.8's "only (i) can become
a pre-open rule".

**"Not a conditioning variable" is only promised.** No code or test enforces it. The
geopolitical list was itself selected partly by market salience. financial_stress is defined
by market stress and sits inside Q7.

**Minimal wording change:**
- §3: "pre/post is decided only for event minutes inside 09:30–16:00 on `date`; an event
  minute before 09:30 or on an earlier calendar day makes every trade post; `unknown` rows
  are excluded from Q5 and labelled 'time unverified' in Q1".
- 13.8: "(ii) released before 09:30 can become a pre-open rule".
- Add to §1.2's test file a test that the conditioning set contains no column derived from
  the session's own prices.

### 7. Multiple testing (§6 Q7, §7, 13.6): FAIL

Not defined before the data:
- **The family size m.** Are cells below the §7 minimums (≥ 30 sessions, ≥ 50 trades) in the
  BH family of 390 or out of it?
- **financial_stress** is reported-only in Q1 but counted inside Q7's 13 categories.
- **The shortlist** when more than 5 survive.
- **Q7's scale.** Its σ unit (the setup's own daily σ or the arm's) and how no-trade days
  count are unstated.
- **C2's family.** 13.6 makes C2 "never PASS/FAIL unless the observed effect exceeds the
  MDE". So whether the Holm family has 1 or 2 members depends on the data.

Holm on a holdout shortlist does avoid double-dipping in principle, but only once item 3's
overlap is fixed. Counted from the CSV, **no geopolitical category reaches 30 holdout
sessions** (max trade_tariff 27). Geopolitical shortlist rows are therefore untestable on
holdout under §7's own rule.

**Minimal wording change:**
- Q7: "m = cells meeting §7 minimums in discovery, counted and committed before any P&L is
  read; financial_stress excluded; if more than 5 survive, the 5 largest |effect|; σ = the
  arm's discovery daily σ; no-trade days = 0".
- §7: "C1 and C2 form a Holm family of m = 2 regardless of 13.6's power label".

### 8. Power (13.6): FAIL

My recomputed numbers:

| test | SE | MDE | matches the text? |
|---|---|---|---|
| C2 | √(1/48 + 1/202) = 0.1606 | 0.450 | yes; 48 distinct 2023 scheduled sessions, no 2023 overlaps |
| C1 | √(1/185 + 1/745) = 0.0821 | 0.230 | yes, but realised distinct scheduled sessions in 2023-01-03 → 2026-09-24 are **170**, giving SE 0.085 and MDE 0.237 |

The text claims more than the arithmetic supports:
- **PASS needs the point estimate itself to reach 0.25σ.** At a true effect of exactly 0.25σ
  the chance of a PASS is **≈ 50%**. 80% needs a true condition − B0 effect of about
  **0.31σ (C1) / 0.36σ (C2)**. "Adequate for 0.25σ" is wrong.
- **Wrong comparison group.** The SE is for condition vs complement. The test statistic is
  condition − B0 = (1−p) × that, with p ≈ 0.20. The two are mixed up.
- **σ = 1 holds only on discovery** (item 5).
- **Options discovery loses most Thursday releases.** Of 141 scheduled sessions in
  2020–2022, 37 fall on Tue/Thu with no SPY 0DTE before Nov 2022: GDP_ADV 11 of 12, CPI 17
  of 35. This does not affect the 2023 holdout.

**Minimal wording change:** 13.6 should read "C1 ≈ 170 scheduled sessions, SE ≈ 0.085σ.
Because PASS also requires |effect| ≥ 0.25σ, power at a true 0.25σ is ≈ 50%; 80% power
requires ≈ 0.31σ (C1), 0.36σ (C2), in condition − B0 units".

### 12. Association language (13.2): FAIL

13.2 renames things but does not edit the text it renames. These still read as causal or
incremental:
- **§7:** "A PASS says news *awareness* shifts that arm's outcomes". The runner was never
  aware of anything.
- **Title and opening question:** "does news information change what the frozen runners
  earn", "does knowing … change the distribution".
- **§5:** "**Incremental value** is the difference…" is still the live sentence.
- **§8:** "A forward pass is the earliest point at which a dated amendment could even be
  proposed". A forward pass of an association is still an association. 13.2 says only a
  forward *policy* experiment can show value.

**Minimal wording change:**
- §7: → "A PASS says that arm's frozen-runner outcomes on scheduled-release sessions differ
  from baseline by ≥ 0.25σ".
- Title: "change" → "differ with".
- §5: "Incremental value" → "Conditional difference from baseline".
- §8: → "a forward pass can motivate a separate forward policy experiment (13.2); only that
  experiment could support an amendment".

### 11. The one bug class: FAIL (scenario found)

**Scenario: $0 days that are really "no option existed" or "dead feed".**

1. §2 says "the replay skips days without a 0DTE expiry and counts them". `backfill.run`
   does not (`backfill.py:346-379`). The day runs, every signal is written as SKIPPED
   `no_0dte` (`runner.py:443-446`), and `daily/<day>.json` records `atm_net = 0`.
2. In 2020-01 → 2022-11 about 40% of sessions (Tue/Thu) become $0 days. Left in B0 and in the
   discovery σ, they cause two problems:
   - **σ shrinks** by about √0.6, so it comes out roughly 23% too small. Every 2023 holdout
     effect "in σ" is inflated about 1.3×, which makes the 0.25σ bar easier to clear. C2 is
     exactly this case.
   - **Weekday-skewed categories look good by construction.** GDP_ADV is 11 of 12 on
     Thursday in options discovery, so its cell is ≈ 0 − B0. If the runner loses on average,
     GDP "beats baseline" for no reason at all.
3. The same failure hides a dead feed. HTTP 472 and empty bodies come back as `[]`
   (`feed.py:159-166`), and `_cached` pickles them (`backfill.py:91-103`). An empty
   `strikes_…` turns a Monday/Wednesday/Friday into `no_0dte` too. An empty `nbbo_…` turns
   every signal into `no_underlying_quote`, while bars still exist so the session passes
   `MIN_RTH_BARS`. Both look exactly like legitimate structure. This is the same shape as
   news_impact's 25 silently dropped QQQ sessions (`audit_results.md` item 11).

**Does the design catch it?** Partly:
- The weekday-matched placebo would flag a category as "selection".
- The §9 cache guard (spec 003) is promised, but `specs/003` does not exist at `094b848`.
- The §10 funnel counts sessions skipped for "no 0DTE / thin / no bars", **not sessions
  that ran with every fill skipped**.

**Minimal wording change:**
- §2: "options sessions where `zero_dte` is None are excluded from every options table, from
  B0 and from σ, and counted by weekday; a Mon/Wed/Fri `no_0dte` or a session with every
  fill skipped for `no_underlying_quote` is a defect that stops the run".
- §10: add "sessions run with zero fills, by skip reason, by year".

---

## QUESTIONABLE

### 5. Metrics (§4)
Standardised daily return is a sound way to compare arms. But σ comes from discovery and is
applied to holdout, so holdout σ ≠ 1. The shares holdout covers a different years mix, and
the options σ is zero-inflated (item 11). So 13.6's "σ = 1 by construction" does not hold on
the data being tested.

The options-specific metrics can be computed from the trade records (`entry_spread_pct`,
`entry_ask`, `exit_reason`, `underlying_return` are all present). But the replay's exit
reasons include `failed` (114), `bars` (102) and `trail_imb` (162), which are not in the
§4/13.7 list (stop/target/time/eod/trail).

*Would change the verdict:*
- report holdout σ next to discovery σ and the ratio;
- fix the mapping of exit reasons to the five §4 classes before the run.

### 6. Baselines and placebo (§5, 13.3)
Condition − B0 is a legitimate estimand. But B0 contains the condition, so the effect is
(1−p)·δ, and p runs from about 0.03 (FOMC) to about 0.20 (`scheduled_today`). A 0.25σ bar
therefore means a different underlying difference for each category. The placebo
statistics are centred at −p·δ, not 0, which leans mildly toward "effect".

"p95 ≥ half the effect ⇒ selection" means an effect must reach about 2 × 1.645 ≈ 3.3 SE to
escape the label. That is stricter than the 1.96-SE CI rule. It is inherited from
news_impact A1 and is a heuristic, not derived. There is no rule for negative effects
(p5? |p95|?), and nothing says whether the flag overrides a C1 PASS.

Leakage: none. 13.3 excludes the condition's sessions from the placebo pool.

*Would change the verdict:*
- state the sign rule for negative effects;
- state whether a placebo flag overrides C1/C2;
- report condition − complement next to condition − B0.

### 9. Overlap (13.4)
For exploratory tables, counting in each cell plus a single-category sensitivity line is
enough; a joint model is not needed at these overlap sizes. Scheduled-with-scheduled is
handled. Counts I reproduced:

| overlap | sessions |
|---|---|
| CPI + FOMC | 7 |
| GDP + PCE | 5 |
| FOMC + GDP | 3 |
| multi-category sessions | 73 of 758 |
| trade_tariff on scheduled days | 16 of 70 |

The sensitivity line is attached to "every Q1–Q7 table", which leaves out **C1/C2**, the
only confirmatory rows.

*Would change the verdict:* extend 13.4's single-category line to C1 and C2.

### 10. 2016–2019 structure (§9)
Reporting every table by year describes the problem but does not stop it steering the
analysis. 2016–2019 is 4 of the 7 discovery years (about 57% of sessions). The BH survivors
and shortlist are pooled over discovery, so they are drawn mostly from the years whose
spreads, tick regimes and ETF volumes are furthest from the 2026 market the shares rules
were written on. The regime variable is also missing for those years (item 3).

The failure is in the safe direction: a 2016–2019 artefact would most likely fail a
2023–2026 holdout. The cost is holdout slots spent on artefacts, not false confirmations.

**Recommendation: QUESTIONABLE.** A later start (e.g. 2018) is defensible, but this audit
does not choose it.

*Would change the verdict:* either the per-year funnel (trades per session, median
`entry_spread_bp`) shows 2016–2019 inside the 2023–2026 range, or the owner fixes the start
date with that reason written down before discovery runs.

---

## PASS

None of the 12 items PASS in full. Some sub-checks did reproduce and need no fix:
- 13.6 arithmetic: 0.1606 / 0.450 and 0.0821 / 0.230.
- §13.3 placebo pool excludes the condition's own sessions.
- `_closed`, `_no_future` and the bisect mechanics are correct.
- The holiday zero-grid is rejected by the zero-price filter (`feed.py:374, 405`) together
  with `MIN_RTH_BARS`.
- `med20` excludes today (news_impact audit item 6, recomputed there).
- Scheduled `session` alignment is correct for the 6 holiday-shifted rows.

---

## Checks I could not run (each is an unknown, not a pass)

- **What vendor stamps mean** (item 1, test h). I had no ThetaData access from here, and the
  §12 probes are blank.
- **Barrier tests (a)–(g) and spec 003.** `backfill_barrier_test.py` and `specs/003-*` do not
  exist at `094b848`. I could not confirm they fail when their guards are removed.
- **Median |per-trade gap| on twinned trades.** `audit.json` holds only bin totals, so I could
  not calibrate the $3/$8 bars.
- **Feed depth** (1-minute stock NBBO to 2016, chains to 2020) on the current tier. Unknown
  until the §12 probe. The ThetaData status is "UNKNOWN" in `CLAUDE.md`.
- **Missingness correlated with outcome** (zero-filled minutes and missing quotes on volatile
  days). This needs replay output, so it is not applicable before gate 2.
- **Overlap matrix including the 63 `unknown`-time rows' true sessions.** That cannot be
  verified from the files.
