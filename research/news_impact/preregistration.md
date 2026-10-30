---
task_id: R-2026-10-30-01
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
parent: research/ARCHITECTURE_AND_BUSINESS_REVIEW_2026-10-23.md
role: orchestrator
repo_sha: ea65bb1
inputs:
  - "owner's request 2026-10-29/30: news impact on SPY and QQQ, ranged-day count, lab profitable days; 'yes to all, agent builds the geopolitical list, go'"
  - research/six_lines_results.md@ea65bb1   (8.2% of QQQ sessions broke no line)
  - live_lab_data/@ea65bb1                   (lab record, 42 sessions)
decision_rule: "section 6, fixed before any feature file exists"
outputs:
  - research/news_impact/events_scheduled.csv        (agent: calendar)
  - research/news_impact/events_geopolitical.csv     (agent: geopolitical list)
  - trade_analysis/news_impact/day_features.py + _test.py   (agent: feature script; runs on the laptop)
  - trade_analysis/news_impact/analyze.py + _test.py         (agent: analysis script)
  - research/news_impact/audit_preregistration.md    (statistical-auditor, gate 1)
  - research/news_impact/redteam.md                  (strategy-contrarian)
  - research/news_impact/results.md                  (orchestrator, after gate 2)
assumptions:
  - "Finlight free plan holds about one month of history and no sentiment; it is NOT a data source for this study, only for a later forward log"
  - "10 years of 1-minute QQQ and SPY bars incl. premarket are reachable on the laptop through levels_test.get_ext (proven 2026-09-21 and 2026-10-26); nothing here runs in the cloud sandbox"
open_questions: []
errors: []
status: approved by the owner 2026-10-30 ("yes to all ... go"); frozen
verified_by: null
budget: {agents: 6, passes: 1 each, laptop_runs: 1 feature extraction}
---

# Pre-registration — did news move SPY and QQQ, how much, and what did the days look like when it did not

## 1. Questions, in the owner's words, and what each becomes

| he asked | the measurable version |
|---|---|
| "what news affected SPY and QQQ and how much" | on days with a dated event, how far did the day's gap and range sit outside its own recent normal, by event type |
| "if it did not affect then how did it move" | on event days inside the normal band, the split of day types (trend / ranged / chop), against non-event days |
| "how many days was it ranged" | count and share of ranged days by two fixed definitions, overall, by year, event vs non-event |
| "how many days was MY strategy profitable" | lab record: green days per arm, both arms, and on event vs non-event days |
| "fuzzy logic metric" | two relative measures with three bands each, thresholds set in §4, never tuned |

## 2. Data

| series | source | period | where it runs |
|---|---|---|---|
| QQQ, SPY 1-minute bars, 04:00–16:00 ET | `levels_test.get_ext` (ThetaData stock history, cached in `live_lab_data/ext_cache`) | 2016-01-04 → 2026-10-29 | laptop |
| lab record | `live_lab_data/trades.jsonl`, `shares/trades.jsonl`, `session_ledger.jsonl` | 2026-08-28 → today | anywhere (in git) |
| scheduled events | public release calendars, one source URL per date | 2016-01 → 2026-10 | agent, cloud |
| geopolitical events | agent-built list, one source URL per row, categories fixed in §3 | 2016-01 → 2026-10 | agent, cloud |

Not used: Finlight (no history on the free plan), anything that needs a purchase.

## 3. Events

### 3.1 Scheduled (objective dates; file `events_scheduled.csv`)

Columns: `date` (ET release date), `type`, `time_et`, `source_url`, `note`.
Types, all included (owner: "yes to all"):

| type | what counts |
|---|---|
| `FOMC` | the statement day of each scheduled meeting (not minutes, not speeches) |
| `CPI` | BLS Consumer Price Index release day |
| `NFP` | BLS Employment Situation release day |
| `GDP_ADV` | BEA advance estimate of quarterly GDP |
| `PCE` | BEA Personal Income and Outlays (PCE price index) release day |

Unscheduled Fed actions (e.g. 2020-03-03, 2020-03-15) go in the geopolitical file under
`financial_stress`, not here: this file is for days the market knew about in advance.

### 3.2 Geopolitical and macro shocks (judgement-laden; file `events_geopolitical.csv`)

Columns: `date` (first US trading day on which the market could react; if news broke after
16:00 ET or on a weekend, the next session), `category`, `headline` (≤ 12 words),
`source_url`, `surprise` (`yes` if unscheduled, `no` if a scheduled vote/deadline).
Categories, fixed, nothing else allowed:

`war_military` · `trade_tariff` · `financial_stress` · `us_fiscal` (shutdown, debt
ceiling, downgrade) · `pandemic_health` · `political_shock` (election result, resignation,
assassination attempt, impeachment vote) · `energy_shock` · `cyber_infrastructure`

Inclusion rule for the agent: the event led the front page of at least two of Reuters, AP,
WSJ, FT or Bloomberg on that day, and had a plausible channel to US equities, rates or the
dollar. Target 150–300 rows over the decade. **The list is built before any feature file
exists and is frozen at its first commit**; rows may be corrected for date errors with a
note, never added after results are seen.

## 4. Day features (file `research/news_impact/day_features_{SYMBOL}.csv`, one row per session)

Built by `trade_analysis/news_impact/day_features.py` on the laptop. Imports
`six_lines.build_six`, `six_lines.breaks_today`, `levels_test.get_ext`,
`walkforward.sessions_between`; edits none of them.

| column | definition |
|---|---|
| `date` | session date |
| `pre_high`, `pre_low` | 04:00–09:29 extremes |
| `open` | first 09:30 1-min bar open |
| `high`, `low`, `close` | 09:30–15:59 |
| `prev_close` | previous session's close |
| `gap_bp` | (open / prev_close − 1) × 10,000 |
| `range_bp` | (high / low − 1) × 10,000 |
| `co_bp` | (close / open − 1) × 10,000 |
| `first60_hi_bp`, `first60_lo_bp` | 09:30–10:29 high and low, as bp from open |
| `med20_range_bp` | median `range_bp` of the previous 20 sessions (today excluded) |
| `med20_absgap_bp` | median \|gap_bp\| of the previous 20 sessions |
| `rel_range` | `range_bp / med20_range_bp` |
| `rel_gap` | \|`gap_bp`\| / `med20_absgap_bp` |
| `n_line_breaks` | six-line close-breaks today (`breaks_today`), lines from yesterday + today's premarket |
| `ranged_a` | `n_line_breaks == 0` (his lines; September tally: 8.2% on QQQ) |
| `ranged_b` | `rel_range < 0.6` |
| `day_class` | `trend` if \|`co_bp`\| / `range_bp` ≥ 0.6; else `ranged` if `ranged_a` or `ranged_b`; else `chop` |
| `gap_reversal` | `rel_gap ≥ 1.0` and sign(`co_bp`) ≠ sign(`gap_bp`) |
| `thin` | fewer than 300 RTH bars (half days, data holes): excluded from every statistic, kept in the file |

**Impact bands (the "fuzzy" part, thresholds fixed here):**

| measure | low | medium | high |
|---|---|---|---|
| `rel_range` | < 0.8 | 0.8 – 1.5 | > 1.5 |
| `rel_gap` | < 1.0 | 1.0 – 2.5 | > 2.5 |

`impact` = the higher of the two bands. A day is **affected** when `impact == high`. The
bands are reported as such; no composite score is invented, because a composite hides which
of the two moved.

**Lab join** (`research/news_impact/lab_days.csv`, built from git, no laptop): per session
date, `options_atm_net`, `shares_net`, `options_green`, `shares_green`, `both_green`,
`ledger_outcome`. Computed from the record as it stands at the run's `repo_sha`.

## 5. Agents, one task each

| role | writes | may not |
|---|---|---|
| calendar agent (general-purpose) | `events_scheduled.csv` | touch any other file |
| geopolitical agent (general-purpose) | `events_geopolitical.csv` | add categories; add rows after features exist |
| feature-script agent (general-purpose) | `day_features.py`, `day_features_test.py` | edit anything under `live_lab/`; read `live_lab_data/` except `ext_cache` |
| analysis-script agent (general-purpose) | `analyze.py`, `analyze_test.py` | change a threshold or a test in §4/§6 |
| statistical-auditor (gate 1, gate 2) | `audit_preregistration.md`, `audit_results.md` | edit anything else |
| strategy-contrarian (once) | `redteam.md` | edit anything else |

Every output file carries the §0 front matter. The orchestrator (this session) runs the
tests, runs nothing on the laptop (the owner does), writes `results.md` only after gate 2,
and is the only one who commits to `main`.

## 6. Decision rules, fixed now

Unit of analysis: the session. Statistic for "how much": the **median** `rel_range` and
median `rel_gap`, because both are heavy-tailed. Uncertainty: bootstrap over sessions,
3,000 reps, seed 20260828, 95% CI. Event days of one type are compared with all
non-event days (no event of any type that day) in the same symbol.

| id | claim tested | pass condition | correction |
|---|---|---|---|
| **H1 (primary)** | FOMC days have a larger `rel_range` than non-event days, QQQ | CI of the median difference excludes 0 and the difference is ≥ 0.2 (a fifth of a normal day's range) | none; one primary |
| H2 | same as H1 for CPI, NFP, GDP_ADV, PCE, and for each geopolitical category with ≥ 30 rows | as H1 | Holm across all types tested |
| H3 | share of **affected** days (impact = high) on event days vs non-event days, per type | CI of the difference in shares excludes 0 | Holm as H2 |
| H4 | ranged-day share (`ranged_a`, `ranged_b`), overall, by year, event vs non-event | descriptive, reported with CIs, no pass/fail | — |
| H5 | on event days that were **not affected**, the `day_class` split vs non-event days | descriptive, chi-square p reported, no pass/fail | — |
| H6 | lab: green days per arm on event vs non-event sessions | descriptive only; 42 sessions cannot carry a test | — |

Minimum sample: a type with fewer than **30** event days is reported, not tested. SPY is run
as a replication of every QQQ result; a QQQ result that does not hold on SPY is reported
as "not replicated", not averaged away.

What a PASS on H1 would and would not mean: it would mean scheduled Fed days are wider than
ordinary days, which is widely believed and would mostly confirm the method works. It
says nothing about direction and nothing about whether any of his setups do better on
those days; that would be a new pre-registration with its own rule.

## 7. Gates

- **Gate 1**, before the laptop run: the statistical-auditor reads this file, both event
  CSVs and both scripts' tests, and emits PASS/FAIL per §4 definition and §6 rule
  (lookahead in `med20_*`, event-date alignment to ET sessions, thin-day handling, the
  Holm family, whether `ranged_a` leaks today's data into the lines). A FAIL is fixed in
  this file with a dated note before anything runs.
- **Gate 2**, after the run: the auditor reads `results.md` against the JSON and the CSVs,
  checks every quoted number, and marks each claim PASS / FAIL / QUESTIONABLE. Nothing
  enters `CLAUDE.md` without a PASS.

## 8. Out of scope, on purpose

Finlight text (forward log only, separate spec); any trading rule; intraday reaction
timing around the release minute (a second study, if H1 passes); anything that changes
`live_lab/` code.
