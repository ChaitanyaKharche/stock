---
task_id: R-2026-10-30-08
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
parent: R-2026-10-30-01
role: orchestrator
repo_sha: e39758a
inputs:
  - research/news_impact/preregistration.md@ecccb5a   (with amendment 9)
  - research/news_impact/day_features_QQQ.csv@e39758a  (2,697 sessions, 2016-01-04 → 2026-09-24, laptop run; the last 25 sessions are missing, see §9)
  - research/news_impact/day_features_SPY.csv@e39758a  (1,714 sessions, 2020-01-06 → 2026-10-29, laptop run)
  - research/news_impact/events_scheduled.csv@550b4a5   (516 rows)
  - research/news_impact/events_geopolitical.csv@3facddc (321 rows, frozen)
  - research/news_impact/lab_days.csv@4350317           (44 sessions)
  - research/news_impact/analysis.json                  (this run's output, unedited)
decision_rule: "preregistration section 6 as amended in section 9; H1 primary"
outputs:
  - research/news_impact/results.md
evidence:
  - "every number below is in analysis.json; the auditor checks them at gate 2"
assumptions:
  - "SPY covers 2020-2026 only: the stock feed could not fetch 2016-2019 SPY before the run; coverage is stated wherever SPY is quoted"
open_questions: []
errors:
  - "first SPY feature run returned zero rows for every uncached date; cause was a dead Theta Terminal upstream link, fixed by restarting the terminal; the same failure produced the zero-trade first his_rules run on 2026-10-26"
status: done; corrected in place 2026-10-30 after gate 2 (see the note under the title)
verified_by: R-2026-10-30-09 (gate 2: H1 reproduced independently; five write-up errors found and corrected below)
budget: {agents: 6, laptop_runs: 2, cloud_run: "analyze.py, 3000 bootstrap reps, 200 placebo draws"}
---

# Results — did news move QQQ and SPY, and what did that do for the strategy

**Corrected 2026-10-30 after gate 2 (`audit_results.md`).** The auditor reproduced the primary
result from the raw CSV (+0.2687, its own seeds) and found five errors in the first draft of
this file, all in the words, none in the numbers of `analysis.json`. Each is fixed in place
and marked ⟨corrected⟩: (1) SPY tariff days are "selection", not "not explained"; (2) SPY's
not-affected event days have *fewer* trend days, not more; (3) QQQ coverage ends 2026-09-24,
not 10-29, because the dead feed dropped 25 sessions (§9); (4) "27% wider" was loose, the
median Fed day is 1.23× normal against 0.96× on control days; (5) SPY was said to "survive
correction" though SPY is in no family. Also added: the thin-by-year table and the funnel
amendment A6 requires, and a corrected §6 sentence.

Run 2026-10-30 on the features the laptop produced the same day. Primary symbol QQQ,
2,655 usable sessions over ten years; SPY as a second-symbol check, 1,681 usable sessions
over seven. "Normal day" = the median of the previous 20 sessions. "rel_range" = today's
high-to-low range divided by that normal. "Affected" = range above 1.5× normal or gap above
2.5× normal gap.

## 1. The verdict

**H1 PASS.** On Fed decision days QQQ's range was **about a quarter of a normal day's range
wider** ⟨corrected⟩: median 1.23× normal on the 84 Fed days against 0.96× on the 2,165 days
with no scheduled release, difference +0.27, 95% CI [+0.12, +0.35]. Holds on SPY (2020–2026). This is what everyone expected and it mostly confirms the
method works.

**For the trader, the answer to "what are we missing" is: nothing that helps.** News days
are wider. Width was never the problem. Direction is, and this study did not touch
direction, by design (`CLAUDE.md`: magnitude, not sign, four times over; now five).

## 2. Scheduled releases: how much (QQQ, Holm-corrected within each family)

| release | days | range vs normal | share of big days (normal: 27%) |
|---|---|---|---|
| Fed decision | 84 | **+27%** PASS (primary) | 39%, Holm p 0.052, just misses |
| GDP advance | 42 | **+26%** PASS | 41%, not significant |
| Jobs report | 126 | +12%, not after correction | **44%** PASS |
| CPI | 127 | +1%, nothing | **39%** PASS |
| PCE | 125 | +7%, nothing | 30%, nothing |

Fed days' share of big days (39% vs 27%) **fails** the pre-registered rule (Holm p 0.052);
it is reported as such and SPY is not used to rescue it ⟨corrected⟩. SPY (2020–2026) is a
robustness check in no family: there the Fed, CPI and jobs shares are 47%, 45%, 44% against
28%, same direction as QQQ; the SPY range medians are reported with CIs only ⟨corrected: the
draft said "survive correction"⟩. GDP has 27 SPY days, under the 30 needed.

Read together: **on CPI and jobs-report days about 4 in 10 are big days against 1 in 4
normally (PASS on QQQ), and most of that "big" is the opening gap after the 08:30 release,
not the range afterwards.** The median release day is only a little wider than usual; the
tail is what moves. The pooled "any release" share was never tested and is not claimed.

## 3. Geopolitical events: mostly selection, descriptive only

Pre-registered as descriptive because the list is chosen by what made headlines, and
headlines follow market moves. Placebo: 200 draws of random days matched by year and weekday.

| category | QQQ days | range vs normal | placebo 95th pct | verdict |
|---|---|---|---|---|
| trade / tariff | 69 | +29% | +11% | placebo 95th pct below observed (SPY: +21% vs +23%, **selection**) ⟨corrected⟩ |
| war / military | 53 | +19% | +9% | placebo 95th pct below observed (SPY: +31% vs +14%, same) |
| pandemic | 16 | +57% | +22% | placebo 95th pct below observed, but 16 days |
| political shock | 61 | +4% | +13% | **selection** |
| US fiscal | 34 | +10% | +20% | **selection** |
| financial stress | 32 | +1% | +21% | **selection** (never testable by construction) |
| energy shock | 31 | +1% | +16% | **selection** |
| cyber / infrastructure | 10 | +29% | +39% | **selection** |

Five of eight categories are no wider than random days from the same years. War days sit
above their placebo on both symbols; tariff days do on QQQ only ⟨corrected⟩. "Above the
placebo" is a description of this list, not a test of anything, and it stays one.

## 4. How often is the day ranged (QQQ, ten years)

| definition | share | in plain terms |
|---|---|---|
| none of his six lines broke on a close | **8.2%** (CI 7.2–9.3) | about 1 day in 12 |
| range under 0.6× normal | **12.8%** (CI 11.5–14.0) | about 1 day in 8 |

By year, "no line broke" has risen from 3–4% in 2016–2018 to 13% in 2024–2025 and 17.5% in
2026 through 09-24 (183 sessions) ⟨corrected: coverage⟩. **QUESTIONABLE per gate 2:** the
rise tracks how wide the premarket range is relative to a normal day (1.5% → 20.7% across
quintiles of that ratio), SPY shows no rise over the same years, and the CSV cannot tell
whether QQQ's premarket got wider or the premarket bar coverage changed. Unresolved until
bar counts from the laptop cache are checked. Event days are slightly more often line-contained (10.2% vs
7.5%) and slightly less often narrow (10.3% vs 13.7%). SPY: 10.8% and 13.7%, flat by year.

## 5. When the news did not move it (QQQ)

On release days that stayed inside the normal band, the day-type split (trend / ranged /
chop) was **147 / 73 / 230**, against **464 / 262 / 711** on quiet non-event days. Chi-square
p = 0.61: the same mix. A release that does not land big leaves an ordinary day. (SPY
p = 0.025, descriptive; the difference is **fewer** trend days, 31.5% vs 33.6%, and more
chop, 52.9% vs 44.6% ⟨corrected: the draft had the direction backwards⟩.)

## 6. Can you tell by 10:30 what kind of day it is? Mostly no.

Share of days whose first-hour move had already exceeded 0.6× a normal full-day range by
10:30 ET:

| the day turned out to be | QQQ | SPY |
|---|---|---|
| trend | **35%** | 24% |
| chop | 27% | 23% |
| ranged | 8% | 9% |

Trend days are somewhat more often already moving by 10:30 than chop days (35% vs 27% on
QQQ; the CIs [31.6, 37.9] and [24.6, 29.4] do not overlap) ⟨corrected: the draft said "look
alike"⟩. But 65% of trend days had **not** moved by then, and a quarter of chop days had, so
the first hour sorts big-from-quiet far better than trend-from-chop. The ranged row is
partly true by construction: a narrow day cannot have made a big first-hour move. The red
team predicted this shape (`redteam.md` §3): size is knowable early, persistence is not.

## 7. The lab on event days (44 sessions, counts only)

| | sessions | options green | shares green | both |
|---|---|---|---|---|
| event day | 14 | 6 of 13 | 6 of 14 | 4 of 13 |
| non-event | 30 | 8 of 25 | 12 of 26 | 6 of 23 |

No test, by pre-registration, and none would be meaningful. There is no sign of the lab
doing better or worse on news days.

## 8. What this does and does not change

- `CLAUDE.md` gets one new settled line, worded to the PASS-level findings only ⟨corrected
  after gate 2⟩: **Fed decision days: QQQ range median 1.23× normal vs 0.96× on
  no-release days, +0.27 [0.12, 0.35], 84 days 2016–2026-09, holds on SPY 2020–26; GDP
  advance +0.26 (Holm 0.011); CPI and jobs days have 4 in 10 big days vs 1 in 4, mostly the
  opening gap. Everything else in `news_impact/` is descriptive. Direction was not
  measured.**
- No day filter follows from this. Conditioning the lab on "Fed day" or "tariff day" would
  select wider days, and width without direction is the thing that has been losing money.
- The geopolitical list stays in the repo as a dated, sourced reference, labelled as what it
  is. Anyone wanting to test it needs a list built without looking at markets.
- The lab's Finlight daily snapshot (owner approved) is a separate small spec; it collects
  forward data and makes no claim.

## 8a. Coverage tables required by amendment A6

QQQ funnel: 2,824 weekdays tried → 127 with no bars (about 102 holidays and **25 sessions
lost to the dead feed, 2026-09-25 → 10-29**, see §9) → 2,697 rows → 21 thin (half days)
excluded → 21 warm-up rows without a 20-day history excluded → **2,655 usable**.
SPY funnel: 1,779 tried → 65 no bars → 1,714 rows → 12 thin → 21 warm-up → **1,681 usable**.

| thin by year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QQQ | 1 | 2 | 3 | 3 | 2 | 1 | 1 | 2 | 3 | 3 | 0 |
| SPY | – | – | – | – | 2 | 1 | 1 | 2 | 3 | 3 | 0 |

Usable event sessions, QQQ: FOMC 84 of 86, CPI 127 of 129, NFP 126 of 129, GDP 42 of 44,
PCE 125 of 128. SPY (2020–26 only): FOMC 53, CPI 80, NFP 79, GDP 27, PCE 80.

## 9. Defects recorded during the run

- Theta Terminal's upstream link was dead for an unknown period before 2026-10-30; every
  uncached history request returned nothing and the scripts counted the days as "no bars".
  Restarting the terminal fixed it. The same state produced the zero-trade first
  `his_rules` run on 2026-10-26. The lab's morning preflight checks this link; manual
  research runs did not. Rule added to the operating notes: a backtest that prints zeros
  for its first hundred days has a dead feed, not a null.
- **The dead feed also removed QQQ's last 25 sessions (2026-09-25 → 10-29) and the cache
  made it permanent** ⟨found at gate 2⟩: `levels_test.get_ext` pickles whatever the feed
  returns, including an empty response, so those dates now read as "no bars" on every
  re-run until the 25 cache files are deleted. Effect on this study: one Fed day and about
  1% of rows; the headline does not move. Effect on the project: a proven helper that
  caches a feed outage as a fact. Fix is a one-line guard (never cache an empty response),
  spec-first because the file sits under `live_lab/`.
- SPY 2016–2019 was not fetched (the link was fixed after the run was restarted from 2020).
  Not re-run: SPY is a robustness check and seven years are enough for it.
