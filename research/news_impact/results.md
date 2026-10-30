---
task_id: R-2026-10-30-08
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
parent: R-2026-10-30-01
role: orchestrator
repo_sha: e39758a
inputs:
  - research/news_impact/preregistration.md@ecccb5a   (with amendment 9)
  - research/news_impact/day_features_QQQ.csv@e39758a  (2,697 sessions, 2016-01-04 → 2026-10-29, laptop run)
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
status: done, awaiting gate 2
verified_by: null
budget: {agents: 6, laptop_runs: 2, cloud_run: "analyze.py, 3000 bootstrap reps, 200 placebo draws"}
---

# Results — did news move QQQ and SPY, and what did that do for the strategy

Run 2026-10-30 on the features the laptop produced the same day. Primary symbol QQQ,
2,655 usable sessions over ten years; SPY as a second-symbol check, 1,681 usable sessions
over seven. "Normal day" = the median of the previous 20 sessions. "rel_range" = today's
high-to-low range divided by that normal. "Affected" = range above 1.5× normal or gap above
2.5× normal gap.

## 1. The verdict

**H1 PASS.** On Fed decision days QQQ's range was **27% wider than a normal day** (median
rel_range difference +0.27, 95% CI [+0.12, +0.35], 84 Fed days against 2,165 days with no
scheduled release). Holds on SPY. This is what everyone expected and it mostly confirms the
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

On SPY (2020–2026): Fed, CPI and jobs days all pass the "share of big days" test (47%, 45%,
44% against 28%); none of the range medians survive correction; GDP has 27 days, under the
30 needed. Read together: **on a Fed, CPI or jobs day, about 4 days in 10 are big ones,
against 1 in 4 normally.** The median release day is only a little wider than usual; the
tail is what moves.

## 3. Geopolitical events: mostly selection, descriptive only

Pre-registered as descriptive because the list is chosen by what made headlines, and
headlines follow market moves. Placebo: 200 draws of random days matched by year and weekday.

| category | QQQ days | range vs normal | placebo 95th pct | verdict |
|---|---|---|---|---|
| trade / tariff | 69 | +29% | +11% | not explained by placebo |
| war / military | 53 | +19% | +9% | not explained by placebo (SPY: +31% vs +14%, same) |
| pandemic | 16 | +57% | +22% | not explained, but 16 days |
| political shock | 61 | +4% | +13% | **selection** |
| US fiscal | 34 | +10% | +20% | **selection** |
| financial stress | 32 | +1% | +21% | **selection** (never testable by construction) |
| energy shock | 31 | +1% | +16% | **selection** |
| cyber / infrastructure | 10 | +29% | +39% | **selection** |

Five of eight categories are no wider than random days from the same years. Tariff and war
days are, on both symbols. That is a description, not a test, and it stays one.

## 4. How often is the day ranged (QQQ, ten years)

| definition | share | in plain terms |
|---|---|---|
| none of his six lines broke on a close | **8.2%** (CI 7.2–9.3) | about 1 day in 12 |
| range under 0.6× normal | **12.8%** (CI 11.5–14.0) | about 1 day in 8 |

By year, "no line broke" has risen from 3–4% in 2016–2018 to 13% in 2024–2025 and 17.5% in
2026 so far (183 sessions). Event days are slightly more often line-contained (10.2% vs
7.5%) and slightly less often narrow (10.3% vs 13.7%). SPY: 10.8% and 13.7%, flat by year.

## 5. When the news did not move it (QQQ)

On release days that stayed inside the normal band, the day-type split (trend / ranged /
chop) was **147 / 73 / 230**, against **464 / 262 / 711** on quiet non-event days. Chi-square
p = 0.61: the same mix. A release that does not land big leaves an ordinary day. (SPY
p = 0.025, descriptive; the difference is a few more trend days.)

## 6. Can you tell by 10:30 what kind of day it is? Mostly no.

Share of days whose first-hour move had already exceeded 0.6× a normal full-day range by
10:30 ET:

| the day turned out to be | QQQ | SPY |
|---|---|---|
| trend | **35%** | 24% |
| chop | 27% | 23% |
| ranged | 8% | 9% |

Trend and chop days look alike at 10:30 (35% vs 27%). The first hour tells you a quiet day
is quiet; it does not tell you a big day will trend rather than chop. The red team predicted
exactly this (`redteam.md` §3): size is knowable early, direction and persistence are not.

## 7. The lab on event days (44 sessions, counts only)

| | sessions | options green | shares green | both |
|---|---|---|---|---|
| event day | 14 | 6 of 13 | 6 of 14 | 4 of 13 |
| non-event | 30 | 8 of 25 | 12 of 26 | 6 of 23 |

No test, by pre-registration, and none would be meaningful. There is no sign of the lab
doing better or worse on news days.

## 8. What this does and does not change

- `CLAUDE.md` gets one new settled line after gate 2: **scheduled releases widen the day
  (Fed +27%, 4 in 10 release days are big vs 1 in 4), direction untouched, trend not
  knowable by 10:30.**
- No day filter follows from this. Conditioning the lab on "Fed day" or "tariff day" would
  select wider days, and width without direction is the thing that has been losing money.
- The geopolitical list stays in the repo as a dated, sourced reference, labelled as what it
  is. Anyone wanting to test it needs a list built without looking at markets.
- The lab's Finlight daily snapshot (owner approved) is a separate small spec; it collects
  forward data and makes no claim.

## 9. Defects recorded during the run

- Theta Terminal's upstream link was dead for an unknown period before 2026-10-30; every
  uncached history request returned nothing and the scripts counted the days as "no bars".
  Restarting the terminal fixed it. The same state produced the zero-trade first
  `his_rules` run on 2026-10-26. The lab's morning preflight checks this link; manual
  research runs did not. Rule added to the operating notes: a backtest that prints zeros
  for its first hundred days has a dead feed, not a null.
- SPY 2016–2019 was not fetched (the link was fixed after the run was restarted from 2020).
  Not re-run: SPY is a robustness check and seven years are enough for it.
