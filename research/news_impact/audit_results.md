---
task_id: R-2026-10-30-09
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
parent: R-2026-10-30-08
role: statistical-auditor
repo_sha: e856c6f
inputs: [results.md, analysis.json, preregistration.md, the two feature CSVs, analyze.py, day_features.py]
decision_rule: "PASS/FAIL/QUESTIONABLE per item"
outputs: [research/news_impact/audit_results.md]
evidence: []
assumptions: []
open_questions: []
errors: []
status: done
verified_by: null
budget: {cap: "one pass"}
---

# Gate 2 audit: news-impact results

**Short version: the main finding holds. On Fed days QQQ's range is wider than on
no-release days. I reproduced that from the CSV on my own (+0.269). The write-up around
it does not hold up as well.** Two sentences contradict the JSON. The QQQ file silently
stops on 2026-09-24, not 10-29. The proposed `CLAUDE.md` line turns descriptive and
untested results into settled ones. None of it should enter `CLAUDE.md` as written.

How I checked: I rebuilt every group, median, share, count and Holm adjustment from the
two feature CSVs and the two event CSVs with my own pandas code. I did not use
`analyze.py`'s functions. The bootstrap I re-ran with my own resampler and other seeds.
Everything in `analysis.json` that I recomputed matches to the printed precision. Where
the write-up disagrees, it disagrees with the JSON, not with the data.

## Verdict table

| # | Claim / item | Verdict | Reason |
|---|---|---|---|
| 1 | Every number in results.md §1-7 traces | **FAIL** | §3 "tariff ... on both symbols" is false for SPY (flag = selection). §5 SPY "a few more trend days" is backwards. The 2026 "so far" figure covers only up to 09-24. |
| 4b | §3 stays descriptive | **FAIL** | "Tariff and war days are, on both symbols." is a claim, and it is wrong for SPY tariff (0.210 vs p95 0.230). |
| 5 | Gate-1 FAIL 11 fixes honoured (A6 thin-by-year table, funnel) | **FAIL** | A6 requires `results.md` to print thin sessions by year. It does not, and there is no days-tried funnel. That omission hid a 25-session QQQ hole (item 11). |
| 9b | "trend and chop days look alike at 10:30 (35% vs 27%)" | **FAIL** | On QQQ the two CIs do not overlap: [31.6, 37.9] vs [24.6, 29.4]. The sentence holds for SPY only (24% vs 23%). |
| 10 | Proposed `CLAUDE.md` line | **FAIL** | Only "Fed +27%" is PASS-level. "Scheduled releases widen the day", "4 in 10 release days", "direction untouched" and "trend not knowable" are untested, null, or not measured. |
| 11 | A confident number that measured nothing | **FAIL (found)** | QQQ ends 2026-09-24. 25 sessions were dropped by the dead feed as `no_bars`, while results.md says 10-29. `levels_test.py:69` caches the empty responses. |
| 3 | Families and Holm; "4 in 10 vs 1 in 4" | QUESTIONABLE | QQQ families and Holm are exact. But the code runs Holm on SPY (`analyze.py:558-559, 582-583`) against A3, and the "4 in 10" sentence includes FOMC, which failed H3 on QQQ (Holm p 0.052). |
| 5 | Gate-1 FAIL 4 and FAIL 9 fixes | QUESTIONABLE | Implemented for QQQ. SPY gets its own Holm families, and §2 uses SPY to support FOMC's failed H3. |
| 8 | ranged_a rise from 3.5% to 17.5% | QUESTIONABLE | It follows premarket range width (quintile 1: 1.5%, quintile 5: 20.7%). Whether that width is a market fact or feed coverage cannot be told from the CSV. |
| 2 | H1 run as pre-registered; point estimate | **PASS** | Reproduced: FOMC median 1.2330 − control median 0.9644 = **+0.2687**, n 84 vs 2,165. Independent bootstrap CI [0.12, 0.36]. |
| 4a | Placebo "selection" flag applied as A1 states | **PASS** | `analyze.py:418-420` is p95 ≥ 0.5 × observed. All 16 rel_range flags (QQQ and SPY) recomputed from the JSON values; all match. |
| 5 | Gate-1 FAIL 3, 7, 11(H5) fixes | **PASS** | Join on `session`; geopolitical days kept in the H1/H2/H3 control (2,165 vs 1,918); H5 both arms not-affected (`analyze.py:648-649`), reproduced. |
| 6 | Lookahead: med20, ranged_a | **PASS** | med20 recomputed on all 2,655 QQQ and 1,681 SPY rows: 0 mismatches (1,534 mismatches if today is included). The lines use only pre-09:30 bars. |
| 7 | Event alignment | **PASS** | All 6 Good Friday rows land on the next session. No geopolitical date falls on a weekend. 9 of 9 late-news rows spot-checked are correct. No double shift. |
| 9a | A7 computed as pre-registered | **PASS** | Strict `>`, 0.6 × med20, max(hi, −lo) (`analyze.py:452-471`). Reproduced 331/953, 349/1297, 34/405. Zero ties. |

---

## FAILs

### Item 1: numbers that do not trace

I checked every number in §§1-7 against `analysis.json`. Almost all trace exactly:
2,655 / 1,681; H1 +0.27 [0.12, 0.35] 84 vs 2,165; the §2 table (diffs, shares, Holm
0.052); the SPY 47/45/44 vs 28; the §3 table (all 8 diffs, n and p95); the §4 shares, CIs
and by-year values; §5 147/73/230 vs 464/262/711, p 0.61, SPY p 0.025; the §6 table;
the §7 lab counts (also recomputed from `lab_days.csv`). These do not:

| where | says | correct |
|---|---|---|
| §3 l.81-82 | "Tariff and war days are [wider than placebo], on both symbols" | SPY trade_tariff: diff +0.210, placebo p95 +0.230, flag **selection**. Only **war** is "placebo not reached" on both symbols. |
| §5 l.99-100 | SPY: "the difference is a few more trend days" | Backwards. SPY not-affected event days vs not-affected non-event days: trend **31.5% vs 33.6% (fewer)**, ranged **15.6% vs 21.8% (fewer)**, chop **52.9% vs 44.6% (more)**. The difference is more chop days. |
| §4 l.91-92 and front matter l.9 | "17.5% in 2026 so far (183 sessions)"; QQQ "2016-01-04 → 2026-10-29" | The QQQ CSV ends **2026-09-24** (`analysis.json` `QQQ.date_range`). The 183 sessions are 01-02 → 09-24. See item 11. |
| §1 l.40 | "QQQ's range was **27% wider than a normal day**" | Normal day is defined (l.34) as the trailing 20-day median. The median Fed day is **1.23× normal** (23% wider). The +0.27 is the gap between Fed days (1.23×) and no-release days (0.96×). Wording, not a wrong number. |
| §2 l.60 | SPY: "none of the range medians survive correction" | SPY is in no family (A3/A4), so there is no correction to survive. And SPY's H1 range median *does* meet A4 (+0.376, CI [0.22, 0.68]). By A4's rule the CPI/NFP/PCE medians fail on the 0.2 floor, not on a correction. |
| §4 l.93 | SPY ranged "flat by year" | By year 7.2%–14.0% (2021 low, 2020 high), no monotone trend. "No trend" is fair; "flat" overstates it. |

**What would change the verdict:** correct the five lines above in place, with the
reason.

### Item 4b: §3 slips from description into claim

The A1 flag rule is implemented exactly (item 4a, PASS). The write-up then says more
than the flag allows:

- l.82: **"Tariff and war days are, on both symbols."** This is a claim, and it is false
  for SPY tariff (flag = selection).
- Table column "verdict" and "not explained by placebo". The code's label is "placebo not
  reached (still descriptive)" (`analyze.py:420`). "Not explained" implies the rest is an
  effect. The placebo matches only year and weekday. It cannot detect salience selection,
  which is the stated reason the list is untestable.
- l.81: "Five of eight categories are no wider than random days from the same years."
  This is a null read as an equivalence, on n = 10–61. Two of the five (cyber +0.29,
  us_fiscal +0.10) have positive observed differences that are simply under-powered.
- Title "mostly selection". A1's "selection" label lands on categories whose observed
  difference is about 0 (energy +0.011, financial_stress +0.008). There is nothing there
  for selection to explain. The flag is a label, not an attribution.

Checklist: E (claim vs measured quantity), D (underpowered null).
**What would change it:** §3 says only "war_military: placebo not reached on both
symbols; trade_tariff: placebo not reached on QQQ, selection on SPY; all others flagged
selection; descriptive, list selected by salience; may be circular". Drop
"not explained".

### Item 5 (gate-1 FAIL 11, data hygiene part): A6 table and funnel not in results.md

A6: "`day_features.py` reports thin sessions by year … **`results.md` prints that
table**." The code writes it (`day_features.py:285`, JSON `thin_by_year`: QQQ 1-3 per
year, 21 total; SPY 12). `results.md` does not print it. Gate-1 11.3 asked for a funnel:
days tried, no bars, thin. The data for it exist in `day_features_QQQ.json`: 2,824
weekdays tried, 127 `no_bars`, 21 thin, 2,697 rows. 127 is about 102 holidays plus 25
dead-feed days. The funnel was the alarm, and it was not printed. 11.5 (share of
"affected" days caused by the gap alone) was also not reported: **265 of 768 (34.5%)**
QQQ affected days.
**What would change it:** print the thin-by-year table and the funnel (tried / no bars /
thin / short history / usable) per symbol in `results.md`.

### Item 9b: "trend and chop days look alike at 10:30"

The QQQ figures are 34.7% [31.6, 37.9] vs 26.9% [24.6, 29.4]. The CIs do not overlap, and
trend days showed a big first hour about 1.3× as often. "Alike" holds only on SPY
(24.2% vs 23.3%). Two more problems in §6:
- "The first hour tells you a quiet day is quiet" is partly true by construction. 274 of
  the 405 `ranged` days are `ranged_b` (whole-day range < 0.6 normal). Those days *cannot*
  have a first-hour excursion > 0.6 normal: 0 of 274. All 34 "early" ranged days are among
  the 131 ranged_a-only days.
- "The red team predicted exactly this: … direction and persistence are not." A7 measures
  no direction. The statistic is also P(big first hour | day type), while the heading asks
  the reverse, P(day type | big first hour). From the CSV that reverse figure is 46.4%
  trend among big-first-hour days vs a 35.9% base rate. That differs from "no".
**What would change it:** "On QQQ, trend days had already shown a big first hour more
often than chop days (35% vs 27%); on SPY about equally (24% vs 23%). Descriptive, no
test." Delete "exactly this" and the direction clause.

### Item 10: the proposed CLAUDE.md line

> scheduled releases widen the day (Fed +27%, 4 in 10 release days are big vs 1 in 4),
> direction untouched, trend not knowable by 10:30.

| clause | status |
|---|---|
| "scheduled releases widen the day" | **Not PASS.** Range passed for FOMC (H1) and GDP_ADV (H2, QQQ only; SPY untested at 27). CPI +0.009, PCE +0.072, NFP fails Holm (0.144). |
| "Fed +27%" | PASS (H1), with the item-1 wording fix. |
| "4 in 10 release days are big vs 1 in 4" | **Not PASS.** The pooled "release day" share was never tested. FOMC H3 failed (Holm 0.052), PCE is null (30%), GDP_ADV not significant. Only CPI and NFP passed H3. For those two, "big" is driven mostly by the gap band, not by range: share with gap > 2.5× is 22.8% / 23.0% vs 14.2% control, share with range > 1.5× is 21.3% / 25.4% vs 17.5%. |
| "direction untouched" | **Not measured.** It reads as a finding. The study never looked at direction (§1 says so). |
| "trend not knowable by 10:30" | **Descriptive (A7, no test), and contradicted on QQQ** (item 9b). |

Same problem in §1 l.47: "magnitude, not sign, four times over; **now five**". A study
that "did not touch direction" cannot be a fifth arm of a sign result.

**Minimal rewording (PASS-level only):**
> Fed decision days are wider on QQQ: median range +0.27 of a normal day above
> no-release days (H1 pass; SPY 2020-26 agrees). On CPI and jobs days about 4 sessions
> in 10 are "big" (range > 1.5× or gap > 2.5× normal) vs about 1 in 4 otherwise.
> Direction was not measured.

### Item 11: the one bug class. Found one.

**QQQ was silently truncated by the dead feed.** The QQQ feature run started
2026-10-29 19:10 local (`day_features_QQQ.json` `run_at`), before the terminal restart.
The SPY run is 23:26. `day_features.py:277-279` turns a day with no bars into "no row",
counted as `no_bars`. QQQ: 2,824 weekdays tried, 127 no_bars. SPY over the same calendar
has **25 sessions after 2026-09-24** that QQQ lacks. So 127 = about 102 holidays + 25
dead-feed sessions. No error, no crash. `results.md` front matter says → 2026-10-29,
§4 says "2026 so far", and §9 "Defects recorded" names the dead feed but does not say it
cost QQQ its last 25 sessions. Worse, `levels_test.py:62-69` pickles an empty response
as `([], [])` (only `FeedOutage` skips the cache). A re-run from the laptop cache would
therefore reproduce the same hole, unless those pickles were deleted. The same applies to
any SPY 2016-2019 dates requested during the dead period. I cannot see the cache
(`live_lab_data/ext_cache` is not in git).

Magnitude on the headline: small. One FOMC session (2026-10-28) and about 1% of rows are
lost. H1 does not move materially. Direction of bias: none expected. It is the bug class
nonetheless: a coverage statement that is confidently wrong, found only by comparing two
files.

Two more, of the "true number, false conclusion" kind: (a) "4 in 10 release days are big"
read as "releases widen the day", when for CPI and NFP the excess is mostly the opening
gap (item 10). (b) "now five" (item 10).

**What would change it:** state QQQ coverage as 2016-01-04 → 2026-09-24. Either delete
the empty QQQ pickles for 2026-09-25 → 10-29 and re-run, or say they are missing. Note
in §9 that the dead feed truncated QQQ.

## QUESTIONABLE

### Item 3: families, Holm, "4 in 10 vs 1 in 4"

- QQQ H2 family = {CPI, NFP, GDP_ADV, PCE}. FOMC is outside H2 and is in H3 = {FOMC, CPI,
  NFP, GDP_ADV, PCE}. Correct per A3.
- I reproduced Holm by hand. H3: 0.000667×5 = 0.0033, 0.00333×4 = 0.0133, 0.01733×3 =
  0.052, 0.0633×2 = 0.127, 0.398. H2: 0.00267×4 = 0.0107, 0.048×3 = 0.144, 0.138×2 =
  0.276, 0.861. All match.
- The floor comes after Holm (`analyze.py:571-573`). GDP_ADV passes (Holm 0.011, diff
  0.255 ≥ 0.2).
- My own-seed bootstrap gives FOMC H3 raw p 0.021 (Holm about 0.062). It fails either
  way, so 0.052 is not a knife-edge artefact.
- **SPY in no family: not honoured in code.** `analyze_symbol` runs Holm for every symbol.
  `analysis.json` holds `SPY.H2.family = [CPI, NFP, PCE]` and `SPY.H3.family = [FOMC, CPI,
  NFP, PCE]` with `p_holm` and `pass`. That contradicts its own `families.SPY = "none"`.
  The A4 replication flag uses the raw CI, so replication is unaffected. Only the wording
  is ("survive correction", "pass the test").
- **"4 in 10 vs 1 in 4."** The shares are FOMC 39.3%, CPI 39.4%, NFP 43.7% vs 26.7%
  (QQQ), and 47/45/44% vs 27.6% (SPY). As arithmetic, "about 4 in 10 vs about 1 in 4" is
  fair. But it names Fed days, which **failed** H3 on QQQ. SPY cannot rescue a failed
  primary under A4.
Caveat wording: "on CPI and jobs days (QQQ, Holm-corrected) about 4 in 10 are big vs
about 1 in 4; Fed days 39%, not significant after correction."
**What would change it:** drop Holm for SPY, or label it "not pre-registered", and
remove "Fed" from the 4-in-10 sentence.

### Item 5 (gate-1 FAIL 4 and FAIL 9)

Implemented as written for QQQ (`analyze.py:78-79`, `replication_flag`), with the label
"not an independent sample" in the JSON. Not fully honoured: SPY Holm (item 3), and §2
"Read together" uses SPY's FOMC share to support a QQQ result that failed. The
`replicated_on_SPY = True` written on QQQ FOMC H3 is meaningless for a failed primary.
**What would change it:** the item-3 fixes.

### Item 8: ranged_a by year (3.5% → 17.5%)

What the CSV establishes: I rebuilt the six lines for every row from `pre_high`,
`pre_low`, `high`, `low` of the `prev_session_date` row plus today's premarket. Every day
whose high and low stayed inside the nearest not-yet-broken lines is `ranged_a` (0
contradictions), so the reconstruction is consistent.
- **ranged_a is mostly a function of how wide the premarket was relative to a normal
  day.** By pooled quintile of (premarket range / med20 range): 1.5%, 2.4%, 4.9%, 11.5%,
  20.7%.
- The median of that ratio rose from **0.55 (2016) to 1.13 (2026)**. Top-quintile days:
  64 of 727 in 2016-18, 249 of 679 in 2024-26.
- Composition is not the whole story. Within the top quintile ranged_a still rose (9.4% →
  24.5%).
- Thin handling is not the driver (1-3 thin days a year). The cause is not "gap-and-go
  pre-breaks" either: all lines on one side pre-broken occurs on ≤ 0.5% of days.
- SPY over the same days, same code, same feed: 14.0% (2020) → 13.0% (2026), no rise.
  QQQ: 8.4% → 17.5%. This argues against a defect in shared code, but not against
  symbol-specific premarket coverage.

What it cannot establish: whether QQQ's wider premarket is real (more premarket trading
and news) or feed coverage. `get_ext` drops zero-filled minutes (`levels_test.py:58-59`),
so a sparse 2016 premarket yields a narrower observed range and tighter lines, which
means more breaks. Telling these apart needs premarket bar counts per day from the laptop
cache, which the CSV does not carry. 2026 covers only up to 09-24 (item 11).
Caveat wording: "the rise in no-line-break days tracks a widening premarket range;
whether that is market structure or premarket data coverage is not established."
**What would change it:** premarket bar counts per session, by year, from `ext_cache`.

## PASS

### Item 2: H1

Run as amended. Control = usable sessions with no *scheduled* event; geopolitical days
are kept (`analyze.py:525`; 2,165 control vs 1,918 no-event-at-all, both reproduced).
Statistic = difference of medians. Bootstrap 3,000 reps, seed 20260828 (`analyze.py:69-70`),
95% percentile CI. Pass = CI excludes 0 and diff ≥ 0.2 (`analyze.py:353-356`), no family.
**My independent point estimate: median rel_range on 84 FOMC sessions 1.2330 − median on
2,165 control sessions 0.9644 = +0.2687.** With my own resampler, seeds 1 and 2: CI
[0.123, 0.356] and [0.121, 0.352], p 0.0003. That matches the JSON [0.124, 0.352].
SPY: +0.3765 (53 vs 1,371), reproduced.
Standing caveat from gate 1, unaddressed but not binding: the CI assumes independent
sessions. FOMC days are about 6 weeks apart, so this matters little here.

### Item 4a: the placebo flag rule

`analyze.py:413-420` flags "selection" when p95 ≥ 0.5 × observed (observed > 0), never
"effect". The pool is all usable sessions of the same year and weekday, minus the
category's own days. The control is no-event days. The statistic is the same as the
observed one. I checked all 8 categories × 2 symbols for rel_range against the rule: all
correct. `financial_stress` is never tested (`analyze.py:82, 620`).

### Item 5 (gate-1 FAIL 3, 7, and 11-H5)

- FAIL 3: scheduled rows join on `session` (`analyze.py:163-168`), and the code refuses a
  file without it. The geopolitical `date` is already the assigned session and is not
  re-shifted.
- FAIL 7: the control fix is in code (above).
- FAIL 11-H5: A5 is implemented (`analyze.py:648-649`). My counts:
  147/73/230 (sum 450) vs 464/262/711 (sum 1,437 = 1,918 − 481 affected).
- Not verifiable: the tightened two-front-page rule.
  `events_geopolitical.README.md` says the two-outlet test is "asserted from the nature of
  the event for every row, not verified". This is contained, because no geopolitical row
  feeds any test.

### Item 6: lookahead

- `med20` uses the previous 20 non-thin sessions, today excluded (`day_features.py:163-171`;
  history is appended only after the row, `:293`). I recomputed it on every row: 0
  mismatches on QQQ (2,655) and SPY (1,681). If today were included, 1,534 QQQ rows would
  mismatch, so the check can tell the two apart. Three rows by hand:

| date | previous 20 | med20_range recomputed / file | incl. today would be |
|---|---|---|---|
| 2016-02-03 | 01-05 … 02-02 | 222.7821 / 222.7821 | 228.4185 |
| 2020-03-17 | 02-18 … 03-16 | 423.391 / 423.3909 | 442.4056 |
| 2026-09-24 | 08-26 … 09-23 | 90.631 / 90.6309 | 94.2573 |

- `prev_close` equals the `prev_session_date` row's close on 100% of rows. No
  `prev_session_date` points at a thin day (A6).
- `ranged_a`: `build_six(prev_ext_bars, pre_bars)` (`day_features.py:215`) uses only the
  last non-thin session's 04:00-15:59 bars and today's 04:00-09:29 bars
  (`levels_test.py:64-67`; `six_lines.py:89-91`, `_hi_lo` is end-exclusive). Nothing after
  09:30 enters the lines. `n_line_breaks` is a same-day label, used only descriptively.

### Item 7: event alignment

- 516 scheduled rows. The 6 rows where `date ≠ session` are exactly the Good Fridays
  2017-04-14, 2020-04-10, 2021-04-02, 2023-04-07, 2024-03-29 and 2026-04-03. Each maps
  to the next session present in the data, and none of the release dates is itself a
  session. `scheduled_rows_shifted_to_next_session = 6`.
- Every scheduled session up to 2026-09-24 is a QQQ row.
- Geopolitical: no weekend dates, and every date up to 09-24 is a QQQ session. Four rows
  land on thin days and are excluded and counted (2018-12-24 ×2, 2020-12-24, 2021-11-26).
- 100 rows have `break_time_et ≥ 16:00`. I spot-checked 9 against the actual event
  timing, and all 9 sit on the next session: 2016-01-06 NK test, 2016-01-29 BoJ,
  2016-06-24 Brexit, 2016-07-15 Nice, 2016-07-18 Turkey, 2016-12-05 Italy, 2017-01-30
  travel ban, 2017-02-14 Flynn, 2017-05-10 Comey.
- Caveat: break times are the list-builder's recollection, and the 18 rows from 2026 are
  dated by signing date with unknown hour (README).

### Item 9a: A7 computation

`analyze.py:452-471`: max(first60_hi_bp, −first60_lo_bp) > 0.6 × med20_range_bp, strict.
The first-hour window is bars 09:30-10:29 (`day_features.py:85, 206`), so the 10:29
1-minute bar closes at 10:30 and is knowable at 10:30. Reproduced QQQ 331/953, 349/1297,
34/405 and SPY 146/603, 183/787, 27/291. Using ≥ instead of > changes 0 rows.

## Checks not run, and why

- **Premarket bar counts by year** (item 8) and **the state of `ext_cache`** (item 11:
  whether empty pickles exist for QQQ 2026-09-25 → 10-29 and for SPY 2016-2019). Both
  need the laptop; `live_lab_data/ext_cache` is not in git.
- **Missingness vs outcome**: only the tail hole was found. Within 2016-01-04 → 2026-09-24
  QQQ has every session SPY has (2020+), and 252 rows in 2016. Holes before 2020 can only
  be checked against a holiday calendar, which I did by count (2,824 − 127 = 2,697), not
  date by date.
- **Geopolitical break times for the other 91 late rows and the 63 `unknown` rows**: not
  verifiable from the repo.
- **Block bootstrap / day clustering** (gate-1 Q6): not run by the study and not
  required by the pre-registration. Unknown for clustered geopolitical categories, which
  are descriptive anyway.
- **Lab H6 Wilson CIs and the `reconstructed` P&L flag** (gate-1 Q10, QUESTIONABLE then):
  not in results.md. The counts reproduce; no claim rests on them.
- **Option pricing, the $0.65 fee, the 75.3% breakeven, 0DTE survivorship**: not
  applicable. The study prices no options and tests no trading rule.
