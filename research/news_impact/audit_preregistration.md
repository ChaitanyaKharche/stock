---
task_id: R-2026-10-30-06
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
parent: R-2026-10-30-01
role: statistical-auditor
repo_sha: a367245
inputs: [research/news_impact/preregistration.md@a367245]
decision_rule: "PASS/FAIL/QUESTIONABLE per item"
outputs: [research/news_impact/audit_preregistration.md]
evidence: []
assumptions: []
open_questions: []
errors: []
status: done
verified_by: null
budget: {cap: "one pass"}
---

# Gate 1 audit: news-impact pre-registration

**Short version: the scheduled-event part is sound. The geopolitical part is circular.**
The agent-built list is picked by how big the news looked, and big-looking news is
mostly news that moved the market. Those days are also removed from the "normal day"
comparison group, so the bias reaches the primary FOMC test as well. Four items need a
one-line wording fix before the laptop run (items 3, 4, 7, 9 and 11).

This is a design audit: no data exist yet, so nothing was reproduced. These verdicts
cover the *rules*, not results. I read the code the rules depend on
(`six_lines.build_six`, `six_lines.breaks_today`, `six_lines.run`, `levels_test.get_ext`,
`walkforward.sessions_between`) and the lab ledger at a367245.

## Verdict table (FAILs first)

| # | Item | Verdict | Reason |
|---|---|---|---|
| 7 | Geopolitical inclusion rule / H3 circularity | **FAIL** | §3.2 l.82: "led the front page of two of Reuters, AP, WSJ, FT, Bloomberg" partly selects on market reaction. Those days are also removed from the §6 l.151 control group, so the bias reaches H1. |
| 3 | Event-date alignment | **FAIL** | §3.1 has no rule for a release on a closed day (NFP 2021-04-02 and 2023-04-07, PCE 2024-03-29, all Good Fridays). §3.2 l.73 lacks a break-time column, the 16:00 edge and holidays. |
| 4 | Holm family | **FAIL** | §6 l.156-157: which tests are in each family, whether H2 and H3 are one family or two, and how Holm gets a p-value from a bootstrap CI rule are all unstated. |
| 9 | SPY replication rule | **FAIL** | §6 l.163-164: "does not hold on SPY" is undefined. The criterion in the brief is not in the file. |
| 11 | "Confident number that measured nothing" | **FAIL** | §6 l.159 H5 restricts event days to "not affected" but leaves non-event days unrestricted. That raises the event-day ranged share by construction. |
| 1 | Trailing-20 medians | QUESTIONABLE | No lookahead. But §4 l.105-106/114 do not say whether thin days are in the 20-day window, what happens in the first 20 sessions, or what "previous session" means after a half day. |
| 5 | ≥ 0.2 effect floor | QUESTIONABLE | Arbitrary but chosen in advance and conservative. §6 l.155 does not say whether 0.2 applies to the point estimate or the CI bound. |
| 6 | Session bootstrap | QUESTIONABLE | Independent session resampling ignores volatility clustering and the 2020 and 2022 event clusters. CIs too narrow for geopolitical categories most of all. |
| 8 | Minimum 30 per type | QUESTIONABLE | At n=30 the smallest detectable difference is about the size of the 0.2 floor. A null there is underpowered, not absent. "Rows" ≠ distinct sessions. |
| 10 | H6 lab, descriptive | QUESTIONABLE | At a367245 the ledger has 44 `is_session` dates, not 42, and 22 are `PARTIAL`. No rule forces k/n and Wilson CIs. |
| 2 | `ranged_a` information timing | PASS (design) | `build_six` uses only data before 09:30 (`six_lines.py:83-106`). `n_line_breaks` is a same-day description, and §6 never uses it to predict. |

---

## FAIL 7: the geopolitical list is chosen partly by market impact

**Is H3 trivially true by construction? For scheduled events, no.** FOMC, CPI, NFP,
GDP_ADV and PCE dates are fixed by the calendar, months ahead and blind to the outcome.
"Affected" (`impact == high`) is defined from the day's own trailing median and never
from event status. So H3 on scheduled types is a real test. It may be unsurprising,
like H1, but it is not circular.

**For geopolitical categories, yes, partly.**
- WSJ, FT and Bloomberg are financial front pages. A geopolitical story leads them far
  more often on days when markets lurched ("Stocks tumble as ..."). Two of the five
  outlets in §3.2 l.82 therefore vote with the market's reaction.
- "Plausible channel to US equities, rates or the dollar" is a hindsight judgement.
- The list is built by an agent whose background knowledge already contains the
  market's reaction to every event in 2016-2026. "Built before any feature file exists"
  (l.84) hides the feature file, not the outcomes; those are public knowledge. Events
  remembered as salient are mostly events that moved prices.
- `financial_stress` is circular by definition. Its examples (2020-03-03, 2020-03-15)
  are reactions *to* market moves; the event is the market move.

**The bias also reaches H1.** §6 l.150-151 compares each type with "non-event days (no
event of any type that day)". If 150-300 salience-selected days (6-11% of ~2,650
sessions) are mostly high-impact days, removing them makes the control group calmer
than an ordinary day. Every event-minus-control difference grows, FOMC (H1) included.
Direction: always toward PASS. Rough size, not reproducible until the data exist: if
control days are about 12% "affected" and geopolitical days about 40%, removing 8% of
days lowers the control share to about 9.6%. That inflates every H3 difference by about
2.4 percentage points. The effect on the median `rel_range` is smaller (a few
hundredths), but it still moves the "CI excludes 0" test in the favourable direction.

Checklist: E (target vs objective, selection), C (selection correlated with outcome).

**Minimal fix (wording, §3.2 and §6):**
> §3.2: "A front page counts toward the two-outlet test only if its lead headline is
> about the event itself; a headline about the market's reaction (stocks, yields, oil,
> the dollar) does not count. `financial_stress` rows are reported, never tested."
> §6: "For H1 and for scheduled types in H2/H3, the control group is days with no
> *scheduled* event; geopolitical days are not removed from it. H2/H3 results for
> geopolitical categories are descriptive and are labelled 'list selected by salience;
> may be circular with market impact.'"

(Leaving geopolitical days in the control group biases toward the null, which is the
safe direction for a test.)

**What would change the verdict:** the wording above, or proof that the list was built
by a procedure blind to market coverage.

## FAIL 3: event dates on closed days and the geopolitical timing rule

- **Scheduled (§3.1 l.57):** `date` is "ET release date" and there is no rule for a
  release that falls on a non-session. NFP was released on Good Friday 2021-04-02 and
  2023-04-07, and PCE on 2024-03-29 (also Good Friday). These dates are from my own
  knowledge; check them against the calendar. With no rule, the event joins to no
  session and silently disappears. The next session, which carries the reaction in its
  gap, gets counted as a non-event day. 08:30 releases on a normal session are correctly
  same-session, since the gap captures them. FOMC at 14:00 is same-session for range,
  but that day's `rel_gap` is measured *before* the statement. So FOMC's `impact`
  (the higher of two bands, l.123) carries a gap that cannot reflect the event. That
  dilutes FOMC's H3 rather than biasing it, so it is not a FAIL on its own; it needs a
  sentence in results.
- **Geopolitical (§3.2 l.73):** "after 16:00 ET or on a weekend, the next session".
  Unstated: (a) whether exactly 16:00 counts; (b) market holidays, which "weekend" does
  not cover; (c) the CSV has no break-time column, so the rule cannot be checked;
  (d) "led the front page ... on that day" (l.82). Evening news hits print front pages
  the *next* morning, so the inclusion day and the assignment day can differ.

**Minimal fix:**
> §3.1: "If the release day is not a trading session, `date` is the next session and
> `note` says 'moved from YYYY-MM-DD (closed)'. `date` is the actual release date, not
> the originally scheduled one."
> §3.2: "Add column `break_time_et`. News at or after 16:00 ET, or on a weekend or market
> holiday, is assigned to the next session. The front-page test may use the edition
> dated the assigned session."

## FAIL 4: the Holm family is not defined

§6 l.155-157: H1 has "none; one primary". H2 has "Holm across all types tested". H3
has "Holm as H2". Three things are unstated: (a) whether FOMC is in the H2 family (the
wording "same as H1 for CPI, ..." suggests not); (b) whether H3, which runs "per type"
and so includes FOMC, is its own family or shares one with H2; (c) the p-value. The pass
rule is "the CI excludes 0", and Holm needs ordered p-values, so two agents could
implement it two ways. Is one primary hypothesis enough protection? Yes for H1 itself.
Across H1+H2+H3 the study runs about 2×(5 + k) tests (k = geopolitical categories with
≥ 30 sessions). With two separate Holm families, the study-wide error rate is
controlled only within each family. That is acceptable if results.md calls only H1
confirmatory.

**Minimal fix:**
> "H2 family = {CPI, NFP, GDP_ADV, PCE} plus geopolitical categories with ≥ 30 distinct
> non-thin QQQ sessions; FOMC excluded (it is H1). H3 is a separate family over the same
> types plus FOMC. Each test's p-value is the two-sided bootstrap p =
> 2·min(share of reps ≤ 0, share of reps ≥ 0). Holm is applied at α = 0.05 within each
> family; the 0.2 floor (H2) is applied after Holm. SPY is not in any family. Only H1 is
> confirmatory."

## FAIL 9: the SPY replication criterion is not written down

§6 l.163-164 says only that a QQQ result which "does not hold on SPY" is reported as not
replicated. The criterion in the brief ("same sign and SPY CI excludes 0") is not in the
file. As a criterion it is reasonable, but it is **lenient as evidence**, for two
reasons. SPY and QQQ share every event date and move in the same direction on ~99% of
days, so SPY is close to the same sample counted again. It checks symbol- and
data-specific defects (SPY has about 3× QQQ's zero-filled minutes), not chance. It is
not strict either, because `rel_*` is normalised per symbol.

**Minimal fix:**
> "'Holds on SPY' = the SPY point estimate has the same sign, its 95% bootstrap CI
> excludes 0, and (H1/H2) it meets the 0.2 floor; no Holm on SPY. SPY replication is a
> data and symbol robustness check, not an independent sample, and is never described
> as one."

## FAIL 11: scenarios where a confident number measures nothing

1. **H5 conditioning asymmetry (the FAIL).** §6 l.159 compares event days that were
   *not affected* (`rel_range` ≤ 1.5 and `rel_gap` ≤ 2.5) with *all* non-event days,
   which include the high-band days. High-band days can almost never be `ranged_b`
   (`rel_range` < 0.6) and rarely `ranged_a` (wide days break lines). Cutting them from
   one group only raises that group's ranged share by about a factor 1/(1 − share_high).
   Result: "news days that did not move were more ranged than ordinary days" appears
   with no news effect at all. **Fix:** "H5 compares not-affected event days with
   not-affected non-event days."
2. **The circular geopolitical list feeds the control group** (FAIL 7): FOMC "wider
   than normal" partly because unusually wild normal days were filtered out.
3. **Silent drops.** `sessions_between` returns every weekday (`walkforward.py:72-78`),
   holidays included. `get_ext` drops non-positive bars (`levels_test.py:60`), so a
   holiday comes back empty and falls under `thin`. That is correct, and it is a real
   trading-day check, not an emptiness check. But `get_ext` **caches whatever comes
   back** (`levels_test.py:69`). Empty or short responses from a lapsed or limited
   ThetaData tier (status UNKNOWN per CLAUDE.md) become permanent `thin` days, and
   whole years could vanish without an error. The pre-registration asks for no funnel.
   Required in results: days tried, thin (holiday / half day / data hole), by year and
   by event status. Reference: the September run had 2,795 tried and 138 thin
   (`six_lines_results.md:4`). A thin rate much higher than that is the alarm.
4. **"Previous session" after a half day.** `six_lines.run` does not update `prev` on
   thin days (`six_lines.py:238-259`). So after a half day (e.g. the Friday after
   Thanksgiving), the next session's "yesterday" lines come from the session *before*
   the half day. If `day_features.py` does the same for `prev_close`, the gap is
   measured across two sessions. Affects about 3 sessions a year. See item 1.
5. **`rel_gap` blow-ups.** The median |gap| over 20 quiet days can be a few bp, so a
   routine gap reads as `rel_gap` > 2.5. These are "affected" days that reflect a small
   denominator, not news. It should be reported as the share of high-band days caused
   by `rel_gap` alone.

## QUESTIONABLE 1: trailing-20 medians

There is no lookahead: earlier sessions' range and gap are known before today's open,
and "today excluded" is enough for that. Leaving thin days out of the history is
defensible: half days have short ranges and would pull the baseline down. But it is
**not stated**. §4 l.114 says thin days are "excluded from every statistic", and it is
unclear whether the window is the previous 20 *non-thin* sessions or 20 rows. Also
unstated: the first 20 sessions (2016-01-04 onward) have no baseline. And `prev_close`
after a half day or holiday (scenario 11.4). One more effect: after an event cluster
(March 2020) the 20-day baseline is inflated, so the next events look *less* affected.
This biases toward the null and is not lookahead; it belongs as a caveat in results.
**Caveat wording:** "the 20-day baseline excludes thin days; events inside a volatile
cluster are measured against an inflated normal." **Changes to PASS with:** "Window =
the previous 20 non-thin sessions; sessions with fewer than 20 are excluded. `prev_close`
and 'yesterday' = the previous session with any RTH bars, half days included."

## PASS 2: `ranged_a` information timing (design)

`build_six` takes yesterday's premarket (04:00-09:29), yesterday's RTH (09:30-15:59),
and today's premarket (`six_lines.py:83-106`, bars split at `levels_test.py:64-67`).
All of it is known before 09:30. After-hours bars are not used at all. Timestamps are
naive ET compared with ET literals and are never converted, so the Arizona-machine
problem does not apply. `n_line_breaks` counts breaks over the whole RTH day, so it is a
**same-day descriptive label**. §6 uses it only in H4 (descriptive) and H5 (same-day
association), never to predict. No leakage into any decision. Caveats: the
half-day "yesterday" issue (11.4), and the H5 asymmetry (11.1), which is a comparison
defect, not a lookahead defect. Note: the "8.2%" in §4 l.110 is 218 no-trade sessions
out of 2,656 (`six_lines_results.md:26`) from `six_lines.run`. If `day_features.py`
does not reproduce it on the same window, the two implementations differ. That is a
free bug detector.

## QUESTIONABLE 5: the ≥ 0.2 floor

`rel_range` is in units of the session's own normal range, so 0.2 = "a fifth of a
normal day". On QQQ at about 130 bp normal range that is about 26 bp, roughly $1.60 on a
$600 share. It is arbitrary but fixed in advance, and it can only turn a PASS into a
FAIL, never the reverse. So it is harmless to validity. It does matter for the
conclusion: it sits about at the n=30 detection limit (item 8), so for small types it
binds. **Ambiguity:** point estimate ≥ 0.2, or CI lower bound ≥ 0.2? **Changes to PASS
with:** "the point estimate of the median difference is ≥ 0.2."

## QUESTIONABLE 6: session-level bootstrap

Normalising by the trailing median removes the slow volatility level but not short runs
(rel_range stays above 1 for days after a shock). Geopolitical events cluster: 2020
(pandemic) and 2022 (war, inflation). A category like `pandemic_health` may be one or
two episodes, not 30 independent sessions. Resampling sessions independently treats
them as independent, so CIs are too narrow, worst for geopolitical categories. FOMC is
spread about 6 weeks apart and is close to independent. **Caveat wording:** "CIs assume
independent sessions; for clustered categories they are too narrow." **Would change
with (robustness lines only, no new hypothesis):** a moving-block bootstrap over the
session sequence (block ≈ 20 sessions, matching the baseline window) reported next to
the primary CI, plus a by-year table of the difference with the sign per year. If the
primary CI excludes 0 and the block CI does not, results must say so.

## QUESTIONABLE 8: minimum 30 per type

Expected scheduled counts over 2016-2026 are about 86 FOMC, about 43 GDP_ADV and about
125-130 each for CPI, NFP and PCE. All clear 30. At n=30 the standard error of a median
`rel_range` is roughly 1/(2·f·√30) ≈ 0.09 (assuming density ≈ 1 at the median, not
measured). That means about 80% power only for differences around 0.25, the same size as
the floor. For H3 shares at n=30 the CI is about ±0.16, so share differences under
about 0.2 are invisible. "Rows" (§6 l.156) can include several rows on one date.
**Caveat wording:** "a type with fewer than ~60 sessions that fails is underpowered, not
evidence of no effect." **Changes with:** counting distinct non-thin sessions, not rows,
and stating the detectable difference beside every null.

## QUESTIONABLE 10: H6, the lab

§6 l.160 says "descriptive only". Nothing makes reports honour it. Over about 2 months,
roughly 10-12 event sessions against about 30 non-event sessions means a split like
7/10 vs 15/32 green *looks* like a finding. The Wilson 95% CIs are about ±25-30 points
and overlap heavily. "Green" also ignores size, and this project's P&L is
tail-dominated. **Count does not reproduce:** at a367245, `session_ledger.jsonl` has 63
dates. 44 are `is_session` (21 `COLLECTED`, 22 `PARTIAL`, 1 `INTERRUPTED`) and 19 are
`NOT_A_SESSION`. 42 may have been right at ea65bb1, but which outcomes count is
undefined. The 09-08 and 09-10 shares P&L includes `reconstructed` stranded positions
(CLAUDE.md). **Caveat wording:** "42-ish sessions; green-day shares are counts with
wide intervals and say nothing about whether news helps the strategy." **Changes to
PASS with:** "H6 reports k/n and Wilson 95% CIs only, never a difference or a rate
without n; states which ledger outcomes count as a session and how `PARTIAL` days are
scored; flags reconstructed P&L."

## Checks not run, and why

- **Event CSVs and both scripts' tests:** §7 l.173-176 asks gate 1 to read them. None
  exist at a367245 (`research/news_impact/` holds only the pre-registration;
  `trade_analysis/news_impact/` only `__init__.py`). Date alignment, thin handling and
  the window definition **as implemented** are unaudited. Gate 1 is incomplete until
  they are read.
- **Any number**: no feature file exists, so nothing was reproduced. Every magnitude
  above is an estimate, marked as such.
- **Missingness vs outcome** (are data holes more common on volatile days): needs the
  laptop cache.
- **ThetaData stock-history depth**: tier status is UNKNOWN per CLAUDE.md. It cannot be
  confirmed from the cloud that 2016 is still reachable rather than coming from cache.
- **Option-pricing selection, 0DTE survivorship, the $0.65 fee, the 75.3% breakeven:**
  not applicable. The study prices no options and tests no trading rule (§8).
- **Lookahead check for the 10-minute bar labels in `breaks_today`**: not applicable to
  a whole-day count. It would apply if anyone used a break time as an entry.
