---
task_id: R-2026-10-30-07
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
parent: R-2026-10-30-01
role: strategy-contrarian
repo_sha: a367245
inputs: [research/news_impact/preregistration.md@a367245, CLAUDE.md@a367245]
decision_rule: "n/a"
outputs: [research/news_impact/redteam.md]
evidence: []
assumptions: []
open_questions: []
errors: []
status: done
verified_by: null
budget: {cap: "one pass"}
---

# Red team: the news-impact study, and the reason for running it

## 0. The strongest attack, in one paragraph

Nothing needs to be "missing" for the trading to be flat. Three settled findings already
explain it. (1) The levels predict how far price travels, not which way it ends up
(CLAUDE.md, "magnitude, not sign", four separate measurements). (2) Option sellers charge
for travel: the unconditional 0DTE variance premium is real (gross t=+7.52,
`vrp_cost_model_results.md` §1), so whoever buys travel overpays a little on average.
(3) The buyer also crosses the spread. Put a coin-flip direction, a slightly overpriced
option and a spread together and you get flat-to-negative results. That is what the lab
shows: options −$4,387 (14 green days out of 38), shares −$19,100 (18 out of 40),
recounted from `live_lab_data/` at a367245. A day-type filter can only fix this if the
day type is **(a)** known before entry, **(b)** not already priced into the option, and
**(c)** tells you the direction. As written, the study tests none of (a), (b) or (c). It
describes days. It cannot find "which days work."

That does not make the study worthless. It is a clean descriptive study. It is not the
missing piece, and the owner should be told so **before** results arrive, not after.

Specifications I examined while writing this: one null simulation (§2, two arms, ~125
cells each). No market data was fetched and no test was run on prices.

---

## 1. Even if H1 passes, the owner does not gain an edge

**Mechanism.** FOMC, CPI and NFP dates are published months ahead. Market makers raise
the implied volatility of 0DTE options on those days, so the bigger range is priced in
before 09:30. Someone buying a call or put on a Fed day pays for the wider day up front.
"Fed days are wider" (H1) and "Fed-day options are cheap" are different claims. Only the
second could make money, and the pre-registration says so itself (§6, last paragraph).

**Prediction.** On scheduled-event days, the ATM 0DTE premium will be higher than on
other days, by roughly the same proportion as the extra range. The payoff-to-premium
ratio will be about the same on event days and non-event days.

**The test (no new data needed).** Use the SPY 0DTE archive, 2020-01-03 → 2023-12-29,
the same 618 sessions `cost_model.py` used. **The 2024 block stays sealed.**
- At the close of the 09:30–09:34 five-minute bar (decision time 09:35, not 09:30), record
  the ATM straddle mid `P` and ask `A`. A straddle is a call plus a put at the same
  strike. Here it is only a **ruler**: its price is the market's forecast of travel in
  either direction. It is not a proposed trade.
- At 16:00 record the payoff `X = |S_close − K|`.
- Per session, `r = X / P` (realised over implied, gross) and `r_net = X / A`.
- Compare the median `r` on event days (FOMC, CPI, NFP, PCE, GDP_ADV, each type alone and
  pooled) with non-event days. Bootstrap over sessions, 3,000 reps, seed 20260828, as in
  §6 of the pre-registration.
- Check against the 30-minute straddle's unconditional gross edge (+$0.0205 to the seller,
  spread $0.0376): the same run should show `median r < 1` on non-event days. If it does
  not, the harness is wrong, not the market.

**Falsifier (fixed now).** The owner's idea survives only if the pooled event-day median
`r_net` exceeds 1.0 **and** is above the non-event median, with a 95% CI that excludes
zero. Anything else means the extra range is already paid for, and H1 tells a buyer
nothing.

**Sample warning.** Before Nov 2022, SPY 0DTE existed only on Mon/Wed/Fri (no Tue/Thu
expiries), and 2021 has only 39 archive sessions. FOMC is on a Wednesday, so it is
covered, but count first: the archive likely holds **fewer than 30 FOMC days**. By the
pre-registration's own rule (§6) that means the FOMC arm is reported, not tested. CPI
dates fall on any weekday, so CPI coverage before late 2022 will be patchy. A design
effect of 1.3–1.7 (CLAUDE.md) gives an effective n of about 18–23 per 30 days. **Cost to
settle:** one script reading the existing archive, about one laptop hour. **This is the
cheapest decisive test in this file and should run before anyone reads H1.**

**My prediction:** event-day `r` will not differ from non-event `r` (CI includes zero).
Both medians will sit below 1.0.

---

## 2. The "which days work" trap

**The count.** The pre-registration creates these day labels:

| axis | levels |
|---|---|
| event type | 13 (5 scheduled + 8 geopolitical), plus "any event" and "none" |
| impact band | 3 (low / medium / high), from 2 measures (`rel_range`, `rel_gap`) |
| day class | 3 (trend / ranged / chop) |
| ranged definition | 2 (`ranged_a`, `ranged_b`) |

**13 × 3 × 3 × 2 = 234 day-type cells per symbol, 468 across QQQ and SPY.** That is before
`gap_reversal` (×2), year (×11 for H4), lab arm (options ATM / ATM±1 / shares) or
setup (15). The confirmatory tests (H1–H3, roughly 26 Holm-corrected comparisons per
symbol) are fine. The danger is what happens after `results.md`: the owner, or an agent,
sets the 42 lab sessions against these labels and reads off "the days that work."

**What that scan finds when there is nothing to find.** I ran a simulation
(`scan_sim.py`, scratch, 300 draws, seed 20261030). It used 42 sessions, about 9
scheduled-event days and 4 geopolitical days in the window, plausible label rates, and
green days drawn **independently of every label** at the lab's own rates (options 14/38,
shares 18/40). Every one-, two- and three-way combination of labels holding at least 3
sessions was scanned:

| arm | cells scanned (n ≥ 3) | cells 100% green by chance | P(≥1 such cell) | cells Fisher p<0.05 | P(≥1) |
|---|---|---|---|---|---|
| options | ~125 | 1.3 | 31% | 2.9 | 45% |
| shares | ~125 | 3.0 | 57% | 3.3 | 47% |

Across both arms, an analyst with **zero real effect** has about a **70%** chance of
finding at least one "significant" day type. The expected number is about 6 such cells.
Most of them rest on 3–5 sessions. Most real 42-session cells hold 0–2 sessions, so any
cell big enough to look like a finding is mostly made of the noisiest overlaps.

**Rule to adopt now.** No statement of the form "his setups work on day type X" can come
from the lab record, at any session count, unless that day type is named in a new
pre-registration **before** the sessions it is tested on are recorded. H6 stays
descriptive, as written. Do not let anyone upgrade it later.

---

## 3. "Ranged vs trend" is decided at 16:00. How much is known by mid-morning?

`day_class` uses `co_bp / range_bp`, the open-to-close move divided by the full-day
range. You only know that at the close. `ranged_b` (`rel_range < 0.6`) is also an
end-of-day number. `ranged_a` ("no line broke all day") can be confirmed false early
(92% of sessions break a line by 09:40) but can only be confirmed true at 16:00. A label
you can only assign after the close cannot be used to choose a morning trade.

**Timing note.** `first60_hi_bp` / `first60_lo_bp` cover 09:30–10:29. Decision time is
the close of the 10:29 bar, i.e. **10:30 ET, not 10:00**.

**The test.** Sessions 2016–2022 are for fitting, 2023–2026-10 for evaluation, QQQ, SPY
as replication. At 10:30, from the features file only:
- `W = (first60_hi_bp − first60_lo_bp) / med20_range_bp`: how wide the first hour was,
  relative to normal;
- `A = (first60_hi_bp + first60_lo_bp) / (first60_hi_bp − first60_lo_bp)`: where the
  open sat inside that first-hour range (−1 to +1), i.e. which way the first hour leaned.

Fit logistic regression, no tuning, both features linear, on two targets:
1. **Magnitude:** `rel_range ≥ 1.5` (the pre-registered "high" band).
2. **Day type:** `day_class == trend`.

Report out-of-sample AUC for each, with a session bootstrap CI. AUC measures how well a
score ranks the days: 0.5 is a coin flip, 1.0 is perfect.

**Built-in bias to name.** `co_bp` includes the first hour. A day that rose 60 bp by 10:29
starts with a head start toward |co|/range ≥ 0.6, so some predictive power for the trend
label is **mechanical**. The file has no 10:29 close, so the clean version (does the
**rest of the day** keep going the first hour's way?) needs one more column. Until that
column exists, any trend AUC is an upper bound.

**Prediction, from the magnitude-not-sign finding.** Target 1 (magnitude) will be
predictable, because volatility clusters and a wide first hour means a wide day. I
expect AUC ≥ 0.75. Target 2 (trend) will be near chance once the mechanical part is
removed. I expect AUC 0.55–0.62 even with the bias left in. That is the four earlier
findings again: you can tell it will be a **big** day by 10:30, but not that it will be a
**trending** day, and not which way.

**Falsifier (fixed now).** If the trend target reaches out-of-sample AUC ≥ 0.65 on QQQ and
replicates on SPY, day type is knowable early enough to act on, and my reading is wrong.
If it comes in below 0.65, "ranged vs trend" is something you can only describe after the
close, and it cannot be the missing variable. **Cost to settle:** about 30 lines on the
features file once it exists. No new data.

---

## 4. The geopolitical list is selected on the outcome

**Mechanism.** The rule "led the front page of two of Reuters, AP, WSJ, FT, Bloomberg" puts
the market inside the selection. Three of those five are financial papers, and they lead
with whatever moved markets. The rule also asks for a "plausible channel to US equities,"
which is easiest to argue after the market has reacted. Most seriously, the list is built
by an LLM agent that already **remembers** which days crashed: it knows 2020-03-16 and
2025-04-03 to 04-07. "Built before any feature file exists" does not make the builder
blind to prices. So the geopolitical arm is biased toward passing H2/H3, and the size of
that bias is unknown.

**Prediction.** If selection drives the result, the curated list's `rel_range` excess will
be much larger than that of an equally "big" list chosen without reference to markets.

**Controls, specified now.**
1. **Placebo dates (cheapest).** 1,000 draws of random sessions, each the same size as
   each category, matched by year and weekday, with real event days excluded. Run the
   identical H2 statistic. A category must beat the 95th percentile of its own placebo
   distribution. This checks the method, not the selection bias.
2. **Market-blind list.** Build a second list by a rule that never looks at markets: for
   example, the top-N days by article volume in GDELT for the same eight categories, or
   the lead item of AP's general (non-business) wire, with a timestamp **before 09:30 ET**
   on the reaction session. Same N per category, frozen before features are read.
3. **Pre-open timestamp rule** for the curated list itself: drop any row whose earliest
   source is timestamped after the reaction session opened. News that only exists in
   coverage of the move is the move, not its cause.

**Falsifier (fixed now).** If the market-blind list's median `rel_range` excess is at
least **half** the curated list's, selection is not the main story and the curated
results can stand. If it is under half, the excess is selection, and the geopolitical
H2/H3 results are reported as "contaminated by selection", not as findings.

**Also:** `pandemic_health` and `financial_stress` rows will cluster in Feb–Apr 2020 and
Mar 2023, while `med20_*` is still catching up to the new regime. Those rows will show
large `rel_range` because the trailing baseline is stale, not because of the news. They
are also not independent events. Report them with cluster-robust CIs (clustered by
episode) or not at all.

**Cost to settle:** placebo is minutes once features exist. The market-blind list takes
one agent pass, the same size as the geopolitical list.

---

## 5. Most likely outcome, and what to do when it arrives

My expectations: H1 passes, with FOMC days roughly 0.2–0.4 of a normal day wider, the
widely known result the pre-registration already expects. CPI likely passes on 2022–2023
data and NFP probably also passes. One or two geopolitical categories "pass", with an
effect I would expect to shrink by at least half under the market-blind control.
Ranged days come out rare: about 8% on `ranged_a` and somewhat more on `ranged_b`. Event
days that were not "affected" will look like ordinary days. H6 will be a 42-session table
with about 9 event days in it, saying nothing. **None of this will show a buyer of calls or
puts how to make money**, because the wider days are priced (§1) and the direction is
not predicted (§3). The likely answer to "what are we missing" is: **nothing. The flat
result is the expected result for a buyer with a magnitude-only signal paying the
variance premium and the spread.**

**What the owner should do if that outcome arrives:**
- Stop looking for the day filter. Do not turn day-type cells into trading rules, and do
  not scan the lab against them (§2).
- Accept that, on every measurement this project has made, his calls and puts carry no
  edge. Day type does not change that.
- Before any more work on "why it doesn't work," check the premise: has the journal's
  **+$3.40/trade over 442 trades** (about $1,500 in total, P&L dominated by a few big
  trades) been checked against **broker statements**, rather than the hand-kept CSV? Has
  it been tested against zero, with day clustering? If it is not distinguishable from
  zero, there is no profitable manual strategy to explain, and the whole search for the
  missing variable has no target. That check costs one statement download and one
  t-test. Of everything in this file, it is the cheapest test that could change the
  conclusion.
