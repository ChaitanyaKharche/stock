# Pre-registration — The package: his entries + his exits + strict ATM + a free public setup filter

**Status: PRE-REGISTERED, NOT YET RUN.** Written 2026-08-26, before any number computed.

## 0. What is new here, stated precisely

This is **not** a re-run of `setup_crosscheck_preregistration.md`. That test asked *does
setup S separate his returns*, in **percent**, on **his own contracts**, across **all
symbols**. It returned 0 of 7 clearing Holm with an **MDE of 11-12 percentage points** —
larger than his entire per-trade edge, so it excluded only very large effects.

Three things differ, and each is load-bearing:

1. **The endpoint is dollars per week, not percent return.** These can point in opposite
   directions. A filter that keeps 20% of his trades can raise mean return *and halve his
   weekly income.* Percent-return tests are structurally incapable of seeing that. His stated
   objective is **$200-500/week**, so dollars/week is the endpoint that matches the decision.
2. **The instrument is the strict ATM contract**, which he has now committed to — nearest
   listed strike, no offset. Not the OTM contracts he actually bought.
3. **His discretionary exit is preserved.** The ATM contract is exited at *his* exit second.
   This is deliberate: his exit is the only thing in this programme that has ever cleared
   Holm (+8.97pp vs a mechanical +25% target). The package keeps what works and varies only
   what he asked about.

## 1. The honest prior, recorded before running

**I expect this to be null**, for two reasons already measured: setup coverage is
**saturated at 91.4%** (at least one setup matches nine entries in ten, so a match carries
almost no information), and no setup separated his returns at 0/7.

**The primary value of this experiment is therefore not its p-values.** It is the
**volume-versus-quality table** in section 6, which is decision-relevant whether or not
anything is significant, and which has never been computed. If a filter improves per-trade
quality but costs more in lost trade count, that is an actionable finding at any p.

I am recording this so a null cannot later be presented as expected all along, and so the
table cannot later be presented as a discovery.

## 2. Sample

QQQ/SPY round trips present in **both** `round_trips.csv` and the cached `setup_match.csv`
(1:1 join on date + symbol + pct, verified 334 QQQ/SPY keys, no collisions), **and** having a
cached strict-ATM quote path covering both his entry second and his exit second.

Single names are excluded — no options entitlement, a pre-existing constraint, not a choice
made here. Funnel reported.

`pct` is used **only as a join key**. It is never an outcome. All outcomes are recomputed
from ThetaData NBBO on the ATM contract.

## 3. Construction, fixed now

| | |
|---|---|
| contract | strict ATM: listed strike nearest underlying spot at his entry second. **No offset.** |
| direction | his actual right (call/put). **Long premium only. No arm sells a contract.** |
| entry | that contract's **ask** at his entry second |
| exit | that contract's **bid** at **his actual exit second** |
| size | **matched capital** — sized to the same dollars he actually deployed on that trade, fractional contracts permitted |
| fees | **$0.0404/contract/side**, his real broker rate, both sides |

Matched capital, not matched contract count. ATM costs ~$1.18 against his ~$0.69 median, so
equal contracts would nearly double his risk and make the comparison capital-unfair.

## 4. Arms

**Baseline (not a test):** all qualifying trades, unfiltered.

**Ten filter arms**, one per setup, **fixed from the already-published external list** and not
re-derived here: `vwap_side`, `vwap_reclaim`, `orb_cont`, `flag`, `pullback_9ema`,
`gap_and_go`, `fade_edge`, `late_breakout`, `rvol`, `pd_level`. A filter arm keeps only
trades where that flag is True and **skips the rest entirely** — no substitute trade.

**No combination of filters is tested.** No threshold inside a setup is swept. Adding
combinations would multiply the family into exactly the search that manufactured this
project's earlier false positives.

## 5. Primary endpoint and statistic

**$/week** = (total dollars across the arm) / (weeks spanned by the sample). The denominator
is a **fixed calendar constant, identical across all arms**, so arms differ only in numerator.

**Statistic:** paired day-clustered bootstrap, 10,000 reps. Each replicate resamples the same
set of activity dates for **every** arm simultaneously, so baseline and filter see identical
resampled days and the comparison is properly paired.

**Comparison:** filter $/week minus baseline $/week.

**Multiplicity: family m = 10. Holm across all ten.** The baseline is not a test.

## 6. Mandatory reporting table — produced whatever the p-values say

For every arm: **trades kept**, **% of baseline trades kept**, **$/trade**, **$/week**, **win
rate**, and **$/week change vs baseline decomposed into a quality term and a volume term.**
This table is reported in full, with no cell omitted for being uninteresting and none
promoted for being interesting.

## 7. Power

The achieved MDE is computed from the realised bootstrap SE and printed beside every effect.
**If |effect| < MDE the result is labelled underpowered regardless of its p-value.** Same
rule as the construction and entry-timing tests; not relaxed.

Filters that keep few trades will be badly underpowered by construction. That is a property
of the filter, not a defect of the test, and will be reported as such rather than hidden.

## 8. Leak audit

| | |
|---|---|
| setup flags | computed at the last completed bar **before** each entry second in the original cross-check. Not recomputed here, so no opportunity to re-tune them against this outcome |
| ATM strike | underlying spot at his entry second — known at entry |
| exit second | his actual exit. Known only afterwards, but applied **identically to every arm**, so it cannot favour a filter |
| filtering | uses only the pre-entry flag. **No arm filters on outcome** |
| join key | `pct` used for identity only, never as a variable |
| quote validity | bid > 0, ask > 0, ask >= bid; 09:31-15:59 |

**Known and unfixable in this design:** the exit second is his, so every arm inherits his
discretion. This test cannot say whether a filter would help a trader who exits mechanically.
It answers only the question asked: does a filter help *him*.

## 9. Decision rule, committed now

| result | conclusion |
|---|---|
| a filter's paired CI excludes 0 and clears Holm, effect > 0 | that setup improves his weekly dollars — first positive filter result in the programme |
| no filter clears | **no free public setup improves the package.** Report the volume-quality table and stop |
| a filter raises $/trade but lowers $/week | quality gained, volume lost — reported explicitly as the trade-off it is, and **not** described as the filter "working" |

## 10. Prohibited

No setup added, removed, or redefined. No combinations. No thresholds swept. No per-symbol or
per-year cell promoted. Family stays 10 if an arm is too thin to execute. **No arm may sell an
option.** No result reported without its trades-kept count beside it.
