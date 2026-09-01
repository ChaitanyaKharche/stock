# Results — The package: his entries + his exits + strict ATM + a free public setup filter

**Executes `research/package_preregistration.md`.** Run 2026-08-26. Zero new API calls.

> ## HEADLINE
>
> **0 of 10 free public setups improve his weekly dollars.** The one that improves per-trade
> quality — ORB continuation, +55% on $/trade — costs almost exactly as much in lost trade
> volume as it gains in quality (+$36.62/wk quality, **-$38.49/wk volume**). Net **-$1.87/week.**
>
> **A separate and more important finding:** NBBO-priced simulation overstates his realised
> cash by **+$6.85/trade** on the same trades. Every backtest number in this project priced at
> NBBO — including this one and C-money — carries that optimism.

## Funnel

```
QQQ/SPY round trips                  357
  no setup flags                      23
  evaluated                          334      120 days, 2025-05-02 -> 2026-02-20, 42.0 weeks
strict-ATM check: strike from spot   median 0.0417%   max 0.0868%
ATM median entry ask $1.18   his median premium $0.67
```

Strike distance confirms the instrument is **strict ATM, no offset** — max 0.087% from spot,
under half a strike on QQQ at $600. This is the constraint as stated, not "ATM +/- 1."

**Long premium only. No arm in this test sells a contract.**

## PRIMARY — $/week vs unfiltered, Holm across ten

```
filter            d $/week                  95% CI          p      Holm    MDE(80%)
vwap_side            +1.98 [  -70.63,   +82.31]   0.9946   1.0000    +109.21  [underpowered]
orb_cont             -1.87 [  -90.03,   +85.86]   0.9742   1.0000    +125.02  [underpowered]
vwap_reclaim        -51.29 [ -208.60,  +102.98]   0.4994   1.0000    +222.04  [underpowered]
rvol               -109.03 [ -265.37,   +38.37]   0.1468   1.0000    +216.77  [underpowered]
fade_edge          -109.48 [ -275.06,   +51.13]   0.1762   1.0000    +232.05  [underpowered]
flag               -105.11 [ -269.86,   +55.84]   0.1904   1.0000    +230.94  [underpowered]
pd_level           -122.11 [ -272.01,    +4.08]   0.0590   0.5900    +196.81  [underpowered]
gap_and_go         -126.78 [ -275.59,    +9.06]   0.0702   0.5900    +202.54  [underpowered]
pullback_9ema      -132.49 [ -287.86,    +7.66]   0.0652   0.5900    +211.18  [underpowered]
late_breakout            -   arm could not execute: 0 trades

clearing Holm: 0 of 10
```

Family size held at **m = 10** per section 10, not reduced because `late_breakout` had no
trades to run.

**Every arm is underpowered for its own effect.** Filters that keep few trades are
underpowered *by construction* — that is a property of the filter, not a flaw in the test.
Seven of nine executable arms point negative.

## SECTION 6 — volume versus quality, in full

| filter | n | kept | $/trade | $/week | win% | quality $/wk | volume $/wk |
|---|---|---|---|---|---|---|---|
| **(unfiltered)** | 334 | 100% | **+13.25** | **+105.37** | 47.0% | — | — |
| vwap_side | 286 | 85.6% | +15.77 | +107.36 | 47.6% | +17.13 | −15.14 |
| **orb_cont** | 212 | 63.5% | **+20.50** | +103.50 | **50.9%** | **+36.62** | **−38.49** |
| vwap_reclaim | 91 | 27.2% | +24.96 | +54.09 | 46.2% | +25.38 | −76.66 |
| flag | 2 | 0.6% | +5.62 | +0.27 | 100.0% | −0.36 | −104.74 |
| rvol | 37 | 11.1% | −4.15 | −3.65 | 59.5% | −15.33 | −93.70 |
| fade_edge | 6 | 1.8% | −28.74 | −4.11 | 16.7% | −6.00 | −103.48 |
| pd_level | 135 | 40.4% | −5.21 | −16.73 | 51.9% | −59.32 | −62.78 |
| gap_and_go | 117 | 35.0% | −7.68 | −21.40 | 53.0% | −58.32 | −68.46 |
| pullback_9ema | 78 | 23.4% | **−14.60** | **−27.12** | 50.0% | −51.73 | −80.77 |
| late_breakout | 0 | 0% | — | — | — | — | — |

**This table is the deliverable.** It was declared in advance to be the point of the
experiment regardless of significance, and it is the part that survives.

### What it says

- **ORB continuation is the only setup that makes his trades better and it changes nothing.**
  $/trade rises 55% (+13.25 → +20.50), win rate rises 3.9pp (47.0% → 50.9%) — and $/week
  falls, because the 36.5% of trades it discards were worth **−$38.49/week** against the
  **+$36.62/week** of quality gained. **The cancellation is near-exact.** This is the
  interaction the earlier percent-return test was structurally unable to see, and it is now
  measured.
- **Several setups raise win rate while destroying dollars.** `rvol` wins 59.5% of the time
  and returns −$4.15/trade. `gap_and_go` wins 53.0% and returns −$7.68. Win rate is not the
  objective and these arms show why.
- **`pullback_9ema` is the worst filter tested.** It keeps 23.4% of his trades and converts
  +$13.25/trade into **−$14.60/trade**. This is the setup he specifically asked about, and it
  is consistent with the well-powered standalone null (**−0.0017 sigma** over 1,542 sessions).
- **`late_breakout` = 0 trades and `flag` = 2.** He never enters at a session extreme after
  14:00 and essentially never waits for a consolidation bar. Already established; re-confirmed.

## The baseline read was INVALID — recorded in full

The unfiltered arm printed **+$105.37/week** against his realised tape's **+$25.77/week** on
the same trades, suggesting strict ATM was worth ~4x. **That reading is wrong** and is
retracted here rather than reported.

Section 4 designates the baseline as context, not a test. It was checked before use. Three
arms, same 276 trades, same matched capital:

| arm | $/trade | $/week | win% |
|---|---|---|---|
| his contract @ NBBO ask→bid | +$15.18 | +$99.75 | 46.7% |
| **ATM @ NBBO ask→bid** | **+$16.13** | +$105.98 | 47.5% |
| his **realised broker cash** | +$8.33 | +$54.71 | 47.5% |

```
ATM - his contract, both at NBBO : +$0.95/trade  CI [-$7.95, +$8.87]  p 0.7754
his NBBO - his realised cash     : +$6.85/trade
```

**The strict-ATM instrument is worth +$0.95/trade and is not significant** — reproducing Test
A's +$0.98/trade and +1.37% independently, on a different sample with a different weighting.
Two independent estimates agreeing at ~$1/trade is the most solid thing in this file.

**The apparent 4x was an artefact of comparing NBBO pricing to realised cash.**

## The $6.85/trade NBBO gap — the most consequential number here

On identical trades at identical capital, NBBO ask→bid simulation produces **$6.85/trade more
than he actually banked**. At ~8 trades/week that is roughly **$55/week of modelling
optimism**.

**Composition unknown, and this dataset cannot resolve it.** Candidates:

1. Real slippage — consistent with the previously established finding that **41.3% of his
   sub-second market buys print outside the reported NBBO**.
2. Quote-timing optimism — the "prevailing quote at his entry second" may be stale in a
   favourable direction.
3. Partial-fill aggregation — round trips VWAP multi-fill exits into one price; the NBBO arm
   uses a single exit second.

No execution-quality claim is available, because the broker's `client_bid/ask_at_submission`
fields are sign-negated on the `strategy_detail` code path. **The gap is stated as a
calibration fact, not attributed to a cause.**

### Consequence for C-money

`entry_timing_results.md` reported a mechanical ATM fixed-hold at **+$0.61/trade / +$4.93/week
at 1 contract**, priced at NBBO. His median position here is ~$250 capital, roughly 2 ATM
contracts, so the gap is ~$2.70-3.40 per contract. Applying it, **the mechanical strategy is
most likely negative in live trading, not +$4.93/week.**

This is a caveat, not a restatement: the gap's composition is unknown and it may not transfer
one-for-one to a different order type. But the direction is unambiguous, and **+$4.93/week
should not be treated as achievable.**

## Verdict against section 9

Row 2: **no filter clears — no free public setup improves the package. Report the
volume-quality table and stop.**

Row 3 also fires for `orb_cont` and `vwap_reclaim`: both raise $/trade and both lower or
fail to raise $/week. Per the pre-registered rule these are **not** described as working.

**Ninth consecutive pre-registered null.**

## What this does not establish

- Not that ORB continuation is useless — that it is **volume-neutral** on his trade
  population, which is a different and more specific claim.
- Not that no filter could help. MDE ranges $109-232/week; only very large effects were
  detectable. Nothing smaller is excluded.
- Nothing out-of-sample. Same May 2025 - Feb 2026 window.
- Nothing about a trader who exits mechanically. Every arm inherits **his** exit second, by
  design, because his exit is the only thing in this programme that has ever cleared Holm.
