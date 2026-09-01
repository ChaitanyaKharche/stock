# Results — Strike choice and exit policy (construction tests)

**Executes `research/construction_preregistration.md` (2026-08-20) in full.** Run 2026-08-26.
Both tests, Holm across the pre-registered family of **five**.

## Funnel

```
QQQ/SPY round trips in round_trips.csv        357
  Test A evaluated (his path AND ATM path)    296     61 excluded: missing_path
  Test B evaluated (his path)                 296     61 excluded: no_his_path
distinct days                                  89
date range                            2025-05-13 -> 2025-11-26
```

**The window is in-sample.** It is set by the quote cache built for the earlier percentile
experiment, not chosen. The PRE/POST held-out blocks are not covered here.

## TEST A — his strike vs ATM

Same entry second, same exit second, both arms priced ask -> bid, so only the strike differs.

| | his strike | ATM |
|---|---|---|
| median distance from spot | 0.261% | **0.045%** |
| median entry ask | $0.68 | $1.06 |
| mean return | +7.59% | **+8.95%** |
| win rate | 47.3% | 48.0% |

```
PAIRED (ATM - his)   +1.37%   95% CI [-0.35%, +3.11%]   p 0.1182   Holm 0.3546
matched capital      +288.82 total   +0.98 / trade
achieved MDE (80%)   +2.49%   -> effect < MDE, UNDERPOWERED
```

**Does not clear.** By the pre-registered rule in section 5 it is reported as underpowered
regardless of p.

The point estimate favours ATM **despite ATM costing 56% more premium** ($1.06 vs $0.68).
The tighter relative spread and higher delta more than offset the lost percentage leverage.
That direction is consistent with the mechanical argument — spread falls from ~1.8% of
premium to ~0.7% — which is an accounting fact and does not depend on this test.

**Conclusion under the section 7 decision rule:** the strike change is *not supported at
p<0.05*. It is also not contradicted. Adopting ATM rests on the mechanical argument, with
the estimate agreeing but not establishing it. Dollar impact at his sizing is ~$1/trade.

## TEST B — exit policies on his real contracts

Entry at his actual fill; arms differ only in exit. Levels fixed in advance from his own
payoff distribution; **no level swept**.

| policy | mean | median | win rate | total $ at his sizing |
|---|---|---|---|---|
| **P0 his actual** | **+5.08%** | -5.98% | 48.3% | **+3,458.34** |
| P1 +25% target | -3.89% | +25.00% | **71.6%** | -2,347.76 |
| P2 -30% stop | +5.64% | -30.00% | 17.6% | +6,563.91 |
| P3 target+stop | -1.26% | +25.00% | 51.7% | -627.65 |
| P4 hold to 15:45 | +6.86% | -57.17% | 32.8% | +16,789.03 |

## Holm across the full family of five

```
comparison            effect            95% CI          p       Holm    MDE(80%)
TEST A  ATM vs his    +1.37%  [ -0.35%,  +3.11%]   0.1182   0.3546    +2.49%  [underpowered]
P1 +25% target        -8.97%  [-16.07%,  -2.42%]   0.0054   0.0270    +9.72%  CLEARS [underpowered]
P2 -30% stop          +0.56%  [-17.26%, +26.79%]   0.9526   1.0000   +32.05%  [underpowered]
P3 target+stop        -6.33%  [-12.21%,  -0.85%]   0.0226   0.0904    +8.12%  [underpowered]
P4 hold 15:45         +1.78%  [-26.67%, +37.76%]   0.9830   1.0000   +46.57%  [underpowered]

clearing Holm: 1 of 5
```

## The one result that clears

**A fixed +25% profit target loses to his discretion by 8.97 percentage points per trade
(Holm p = 0.027).**

Mechanism, and it is arithmetic rather than interpretation:

- P1 raises the win rate from 48.3% to **71.6%** and still **loses money** (-$2,348).
- A +25%/-100% structure has its own breakeven at **75.3% wins**. P1 delivers 71.6%. Below
  the line.
- His realised structure — average win **+48.1%**, average loss **-37.0%** — has a breakeven
  of **43.5%**, and he runs **45.8%**.

Capping every winner at +25% while losers still run destroys precisely the asymmetry that
makes his record positive at all. P3 shows the same thing more weakly (-6.33%, Holm 0.090,
does not clear).

**This maps to section 7 row 3:** his discretionary exit management is adding measurable
value relative to the mechanical alternatives tested.

## What does NOT follow from this

- **P2 and P4 are not endorsements.** P4's +$16,789 has a **-57.17% median** and a 32.8%
  win rate: a lottery-ticket payoff nobody sits through, with a 46.6pp MDE. P2's +$6,564
  comes with a **17.6% win rate**. Both CIs are enormous and both are noise at this sample.
- **No level is validated.** -30% and +25% were fixed, not swept. "A different target might
  work" is a new pre-registered question, not an inference from this table.
- **Nothing here is out-of-sample.** May-Nov 2025 only.
- **Every comparison is underpowered for its own effect.** Directions are credible;
  magnitudes are likely inflated by the usual winner's-curse mechanism.

## Standing corrections carried forward

- The 75.3% breakeven is **exit-structure-specific**, not a universal constant. It applies to
  a +25% target. His actual tape's breakeven is 43.5%.
- No execution-quality claim is available from the broker export: `client_bid/ask_at_submission`
  is sign-negated on the `strategy_detail` code path, and 41.3% of sub-second market buys print
  outside the reported NBBO. Both arms here are priced from ThetaData NBBO, not broker fields,
  so this test is unaffected.
