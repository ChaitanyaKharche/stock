# Results — Historical walk-forward over the frozen setups (2022-2026)

Run 2026-08-28. `trade_analysis/live_lab/walkforward.py` + targeted option pricing.
**Context for the forward test. Changes no frozen rule, resets no clock, promotes nothing.**

## Sample

**23,519 signals over 2,316 symbol-days (QQQ + SPY, 2022-01-03 -> 2026-08-27).**
Strictly walk-forward: session D seeded only from sessions before D, bars admitted one at
a time, same caps and exits as the live runner.

## Stage 1 — underlying only: does any setup call DIRECTION?

| setup | n | win% | mean move | 95% CI | p | Holm |
|---|---|---|---|---|---|---|
| **IntradayMomentumBoundary** | 2593 | 26.6% | **+0.0356%** | [+0.0162%, +0.0563%] | 0.0013 | **0.0173** |
| Crabel_Stretch | 2032 | 46.0% | +0.0455% | [-0.0003%, +0.0944%] | 0.0507 | 0.5680 |
| Gap_Fade | 678 | 44.7% | +0.0548% | [+0.0003%, +0.1118%] | 0.0473 | 0.5680 |
| **MOMO_CHASE** | 5141 | 51.3% | +0.0067% | **[-0.0021%, +0.0156%]** | 0.1380 | 1.0000 |
| ORB_5min | 1867 | 45.2% | +0.0187% | [-0.0141%, +0.0544%] | 0.2787 | 1.0000 |
| VWAP_Reclaim | 2505 | 39.1% | -0.0090% | [-0.0239%, +0.0074%] | 0.2787 | 1.0000 |
| TTM_Squeeze | 2450 | 44.4% | +0.0036% | [-0.0098%, +0.0159%] | 0.5827 | 1.0000 |
| ORB_15min | 2195 | 35.2% | +0.0084% | [-0.0137%, +0.0313%] | 0.4400 | 1.0000 |
| PDH_PDL_Breakout | 1978 | 35.0% | +0.0096% | [-0.0088%, +0.0289%] | 0.3053 | 1.0000 |
| PDH_PDL_FailedBreak | 1538 | 38.3% | -0.0101% | [-0.0448%, +0.0242%] | 0.5480 | 1.0000 |

**1 of 13 clears Holm.** Its fragility was visible immediately:

- **tail:** top 1% (25 of 2,593) = **66.5%** of the total move; strip them -> +0.0121%. Median trade **-0.0642%**.
- **decay:** 2022 +0.0699% -> 2023 +0.0325% -> 2024 +0.0276% -> 2025 +0.0254% -> 2026 **+0.0111%**. The SSRN paper published 2024; the years since are its weakest.
- **not beta:** longs +0.0484%, shorts +0.0229%. Both positive, in a mostly-rising market. This part is genuine.

### MOMO_CHASE — the best-powered null in the programme

n = 5,141. The **entire 95% upper bound (+0.0156%) sits below the ~0.05% conversion floor.**
Not "underpowered": *any effect that exists is too small to pay for an option.*

## Stage 2 — IMB with REAL historical ATM 0DTE quotes

1,996 signals from 2023 (when QQQ daily expiries begin), entry ask -> exit bid, fee
$0.0404/side. **Zero pricing skips.**

The mean-vs-floor comparison was the WRONG test and I had been making it. Option payoffs are
**convex** in the underlying move, and IMB is almost pure right tail — exactly the profile
that converts better than its mean predicts. Measured:

```
und +0.30%..    n= 257   option +173.6%   win 98.4%
und +0.15%..    n= 117   option  +40.3%   win 78.6%
und +0.05%..    n= 103   option   -6.3%   win 35.0%
und  0.00%..    n=  51   option  -36.6%   win  0.0%
```

The convexity is real. Crossover sits between +0.05% and +0.15%.

## The finding: 95.7% of the edge was a one-minute lookahead

A 1-minute bar stamped `T` covers `[T, T+60s)`. Its **close** triggers the signal and is not
knowable until `T+60s`. The first pricing pass used the option quote stamped `T` — the price
at the *start* of the minute whose close confirms the breakout. For a momentum setup that is
buying before the move completes.

| | with lookahead | honest timing |
|---|---|---|
| mean option return | +16.46% | **+0.70%** |
| p | 0.0003 | **0.7900** |
| 95% CI | [+10.45%, +22.71%] | **[-4.40%, +5.73%]** |
| total @ 1 contract | +$33,508.72 | **+$3,432.80** |
| per trade | +$16.79 | **+$1.72** |
| win rate | 35.5% | 19.2% |

Re-running with a clean +1 min on both legs (an earlier attempt over-penalised the exit by
two minutes) moved it only +1.05% -> +0.70%: **the kill is the lookahead, not the penalty.**

What survives does not survive scrutiny:

- top 1% (19 of 1,995) = **756.5%** of total return; without them **-4.66%**
- **2025 -5.59%, 2026 -1.34%** — both recent years negative
- CI spans zero

**IntradayMomentumBoundary does not survive honest execution timing.**

## Verdicts

1. **0 of 13 setups produce a tradeable edge** over 4.6 years and 23,519 signals, with real
   option quotes and honest timing.
2. **MOMO_CHASE is ruled out by its own confidence interval**, not merely unproven.
3. **The convexity thesis is confirmed but insufficient.** Options do amplify a fat right
   tail — the +0.30% bucket returns +173.6% at a 98.4% win rate. IMB just cannot reach that
   bucket often enough once you wait for the bar to close.
4. **The live runner is unaffected.** It waits `bar_close + 1500ms` before touching the
   chain, which is exactly why `store.write_decision()` fsyncs the decision before any price
   is requested. Day 1's -$1,201.18 was measured honestly.

## What this does NOT license

No frozen rule changes. No clock resets. No setup promoted or demoted. No parameter loosened
to rescue anything. This is a backtest with publication bias baked into nine of its thirteen
members; the forward test remains the only instrument that can settle the question, and it
runs Monday at 05:30 regardless of anything written here.
