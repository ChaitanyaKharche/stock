# Results — IntradayMomentumBoundary on QQQ shares, stress test

Run 2026-08-29 before any decision to build on this strategy. Code
`trade_analysis/backtesting/imb_stress.py`, reading the trade records already written by
`trade_analysis/live_lab/sharewf.py` (2,905 IMB trades, 1,560 sessions, 2016-01-20 →
2026-08-27, $10,000 notional per trade, real NBBO both sides, next-bar fills).

Nothing here re-fits or re-tunes. Every test was listed in the module docstring before any of
them was read.

---

## Why this strategy and not another

IMB is the only item in this project that has **cleared a multiplicity correction on a large
sample and survived honest execution timing**. It is also the only setup in the lab whose
parameters were not invented here — they come from Zarattini/Aziz/Barbon, SSRN 4824172, and
`setups.py` flags it as *"the only item needing no invented parameter."* There is no tuning to
undo.

It matters that IMB **died on options** (+16.46% → +0.70%, p=0.79 once fills are lagged a
minute) and **lived on shares**. That is not inconsistency: QQQ's spread is ~0.5 bp round
trip, an ATM 0DTE costs ~1% of premium plus theta. A +3.3 bp signal is hopeless against the
second cost floor and comfortable against the first.

---

## 1. A correction to my own number first

My script initially reported **15.7%/yr**. That was wrong. It annualised on
*active-trading-days ÷ 252*, but IMB only fires on **1,560 of ~2,657 sessions** — the capital
has to sit in the account on the other 1,097 too.

**On calendar time: $9,717 over 10.60 years = $917/yr = 9.2%/yr on $10,000 = $18/week.**

---

## 2. What survives the audit

| test | result |
|---|---|
| headline | +$3.34/trade, CI [+1.75, +4.96], **p = 0.0001** |
| **split-half, independent windows** | first half **+$2.92 (p 0.0040)**, second half **+$3.78 (p 0.0004)** |
| per-year | **10 of 11 years positive** |
| exclude best year (2022) | +$2.85, **p = 0.0001** |
| exclude worst year (2019) | +$3.80, p = 0.0001 |
| direction | long +$3.51 (p 0.0004), short +$3.16 (**p 0.0094**) — both sides |
| exclude best checkpoint (10:01) | +$3.19, **p = 0.0006** |
| day concentration | best day 9.4% of total, top 10 days 33.4% — broad |
| slippage +1c/share/side | +$2.48, p = 0.0018 |
| slippage +2c/share/side | +$1.62, **p = 0.0450** — still clears |
| breakeven extra slippage | **3.88 cents per share per side** |
| monthly | 128 months, **62.5% positive**, **annualised Sharpe 1.39** |

**Both halves of the sample are independently significant.** That is the single most
reassuring line in this document, and it is the test that most of the failed work in this
project would not have passed.

---

## 3. What does not survive, and it is serious

### Tail dependence

| | share of total P&L | remaining |
|---|---|---|
| top 1 trade | 10.8% | +$8,664 |
| top 10 | 35.3% | +$6,286 |
| **top 1% (29 trades)** | **73.9%** | +$2,538 |
| top 5% (145 trades) | **216.4%** | **−$11,308** |

Removing the top 1% leaves **+$0.88/trade, CI [−0.40, +2.19], p = 0.1764 — not significant.**

Median trade **−$6.81**. Win rate **27.2%**. Skew **6.08**.

**How to read this honestly.** A positive-skew strategy has its edge *in the tail by
construction* — trend-following fails an "ex top 1%" test universally, and that is not by
itself disqualifying. The question is whether the tail recurs. The evidence that it does:
both halves significant, 10 of 11 years positive, both directions positive. The evidence that
it might not: 29 trades is a thin reed to hang a decade on, and the mean is estimated with
correspondingly little precision.

### The experience of trading it

| | |
|---|---|
| max drawdown | **$1,236 = 12.4% of capital** |
| trades spent below a prior equity peak | **93.5%** |
| longest stretch below a prior peak | **2018-12-27 → 2021-03-02, 796 days (26.2 months)** |
| second longest | **2025-05-21 → 2026-06-30, 405 days (13.3 months)** |
| worst month | −$549 (5.5% of capital) |

**You would have spent 26 consecutive months below a previous high, at a 27% win rate, on a
strategy paying $18 a week.** That is the honest description of what running this feels like,
and it is the part no backtest statistic conveys.

---

## 4. A finding I am deliberately not acting on

| exit | n | share | mean | total |
|---|---|---|---|---|
| trailing stop | 2,281 | 78.5% | **−$11.69** | **−$26,655** |
| end of day | 624 | 21.5% | **+$58.29** | **+$36,372** |

Winners hold 195 minutes, losers 33. It looks exactly like *"the trailing stop is destroying
the strategy — remove it."*

**That inference is invalid, and it is the same trap as this morning's MFE result.** Trades
that survive to the close are *selected* for not having gone against you. You cannot delete
the trailing stop and keep the EOD column; the 2,281 stopped-out trades would have run on with
unknown, probably worse, results. Settling it needs a full re-run with the stop removed — and
that would also mean deviating from the published parameters, which is the one thing making
this strategy credible. **Logged as a hypothesis, not a change.**

---

## 5. Peer context

| setup | n | $/trade | total | p |
|---|---|---|---|---|
| Crabel_Stretch | 2,438 | **+4.21** | +10,263 | 0.0164 |
| **IntradayMomentumBoundary** | 2,905 | +3.34 | +9,717 | **0.0001** |
| ORB_5min | 2,330 | +1.22 | +2,838 | 0.3590 |
| ORB_15min | 2,485 | +0.92 | +2,285 | 0.3258 |
| VWAP_Reclaim | 3,000 | +0.35 | +1,056 | 0.6046 |
| **MOMO_CHASE** (his own behaviour) | 5,586 | **+0.05** | +295 | 0.8894 |
| TTM_Squeeze | 2,744 | −0.29 | −783 | 0.5800 |
| PDH_PDL_FailedBreak | 1,746 | −2.05 | −3,586 | 0.1448 |

`Crabel_Stretch` has a **higher mean** than IMB and does not clear Holm. It is worth watching
but has invented parameters, so it carries the tuning risk IMB does not.

**MOMO_CHASE — the encoding of his own trading — is +$0.05 per trade over 5,586 trades.**

---

## 6. The verdict, and the thing that makes it hard

IMB **passes** every structural robustness test: split-half, per-year, per-direction,
per-checkpoint, slippage to 2 cents a share, and day concentration. It **fails** the tail test,
and it delivers 9.2%/yr with 26-month underwater stretches.

**The awkward part: a forward test cannot settle this.** It took **10.6 years to accumulate
2,905 trades**, and the 95% CI on the mean is still [+1.75, +4.96]. A year of paper trading
adds ~264 trades — about 9% more information. There is no version of "run it forward and see"
that produces an answer on a useful timescale.

So the decision does not get to wait for better evidence. The 10.6-year record, with untuned
published parameters and two independently significant halves, **is** the evidence, and the
tail dependence **is** the caveat. Both are now measured.

---

## 7. SPY — the cross-instrument test, run 2026-08-29

The same published rule, the same code, the same honest fills, on **SPY 2022-01-03 →
2026-08-27** — an instrument this had never been run on. Nothing was re-tuned.

```
setup                           n   win%    mean$    total$      p     Holm
IntradayMomentumBoundary     1316  25.6%    +2.60    +3,418  0.0007   0.0087   <- clears
Crabel_Stretch                975  45.5%    +3.21    +3,125  0.1487   1.0000
Gap_Fade                      281  42.0%    +7.19    +2,020  0.0347   0.3813
MOMO_CHASE                   2568  50.9%    +0.28      +712  0.5340   1.0000
...
clearing Holm with POSITIVE P&L (m=13): 1 -> ['IntradayMomentumBoundary']
```

**Again the only one of thirteen to clear.** And `MOMO_CHASE` — his own behaviour — is again
nothing (+$0.28, p = 0.534) on a second instrument.

### But it adds less than it looks, and one part of it points the wrong way

| | |
|---|---|
| daily P&L correlation with QQQ on shared days | **0.717** |
| sign agreement on shared days | 78.5% |
| QQQ active days / SPY active days / both | 1,560 / 711 / **591** |

At r = 0.72 roughly half the variance is shared, so this is substantially the *same* signal
seen twice rather than two independent confirmations.

**And the genuinely independent part is negative.** On the **120 days where SPY fired and QQQ
did not** — the only days carrying information QQQ does not already contain — the strategy
lost **−$1,191, −$9.93 per day**. Small sample, and those days are plausibly the weaker-trend
days that failed to trigger QQQ at all, but it is the right test and it does not go the
strategy's way.

### Diversifying across both makes it worse per dollar

| | total | capital | annualised |
|---|---|---|---|
| QQQ alone | +$9,717 | $10,000 | **9.2%/yr** |
| both, one position each | +$13,135 | $20,000 | **6.2%/yr** |

Same window head to head, QQQ is also the better vehicle: **+$4.08/trade (QQQ 2022-2026) vs
+$2.60 (SPY)**.

**Net:** SPY confirms the rule is not a QQQ-specific artefact, which is worth having. It does
not double the evidence, and it does not improve the strategy.

---

## 8. What a forward test can and cannot do here

It took **10.6 years to accumulate 2,905 trades** and the CI on the mean is still
[+1.75, +4.96]. A year of forward paper trading adds ~264 trades — about **9% more
information**. Running it forward will not settle whether the edge is real.

What running it forward *does* settle is whether the **plumbing** is right: signal timing,
fills, position management, the MST/ET clock, restart behaviour, feed outages. Those are
failure modes that have already bitten this project repeatedly, and they are exactly what a
prospective paper run catches cheaply. That is the honest reason to do it — engineering
validation, not statistical validation.
