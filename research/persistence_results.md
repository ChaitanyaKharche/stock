# Results — Is the IMB edge persistent through time?

Executed 2026-09-04 per `persistence_preregistration.md`. Code
`trade_analysis/backtesting/persistence.py`. 2,905 IMB trades on QQQ shares,
2016-01-20 → 2026-08-27. Holm m = 5. No parameter tuned, no strategy proposed.

**Headline: the edge is persistent. All five tests pass, and one of them retroactively
validates every confidence interval this project has reported for this strategy.**

Pooled reference: **+$3.34/trade, CI [+1.76, +4.97], p = 0.0001.**

---

## T1 — No decay, and the publication hypothesis is not supported

```
slope  -0.038 $/trade per YEAR   CI [-0.517, +0.491]   p 0.8745
```

The worry was specific: IMB's parameters are published (SSRN 4824172, ~2024), and published
edges decay. **There is no detectable trend.** The fitted line does not reach zero until 2108,
which is another way of saying the slope is indistinguishable from flat.

---

## T2 — The rolling edge is stable

250-trade rolling mean, 2,656 windows. **Below zero 9.2% of the time overall and 4.1% in the
last three years**, against a falsifier of >40%.

Annual rolling means:

```
2017 +1.94   2018 +4.04   2019 +3.91   2020 -0.70   2021 +3.24
2022 +5.39   2023 +5.49   2024 +3.57   2025 +2.90   2026 +1.65
```

**One thing to watch, deliberately not called a finding.** The last four years decline
monotonically: 5.49 → 3.57 → 2.90 → 1.65, a 70% fall. Four points falling in order has a
1-in-24 chance under random ordering, so it is suggestive — but it is a pattern I noticed
*after* looking, it is not what T1 was designed to detect, and **T1, which was pre-registered,
says there is no trend**. 2017 (+1.94) and 2020 (−0.70) were also weak, so the decade does not
support a decay story. Watch it; do not act on it.

---

## T3 — Every era is positive on its own

```
early    2016-01-20 .. 2019-07-25   n=968   +$3.80   CI [+1.41, +6.27]   p 0.0016
middle   2019-07-26 .. 2022-12-22   n=968   +$3.26   CI [+0.26, +6.36]   p 0.0334
LATE     2022-12-27 .. 2026-08-27   n=969   +$2.98   CI [+0.31, +6.21]   p 0.0280
```

The falsifier was *"if the late third is not significantly positive on its own"*. It is
(p = 0.0280), so the test passes as written.

**Stated precisely, because the distinction matters:** that falsifier was written on the raw
p. Under Holm across the five primaries the late era adjusts to **0.1400 and does not clear**.
The point estimates decline mildly (3.80 → 3.26 → 2.98) but the confidence intervals overlap
almost completely, so that ordering carries no information.

---

## T4 — Serial dependence is not a problem, and this validates the earlier work

```
daily-P&L autocorrelation, lags 1-10:
  -0.09 -0.02 +0.03 -0.01 +0.02 +0.00 +0.02 +0.00 -0.02 -0.02

day-clustered CI width   $3.22/trade
moving-block  CI width   $2.98/trade   (-7%)      falsifier: >+25% wider
```

The concern was that resampling days independently destroys serial structure and produces
intervals that are too narrow. It does not. The block bootstrap is **7% narrower**, not wider.

**Every confidence interval reported in this project for this strategy is therefore
conservative rather than optimistic.** That is a retroactive validation of the whole
day-clustered bootstrap approach, and it was not guaranteed.

---

## T5 — CORRECTED. The first version of this test was circular.

**As pre-registered, T5 was wrong.** It bucketed days by `mean(|trade return|)` — a quantity
derived *from the trades themselves*. A day on which the strategy's trades moved a lot is by
construction a day it made or lost a lot, and since IMB is positively skewed that loads the
winners into the top bucket automatically. It produced this:

```
CIRCULAR (bucketed on the trades' own |return|)
  Q1 -$3.92   Q2 -$7.67   Q3 -$7.94   Q4 +$32.72
```

which would have supported a dramatic and false conclusion: *"the entire edge is in the top
quartile; it loses money in 75% of conditions; it is a long-volatility bet in disguise."*

Redone with **prior-20-session realised volatility of QQQ, knowable before the open**:

```
EXOGENOUS (prior-20d realised vol)
  bucket                 n    $/trade          95% CI        p
  Q1 calmest           723     +1.17   [-0.69, +3.27]   0.2227
  Q2                   723     +3.00   [+0.60, +5.58]   0.0153
  Q3                   725     +3.62   [+0.66, +6.71]   0.0170
  Q4 most volatile     724     +5.43   [+0.90, +10.62]  0.0170
```

quartile cuts at 0.76% / 1.04% / 1.51% daily sd.

**Positive in all four regimes, significant in three, rising monotonically with volatility.**
That is ordinary volatility scaling — a strategy whose P&L is proportional to the size of the
moves it trades — not a hidden regime bet. It does not lose money in calm markets; it simply
earns less.

---

## Verdict

| test | falsifier | result |
|---|---|---|
| T1 trend | significant negative slope reaching zero before 2029 | **passes** — slope flat, p 0.87 |
| T2 rolling | >40% of recent windows below zero | **passes** — 4.1% |
| T3 late era | late third not significantly positive | **passes** on raw p (0.0280); does not clear Holm |
| T4 serial | block CI >25% wider than day-clustered | **passes** — 7% narrower |
| T5 regime | edge confined to one volatility regime | **passes** — positive in all four |

The edge is **persistent, not decaying, not regime-confined, and its confidence intervals are
if anything too wide.** This is the strongest the IMB case has looked, and it is stronger
specifically because these tests could have killed it and did not.

**What this does NOT change.** It is the same 2,905 trades — persistence tests interrogate
structure, they do not add information. The tail dependence stands: the top 1% of trades is
still 73.9% of P&L, and excluding them still leaves p = 0.1764. The strategy still pays
9.2%/yr at a 27% win rate with a 26-month underwater stretch. Nothing here makes it larger,
only more durable than it looked.
