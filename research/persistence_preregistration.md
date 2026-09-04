# Pre-registration — Is the IMB edge persistent through time?

Written **2026-09-04**, before any of the tests below were computed. Amendments appended,
never substituted.

---

## Why this is not another strategy search

Everything measured so far pools 10.6 years into a single number: +$3.34/trade, Holm 0.0043.
The coarse time checks that exist — split-half and per-year — ask only *"is it positive in
both halves"* and *"how many years are green"*. Neither can detect a **decaying** edge, and a
decaying edge is the single most likely failure mode here for a specific reason:

**IMB's parameters are published** (Zarattini/Aziz/Barbon, SSRN 4824172, ~2024). The untuned,
published nature of those parameters is the main reason the result is credible. It is also
exactly why the edge might be arbitraged away after publication.

If the edge is decaying, the pooled estimate is not the forward estimate, and a forward test
sized against +$3.34 is sized against a number that no longer exists.

**No parameter will be tuned and no new strategy proposed as a result of this.** The only
possible outcomes are: the pooled estimate stands, or it must be replaced by a
recent-era estimate, or the whole thing is regime-dependent and unusable.

---

## Sample

`live_lab_data/sharewf_trades_QQQ.json`, `IntradayMomentumBoundary` only: **2,905 trades,
1,560 sessions, 2016-01-20 → 2026-08-27**, $10,000 notional, real NBBO both sides, next-bar
fills. Plus SPY 2022-2026 (1,316 trades) as a secondary panel.

Frozen now: no trade may be added, dropped or re-priced for the remainder of this study.

---

## The five tests, and what each would falsify

Seed 20260904. All bootstraps resample **dates**, not trades.

### T1 — Linear trend in the per-trade edge
OLS of trade P&L on days-since-start; slope bootstrapped by date.
**Falsifier: if the slope is significantly negative (day-clustered p < 0.05) AND the fitted
edge reaches zero before 2029, the pooled +$3.34 is retired and replaced by the recent-era
estimate.**

### T2 — Rolling 250-trade mean
Rolling window with a day-clustered CI on each window.
**Falsifier: if the rolling mean spends more than 40% of the last three years below zero, the
edge is not stable enough to size against.**

### T3 — Era split (thirds, equal trade counts)
Early / middle / late, each with its own CI.
**Falsifier: if the late third is not significantly positive on its own, the case rests
entirely on history that no longer applies.**

### T4 — Serial dependence
Autocorrelation of daily P&L at lags 1–10, plus a **moving-block bootstrap** compared against
the day-clustered bootstrap already in use.
**Falsifier: if the block-bootstrap CI is more than 25% wider than the day-clustered one, then
every confidence interval reported in this project for this strategy is too narrow and must be
restated.**

### T5 — Regime conditioning (diagnostic only, not a filter)
Edge conditioned on trailing realised volatility quartile.
**This cannot produce a trading rule.** It answers one question: is the edge concentrated in a
volatility regime that may simply not recur? If the top quartile carries everything, the
strategy is a long-volatility bet wearing a momentum costume.

---

## Multiplicity

Five primaries, **Holm m = 5**. The family size will not be reduced if a test fails to
execute.

## What cannot happen as a result of this study

- No parameter change. No new setup. No filter.
- No exclusion of any period from the pooled estimate *unless* T1 or T3 fires, in which case
  the recent era becomes the headline and the pooled number is reported as historical.
- Nothing here touches either live arm or either config hash.
