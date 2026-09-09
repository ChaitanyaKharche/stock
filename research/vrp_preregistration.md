# Pre-registration — Can intraday realised variance be forecast well enough to price the 0DTE variance risk premium?

**Status: PRE-REGISTERED, NOT YET RUN.** Written 2026-09-08. Everything below is frozen
before the held-out block is touched.

## 0. Why this is not a ninth attempt at the same thing

Eight prior nulls closed **direction** prediction in this project, and
`research_programme_verdict.md` says so. This is deliberately not that question.

The distinction is not rhetorical, and it surfaced by accident on 2026-09-08 while fixing
the showcase app. `IntegratedMomentumEngine._calculate_momentum_score` turned out to be a
pure **magnitude**: every term is `abs()`-wrapped or non-negative by construction. It
answers *how much is happening*, never *which way* — and the code had been forcing that
magnitude to produce a directional CALL/PUT signal, which is why `_convert_signal_format`
hardcoded `return 'CALLS'` with the comment "momentum typically bullish for options".

That is a bug, and it was fixed. But it is also a hint worth taking seriously: **the part
of this system that was measuring something real was measuring size, not sign.** No
experiment in the programme has tested the magnitude channel on its own.

For 0DTE options the magnitude channel is directly tradeable without any directional call.
A straddle or strangle held to expiry pays the difference between realised variance and the
variance implied by the price paid. The systematic tendency of implied to exceed realised
— the **variance risk premium (VRP)** — is the economic motivation for selling on expiry
days, and it is a variance question end to end.

**This is therefore a genuinely new hypothesis class, not a retry.** It may still null.

## 1. The data, and why the held-out block is irreplaceable

`Desktop/data/raw/option_quote_1m_0dte/SPY/` — **769 sessions**, minute-resolution 0DTE SPY
option quotes with **real bid and ask** (not just last trade), spanning 2020–2024:

    2020: 159   2021: 39   2022: 170   2023: 250   2024: 151

Underlying minute bars come from `stock_ohlc_1m/SPY/`, and `stock_quote_1m/SPY/` supplies
NBBO for the underlying.

**This dataset cannot be extended.** The ThetaData subscription lapsed to
`Options: FREE` on or before 2026-09-08, so no further option history can be pulled at any
price the project is willing to pay. Every 0DTE session that exists, exists now. Burning
the held-out block on an exploratory run is therefore **permanent**, not merely bad
practice, and that is the single most important constraint on this design.

**Split, frozen now:**

| block | sessions | use |
|---|---|---|
| TRAIN | 2020-01-01 → 2022-12-31 | ~368 | fit, tune, select — unlimited looks |
| VALIDATION | 2023-01-01 → 2023-12-31 | ~250 | model selection, early stopping, one number per candidate |
| HELD OUT | 2024-01-01 → 2024-12-31 | ~151 | **touched exactly once, at the end, for the winner only** |

Temporal, never random. A random split across minutes of the same session leaks almost
perfectly: adjacent minutes of one day share the same realised variance path.

## 2. The target, stated precisely enough to be wrong

For a forecast origin at minute `t` on session `d`:

    RV(d, t, h) = sum over the next h one-minute log returns of r^2, annualised

`h = 30` minutes is the primary horizon. `h = 60` and `h = to-close` are secondary and
reported, never promoted without their own correction.

The tradeable quantity is the signed premium:

    VRP(d, t, h) = IV_atm(d, t)^2 * (h / minutes_per_year) − RV(d, t, h)

`IV_atm` is backed out from the **mid** of the nearest-to-ATM 0DTE straddle at `t`, using
bid and ask that were quoted at or before `t`. Positive VRP means the option was expensive
relative to what actually happened.

**Forecast origins** are restricted to `10:00 ≤ t ≤ 15:00` ET. Before 10:00 the morning
realised-variance feature does not exist yet; after 15:00 the 0DTE gamma profile makes the
minute return distribution a different object, and pretending one model spans both is how
you manufacture an edge that is really a regime mixture.

## 3. Features — every one computable strictly before `t`

The 2026 literature converges on two features dominating intraday realised-variance
forecasting: **morning realised variance from the open**, and **the prior day's volatility
index**. Both are in the set. The rest are there to give the model a chance to beat HAR,
not because they are believed.

**Realised (from bars closed at or before `t`)**
1. RV from 09:30 to `t`, annualised
2. RV over the trailing 30 minutes
3. RV over the trailing 5 minutes
4. Bipower variation over the trailing 30 minutes (jump-robust)
5. Realised quarticity, trailing 30 minutes (for the HAR-Q term)
6. Prior session's full-day RV
7. Prior 5 sessions' mean RV
8. Prior 22 sessions' mean RV

**Diurnal / state**
9. Minutes since open (the U-shape is a fact, not a discovery)
10. Minutes to close
11. Day of week

**Option-implied (from quotes at or before `t`)**
12. ATM 0DTE implied variance
13. 25-delta risk reversal (skew)
14. 25-delta butterfly (smile curvature)
15. ATM straddle relative bid–ask spread, in bp of mid
16. Prior day's VIX close

**Prohibited outright**, because each has already destroyed a result in this project:
- any bar whose close timestamp is `> t`, including the bar *containing* `t`;
- the settlement print;
- any quote with `bid_size == 0 and ask_size == 0` (the placeholder rows visible at 09:30
  in the raw files are not tradeable prices);
- forward-filled option quotes across a gap longer than 5 minutes.

**A timestamp audit is a gate, not a checkbox**, and it reads the opposite way to the
obvious guess. `har_baseline.py --audit` shifts every feature one minute into the future,
DELIBERATELY injecting a lookahead. The audit scoring BETTER is therefore the PASS: it
proves the honest features did not already contain that minute. The audit scoring the SAME
is the failure, because it means the minute was already inside the honest run.

Measured before any modelling, on 623 sessions: honest QLIKE 0.187365, audit 0.184776 —
audit better by 1.4%, so the pipeline is clean. That 1.4% is retained as a calibration:
one minute of outright cheating buys 1.4% QLIKE here, so any model claiming a materially
larger margin over HAR is claiming more than the future itself is worth, and should be
disbelieved until re-audited.

## 4. The benchmark that must be beaten

**HAR-RV** (Corsi), and its jump-robust variant **HAR-RV-J**, fit on the same features 1–8
with OLS. This is the standard in the realised-volatility literature and it is hard to
beat. A neural model that does not beat HAR has produced nothing, however good its
absolute numbers look.

The comparison metric is **QLIKE** on the primary horizon, which is the standard loss for
variance forecasts because it is robust to the noise in the RV proxy in a way that MSE is
not. RMSE is reported alongside, never instead.

Statistical test: **Diebold–Mariano on the QLIKE loss differential, clustered by session
date.** Minute-level observations inside one day are not independent, and treating them as
independent would inflate the effective sample by ~390× and manufacture significance.

## 5. The model family, frozen

| # | model | why it is here |
|---|---|---|
| 1 | HAR-RV | benchmark |
| 2 | HAR-RV-J | benchmark, jump-robust |
| 3 | Chronos-2, zero-shot | 2026 baseline. Costs nothing to evaluate. |
| 4 | TimesFM-2.5, zero-shot | second zero-shot opinion |
| 5 | Chronos-2, fine-tuned | does adaptation to this series help |
| 6 | TFT, statics **populated** | the existing architecture, with its bug fixed |
| 7 | Gradient-boosted trees (LightGBM) | the honest tabular baseline everyone forgets |

**Family size 7. Holm–Bonferroni at α = 0.05 across the family.** Expected false positives
if nothing works: 7 × 0.05 = 0.35, so a single uncorrected "hit" is not evidence.

Model 6 exists because the checkpoint's `scaler_static` has every `scale_` equal to 1.0 —
the static branch was trained on a constant placeholder and therefore contributed nothing.
That is a real bug with a genuinely unknown answer. Statics to populate: day of week,
prior-day VIX bucket, prior-day RV quintile, and expiry-cycle position.

**A zero-shot foundation model is now the first thing to try, not the last.** In 2024 the
default was to train a TFT. In 2026 Chronos-2 / TimesFM-2.5 / Moirai-2 are strong enough
zero-shot that training before checking them wastes compute and risks reporting a trained
model that a free API call would have beaten.

## 6. Decision rule, committed now

**Primary:** model beats both HAR variants on QLIKE at `h=30`, on VALIDATION, with
DM p < 0.05 after Holm across the 7-model family.

If and only if that passes, the model is run **once** on HELD OUT. Success requires:
- QLIKE improvement over HAR in the same direction, and
- the sign of the improvement stable across all four quarters of 2024, and
- the improvement surviving a realistic cost model: the measured ATM straddle bid–ask
  spread at the forecast origin, not an assumed one.

**A model that only wins gross of spread is a null.** The spread on 0DTE ATM SPY is in the
data; there is no excuse for reporting a gross number.

**Failure is a result and gets written up with the same care as success.** Specifically, a
powered null on the VRP question — with the timestamp audit passing, the cost model
applied, and confidence intervals on the loss differential — is a stronger portfolio
artifact than another backtest with a good-looking equity curve. Most candidates can show a
model that worked. Almost none can show they caught their own lookahead and then measured
the thing properly anyway.

## 7. Power, stated before running

Effective sample is **sessions, not minutes**: ~368 train, ~250 validation, ~151 held out.
Forecast origins within a session are ~0.9 correlated at 30-minute separation, so the
design effect is large and the honest n is close to the session count.

At n=151 held-out sessions, a DM test has ~80% power to detect a standardised loss
differential of **0.23** and is essentially blind below **0.10**. If the true effect is
smaller than that, this design cannot see it, and **that is a limitation of the experiment,
not evidence of absence.** It is recorded here so the write-up cannot later claim more.

## 8. Scope note

This tests forecastability of intraday variance and whether any edge survives the 0DTE
spread. It does **not** test a full trading strategy: no position sizing, no gamma risk
management, no assignment handling, no path-dependent stop. Those are separate questions
and mixing them in would make a variance result unfalsifiable.

The live lab (`live_lab_data/`) is untouched by this. Its clock never resets and nothing
here may alter its frozen config hashes.

## 9. Prohibited

- Adding a feature after seeing a result and calling it the same experiment.
- Reporting the best of several horizons as if `h` had been fixed.
- Any use of the 2024 block before §6's validation gate passes.
- Reporting a gross-of-spread number as the headline.
- Presenting a zero-shot foundation model's win as evidence that *this project's* modelling
  added value. If Chronos-2 zero-shot wins, the finding is that the problem needed no
  bespoke model, and it must be stated that way.
