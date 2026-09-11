# Pre-registration — is the 0DTE variance premium harvestable CONDITIONALLY?

*Written 2026-09-11, **before any result was computed**. Every threshold, window and gate
below is fixed by this document. §9 of `vrp_preregistration.md` applies: a gate may not be
moved after seeing an outcome, and a gate that is moved voids the arm.*

---

## 1. The question, and why it is not a tenth attempt at the same thing

Nine nulls stand. Eight tested **direction**. The ninth tested the **unconditional** short
0DTE straddle and is decisive:

| | mean/trade | win rate | session-clustered t |
|---|---|---|---|
| gross (mid→mid) | +$0.0205 | 68.6% | **+7.52** |
| net (bid→ask) | **−$0.0171** | 61.0% | **−5.38** |

Spread $0.0376 = 1.84× the gross edge. Replicated four independent ways: our cost model,
Almeida/Freire/Hizmeri (CBOE 1-min SPX/SPXW 2012–2025, net Sharpe negative at all nine
intraday entry times), Bevilacqua & Hizmeri (gross Sharpe 0.64 → negative net), and the
corrected Vilkov package ("no structure retains a materially positive net Sharpe ratio").

**This tests something the nine did not: whether the premium is harvestable on a SUBSET of
origins selected in advance by a signal.** The unconditional trade pays the spread on every
origin including the ones where there is no premium to collect. That is a different claim
from "the premium does not exist," and the literature contains a positive on it — Almeida
et al.'s SSD-violation rule earns net Sharpe **0.101–0.159** across the same nine entry
times where unconditional carry earns **−0.010 to −0.042**, and costs barely dent it.

The published conditioning device is crude: a historical return histogram rescaled by
current ATM IV. **That is the bar, and it is a low one.**

---

## 2. Two arms, both pre-registered here

Order of implementation is for tractability and carries no inferential weight. Both gates
are fixed now, so reading one first cannot license moving the other.

### Arm A (PRIMARY) — replicate the published rule

Almeida/Freire/Hizmeri's SSD-violation strategy on the **Vilkov SPXW 0DTE panel**
(`research_data/0dte-strategies/data/data_opt.parquet`; 1,259,071 rows, 1,397 sessions,
2016-09-02 → 2024-05-01, 30-minute bars, 10:00–15:30 ET, real bid/ask).

Primary because it has a published number to check against. If our implementation cannot
reproduce a positive where they report one, that is a fact about our implementation, and we
need to know it before trusting any variant of ours.

**Panel traps that must be handled (from prior direct inspection, not the repo's README):**
`mnes_rel` is strike/spot **at that bar**, so it is constant-moneyness and following one
contract requires recomputing K/S_t; and `mid`/`bas`/`payoff` are each divided by **their
own bar's** spot, so values from different bars are in different units and must be rescaled
before being compared. The upstream package's own 2026-08 correction was caused by exactly
this trap — it charged the half-spread at 1/100 of true size.

### Arm B (SECONDARY, single pre-declared variant) — condition on our own forecast

Condition the **already-measured** unconditional straddle on the HAR-IV forecast.

This is not a new backtest. `data/cost_model_trades.parquet` holds 37,517 origins with
`entry_bid` / `exit_ask` / `net` already measured against the raw archive. Arm B adds a
selection rule and re-reads the same P&L, so the comparison to the −$0.0171 null is exact
and nothing is re-priced.

**Why this signal:** `research/vrp_harp_specification.md` establishes that HAR-IV forecasts
realised variance 22% better than HAR-RV and 18% better than IV alone, robustly (79.2% of
sessions, bootstrap P(t<−1.96) = 0.9995). Pollok (arXiv:2506.07928) shows the economic test
that matters is a sort on the **forecast-RV-minus-IV spread**. We now have an audited
forecast to build that spread from, and it is a better instrument than the published
histogram.

---

## 3. Arm B, specified exactly

**Signal.** At origin *t*:

```
expected_VRP(t) = iv_var_atm(t) − HAR_IV_forecast( rv_fwd_30 | t )
```

Positive means implied variance exceeds the forecast of realised variance — the premium is
expected to be collectable at this origin. `HAR_IV_forecast` is the `HAR-IV` specification
of `harp_baseline.py`, unchanged, with the `exp(s²/2)` lognormal correction.

**Rule.** Sell the ATM straddle only when `expected_VRP(t)` is in the **top tercile**.
Otherwise no trade.

**The threshold is fixed on the TRAIN block.** The tercile cut is the 66.67th percentile of
`expected_VRP` computed over **train origins only (dates < 2023-01-01)** and then applied
unchanged to validation. Computing the cut on the evaluation sample would be a look-ahead,
and it is the specific look-ahead most likely to manufacture a result here.

**Evaluation window.** Validation block only — **dates in 2023**, matching the frozen split,
because HAR-IV is fitted on train and is only out-of-sample there. The 2020–2022 origins in
`cost_model_trades.parquet` are in-sample for the forecast and are **excluded**.
**2024 remains sealed.**

**Horizon.** 30 minutes, unhedged short straddle — identical to the measured null, because
the whole point is comparability. No delta hedge is claimed.

**Costs.** Unchanged: sell at `entry_bid`, buy back at `exit_ask`. The `net` column already
in the parquet. No mid fills anywhere.

**Inference.** Mean net P&L per trade, **session-clustered** t (one mean per session, then a
one-sample t across sessions), exactly as `dm_test` and the cost model do. Intraday origins
are ~0.9 autocorrelated; a naive per-trade t inflates n ~300×.

### The gate — all four must hold

1. **net mean per trade > 0** on the conditioned subset
2. **session-clustered t > 2.0**
3. conditioned net **significantly better than unconditional net** on the *same* 2023
   origins, by a paired session-clustered test at p < 0.05
4. the conditioned subset retains **≥ 50 sessions** and **≥ 1,000 trades** — a rule that
   fires on a handful of origins is unmeasurable regardless of its mean

### Declared in advance, because these are the ways it will look good and be wrong

- **Concentration.** `dm_concentration`-style audit is mandatory on the result: sign test,
  net/gross, leave-k-out, session bootstrap. Every apparent edge in this project has lived
  in a tail — IMB 74% of P&L in the top 1%, the 0DTE straddle's worst 1% = 94% of net loss.
  **If ≥ 50% of the conditioned P&L sits in the top 5% of sessions, the result is reported
  as fragile regardless of its t.**
- **Short-gamma masking.** 61% of unconditional trades were winners while the mean was
  negative. A positive median with a negative mean is the signature. Both are reported.
- **Selection by survivorship.** The tercile rule must not preferentially select low-spread
  origins, which would smuggle in a cost advantage rather than a premium. `entry_spread_bp`
  is compared across selected and rejected origins and reported.

### Multiplicity

Arm B is **one** signal, **one** threshold rule, **one** horizon: m = 1. No grid over
terciles, horizons or percentile cuts is authorised by this document. Adding any would
require a new pre-registration and Holm correction across the enlarged family.

---

## 4. What each outcome means — committed now

| outcome | reading |
|---|---|
| Arm A reproduces a positive **and** Arm B passes its gate | The conditional story is real on two instruments. This is the first live positive in the programme and warrants a forward test, not a claim. |
| Arm A reproduces, Arm B fails | Our forecast is not a useful conditioner even though it forecasts better. That is the Pollok warning made concrete — QLIKE and P&L are loosely coupled — and it is a publishable-quality negative about our own instrument. |
| Arm A fails to reproduce | Stop and debug the implementation against their reported Sharpe before reading Arm B at all. A failed replication is about our code until proven otherwise. |
| both fail | **The 0DTE arm closes completely.** Unconditional was already dead four ways; conditional dead too means the premium is not harvestable by us at any selection, and the arm is finished rather than merely paused. |

**No outcome here reopens the forecasting arm.** §6 of `vrp_preregistration.md` is a
separate gate on a separate question, and its target is now HAR-IV-TOD at 0.330489.

**No outcome here authorises touching 2024.** That block is opened once, by §6, and this
document does not amend §6.

---

## 5. What is deliberately not attempted

No delta hedge, so this measures the retail expression rather than a clean variance swap —
the same limitation the unconditional null carries, kept identical for comparability.

No position sizing, no portfolio, no compounding. Overlapping 30-minute origins mean six
positions open at once; that is fine for per-trade economics and meaningless as a return
series.

No parameter search. The tercile cut, the horizon and the train-only threshold are the three
free choices and all three are fixed above.
