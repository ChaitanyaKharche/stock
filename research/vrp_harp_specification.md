# #2 — the diurnal test, and the benchmark it accidentally replaced

*Run 2026-09-11. `python -m trade_analysis.hpc.harp_baseline --data data/vrp`.
Validation block (2023). 2024 held-out never touched.*

## What was asked, and what came back

The question was whether HAR lacks an intraday time-of-day term that implied variance
carries — the periodicity hypothesis from Dumitru/Hizmeri/Izzeldin (*JBF* 170, 2025). It was
pre-committed as **confirmatory**: a better HAR would strengthen the variance arm's
negative, not reopen it.

The diurnal answer is small. The side effect is not.

| model | QLIKE | vs HAR-RV | vs IV alone |
|---|---|---|---|
| **HAR-IV-TOD** | **0.330489** | **−23.26%** | −19.62% |
| HAR-IV | 0.335902 | −22.00% | −18.30% |
| implied_variance | 0.411136 | −4.53% | — |
| HAR-TOD-bins | 0.423654 | −1.62% | +3.04% |
| HAR-TOD | 0.424892 | −1.34% | +3.35% |
| HAR-RV | 0.430642 | — | +4.75% |
| HAR-RV-J | 0.430788 | +0.03% | +4.78% |

**Diurnal terms buy 1.3–1.6%.** Real but minor, and the 59-parameter binned spec beats the
6-parameter Fourier spec by only 0.3%, so it is a genuine mild clock rather than train-block
overfitting. **The 5-minute RV features already absorb most of the intraday U-shape.** The
"maybe HAR was just misspecified on time-of-day" objection is closed by measurement, and the
estimator-side HARP rebuild — filtering returns and recomputing RV from the 1-minute
archive — is not worth its cost on this evidence.

## The finding that matters: HAR-IV

**Combining HAR features with implied variance beats either alone by ~22%.** This is the
benchmark the literature actually uses and the pre-registration does not —
Kambouroudis/McMillan/Tsakou (*JFM* 2021) find only HAR specifications that include implied
volatility enter the Model Confidence Set. Comparing IV-alone to HAR-alone, as §4 does, is
not the standard comparison and it understates both.

It survives every robustness check, and does so far more convincingly than the IV-alone
result ever did:

| check | HAR-IV vs HAR-RV | (IV alone vs HAR-RV, for contrast) |
|---|---|---|
| DM t / p, session-clustered | **−2.95 / 0.0035** | −0.55 / 0.5832 |
| sessions won | **198 / 250 = 79.2%** | 99 / 250 = 39.6% |
| sign test | **p ≈ 0.0000** | p = 0.0012 *against* |
| net / gross | **−0.888** | −0.104 |
| bootstrap P(t < −1.96) | **0.9995** | 0.0107 |
| mean 95% CI | **[−0.167, −0.044]** | [−0.100, +0.036] |

Note the leave-k-out curve runs the *opposite* way to a fragile result: dropping the most
favourable sessions makes t **more** negative (−2.95 → −4.22 → −8.69 at k=10), because those
few extreme sessions inflate the variance more than the mean. That is the signature of a
broad effect with a couple of noisy outliers, not an effect carried by a tail. **It is the
first result in this project that gets stronger when you remove its best days.**

**Lookahead audit PASSES.** Features shifted +1 bar score better: HAR-IV 0.3359 → 0.3327,
HAR-RV 0.4306 → 0.4224. One bar of real future information is worth ~1–2% here.

## The mechanism, and why a 22% margin is not automatically suspicious

One bar of genuine lookahead buys ~1.9%; HAR-IV's margin is 22%. The project's own heuristic
says to distrust a margin much larger than cheating with the future would buy. That heuristic
does not apply here, and it is worth saying why.

`iv_var_atm` is annualised implied variance, and `_rv` is annualised realised variance
(validated on build: 8.8% annualised vs 8.7% from an independent source), so the two are on
the same scale — this is not a units artifact. What raw `implied_variance` gets wrong is
**bias, not scale**: implied exceeds realised at 80.1% of origins, which is the variance risk
premium, and QLIKE punishes a systematically high forecast. HAR-IV fits a coefficient and an
intercept on log IV, which estimates and removes that premium.

So **HAR-IV's gain is substantially the premium being measured and subtracted.** That is
information from a different source than price history, which is exactly why it can exceed
what a bar of past-price lookahead buys. It is also the reason the result is interesting
rather than merely accurate.

## Consequences

**1. The bar for the neural arm went up, not down.** §6 requires beating the best HAR-family
spec at Holm-corrected p < 0.05. That spec is now **HAR-IV-TOD at 0.330489**, not HAR-RV at
0.430642 — a 23% harder target, reached by a linear model with two extra regressors. A
sub-1M-parameter MLP on HAR-type features has less room than before, not more. The decision
not to run the GPU sweep is further over-determined.

**2. The variance arm stays closed, and this does not reopen it.** The arm is closed on the
cost model — spread $0.0376 against a $0.0205 gross edge, 1.84×, replicated four independent
ways. No forecast improvement touches that. Pollok (arXiv:2506.07928) is explicit that QLIKE
and P&L are only loosely coupled in both directions; a 22% QLIKE gain is not 22% of anything
tradeable.

**3. But it hands #3 a better instrument than the published one.** The conditional 0DTE
question needs something to condition on, and Pollok's economic test sorts on a
**forecast-RV-minus-IV spread**. Almeida/Freire/Hizmeri's SSD-violation rule conditions on a
crude device — a historical return histogram rescaled by current ATM IV — and still nets
Sharpe 0.101–0.159 against −0.010 to −0.042 for unconditional carry. We now have a measured,
audited, robustly-better RV forecast to build that spread from. **That is the one live thread
this result feeds, and it is a trading test, not a forecasting one.**

## Multiplicity

Six non-benchmark models were tested, so Holm was applied across all six and is reported
beside the raw p. Every model loses to HAR-IV-TOD at Holm p < 0.05 except nothing — HAR-IV
itself loses at Holm p = 0.0204. No raw-p-survives / Holm-p-fails cases arose, so nothing is
being reported on an uncorrected p.

## Isolation

`harp_baseline.py` imports the split, cleaning, target, QLIKE, DM test and feature lists from
`har_baseline` and edits none of it, per the project rule on experimental code. It adds its
own log-space fit only because the diurnal and IV terms need linear (unlogged) columns that
`fit_har` cannot express — and it applies the same `exp(s²/2)` lognormal correction, whose
omission is what produced the 24.1% phantom this file was originally written to explain.

`dm_concentration.py` gained `--models harp` and `--benchmark` so any pair in the widened set
can be audited with the same instrument, rather than a second copy of the audit drifting from
the first.
