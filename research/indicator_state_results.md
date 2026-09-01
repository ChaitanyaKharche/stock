# Results — Do the trader's own declared indicators separate his outcomes?

Executed 2026-08-20 per `indicator_state_preregistration.md`. No definition changed after a
result was seen.

## 1. Sample and funnel

**443 round trips → 418 with features.** Exclusions: 20 entries too early in the session for
a 15-minute trailing window, 5 with no completed 5-minute bar before entry. **None
outcome-related.** 151 activity dates, 13 underlyings, single names retained.

Minute bars for all 13 symbols came from cache — zero live calls, zero failures.

## 2. His indicators do fire as he describes

Measured at the **last fully completed 5-minute bar before each entry second**:

| | median at entry | gate rate |
|---|---|---|
| MACD histogram (9/17/9, OHLC4), direction-aligned | **+0.0696** | **69.4%** |
| +DI − −DI (Wilder 14), direction-aligned | **+10.17** | **78.5%** |
| ADX (Wilder 14) | **24.3** | 69.6% above 20 |
| volume ÷ EMA20(volume) | **0.986** | **48.3%** |
| 10m + 15m MACD & DMI agreement | — | 40.7% |
| **all five** | — | **14.1%** |

So the description is accurate: he does enter when MACD and DMI are aligned with his
direction, roughly 70–79% of the time. That is consistent with the established chase result
(+0.20σ trailing move) — momentum alignment and momentum chasing are the same thing measured
two ways.

**One exception worth naming: volume is a coin flip.** He cites volume against its 20-EMA as
a confirmation, but the median entry sits at **0.986× the EMA** and only 48.3% clear it. He
is not, in practice, requiring volume.

## 3. Primary endpoint — null

Ridge on the six continuous features, GroupKFold(5) by date, 1,000 day-level permutations,
return space:

| λ | OOS R² | permutation null mean | p |
|---|---|---|---|
| 0.1 | −0.0626 | −0.0410 | 0.854 |
| **1.0 (primary)** | **−0.0623** | −0.0415 | **0.835** |
| 10.0 | −0.0598 | −0.0392 | 0.846 |

Out-of-sample R² is **negative and indistinguishable from the permutation null** at every λ.

**Secondary, OOS AUC on win/loss: 0.3919** against a null of 0.5011, p = 1.000 — the model
ranks winners *below* losers out of sample. Dollar-space repeat: OOS R² −0.0282, p = 0.298.

## 4. The checkbox test — the direct form of the request

| | n | mean return | median | win rate | 95% CI |
|---|---|---|---|---|---|
| **all five gates met** | **59** | **−8.26%** | −22.18% | 40.7% | [−21.95%, +6.76%] |
| not all met | 359 | **+2.69%** | −8.00% | 46.0% | [−3.42%, +9.43%] |
| **difference** | | **−10.95%** | | | **[−26.36%, +4.85%], p = 0.163** |

Individual gates, Holm across five:

| gate | n | effect on return | p raw | p Holm |
|---|---|---|---|---|
| gate_mtf (10m/15m agreement) | 170 | **−8.16%** | 0.141 | 0.707 |
| gate_macd | 290 | −5.22% | 0.369 | 1.000 |
| gate_vol | 202 | −3.62% | 0.533 | 1.000 |
| gate_dmi | 328 | −3.22% | 0.621 | 1.000 |
| gate_adx | 291 | +4.79% | 0.511 | 1.000 |

Four of five point negative; nothing is significant.

Dose response by number of gates met: **−9.49% (0), −6.42% (1), +6.78% (2), +7.44% (3),
−3.42% (4), −8.26% (5)**. Non-monotone, inverted-U. **This is the shape noise makes.** The
+7.44% at three gates must not be read as a rule — selecting it would be exactly the
profitable-subgroup search the pre-registration prohibits.

## 5. Verdict

Pre-registered rule: *OOS R² ≤ 0.05 with p > 0.05 **and** AUC ≤ 0.55 ⇒ no detectable
information in his indicator stack; stop; no 2018 extension.*

**Both conditions met decisively** (R² = −0.062 at p = 0.835; AUC = 0.392). **Verdict: no
detectable information. The historical extension is not authorised by this result.**

## 6. What this does and does not establish

**Does:** his own declared indicator stack, evaluated at his own real entry timestamps, with
his own stated parameters, carries no out-of-sample information about the return of the
option he then bought. This was the last live version of "the indicators work, the backtest
just tested them at the wrong time." It does not survive.

**Does not:** establish that checking all five boxes is *harmful*. The −10.95% has a CI
spanning zero, and the `all_gates` cell is n=59 with an MDE near 14 percentage points. The
honest statement is *no evidence it helps*, with a point estimate that happens to be
negative.

**Power, as pre-stated:** return sd ≈ 52%, effective n ≈ 334 after DEFF 1.25, so this
excludes mean shifts above roughly **8 percentage points** and nothing smaller.

**Sixth consecutive null** on this record: C1 (strike distance), F1 (size — scale only), F2
(within-day ordinal), the duration-matched entry percentile, the 19-feature ceiling test, and
now the declared indicator stack. Every one measured with a pre-registered falsifier on the
full sample.
