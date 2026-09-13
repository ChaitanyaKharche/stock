# Results — MOMO_CHASE mis-specification test (momo_v2.py)

Executed **2026-09-13** per the pre-registration in the `momo_v2.py` docstring (written
2026-09-04). Variants were frozen before running; all three are reported.

**The pre-registration sat unexecuted for nine days.** It was committed in the bulk
`c0c869d` and never run. That is recorded here because an unrun pre-registration is
indistinguishable from a suppressed result in the file tree, and the discipline only works
if the gap between writing and running is visible.

## Sample

QQQ 2,657 sessions (2016-01-04 → 2026-08-27) + SPY 1,158 (2022-01-03 → 2026-08-27).
Sessions with fewer than 300 one-minute bars are dropped by `sessions()`, which removes
early closes. $10,000 notional, signal from a closed 5-minute bar, fill at the next
1-minute bar's open, exit 25 minutes later at the open, both legs lagged identically.

## Result

| | variant | n | $/trade | 95% CI | p raw | p Holm |
|---|---|---|---|---|---|---|
| V0 | as frozen (sanity check) | 8,598 | **+0.03** | [−0.47, +0.53] | 0.8982 | 1.0000 |
| V1 | + px_vs_ema9 ≥ 0.71 ATR | 8,104 | −0.03 | [−0.54, +0.48] | 0.9126 | 1.0000 |
| V2 | 1-min DMI instead of 5-min | 13,255 | −0.09 | [−0.47, +0.29] | 0.6364 | 1.0000 |

Day-clustered bootstrap, 10,000 resamples. Holm across m=3.

**Falsifier: a variant must beat +$1.00/trade AND clear Holm at 0.05. NONE DID.**
Per the pre-registered decision rule, **MOMO_CHASE is closed** as a question about
mis-specification, and no further variant is tried *under this pre-registration*.

## Two things the table does not say on its face

**1. V0's sanity check passed, but barely informatively.** The docstring predicted V0 would
reproduce "~+$0.05" and it came in at +$0.03. The CI is [−0.47, +0.53], which contains
+0.05, 0.00 and −0.30 alike. The reproduction is consistent, but a ±$0.50 interval around a
$0.03 point estimate cannot confirm much, and reading it as a tight match would be
overclaiming.

**2. Every number above is GROSS, and the costs are larger than the entire effect.** At
$10,000 notional on a ~$400 QQQ that is ~25 shares. A one-cent-wide round trip crossing
half the spread each way is ~$0.25 per trade. So V0's +$0.03 gross is approximately
**−$0.22 net**, and V2's −$0.09 is approximately **−$0.34**. This project has made exactly
this error before in the other direction — a setup measured at gross +$0.0205 and net
−$0.0171 — which is why the gross figure is never the finding.

The direction of the V1 and V2 results is also worth stating plainly: correcting the
specification toward fidelity with the trader's actual behaviour made it **worse**, not
better, in both cases. That matches the prior stated in the pre-registration itself (the
journal study's Q3 zone ran significantly negative at −0.149 bp) and it is now observed a
second time on a different sample and a different exit rule.

## What this closes and what it does not

**Closed:** the hypothesis that MOMO_CHASE underperforms because it uses the wrong DMI
timeframe or omits `px_vs_ema9`. It does not, and fixing those does not help.

**Not closed, and explicitly out of scope here:** whether *any* parameterization of the
momo-chase FORM has an edge as a standalone strategy. This test moved three hand-picked
points in an 11-dimensional space, and those three were chosen after looking at the journal
study's AUC table — a search whose size is unknown and therefore uncorrectable.

That is a weaker epistemic position than an exhaustive grid, not a stronger one. A full
enumeration has a *known* universe, which is the precondition for a valid data-snooping
correction (White's Reality Check / Hansen SPA). See
`research/momo_sweep_preregistration.md`, which forms a **new family with its own
correction**, as required by §2 of `setup_sweep_preregistration.md`.

**This result does not license that sweep to be read as a continuation of this one.** It is
a separate family and its own falsifier applies.
