# Results — an early R1 break DOES predict continuation, and the exit throws it away

**Run 2026-09-23.** `trade_analysis/live_lab/after_break.py`, QQQ, 2,656 sessions,
2016-01-05 → 2026-08-27. Executes [`after_break_preregistration.md`](after_break_preregistration.md),
frozen before any number here was computed. Offline, from the stored six-line sweep.

**"First 15 minutes" is the first two 10-minute bars, 09:30 and 09:40** — 20 minutes. The
stored grid cannot express 15.

## 0. The three answers, in one line each

1. **It dips back first, then goes.** An early R1 break reaches R2 70.7% of the time vs
   59.3% for a late break. Real, and it survives a time-matched control.
2. **R2 does not stall it.** Of the sessions that reach R2, 93% close beyond it, and 46%
   carry on through R3.
3. **Round numbers: nothing.** Flat null at every multiple tested, in both price eras.

**And all of it is upside-only. The S side is dead.**

## 1. The R side — the first real conditional signal this project has found

| outcome, after R1 breaks | early (≤09:40) | late (≥09:50) | gap | p |
|---|---|---|---|---|
| reached R2 | **70.7%** (224/317) | 59.3% (302/509) | +11.3 pp | 0.001 |
| closed beyond R2 | **65.6%** (208/317) | 54.0% (275/509) | +11.6 pp | 0.001 |
| closed beyond R3 | **46.1%** (146/317) | 32.2% (164/509) | +13.8 pp | 0.000 |

### The control that makes it believable

An early break has six hours left to reach R2; a 15:30 break has twenty minutes. That
alone would produce the table above with no momentum anywhere. So both groups were given
the **same 60-minute window**, restricted to breaks by 14:50:

| R2 broke within 60 min of R1 | rate |
|---|---|
| early (≤09:40) | **49.2%** (156/317) |
| late (09:50–14:50) | 37.9% (181/478) |
| **gap** | **+11.3 pp, p=0.002** |

**Identical gap to the raw comparison.** The effect is not time exposure.

## 2. The exit looked like it was throwing away winners — IT IS NOT. Tested and falsified.

> **CORRECTION, 2026-09-23.** The measured figures in this section are right. **The
> conclusion drawn from them is wrong and was falsified by the experiment it motivated.**
>
> This section claimed the close-back-inside stop "is throwing away more than half of its
> winners" and called removing it the highest-value thing left to measure.
> [`early_r1_hold_results.md`](early_r1_hold_results.md) removed it on these same 317
> sessions: **worth +1.84 bp, CI [−13.02, +16.81], p=0.829.** Nothing. Two-thirds of a
> cent on a $740 stock.
>
> **Why the inference failed:** "R2 close-broke later that day" and "the trade would have
> been profitable at 16:00" are different statements. R2 can break at 11:20 and price can
> be back under R1 by the close. A held trade exits at the **session close**, not when R2
> breaks. I read a fact about the *path* as a fact about the *endpoint*.
>
> Everything below stands as measured. Do not read it as a case for changing the exit.

The same 314 early R1 breaks, traded by the six-line rule (enter next bar, exit on a close
back inside R1):

| | |
|---|---|
| exited "failed" — closed back inside R1 | **69.7%** (219/314) |
| median MFE (best unrealised move) | +25.6 bp |
| median end move | **−16.7 bp** |
| median giveback | **35.2 bp** |

Those two facts have to be reconciled: 65.6% of these sessions close beyond R2, yet the
median trade ends at −16.7 bp. They reconcile like this —

| of the 219 early R1 breaks that got stopped out by "close back inside" | |
|---|---|
| **still closed beyond R2 later the same day** | **119 (54.3%)** |
| of the 95 that did NOT get stopped out, reached R2 | 87 (91.6%) |

**The shakeout exit closes 219 trades, and 119 of them go on to do exactly what the setup
predicted.** The median failed trade books −26.1 bp after showing +13.6 bp.

This is the same shape as the settled cap finding — an exit rule destroying the thing the
entry found — but it is a **different exit**. The cap truncates winners; this one converts
winners into losers. `six_lines_results.md` measured the cap. Nothing has yet measured
removing the close-back-inside stop.

## 3. Round numbers: a clean null

Distance from R1 to the nearest multiple of M. Under no clustering the mean is M/4, so
**ratio 1.00 means no effect**.

| M | mean distance | expected M/4 | ratio |
|---|---|---|---|
| 1 | $0.254 | $0.250 | 1.02 |
| 5 | $1.206 | $1.250 | 0.96 |
| 10 | $2.481 | $2.500 | 0.99 |
| 25 | $6.587 | $6.250 | **1.05** |

Same in both price eras (QQQ ran $110 → $740), same on the S side. The pre-specified
near/far split at the median distance to a multiple of 25 also returns nothing on the R
side. **R1 and S1 have no affinity whatsoever for round numbers.** There is no reason to
weight a level because it sits near 700 or 725.

## 4. The S side is dead, and that asymmetry is a warning

| outcome, after S1 breaks | early | late | gap | p |
|---|---|---|---|---|
| reached S2 | 66.8% | 62.4% | +4.4 pp | 0.196 |
| closed beyond S2 | 57.7% | 54.6% | +3.1 pp | 0.382 |
| closed beyond S3 | 38.2% | 33.4% | +4.9 pp | 0.149 |
| **time-matched, 60 min** | 48.3% | 44.1% | +4.2 pp | **0.242** |

Nothing. **The finding in §1 is a long-only effect, measured over a decade in which QQQ
roughly tripled.** The early-vs-late comparison is internal to the R side so drift cannot
directly manufacture the gap, but a decade-long uptrend is exactly the environment where
upside continuation is easiest, and this has not been tested in a falling market. Treat
the effect as conditional on regime until a bear sample says otherwise.

## 5. A DEFECT IN ONE OF MY OWN METRICS, recorded rather than quietly dropped

The pre-registration included `full_reversal` = "the session traded through the opposite
side". It reported early breaks reversing *less* (R side 59.6% vs 82.7%, p<0.001), which
looked like a fourth signal.

**It is not usable.** It is computed from the whole session's extreme, which includes the
hours *before* R1 broke. A late R1 break happens on a day that already swung around, so
the metric is measuring "the day was wide", not "it reversed after the break". The stored
sweep has no post-break extreme, so it cannot be repaired here. The row is printed with a
`[DEFECTIVE]` tag and its early-vs-late gap must not be read.

## 6. What is NOT concluded

- **No P&L. No options. No cap.** Nothing here is a strategy, and the ~5 bp an ATM 0DTE
  needs for spread and theta is not paid anywhere in this document.
- **Nothing about SPY.** `six_lines_SPY.json` does not exist; that leg never finished.
- **Nothing about what happens inside a 10-minute bar.**
- §2's overlap figure (54.3%) is **exploratory** — it was computed after seeing §1, and it
  was not in the pre-registration.

## 7. The experiment this pointed at — RUN, AND IT FAILED

This section originally proposed re-running the six-line trade on early R1 breaks with
the close-back-inside exit removed, and called it the highest-value thing left to
measure.

**It was pre-registered, run the same day, and falsified on all three of its own
criteria.** See [`early_r1_hold_results.md`](early_r1_hold_results.md):

| gate | result |
|---|---|
| removing the stop is worth > 5 bp | **FAIL** — +1.84 bp, CI [−13.02, +16.81] |
| the setup beats the drift benchmark | **FAIL** — −2.56 bp, wrong sign |
| the sign holds in both halves | **FAIL** — +4.87 then −9.97 |

The benchmark arm — buying at 09:50 on days the signal did **not** fire — returned
+4.25 bp, more than either treatment arm. §1 of this document still stands: early breaks
genuinely reach R2 more often. It simply does not reach the closing price.

**The six-line family is now closed on all four of its parts: entry timing, level set,
profit cap, and stop.**
