# Phase 0 — is the IV-vs-HAR differential robust?

*Run 2026-09-11. `python -m trade_analysis.hpc.dm_concentration --data data/vrp`.
Validation block (2023) only. The 2024 held-out block was not touched.*

## Why this ran

A GPU sweep was queued on the strength of "the option market beats HAR." This was meant to
be the cheap gate in front of it: **count how many sessions carry the reported t = −2.30**,
because a 24% QLIKE reduction that only reaches t = −2.30 over ~250 clustered sessions
implies a mean carried by a handful of days.

The gate fired immediately, for a reason that had nothing to do with concentration.

## 1. The premise was already superseded

There is no t = −2.30. The figure came from a HAR fitted **without the lognormal
retransformation correction** — `exp(X @ beta)` is the conditional median, and the missing
`exp(s²/2)` factor (1.423 here, s² = 0.7056) cost HAR 20% of its QLIKE.

| | reported | actual (corrected fit) |
|---|---|---|
| implied_variance QLIKE | 0.411136 | 0.411136 |
| best HAR QLIKE | 0.541358 (HAR-RV-J) | **0.430642** (HAR-RV) |
| gap | 24.1% | **4.5%** |
| DM t | −2.30 | **−0.55** |
| DM p | 0.0222 | **0.5832** |

**`research/vrp_baseline_results.md` §2 and `har_baseline.predict_har`'s docstring have
both recorded this since 2026-09-09.** The repo was right. The stale figure survived in an
agent memory file and was used to brief a literature review, which spent its effort
explaining a 24.1% anomaly that does not exist. At 4.5% the result is unremarkable — it
sits inside the published single-digit range for IV augmentation and needs no explaining.

Frame note, also corrected: **769 sessions on disk, 747 usable.** The split
(346 / 250 / 151) sums to 747, not 769, so the "0 dropped" I had recorded was contradicted
by its own numbers. **The 22 are benign and `clean()` says so:** they are the first 22
sessions in the archive, lost because `rv_prev_22` needs 22 prior sessions that do not yet
exist. No rows are dropped inside a surviving session and all 22 sit in TRAIN, so
validation and heldout are untouched. A lookback warmup, not a defect -- my first framing
of this omitted that and read as more alarming than the facts warrant.

## 2. What the differential actually contains

Running the audit anyway produced a **stronger negative than "not significant."**

| check | result |
|---|---|
| sessions where IV wins | **99 / 250 = 39.6%** |
| sign test | **p = 0.0012, significant in HAR's favour** |
| net differential | −4.89 |
| gross flow | 46.82 (pro-IV −25.86 / pro-HAR +20.97) |
| **net / gross** | **−0.104** |
| 2 most pro-IV sessions | −9.66, ≈ **2× the entire net** |
| worst single session | **−7.83** |
| drop 4 sessions (1.6%) | **HAR significantly better, t = +2.36** |
| bootstrap P(t < −1.96) | **0.0107** |

The direction is not merely weakened — **on the typical session it is reversed, and that
reversal is significant.** HAR beats IV on 60.4% of days. IV's tiny net edge is a residual
of ~90% cancellation between two large opposing tails: it wins rarely and enormously on the
few days HAR badly underreacts, and loses on the routine majority.

That is the same pathology as every other apparent edge in this project — IMB with 74% of
P&L in the top 1%, the 0DTE straddle with 94% of net loss in the worst 1%. **Here it
appears in both directions at once, which is why the mean is a wash.**

## 3. Consequence

§6 of the pre-registration requires beating the best HAR on QLIKE at Holm-corrected
p < 0.05 across 7 models before the 2024 block may be opened.

**The bar is HAR-RV at 0.430642, and the option market's own forecast does not clear it.**
A sub-1M-parameter MLP on HAR-type features is not going to. There is no route by which the
queued sweep produces a positive, and the earlier decision not to run the GPU sweep — taken
on cost-model grounds — is now over-determined.

**Nothing was spent on the cluster to establish this.** That is what the gate was for.

## 4. What this does and does not settle

It does **not** test the periodicity hypothesis. HARP (periodicity-filtered HAR
predictors, Dumitru/Hizmeri/Izzeldin, *JBF* 170, 2025) is still worth building, but its
role has inverted: with the corrected fit there is no IV advantage to explain away, so HARP
would make **HAR better** and push it further ahead. That strengthens the negative rather
than deciding anything. It is no longer urgent.

The genuinely open question the literature review surfaced is unaffected by all of this:
whether the 0DTE variance premium is harvestable **conditionally** rather than as carry
(Almeida et al.'s SSD-violation rule nets Sharpe 0.101–0.159 against −0.010 to −0.042
unconditional). That needs a different instrument, not a better RV forecast.

## 5. Method

`trade_analysis/hpc/dm_concentration.py` imports the split, cleaning, HAR fit, QLIKE and DM
test from `har_baseline` and re-implements none of them, so it cannot drift from the result
it audits. Reproduction is asserted before anything else prints: this run's session-mean
differential matched `dm_test`'s `mean_diff` to 1e-12.

The session is the unit throughout, matching `dm_test` — intraday origins are ~0.9
autocorrelated and treating them as independent inflates n roughly 300×.

Two defects in the first version of the tool itself, both fixed and worth recording. It
reported `share_of_total`, which returned **356%** when the net is a near-zero residual of
two large sides — replaced with gross flow and a net/gross ratio. And its leave-k-out
verdict printed "dropping 0 of 250 sessions removes significance," which is meaningless
when the challenger was never significant; it now detects that case and reads the curve in
the other direction. A tool that reports nonsense confidently is the defect this project is
organised against, so it does not get a pass for being the auditor.
