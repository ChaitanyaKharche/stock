# Results — Three-Endpoint Pass (C1, F1, F2)

Executed 2026-08-18 per `c1_f1_f2_preregistration.md`. No definition was altered after a
result was seen. Day-clustered bootstrap, 10,000 replicates, Holm across exactly three
primaries.

## 1. Sample construction

| | |
|---|---|
| eligible round trips | **318** (ETF 266, single-name 52) |
| C1 sample (ETF only, per §1 of the pre-registration) | **266**, 89 date clusters |
| F1 / F2 sample | **318**, 103 date clusters |
| split/adjustment filter exclusions | **0** |
| outcome Y ($/contract, net of fees) | mean **+4.88**, sd 39.81, median −2.08 |
| C1 predictor `d` (daily sigmas OTM) | mean 0.591, sd 0.872, p10 −0.545, p90 1.816 |
| F1 predictor `q` (contracts) | mean 4.63, median 5, range 1–16 |
| F2 predictor `k` (within-day ordinal) | mean 2.58, median 2, max 9 |

Single names were excluded from C1 on the pre-registered comparability grounds (23% vs 94%
0DTE) and included in F1/F2, where the predictors need no market data. R1 confirms they do
not drive either result.

## 2. Leak audit outcome

No endpoint uses post-trade information. Verified as specified: `d` uses only the strike and
prior-session data; `q` is chosen at entry; `k` is known at the moment of that entry, and the
day's *total* count was never used as a regressor. The premium linkage in C1/F1 is a
mechanical confound, addressed by covariate adjustment, not outcome leakage. **No endpoint
was disqualified.** The averaging-down comparison remains excluded for leakage.

## 3. Primary results

| endpoint | coefficient | 95% CI | p raw | **p Holm** |
|---|---|---|---|---|
| **C1** `Y ~ d + premium` | **−1.003** $/contract per sigma | [−6.108, +4.218] | 0.6912 | **1.0000** |
| **F1** `Y ~ q` | **−2.034** $/contract per +1 contract | [−3.793, −0.383] | 0.0160 | **0.0480** |
| **F2** `Y ~ k` | **+0.301** $/contract per +1 step later | [−2.885, +3.844] | 0.8440 | **1.0000** |

Only F1 survives Holm, and only barely (0.048 against 0.05).

## 4. F1 — exposure versus prediction

The pre-registered diagnostics, and they matter more than the primary:

| quantity | result |
|---|---|
| `corr(q, premium)` | **−0.181** — he sizes up on cheaper options |
| `\|Y\| ~ q` slope | **−2.261**, CI [−3.404, −1.245], p = **0.0005** |
| `total$ ~ q` slope | −8.174, CI [−18.540, +2.398], p = 0.126 — **not significant** |
| `Y ~ q + premium`, q slope (R7) | −2.062, CI [−3.807, −0.426], p = 0.010 |
| **percent-return outcome (R3)** | **−0.0177, CI [−0.0409, +0.0046], p = 0.139 — not significant** |

**The pre-registered exposure rule was written for a case that did not occur.** It said:
zero slope on `Y` plus a *positive* slope on `|Y|` ⇒ exposure only. Here the slope on `Y` is
non-zero and the slope on `|Y|` is **negative** — larger positions have *smaller* per-contract
magnitude, not larger. I am not substituting a new rule to reach a verdict; I am reporting
what the numbers support.

What they support: the dollar effect is significant, the **percent-return effect is not**
(p = 0.139), and the divergence is explained by the −0.181 size↔premium correlation. Dollars
per contract = percent × premium × 100, so if bigger positions sit on cheaper options, their
dollar magnitude compresses on both sides — which is exactly the negative `|Y|` slope. A
linear premium covariate cannot fully remove a multiplicative channel, so the surviving
−2.06 in R7 does not settle it either. And at the level that determines the account,
**total dollars per position, the effect is not significant** (p = 0.126).

## 5. Robustness (all pre-specified, none promoted)

| check | C1 `d` | F1 `q` | F2 `k` |
|---|---|---|---|
| R1 QQQ/SPY only | n/a | −2.157, p 0.008 | +0.146, p 0.931 |
| R2 winsorised Y | −0.952, p 0.739 | −2.000, p 0.012 | +0.176, p 0.898 |
| R3 percent return | −0.0157, p 0.761 | **−0.0177, p 0.139** | +0.0095, p 0.654 |
| R4 0DTE only | −1.262, p 0.670 | n/a | n/a |
| R5 gross of fees | −1.002, p 0.679 | −2.034, p 0.017 | +0.301, p 0.829 |
| R6 day fixed effects | n/a | n/a | +1.125, p 0.484 (n=300) |

C1 and F2 are stable nulls across every check. F1 is stable in dollars and absent in percent.

Descriptive mean Y by ordinal (no binning was used in any test): k=1 +4.89 (n=102), k=2 +3.35
(84), k=3 +2.86 (55), k=4 +11.68 (34), k=5 +9.76 (24), k=6 −7.14 (9), k=7 +5.25 (6). No
monotone pattern.

## 6. Blunt conclusions

**C1 — no detectable information.** The sigma-normalised distance of his chosen strike carries
no measurable relationship to realised dollar outcome: −$1.00 per sigma with a CI spanning
−$6.11 to +$4.22, and the same null in percent space, 0DTE-only, and winsorised. **The
descriptive SPY-versus-QQQ gap (+$42 on 69 round trips against +$2,995 on 206) is not
explained by normalised strike distance.** Whatever produced it, this variable is not it. This
was the highest-priority remaining question and it is now closed.

**F1 — exposure/scale effect; prediction not demonstrated.** Significant in dollars at
Holm p = 0.048, absent in percent returns (p = 0.139), absent in total dollars per position
(p = 0.126), and accompanied by a strongly negative `|Y|` slope indicating magnitude
compression rather than worse selection. The most defensible reading is that his larger
positions sit on cheaper contracts and therefore produce smaller per-contract dollar swings
in both directions. **This is not evidence that he sizes up on trades he predicts worse.** It
is marginal, scale-driven, and one failed robustness check from nothing.

**F2 — no detectable information.** +$0.30 per ordinal step, CI [−2.89, +3.84], p = 0.844;
with day fixed effects +1.13, p = 0.484; no monotone pattern in the descriptive means. **This
retires the previously circulated entries-per-day gradient** ($14.47 for 1/day down to $3.36
for 5+/day). That gradient used the day's total entry count, which is known only at the close;
the clean ex-ante ordinal shows nothing. Later trades in a day are not worse.

**Family-wise:** one of three endpoints reaches significance, at Holm p = 0.048, and it fails
its own scale-invariance check. Treated as a family, this pass found **no robust behavioural
leak.**

## 7. What uncertainty remains that is materially important

**One thing, and it is the same thing as before: whether the trading has an edge at all is
unresolved and cannot be resolved by this dataset.** Dollar record t ≈ 0.74 on 333 round
trips; a true $10/contract edge needs ≈2,357. This pass did not address that question and was
never going to — it was leak detection, and it found no leak.

**Two specific uncertainties that are now smaller, and worth recording as closed:** contract
selection by volatility-adjusted strike distance carries no information (C1), and within-day
sequencing carries no information (F2). Both were live hypotheses an hour ago. Neither should
be revisited on this data.

**One that is materially important and remains open:** the entire intraday decision layer —
entry timing, exit timing, and whether retest/rejection differentiates anything — is
unidentified without execution timestamps, not merely underpowered. No re-analysis of the
activity export reaches it.

**One that is structurally unreachable here:** the value of abstention. He traded 74% of
sessions; the dataset contains only the treated arm, so what his skipping was worth cannot be
estimated from it at all.

No further experiment is proposed.
