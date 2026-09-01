# Results — Journal reverse-engineering, full ML toolkit

Executed 2026-08-29 per `journal_reverse_engineering_preregistration.md` and its amendments
A1–A4. No definition was changed after a result was seen. Sections still computing are marked
**PENDING** and will be appended, not substituted.

---

## The one-paragraph answer

**His rule is recoverable. The rule does not work. He is not measurably better than the rule.**

A classifier separates the minutes he entered from same-session placebo minutes at
**out-of-fold AUC 0.847** (permutation p = 0.0020) — the strongest, cleanest result this
project has ever produced. The recovered rule is momentum chasing, and it is not subtle. But
running that rule blind over 34,534 signals from 2016–2026 returns **−0.15 to −0.22 bp**, and
the strongest possible version — scoring 721,887 out-of-sample minutes with the full
45-feature model and taking only the top 1% — has a 95% CI upper bound of **+2.34 bp** against
the **+5 bp** an ATM 0DTE needs to clear spread and theta. His own entries, once a measurement
bias of mine is removed, run **+0.85 to +1.67 bp with every CI crossing zero.**

---

## 1. Sample

443 journal round trips → **418 with a complete feature vector** (25 dropped: entry too early
in the session for the indicator window; no exclusion is outcome-related), **3,274 placebos**,
**151 date clusters**, 12 symbols, window 2024-09-11 → 2026-02-20.

**51 features**, split by availability rather than by hand: 45 are **market-state** (populated
on placebo rows too) and 6 are **trade-only** (`qty`, `dte`, `opt_spread_pct`,
`trade_ordinal`, `prev_trade_net`, `prev_trade_loss`). The trade-only six are barred from Q1 —
handing them to a model that separates entries from placebos would let it win on NaN-ness
alone and score AUC 1.0. That is the leak this study most had to avoid, and it is not the one
that actually bit (see §3).

---

## 2. Q1 — SELECTION. The rule is recoverable.

| model | OOF AUC (GroupKFold-5 by date) |
|---|---|
| logistic | 0.8279 |
| **random forest** | **0.8469** |
| gradient boosting | 0.8410 |

**500 within-session label permutations:** null mean **0.5003**, sd 0.0186, p95 0.5301.
**Real 0.8469, empirical p = 0.0020.**

Pre-registered falsifier was AUC ≤ 0.55 or p > 0.05. **Cleared, decisively.**

**It is not time-of-day.** That was the first thing I checked, before any null: a model given
only the four clock features scores **0.556**; a model given everything *except* the clock
still scores **0.846**.

### What the separation actually is

Univariate AUC, his 418 entries vs 3,274 placebos:

| feature | AUC | his median | placebo median |
|---|---|---|---|
| `px_vs_ema9_al` (ATR above the 5-min 9-EMA, aligned) | **0.809** | **+1.134** | +0.141 |
| `di_diff_1m_al` (1-min DI+ − DI−, aligned) | **0.806** | **+16.04** | +0.87 |
| `trail15_sigma_al` (15-min move in 5-min sigmas) | **0.791** | **+0.854** | +0.015 |
| `ret_15m_al` (15-min move, bp) | **0.787** | **+16.17** | +0.80 |
| `ret_5m_al` | 0.764 | +8.40 | +0.17 |
| `ret_30m_al` | 0.741 | +17.73 | +1.35 |
| `px_vs_ema20_al` | 0.735 | +1.528 | +0.396 |
| `upbar_frac_al` | 0.730 | +0.200 | +0.067 |
| `macd_hist_slope_al` | 0.702 | +0.548 | −0.005 |
| `macd_hist_al` | 0.681 | +1.763 | −0.141 |
| `adx1` | 0.641 | 26.89 | 21.75 |
| `di_diff_al` (5-min DMI) | 0.634 | 10.17 | 4.25 |
| … | | | |
| `gap_al`, `prior_day_ret_al`, `weekday`, `dist_round5_atr` | **0.501–0.503** | — | — |

**He is a momentum chaser, and the description is now quantitative rather than anecdotal.**
He enters when price is roughly **1.1 ATR extended above the 5-minute 9-EMA**, the **1-minute**
DMI spread is **+16**, and the trailing 15-minute move is **+0.85 sigma** in the direction he
takes. His declared MACD/ADX/volume stack ranks *below* all of that, and the levels he cites —
round numbers, prior-day levels, the gap — are **at chance** (AUC 0.50).

Note also that the discriminating DMI feature is the **1-minute** one (AUC 0.806), not the
5-minute one (0.634). The frozen `MOMO_CHASE` setup gates on the 5-minute version. That is a
real mismatch between the setup and the behaviour it was meant to encode.

---

## 3. The correction that mattered — a bias I introduced and then found

My first measurement of his entries' forward returns anchored on **the last bar whose close
was knowable at his entry second**. That is the correct anchor for *features*. It is the wrong
base for a *fill*.

He enters on average **31.9 seconds into the following bar**, and that bar moves **+1.905 bp
in his direction**. Anchoring on the previous close hands him that entire move for free.

| 5-minute forward move | mean | 95% CI | p Holm |
|---|---|---|---|
| previous-bar-close base (**biased**) | **+2.271 bp** | [+0.745, +3.787] | **0.0096** |
| interpolated at his actual entry second | **+0.850 bp** | [−0.744, +2.414] | 0.9736 |
| entry-bar close (conservative) | +0.194 bp | — | — |

**The biased version cleared Holm. The unbiased version is not distinguishable from zero.**
This is the same one-minute execution artefact that took `IntradayMomentumBoundary` from
p = 0.0003 to p = 0.79, and it caught me here before I reported it. The interpolated fill —
price at his actual entry second, exit the same number of seconds later — is the primary
measure everywhere below.

The walk-forward tests (§4, §5) were never affected: they signal from bar *k*'s close and fill
at bar *k+1*'s **open**, exactly one minute later, with no intra-bar credit.

> **This also puts a question mark over the previously recorded "+0.0254%/trade" figure for
> his underlying capture.** That is +2.54 bp, which sits right on top of my *biased* numbers
> (+2.27 to +3.06) and well above the unbiased ones. It should be recomputed with an
> interpolated fill before it is quoted again.

---

## 4. Q3 — does the recovered rule pay? No.

The zone is the 25th percentile of his own entries on the four discriminating features:
`px_vs_ema9_al ≥ 0.71`, `di_diff_1m_al ≥ 9.39`, `trail15_sigma_al ≥ 0.39`,
`ret_15m_al ≥ 6.65 bp`. It captures **55.5%** of his entries and fires on **13.6%** of random
same-session minutes — a **4.1x lift**, so it is a faithful coarse copy of him.

Run blind on QQQ 2016-2026 and SPY 2022-2026. **34,534 signals, 3,811 symbol-days**, 9.06 per
session, 51.3% long. Fills lagged one minute on both legs.

| horizon | mean bp | win% | 95% CI | p Holm |
|---|---|---|---|---|
| 5 min | **−0.149** | 47.9% | [−0.262, **−0.033**] | **0.0456** |
| 10 min | −0.159 | 48.1% | [−0.319, +0.002] | 0.1584 |
| 15 min | −0.170 | 48.9% | [−0.367, +0.035] | 0.1756 |
| 25 min | −0.218 | 49.2% | [−0.466, +0.033] | 0.1756 |

Negative at every horizon, **significantly negative at 5 minutes**, win rate below 50%
throughout, negative in 7 of 11 years, negative in both directions. Every CI upper bound is
below +0.04 bp against a +5 bp requirement — short by a factor of roughly 140.

Chasing extension into the 9-EMA mean-reverts slightly. He is paying the spread to buy the top
of a move.

---

## 5. Q3b — the strongest possible version. Also no.

Objection to §4: the zone fires 9.06 times a session against his ~2.8 trades a day, so it is
broader than he is and his sub-selection might be the edge. Q3b removes that objection by
using the **entire Q1 classifier** — all 45 features, AUC 0.847 — as the signal, scored over
**721,887 QQQ minutes from 2016-01-04 to 2024-09-10**, which ends the day before his first
journal trade and is therefore out-of-sample in time by construction.

**Out-of-sample, top centile (n = 2,419, 1,403 days):**

| horizon | mean bp | win% | 95% CI | p Holm |
|---|---|---|---|---|
| 5 min | −0.261 | 45.6% | [−0.898, +0.360] | 0.8210 |
| 10 min | −0.112 | 45.7% | [−1.056, +0.807] | 0.8210 |
| 15 min | +0.859 | 48.3% | [−0.253, +1.905] | 0.4980 |
| 25 min | **+1.035** | 48.4% | [−0.256, **+2.339**] | 0.4980 |

Top decile (n = 12,986): −0.15 bp at every horizon, Holm 1.0000. In-period 2024-2026 top
centile (n = 516): every CI crosses zero, best upper bound +2.10 bp.

**Pre-registered falsifier: top-centile CI upper bound below +5 bp at every horizon.** The
best is **+2.34**. **Fired.** Win rates of 45.6–48.4% in the *most confident 1%* say the model
is, if anything, mildly anti-predictive.

A baseline worth recording: across all 721,887 scored minutes, entering in the direction of
the trailing move returns **−0.081 bp** at 5 minutes (Holm 0.0010). Intraday QQQ momentum
mean-reverts at short horizons. That is the sea the whole strategy is swimming in.

---

## 6. His own entries, measured honestly

Interpolated fill at his actual entry second, n = 425:

| horizon | mean bp | win% | 95% CI | p raw | p Holm |
|---|---|---|---|---|---|
| 5 min | +0.850 | 53.4% | [−0.744, +2.414] | 0.2956 | 0.9736 |
| 10 min | +1.286 | 52.9% | [−1.186, +3.626] | 0.2954 | 0.9736 |
| 15 min | +1.671 | 54.8% | [−1.161, +4.411] | 0.2434 | 0.9736 |
| 25 min | +1.637 | 56.0% | [−1.937, +5.132] | 0.3692 | 0.9736 |

ETF-only (QQQ/SPY, n = 347): +0.313 at 5 min, +1.461 at 15, **+2.147** at 25.

**Every point estimate is positive. Not one is significant.** Win rates of 53–56% are mildly
encouraging and entirely consistent with noise at this n.

What is worth saying precisely: **his entries do beat every mechanised version of himself.**
+0.85 to +1.67 bp against the zone's −0.15 to −0.22 and the top-centile model's −0.26 to
+1.04. That gap is the part of his judgement the 45 features do not contain. It is also,
at this sample size, indistinguishable from luck — and it is below the conversion floor at
every horizon but the longest.

---

## 7. Q4 — the EXIT

Reconstructed from real 1-minute NBBO on **his actual contracts**: 443 round trips → 16 dropped
as multi-day, 5 with unusable paths, **422 reconstructed** over **154 date clusters**. Entry at
the ask, exit at the bid, $0.0404/side.

**Sanity check:** simulated **+$0.78/trade** against his broker-reported **+$3.43/trade**.
Crossing the spread both ways is $2.65/trade worse than what he actually got — he uses limit
orders and gets real price improvement. That is a small, genuine, and previously unmeasured
execution skill.

### Q4a — 0 of 14 mechanical alternatives clear Holm

| rule | $/trade | vs his | 95% CI on diff | p Holm |
|---|---|---|---|---|
| target +25% | +5.00 | +4.22 | [−7.98, +16.33] | 1.0000 |
| target +50% | −2.61 | −3.39 | [−10.61, +3.22] | 1.0000 |
| target +75% | −3.80 | −4.59 | [−11.06, +0.47] | 1.0000 |
| target +100% | −2.43 | −3.21 | [−9.09, +1.58] | 1.0000 |
| stop −25% | −2.62 | −3.40 | [−12.87, +6.66] | 1.0000 |
| stop −50% | +0.56 | −0.22 | [−5.34, +4.86] | 1.0000 |
| time 5 min | −3.81 | −4.59 | [−19.15, +9.25] | 1.0000 |
| time 10 min | −5.05 | −5.83 | [−17.83, +5.07] | 1.0000 |
| time 15 min | −7.57 | −8.36 | [−20.76, +2.93] | 1.0000 |
| time 25 min | −5.02 | −5.80 | [−15.22, +2.14] | 1.0000 |
| time 45 min | +0.87 | +0.09 | [−5.33, +5.18] | 1.0000 |
| trail −20% from peak | −5.73 | −6.51 | [−21.12, +7.27] | 1.0000 |
| trail −30% | −6.05 | −6.83 | [−21.28, +6.54] | 1.0000 |
| trail −50% | −6.01 | −6.79 | [−21.04, +6.38] | 1.0000 |
| **his exit** | **+0.78** | — | — | — |

**Pre-registered falsifier fired: no counterfactual differs from him at Holm p < 0.05.**

**This weakens the one positive result this project had.** Test B found a +25% target *losing*
to his discretion by 8.97pp at Holm 0.027, measured in **return** space. Measured in **dollar**
space on 422 trades with fall-through to his own exit, the same rule **gains** +$4.22 and is
nowhere near significance (p = 0.487). The sign flips with the weighting and neither version is
robust. Test B should be treated as fragile, not as established.

### Exit quality diagnostics — the largest number in this study

| | |
|---|---|
| mean maximum favourable excursion | **+$81.68** |
| mean maximum adverse excursion | −$76.67 |
| mean realised | **+$0.78** |
| median capture of MFE | **32.4%** |
| mean hold | **40.6 min** |
| mean minute the peak bid arrives | **19.1 min** |
| **trades exited after the peak** | **74.6%** |
| winners | hold 37.8 min, capture **86.0%** of MFE |
| losers | hold **43.0 min** |

(The *mean* capture ratio of −125% is not reportable — trades whose MFE is a cent give it an
exploding denominator. The median is the honest statistic.)

The average trade shows **$81.68 of peak paper profit and realises $0.78 of it**. The peak
arrives at minute 19; he holds to minute 41. He exits after the peak three times in four, and
he **holds losers longer than winners** — textbook disposition effect.

**And yet — none of the 14 rules that try to exploit this help.** Time exits at 5/10/15/25 min
are all *worse*. Every trailing stop is worse. Both fixed stops are worse. MFE is a maximum
over a path and every random walk has a big one; the gap between MFE and realised is mostly
arithmetic, not opportunity. **This is exactly the trap the toolkit warns about: a large,
interpretable, intuitive number that no implementable rule can convert.**

### Q4b — his exit IS mechanisable

Per-minute panel, 17,148 rows over 422 trades, within-trade permutation null (permute which
minute of the hold is labelled "exit", so the trade is held constant).

**OOF AUC 0.7036** against a null of **0.6760** (sd 0.0113), **p = 0.0100**. Pre-registered
falsifier was AUC <= 0.55 or p > 0.05. **Cleared.**

The null is high because "minutes elapsed" trivially flags the end of a hold; the
within-trade permutation absorbs most of that, and a 0.028 margin survives it.

**Read Q4a and Q4b together, because they say opposite-sounding things and both are true:**
his exit timing *is* a function of observables (drawdown from peak, unrealised return, time
held) — so it can be copied — **and** copying it buys nothing, because no mechanical version
of it beats him and he beats none of them either. He is running a reproducible exit policy of
no measurable value.

---

## 8. M4 — Lo–Mamaysky–Wang kernel patterns: 0 of 10

Nadaraya–Watson smoothing, Gaussian kernel, LOO-CV bandwidth h = 0.60 → LMW's 0.3 multiplier
→ **h = 0.18**, 38-bar within-session windows on 5-minute bars. **129,693 windows** across
QQQ (2,657 sessions) and SPY (1,158).

| pattern | n | freq | mean (sigma, 15 min) | KS p | KS Holm |
|---|---|---|---|---|---|
| HS | 20 | 0.02% | +0.345 | 0.774 | 1.000 |
| IHS | 21 | 0.02% | −0.671 | 0.094 | 0.751 |
| BTOP | 2,619 | 2.02% | +0.001 | 0.494 | 1.000 |
| BBOT | 2,698 | 2.08% | +0.064 | 0.137 | 0.824 |
| TTOP | 2,503 | 1.93% | +0.012 | 0.015 | 0.135 |
| TBOT | 2,839 | 2.19% | −0.002 | 0.007 | 0.065 |
| RTOP / RBOT | 6 / 5 | — | too few | — | — |
| DTOP | 3,052 | 2.35% | +0.016 | 0.095 | 0.751 |
| DBOT | 2,739 | 2.11% | +0.053 | 0.268 | 1.000 |

**Nothing clears Holm at 15 or 25 minutes.** Head-and-shoulders and its inverse are essentially
nonexistent intraday (n = 20 and 21 in 129,693 windows) — worth knowing on its own, given how
much chart commentary is built on them. The two triangle patterns have low raw KS p-values with
mean effects of +0.012 and −0.002 sigma: a distributional difference with no economic content,
which is what a KS test on 2,500 observations will find when shapes differ trivially.

---

## 9. Methods 2 and 3, and Q2

### Method 2 — per-feature null importance, and a flaw in my own pre-registration

Q1, gradient boosting, 200 label-permuted refits:

| feature | importance | null mean | z | p | q(BH) |
|---|---|---|---|---|---|
| `px_vs_ema9_al` | 0.04492 | −0.00028 | **11.44** | 0.005 | 0.112 |
| `ret_5m_al` | 0.01415 | −0.00018 | **3.14** | 0.005 | 0.112 |
| `ret_1m_al` | 0.01124 | +0.00015 | 2.09 | 0.025 | 0.373 |
| `di_diff_1m_al` | 0.00713 | +0.00014 | 1.54 | 0.080 | 0.597 |
| `tod_min` | 0.00644 | −0.00012 | 1.57 | 0.060 | 0.582 |
| … 40 more, all z < 1.6 | | | | | |

**Features clearing BH q < 0.05: 0. That number is an artefact of my own design and must not
be read as a null.** With 200 permutations the smallest attainable p is 1/201 = 0.00498. BH at
q = 0.05 across 45 features requires the top-ranked feature to reach p ≤ 0.05/45 = 0.0011.
**No feature could have passed, whatever the data said.** The pre-registration specified a
criterion its own permutation count could not satisfy; that is my error, recorded here rather
than quietly restated.

What the data does say, from the statistic that is not resolution-limited: `px_vs_ema9_al`
sits **11.4 null standard deviations** above its permuted distribution and `ret_5m_al` sits
3.1 above. Both are pinned at the permutation floor. Everything else is inside noise.
Concentration confirms it: the top 5 features hold **67.8%** of positive importance against
the 11.1% an even spread would give.

Correcting this properly needs ~5,000 permutations (a ~70-hour job as configured) or a
parametric tail on the z-scores. Neither changes the economics in §4–§6, so neither was run.

### Method 3 — the decision tree, and its noise control

The control worked exactly as intended, and it separates the two questions cleanly:

| tree | base rate | best OOF leaf | permuted p95 | verdict |
|---|---|---|---|---|
| **Q1 selection** | 0.113 | **0.410** | 0.231 (max 0.346) | **ABOVE the band, p = 0.0020** |
| **Q2b win/loss** | 0.452 | 0.593 | **0.704** (max 0.776) | **INSIDE the band, p = 0.4331** |

A depth-3 tree on pure noise routinely finds a 70% "win rate" leaf in 418 trades. **The real
win/loss tree found 59.3% — worse than the median noise result would suggest is impressive,
and well inside the band.** This is the single clearest demonstration in the study of why an
interpretable rule is not evidence. The same tree fit against *what he does* rather than
*whether it worked* clears the band by a mile.

Both Q1 splits are on `px_vs_ema9_al` and `trail15_sigma_al` — the same two features
everything else points at.

### Q2a — entry state vs the forward underlying move: 0 of 4

| horizon | best OOF R² | AUC(sign) | p | 
|---|---|---|---|
| 5 min | +0.0001 | 0.5459 | 0.0448 raw → **0.179 Holm** |
| 10 min | −0.0527 | 0.4308 | 0.4975 |
| 15 min | −0.0785 | 0.4733 | 0.8259 |
| 25 min | −0.0769 | 0.4680 | 0.7463 |

An R² of +0.0001 is not an edge in any sense. Ridge is worse than the tree models at every
horizon (R² −0.21 to −0.51), and three of four sign-AUCs are **below 0.50**.

### Q2b — entry state vs his realised money: null

OOF AUC **0.5117** (logistic 0.4545, random forest 0.4745), 200-permutation null mean 0.4867,
**p = 0.2836**. Nothing.

**The contrast is the whole study in two lines:** predicting *whether he enters* → AUC 0.847,
p = 0.002. Predicting *whether it worked* → AUC 0.512, p = 0.284.

---

## 10. Verdict

| # | question | answer |
|---|---|---|
| Q1 | Is his rule recoverable? | **Yes. AUC 0.847, p = 0.0020.** Momentum chasing, quantified. |
| Q3 | Does the recovered rule pay? | **No.** −0.15 to −0.22 bp over 34,534 signals; −0.149 bp significantly negative at 5 min. |
| Q3b | Does the *full model* pay? | **No.** Best CI upper bound +2.34 bp against a +5 bp floor. |
| — | Do his own entries pay? | **Positive but not significant**, +0.85 to +1.67 bp, all CIs cross zero. |
| Q4a | Does his exit beat mechanical rules? | **No** — and no mechanical rule beats him either. 0 of 14. |
| M4 | Do kernel chart patterns pay? | **No.** 0 of 10. |

The methodology worked exactly as it is supposed to. It found a strong, real, reproducible
pattern — **what he does** — and then refused to confuse it with **what works**. The null
importance and permuted-tree controls exist to stop the second from being read off the first,
and here the economics answered before they needed to.

**Nothing in this study touches the frozen forward test.** `forward_test_preregistration.md`
§5 stands: the clock never resets. Nothing here proposes a new `setup_id`, because nothing here
earned one.

**The single most useful thing produced:** his entries occupy a measurable, specific state
(1.1 ATR above the 5-min 9-EMA, 1-minute DMI +16, trailing 15-min move +0.85 sigma), and that
state is worth **−0.15 bp** when traded mechanically. Whatever separates him from it is not in
these 51 observables, is worth at most 1–2 bp, and cannot be copied from this data.
