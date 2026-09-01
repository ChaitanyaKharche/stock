# Results — Percentile of Achievable Return

Executed 2026-08-18 per `percentile_return_preregistration.md`. Primary endpoint and
pricing convention confirmed by the account holder before running. Nothing in the
pre-registration was altered after results were seen.

## 1. Data coverage

| | |
|---|---|
| eligible round trips | **266** (QQQ 206, SPY 60) |
| chain fetches | 266 attempted, **0 failures**, 0 retries |
| activity-date clusters | 89 |
| valid minutes/contract (QF1) | min 196, **median 389**, max 389 |
| contracts below the 60-minute floor | **0** |
| median pairs enumerated per trade | **75,466** |
| excluded upstream | 9 multi-session, 1 quantity-unbalanced, 57 single-name contracts (no entitlement) |

## 2. Controls — passed before the primary was computed

**C1, implementation check.** Random pair priced mid, ranked against the mid set:
**0.4978** (n=53,200), inside the [0.49, 0.51] pass window. Mid-rank tie handling and the
searchsorted path are correct. **PASS** — verdict (D) not triggered.

**C2, pricing-convention bias.** Random pair priced **ask→bid**, ranked against the
**mid→mid** enumeration: **0.4681**. So a trader crossing the full spread at random times
scores 0.468, not 0.500. The appropriate null is therefore the interval **[0.4681, 0.5000]**,
its position depending on how much price improvement his fills actually received. This makes
the primary result *stronger* relative to the null, not weaker.

**C3, support check.** **0 of 266** realised returns fell outside the enumerated support.

## 3. Primary result

Unconstrained enumeration, mid-to-mid, QF1:

```
mean percentile      0.5522
median percentile    0.5850
95% CI (day-cluster bootstrap, 10,000 reps)   [0.5143, 0.5885]
measured design effect                         1.37
bootstrap two-sided p vs 0.50                  0.0054
sign test                                      158 of 266 above 0.5, p = 0.0026
```

**Against the pre-registered decision rule this is INCONCLUSIVE.** The CI excludes 0.50,
but the rule required the CI *lower bound* to exceed **0.53** to declare (A), and it is
**0.5143**. The rule is not re-litigated after the fact: the bar was set in advance and was
not met.

## 4. Secondary — and the decomposition that governs interpretation

Distribution of P: p25 **0.3687**, p50 **0.5850**, p75 **0.7442**. Proportion above 0.50:
**0.594**. Above 0.53: **0.549**.

**Duration grid (pre-registered, all five reported, none promoted):**

| duration | n | mean P | 95% CI |
|---|---|---|---|
| d = 10 min | 266 | **0.5057** | [0.4528, 0.5584] |
| d = 20 min | 266 | **0.5151** | [0.4679, 0.5652] |
| d = 40 min | 266 | 0.5333 | [0.4908, 0.5748] |
| d = 60 min | 266 | 0.5444 | [0.5074, 0.5823] |
| d = close | 266 | **0.6895** | [0.6346, 0.7453] |

**This is the finding.** The statistic rises monotonically with the comparison duration, and
at short matched durations it is flat against 0.50 with CIs that contain it. The
unconstrained primary is dominated by long holds — the mean pair duration over a 389-minute
session is ~130 minutes — and a 0DTE option held for hours decays toward zero. His returns
beat that set because **he exits early**, which is a fixed policy requiring no information,
not a timing skill. The `d = close` cell (0.6895) is the pure form of the same artifact.

This is exactly the distribution-mismatch hazard the brief asked to be tested for: his
realised returns come from short holds, the enumerated opportunity set is mostly long holds,
and the comparison is not like-for-like on duration. C2 caught the pricing half of the
mismatch; the duration grid caught the larger half.

**Other secondaries:**

| cut | n | mean P | 95% CI |
|---|---|---|---|
| QQQ (descriptive) | 206 | 0.5598 | [0.5183, 0.6009] |
| SPY (descriptive) | 60 | 0.5258 | [0.4518, 0.5948] |
| ask→bid pricing | 266 | 0.5745 | [0.5361, 0.6103] |
| QF2 spread ≤ 50% | 266 | 0.5459 | [0.5080, 0.5818] |
| QF3 both sizes > 0 | 266 | 0.5459 | [0.5095, 0.5831] |
| fee-inclusive realised return | 266 | 0.5516 | [0.5142, 0.5871] |
| **single-fill subset** (1 BTO, 1 STC) | **130** | **0.5465** | **[0.4980, 0.5926]** |

The result **survives every quote-quality filter** (0.546–0.552, stable). The ask→bid
variant is higher as predicted, confirming that convention favours him. On the **single-fill
subset — where the one-entry/one-exit enumeration matches his trade structure exactly — the
CI includes 0.50**, so part of the primary effect comes from multi-tranche averaging the
enumeration cannot represent.

## 5. Conclusion

**Verdict: (B), evidence consistent with random timing, once duration is matched.**

Stated precisely: at short matched holding durations (d=10, d=20) his percentile is
**0.506 and 0.515 with CIs containing 0.50**. The above-50 unconstrained result (0.5522,
p=0.0054) is real arithmetic but is explained by his early-exit policy beating a comparison
set dominated by multi-hour 0DTE holds. Truncating the hold on a decaying asset is a policy,
not information.

The pre-registered bar for (A) was a CI lower bound above 0.53. It came in at 0.5143. **The
bar was not met and no follow-up experiment is proposed** — per the brief, the primary being
consistent with 0.50 means stop.

**What this establishes:** his entry/exit minutes are not distinguishably better than random
minutes in the same contract-session at comparable holding durations, to a resolution of
about 0.058 percentile units.

**What it does not establish:** that he has no edge. It cannot see an effect smaller than
0.058; it says nothing about day selection, direction choice, or contract selection, all of
which are outside this design; and it covers one regime (2025-05-05 → 2025-11-26) with no
out-of-sample period. A null here is not a verdict on the trader.

**One methodological debt to record.** The unconstrained primary was chosen because holding
duration is unobservable, which is correct, but it conflates *duration policy* with *timing
skill* — and that conflation turned out to drive the headline. Had this been foreseen, the
pre-registration would have named the short-duration cells as co-primary. It was not
foreseen, the duration grid was pre-registered anyway, and it is what exposed the artifact.
