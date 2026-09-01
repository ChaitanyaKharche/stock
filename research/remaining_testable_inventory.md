# Research Inventory — What Remains Testable Without Execution Timestamps

Compiled 2026-08-18 after the pre-registered percentile experiment returned **(B),
consistent with random timing**. No experiments were run to produce this document; it is a
scoping exercise only.

## Standing facts this inventory is built on

| | |
|---|---|
| option fills / contracts | 1,051 / **333** |
| eligible QQQ+SPY single-session round trips | **266** (QQQ 206, SPY 60) |
| out of scope (9 single names, no options entitlement) | 57 contracts, 197 fills (18.7%) |
| window | 2025-05-05 → 2025-11-26, **105 active days of ~144 sessions (74%)** |
| verified P&L | **+$2,341.12**, mean **+$7.03/contract**, dollar **t = 0.63–0.91** |
| design effect (day clusters) | **1.31–1.43** (measured 1.37 in the percentile test) |
| dollar P&L SD | **$173/contract** → MDE **±$27/trade** at 80% power |
| n needed to resolve a true $10/contract edge | **≈2,357** round trips |
| binary antecedent MDE (winners vs losers) | **±15–16 percentage points** |
| day-selection detectable odds ratio | **≥ 2.5** and nothing smaller |
| **what timestamps block** | any within-day *timing* question |
| **what reverse chronology unblocks** | within-day decision *order* (2 violations / 1,051 fills) |

**The arithmetic that governs everything below:** 333 round trips cannot establish whether
he has an edge. That is not pessimism, it is a power calculation. What 333 trades *can*
plausibly detect is a **large leak** — a behaviour costing far more than $27/trade. The
inventory is therefore sorted by whether a question can find a leak, not by whether it can
find an edge.

---

## A. Day selection

### A1 — Do pre-open observables predict whether he trades that day?

| | |
|---|---|
| **Hypothesis** | The probability he opens a position is a function of pre-open state (overnight gap, prior-day range, premarket range, trailing sigma, weekday) |
| **Data required** | Activity dates; prior-close and premarket stock minute bars |
| **Present?** | **Yes** — ThetaData stock line covers both, no fetch gaps |
| **Causal/descriptive** | **Predictive** — all regressors strictly pre-open |
| **Sample** | 144 sessions, 106 traded |
| **Power** | **Poor.** A 74% base rate leaves little variance to explain; detects OR ≥2.5 only |
| **Falsifier** | No pre-registered variable reaches OR ≥2.5 at Fisher p<0.05 |
| **Circularity** | Low. **Multiplicity high** — the variable space is unbounded; must pre-register one |
| **Duplicates?** | Partially. Independence from the *mechanical rule* was tested (Fisher p=0.221–1.000). General pre-open observables were not |
| **Priority** | **LOW** — he trades nearly everything; there is almost no selection to detect |

### A2 — Do the days he skipped differ from the days he traded?

| | |
|---|---|
| **Hypothesis** | Skipped sessions differ systematically in pre-open state from traded sessions |
| **Data required** | Same as A1 |
| **Present?** | **Yes** |
| **Causal/descriptive** | Descriptive. **The *value* of abstention is unidentified in principle** — the dataset contains only the treated arm |
| **Sample** | **38 skipped** vs 106 traded |
| **Power** | Weak; binary MDE ≈ ±18 points |
| **Falsifier** | No difference exceeding 18 points on the pre-registered variable |
| **Circularity** | Low |
| **Duplicates?** | No |
| **Priority** | **LOW** — the only window onto abstention, but it cannot measure whether abstaining *helped*, only whether skipped days looked different |

---

## B. Direction selection

### B1 — Is his call/put choice better than a coin flip?

| | |
|---|---|
| **Hypothesis** | On days he commits to one side, that side matches the session's realised direction more than 50% of the time |
| **Data required** | Right per contract; session open/close |
| **Present?** | **Yes** |
| **Causal/descriptive** | **Predictive** |
| **Sample** | **≈46 one-sided symbol-days** (he trades both sides on 59 of 105 active days) |
| **Power** | **Poor** — n=46 gives MDE ≈ ±20 points against 50% |
| **Falsifier** | Hit-rate 95% CI contains 0.50 |
| **Circularity** | Low, provided "correct direction" is close-vs-open and never references an inferred entry |
| **Duplicates?** | No |
| **Priority** | **LOW–MEDIUM** — clean question, sample too small to answer it |

### B2 — On both-sided days, does the side he entered *first* carry information?

| | |
|---|---|
| **Hypothesis** | Given the recovered chronology, the first side taken on a two-sided day matches the session direction more than 50% of the time |
| **Data required** | Recovered within-day order + session open/close |
| **Present?** | **Yes** — newly available from the reverse-chronology finding |
| **Causal/descriptive** | **Predictive** |
| **Sample** | **59 both-sided days** |
| **Power** | Weak; MDE ≈ ±18 points |
| **Falsifier** | First-side hit-rate CI contains 0.50 |
| **Circularity** | Low |
| **Duplicates?** | No — not previously possible |
| **Priority** | **LOW–MEDIUM** — genuinely new capability, still underpowered |

---

## C. Contract / strike selection

### C1 — Does sigma-normalised strike distance predict his per-trade outcome?

| | |
|---|---|
| **Hypothesis** | Per-contract dollar outcome declines in `(strike − prior close)/(σ₂₀ × spot)`; i.e. the further out he reaches in volatility-adjusted terms, the worse he does |
| **Data required** | Strike (from Description), prior daily close, trailing 20-day σ, realised P&L |
| **Present?** | **Yes, entirely. No timestamps needed** — strike and σ are both fixed before any entry, whenever it occurred |
| **Causal/descriptive** | **Predictive**, and the cleanest such question left: regressor strictly ex-ante, outcome strictly ex-post |
| **Sample** | **266** (QQQ+SPY); ~333 if single-name stock bars prove entitled |
| **Power** | n_eff ≈ 194 at DEFF 1.37. Detects a slope explaining ≳4% of outcome variance. The observed contrast is large — SPY strikes sit **~2.8× further out in σ** and netted **+$42 on 69 round trips** against QQQ's **+$2,995 on 206** |
| **Falsifier** | Slope 95% CI (day-clustered) contains zero |
| **Circularity** | **Low.** No outcome enters the regressor |
| **Duplicates?** | No. The QQQ/SPY gap was observed descriptively; the normalised relationship was never tested |
| **Priority** | **HIGH** — largest clean effect in the dataset, fully answerable, and it targets a *leak* rather than an edge |

### C2 — Does premium paid predict outcome?

| | |
|---|---|
| **Hypothesis** | Cheaper premium → worse outcome |
| **Data required** | Fill prices, realised P&L |
| **Present?** | Yes |
| **Causal/descriptive** | Descriptive and **mechanically confounded**: cheap ⇒ far OTM ⇒ low delta ⇒ lower win rate by option geometry, independent of any decision quality |
| **Sample** | 266–333 |
| **Power** | Adequate, but measuring a tautology |
| **Falsifier** | n/a — the geometry guarantees the sign |
| **Circularity** | High in interpretation, not in construction |
| **Duplicates?** | **Yes** — already measured by premium quartile (−$2.90 / +$11.75 / +$22.79 / +$14.96, all \|t\|<1) |
| **Priority** | **LOW — do not run.** C1 is the same question with the confound removed |

### C3 — Was his strike better than the alternatives on the same chain?

| | |
|---|---|
| **Hypothesis** | Given day and direction, his chosen strike outperformed neighbouring strikes |
| **Data required** | Full chain **priced at his entry moment** |
| **Present?** | **No.** Requires the entry minute to price alternatives at a common instant |
| **Causal/descriptive** | Would be predictive |
| **Sample** | n/a |
| **Power** | n/a |
| **Falsifier** | n/a |
| **Circularity** | Any version using a reconstructed entry time is circular — the reconstruction pins moneyness, which is the answer |
| **Duplicates?** | The duration-free variant collapses into the percentile experiment, already returned (B) |
| **Priority** | **BLOCKED — requires execution timestamps** |

---

## D. Entry timing

### D1 — Anything about *when* he entered

**BLOCKED and CLOSED.** Requires execution timestamps. The achievable-return formulation was
the pre-registered percentile experiment and returned **(B)**: 0.5057 at d=10 and 0.5151 at
d=20, both CIs containing 0.50. **Do not re-run in another formulation.** The
fill-price→NBBO inversion is excluded by instruction and independently measured as
many-to-one (P(unique) ≈ 0% in-spread, ~149-minute candidate spans).

### D2 — Does averaging down predict worse outcomes?

| | |
|---|---|
| **Hypothesis** | Contracts where a later buy was below the first ("averaging down") underperform single-entry contracts |
| **Data required** | Recovered order + fill prices + P&L |
| **Present?** | Yes |
| **Causal/descriptive** | **Neither — it leaks.** |
| **Sample** | 46 average-down vs 215 single-entry |
| **Power** | Marginal; two-sample MDE ≈ ±$79 |
| **Falsifier** | Would be a CI containing zero, but see below |
| **Circularity** | **HIGH — disqualifying. "Averaged down" is definitionally "the position was losing when he added," and a contract that fell after the first buy is more likely to end down.** The grouping variable is a function of the outcome path. De-leaking requires comparing against contracts that *also* fell after the first buy but where he did not add — which needs the first buy's timestamp to establish "after". **BLOCKED.** |
| **Duplicates?** | Measured by two agents (−11.1%/33% win vs +5.8%/50%); **neither de-leaked it, and the figures should not be quoted as a behavioural finding** |
| **Priority** | **DO NOT RUN.** Record as not cleanly testable without timestamps |

---

## E. Exit timing

### E1 — Anything about *when* he exited

**BLOCKED and CLOSED**, same grounds as D1. The percentile experiment's `d=close` cell
(0.6895) already quantifies the one identified exit effect.

### E2 — Does his early-exit policy beat holding the same contracts to expiry?

| | |
|---|---|
| **Hypothesis** | Realised P&L exceeds the counterfactual of holding each contract from his fill price to expiry |
| **Data required** | Strikes, fill prices, session closes |
| **Present?** | Yes, no timestamps needed |
| **Causal/descriptive** | Descriptive policy comparison, fully identified |
| **Sample** | 266–333 |
| **Power** | Excellent — the effect is large and one-directional |
| **Falsifier** | None meaningful; the answer is already known directionally |
| **Circularity** | Low |
| **Duplicates?** | **Yes, twice.** 63% of his QQQ and 83% of his SPY 0DTE strikes finished OTM, and the percentile `d=close` cell measures the same thing |
| **Priority** | **LOW — do not run.** Already established: not letting 0DTE decay is a *policy*, not information |

### E3 — Does tranched exiting beat a single exit?

| | |
|---|---|
| **Hypothesis** | Proceeds from multi-tranche exits exceed selling the whole position at the first tranche's price |
| **Data required** | Recovered order + fill prices |
| **Present?** | Yes |
| **Causal/descriptive** | The accounting contrast is clean; **selection into tranching is not** — he may tranche in response to the path |
| **Sample** | **68** genuine tranched exits (52 of 120 multi-STC contracts sold every tranche at one price — partial fills of one order, not decisions) |
| **Power** | Weak at n=68 |
| **Falsifier** | Median uplift CI contains zero |
| **Circularity** | Moderate — path-dependent selection into the treatment |
| **Duplicates?** | **Largely** — median uplift of last tranche over first already measured at **+0.0%** |
| **Priority** | **LOW — do not run** |

---

## F. Position sizing

### F1 — Does position size predict per-contract outcome?

| | |
|---|---|
| **Hypothesis** | Larger positions carry worse per-contract returns — he sizes up on his worse trades |
| **Data required** | Quantity per contract, realised P&L |
| **Present?** | **Yes, fully. No timestamps** |
| **Causal/descriptive** | **Predictive** — size is chosen at entry, outcome follows |
| **Sample** | 266–333 |
| **Power** | Marginal at the observed effect: correlation ≈ **−0.083**, SE ≈ 0.061 at n=266, so t ≈ 1.4. **Underpowered for what has been observed**, adequate for a leak twice that size |
| **Falsifier** | Day-clustered correlation/slope 95% CI contains zero |
| **Circularity** | **Low** |
| **Duplicates?** | Measured descriptively (equal-weight +6.2% vs dollar-weighted +3.13%); never tested with clustered inference |
| **Priority** | **MEDIUM–HIGH** — clean, actionable, targets a leak. Honest about being underpowered at the effect already seen |

### F2 — Does within-day entry sequence predict outcome?

| | |
|---|---|
| **Hypothesis** | Per-contract outcome declines with the entry's ordinal position within that day |
| **Data required** | Recovered within-day order + P&L |
| **Present?** | **Yes — newly unblocked.** Previously recorded as blocked on timestamps; *order* suffices and order is established |
| **Causal/descriptive** | **Predictive for entries 2..n** — the ordinal is known at the moment of that entry |
| **Sample** | **≈228** non-first entries across 105 days |
| **Power** | Moderate; a continuous ordinal regressor at n_eff ≈ 166 |
| **Falsifier** | Slope 95% CI (day-clustered) contains zero |
| **Circularity** | **Low for the ordinal version. HIGH for the "total entries that day" version** — that is known only at the close and must not be used as a regressor. The published bucket figures (1/day $14.47 → 5+/day $3.36) use the ex-post version and are contaminated |
| **Duplicates?** | The contaminated bucket version was measured; the clean ordinal version was not |
| **Priority** | **MEDIUM–HIGH** — newly answerable, clean, and it separates a real leak (deterioration through the day) from an artifact |

---

## G. Market-regime selection

### G1 — Does pre-open regime predict his per-trade outcome?

| | |
|---|---|
| **Hypothesis** | Per-contract outcome depends on one pre-registered pre-open regime variable (trailing σ₂₀, overnight gap, or prior-day range) |
| **Data required** | Stock minute/daily bars before the open; P&L |
| **Present?** | **Yes** |
| **Causal/descriptive** | **Predictive** |
| **Sample** | 266–333 |
| **Power** | Weak — MDE ±$27/trade overall, ≈±$47 for a tercile contrast |
| **Falsifier** | Coefficient 95% CI (day-clustered) contains zero |
| **Circularity** | Low. **Multiplicity is the hazard** — three variables × several functional forms is a garden of forking paths; exactly one must be pre-registered |
| **Duplicates?** | No. Prior vol-tercile work was on the *mechanical* strategy, not his trades |
| **Priority** | **MEDIUM** — clean but underpowered, and the multiplicity temptation is high |

### G2 — Does he *choose* to trade in particular regimes?

Same structure as A1 and defeated by the same 74% base rate. **Priority: LOW.**

---

## Blocked — requires execution timestamps

D1, E1, C3, and the de-leaked version of D2. Also anything about retest/rejection, which is
a 5–20 minute intraday sequence. **Robinhood carries execution times in trade confirmations
and monthly statements, not in the activity export.** That single request unblocks this
entire class.

## Blocked — requires additional history

Any out-of-sample validation. The window is seven months and one regime, with **no held-out
period by construction**, so anything fitted here has nowhere to be tested.

---

## The smallest set that meaningfully reduces uncertainty

Three experiments, in this order. All are **leak-detection**, not edge-detection — because
333 trades cannot resolve an edge and no re-analysis changes that.

**1. C1 — sigma-normalised strike distance → per-trade dollar outcome. (HIGH)**
One pre-registered regression, day-clustered. Fully answerable now, no timestamps, no fetches
beyond the daily bars already held. It is the largest clean contrast in the dataset and it
carries a concrete consequence: if the slope is real, the SPY book's ~2.8× further reach is a
measurable cost, and the remedy is a contract-selection change rather than a strategy change.

**2. F2 — within-day entry ordinal → per-trade outcome. (MEDIUM–HIGH)**
Newly answerable because decision order is recovered. Uses the ordinal, never the day's
total. Separates genuine intraday deterioration from the contaminated bucket figures
currently in circulation.

**3. F1 — position size → per-trade outcome. (MEDIUM–HIGH)**
Clean, ex-ante, actionable. Report it as underpowered at the effect already observed
(t ≈ 1.4) — it can confirm a large leak and cannot rule out a small one.

Run all three as one pre-registered pass with **three primary endpoints and Holm correction**,
one dollar-denominated outcome, day-clustered inference throughout. Not as three separate
searches.

**Everything else on this list is either blocked, duplicated, leaking, or too underpowered to
change a decision.** In particular: do not re-run entry or exit timing in a new formulation,
do not quote the averaging-down figures, and do not scan regime variables.

**And the honest statement about what these three can deliver:** they can find a behaviour
that is costing money. **None of them — nor any combination of them — can establish that the
trading has an edge.** That question needs roughly 2,357 round trips against the 333 on hand,
or ~3.2 years at the current rate. The only actions that move it are collecting more trades
and obtaining execution timestamps.
