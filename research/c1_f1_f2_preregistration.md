# Pre-registration — Three-Endpoint Pass (C1, F1, F2)

**Status: PRE-REGISTERED, NOT YET RUN.** Written 2026-08-18 after feasibility probing and
before any endpoint was computed. Definitions here are final. Deviations must be appended as
dated amendments, never edited in place.

Follows `remaining_testable_inventory.md`. All three endpoints are **leak detection**, not
edge detection: 333 round trips cannot resolve a $10/contract edge (needs ≈2,357).

---

## 1. Sample construction

Source `trade_analysis/data/robinhood_trades.csv`, parsed with `csv.reader`, rows with
`len(row) != 9` dropped. A round trip is one unique contract `(symbol, expiry, right,
strike)`. Base eligibility: **all fills on a single activity date** and **bought quantity ==
sold quantity**.

| | QQQ+SPY | single names | total |
|---|---|---|---|
| contracts | 276 | 57 | 333 |
| single-session AND balanced | **266** | **52** | **318** |

### Single-name inclusion — decided before running, per endpoint

Probed and established:

- The ThetaData **stock** line **does serve** NVDA/WMT (minute bars returned; `interval=1d`
  is rejected, so daily closes will be aggregated from minute bars). Prior closes and
  trailing sigma are therefore obtainable. No options entitlement is needed by any of these
  three endpoints.
- **But the DTE structure differs fundamentally.** Single names: DTE 0→12, 1→12, 2→5, 3→11,
  4→11, 7→1, i.e. **23% 0DTE**. QQQ+SPY: DTE 0→249, 1→14, 2→1, 3→2, i.e. **94% 0DTE**.
- Premium scale also differs: single-name median BTO **$1.24** vs QQQ/SPY **$0.71**.

**Decision:**

- **C1 excludes single names. n = 266.** A strike distance measured in *daily* sigmas is not
  comparable across a 0-to-7-day horizon — the probability of traversing one sigma differs by
  roughly √8. Making them comparable requires a time-scaling term, and choosing one after
  inspecting the DTE mix is exactly the post-hoc normalisation search that is prohibited.
  **The exclusion is on comparability grounds, not on the results, which have not been seen.**
- **F1 and F2 include single names. n = 318.** Their predictors are quantity and within-day
  ordinal — no market data, no DTE dependence, identical definitions across symbols.
- **F2 requires them.** The ordinal is computed over *all* contracts first-bought that day.
  Dropping single names would misnumber every QQQ/SPY trade on a mixed day.

Multi-session contracts (9 + 5) and the one quantity-unbalanced contract are excluded from
all endpoints: no complete single-session round trip exists.

**Pre-registered data-integrity filter (not a threshold search):** any contract with
`|ln(K / S₀)| > 0.5` is flagged and excluded as a probable split/adjustment artifact. Count
reported.

---

## 2. Pre-registered definitions

### Primary outcome — identical for all three endpoints

```
Y = ( Σ_all fills  Amount ) / quantity          [dollars per contract]
```

`Amount` is the broker's signed cash column, so **Y is net of the ~$0.085/contract
round-trip fees actually paid**. Per-contract rather than per-position, so that **position
size is not mechanically inside the outcome** — this is what makes F1 a test of prediction
rather than of arithmetic.

### C1 predictor — one definition, no alternatives

```
S₀   = underlying regular-session close on the last trading day STRICTLY BEFORE the
       activity date
σ₂₀  = sample SD of the 20 most recent daily log returns, all STRICTLY BEFORE the
       activity date
d    = (K − S₀) / (σ₂₀ · S₀)     for calls
       (S₀ − K) / (σ₂₀ · S₀)     for puts
```

Positive `d` = further out of the money, in units of daily sigma. Daily closes aggregated
from ThetaData stock minute bars, filtered to positive prices, last regular-session minute.

### F1 predictor

```
q = total contracts bought in the round trip     [continuous, as inventoried]
```

Continuous per the inventory. No binning, no thresholds.

### F2 predictor

```
k = rank of this contract's chronologically-FIRST BTO among all option contracts
    first-bought on that activity date, ranked 1 … m
```

Chronology from the reverse-chronological export (established: 2 negative-position
violations in 1,051 fills chronologically, vs 682 in file order, vs ~357 shuffled).
**No clock time is inferred or used anywhere.**

---

## 3. Statistical specification

**Primary models** (OLS point estimates; all inference by bootstrap):

| endpoint | model | reported coefficient |
|---|---|---|
| C1 | `Y ~ d + premium` | slope on `d` |
| F1 | `Y ~ q` | slope on `q` |
| F2 | `Y ~ k` | slope on `k` |

`premium` enters C1 as a covariate because it is a **mechanical** channel, not because it
was found to matter: a further-OTM option is cheaper, and a cheaper option has a smaller
maximum dollar loss, so `d` and `|Y|` are linked by arithmetic independent of any decision
quality. Partialling premium isolates selection from scale. The univariate `Y ~ d` is
reported as robustness, not as primary.

**Inference:** day-clustered bootstrap over **activity dates**, resampled with replacement,
**10,000 replicates**, percentile 95% CI, two-sided bootstrap p against a null slope of 0.
Measured design effect on these data is **1.31–1.43**; the cluster bootstrap handles it
directly. The retracted "9.4x" figure is not used.

**Multiplicity:** **Holm** across exactly the three primary p-values. Sorted ascending,
compared against α/3, α/2, α/1 at α = 0.05. No fourth endpoint is admitted to the family.

**Power, stated in advance.** Dollar SD ≈ $173/contract; at n = 266–318 with DEFF ≈ 1.37,
effective n ≈ 194–232, giving SE ≈ $11–12 and **MDE ≈ ±$31/contract at 80% power**. The
observed size correlation is ≈ −0.083, which at n = 318 implies t ≈ 1.5 — **F1 is
underpowered for the effect already glimpsed** and this is stated before running, not after.
All three endpoints can confirm a large leak and none can rule out a small one.

---

## 4. Leak audit — required before computing

| | C1 (`d`) | F1 (`q`) | F2 (`k`) |
|---|---|---|---|
| **1. Known before the trade** | strike (chosen at entry), prior close, prior-20-day sigma | contracts bought | that this is the k-th contract of the day |
| **2. Known only after** | `Y` | `Y` | `Y` |
| **3. Any part derived from the outcome?** | **No** | **No** | **No** |
| **4. Incorporates future same-day trades?** | **No** — uses no same-day data at all | **No** | **No** for `k` itself. The day's *total* `m` WOULD, and is therefore **not used as a regressor** |
| **5. Mechanically tied to size or premium?** | **Yes, to premium** — handled by including premium as a covariate | Partially — `Y` is per-contract so size is out of the outcome by construction; `q`↔premium correlation is measured and reported | Not by construction; measured and reported |

**No endpoint uses post-trade information. No endpoint is disqualified.** The premium
linkage in C1 and F1 is a *mechanical confound*, not outcome leakage, and is addressed by
pre-registered covariate adjustment rather than by exclusion.

**Explicitly excluded for leakage** (from the inventory, restated so it is not revisited):
the averaging-down comparison, because "averaged down" means "the position was losing when
he added" and the grouping variable is a function of the outcome path; and any
entries-per-day *total* as a regressor, because it is known only at the close.

---

## 5. Robustness checks — specified now, before any result

| | check |
|---|---|
| R1 | F1/F2 recomputed on QQQ+SPY only (n=266), to confirm single names are not driving them |
| R2 | `Y` winsorised at the 1st/99th percentile |
| R3 | percent-return outcome as a secondary scale (`net / cost`), reported for all three |
| R4 | C1 restricted to 0DTE only (excludes the 17 non-0DTE QQQ/SPY contracts) |
| R5 | `Y` gross of fees (`Price × qty × 100` basis) |
| R6 | F2 with **day fixed effects** (within-day demeaned `k`) — see §6 |
| R7 | F1 with premium as a covariate, and `|Y| ~ q` for the exposure channel |

None of these may be promoted to primary.

---

## 6. Interpretation rules fixed in advance

**F1 — exposure versus prediction.** Three quantities decide it, all pre-registered:
`Y ~ q` (prediction), `|Y| ~ q` (dispersion/exposure), and `corr(q, premium)` (proxy). If
the slope on `Y` is indistinguishable from zero while `|Y| ~ q` is positive, the verdict is
**exposure effect only** — bigger bets produce bigger swings in both directions, which is
arithmetic, not skill.

**F2 — association versus predictiveness, and versus cause.** This endpoint can establish
only that *later trades have different outcomes*. It **cannot** establish that later trades
are *more predictive*, which would require a measure of forecast accuracy that this dataset
does not contain. It **cannot** establish causality: a later ordinal may proxy for market
conditions, for the day being busy, or for state that arises after an early loss. The
primary spec omits day fixed effects deliberately, because `k` without them is the honestly
ex-ante predictor; **R6** adds them to decompose within-day progression from between-day
differences. The gap between the two is reported, not resolved in favour of either.

**C1 — the question being asked.** Not "which strike distance makes money." The question is
whether `d`, a variable he controls at entry, carries measurable information about the
realised dollar outcome. A significant negative slope means reaching further out in
volatility-adjusted terms costs him money; that is a leak, not an edge.

---

## 7. Prohibited

No threshold optimisation. No bin searching. No post-hoc subgroups. No indicators. No
execution-time inference of any kind. No future information. No alternative normalisations
for `d`. No promotion of a robustness check or secondary scale to primary. No change to any
definition above once a result has been seen.

## 8. Verdict categories, committed now

Each endpoint resolves to exactly one, judged on the **Holm-adjusted** primary p-value and
the CI:

- **C1:** evidence of useful contract-selection information / no detectable information /
  invalid or underpowered
- **F1:** evidence of useful sizing information / **exposure effect only** / no detectable
  information / invalid or underpowered
- **F2:** evidence of useful sequential information / no detectable information / invalid or
  underpowered

An endpoint whose CI excludes zero **before** Holm but not after is reported as **not
significant**, with the unadjusted value shown.
