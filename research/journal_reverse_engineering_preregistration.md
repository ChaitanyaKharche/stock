# Pre-registration — Journal reverse-engineering, full ML toolkit

Written **2026-08-29**, after the feature table was built and its funnel counted, and
**before any model was fit or any outcome inspected**. Nothing below may be edited after a
result is seen. Amendments are appended, never substituted.

This supersedes nothing. `indicator_state_preregistration.md` ran the *linear* version of
Method 1 on 6 features and returned null. This runs the non-linear version, adds per-feature
null importance, adds the tree, adds kernel pattern detection, and adds the selection
question that study never asked.

---

## 0. Why this is not a re-run

| | prior study (2026-08-20) | this study |
|---|---|---|
| features | 6 | **51** |
| model class | Ridge only (linear) | Ridge + RandomForest + HistGradientBoosting |
| importance | none (model-level R² only) | **per-feature permutation importance** |
| null | 1,000 permutations of model R² | **per-feature null-importance distribution** |
| decision tree | not run | **depth-3, with a permuted-best-leaf control** |
| kernel patterns | not run | **Lo–Mamaysky–Wang, 10 patterns** |
| target | his option return only | option return **+ exit-free forward underlying move** |
| question | "which of his trades won?" | that, **plus "what makes a minute one he enters?"** |

The target change matters most. His realised option return embeds his discretionary **exit**,
which is the only effect in this entire project that ever cleared a multiplicity correction
(Test B, Holm p = 0.027). A model trained on that target can learn his exit skill and present
it as entry quality. Every outcome claim here is therefore reported twice: once on the
exit-free forward underlying move, once on the money.

---

## 1. Sample — frozen

| | |
|---|---|
| journal round trips | 443 |
| dropped: entry too early in session for the indicator window | 25 |
| **entries with a complete feature vector** | **418** |
| placebo rows (8 per trade, same session, uniform over eligible minutes) | **3,274** |
| date clusters | **151** |
| symbols | 12 (ETF entries 341, single names 77) |
| window | 2024-09-11 → 2026-02-20 |
| features | **51** |

No exclusion is outcome-related. The sample is now closed: **no row and no feature may be
added, dropped, or redefined for the remainder of this study.**

**Placebo definition.** For each trade, 8 minutes drawn without replacement from that same
session, uniform over indices 5 .. n−26, evaluated 90 seconds after the bar stamp, carrying
the same direction as the real trade. Direction is inherited so that every `*_al` feature is
signed identically for the real and placebo rows — otherwise the classifier could separate
the two on direction bookkeeping alone.

---

## 2. Timing rules — the thing that has voided work here before

1. A 1-minute bar stamped `T` covers `[T, T+60s)`. Its close is knowable only at `T+60s`.
   The anchor bar for an event at second `E` is the last bar with `T + 60 ≤ E`.
2. 5-minute buckets are **close-stamped**; the anchor is the last bucket closing at or before
   `E`. A bucket is emitted only when all five constituent minutes exist.
3. Indicators are seeded from **2 prior sessions**. Nothing resets at 09:30.
4. `fwd_*` and `opt_*` columns use the future **by construction** — they are targets. They
   are never permitted into a feature matrix. The feature list is enumerated explicitly in
   code and every target column is excluded by name, not by convention.

---

## 3. The four questions and their falsifiers

Seed **20260829** throughout. All cross-validation is `GroupKFold(5)` grouped by **date**.
All bootstraps resample **dates**, not rows.

### Q1 — SELECTION. What distinguishes a minute he entered from one he did not?

- Target `is_entry` ∈ {0,1}; n = 418 vs 3,274.
- Metric: **out-of-fold AUC**.
- Null: permute `is_entry` **within each (symbol, date) session**, refit, recompute OOF AUC.
  **500 permutations.** Within-session permutation is the correct null: it holds constant
  everything about the day and asks only whether *the minute he picked* was special.
- **Falsifier: if the best of the three models has OOF AUC ≤ 0.55, or its permutation
  p > 0.05, there is no recoverable entry rule in this feature set.**

### Q2a — OUTCOME, exit-free. Does entry state predict the forward underlying move?

- Targets: `fwd_5m_bp`, `fwd_10m_bp`, `fwd_15m_bp`, `fwd_25m_bp` (direction-aligned, bp),
  on his 418 entries only.
- Metrics: **OOF R²** (regression) and **OOF AUC on the sign**.
- Family size **m = 4**, Holm.
- **Falsifier: OOF R² ≤ 0 or Holm-adjusted p > 0.05 ⇒ no information.**

### Q2b — OUTCOME, the money. Does entry state predict his realised option result?

- Targets: `win` (net > 0) and `opt_ret` (%).
- Metrics: OOF AUC and OOF R².
- Family size **m = 2**, Holm.
- **Falsifier: AUC ≤ 0.55 / R² ≤ 0, or Holm p > 0.05 ⇒ no information.**
- Reported *alongside* Q2a. A result that appears here and not in Q2a is evidence about his
  **exit**, not his entry, and will be labelled that way.

### M4 — Kernel pattern detection (Lo, Mamaysky & Wang 2000)

- Nadaraya–Watson smoothing of the 5-minute close series, Gaussian kernel, bandwidth
  `h = 0.3 × h_CV` (LMW's own choice), rolling window **38 bars**.
- Local extrema from sign changes of the smoothed first difference.
- The **10 canonical LMW patterns**: HS, IHS, BTOP, BBOT, TTOP, TBOT, RTOP, RBOT, DTOP, DBOT.
- Two uses:
  (a) **descriptive** — pattern incidence at his entry minutes vs the unconditional base rate;
  (b) **predictive** — conditional forward-return distribution vs unconditional, on all QQQ
      5-minute bars 2020-2026. Two-sample **Kolmogorov–Smirnov** on the standardised forward
      return, plus a mean difference with a day-clustered bootstrap CI.
- Family size **m = 10**, Holm.
- **Falsifier: a pattern counts only if its KS test clears Holm at 0.05.**

---

## 4. Method 2 — per-feature null importance (the correction that is usually skipped)

For the best model of each question:

1. Compute **permutation importance on held-out folds only** (never in-sample), 20 repeats.
2. Refit the identical pipeline on **200 label-permuted** datasets (same permutation scheme
   as that question's null) and recompute permutation importance each time.
3. For every feature report `z = (real − null_mean) / null_sd` and the empirical
   `p = P(null ≥ real)`.
4. **A feature is declared real only if its empirical p clears Benjamini–Hochberg at
   q = 0.05 across all 51 features.** Raw importance rankings are reported but carry no
   inferential weight on their own.

Declared in advance, per the standard failure mode: **if importance is spread near-evenly
across features, or no feature clears BH, the model learned nothing and the ranking is noise
regardless of how interpretable the top feature looks.**

---

## 5. Method 3 — decision tree, and the control that keeps it honest

- `DecisionTreeClassifier`, **max_depth 3, min_samples_leaf 25**, GroupKFold(5) by date.
- Rules are extracted and printed in if-then form.
- **Control, pre-declared:** the identical fit is repeated on **500 label-permuted** datasets,
  recording each time the **best leaf's out-of-fold rate**. This measures how good a rule a
  depth-3 tree finds *in pure noise at this sample size*.
- **A discovered rule is reported as noise unless its best-leaf metric exceeds the 95th
  percentile of that permuted distribution.** No rule from this section may be traded, and
  none may enter the frozen forward test except as a brand-new `setup_id` starting at zero.

---

## 6. Multiplicity, in full, declared now

| family | m |
|---|---|
| Q1 selection | 1 |
| Q2a exit-free horizons | 4 |
| Q2b money | 2 |
| M4 kernel patterns | 10 |
| **total declared primaries** | **17** |

Holm within family; families reported separately. **The family size will not be reduced if a
member fails to execute** — a member that cannot be computed is reported as a failure and
still counts against m. Feature-level tests use BH at q = 0.05 across all 51 and are
explicitly *secondary*.

---

## 7. What "there is something here" requires

All three, jointly:

1. **Q1 OOF AUC > 0.60** with permutation p < 0.05; and
2. **at least one feature** clearing BH null-importance; and
3. **the depth-3 tree's best leaf above the 95th percentile** of its permuted-null distribution.

Any weaker combination is reported as **no detectable structure**. In particular, a high
importance ranking with no BH survivor, or an interpretable tree rule inside the permuted
band, is a null.

---

## 8. Standing constraints this study cannot override

- **Nothing here modifies the frozen forward test.** `research/forward_test_preregistration.md`
  §5 stands: the clock never resets. A discovery here can only propose a new `setup_id` at
  count zero.
- **No result licenses a trade.** This is measurement.
- Power is what it is: 418 trades, day-clustered, DEFF ≈ 1.3. This can detect a **large**
  structure and nothing subtle. A null here does not prove he has no skill; it bounds how
  much of it is recoverable from these 51 observables.

---

## 9. Amendments

*(append only)*

### A1 — importance model, 2026-08-29, before any importance was computed

Method 2 as written says "the best model of each question." On Q1 the best model is
RandomForest-400 (OOF AUC 0.8469) but its `predict` cost makes 200 null refits with
permutation importance a ~7-hour job. Importance is therefore computed with
HistGradientBoosting (OOF AUC 0.8410, a 0.006 difference), **and its null is computed with
the same model**, so the real-vs-null comparison stays internally valid. A linear
(logistic) importance table is reported alongside as a cross-check. Null refits use
`n_repeats=5` against the real table's 20; fewer repeats makes each null draw noisier and
the null distribution **wider**, which is conservative.

### A2 — Q3, the zone walk-forward, declared 2026-08-29 before it was run

Q1's first diagnostic (univariate AUC per feature, run before any null) showed the
separation is **not** time-of-day (TOD-only AUC 0.556; everything-but-TOD 0.846) and is
instead concentrated in momentum-alignment features. That makes the recovered rule
*describable*, which raises a question Q1 cannot answer: **does occupying that state pay?**

Q1 measures only whether his minutes are distinguishable. A rule can be perfectly recovered
and worth nothing. Q3 is therefore declared now, before execution:

- Take the interquartile **zone** his own 418 entries occupy on the features that discriminate.
- Scan **every** QQQ and SPY minute available (2016-2026 QQQ from `live_lab_data/bars_cache`),
  blind, with no knowledge of his trades.
- Direction is set by the trailing move, exactly as the recovered rule specifies.
- **Execution is lagged one full minute** past the signal bar stamp, on both legs. This is
  the rule that killed `IntradayMomentumBoundary`; it is not optional.
- Endpoint: forward underlying move at 5/10/15/25 minutes, direction-aligned, in bp.
  Family size **m = 4**, Holm. Day-clustered bootstrap, 10,000 replicates, resampling dates.
- **Falsifier: if the Holm-adjusted p exceeds 0.05 at every horizon, or the 95% CI upper
  bound sits below +5 bp (the ~0.05% an ATM 0DTE needs to clear spread and theta), the
  recovered rule has no tradeable edge and no amount of further feature work will give it
  one.**

Q3 adds 4 primaries. **Total declared primaries for this study is therefore 21, not 17.**
This count will not be reduced.

### A3 — Q4, the EXIT study, declared 2026-08-29 before it was run

Everything above reverse-engineers the **entry**. That is the wrong end of this particular
trader. Test B (`construction_results.md`) found that a mechanical +25% profit target loses
to his discretion by 8.97 percentage points at Holm p = 0.027 — **the only comparison in this
entire project that has ever cleared a multiplicity correction.** If there is skill in this
journal, the evidence says it is in the exit, and no study here has ever tried to recover it.

The local ThetaData gateway serves `/v3/option/history/quote` at 1-minute resolution for his
actual contracts, so his realised P&L path can be reconstructed rather than modelled.

**Q4a — economic. Does his discretionary exit beat mechanical alternatives on his own trades?**

Paired per trade, entry at the ASK, exit at the BID, fee $0.0404/contract/side, day-clustered
bootstrap of 10,000 replicates resampling dates. Counterfactual grid, **fixed now**:

| family | members |
|---|---|
| fixed profit target | +25%, +50%, +75%, +100% |
| fixed stop | −25%, −50% |
| pure time exit | 5, 10, 15, 25, 45 minutes |
| trailing give-back from peak | 20%, 30%, 50% |

**14 counterfactuals, Holm m = 14.** Where a target or stop never triggers, the position
falls through to **his own exit time** — this isolates the effect of the rule rather than
confounding it with a different terminal. A secondary pass falls through to 15:55 instead.

- **Falsifier: if no counterfactual differs from his realised result at Holm p < 0.05, his
  exit is statistically indistinguishable from mechanical alternatives and the Test B result
  does not generalise beyond the +25% target.**

**Q4b — mechanisability. Is his exit timing a function of observables?**

Per-minute hazard framing over each holding period: label 1 at the minute he exited, 0 at
every other minute held. Features: minutes elapsed, unrealised option return, drawdown from
the running peak bid, run-up from the running trough, underlying move since entry (aligned),
and the same momentum panel used in Q1. Null: permute the exit minute **within each trade**,
which holds the trade constant and asks only whether *that* minute was special.

- **Falsifier: OOF AUC ≤ 0.55 or permutation p > 0.05 ⇒ his exit timing is not recoverable
  from these observables, and therefore cannot be mechanised from them — whatever Q4a says
  about its value.**

Q4 adds **15 primaries** (14 + 1). **Total declared primaries for this study is therefore 36.**
This count will not be reduced. A member that cannot be computed is reported as a failure and
still counts against m.

### A4 — Q3b, the full-model zone, declared 2026-08-29 before it was run

Q3 tests a **4-feature** zone taken from the quartiles of his entries. It fires 9.06 times
per session against his own ~2.8 trades per day, so it is strictly broader than what he
actually does: he takes roughly one in three of its opportunities. A fair reading of a
negative Q3 therefore has an obvious objection — *the zone is not him, it is a coarse shadow
of him, and his sub-selection is the part that matters.*

Q3b removes that objection by using the **entire Q1 classifier** — all 45 features, the model
that scores OOF AUC 0.85 at identifying his minutes — as the signal.

- Fit the Q1 model on all 418 entries and 3,274 placebos.
- Score **every QQQ minute from 2016-01-04 to 2024-09-10**, i.e. strictly before his first
  journal trade. That window is out-of-sample in time by construction; no fold logic is
  needed and none is used.
- Rank minutes by P(he would enter here) and take the **top decile** and **top centile**.
- Same one-minute-lagged fills on both legs, same 4 horizons, same day-clustered bootstrap.
- Reported separately and explicitly labelled as in-period: the same scoring over
  2024-09-11 → 2026-08-27.

- **Falsifier: if the top-centile 95% CI upper bound is below +5 bp at every horizon, then
  even a model that reproduces his selection at AUC 0.85 cannot produce a tradeable edge, and
  the failure is in the strategy rather than in the fidelity of the reconstruction.**

Q3b adds 4 primaries (the top-centile arm; the top-decile arm is secondary).
**Total declared primaries is therefore 40.** This count will not be reduced.
