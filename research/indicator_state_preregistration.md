# Pre-registration — Do the trader's own declared indicators separate his outcomes?

**Status: PRE-REGISTERED, NOT YET RUN.** Written 2026-08-20 before any endpoint was
computed. Definitions final; deviations appended as dated amendments only.

## 0. The question, and the flaw it corrects

The request was: look at the *successful* trades, find which indicator conditions were met
around entry, and use those as checkboxes. **Selecting on the outcome is fatal** — the
worked demonstration is on file: classifying exits by whether the typed target filled gives
44 trades, +$5,175, win 86.4% against −$3,722 for the rest, which is a tautology because a
target only fills if price rose.

This pre-registration keeps the substance and removes the flaw: **all trades, outcome never
used for selection, out-of-sample validation.**

What makes it worth running at all: his declared stack (MACD 9/17/9 on OHLC4, DMI/ADX
Wilder 14, volume vs 20-EMA, 5m execution with 10m/15m confirmation) has only ever been
evaluated inside a 09:30–09:45 ORB rule — a window containing **5.7%** of his entries. It
has never been evaluated at his real entry timestamps, which are now known.

**Stated prior: under 15%.** C1 null, F1 scale-only, F2 null, percentile null, and a
19-feature ceiling test null (OOS R² −0.078, AUC 0.431). This is the last live version of
"my indicators work, the backtest just tested them at the wrong time."

## 1. Sample

All **443** round trips from `research/round_trips.csv` (2024-09-11 → 2026-02-20, 158 days).
No filtering on outcome, symbol, size, or profitability. Requires stock minute bars for the
underlying; exclusions counted and reported as a funnel. Expected: 357 QQQ/SPY, 86 across
NVDA, TSLA, MSFT, AMD, AMZN, AAPL, RIVN, GOOGL, META, BMY. **Single names are retained** —
they are where the losses are, and dropping them would be selection.

## 2. Timing rule — the discipline this project learned the hard way

Every feature is computed from the **last fully completed bar strictly before the entry
timestamp**. For an entry at 11:44:23 on 5-minute bars, that is the bar spanning
[11:35, 11:40) — not the in-progress bar, not the bar labelled 11:40. Bars are built from
minute bars with `label=left, closed=left` and a bar is eligible only if
`label + width <= entry_time`.

No feature may use any bar at or after the entry second. Verified by assertion in code.

## 3. Features — his settings, no search

Direction sign `s = +1` for a call, `−1` for a put. Known at entry; not an outcome.

| feature | definition |
|---|---|
| `macd_hist_al` | MACD(9,17,9) on (O+H+L+C)/4, EMA oscillator and EMA signal; histogram = macd − signal; × s |
| `di_diff_al` | Wilder 14: (+DI − −DI) × s |
| `adx` | Wilder 14 ADX (unsigned trend strength) |
| `vol_ratio` | bar volume / EMA20(volume) |
| `m15_al` | trailing 15-min underlying move ÷ σ₂₀, × s (the established chase measure) |
| `atr_pct` | Wilder 14 ATR ÷ price (unsigned regime scale) |

σ₂₀ = SD of the 20 most recent daily log closes, all strictly before the entry date.

**Boolean checkboxes** — the literal form of the request:
`gate_macd = macd_hist_al > 0`, `gate_dmi = di_diff_al > 0`, `gate_adx = adx > 20`
(the ADX floor already fixed in the repo), `gate_vol = vol_ratio > 1`,
`gate_mtf` = `gate_macd AND gate_dmi` agree on both 10m and 15m bars,
and **`all_gates`** = all five true.

Primary timeframe is **5-minute** (his execution timeframe). 10m/15m are used only for
`gate_mtf`. No other timeframe is examined.

## 4. Outcome

**Primary: per-trade return** `pct` from `round_trips.csv` (net of real fees, ÷ cost basis).
Return space rather than dollars because median cost basis swings **7.3x** across the sample
($350 → $48), which makes a dollar-mean test misspecified. Dollars reported as a secondary.

## 5. Primary endpoint and decision rule

**Primary: joint out-of-sample predictive power.** Ridge regression of `pct` on the six
continuous features, standardised, `GroupKFold(5)` grouped by **activity date** so no date
spans folds. Statistic: out-of-sample R². Null: **1,000 permutations** of `pct` within the
group structure. λ ∈ {0.1, 1.0, 10.0} all reported; **λ = 1.0 is primary.**

"Do my indicators work" is a joint question, so the joint test is primary and the individual
coefficients are a band, not the headline.

**Secondary, pre-registered:**
- OOS AUC on win/loss, same folds, same permutation null
- **the checkbox test**: mean `pct` when `all_gates` is true vs false, day-clustered
  bootstrap (10,000 reps), plus each individual gate reported with **Holm** across the six
- gate-count dose response: mean `pct` by number of gates met (0–5), reported with n per cell
- dollar-space repeat of the primary

**Decision rule, committed now:**

| result | conclusion |
|---|---|
| OOS R² ≤ 0.05 with permutation p > 0.05 **and** AUC ≤ 0.55 | **no detectable information in his indicator stack.** Stop. No 2018 extension. |
| OOS R² > 0.05 at p < 0.05 **or** AUC > 0.60 at p < 0.05 | information present → consider the historical extension in §7 |
| between | **inconclusive**; report as such, do not resolve by choosing a secondary |

**Power, stated before running.** Return sd ≈ 52.3%, n = 443, measured DEFF ≈ 1.25 →
effective n ≈ 354, SE ≈ 2.8pp, so the **MDE on a mean shift is ≈ 7.8pp** and on the
`all_gates` split ≈ 11pp given the expected cell imbalance. The permutation distribution
supplies the R² threshold empirically. **A null here excludes effects above roughly 8
percentage points of return and nothing smaller** — that will be stated with the result.

## 6. Leak audit

| | |
|---|---|
| known before entry | all six features (completed bars only), the call/put sign, σ₂₀ from prior sessions |
| known only after | `pct`, `net`, hold duration, exit price |
| any feature derived from the outcome? | **No** |
| does grouping leak? | No — folds are grouped by date, so same-day trades never split across train/test |
| mechanically tied to size or premium? | No — all six are market-state; cost basis is not a regressor |

**Explicitly excluded:** hold duration (chosen after seeing the path), anything keyed to the
typed limit price filling, and any winners-only subset.

## 7. If and only if the primary survives

Historical extension is a **second** experiment, not part of this one, and it carries a hard
data limit: **options quotes floor at 2020-01-01**; stock reaches 2016. A pre-2020 test can
therefore only measure whether the *underlying* moved, never whether the *option* paid —
and that proxy previously diverged from real option outcomes by ~10 points of win rate, in
the unfavourable direction. Any extension must be pre-registered separately with that
divergence stated as a known bias.

## 8. Prohibited

No indicator-parameter search (his stated settings only). No threshold tuning, including the
ADX floor. No timeframe search beyond the declared 5/10/15. No winners-only or
profitable-subgroup analysis. No promotion of a secondary or a λ to primary. No change to
any definition above once a result is seen.
