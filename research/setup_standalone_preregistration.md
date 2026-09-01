# Pre-registration — Do the ten setups work as standalone rules?

**Status: PRE-REGISTERED, NOT YET RUN.** Written 2026-08-20.

## 0. What changes from the previous test

`setup_crosscheck_results.md` asked whether setup-matching *at his entries* predicted *his*
outcomes: coverage saturated at 91.4%, 0 of 7 cleared Holm. This asks a different question
with far more power: **do these setups generate a directional edge on their own, on every
session, independent of him?**

## 1. Setup definitions — inherited verbatim, not re-specified

All ten entry conditions are taken **unchanged** from
`setup_crosscheck_preregistration.md` §2. They were fixed from external sources before any
result was seen and may not be redefined here. Re-specifying them now would reintroduce the
researcher discretion that pre-registration exists to remove.

Direction `s ∈ {+1, −1}` is now generated **by the setup itself** rather than taken from his
call/put choice: each setup fires long or short according to its own stated logic (e.g. VWAP
side fires long above VWAP, short below).

## 2. Sample

QQQ and SPY, **2020-01-01 → 2026-02-20**. This window is chosen to match the **options data
floor (2020-01-01)** so that any survivor can be priced in stage 2 on the same period — not
because of anything observed in the data. Stock minute bars reach 2016; the extra years are
deliberately not used, to avoid a regime-mismatch objection at the pricing stage.

**One signal per setup, per symbol, per session: the first qualifying bar.** This matches how
the ORB work was done and avoids an unbounded within-day multiplicity. Entry window
09:45–15:30 ET so that a 25-minute hold completes inside the session.

## 3. The exit rule — one, fixed, no grid

**Exit exactly 25 minutes after entry.** Taken from his measured median hold (25.4 min,
`round_trips.csv`), fixed before running.

**No stop, no target, no trailing rule, no alternative durations are tested.** Sweeping exits
across ten setups is thousands of specifications and would guarantee a winner. If a setup
shows drift at a fixed horizon, exit engineering becomes a separate pre-registered question;
if it shows none, no exit rule can rescue it.

## 4. Outcome

**Signed forward return in sigma units:**

```
r = s · (P(t+25min) / P(t) − 1) / sigma20
```

`sigma20` = SD of the 20 most recent daily log closes, strictly before the session. Sigma
normalisation pools QQQ and SPY and the 2020 and 2024 volatility regimes on one scale.

Entry price is the close of the first bar at or after the signal bar's completion — decision
time is the bar's close, never its label.

## 5. Primary endpoint and decision rule

**Primary:** per-setup mean signed forward return, **day-clustered bootstrap** (10,000 reps,
clustering on date across both symbols, since QQQ and SPY fire together on ~99% of days),
**Holm across all ten setups**.

**Advancement threshold, committed now.** A setup proceeds to stage 2 (real option NBBO
pricing) only if **both**:
- mean signed forward return **≥ 0.05σ**, and
- Holm-adjusted p **< 0.05**

0.05σ is a deliberately low bar. For QQQ (σ ≈ 1.2% daily) it is ≈ 0.06% of drift, against
roughly **0.20%** of favourable travel needed for a 25% option gain. A setup below 0.05σ
cannot pay for premium under any exit rule, so the screen is generous by design.

| result | conclusion |
|---|---|
| no setup clears both | **stop.** None of the ten carries directional drift; no option pricing is authorised. |
| one or more clears | pre-register stage 2 separately and price those setups only |

**The screen can only kill.** Clearing it establishes a *necessary* condition, never a
sufficient one: the MFE-proxy diverged from real option outcomes by ~10 points of win rate
earlier in this project, in the unfavourable direction.

## 6. Power

~1,540 sessions × 2 symbols. At observed signal rates a typical setup should produce
1,000–3,000 signals. With SD of a 25-minute signed move ≈ 0.30σ and a day-level design effect
≈ 1.5, SE ≈ 0.008–0.013σ, giving an **MDE near 0.03σ** — comfortably below the 0.05σ
threshold. **This test is genuinely well-powered**, unlike everything run on his 443 trades.

A null here therefore means something: it excludes drift above ~0.03σ, which is below the
level that could pay for premium.

## 7. Leak audit

| | |
|---|---|
| known at entry | VWAP, opening range, prior-day levels, EMAs, RVOL, session extremes — all from completed bars strictly before the entry bar's close |
| known only after | the 25-minute forward return |
| decision timing | entry at the **close** of the signal bar, never its label — the defect that voided this project's earlier work |
| survivorship | none: every session in the window is evaluated, no symbol or date filtering |
| holidays / zero-fills | sessions with fewer than 300 positive-price minutes are dropped and counted |

## 8. Prohibited

No exit-rule sweep. No alternative hold durations. No threshold tuning on any entry
condition. No added or redefined setups. No symbol or period selection after seeing results.
No promotion of a per-symbol or per-year cell to primary.

**Stated prior: under 10%.** ORB continuation is one of these ten and already failed
decisively on real option pricing (−9.86%/trade, t = −3.53). Seven consecutive pre-registered
nulls precede this.
