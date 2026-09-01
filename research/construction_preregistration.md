# Pre-registration — Strike choice and exit policy, on his own trades

**Status: PRE-REGISTERED, NOT YET RUN.** Written 2026-08-20.

## 0. Why these two questions are different from the seven nulls

Every prior test in this programme was a **prediction** test — does some observable forecast
the outcome. All seven returned null, and all but one were underpowered.

These are **construction** tests: given that he took exactly these trades at exactly these
seconds, would a different strike or a different exit have done better? **No edge is required
for a construction improvement to be real.**

They are also far better powered, for a specific reason: prior tests compared *different
trades*, so the market move was the dominant variance term. These compare **the same trade
under two constructions** — same second, same direction, same session — so the market move is
common to both arms and cancels in the paired difference.

## 1. Sample

All QQQ/SPY round trips from `research/round_trips.csv` (357 of 443). Single names excluded:
no options entitlement. All are post-2020, so the options floor does not bind. Contracts
failing to return a two-sided quote path are excluded and **counted in a funnel**.

## 2. TEST A — strike choice: his actual strike vs ATM

**Arms.** Same entry second, same direction, same exit second (his actual exit timestamp,
so the hold is identical and only the strike differs).

- **Arm 1 (actual):** his real contract, entered at his real fill price.
- **Arm 2 (ATM):** the listed strike nearest the underlying spot at his entry second, same
  expiry and right. Entered at that contract's **ask** at the same second, exited at its
  **bid** at the same second as his exit.

For fairness, Arm 1 is *also* re-priced ask→bid at the same two timestamps, so both arms
face identical execution assumptions. His realised fill is reported separately as context,
never as an arm.

**Primary outcome:** per-contract return, `exit_bid / entry_ask − 1`. Scale-free, so the
different premium levels are directly comparable.

**Secondary:** dollars at **matched capital deployed** — the ATM position sized to the same
total cost as his actual position, fractional contracts permitted, since the ATM contract
costs roughly 3x more and an equal-contract comparison would not be capital-fair.

**Statistic:** mean paired difference (ATM − actual), day-clustered bootstrap, 10,000 reps.

## 3. TEST B — exit policy, on his actual contracts

Applied to his real contract from his real entry fill. Five policies, **fixed now**:

| policy | rule |
|---|---|
| **P0 actual** | his realised exit (context arm, not counted in the multiplicity family) |
| **P1 target** | exit at the first minute whose **bid ≥ 1.25 × entry fill**; else at 15:45 bid |
| **P2 stop** | exit at the first minute whose **bid ≤ 0.70 × entry fill**; else at 15:45 bid |
| **P3 target+stop** | whichever of P1/P2 triggers first; else 15:45 bid |
| **P4 hold** | bid at 15:45 regardless |

All exits at the **bid**; entry at his actual fill price so the arms differ only in exit.

**The +25% / −30% levels are taken from his own measured payoff distribution** (avg win
+48.1%, avg loss −37.0%, breakeven 43.5%) and are **fixed before running. No stop or target
level is swept.** Sweeping levels across policies is how this project produced its earlier
false positives; if a policy shows promise, level selection becomes a separate pre-registered
question.

**Primary outcome:** per-contract return. **Statistic:** mean paired difference versus P0,
day-clustered bootstrap.

## 4. Multiplicity

**Five primary comparisons in one family:** Test A (1) + Test B's P1–P4 (4). **Holm across
all five.** P0 is the baseline, not a test. Any result significant before Holm but not after
is reported as **not significant** with the raw value shown.

## 5. Power

Unpaired return SD is 52%. The paired SD is expected to be materially lower because the
underlying move cancels, but **I do not know by how much and will not guess**: the realised
paired SD and the achieved MDE are computed from the data and **reported alongside the
result**, whichever way it lands. If the achieved MDE exceeds the observed effect, the result
is reported as underpowered regardless of its p-value.

## 6. Leak audit

| | |
|---|---|
| ATM strike selection | uses underlying spot at the entry second — **known at entry** |
| exit policies | use the forward NBBO path, which is the policy's *mechanism*, not a predictor. Every policy is applied identically to every trade, with **no per-trade selection** |
| his actual exit (P0) | a discretionary choice made with information the mechanical policies lack. This asymmetry **favours P0** and is the point of the comparison |
| outcome selection | none — all trades enter every arm; no winners-only subset at any stage |
| quote validity | `bid > 0`, `ask > 0`, `ask ≥ bid`; 09:31–15:59 only, excluding the 09:30 null row and the 16:00 settlement stub |

## 7. Decision rule, committed now

| result | conclusion |
|---|---|
| ATM paired difference > 0, Holm p < 0.05 | **strike change is supported** — adopt ATM, no edge required |
| any P1–P4 beats P0, Holm p < 0.05 | that mechanical exit beats his discretion |
| P0 beats all of P1–P4 | his discretionary exit management is adding measurable value |
| nothing clears | construction is not where the money is either; report and stop |

**Stated prior.** On the strike test I am genuinely uncertain — the spread saving (1.82% →
~0.7% of premium) and the lower breakeven move favour ATM, while OTM's higher percentage
leverage per unit of move favours the status quo. I decline to predict it. On the exit test I
expect P0 to be hard to beat, because his loss-truncation is already what makes his record
positive at all (breakeven 43.5%, realised 45.8%).

## 8. Prohibited

No level sweeping. No added policies. No alternative exit timestamps for Test A. No
per-symbol or per-year cell promoted to primary. No definition changed after a result is seen.

---

## AMENDMENT 2026-08-20 — Test A deferred, Test B run on cached data

**Cause, not choice.** The ThetaData Terminal fails to start: `Invalid credentials` plus a
401 on version fetch. The API key in `.env` is no longer valid, so the 180 whole-chain
requests Test A requires cannot be made. All 180 failed (788 retries, 0 live calls).

**Test A (strike: ATM vs actual) is DEFERRED**, not cancelled or modified. It needs the ATM
strike's quote path, which only a chain request provides.

**Test B (exit policies) runs now** on per-contract quote paths already cached by the
earlier percentile experiment (266 contracts, full 389-minute bid/ask series each), joined
to the API round trips by (symbol, expiry, right, strike). Sample reported as a funnel.

**Multiplicity is NOT relaxed.** Holm is applied with **family size m = 5**, the number
pre-registered, even though only 4 comparisons run. Correcting across 4 would make the
surviving tests easier to pass purely because a sibling test failed to execute, which is
not a legitimate reason to lower a bar.

Nothing else changes: policies, levels, outcome, and decision rule are as written above.
