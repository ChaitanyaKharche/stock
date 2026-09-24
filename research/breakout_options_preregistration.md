# Pre-registration — the 6-line breakout, ATM options, capped, with a 0DTE→1DTE switch

**Status: PRE-REGISTERED, PARTIALLY UNMEASURABLE.** Written 2026-09-18, frozen before any
number is computed. §1 states which arms can be run at all; that inventory is the most
important section and it is first for that reason.

The strategy is the trader's, described on 2026-09-17 and 2026-09-18 from his own chart
setup. This makes it falsifiable without changing it.

## 0. The strategy as stated

> Mark the previous 3 days' highs and lows — premarket and market hours — 6 lines.
> Take the breakout. Buy ATM naked calls or puts. Cap profit at 25–50% of the premium.
> If the breakout happens after 1–2pm ET and the 0DTE has lost most of its premium, use
> the next day's expiry instead, so delta doesn't work against us.

Four separable claims, and they fail or survive independently:

| # | claim | what would falsify it |
|---|---|---|
| A | the 6-line breakout predicts directional continuation | underlying move after the signal is indistinguishable from zero |
| B | the move is big enough to pay for an ATM option | move < the measured spread + theta floor |
| C | capping at 25–50% of premium beats not capping | capped P&L < uncapped, in dollars |
| D | switching to 1DTE after 13:00 beats staying 0DTE | — |

## 1. WHAT CAN AND CANNOT BE MEASURED. READ THIS BEFORE ANYTHING ELSE.

| arm | data | status |
|---|---|---|
| **A — signal, QQQ + SPY underlying** | `stock_ohlc_1m/`, `stock_quote_1m/`, 2016–2026 | **RUNNABLE** |
| **B/C — SPY 0DTE, ATM, capped** | `option_quote_1m_0dte/SPY/`, 769 sessions 2020–2024 | **RUNNABLE** |
| **B/C — QQQ options, any expiry** | — | **IMPOSSIBLE. No QQQ option archive exists.** |
| **D — 1DTE, any symbol** | — | **IMPOSSIBLE. The archive is 0DTE by construction.** |

The ThetaData subscription lapsed to `Options: FREE` on or before 2026-09-08, so no
further option history can be pulled at any price this project will pay. `hpc/README.md`
records the inventory. **Claim D is therefore pre-registered and permanently
unanswerable with available data, and any statement about it in future must say so.**

This is not a gap to be worked around. A 1DTE result inferred from 0DTE quotes plus an
assumed vol surface would be a model output presented as a measurement, which is the
single failure mode this project has spent nine nulls learning to avoid.

**What §2–§7 therefore test: A and B on both symbols, C on SPY 0DTE only.** If A fails,
D is moot — no expiry rescues a signal with no move behind it. That ordering is the
reason this design is worth running despite D being blocked.

## 2. The levels, frozen

**Primary level set: EIGHT lines, not six.**

- prior 3 completed sessions, each contributing its **high** and its **low** over the
  full extended session 04:00–20:00 ET → 6 lines;
- **today's premarket high and low**, 04:00–09:29 ET → 2 lines.

**Deviation from the stated spec, named explicitly.** He said "6 lines total". Both trades
he described, however, reference premarket: 2026-09-17's no-trade was about today's
premarket range containing RTH, and 2026-09-18's breakout was of *today's* premarket high.
Excluding today's premarket would exclude the example that generated the hypothesis. The
6-line version (prior days only) is pre-registered as **secondary S1** and is a one-flag
change if this reading is wrong. **This is the one thing to confirm before the run.**

Levels use **completed prior sessions only**. Today's premarket is complete at 09:30, so
it is available to every RTH decision and introduces no lookahead.

## 3. The signal, frozen

Identical mechanics to `orb_veto_preregistration.md`, deliberately — two hypotheses from
the same trader read off the same chart should not disagree about what a break is.

| element | value | source |
|---|---|---|
| bar interval | **10 minutes**, aligned to 09:30 | the timeframe he reads |
| trigger | a 10-min bar **CLOSES** beyond a level | §2 of the veto prereg: a rule one tick can flip is not a rule |
| buffer | **0.05% of price** | `levels_test.BUFFER`, his `VWAP_Reclaim` convention |
| direction | long above, short below | |
| level used | the **nearest un-breached** level in the break direction | |
| one signal per level per day | a level already broken cannot re-trigger | |
| window | 09:40–15:30 ET | after the opening range; before the flatten |
| fill | **next bar**, at the **ask** | always long premium, so always pay the ask |

## 4. The expression, frozen

- **ATM only**, the strike nearest spot at the entry bar. No adjacent strikes; that is a
  separate hypothesis and the frozen options lab already tests ATM±1.
- **0DTE before the switch, 1DTE after.** Primary switch time **13:00 ET** — the earlier
  end of his stated 1–2pm, which is the conservative choice about theta. 14:00 is
  secondary **S2**.
- **1DTE cannot be priced (§1).** Signals after the switch are therefore **recorded and
  excluded from the primary P&L**, with their count and their underlying move reported
  separately. They are not silently dropped and they are not filled at an invented price.

## 5. The exits, frozen

1. **Profit cap: +50% of premium paid.** PRIMARY.
2. **Failed breakout: a 10-min close back inside the level.** [R] — chosen here, not from
   a source, because he did not state a stop. It is the exit most faithful to the logic:
   the thesis was the break, so the thesis dying is the exit. Symmetric with §3's trigger.
3. **15:55 ET flatten**, matching the live lab.

**Why +50% and not +25%, stated before running.** The journal work already measured the
low end of his range and it loses: a +25% target raises his win rate to 71.6% and *still*
loses money, because that structure breaks even at 75.3%. His realised structure — avg
win +48.1%, avg loss −37.0% — breaks even at 43.5% against a 45.8% hit rate. So **+25% is
a pre-registered expected LOSS, not a candidate**, and picking it as primary would be
choosing a parameter already known to fail. The full 25/30/40/50% ladder is secondary
**S3**, reported with correction, to measure the shape of the cap's effect rather than to
find a winner.

**Uncapped is the control.** The same trades held to exit 2 or 3 only. Claim C is about
whether the cap *adds* anything, and without the control it cannot be evaluated —
"capping made money" is not evidence for capping.

## 6. Costs, and the number claim B must clear

Every fill uses the **real quoted bid/ask at the fill bar**, never the mid. On the VRP
arm the mid-fill assumption was the difference between `t=+7.52` and `t=−5.38` on
identical trades — between a strategy and its opposite.

The bar claim B must clear, from `shares_runner.py`: **an ATM 0DTE needs roughly +5 bp of
underlying move to clear spread and theta.** So arm A reports, for every signal, the
underlying move from fill to each exit, in basis points, and the fraction of signals
clearing 5 bp. **If most signals do not clear 5 bp, claim B is dead on the underlying
alone and no option result is needed.**

## 7. Inference, power, and the tail

- Unit is the **session**, not the trade. Day-clustered stationary bootstrap, 3,000 reps,
  seed 20260828, reusing `sharewf.boot`.
- Arm A: ~2,650 sessions per symbol, 2016–2026.
- Arm B/C: **769 SPY sessions, 2020–2024**, and only the subset with a pre-switch signal.
  At a plausible 30–50% signal rate that is ~230–385 events. **This is thin for a
  long-premium payoff**, whose distribution is right-skewed by construction: capped
  upside and −100% floor. State the MDE with the result; do not report a point estimate
  without it.
- **Tail reporting is mandatory, per the pattern in every prior result here.** IMB carries
  73.9% of its P&L in the top 1% of trades; the VRP short straddle carries 94% of its
  loss in the worst 1%. Report the top-1% and bottom-1% P&L share before any verdict. A
  capped strategy that survives only on its top 1% has not been shown to work at n≈300.

## 8. Decision rule, committed now

**A survives** if mean underlying move in the signal direction, over the holding window,
has a day-clustered 95% CI excluding zero, in the signalled direction, on **both** QQQ
and SPY. One symbol is not two observations.

**B survives** only if A survives **and** the median signal clears the 5 bp floor.

**C survives** only if capped net P&L exceeds uncapped net P&L, in **dollars**, with the
CI on the difference excluding zero. Return space is prohibited as the headline here: the
journal work found the sign flips between return and dollar space (8.97pp at Holm
p=0.027 in return space, +$4.22 at p=0.487 in dollars), and dollars are what is traded.

**D is not decided.** It is recorded as blocked.

**Any of these failing is a result and gets written up with the same care as a success.**

## 9. Multiplicity

Primary is the single specification in §2–§5. Secondary, Holm-corrected across the whole
family and **never promoted**: S1 six-line set · S2 14:00 switch · S3 the cap ladder ·
lookback ∈ {2, 5} days · interval ∈ {5, 15} min.

That grid is 2 × 2 × 4 × 3 × 3 = 144 cells before symbols. **The MOMO_CHASE sweep already
demonstrated that 2,822,400 cells of a plausible-looking form return a null at p=0.8227.**
A secondary cell beating the primary is evidence about the size of the search space.

## 10. Prohibited

- Reporting any 1DTE number, including one inferred from a vol model.
- Any QQQ option result. The archive does not exist.
- Mid-price fills.
- Dropping post-switch signals silently rather than reporting them as excluded.
- Reporting capped results without the uncapped control.
- Changing the cap, switch time or level set after seeing §8 and calling it the same
  experiment.
- Touching `live_lab_data/`. This is a backtest; the forward test is frozen.
