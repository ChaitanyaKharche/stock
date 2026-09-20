# Results — the 8-line breakout is a null on QQQ, and the profit cap subtracts value

**Run 2026-09-19.** `trade_analysis/live_lab/breakout_sweep.py`, QQQ, 2016-01-04 →
2026-09-17. **2,794 days tried, 2,657 usable, 137 thin, 8,843 signals over 2,600
sessions.** Frozen by [`breakout_options_preregistration.md`](breakout_options_preregistration.md).

This answers **claim A** (does the breakout predict continuation?) and **claim C** (does
the profit cap add anything?) on the underlying. Claim D (1DTE) is permanently
unmeasurable and stays that way; §1 of the pre-registration records why.

## 0. CORRECTION — this tested the wrong level set

**Added 2026-09-20, after the trader specified the levels precisely.** His set is:

    yesterday's PREMARKET high/low  +  yesterday's MARKET-HOURS high/low
                                    +  today's PREMARKET high/low          = 6

**One** prior day, with premarket and market hours kept as SEPARATE levels. What was
tested below is three prior sessions, each reduced to a single high/low over the whole
04:00-20:00 span. That merge deletes yesterday's premarket lines whenever they sit inside
yesterday's RTH range, which is most days.

So this null is a null about levels he does not draw. Everything below stands as a
measurement of the 8-line set and of nothing else. `six_lines.py` implements the correct
set; `six_lines_test.py` pins it, including the case that caused the error.

## 1. The verdict

Pre-switch signals only (7,938 of 8,843; the 905 post-13:00 signals would need 1DTE and
are excluded, not dropped). Day-clustered bootstrap, 3,000 reps, seed 20260828.

| series | mean bp | median bp | 95% CI | p |
|---|---|---|---|---|
| **uncapped** (hold to failure or 15:55) | **+1.44** | −13.79 | [−0.97, +3.91] | 0.259 |
| **capped at +20 bp** (the stated strategy) | **+0.27** | +20.00 | [−0.88, +1.46] | 0.640 |

**Claim A fails.** No directional drift after the signal, over 2,657 sessions.

**Claim C fails, and fails in the informative direction: the cap is worth −1.17 bp per
signal.** It does not merely fail to help; it actively subtracts.

Supporting figures: 64.8% of signals touch the cap first, win rate 65.9%, median hold
**10 minutes**. Only 37.1% of uncapped signals clear the 5 bp an ATM 0DTE needs for
spread and theta.

## 2. Why the cap subtracts, and why that is the second time this has been measured

The uncapped distribution is violently two-tailed — worst signal **−628.4 bp**, best
**+479.0 bp**. Capping at +20 bp truncates the entire right tail and leaves the left tail
untouched. Every large winner is converted into a +20, and nothing is done about the
−628s.

**This is the same finding the journal work produced, from independent data.**
`PROJECT_REPORT.md` records that a +25% target raises his win rate to 71.6% and *still*
loses money because that structure breaks even at 75.3%, and concludes: *"Capping winners
destroys the asymmetry that makes his record positive."*

That was 422 discretionary trades in dollar space. This is 7,938 mechanical signals in
underlying-move space, over a different instrument and a different decade. Same
conclusion. **Two independent measurements of one effect is the strongest statement this
project has about the trader's own exits**, and both say the same thing: the win rate
goes up, the expectancy goes down.

## 3. A FLAW IN THE PRE-REGISTRATION'S OWN LOGIC, recorded rather than quietly used

§8 states *"B survives only if A survives."* **That is too strong for a long-premium
expression, and it should not be relied on.**

A long option has a floor: it cannot lose more than the premium. In this sample the worst
signals are −185 to −628 bp, and at delta ≈ 0.5 a total loss of premium is only about
−40 bp of underlying-equivalent. So **the underlying series overstates the losses of a
long-call/put expression by a large factor** — the option truncates the left tail and the
underlying does not.

A convex payoff can therefore be positive on an underlying process whose mean move is
zero. Arm A's null does not logically kill the option arm.

What it does do is remove the reason to expect the option arm to work, and the reason is
economic rather than statistical: **you pay for that convexity.** The price is the
variance risk premium, and this project has already measured it —
[`vrp_cost_model_results.md`](vrp_cost_model_results.md) found implied exceeding realised
with a gross short-straddle edge of +$0.0205/trade at `t=+7.52`. Buying options against
a zero-drift signal means paying that premium for nothing.

So the honest statement is: **a signal with no directional drift, expressed as a long
call or put, pays the VRP and is expected to lose** — not "A is null, therefore B is
dead."

## 4. The thing the data does say, which is not what the strategy assumed

| | |
|---|---|
| directional drift | **+1.44 bp, CI spans zero** |
| median MFE in the signalled direction | **+34.26 bp** |
| share reaching 20 bp at some point | **64.8%** |

The move is there. The *direction* is not. A signal whose favourable excursion is 34 bp
while its terminal drift is indistinguishable from zero is describing **magnitude, not
sign**.

Which is the same thing the VRP arm found by accident: `_calculate_momentum_score` was a
pure magnitude that the code was forcing to emit a directional CALL/PUT, and
[`vrp_preregistration.md`](vrp_preregistration.md) §0 notes *"the part of this system
that was measuring something real was measuring size, not sign."* Three arms of this
project have now arrived at that sentence independently.

**The expression implied by a magnitude signal is a straddle, not a call or a put.** That
is a different hypothesis, it needs its own pre-registration, and it must not be fitted
onto this one — reinterpreting a null by changing the instrument after seeing the result
is precisely the garden of forking paths the MOMO_CHASE sweep spent 2,822,400 cells
demonstrating. It is also measurable: SPY 0DTE, 769 sessions, is the one option archive
that exists.

## 5. A discrepancy between the spec and the trader's behaviour

**8,843 signals over 2,600 sessions is 3.4 per day.** His journal shows a median of
roughly 1–2 trades, entering around 11:39.

So the mechanical rule fires about three times more often than he does, and whatever
selects *which* breakout he takes is not written down anywhere. Given that the entry
signal is null and the exits are where the journal says his edge lives, that unwritten
selection is the remaining unexplained part of his process — and it is not in this
experiment.

## 6. Two defects found and fixed during the run, both recorded

**(a) The profit cap was not implemented.** The first version of `evaluate_signal` coded
exits 2 and 3 and omitted exit 1 — the cap the strategy is built around — and reported a
−13.79 bp median as though that were the strategy. It measured "hold until the breakout
fails", nearly the opposite. Found because the trader pushed back on the result.

**(b) Fixing (a) introduced lookahead.** The failed-breakout branch read
`if cap_px is not None: break`, so when the breakout failed before the cap was touched
the loop carried on and could register a cap fill on a later bar — after the position had
already exited. Caught because two runs over identical data disagreed: median MFE
34.26 → 40.70 bp, cap-hit 64.8% → 74.5%. That inflated run reported **+4.98 bp,
CI [3.82, 6.14], p=0.0003**, which is **VOID** and appears here only so the record shows
it was published and withdrawn.

Both are pinned by tests in `breakout_levels_test.py`, each verified to fail against the
defective version.

## 7. What is NOT concluded

- Nothing about SPY. The sweep has not completed there. Per §8 both symbols must survive
  A, and QQQ already has not, so SPY is confirmatory rather than decisive.
- Nothing about any option P&L. There is no option price anywhere in this run.
- Nothing about QQQ options or any 1DTE, now or ever — no archive exists and
  `Options: FREE` since 2026-09-08.
- Nothing about the six-line variant (secondary S1), which was not run.
