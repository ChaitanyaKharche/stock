# Results — the six-line breakout, exactly as specified, is a null on QQQ

**Run 2026-09-21.** `trade_analysis/live_lab/six_lines.py`, QQQ, 2016-01-04 → 2026-09-19.
**2,795 days tried, 2,656 sessions recorded, 138 thin.**

This is the trader's own level set, at his own trade frequency, with his own profit cap.
It supersedes [`breakout_options_results.md`](breakout_options_results.md), which tested
a different set of levels (three prior sessions, premarket merged into market hours) and
is therefore a null about something he does not trade.

## 1. The strategy, as specified

Six lines, rolling one session at a time:

| | source |
|---|---|
| R1 R2 R3 | yesterday's premarket high · yesterday's market high · today's premarket high |
| S1 S2 S3 | yesterday's premarket low · yesterday's market low · today's premarket low |

R1 nearest above, S1 nearest below. Break = a 10-minute RTH bar **closing** beyond a
line. One trade per session, on the **first** break. Entry next bar's open. Exits: +20 bp
(≈ +50% on an ATM option at delta 0.5) | close back inside the broken line | 15:55.

## 2. The verdict

2,438 trades, 218 no-trade sessions (8.2%). Day-clustered bootstrap, 3,000 reps.

| series | mean bp | median bp | 95% CI | p |
|---|---|---|---|---|
| uncapped | **+0.54** | −13.94 | [−1.70, +2.87] | 0.611 |
| capped at +20 bp | **−0.18** | — | [−1.29, +0.88] | 0.764 |

**Null, both ways.** And both means sit **below the 5 bp** an ATM 0DTE needs for spread
and theta, so even a real effect of this size would not pay for the instrument it is
meant to be traded in.

**The cap subtracts again: +0.54 → −0.18, a cost of 0.72 bp per trade.**

## 3. The cap has now subtracted three times, on three different datasets

| measurement | sample | finding |
|---|---|---|
| journal, dollar space | 422 discretionary trades | +25% target lifts win rate to 71.6%, still loses; breakeven needs 75.3% |
| 8-line mechanical | 7,938 signals | cap worth **−1.17 bp** |
| 6-line, his exact set | 2,438 trades | cap worth **−0.72 bp** |

Three independent samples, three different level definitions, one answer. **This is the
most robust finding in the project about the trader's own rules**, and it is a negative
one: capping winners while leaving losers uncapped removes the asymmetry that any
positive expectancy would have to come from.

## 4. Why the level set was not the problem

The 6-line tally explains the null better than the P&L does.

**A break is not a selective event.** 92.0% of sessions break at least one line, and the
median first close-break is **09:40 ET** — the second bar of the session. A condition
that is true almost every day, almost immediately, cannot separate good days from bad
ones.

**Break rates are explained by distance alone.** Normalised for how often each line is
even live (a line already beyond the open is excluded as `pre_broken`):

| | R1 | R2 | R3 | | S1 | S2 | S3 |
|---|---|---|---|---|---|---|---|
| break rate, of live sessions | 55.4% | 42.8% | 38.1% | | 46.7% | 33.3% | 27.0% |

Perfectly monotone in distance from the open, on both sides. That is what *any* six lines
at those distances would produce. Nothing in this table distinguishes "yesterday's
premarket high" from a price level chosen arbitrarily at the same distance.

**Movement was never the constraint.** Median RTH range is **125 bp**, against the ~20 bp
a +50% ATM gain needs. The move is there six times over; the direction is not.

## 5. The long/short split, and why it is not an edge

56% long, long mean **+2.30 bp**, short mean **−1.74 bp**.

QQQ rose over 2016–2026. A rule that goes long more often than short during a decade-long
uptrend collects drift, and that is the most likely explanation for a 4 bp spread with no
per-direction CI reported. It should not be read as the breakout working better upward.
Testing it properly would need the direction split pre-registered as its own hypothesis
against a buy-and-hold benchmark.

## 6. What is now closed, and what is not

**Closed:** the six-line close-based breakout, taken once per session, as a directional
entry on QQQ. Capped or uncapped. This is the trader's specification, run on his levels
at his frequency, and it does not predict direction.

**Not closed, and not tested here:**

- **SPY.** The sweep has not completed there. Given QQQ's result, SPY is confirmatory.
- **Whether these levels beat random lines at the same distances.** §4 suggests they do
  not, but suggestion is not measurement. One run would settle whether the whole family
  is distance-to-a-line rather than which-line, and a null there closes every variant at
  once instead of one at a time.
- **Whatever selects which break he takes.** 92% of sessions break something by 09:40,
  and this rule takes one trade per session against his **2.90 per day**
  (`entry_timing_results.md:17-21`, 357 round trips / 123 days). So this rule trades
  *less* than he does, not more.

  **CORRECTED 2026-09-22.** This bullet originally read "his journal median is 1–2 trades
  entering at 11:39. The mechanical rule and the trader are not doing the same thing."
  Both halves were wrong. The count is 2.90/day, not 1–2. And 11:39 is a *median*, which
  was written up as though it were a cluster — for a two-humped distribution the median
  lands in the gap between the humps, at an hour that may hold almost no trades. The 17
  `entry_too_late` exclusions in the same funnel show entries running late into the
  afternoon. See `breakout_options_results.md` §5 for the full correction.

  What remains true is only that the selection rule has never been written down. The claim
  that the rule and the trader "are not doing the same thing" was inferred from a frequency
  gap that does not exist, and is withdrawn. Every measurement so far still says the entry
  is null and the exits carry his record, so that unwritten selection is the last
  unexamined part of his process — but this document has produced no evidence about its
  shape.

## 7. What is NOT concluded

- Nothing about options. No option price appears in this run; the 20 bp cap is an
  underlying-move proxy for +50% of premium at delta 0.5, and theta only makes it worse.
- Nothing about the touch-based variant as a strategy. It is tallied, not traded.
- Nothing about SPY, or about any 1DTE expression, which remains permanently unmeasurable
  — no archive exists and ThetaData has been `Options: FREE` since 2026-09-08.
