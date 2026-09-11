# Arm B results — conditioning the measured straddle on HAR-IV

*Run 2026-09-11 against `research/conditional_0dte_preregistration.md`, committed before
any number below existed. Validation block (2023) only. 2024 sealed.*

## Result

**Arm B clears all five pre-registered gate conditions, and is reported FRAGILE on two
pre-declared criteria.** Both halves of that sentence are load-bearing.

| | per trade | session-mean | session-t | win rate | abs spread cost |
|---|---|---|---|---|---|
| **SELECTED** (top-tercile expected VRP) | **+$0.01002** | +$0.04541 | **+2.05** | 56.6% | 0.03089 |
| all 2023 origins (unconditional) | −$0.00797 | −$0.00758 | −1.81 | 63.4% | 0.02460 |
| rejected (other two terciles) | −$0.00941 | −$0.00899 | −1.49 | 63.9% | 0.02410 |

Paired, per session: **+$0.06297 at t = +3.16** over 73 sessions. 1,113 selected origins of
14,931 (7.5%), 73 sessions of 250.

| gate | value | |
|---|---|---|
| net mean per trade > 0 | +0.01002 | PASS |
| session-clustered t > 2.0 | +2.05 | PASS |
| paired vs unconditional, \|t\| > 1.96 | +3.16 | PASS |
| ≥ 50 sessions | 73 | PASS |
| ≥ 1000 trades | 1113 | PASS |

## Why it is fragile anyway

**1. One session removes significance.** t decays 2.05 → 1.83 → 1.67 → 1.51 as the best
sessions are dropped one at a time, and is under 1.0 by k = 6. The sign is robust; the
*magnitude* is marginal.

**2. The tail is 16× the median win, in a year with no crisis.** Worst 0.1% of selected
trades is −2.298 against a median win of +0.140. Worst single session −12.37. And 2023's
median `iv_var_atm` is **0.42× the train block's** — this is a calm sample, and the rule
selects the highest-volatility origins within it. The unconditional version of this trade
had its **worst 1% of trades account for 94% of net loss**; that pathology is not absent
here, it is merely unsampled.

This is short gamma. It shows a rising curve and a decent hit rate for months, then erases
it. Nothing in a 2023-only test can speak to that.

## What is NOT wrong with it

Three things the pre-registration told me to suspect, which the measurement clears:

- **Not a cost advantage.** I flagged this first as "selected origins are 35% cheaper to
  trade (105 vs 141 bp)" — and that was **backwards**. Relative spread is lower only
  because the straddle mid is larger on high-vol days. In absolute terms selected origins
  pay **28% MORE** spread (0.03089 vs 0.02410). Only absolute cost moves P&L, so the edge
  cannot be a cost artifact. The gross edge is 2.8× larger (+0.04091 vs +0.01468) and that
  is what carries it.
- **Not tail-concentrated.** Top 5% of sessions hold **19.5%** of gross flow — unlike every
  other candidate in this project. net/gross +0.105.
- **Broad, not a handful of days.** **51 of 73 sessions profitable = 69.9%, sign test
  p = 0.0009.** Most selected sessions make money.

## What it actually is, economically

The rule selects origins where implied variance is **3.35×** the rejected median. Stripped
of the machinery: **sell 0DTE straddles when implied volatility is high, skip when it
isn't.** In 2023 that fires on 7.5% of origins and nets a cent per trade after real bid-ask.

That is a recognisable trade with a recognisable failure mode, and the 22%-better HAR-IV
forecast is doing less work than it looks like — most of the selection is just "IV is high."
A useful follow-up, **not authorised by the current pre-registration**, would be whether
`iv_var_atm` alone selects the same origins. If it does, the forecast adds nothing and the
result is simpler than it appears.

## Two errors in my own tooling, both corrected here

**The >100% concentration ratio, for the second time in one day.** The first version divided
the top-5% session sum by the NET total and printed **185.3%** — the identical defect found
and fixed in `dm_concentration.py` hours earlier. Writing it the broken way again, in the
file auditing the only live positive in the programme, is the best possible argument for the
ratio being taken against gross flow. It now is.

**A 4.5× overstatement from a mislabelled metric.** `session_t` returns the mean of session
means; the gate names "mean net P&L per trade". I reported the former under the latter's
label, so the first run showed **+$0.04541/trade** when the per-trade figure is
**+$0.01002**. They differ because sessions hold unequal numbers of selected origins. The
session mean is the right basis for the *t* (the session is the independent unit); the
per-trade mean is the right thing to *gate*. Both are now printed side by side, and the
gate reads the per-trade one.

Neither error changed the gate outcome. Both changed the headline number, one of them by a
factor of 4.5.

## Status against §4 of the pre-registration

§4 reads "Arm A reproduces **and** Arm B passes" as the branch that would make the
conditional story real on two instruments. **Arm A has not been run.** So this is not that
branch, and Arm B alone supports no claim about the published effect. The correct next step
is the replication, and per §4 a failure there is about our code until proven otherwise.

Also unchanged by this result, explicitly: the forecasting arm's §6 gate (target now
HAR-IV-TOD at 0.330489), and the seal on 2024.
