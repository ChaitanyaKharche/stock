# Pre-registration — Live lab, SHARES arm

Written **2026-08-31**, before the arm has traded a single session. Amendments are appended,
never substituted.

---

## 1. Why this arm exists

The options lab has forward-tested 13 setups on ATM/ATM±1 0DTE contracts since 2026-08-28.
Everything in it is at or below zero, and the historical work says that is **structural**: an
ATM 0DTE needs roughly **+5 bp** of underlying move to clear spread and theta, and none of
these signals reliably produce that.

One strategy runs the other way, and it was not being tested at all.

`IntradayMomentumBoundary` is the only item in this project that has ever cleared a
multiplicity correction on a large sample **and** survived honest execution timing:

| | |
|---|---|
| QQQ 2016-2026 | 2,905 trades, **+$3.34/trade**, p = 0.0001, **Holm 0.0043** |
| SPY 2022-2026 | 1,316 trades, **+$2.60/trade**, p = 0.0007, **Holm 0.0087** |
| split-half (QQQ) | +$2.92 (p 0.0040) / +$3.78 (p 0.0004) — **both independently significant** |
| per-year | 10 of 11 positive |
| parameters | **published, untuned** (Zarattini/Aziz/Barbon, SSRN 4824172) |

And on **options** the same signal is +0.70% at p = 0.79 once fills are lagged one minute,
and 0-for-11 in the live lab so far. That is not a contradiction: QQQ's spread is ~0.5 bp
round trip against an option cost floor roughly 450x higher.

**So the strategy with the best evidence in the project was the one strategy not being
forward tested.** This closes that gap. See `research/imb_stress_results.md`.

---

## 2. Relationship to the frozen options arm — total separation

| | options arm | shares arm |
|---|---|---|
| store | `live_lab_data/` | **`live_lab_data/shares/`** |
| config hash | `1f7247d7839d9950` (frozen 2026-08-28) | its own, over shares definitions |
| instrument | ATM / ATM±1 0DTE | shares, $10,000 notional |
| code | `runner.py` | **`shares_runner.py`** (new; imports, modifies nothing) |

The shares arm **cannot add a trade to or remove one from any options setup's count**, and
the options freeze is untouched. `research/forward_test_preregistration.md` §5 still governs
the options arm and is unaffected by anything here.

---

## 3. All thirteen setups run, not only IMB

IMB won a 13-way sweep. Forward-testing **only the winner** would be selection on the
outcome, and its forward p-value would mean nothing. The whole family runs, so **Holm m = 13**
stays valid and IMB's result can be read against its own peers.

`Gap_Fade` and `EMA_9_20_Pullback` remain labelled SLOW and `VWAP_2sigma_Fade` /
`ThreeBarPlay` DEAD, on measured signal *frequency* — never on P&L.

---

## 4. Execution — mirrors `sharewf.replay_day` exactly

The forward record only extends the 10.6-year backtest if it is generated the same way:

1. One open position **per setup** at a time.
2. Open positions are **managed before** new entries are considered (production ordering).
3. Exit precedence: **stop → target → time → bars → trailing → eod**.
4. Stop/target detected on a **closed bar's** high/low; filled at the **live NBBO**.
5. **Long buys the ask and sells the bid; short sells the bid and buys the ask.** No mid.
6. `max_per_day` and `max_per_direction` honoured; every capped signal is written as a SKIP
   so the denominator stays honest.
7. **$10,000 notional per trade**, shares = notional ÷ fill price.
8. A fill is never taken against a quote older than **5 seconds**.
9. Flat by **15:55 ET**, or earlier — see §5.

**No orders are placed. There is no broker path in this file and there will not be one
without a separate, explicit decision.**

---

## 5. Early close

`EOD_FLAT` is 15:55, but on a half day the tape stops at 13:00. A hardcoded flatten would
hold positions through three hours of dead air and stamp the exits 15:55 at 13:00 prices,
inflating `hold_minutes` by ~175 minutes inside a frozen record. This arm therefore treats
**no new bar for 12 minutes, after 12:30 ET** as the close and flattens against the last real
quote. (The options arm still has this defect; it is logged and not yet fixed.)

---

## 6. Checkpoints — and an honest statement about what they can settle

| checkpoint | meaning |
|---|---|
| 50 total trades | plumbing only |
| 100 total | plumbing only |
| **200 per setup** | first look, **underpowered** — see below |
| **666 IMB trades** | the first n at which the backtest effect would be detectable |

Historically IMB fires **1.09 trades per QQQ session** and **1.14 per SPY session**, so
roughly **2.2 per day** across both. That gives:

- 200 IMB trades ≈ **90 sessions ≈ 4.3 months**
- 666 IMB trades ≈ **300 sessions ≈ 14 months**

**The arithmetic of what a forward test can settle, stated before any data arrives.** The
backtest's day-clustered CI half-width is ±1.6 on 2,905 trades. Scaling as 1/√n:

| n | approximate CI half-width | contains zero at +$3.34? |
|---|---|---|
| 200 | ±6.1 | **yes** — cannot resolve |
| 666 | ±3.3 | marginal, single test |
| 2,905 | ±1.6 | the backtest |

**So this arm will not confirm or refute IMB inside a year, and no reading of it before
~666 trades should be treated as evidence about the edge.** That is not a reason to skip it.
What it *does* buy, immediately and cheaply:

1. **Execution truth** — real spreads, real fills, real outages, at real times of day. The
   backtest assumed next-bar NBBO; this measures what actually happens.
2. **Regime evidence** — whether the signal still fires at the historical rate at all.
3. **The plumbing** — clock, restarts, recovery, half-days — proven before any money.

---

## 7. What would make this arm promotable

All four, jointly, exactly as the options arm requires:

1. n ≥ 200 for the setup **and** n ≥ 666 for any claim about the effect size
2. clears Holm within m = 13
3. |effect| exceeds its own achieved MDE
4. first-half / second-half agree in sign

**Failing any one of these means it stays a paper arm.** A good month is not a result.

---

## 8. The clock rule applies here too

Cumulative from the arm's first session. **No session, setup, date, regime or symbol may be
excluded, ever.** A setup performing badly is a result, not a reason to restart its count.
The only legitimate reset is a definition change, which forks a new `setup_id` at zero and
leaves the old history intact.

Missing sessions are reconciled by the dashboard's coverage report, so a day the lab silently
failed to run is visible rather than invisible.

---

## 9. Amendments

*(append only)*
