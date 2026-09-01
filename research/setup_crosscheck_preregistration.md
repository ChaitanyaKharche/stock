# Pre-registration — Do his entries coincide with any standard intraday setup?

**Status: PRE-REGISTERED, NOT YET RUN.** Written 2026-08-20. The setup list below was fixed
from external sources *before* any coverage or outcome number was computed.

## 0. The flaw being corrected

The request was "find all setups that are profitable and cross-check — at least some setup
should work." Searching a menu until one clears is guaranteed to succeed spuriously: at
n=418 with ~8pp of return noise, roughly 1 in 20 arbitrary setups clears p<0.05 by chance.
"At least one should work" is the expectation that turns a search into a false positive.

Two corrections. **(1) The list is fixed in advance from an external source**, so it is not
my choice and cannot be extended after seeing results. **(2) Every setup on the list is
reported whether it works or not**, with multiplicity correction across the full count.

## 1. The primary endpoint is COVERAGE, not profitability

This is the important design decision. The outcome test is underpowered (below); the
**coverage** test is not — it is a proportion at n=418, SE ≈ 2.4pp.

And coverage answers the real question. If his entries satisfy most setups at high rates,
then "his trade matched a setup" carries no information *by construction*, and the whole
cross-check is uninformative regardless of what the outcome test returns. This measures
directly what was previously only estimated: the level/setup saturation argument, which put
coverage of ~9 candidate reference lines at **141%** of typical session travel.

**Primary:** per-setup coverage, and the fraction of entries matching **at least one** and
**at least half** of the list.

**Secondary:** mean return by setup, day-clustered bootstrap, **Holm across all setups
tested** — including every one that fails.

## 2. The setup list — fixed, external, complete

From [TradingSim](https://www.tradingsim.com/blog/day-trading-setups) (six classic setups:
breakout, fade, range, late-day breakout, flag, gap-and-go),
[Bulls on Wall Street](https://www.bullsonwallstreet.com/post/what-is-the-vwap-trading-indicator-and-how-to-use-it-as-a-day-trader)
(VWAP reclaim), and
[Warrior Trading](https://www.warriortrading.com/vwap/) (VWAP as pullback target, RVOL filter).

`s = +1` for a call, `−1` for a put. All quantities from bars **strictly before** the entry
second; the 5-minute bar must satisfy `label + 5min <= entry_time`.

| # | setup | condition at entry |
|---|---|---|
| 1 | **VWAP side** | `s·(price − session VWAP) > 0` |
| 2 | **VWAP reclaim** | price crossed VWAP in direction `s` within the prior 15 min and is still on that side |
| 3 | **ORB continuation** | price beyond the 09:30–09:45 range in direction `s` |
| 4 | **Flag** | 15-min move `≥ +0.5σ` in direction `s`, then the last completed 5-min bar's range `<` half the prior bar's range |
| 5 | **Pullback to 9 EMA** | price within 0.1% of the 5-min 9-EMA, and `s·(9EMA − 20EMA) > 0` |
| 6 | **Gap and go** | `s·(open − prior close) > 0` and `s·(price − open) > 0` |
| 7 | **Fade / range edge** | price within 10% of the session range at the extreme *opposite* to `s` |
| 8 | **Late-day breakout** | entry after 14:00 ET and price at the session extreme in direction `s` |
| 9 | **RVOL** | 5-min bar volume `≥ 1.5 ×` its EMA20 |
| 10 | **Prior-day level break** | price beyond the prior session's high (s=+1) or low (s=−1) |

Ten setups. **No setup may be added, removed, or redefined after a result is seen.**

## 3. Decision rule, committed now

**On coverage (primary):**

| result | conclusion |
|---|---|
| ≥ 90% of entries match at least one setup | **saturated — the cross-check is structurally uninformative.** A "match" carries no information and no outcome result from it should be believed. |
| < 60% match at least one | setups are discriminating; the outcome test is meaningful |
| between | partial; report and interpret with the coverage rate attached |

**On outcome (secondary):** a setup is reported as separating outcomes only if its
day-clustered CI excludes zero **after Holm across all ten**. Anything significant before
Holm but not after is reported as **not significant**, with the raw value shown.

## 4. Power, stated before running

Return sd ≈ 52%, n = 418, DEFF ≈ 1.25 → effective n ≈ 334, SE ≈ 2.9pp. Holm across ten
tests puts the effective per-test α near 0.005, which raises the **MDE from ~8pp to roughly
11–12 percentage points** of return. Nothing measured anywhere in this project has
approached that.

**Stated prior: under 10% that any setup clears.** Six pre-registered nulls precede this —
strike distance, position size, within-day ordinal, entry-timing percentile, a 19-feature
predictive ceiling, and his own declared indicator stack.

**Predicted coverage pattern, committed in advance so it cannot be claimed as a discovery:**
given the established chase result (+0.20σ trailing move, 89% of entries), his entries should
match the momentum-continuation setups (1, 3, 4, 6, 8) at **high** rates and the fade setup
(7) at a **low** rate. If that is what appears, it is a re-measurement of the chase, not a
new finding.

## 5. Leak audit

All conditions use completed bars strictly before entry. `s` is the call/put choice, known at
entry. Session VWAP, opening range, prior-day levels, and EMAs all use only prior data. **No
condition uses the outcome, the exit, the hold duration, or the realised return.** No
winners-only subset is examined at any point.

## 6. Prohibited

No setup added after seeing results. No threshold tuning on any condition. No combination
search across setups beyond the pre-declared "at least one" / "at least half" counts. No
profitable-subgroup selection. No promotion of a secondary to primary.
