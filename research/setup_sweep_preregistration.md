# Pre-registration — Frozen sweep of 24 simple setups against the trader's real entries

**Status: PRE-REGISTERED, NOT YET RUN.** Written 2026-08-20. The setup list below is
**frozen**. Any setup added after results are seen forms a new family with its own
correction and must be labelled as such.

## 0. What this is and what it corrects

The request was to find profitable setups and cross-check them against his signals, on the
expectation that "at least some setup should work." Searching a space and keeping the
winners is the garden of forking paths; **stating the expected conclusion in advance makes
it worse, not better.** This design keeps the search and removes the bias: fixed universe,
every result reported, family-wise correction, held-out confirmation.

**Expected false positives if nothing works: 24 × 0.05 = 1.2 setups significant at raw
α=0.05.** That number is committed here so that finding one or two "hits" cannot be read as
success.

## 1. Sample and the held-out reserve

Feature frame from `indicator_state_results.md`: **418 round trips** with complete pre-entry
features, 151 dates, 13 underlyings.

- **DISCOVERY set:** entries 2025-05-05 → 2025-11-26 (the window in which all prior beliefs
  were formed). Sweep runs here only.
- **HELD-OUT set:** everything before 2025-05-05 and after 2025-11-26. **Not touched until
  a setup survives correction on the discovery set.**

Splitting this way, rather than randomly, because the held-out blocks are already known to
be negative in aggregate — so a setup that survives there is surviving a hostile test.

## 2. The frozen universe — 24 binary setups

All direction-aligned where applicable (`s = +1` call, `−1` put); all computed from bars
**strictly completed before** the entry second, per the timing rule already in force.

**Trend / location (6)**
1. price > session VWAP, aligned
2. price > prior-day close, aligned
3. opening-range (09:30–09:45) high/low cleared, aligned
4. position in day's range so far > 0.5, aligned
5. price > day's open, aligned
6. prior-day direction aligned

**Momentum (6)**
7. MACD(9,17,9) histogram aligned > 0
8. (+DI − −DI) aligned > 0
9. ADX > 20
10. ADX > 25
11. RSI(14) aligned > 50
12. trailing 15-min move aligned > 0

**Volatility / participation (5)**
13. volume > EMA20(volume)
14. volume > 1.5 × EMA20(volume)
15. ATR% above its own 20-session median
16. day's range so far > prior day's full range
17. overnight gap aligned

**Time of day (4)**
18. entry 09:30–10:30 ET
19. entry 10:30–14:00 ET
20. entry 14:00–15:30 ET
21. entry 09:00–10:00 **local (Arizona)** — his own anchor, per the DST finding

**Structure (3)**
22. 0DTE
23. DTE ≥ 1
24. 10m + 15m MACD & DMI agreement

## 3. Test and correction

For each setup: mean per-trade **return** when TRUE vs FALSE, difference tested by
**day-clustered bootstrap** (10,000 resamples of activity dates), two-sided p.

- **Holm** across all 24 (primary correction)
- **Benjamini–Hochberg** FDR at q=0.10 (secondary, reported alongside)
- Cells with n < 20 on either side are reported but **excluded from the correction family**
  and cannot be declared a survivor.

**Every one of the 24 is reported**, ranked by raw p, including the 20-odd that will fail.

## 4. Decision rule, committed now

| result | conclusion |
|---|---|
| no setup survives Holm | **no simple setup separates his outcomes.** Stop. This closes the search. |
| ≥1 survives Holm | **not a result — a hypothesis.** Test it on the HELD-OUT set. Only if it holds there with the same sign is it worth anything. |
| something significant at raw α but not after correction | reported as **expected noise**, explicitly counted against the 1.2 expected false positives |

## 5. Power, stated before running

Return sd ≈ 52%; discovery n ≈ 368; DEFF ≈ 1.25 → effective n ≈ 294. On a balanced split,
SE on the difference ≈ 6.1pp, so the **uncorrected MDE is ≈ 17pp** and, after Holm across
24, the effective MDE is roughly **22–24pp**.

**This sweep can only find very large effects.** A genuine 5pp edge is invisible to it. A
null therefore means "no setup separates outcomes by more than ~20 percentage points of
return," not "no setup works."

## 6. Scope note

This tests whether any setup **separates his existing trades**. It does not backtest any
setup as a standalone strategy — that is a different and much larger project, bounded by the
**2020-01-01 options floor**, and it is not authorised by this pre-registration.

## 7. Prohibited

No setup added after seeing results. No threshold tuning on any of the 24. No reporting of
survivors without the full 24. No promotion of an FDR survivor over a Holm failure. No use
of the held-out set unless step 4 is triggered.
