# Results — Spread survey for broadening the SHARES arm

Run 2026-09-02 (after the close) with `trade_analysis/backtesting/spread_survey.py`.
51 candidates, 5 sessions (2026-08-27 → 2026-09-02), RTH window 09:45–15:45 ET only, since
the open and close are structurally wide and would punish names unevenly. Read-only; no lab
state or frozen definition touched.

**The question:** the shares edge measured on IMB is **+3.34 bp per trade gross**. Round-trip
spread comes straight off that. So which symbols can actually carry it?

---

## 1. Spread eliminates most of "top 10 per sector"

**16 of 51 names are outright untradeable** — their *median* round-trip spread exceeds the
entire gross edge:

| symbol | spread bp | headroom |
|---|---|---|
| COIN | 9.56 | −6.22 |
| GS | 7.60 | −4.26 |
| LLY | 7.25 | −3.91 |
| F | 7.18 | −3.84 |
| CRM | 4.99 | −1.65 |
| ORCL | 4.98 | −1.64 |
| JNJ | 4.88 | −1.54 |
| COST | 4.53 | −1.19 |
| BA, AMD | 4.36 | −1.02 |
| UNH | 4.10 | −0.76 |
| T | 3.84 | −0.50 |
| MU, PFE | 3.53 | −0.19 |
| HD | 3.42 | −0.08 |
| JPM | 3.37 | −0.03 |

These are exactly the high-options-volume household names the broadening idea was aimed at.
**AMD, JPM, COST, CRM and COIN cannot pay for a 3.34 bp signal, whatever their option
volume looks like.** A further 13 (META, PLTR, AVGO, TSLA, CVX, SMH, XLU, PG, VZ, XOM,
AMZN, DIS …) are "thin" — under 1.5 bp of headroom, so a single bad fill erases several
good trades.

## 2. The tightest names are all ETFs

```
SPY   0.26 bp      IWM   0.34      QQQ   0.41      DIA   0.56
XLY   0.86         NVDA  0.89      AAPL  0.92      WMT   0.97
XLK   1.08         XLI   1.12      KO    1.12      INTC  1.12
XLV   1.16         XLP   1.17      WFC   1.18      GOOGL 1.18
```

The four broad ETFs are **2–10× tighter than any single stock.** That is the whole ballgame
for a 3 bp edge.

## 3. A claim of mine that the data does not support

Before running this I asserted that *"the names liquid enough to carry the edge are the names
most correlated with what we already trade."* Measured across 51 names, the correlation
between spread and correlation-with-QQQ is **−0.13** — essentially nothing. There are tight,
low-correlation names (AAPL 0.92 bp, WMT 0.97 bp, XLV 1.16 bp, XLP 1.17 bp). **I stated that
trade-off as fact before measuring it, and it is not there.**

**However, the correlation column is not trustworthy either.** It is estimated on **5
sessions**, which is far too few for a single name where a day of idiosyncratic news
dominates. AAPL prints 0.05 against QQQ over this window because it genuinely decoupled
(2026-09-01: QQQ +0.09%, AAPL +2.68%). Verified the underlying bars are clean — that is real,
not a data fault, but it is a 5-day accident and not an estimate. **The correlation numbers
are therefore excluded from the selection below.** Establishing them properly needs ~60
sessions per name.

## 4. The binding constraint turned out to be operational

Measured against the live gateway: **median 477 ms per call**, max 2,431 ms. The shares
runner makes two sequential calls per symbol per tick (bars, then NBBO):

| symbols | seconds per tick | vs the 5 s poll |
|---|---|---|
| 2 (today) | 1.91 | OK |
| 6 | 5.73 | **exceeds** |
| 10 | 9.55 | **exceeds** |
| 15 | 14.32 | **exceeds by 3×** |

**Broadening naively would put the loop permanently behind**, which is precisely how the
40-minute stale fills happened on 2026-09-01 — and those cost ~$500 across both arms.

This is a prerequisite, not a caveat. Two fixes, either sufficient:

1. **Cache bars per minute.** A 1-minute bar changes once a minute; re-pulling the whole
   session every 5 seconds is ~12× waste. Biggest win, no concurrency risk.
2. **Parallelise the per-symbol calls.** 15 symbols concurrently ≈ one call's latency.

---

## 5. Proposed universe — 15 names, selected on spread only

| tier | symbols | spread bp |
|---|---|---|
| broad index | SPY, QQQ, IWM, DIA | 0.26–0.56 |
| sector | XLK, XLY, XLI, XLV, XLP, XLE, XLF | 0.86–1.73 |
| single name | NVDA, AAPL, GOOGL, WMT | 0.89–1.18 |

Every one is ≤ 1.73 bp (≥ 1.61 bp of headroom) and clears $400k+ of median dollar volume per
minute, so a $10,000 clip is well under 3% of a minute's flow.

**Deliberately ETF-weighted, 11 of 15.** ETFs have no earnings. The shares arm has no
earnings filter, and the prior-day setups — `PDH_PDL_Breakout`, `PDH_PDL_FailedBreak`,
`Gap_Fade`, `Crabel_Stretch` — behave pathologically on the session after an earnings gap.
Four single names is enough to test whether the family generalises beyond ETFs at all.

## 6. What it buys, and what it costs

**Buys:** trades ×7.5. Information ×2–3, not ×7.5, because same-day observations across
correlated names are not independent (`DEFF = 1 + (m−1)ρ`). Time-to-verdict on IMB drops from
~14 months to roughly **5–7**.

**Costs:**
- ~~`symbols` is inside the shares config hash, so this **forks it and restarts the shares
  count at zero**. Currently 31 trades — cheap now, expensive in three months.~~

  **CORRECTED 2026-09-05 — this was wrong, and I asserted it without checking.** The hash
  does fork (`53229d6f1f24df10` → `b53ca8a58aa11718`), but the count does **not** restart,
  because the shares arm has no cross-symbol coupling anywhere:
  `max_per_day` is counted against `(setup_id, symbol)`, `max_per_direction` against
  `(setup_id, symbol, direction)`, the one-open-per-setup rule filters `p.symbol == sym`
  first, and there is no global position or capital cap. A QQQ signal is therefore
  evaluated, capped and filled identically whether the universe holds 2 names or 15, so
  QQQ trades under the two hashes are draws from the same distribution and pool validly.
  **The 80 trades collected under the narrow universe are kept.** Recorded in
  `live_lab_data/shares/FREEZE.json` under `why_both_hashes_pool`, with the residual
  operational coupling (shared feed connection) named as the condition that would void it.
- 1,040 (setup × symbol) cells if analysed per name. The design must keep the family at
  **13 setups**, pool across symbols, and treat symbol as a clustering variable rather than
  a hypothesis.

**The options arm cannot broaden at all**: no single-name entitlement on this tier, no 0DTE
on single names, and adding symbols would fork the frozen hash and discard 73 trades.

## 7. Recommended order

1. Fix the feed loop (per-minute bar cache), re-measure latency.
2. Stage: **6 symbols first** — SPY, QQQ, IWM, DIA, XLK, NVDA — for a week, watching fill
   latency in the dashboard.
3. If median lag stays ≈1 min, expand to the full 15 and fork the hash once, not twice.

Doing step 3 before step 1 would reproduce the stale-fill failure on 15 symbols instead of 2.

**DONE 2026-09-05, and the staging in step 2 turned out to be unnecessary** — see
`runner_capacity_results.md`. The cache alone did not clear 15 symbols: it fixes
*sustained* load (~5 → ~29 symbols) but not the *peak*, because every symbol waits on the
same minute boundary and their refetches collide on one tick. Concurrent fan-out was
needed as well, and with both, the measured boundary tick at 15 symbols is **1.59 s**
against a 5 s poll. The week-long 6-symbol stage was there to watch for lag that the
measurement shows cannot occur, so it would have cost a week and learned nothing.
