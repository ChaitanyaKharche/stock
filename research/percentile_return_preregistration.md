# Pre-registration — Percentile of Achievable Return

**Status: PRE-REGISTERED, NOT YET RUN.** Written 2026-08-18 after data inspection and
before any percentile was computed. Nothing below may be changed once the first result is
produced. Deviations, if any become necessary, must be appended as dated amendments with
the reason, never edited in place.

## 0. The question

Does the trader's realised entry/exit outcome sit above what random timing in the *same
contract on the same session* would have produced? This tests whether his execution
decisions carry information. It does **not** test whether he has a profitable strategy,
and it cannot identify *what* the information is.

## 1. Data inspected, and what is available

**Quote source.** ThetaData `/option/history/quote`, 1-minute, Options Value tier, local
disk cache at `.cache/thetadata/option_history_quote` (2,119 payloads already present).
Fields per row, verified from a cached payload:

```
ask, bid, ask_size, bid_size, ask_exchange, bid_exchange,
ask_condition, bid_condition, timestamp
```

Contract metadata: `symbol, expiration, strike, right`.

**Session length.** 391 rows per contract-session, `09:30:00` through `16:00:00` inclusive.
Verified: the **09:30 row is universally `bid=0, ask=0, size=0`** and the **16:00 row is a
settlement artifact** (observed `ask=0.01, bid=0.0, bid_size=0`). Both are excluded.

> **Maximum usable session duration: 389 minutes (09:31 – 15:59 inclusive).**

**Broker source.** `trade_analysis/data/robinhood_trades.csv`, parsed with `csv.reader`,
rows with `len(row) != 9` dropped. Reverse-chronological within and across days (verified:
2 negative-position violations chronologically vs 682 in file order vs ~357 under random
shuffle). **Chronology is used for nothing in this experiment** — it is noted only to
record that the ordering fact is established.

## 2. Eligibility — fixed before running

A round trip is one **unique contract** `(symbol, expiry, right, strike)`, P&L formed from
all its BTO and STC fills. Inclusion requires all of:

1. `symbol ∈ {QQQ, SPY}` — no ThetaData options entitlement exists for the 9 single names.
2. All fills on a **single activity date** (enumeration is within-session).
3. Bought quantity equals sold quantity (a complete round trip).
4. ≥ **60** minutes in that contract-session passing the quote filter of §4.

**Counts established at inspection (before any result):**

| | n |
|---|---|
| option fills | 1,051 |
| distinct contracts | 333 |
| QQQ / SPY contracts | 276 |
| — excluded, spans > 1 session | 9 |
| — excluded, quantity unbalanced | 1 |
| **candidate round trips** | **266** (QQQ 206, SPY 60) |
| out of scope, 9 single names | 57 contracts / 197 fills (18.7%) |
| chain requests required | 144 `(symbol, expiry, date, right)` |
| distinct activity dates | 89 |

DTE at trade among candidates: 0DTE 249, 1DTE 14, 2DTE 1, 3DTE 2. Non-0DTE contracts are
retained — they quote normally on the activity date.

Any candidate failing criterion 4 after fetching is excluded and **counted in the report**.

## 3. Return definitions

**The trader's realised return — from actual broker fill prices**, per the instruction to
prefer what was actually achieved:

```
r_user = ( Σ_STC Price_i × Qty_i ) / ( Σ_BTO Price_j × Qty_j ) − 1
```

Uses the `Price` column, i.e. **gross of fees**, so that it is measured on the same basis
as the enumerated opportunities, which carry no fees. A **fee-inclusive sensitivity** using
the `Amount` column (which embeds the ~$0.085/contract round-trip ORF/OCC pass-through,
≈0.11% of a median position) will be reported alongside. It is not expected to matter and
is reported regardless of whether it does.

**Enumerated opportunity return**, for entry minute `i` and exit minute `j > i`:

```
r(i,j) = Exit(j) / Entry(i) − 1
```

**Pricing convention — PRIMARY is mid-to-mid:**

```
Entry(i) = ( bid_i + ask_i ) / 2        Exit(j) = ( bid_j + ask_j ) / 2
```

**Secondary is executable ask-to-bid:** `Entry(i) = ask_i`, `Exit(j) = bid_j`.

*Why mid is primary.* The trader's fills carry Robinhood price improvement and generically
sit **inside** the NBBO. Enumerating opportunities at ask→bid charges them a full spread he
did not pay, which inflates his percentile — i.e. it biases toward concluding he has
information. Mid→mid removes that advantage and is therefore the conservative choice given
this project's documented history of false positives. Both are reported; if they disagree
on the conclusion, that disagreement is the finding.

## 4. Quote filtering — fixed before running

A minute is **valid** iff all hold:

- timestamp within `09:31:00 … 15:59:00` (excludes the 09:30 null row and the 16:00 stub)
- `bid > 0` and `ask > 0`
- `ask >= bid`
- for the ask→bid secondary only: `ask_size > 0` and `bid_size > 0`

Invalid minutes are **dropped, never forward-filled**. Enumeration runs over valid minutes
only; `j > i` is enforced on the surviving index.

**Pre-registered quote-quality sensitivity** (item 9 of the required output), one pass each,
no selection among them:

- **QF1** baseline as above
- **QF2** additionally require relative spread `(ask − bid) / mid <= 0.50`
- **QF3** additionally require `bid_size > 0 and ask_size > 0` on both legs

## 5. The duration grid — and a declared deviation from the brief

**PRIMARY endpoint: unconstrained enumeration.** All pairs `(i, j)` with `j > i` over valid
minutes. No duration constraint.

**This deviates from the brief's "fixed set of holding durations" and the reason must be on
the record.** His true holding duration is **unobservable** — the export has no execution
times, and the fill-price inversion is excluded by instruction. A duration-matched
enumeration therefore requires assuming an unknown quantity, and the assumption drives the
answer: return dispersion grows with duration, so comparing a short real hold against a
long enumerated duration pulls the percentile toward 0.50 as an artifact, while the reverse
pushes it away. The unconstrained set requires no unobservable input and answers exactly
the stated interpretation — "is his outcome consistent with random timing in this contract
today."

**SECONDARY, pre-registered, all five reported, none selected among:** duration
`d ∈ {10, 20, 40, 60, close}` minutes, enumerating pairs separated by exactly `d` valid
minutes (`close` = exit at the last valid minute). These are a sensitivity band. **No
duration may be promoted to primary after seeing results.**

## 6. The percentile statistic

With `R = { r(i,j) }` the enumerated set for a round trip, **mid-rank** to handle ties,
which are guaranteed by penny quoting:

```
P = ( #{ r ∈ R : r < r_user }  +  0.5 × #{ r ∈ R : r = r_user } ) / |R|
```

Mid-rank is required: the naive `#{r <= r_user}/|R|` is upward-biased under ties and would
break the 0.50 null. The §8 control exists to verify this.

If `r_user` lies outside `[min R, max R]`, `P` is 0 or 1 by construction. Such trades are
**retained** and their count reported separately — they are real outcomes, not errors.

## 7. Primary analysis and pre-committed decision rule

- **Primary statistic:** mean of `P` across eligible round trips.
- **Null:** 0.50.
- **CI:** cluster bootstrap resampling **activity dates** (89 clusters) with replacement,
  **10,000** replicates, percentile method, 95%. Day clustering is mandatory — same-day
  trades share contract, regime and direction; measured design effect on these data is
  **1.31–1.43** (the "9.4x" figure previously circulated is retracted and must not be used).
- **Secondary tests:** median `P`; two-sided sign test of `P > 0.5` against 0.5; both with
  day-clustered inference.
- **Power:** SD of a percentile ≈ 0.289; at n = 266 with DEFF ≈ 1.35, effective n ≈ 197,
  SE ≈ 0.021, **MDE ≈ 0.058 at 80% power**. The test resolves a shift from the 50th to
  roughly the 56th percentile and **nothing smaller**. A null result means "no effect of
  0.058 or larger," never "no effect."

**Decision rule, committed now:**

| Result | Conclusion |
|---|---|
| 95% CI ⊆ [0.47, 0.53] | **(B) consistent with random timing → STOP.** No follow-up experiment. |
| CI lower bound > 0.53 | **(A) evidence of non-random execution information** → proceed to the smallest discriminating follow-up |
| CI upper bound < 0.47 | **(C) execution timing systematically poor** |
| §8 control fails | **(D) invalid** — report the diagnosis, no substantive conclusion |
| none of the above | **Inconclusive** — report as such; do not resolve by choosing a secondary |

## 8. Critical controls — run and reported BEFORE the primary result

**C1 — Synthetic-trade calibration.** For each eligible contract-session, draw a random
valid `(i, j)` pair, price it under the primary convention, and compute its percentile
against that same contract's enumerated distribution. Repeat 200 times per contract.
**Pass:** grand mean ∈ [0.49, 0.51]. Failure indicates a tie-handling or indexing defect
and triggers verdict (D).

**C2 — Pricing-convention bias, the mismatch check.** The trader's return comes from
**fills**; the enumeration comes from **quotes**. Even random timing could therefore score
off-centre. Measure it: draw a random valid `(i, j)`, price that draw at **ask→bid**
(simulating a real fill crossing the spread), and compute its percentile against the
**mid→mid** enumeration. The resulting mean is the **bias offset `b`**. If `b` differs
materially from 0.50, the primary result is interpreted against `b`, not against 0.50, and
that comparison is stated explicitly in the report. This control quantifies the exact
concern raised in the brief and its output is reported whether or not it is convenient.

**C3 — Support check.** Count trades whose `r_user` falls outside the enumerated support,
and the count of eligible contracts with fewer than 60 valid minutes.

## 9. Secondary / descriptive output (never promoted to primary)

- Full distribution of `P`: 25th / 50th / 75th percentiles, histogram
- Proportion with `P > 0.50`; proportion with `P > 0.53`
- QQQ and SPY separately — **descriptive only**
- The five duration cells of §5
- The three quote-quality filters of §4
- Fee-inclusive realised return
- Restricted to round trips with exactly one BTO and one STC fill, where the single-entry/
  single-exit enumeration matches his actual structure exactly (multi-tranche trades give
  him an averaging effect the enumeration cannot represent — a stated limitation)

## 10. Prohibited, explicitly

No duration selection after seeing results. No threshold tuning. No winner/loser split as
primary — the percentile encodes the outcome, so conditioning on profitability is circular
by construction. No inferred timestamps. No fill-price fingerprint. No modification of the
broker reconstruction to change the result. No new indicators, parameters, or strategy code.
Existing backtest logic is not touched.

## 11. What this experiment can and cannot establish

**Can:** whether his realised outcomes are exchangeable with random entry/exit in the same
contract-session, at a resolution of ≈0.058 percentile units.

**Cannot:** (a) identify the source of any information found — retest behaviour, general
timing, or contract selection are indistinguishable here, which is what the follow-up in
§7 would address; (b) establish profitability — the enumerated distribution is a control,
not an attainable strategy, and must never be reported as one; (c) rule out an effect
smaller than the MDE; (d) speak to day selection or direction choice, both of which are
outside this design; (e) generalise beyond 2025-05-05 → 2025-11-26, one regime, no
out-of-sample period.

**A significant result would not by itself demonstrate an edge.** It would demonstrate
non-exchangeability, which is a necessary and far-from-sufficient condition.
