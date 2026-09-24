# Pre-registration — what happens AFTER R1/S1 breaks early

**Frozen 2026-09-23, before any number below was computed.** Executed by
`trade_analysis/live_lab/after_break.py`. Data: `live_lab_data/six_lines_QQQ.json`,
2,656 QQQ sessions, 2016-01-05 → 2026-08-27, already on disk from the six-line sweep.

## 0. The three questions, as asked

1. When R1 (or S1) breaks in the first 15 minutes, does price then run to R2 (or S2), or
   reverse straight away?
2. If it reaches R2/S2, does it break that too, or stall there?
3. Do R1/S1 tend to sit at round numbers — multiples of 25?

## 1. Definitions frozen in advance

**"First 15 minutes" becomes the first two 10-minute bars, 09:30 and 09:40.** The stored
sweep is on a 10-minute grid, so 15 minutes is not expressible. This is the closest honest
bucket and it is 20 minutes, not 15. Stated here so the widening is on the record.

- **Break** = close-based, as everywhere else in this project. A 10-minute RTH bar closing
  beyond the line.
- **Early break** = that line's `close_break` is `09:30` or `09:40`.
- **Live line** = not `pre_broken` (not already beyond the 09:30 open). Pre-broken lines
  are excluded from every rate, same convention as `six_lines.py`.
- R-side and S-side are analysed **separately**, never pooled. A session may appear in both.

## 2. Outcomes, fixed now

For each session with an early R1 break (mirror for S1):

| outcome | definition |
|---|---|
| `reach_R2` | `rth_high >= R2.price` — price got there at all |
| `break_R2` | `R2.close_break` is not null |
| `break_R3` | `R3.close_break` is not null |
| `minutes_R1_to_R2` | clock gap between the two close-breaks, when both occur |
| `dist_R1_R2_bp` | `(R2.price / R1.price − 1) × 10,000` |
| `failed` | the session's trade record exits `"failed"` (closed back inside R1) |
| `mfe_bp`, `move_bp` | from the trade record, when R1 was also the session's FIRST break |
| `giveback_bp` | `mfe_bp − move_bp` — how much of the best move was handed back |
| `full_reversal` | `rth_low < S1.price` — broke up through R1, then down through S1 |

## 3. The control, without which none of this means anything

Every rate above is reported **three ways**:

- early R1 break (09:30–09:40)
- late R1 break (09:50 or later)
- all sessions where R1 was live

**The claim "an early break predicts continuation" requires the early column to beat the
late column.** A high `break_R2` rate on its own says nothing: R2 breaks often anyway.
If early and late are within noise of each other, timing carries no information and that
is the finding.

## 4. Round numbers

Distance from R1 and from S1 to the nearest multiple of **M ∈ {1, 5, 10, 25}**, in dollars
and in basis points (bp = one hundredth of a percent; on a $740 stock 1 bp ≈ 7.4 cents).

**Null:** if levels have no affinity for round numbers, `price mod M` is uniform, so the
distance to the nearest multiple is uniform on `[0, M/2]` with **mean M/4**. Clustering
means the observed mean comes in **below M/4**.

QQQ ran from ~$110 to ~$740 over this sample, so a multiple of 25 is a far coarser grid in
2016 than in 2026. The test is therefore also run **split by price era** (below/above $300)
so a result cannot be an artifact of the stock simply getting bigger.

## 5. One pre-specified subgroup, and only one

Split early-R1-break sessions at the **median distance from R1 to the nearest multiple of
25, in bp**. Compare `reach_R2` and `break_R2` across the two halves.

One split, chosen now, at the median — no threshold tuning. Any other cut of this data is
exploratory and will be labelled as such. `setup_sweep_results.md` spent 2,822,400 cells
establishing what happens when that rule is not held.

## 6. Inference

One observation per session, so no day-clustering is needed. Proportions get **Wilson 95%
intervals**. Early-vs-late differences get a **two-proportion test**; with ~317 early and
~950 late the detectable difference is roughly 9 percentage points, so a null here means
"no large effect", not "no effect".

## 7. What this cannot conclude

- **Nothing about P&L, options, or the cap.** No price is paid anywhere in this design.
- **Nothing about SPY.** `six_lines_SPY.json` does not exist; that leg never finished.
- **Nothing about the exact path inside a 10-minute bar.** Whether price touched R2 at
  09:52 and reversed by 09:57 is invisible at this resolution.
- A positive result here is a **description of price behaviour, not a strategy.** Turning
  it into one needs its own pre-registration with costs in it — the ~5 bp an ATM 0DTE
  needs for spread and theta has killed every edge this project has measured so far.
