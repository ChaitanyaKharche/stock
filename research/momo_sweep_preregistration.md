# Pre-registration — Exhaustive standalone backtest of the MOMO_CHASE family

**Status: PRE-REGISTERED, NOT YET RUN.** Written 2026-09-13. The universe below is
**frozen** and is enumerated in code at `trade_analysis/momo_sweep/grid.py`; its size is
asserted by the sbatch script before any cell runs.

## 0. What this is, and why it is a NEW FAMILY

Two earlier pieces of work bound this one.

`setup_sweep_preregistration.md` §6 (2026-08-20) tested whether any setup **separates the
trader's existing trades**, and said in terms: *"It does not backtest any setup as a
standalone strategy — that is a different and much larger project… not authorised by this
pre-registration."* That project was never authorised since. **This is it.**

`momo_v2.py` (run 2026-09-13, `momo_v2_results.md`) tested three hand-picked variants and
closed MOMO_CHASE as a *mis-specification* question. Its three variants were chosen after
reading an AUC table, so the size of that search is unknown and therefore uncorrectable.

Per §2 of the setup sweep — *"Any setup added after results are seen forms a new family with
its own correction and must be labelled as such"* — **this forms a new family with its own
correction.** It is not a continuation of either, and a result here may not be reported as
though the earlier nulls were a prior in its favour.

**Nothing found here may enter the frozen live lab.** `live_lab_data/FREEZE.json` states the
setup list may not be added to. A survivor here is a candidate for a *separate, future*
forward test, not a change to the running one.

### Why exhaustive is more honest than selective, not less

The instinct is that a 2.8-million-cell sweep is worse science than three careful variants.
It is the reverse, for one precise reason: **a data-snooping correction needs a
denominator.** White's Reality Check and Hansen's SPA test the null *"the best of a FIXED,
PRE-SPECIFIED set beats the benchmark"*. A hand-picked shortlist has no knowable set size. A
fully enumerated grid has one by construction, every cell is run, and every cell is reported.

The grid is dense along continuous axes and sparse along structural ones, deliberately.
Under a bootstrap correction a near-duplicate cell costs almost nothing, because the
resampling preserves the ~0.99 correlation between `trail_min=15` and `trail_min=20`. A
structurally new cell — the opposite direction, a different window — costs nearly a full
trial. Density is cheap; structure is not. **This is also why the nominal cell count is not
the honest search size**, and why §7 reports a measured effective count instead.

## 1. Sample and the split

Packed by `momo_sweep/pack.py` from `live_lab_data/bars_cache`, admitting sessions with
≥300 one-minute bars (identical filter to `journal_zone_wf.sessions()`, which drops early
closes).

| | sessions | span |
|---|---|---|
| QQQ | 2,657 | 2016-01-04 → 2026-08-27 |
| SPY | 1,158 | 2022-01-03 → 2026-08-27 |

- **DISCOVERY: QQQ only, 2016-01-04 → 2021-12-31.** The full grid runs here and nowhere else.
- **HELD-OUT: QQQ and SPY, 2022-01-03 → 2026-08-27.** Untouched unless §8 triggers.

The split is by time and the holdout adds a **second symbol**, so a survivor must generalise
across both a period and an instrument. SPY is entirely inside the holdout, which is why it
is absent from discovery rather than being an oversight.

**Rolling, never anchored.** Hansen's Assumption 1 admits fixed and rolling schemes but not
recursive/expanding estimation. Nothing here is fitted per-period — the rules are fixed — but
the split is stated as a single fixed partition so the question does not arise.

## 2. The frozen universe — 2,822,400 cells

Thirteen axes, enumerated in `grid.py`, de-duplicated (`gate_ref` is inert when
`direction="chase"` or when every gate is off), **2,822,400 unique cells** from a nominal
product of 3,870,720. At the measured 3,731 cells/CPU-hour that is ~756 CPU-hours.

```
trail_min    7   5, 10, 15, 20, 30, 45, 60
sigma_mult   8   0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0
time_exit    6   5, 15, 25, 40, 60, 90
adx_min      4   0 (off), 20, 25, 30
dmi_tf       3   off, 1m, 5m
macd_gate    2   off, on
window       5   full, open hour, 10:00-13:00, 10:30-14:30 (frozen), afternoon
direction    2   chase, fade
ema9_min     2   off, 0.71 ATR
max_per_day  2   cooldown only, 3
gate_ref     2   gates confirm the TRADE sign / the MOMENTUM sign
sig_norm     2   session-to-date sigma, time-of-day-conditional sigma
vol_regime   3   off, high tercile, low tercile
```

**The frozen MOMO_CHASE is cell #928,056 and is in the grid.** The sweep is therefore
anchored: its V0 cell must reproduce `momo_v2.py`, and that reproduction is a hard
precondition (§6).

**Fixed, NOT swept:** `macd=(9,17,9)` and `dmi_len=14`. These were calibrated to the
trader's behaviour in the journal study and never to P&L — measurements, not free
parameters. Sweeping them would convert a constant into a fitted value and spend correction
budget doing it.

**`direction` and `gate_ref` exist because the family's core assumption may simply be
backwards.** "Momentum continues" is the hypothesis; "a stretched move reverts" is the same
signal with a flipped sign and costs one bit to test. `gate_ref` separates a *divergence*
fade from an *exhaustion* fade, which are different claims. Testing only `chase` would
assume the answer.

**`sig_norm` and `vol_regime` are in on published grounds, not intuition.** Intraday
volatility is U-shaped, so a session-wide sigma makes a fixed threshold much easier to clear
near the open — the threshold silently becomes a time-of-day filter. And the published
intraday-momentum coefficient is insignificant in the low-volatility tercile (R² 0.6% vs
3.3%). Both are documented in `research/vendor_and_method_surveys/`.

**Not in the grid, with reasons:** stop-loss and profit-target structure (the literature
this family comes from holds the full window with no stop, and there is essentially no
peer-reviewed support for tuned stops — the engine is time-exit-only by design); option
expression (0DTE round-trip friction is ~5–6% of premium and 0DTE trades underperform other
option trades by 4.7%, t=−10 — a shares test is the *generous* case, and if it fails on
shares the option version cannot be rescued).

## 3. Costs, which decide this outright

Every headline number is **net**. Gross is reported only alongside net, never alone.

Cost is charged **per share, not per basis point**: `cost = shares × spread_per_share`, with
`spread_per_share = $0.01` for both QQQ and SPY (both penny-wide throughout the sample) and
$0 commission. The engine returns the exact share count per session so the charge is exact
and the assumption can be varied without re-running the grid.

Per-basis-point costing would be wrong here in a way that correlates with time: QQQ spans
roughly $100 to $700 over the sample, so a fixed $10,000 position turns over 100 shares in
2016 and 14 in 2026. A flat bp rate would overstate early costs and understate late ones.

**The primary reported diagnostic is the BREAKEVEN SPREAD** — gross dollars per share
traded. A cell whose breakeven is below $0.01 loses money at any realistic fill, whatever its
t-statistic. This is stated in advance because the frozen cell's own breakeven is on the
order of **$0.001/share, roughly ten times too small**, and because this project has already
reported a setup at gross +$0.0205 / net −$0.0171 once.

## 4. The test

Loss differentials are the per-session net P&L series; the benchmark is **not trading**
(loss 0). The null is `max_k E[d_k] ≤ 0`: **no cell in the grid makes money**.

- **Hansen's SPA, studentized**, stationary bootstrap (Politis–Romano), block length from
  the Politis–White rule **with the Patton (2009) correction**, B = 10,000. All three
  recentrings (`lower`, `consistent`, `upper`) reported; `consistent` governs.
- **StepM (Romano–Wolf)** to identify *which* cells survive, not merely whether any does.
  Holm is not used on this family: at 2.8M cells it divides α by 2.8M and could not detect
  anything, while StepM exploits the cell correlation the bootstrap already models. (Holm
  remains correct for the live lab's family of 13, which is weakly correlated.)
- **SPA is implemented in this repo, not taken from `arch`.** `arch` 8.0.0's
  `SPA(studentize=True)` silently ignores the flag — verified in source and by 40/40
  bit-identical p-values — making it White's Reality Check. On a comparison set dominated by
  poor models, which an exhaustive grid definitionally is, that costs about an order of
  magnitude of power; measured here at 0.57 vs 0.00 on Hansen's own design.
- **Deflated Sharpe and PBO are reported as secondary and are not decisive.** PBO is high by
  construction for a dense grid (its own authors note overfitting "among many skillful
  strategies"), and DSR is a sensitivity surface in the variance of trial Sharpes rather
  than a number.

## 4b. The SECOND gate: a MinBTL-derived Sharpe floor

**Added 2026-09-14, before the sweep has been run.** SPA asks whether the best cell's
statistic could have arisen under the null. It does **not** ask whether the winner's Sharpe
is large enough to be worth having *after a search that size*. Those are different
questions and a cell can pass the first while failing the second.

Bailey/Borwein/López de Prado/Zhu, Theorem 2: `MinBTL < 2·ln(N) / E[max_N]²` years.
Inverted for the quantity actually needed, `SR_min(N) = E[max_N] / sqrt(years)`, using the
**Gumbel** expression for the expected maximum of N standard normals — *not* the
`sqrt(2·ln N)` asymptotic, which overstates the floor by 23% at these sample sizes.
Calibration: the corrected form reproduces Bailey et al.'s own worked example (45
configurations, 5 years → Sharpe 1.0) at **0.9998**; the asymptotic gives 1.234. Asserted
by `stats_test.py`.

On this discovery window (1,499 sessions = **5.95 years**):

| effective trials | minimum in-sample Sharpe |
|---|---|
| 10 | 0.65 |
| 50 | 0.93 |
| 200 | 1.13 |
| 1,000 | 1.33 |
| 10,000 | 1.58 |
| 2,822,400 (nominal) | 2.08 |

**The floor is computed from `effective_trials()`, never from the nominal cell count.**
MinBTL assumes independent trials; this grid is ~0.99 correlated between neighbours, and the
SPA bootstrap prices that dependence exactly while MinBTL cannot. Feeding it 2.8M would be
absurdly conservative; feeding it the measured participation ratio is the honest reading.

**Committed rule: the winning cell must clear BOTH the SPA p-value AND
`minbtl_sharpe_floor(n_eff, 5.95)`. Either alone is insufficient.** Both are fixed in
advance — the floor depends only on the search size and the sample length, neither of which
is a function of which cell wins.

For calibration: this window supports only **~20 independent configurations** at a target
in-sample Sharpe of 1.0. Harvey/Liu/Zhu's corresponding t-ratio hurdles for this literature
are **3.39–3.78** (BHY 1%, Bonferroni), rising to **3.68** under their correlation
adjustment — not 1.96. The live lab's separate Holm-over-13 bar implies roughly t ≈ 2.7–3.0,
which is in the right region; the sweep is not governed by a fixed t at all, which is
precisely why SPA replaced Holm there.

## 5. Power and feasibility, stated before running

The minimum-backtest-length literature implies ~6 years supports roughly **45–50
INDEPENDENT trials**, and E[max Sharpe | true Sharpe = 0] rises to ≈1.65 annualised at 5,000
independent trials. **2.8M nominal cells are nowhere near 2.8M independent trials** — they
are heavily correlated — but the honest count is not the nominal one either. §7 reports the
measured effective count via the eigenvalue participation ratio.

**If the measured effective trial count materially exceeds what the sample supports, that is
reported as a limitation of this design, not quietly omitted.**

The bar is blunt: the frozen cell's gross edge is ≈0.034 bp/trade against a round-trip cost
of ≈2.5 bp. A cell must beat the frozen one by roughly **two orders of magnitude in
per-share terms** to be net positive at all. Published time-series intraday momentum is
~2.8 bp/day gross, and post-cost anomaly decay is ~93%. **The prior is strongly against
finding anything, and that is stated here so a null cannot later be framed as a surprise.**

## 6. Mandatory pre-conditions — the sweep does not run unless all pass

1. **V0 reproduction.** The grid's frozen cell must reproduce `momo_v2.py` at **n = 8,598
   trades**. Verified 2026-09-13: n = 8,598 exactly, +$0.0339/trade.
2. **`stats_test.py` passes**, including that shard-reduced maxima equal whole-grid maxima.
3. **Lookahead audit.** A run with the signal shifted one bar into the future must score
   **materially BETTER**. If peeking does not help, the honest path already contains future
   information and every number is void. This project's worst bug was a five-minute
   lookahead that produced an entire measured edge.
4. **Direction placebo.** Randomising trade direction must produce ≈0 net edge across the
   grid, confirming the machinery is not manufacturing P&L from the entry/exit mechanics.

## 7. Reported regardless of outcome

Full grid size; the measured **effective trial count**; the best cell by net mean and its
breakeven spread; the SPA p-value under all three recentrings; the StepM survivor set; the
distribution of net means across all cells (not just the top); per-axis marginals; and the
frozen cell's own position in the distribution.

## 8. Decision rule, committed now

| result | conclusion |
|---|---|
| SPA p ≥ 0.05 on discovery, net | **No parameterization of the momo-chase form beats not trading, net of costs, anywhere in 2.8M cells.** MOMO_CHASE is closed as a standalone strategy. The held-out set is never touched. |
| SPA p < 0.05 net | **Not a result — a hypothesis.** StepM gives the survivor set; it goes to the held-out set (2022–2026, QQQ **and** SPY) once. Only a same-sign, net-positive survivor there is worth anything, and even then it earns a new forward-test pre-registration, not a live promotion. |
| significant gross but not net | reported as **a cost failure, not an edge** — the explicit category this project has been burned by before |
| significant on QQQ but not SPY in holdout | reported as **failed**; a single-instrument survivor of a 2.8M-cell search is the expected shape of a false positive |

## 9. Prohibited

No cell added, removed, or re-ranged after any result is seen. No re-running with a
different cost assumption to rescue a cell (the cost model is §3; breakeven is reported so
sensitivity is visible without re-running). No reporting a survivor without the full grid
distribution. No promotion of a DSR or PBO figure over the SPA verdict. No use of the
held-out set unless §8 triggers, and then exactly once. No addition to the frozen live lab
under any outcome.
