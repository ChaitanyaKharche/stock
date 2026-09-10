# Results — VRP baseline and lookahead audit

**Interim. The pre-registered experiment is NOT complete.** This records step 2 of
`vrp_preregistration.md`: the benchmark, and the audit that must pass before any number
here is believed.

**Final frame:** 769 sessions built, **5-minute bars**. **747 reach the models**, and
346 + 250 + 151 = 747 — so the earlier wording here, *"zero sessions dropped"*, was
wrong on its own arithmetic and is corrected rather than quietly deleted.

Train 20,634 origins / 346 sessions · Validation 14,940 / **250** · Held out 9,030 / **151**.

The 22 that do not arrive are a **warmup burn-in, not a loss**, and that was checked
rather than assumed (2026-09-09):

- They are the first 22 sessions in the archive, 2020-01-03 … 2020-02-21, consecutively.
- `rv_prev_22` needs 22 prior sessions that do not exist yet. The nesting is exactly what
  a lookback warmup produces: `rv_prev_day` kills 1, `rv_prev_5` kills 5, `rv_prev_22`
  kills 22.
- All 1,320 dropped rows are 22 × 60 whole sessions. **Zero rows are dropped inside a
  surviving session.**
- Every one falls in the train block, so validation and held out are untouched and the
  bar in §3 is unaffected.

They are *less* volatile than the sample (median 7.44% vs 11.00% annualised), which is a
calendar fact about January 2020 being calm, not selection on volatility — the drop rule
never reads the target. Had the direction been the other way it would have mattered a
great deal, which is why it was measured.

Three data defects were found and fixed before these numbers were trusted; §4 records them
because each was **silent**, and the run printed a confident-looking table through all of
them.

---

## 1. The audit passes

`--audit` deliberately shifts every feature one minute into the future.

| run | HAR-RV-J QLIKE |
|---|---|
| honest | 0.541358 |
| audit (lookahead injected) | **0.530811** |

Audit better by **1.95%** → **PASS**. The honest features did not already contain that
minute. Equal scores would have meant the minute was already inside the honest run and
everything downstream was void.

Given this project lost an entire result set to a 1-minute lookahead that produced 95.7%
of a measured edge, this is the most important line in the file. It also **calibrates**:
one minute of genuine future information is worth ~2% QLIKE here, so a model later
claiming a much larger margin is claiming more than cheating with a minute of the future
would buy.

## 2. CORRECTION — the first version of this file overstated its own headline

**Reported first, then corrected on the same day.** The original §2 said *"the option
market beats HAR significantly (p=0.0222)."* **That was substantially an artifact of a
missing bias correction in my own code**, found while porting the model into the app.

`predict_har` computed `exp(X @ beta)`. For a log-space fit that is the conditional
**median**, not the mean: for a right-skewed variable, `E[exp(Z)] = exp(E[Z] + s²/2)`. The
`exp(s²/2)` factor was absent, so HAR forecast systematically **low** — and QLIKE penalises
under-forecasting asymmetrically, so the omission did not merely add noise, it made the
model look worse than it is.

Measured: `s² = 0.7056`, correction factor **1.423** — a 42% upward adjustment, large
because a 6-return realised-variance target is very noisy.

| | uncorrected | corrected |
|---|---|---|
| HAR-RV QLIKE | 0.541610 | **0.430642** |
| IV vs HAR-RV | −0.12958, t=−2.29, **p=0.0230** | −0.01957, t=−0.55, **p=0.5832** |

**The corrected conclusion:** implied variance and a properly specified HAR are
**statistically indistinguishable** at forecasting 30-minute realised variance. IV is still
marginally ahead in level (0.4111 vs 0.4306) but nothing in this sample separates them.

## 3. The bar, corrected

Validation (2023), target `rv_fwd_30`, lower QLIKE better:

| model | QLIKE | RMSE |
|---|---|---|
| implied_variance | 0.411136 | 0.016123 |
| **HAR-RV** | **0.430642** | 0.013771 |
| HAR-RV-J | 0.430788 | 0.013765 |
| persistence_rv30m | 0.803472 | 0.016252 |

Diebold–Mariano vs HAR-RV, clustered by session (n=250):

| model | mean dQLIKE | t | p |
|---|---|---|---|
| persistence_rv30m | +0.37350 | +8.27 | 0.0000 |
| implied_variance | −0.01957 | −0.55 | 0.5832 |
| HAR-RV-J | +0.00015 | +0.26 | 0.7956 |

HAR beats naive persistence decisively (t=+8.27). Nothing separates it from the option
market. Jump-robustness still adds nothing.

**§4's observation about the pre-registration naming the wrong benchmark is weakened but
not void:** IV is no longer *significantly* better than HAR, so §6's gate is defensible as
written. It remains true that a model beating HAR by a hair while sitting behind IV has
produced nothing useful, and that is still worth stating.

## 4. Three silent defects, found and fixed

Each produced no error and no crash. The pipeline reported success throughout.

**(a) The 1-minute archive has an unfillable hole.** SPY `stock_ohlc_1m` is missing
2024-01..2024-07, and refetching returns `NOT ENTITLED` — that layer is gone at any date,
despite the terminal reporting `Stock: STANDARD`. This alone capped the held-out block at
**5 sessions of 151**.

Fixed by moving to **5-minute bars**, which is a correction rather than a workaround:
1-minute equity returns carry microstructure noise that inflates realised variance, and
5-minute sampling is the long-standing standard (Andersen–Bollerslev) for that reason. The
5m archive is complete 2016–2026. Applied uniformly across all three blocks, so one
measurement definition covers the sample; the frame directory is cleared rather than mixed,
and the manifest now records `interval` / `bars_per_day` / `bars_per_year`.

**(b) A truncated month removed the COVID onset.** Native `SPY_2020-02` holds three days
where March holds twenty-two. That silently dropped ten sessions from the start of the
COVID volatility regime — including 2020-02-28 at **51.2%** annualised realised vol,
exactly the observations a variance model most needs to have seen.

Fixed by falling back per day to aggregating the 1-minute archive. **Verified as an
equivalence, not an approximation:** on 2020-02-04, where both archives have data, 1m
aggregated to 5m reproduced the native 5m closes with **max absolute difference 0.000000**
across all 79 overlapping bars.

**(c) One halt bar poisoned 170 later sessions.** SPY hit circuit breakers on 2020-03-09
and the archive fills 09:35 and 09:40 with all-zero OHLCV rows. `log(0) = −inf`, so a
single halt bar made that session's realised variance infinite — and because `rv_prev_22`
carries a 22-session window, it propagated into **170 subsequent sessions**. Downstream,
`dropna` removed **193 of 769 sessions** without a word.

Fixed by dropping non-positive closes at source, which is also the correct treatment: the
return then spans the halt (09:30 close → 09:45 close), the actual price change across it,
rather than inventing two enormous returns for volatility that never traded. 2020-03-09 now
reads **56.5%** annualised instead of infinity. `har_baseline` now prints what cleaning
costs and shouts when sessions are lost, so a silent quarter-sample loss cannot recur.

That counter now reads **22, all warmup** (see the note under the header). The shouting
is deliberately kept even though the remaining loss is benign: a line that only appears
when something is wrong is a line nobody recognises when it appears.

**(d) A safety net with a hole, found 2026-09-09.** `dm_test` falls back to a normal
approximation when scipy is missing — the case it exists for, since some compute nodes
lack it. The fallback called `np.math.erf`, and **`np.math` was removed in numpy 2.0**, so
it raised `AttributeError` and took the whole run down instead of degrading. It never
fired on the cluster only because scipy happens to be installed there. Fixed to stdlib
`math.erf`; the approximation costs about 0.0005 on a p-value at n=250 (0.5827 vs
scipy's 0.5832), which is the difference between a normal and a t with 249 df.

**Every number reported before 2026-09-09 on the 1-minute frame — including HAR-RV
0.187365 — was computed with defect (c) present and is superseded.** They are not deleted:
a record that quietly removes its own wrong numbers is not a record.

## 5. What is NOT concluded here

- Nothing about whether a neural model beats anything. The sweep is unrun.
- Nothing about tradeability. §6 requires the measured straddle spread applied before any
  edge is claimed, and no cost model has been run.
- Nothing about the held-out block, which per §6 stays untouched until validation passes.
