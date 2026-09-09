# Results — VRP baseline and lookahead audit

**Interim. The pre-registered experiment is NOT complete.** This records step 2 of
`vrp_preregistration.md`: the benchmark, and the audit that must pass before any number
here is believed.

**Final frame:** 769 sessions, **5-minute bars**, zero sessions dropped.
Train 20,634 origins / 346 sessions · Validation 14,940 / **250** · Held out 9,030 / **151**.

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

## 2. The headline: the option market beats HAR

Validation (2023), target `rv_fwd_30`, lower QLIKE better:

| model | QLIKE | RMSE |
|---|---|---|
| **implied_variance** | **0.411136** | 0.016123 |
| HAR-RV-J | 0.541358 | 0.013963 |
| HAR-RV | 0.541610 | 0.013968 |
| persistence_rv30m | 0.803472 | 0.016252 |

Diebold–Mariano vs HAR-RV-J, **clustered by session** (n=250, not n=14,940):

| model | mean dQLIKE | t | p |
|---|---|---|---|
| persistence_rv30m | +0.26372 | +7.82 | 0.0000 |
| **implied_variance** | **−0.12934** | **−2.30** | **0.0222** |
| HAR-RV | +0.00024 | +0.29 | 0.7709 |

**Implied variance forecasts 30-minute realised variance significantly better than HAR.**
Stable at the secondary horizon, so it is not an artifact of the short window:

| horizon | returns in window | IV | HAR-RV | IV vs HAR |
|---|---|---|---|---|
| h=30 | 6 | 0.411136 | 0.541610 | −0.1296, t=−2.29, p=0.0230 |
| h=60 | 12 | 0.297063 | 0.403153 | −0.1055, t=−1.96, p=0.0507 |

## 3. The pre-registration named the wrong benchmark

§4 committed to HAR as the thing to beat, calling it "the standard in the
realised-volatility literature and hard to beat." **In this data it is not the hard
benchmark. Implied variance is, and it beats HAR.**

That is a design flaw in the pre-registration, found by running it. §9 forbids changing the
gate after seeing results, so **§6 stands exactly as written** — but it is now known to be
too weak: a model can pass it while still being worse than reading the option chain.

**Stated as a post-hoc observation, not a pre-registered criterion:** any model that beats
HAR but not implied variance has produced nothing of use, because the free alternative is
to read the quote that is already on the screen.

**This does not refute the variance risk premium, and reading it that way would be an
error.** IV can be simultaneously (a) more informative about the level of variance than
HAR and (b) systematically above realised variance. Those are different properties —
accuracy and bias — and the premium lives in the second. Supporting number: implied
exceeded realised at **80.1%** of origins in the smoke sample.

What it does change is the **expected value of the neural arm**. If the option market
already forecasts variance better than the standard econometric model, a small MLP on
HAR-type features is unlikely to beat it, and the interesting question moves from *can I
forecast variance better than the market* to *is the premium harvestable after the
measured spread* — which needs the cost model, not a GPU.

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

**Every number reported before 2026-09-09 on the 1-minute frame — including HAR-RV
0.187365 — was computed with defect (c) present and is superseded.** They are not deleted:
a record that quietly removes its own wrong numbers is not a record.

## 5. What is NOT concluded here

- Nothing about whether a neural model beats anything. The sweep is unrun.
- Nothing about tradeability. §6 requires the measured straddle spread applied before any
  edge is claimed, and no cost model has been run.
- Nothing about the held-out block, which per §6 stays untouched until validation passes.
