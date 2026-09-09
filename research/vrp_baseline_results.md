# Results — VRP baseline and lookahead audit

**Interim. The pre-registered experiment is NOT complete.** This records only step 2 of
`vrp_preregistration.md`: the benchmark, and the audit that has to pass before any number
here is believed. Run 2026-09-09 on Discovery, job `10209061`, 20 seconds on CPU.

Sample: 623 built sessions — train 71,850 origins / 241 sessions; validation 52,896 / 177;
**held out 1,500 / 5** (see §3, this is a problem).

## 1. The audit passes

`--audit` deliberately shifts every feature one minute into the future.

| run | HAR-RV QLIKE |
|---|---|
| honest | 0.187365 |
| audit (lookahead injected) | **0.184776** |

Audit better by **1.38%**. That is the PASS: the honest features did not already contain
that minute. Had the two been equal, the minute was already inside the honest run and
everything downstream would be void.

Given this project lost an entire result set to a 1-minute lookahead that produced 95.7%
of a measured edge, this is the single most important line in the file.

The gap is also a **calibration**: one minute of genuine future information is worth
~1.4% QLIKE here. Any model later claiming a much larger margin over HAR is claiming more
than cheating with a minute of the future would buy, and should be disbelieved until
re-audited.

## 2. The bar

Validation (2023), target `rv_fwd_30`, lower QLIKE better:

| model | QLIKE | RMSE |
|---|---|---|
| **HAR-RV** | **0.187365** | 0.012023 |
| HAR-RV-J | 0.187719 | 0.012016 |
| implied_variance | 0.207442 | 0.015499 |
| persistence_rv30m | 0.226525 | 0.012646 |

Diebold–Mariano vs HAR-RV, **clustered by session** (n=177, not n=52,896):

| model | mean dQLIKE | t | p |
|---|---|---|---|
| persistence_rv30m | +0.03913 | +5.80 | 0.0000 |
| implied_variance | +0.02016 | +1.04 | 0.2992 |
| HAR-RV-J | +0.00036 | +0.82 | 0.4108 |

**HAR-RV is the benchmark at 0.187365.** Jump-robustness adds nothing (p=0.41). Naive
persistence loses decisively (t=5.80), which is the sanity check that the pipeline
discriminates at all.

## 3. The finding that actually matters, and how not to misread it

**Implied variance loses to HAR numerically (+0.0202) but NOT significantly (p=0.2992).**
At 177 sessions we cannot distinguish the option market's variance forecast from HAR's.

It is tempting to read "IV is a worse forecast" as evidence against the variance risk
premium. **That inference is wrong, and getting it backwards would kill the experiment for
the wrong reason.**

Implied variance is a *risk-neutral* expectation. It contains the premium by construction.
A forecast that is systematically too high scores badly on QLIKE **precisely because** it
carries a premium — the bias IS the thing being harvested. QLIKE measures accuracy; the
VRP is a bias. They are different quantities and a model can be bad at one while the other
is real and tradeable.

Supporting number from the smoke sample: implied exceeded realised at **80.1%** of forecast
origins.

So this result neither confirms nor refutes the VRP. It establishes that **HAR is the
forecasting bar**, which is what §6 requires before the neural arm is justified.

## 4. Blocking: the held-out block is 5 sessions, not 151

The manifest shows **146 sessions returning `rows: 0, reason: no usable data`** — every
2024 session from 2024-01-02 to roughly 2024-07-31.

Cause: **the SPY minute-bar archive has no 2024-01 through 2024-07.** It jumps from
`2023-12` to `2024-08`. The 0DTE option quotes for those sessions exist; the underlying
bars they must be paired with do not, so `build_session` correctly returns nothing.

    stock_ohlc_1m/SPY/2024/  ->  2024-08 2024-09 2024-10 2024-11 2024-12   (5 of 12)

**This is fixable and must be fixed before the held-out test means anything.** §7 stated
power at n=151 held-out sessions; at n=5 there is no test at all. And unlike the option
archive, **the stock archive can still be extended** — the ThetaData subscription is
`Stock: STANDARD`, only `Options: FREE`. The seven missing months are re-downloadable.

    python -m trade_analysis.bulk_download.download --layers stock.ohlc.1m \
        --symbols SPY --start 2024-01-01 --end 2024-07-31

Then rebuild; `build_vrp_dataset` skips nothing and overwrites cleanly.

## 5. What is NOT concluded here

- Nothing about whether a neural model beats HAR. That is the sweep, still unrun.
- Nothing about tradeability. §6 requires the measured straddle spread applied before any
  edge is claimed, and no cost model has been run.
- Nothing about the held-out block, which per §6 stays untouched until validation passes,
  and which currently could not be tested even if it did.
