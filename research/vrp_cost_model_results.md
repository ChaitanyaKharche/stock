# Results — the 0DTE variance risk premium is real, and 1.84× too small to trade

**Run 2026-09-09.** `trade_analysis/hpc/cost_model.py`, 5-minute frame,
**37,517 trades over 618 sessions**, 2020-01-03 → 2023-12-29.
**The 2024 held-out block was never touched** — §6 seals it until a validation gate passes,
and this analysis does not need it.

This answers §6 of `vrp_preregistration.md`: *"the improvement surviving a realistic cost
model: the measured ATM straddle bid–ask spread at the forecast origin, not an assumed
one"* and *"A model that only wins gross of spread is a null."*

## 1. The verdict

Sell the ATM 0DTE straddle at each origin, hold 30 minutes, buy it back.

| | mean/trade | median | win rate | session-clustered t |
|---|---|---|---|---|
| **GROSS** (mid → mid) | **+$0.0205** | +$0.0700 | 68.6% | **+7.52** |
| **NET** (bid → ask) | **−$0.0171** | +$0.0400 | 61.0% | **−5.38** |
| spread cost | $0.0376 | $0.0300 | | |

Both halves matter and they say different things.

**The premium is real.** Gross of costs it is strongly significant — `t=+7.52` over 618
sessions, 68.6% of trades profitable. This is not a marginal effect and it is not noise. A
mid-fill backtest would print a convincing, publishable-looking edge.

**As a taker you lose, significantly.** `t=−5.38`. Not "roughly breakeven" — reliably
negative.

**The arithmetic is simple and decisive.** The ATM straddle mid is $1.79 at a 152 bp
spread. Round-tripping costs **$0.0376**. The gross edge is **$0.0205**. The spread is
**1.84× the edge.** You would pay nearly twice your edge for the privilege of collecting
it.

Selling at the bid and buying back at the ask is the entire point. Assuming mid fills is
the most common way an options backtest manufactures an edge that does not exist, and here
it is the difference between `t=+7.52` and `t=−5.38` — between a strategy and its opposite.

## 2. The part that would have fooled you for months

Net win rate is **61.0%** and the net median is **+$0.04**. The mean is still negative.

| percentile | net P&L |
|---|---|
| p0.1 | **−$2.295** |
| p1 | −$1.080 |
| p5 | −$0.482 |
| p50 | +$0.040 |
| p95 | +$0.240 |
| p99 | +$0.408 |

**The worst 1% of trades — 375 of 37,517 — account for 94% of the total net loss.**

That is the short-gamma payoff in its purest form: win a little, six times out of ten, then
give it all back at once. The p0.1 loss is **57× the median win**. Traded live, this
strategy would show a rising equity curve and a 61% hit rate for a long stretch, and feel
like it was working, right up until one session erased the record.

It is worth putting beside [`imb_stress_results.md`](imb_stress_results.md), where 74% of
P&L sat in the top 1% of trades. Same pathology, mirror image: **in this project, every
apparent edge so far has lived in the tail.** A strategy whose result is decided by 1% of
its observations is not characterised by its mean, and a mean estimated from 618 sessions
of such a distribution is not a stable quantity.

## 3. The trend is toward viability, and has not arrived

| year | trades | gross | net | spread | spread (bp) |
|---|---|---|---|---|---|
| 2020 | 9,641 | +0.0228 | −0.0346 | 0.0575 | 228 |
| 2021 | 2,352 | +0.0109 | −0.0207 | 0.0316 | 171 |
| 2022 | 10,338 | +0.0259 | −0.0133 | 0.0392 | 127 |
| 2023 | 15,186 | +0.0168 | −0.0080 | 0.0248 | 139 |

**Negative in every year**, but the gap is closing — and it is closing because **spreads
are tightening**, not because the premium is growing. Gross edge wanders without trend
(0.011–0.026) while spread cost falls by more than half (0.0575 → 0.0248). 2023 net is
−$0.008, roughly a third of 2020's loss.

Honest extrapolation: **if the 0DTE market keeps tightening, this becomes marginal rather
than clearly negative.** That is not a prediction and it is not permission to trade it. It
is the one direction in which this result could change, recorded so it can be re-tested
rather than re-argued.

## 4. What this settles, and what it costs

**§6 is answered, and it is a null of the strongest kind.** Not "nothing was found" but
"the thing was found, measured, and is 1.84× too small to harvest."

**It also disposes of the neural arm.** A forecasting model would have to beat the premium
itself by a factor approaching two, purely to reach breakeven — while
[`vrp_baseline_results.md`](vrp_baseline_results.md) shows implied variance already beating
HAR at forecasting realised variance (p=0.0222). There is no plausible route by which a
20k-parameter MLP on HAR-type features closes an 84% cost gap. **The GPU sweep was not
run, and that is the correct decision rather than an abandoned one.**

## 5. Limitations, stated plainly

- **No delta hedge.** An unhedged short straddle is a bet on magnitude *and* carries
  directional risk from drift. This measures the retail instrument, not a clean variance
  swap. A delta-hedged version isolates variance better and would be the natural
  refinement — its economics are not claimed here.
- **Taker assumption.** Someone who reliably earns mid, or a market maker, faces different
  economics; §1's gross column is their upper bound. For anyone crossing the spread, net is
  the trade.
- **Per-trade, not a portfolio.** Six positions overlap at h=30. These are per-trade
  economics clustered by session, not a return series, and no sizing or compounding is
  implied.
- **One horizon, one instrument, one symbol.** 30-minute hold, ATM straddle, SPY.
- **The fat tail makes the mean fragile.** §2 is not a footnote to §1; it is a reason to
  distrust any point estimate from this distribution, including the negative one.
