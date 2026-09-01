# Specification — Live Paper-Trading Laboratory (QQQ 0DTE)

**Status: DRAFT FOR APPROVAL. No code written. Nothing frozen until you sign off.**
Written 2026-08-27. Appendices A and B appended same day after setup research completed.

**This system will never place an order.** It observes, decides, records, and prices
hypothetical positions. No broker credentials, no order routing, no execution. That is a
design boundary, not a phase-1 limitation.

---

## 0. BLOCKERS — read before anything else

| blocker | detail | consequence |
|---|---|---|
| **Option Data cancels 2026-09-05** | 9 days out | The entire options half dies. Underlying signals keep working; **not one contract can be priced.** Must be renewed or this is an underlying-only study. |
| **ThetaData key is in a chat transcript** | pasted 2026-08-26 | Rotate before the runner holds it in a long-lived process. |
| **Terminal binds 0.0.0.0:25503** | LAN-exposed paid feed | Set `host` to `127.0.0.1` in `config.toml` before running a persistent service. |
| **No Greeks/IV on your tier** | all routes 404 | IV and delta computed via Black-Scholes from the mid. Recorded as `iv_derived` / `delta_derived`. Never labelled vendor data. |

---

## 1. The honest power calculation — read this before approving

This decides whether the lab is worth building.

Per-trade option return SD in this project is **~52%**. To detect a mean return of **+10%**
(roughly the size that would pay $400-500/week at 4 ATM contracts) at 80% power, corrected
across the final family of **13** setups (Holm, worst-case alpha = 0.00385, z = 2.89):

```
n = ((2.89 + 0.84) x 52/10)^2  ~=  377 completed trades PER SETUP
```

| what you want to detect | trades needed per setup |
|---|---|
| +20% mean return (huge) | ~94 |
| **+10% (pays ~$450/wk)** | **~377** |
| +5% (your historical rate) | ~1,507 |

**Per-setup calendar varies enormously with signal rate — see Appendix B.** The fastest
(`MOMO_CHASE`) reaches a verdict in **~3.4 months** on QQQ+SPY; the slowest never does.
Anything any setup says before ~200 trades is noise. If that timeline is unacceptable, say so
now and we design something else instead of discovering it in November.

The one thing it does deliver immediately: **a clean prospective record.** Every prior number
in this project is in-sample and backward-looking. From day one this is not.

---

## 2. Setups (item 1 + 2 + 3 + 4)

Every setup below is evaluated **only at the close of a completed bar.** Definitions are
frozen by SHA-256 hash. **Changing any parameter forks the setup into a new id with a fresh
history — old trades are never retro-fitted to a new definition.**

### Your three, taken verbatim — not redefined

**`ORB_5min`** *(source: edgeful)*
| | |
|---|---|
| range | high/low of 09:30-09:35 ET |
| gate | range width >= 0.15% of the 09:35 price |
| entry | first 1-min **close** beyond the range, AND that bar's volume >= 1.5x the trailing 10-bar average |
| direction | long above the range, short below |
| stop | opposite side of the range |
| target | 1.5R |
| cutoff | no new entries after 11:00 ET |

**`ORB_15min`** *(source: journalplus)*
| | |
|---|---|
| range | high/low of 09:30-09:45 ET |
| entry | first 1-min **close** beyond the range AND price on the correct side of session VWAP |
| stop | mid-range ((high+low)/2) |
| target | 2R |
| cutoff | no new entries after 10:30 ET |

**`VWAP_Reclaim`**
| | |
|---|---|
| entry | 1-min close crosses session VWAP by a buffer of **0.05% of price**, AND the cross direction agrees with the sign of the 20-bar EMA slope |
| stop | last 5-bar swing (min low / max high of the prior 5 bars) |
| target | 1.5R |
| cap | **max 2 trades per ticker per day** |

### Your discretionary behaviour, reconstructed — `MOMO_CHASE`

Defined **from your measured behaviour only**. Every threshold below is a number this project
already measured from your tape, not a number chosen to make your history look good.

| component | value | where it came from |
|---|---|---|
| instrument | QQQ | 279 of your 357 index round trips |
| window | **10:30-14:30 ET** | your median entry is 11:39 |
| trigger | trailing **15-min return >= +0.20 sigma** in the trade direction | your measured median is **+0.2029 sigma**, present on 89.3% of entries |
| sigma | that session's realised 1-min return SD x sqrt(15) | — |
| filter | MACD(9,17,9) histogram sign aligned with direction | fires on 69.4% of your entries |
| filter | (+DI - -DI) sign aligned, Wilder 14 | 78.5% |
| filter | ADX(14) > 20 | 69.6% |
| **volume** | **NO VOLUME CONDITION** | you believe it confirms; your median entry is **0.986x** the 20-EMA and only 48.3% clear it. Including it would be reverse-engineering a belief, not a behaviour |
| direction | call after bullish momentum, put after bearish | matches your tape |
| exit | **25-minute time exit** | your stated typical hold |
| cap | max 3 trades/day | your observed rate is ~2.6/session |

**This is a hypothesis and is expected to fail.** Nine pre-registered tests found no
predictive content in these exact indicators at these exact entry times, including a direct
test of this stack (OOS R2 **-0.062**, AUC **0.392**). It is included because you asked for it
to earn its place, and because a prospective test is the one thing never done. **It gets no
special treatment and no post-hoc adjustment.**

### Additional public setups

**Nine, in Appendix A**, with sources, every parameter an explicit number, and every
researcher-chosen value tagged **[R]**. Seven further candidates were **excluded with the
reason recorded**, including bull/bear flag (cannot be evaluated without future information)
and gap-and-go (published parameters are small-cap only).

---

## 3. Contract selection (item 5)

**ATM is defined once and never varies:**

> **ATM = the listed strike with the smallest absolute distance to the underlying's NBBO
> midpoint at the signal bar's close timestamp.** Ties break to the **lower** strike.
> `ATM-1` and `ATM+1` are the adjacent listed strikes by strike ladder position, not by dollars.

| rule | |
|---|---|
| expiry | **0DTE only** — same-day expiration. If none is listed, the signal is recorded as `SKIPPED: no_0dte` |
| strikes tracked | **ATM, ATM-1, ATM+1 — all three, on every signal, always** |
| primary arm | **ATM.** Declared now. The +/-1 strikes are recorded for measurement and are **never** promoted after the fact |
| direction | call for long signals, put for short. **Long premium only. The system cannot express a short-option position.** |
| entry price | the contract's **ask** at decision time |
| exit price | the contract's **bid** at exit time |
| rejection | if spread > 25% of the mid, or bid <= 0, record `SKIPPED: unusable_quote` and do not open |

**Anti-cherry-pick guarantee:** the primary arm is ATM, fixed in this document before any
data exists. ATM+/-1 results are reported in a separate descriptive table that **cannot be
promoted to primary** under any result.

---

## 4. Hindsight prevention (item 10) — the architectural core

**The decision is written to disk before the outcome is knowable.** This is the central
guarantee and it is structural, not a convention:

```
bar closes at T          ->  wait SETTLE_MS (default 1500ms) for late prints
evaluate setups at T     ->  using ONLY bars with close_time <= T
if signal:
    1. assign uuid, write DECISION record to signals.jsonl  <-- flushed & fsync'd
    2. ONLY THEN fetch the option chain
    3. write the FILL record referencing that uuid
```

A decision is durable on disk before any price that could influence it is requested. **No
code path can reorder these.**

| rule | enforcement |
|---|---|
| bar-close only | a bar stamped 09:30:00 covers 09:30:00-09:30:59.999 and becomes usable at **09:31:00 + settle**. The forming bar is never readable by setup code |
| no future bars | setup functions receive an immutable slice ending at the signal bar; the buffer is append-only and slices are copies |
| no future quotes | option quotes are fetched **after** the decision record is fsync'd |
| exits | evaluated only on subsequent polls, never on the entry poll |
| stale quotes | if a quote timestamp is older than **5s** during RTH, mark `stale` and skip the action. **Never interpolate. Never fabricate.** |
| clock | all timestamps stored as ET **and** as UTC epoch ms. The ThetaData snapshot stamps are ET; verified against `/v3/stock/history/ohlc` |
| setup isolation | each setup is a pure function of `(bars, session_state, its own config)`. It cannot read another setup's state, signals, or P&L |
| replay test | before go-live, every setup is replayed against a historical day and must produce **identical** signals to a from-scratch batch computation. A mismatch blocks the freeze |

---

## 5. Data sources (item 6)

Verified against the running terminal on 2026-08-27:

| need | endpoint | verified |
|---|---|---|
| underlying NBBO | `/v3/stock/snapshot/quote?symbol=QQQ` | 719.69/719.73, sub-second stamps |
| session OHLC | `/v3/stock/snapshot/ohlc?symbol=QQQ` | ok |
| 1-min bars (warmup + VWAP) | `/v3/stock/history/ohlc?interval=1m` | ok |
| **full 0DTE chain** | `/v3/option/snapshot/quote?symbol=QQQ&expiration=<0DTE>` | **333 rows, 310 ms** |
| single contract | same + `&right=&strike=` | ok |
| open interest | `/v3/option/snapshot/open_interest` | ok |
| Greeks / IV | **none exist on this tier** | computed Black-Scholes from mid, stored as `*_derived` |

**Poll cadence:** chain every **5s** while any position is open, else every **15s**.
Underlying quote every **1s**. One chain call covers all setups and all three strikes — the
cost does not scale with setup count.

**Every setup needs only:** 1-min QQQ OHLCV, session VWAP, and the indicator set
(EMA 9/20, MACD 9/17/9, ADX/DMI 14, ATR 14, RVOL vs 10- and 20-bar). All derived from one bar
stream. No setup requires a data source another does not.

**Resilience:** exponential backoff reconnect (1s -> 60s cap); every gap written to
`outages.jsonl` with start, end, and duration; duplicate suppression keyed on
`(symbol, timestamp, endpoint)`; a bar with a gap in its inputs is marked `degraded` and
setups are **skipped** for that bar rather than run on partial data.

---

## 6. What is measured (item 7)

**Per signal** — including every rejected one, with the reason:
`uuid, seq, config_hash, setup_id, ts_et, ts_utc_ms, direction, underlying_bid/ask/mid,
session_vwap, ema9, ema20, macd_hist, adx, plus_di, minus_di, atr14, rvol10, rvol20,
range_high/low, bars_since_open, skip_reason`

**Per hypothetical trade** — for each of ATM, ATM-1, ATM+1:
`expiration, strike, right, entry_bid/ask/mid/spread, spread_pct_of_mid, iv_derived,
delta_derived, open_interest, entry_ts, exit_ts, exit_bid/ask/mid, exit_reason
(target|stop|time|eod|no_quote), hold_minutes, pnl_gross, pnl_net (fees $0.0404/contract/side),
return_pct, MFE, MAE, MFE_time, MAE_time`

**Underlying-only twin — the decomposition you asked for.** Every signal also records the
underlying's forward path over the identical holding period: `underlying_return,
underlying_return_sigma, underlying_MFE, underlying_MAE`.

This separates **"the setup has no directional edge"** from **"the setup is directionally
right and the option structure eats it."** Those demand opposite responses and no prior test
in this project could tell them apart.

**Per setup, rolling:** signals, trades, skips by reason, win rate, avg win, avg loss,
expectancy, mean/median return, max drawdown, mean hold, mean entry spread, mean exit spread,
underlying-twin expectancy.

---

## 7. Promising vs failure (items 8 + 9) — committed now

### Inference families, declared before data exists

- **Family A (primary), m = number of setups:** option expectancy vs 0, ATM arm, native
  exit. Holm across all setups.
- **Family B (co-primary, separate question), m = number of setups:** underlying-twin
  expectancy vs 0. Holm across all setups.
- Everything else — ATM+/-1, alternate exits, per-regime cuts, cross-setup comparisons — is
  **descriptive, reported without p-values**, and cannot be promoted.

Day-clustered bootstrap, 10,000 reps, as in every prior test here.

### PROMISING requires all four

1. **>= 200 completed trades** for that setup. Below this the result is reported as
   `INSUFFICIENT`, whatever it looks like.
2. Clears **Holm within its family**.
3. Effect **exceeds its own achieved MDE** — the same underpowered rule used throughout this
   project, not relaxed.
4. **Survives a split:** first half vs second half of its own trade sequence, same sign, and
   the second half's CI excludes 0. Declared now so it cannot be waived later.

### FAILURE

- No setup satisfies all four after **200 trades each** -> **no mechanical setup in this
  panel has a detectable edge.** Report and stop.
- Family B null while Family A null -> the setups have no directional content; the options
  are not the problem.
- Family B positive while Family A null -> direction is real, **the 0DTE option structure
  destroys it.** That would redirect the whole project toward instrument choice.
- `MOMO_CHASE` fails while a mechanical setup passes -> your discretionary process is not the
  source of your record.

### Anti-overfit rules, binding on me

- **No setup is ever deleted, disabled, or hidden for performing badly.**
- **No definition is edited in place.** A change forks a new `setup_id` with an empty history.
- **No setup is added mid-experiment** without its own dated amendment, and it enters the
  family — the family size never shrinks.
- **No parameter is swept.** Sweeping is what produced this project's earlier false positives.
- **No result is reported without its trade count beside it.**
- The dashboard shows P&L (you asked, and you would see it anyway), but **ranking is by
  expectancy with CI, never by raw P&L**, because setups fire at wildly different rates.

---

## 8. Persistence and dashboard

```
live_lab/
  config/<config_hash>.json        frozen definitions + git sha + spec version
  events.jsonl                     append-only, monotonic seq, every observation
  signals.jsonl                    every signal incl. rejected, with skip_reason
  positions_open.json              atomic-replace snapshot for crash recovery
  trades.jsonl                     completed, one line per (signal x strike arm)
  outages.jsonl                    disconnects, stale-quote windows, degraded bars
  daily/<YYYY-MM-DD>.json          end-of-session summary
```

Append-only + fsync on decision records. **Restart replays `positions_open.json` and resumes;
a 3-hour crash loses nothing.** Every trade carries `uuid`, `setup_id`, `config_hash`.

Dashboard (local, read-only over the JSONL): today's market state, live signals by setup,
open hypothetical positions, P&L by setup, `MOMO_CHASE` broken out, signal and trade counts;
plus the historical table ranked by **expectancy with CI and n**, never by raw P&L.

---

## 9. What I need from you before freezing

1. **Is the Option Data subscription staying alive past Sep 5?** Everything on the options
   side depends on it.
2. **Is a ~9-month horizon acceptable?** If not, we should design a different instrument now.
3. ~~QQQ only, or QQQ + SPY?~~ **Answered by Appendix B: QQQ + SPY is mandatory.** On QQQ
   alone only 3 of 13 setups reach a verdict inside 18 months; with SPY, 9 of 13 do. This is
   arithmetic, not preference. Confirm you are happy with SPY included.
4. **Approve or amend the `MOMO_CHASE` definition** — it is my reconstruction of your
   behaviour and you are the authority on whether it resembles what you actually do.

Once you answer these and the researched setups land, I freeze the spec, hash it, and build.

---

# APPENDIX A — Researched setups (appended 2026-08-27)

Sourced with explicit parameters. Tags: **[R]** = parameter chosen by the researcher, not by a
source. Every **[R]** is a place where we invented a number, and each is a candidate false
positive if it later "works."

All 5-min bars are aggregated from the 1-min feed and **stamped at close** (09:30-09:35 -> 09:35).

### A1. `IntradayMomentumBoundary` — the strongest item found
*Zarattini/Aziz/Barbon, [SSRN 4824172](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4824172); [Concretum](https://concretumgroup.com/backtesting-riding-intraday-trends-in-us-markets-using-matlab/)*

Peer-reviewed, zero discretionary language, fires near-daily.
```
sigma(m) = mean over 14 PRIOR sessions of |Close_d(m)/Open_d(0930) - 1|   (today excluded)
UB(m) = max(Open_0930, PrevClose) * (1 + 1.0*sigma(m))
LB(m) = min(Open_0930, PrevClose) * (1 - 1.0*sigma(m))
```
**Entry** only at bar closes stamped `HH:00`/`HH:30`, 10:00-15:30 (9 checkpoints/day):
long if `Close > UB` and `Close > VWAP`; short if `Close < LB` and `Close < VWAP`.
**Stop** trailing, checked each minute: long exits on first close below `max(VWAP, UB)`.
**Target** none — trailing stop only. **R is undefined**; record realised move / initial stop.
**Flat 15:59.** Cap **4/day [R]**.

### A2. `TTM_Squeeze` (5-min)
*[StockCharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ttm-squeeze)*

BB(20, 2.0) vs KC(20 EMA, 1.5xATR20). **Squeeze ON**: `UpperBB < UpperKC` and `LowerBB > LowerKC`.
**Fires** when ON at `t-1` and OFF at `t`.
Momentum `delta(t) = Close - ((HH20+LL20)/2 + SMA20)/2`, then 20-bar linear-regression **endpoint**.
**Entry** at close of firing bar: long if `hist>0` and `hist>hist(t-1)`.
**Stop** `entry -/+ 1.5*ATR20` **[R]**. **Target 2.0R [R]** + **time stop 9 bars** (source says 8-10).
No entries after 15:00, flat 15:55 **[R]**. Cap **2/day [R]**.

> Implementation trap: the regression endpoint must be at bar `t`. A centred/refit regression is lookahead.

### A3. `PDH_PDL_Breakout` (5-min)
*[NetPicks](https://www.netpicks.com/previous-day-high-low-strategy/)*

`PDH`/`PDL` = prior **RTH-only** high/low. **Entry** at close of first bar with
`Close >= PDH + 0.2*ATR14` (buffer = midpoint of NetPicks' 0.1-0.3 range) **and** body filter
`|Close-Open| >= 0.5*(High-Low)` **[R]**. **Stop** = that bar's Low. **Target 2.0R**.
No entries after 15:00 **[R]**. Cap **1 per direction**.

### A4. `PDH_PDL_FailedBreak` (5-min)
*[TradeMomentum](https://www.trademomentum.org/blog/failed-breakout-reversal-strategy)*

> **Honest caveat: the sources are almost entirely qualitative here. Nearly every number is [R].**
> Included as the natural counterpart to A3, not because a source specified it.

Short at PDH: some bar `p<=t` had `High >= PDH + 0.1*ATR14`; `Close(t) < PDH`; `t-p <= 12` bars **[R]**.
**Stop** = running session high as of `t` (never the eventual day high). **Target 2.0R**.
Entries 09:45-15:00 **[R]**. Cap **1 per level**.

### A5. `VWAP_2sigma_Fade` (5-min)
*[CrossTrade](https://crosstrade.io/learn/trading-strategies/vwap-reversion)* — the true opposite of `VWAP_Reclaim`

`sigma_v(t) = sqrt(SUM v_i*(typ_i - VWAP_t)^2 / SUM v_i)`, bands at **2.0 sigma_v**, cumulative from 09:30.
**Entry (short)**: `High >= Upper`, `Close < Upper`, `Close < Open`, upper wick `>= 0.5*range` **[R]**,
`Volume >= 1.5x` 20-bar mean **[R]**.
**Regime skips (both source-stated):** `ADX(14) > 25`; or 09:30-10:30 range `> 2.0x` its 20-day average.
**Stop** `High + 1.0*ATR14`. **Target** session VWAP (R is non-constant — record realised R).
Entries **10:00-11:30 and 13:30-14:30** only. Cap **3/day**.

> **Source rule deliberately NOT applied:** CrossTrade restricts this to Wednesday/Thursday.
> Unexplained day-of-week seasonality with no mechanism, cutting the sample 60%. **Day-of-week is
> recorded as a covariate instead.** This lab exists to falsify that kind of filter, not inherit it.

### A6. `EMA_9_20_Pullback` (5-min)
*[Bulls on Wall Street](https://www.bullsonwallstreet.com/post/first-pullback-trading-strategy)*

Long: `EMA9>EMA20`, `EMA20(t)>EMA20(t-3)` **[R]**; a prior bar this session had `Low>EMA9`; bar `t` is the
**first** with `Low<=EMA9` after that; `Close>=EMA20`, `Close>Open`, `Close>EMA9` **[R]**;
`Volume <` 5-bar mean **[R]**.
**Stop** = `min(Low(t), EMA20(t))` — the **wider** of the source's two options, stated.
**Target 3.0R** (source). Entries **09:45-11:00**. Cap **1 per direction**.

> Flagged by the researcher: 3R on 5-min QQQ is low-hit-rate and slow to separate from zero.

### A7. `ThreeBarPlay` (5-min)
*[TradingSim](https://www.tradingsim.com/blog/3-bar-play), [HowToTrade](https://howtotrade.com/chart-patterns/3-bar-play/)*

Igniting bar: `Close >= Low + 0.8*range` (source), `Volume >= 1.5x` 20-bar mean (source),
`range >= 1.5*ATR14` **[R]**. Pullback: 1-2 inside bars, each `Low >= High(i) - 0.5*range(i)` (source).
**Entry** at close of next bar if `Close > max(pullback highs)`. **Stop** = last pullback low. **Target 2.0R**.
Entries 09:35-15:00 **[R]**. Cap **3/day [R]**.

> **Deviation from source, stated:** the source triggers intrabar on a stop order. Converted to a
> bar **close** per our no-lookahead rule. **This is a different trade** — later entry, worse price,
> higher quality. Do not compare against the source's claimed win rate.

### A8. `Crabel_Stretch`
*Crabel TASC V.6:9; [Oxford Strategies](https://oxfordstrat.com/trading-strategies/opening-range-breakout/)*

```
Noise(d) = min(Open-Low, High-Open) on prior RTH daily bars
Stretch  = SMA(Noise, 10)
Buy/Sell = Open_0930 +/- 1.0*Stretch
```
Multiple **1.0** = Crabel's original. (Oxford Strat's tested 2.0 was fit on 42 *futures* markets with
daily holds — not transferred.)
**Entry** first 1-min close beyond a level. **Stop** = the opposite level. **Target** none — exit
15:55 or opposite level. Entries 09:31-11:00 **[R]**. Cap **1/day** (structural).

> **Overlap warning:** strongly correlated with `ORB_5min` in outcome. Track trade-level overlap.
> **Deviation:** Crabel uses resting stop orders (intrabar). Converted to bar close.

### A9. `Gap_Fade`
*[SharePlanner](https://www.shareplanner.com/blog/strategies-for-trading/fading-the-gap-how-large-overnight-moves-in-spy-and-qqq-play-out-during-the-trading-day.html), [QuantifiedStrategies](https://www.quantifiedstrategies.com/gaps/)*

Replaces gap-and-go, which was excluded. Across 6,005 QQQ sessions: 0.5-0.99% gaps fill same-day
**77%** (down) / **72%** (up); after a >=1% gap **up** the average open-to-close drift is **-0.5%**.
**Continuation is the weaker side on QQQ** — the published gap-and-go thesis points the wrong way here.

Qualify `|Gap%| >= 0.5%` (0.5% chosen over 1.0% for frequency **and** better fill rate; tradeoff stated).
Gap up -> short. **Entry** at first 5-min close back through `Open_0930` by a 0.05% buffer **[R]**,
window 09:45-12:00 **[R]**. **Stop** running session high/low. **Target** = prior RTH close.
Cap **1/day**.

## EXCLUDED, with reasons — recorded so they are not silently revisited

| setup | why excluded |
|---|---|
| **Bull/bear flag** | Every published definition is **retrospective**. "The consolidation ended" is only knowable after the breakout — **cannot be evaluated without future information.** Quantified versions exist only in vendor marketing with no verifiable parameters. `ThreeBarPlay` covers the same impulse-rest-impulse rhythm with stated numbers. |
| **Gap-and-go as published** | Parameters are written for low-float small caps: 2-4% gaps, 3x RVOL, 100k pre-market shares, "$5-$100", "clear catalyst". QQQ rarely gaps 2%+ and catalyst is not mechanizable. Transplanting them would be inventing parameters and attributing them to a source. Replaced by A9. |
| **NR7 / inside day** | Clean definition but a **daily** pattern: <2 signals/month on QQQ after the 20-EMA filter. Source also says "do not follow the rules mechanically" — disqualifying here. `TTM_Squeeze` captures the same volatility-contraction thesis intraday. |
| **Opening drive** | No source gives a single number. Also a lookahead hazard: "no return to the open within 15 min" is only confirmable at 09:45. Would require inventing 4+ parameters. |
| **Bollinger/Keltner MR** | Sources disagree on every parameter (EMA 10/15/20, mult 1.5/2.0, one gives an unusable 5.5xATR35 stop). `VWAP_2sigma_Fade` is the same thesis with one self-consistent set. |
| **ADX/DMI breakout** | Every source calls ADX a **filter**, not a signal. Already used that way in A5. |
| **Connors RSI(2)** | Fully mechanical but a **daily** swing system, 3-7 day holds. No credible intraday parameterization. |

## Cross-cutting implementation rules — binding

1. **13 setups, ~3 independent bets.** A1/A3/A8/ORB_5min/ORB_15min are all early-session directional
   breakouts; A5/A9 are both fades; A6/A7 are both continuation-after-impulse. **Correlated positives
   will look like confirmation and are not.** Trade-level overlap and pairwise return correlation are
   computed and reported alongside every result. This is a reporting requirement, not optional.
2. **RVOL is time-of-day normalised:** cumulative volume to minute `m` / mean cumulative volume to the
   same minute `m` over the prior 20 sessions. Dividing by a full-day average makes every morning bar
   look high-volume — a normalisation bug, not a signal.
3. **Prior-day levels are RTH-only.** Including 04:00-09:30 / 16:00-20:00 silently changes QQQ's
   PDH/PDL and degrades A3, A4, A9.
4. **VWAP and its sigma-bands are cumulative from 09:30**, never full-session. This is the single most
   common lookahead bug in the VWAP family.
5. **A1, A5, A8 have no fixed R by design.** MFE/MAE are logged per trade so R-based comparisons can be
   reconstructed without re-running.
6. **A7 and A8 deviate from source** (intrabar stop -> bar close). Their results are **not** comparable
   to the sources' published win rates and will never be presented as such.
7. **Sanity benchmark:** an independent 2-year QQQ study found ORB fires ~1.78x/day with 49-57% win
   rates and **near-zero expectancy per trade**. Any setup here reporting materially better on a small
   sample is treated as noise until it clears the section 7 bar.

---

# APPENDIX B — Time to verdict. This changes the design.

Family is now **m = 13** (3 fixed + `MOMO_CHASE` + 9 researched). Holm worst-case alpha = 0.00385,
z = 2.89. At the **+10% effect that would pay ~$450/week**: **n = 377 trades per setup.**

| setup | signals/mo (1 symbol) | months, QQQ only | months, QQQ+SPY |
|---|---|---|---|
| MOMO_CHASE | 55 | 6.8 | **3.4** |
| ORB_5min | 37 | 10.2 | **5.1** |
| IntradayMomentumBoundary | 21 | 17.9 | **9.0** |
| Crabel_Stretch | 21 | 17.9 | **9.0** |
| ORB_15min | 20 | 18.8 | **9.4** |
| VWAP_Reclaim | 16 | 23.5 | **11.8** |
| VWAP_2sigma_Fade | 8-20 | 26.9 | **13.5** |
| TTM_Squeeze | 8-16 | 31.4 | **15.7** |
| PDH_PDL_Breakout | 8-14 | 34.2 | **17.1** |
| EMA_9_20_Pullback | 6-12 | 41.8 | 20.9 — too slow |
| ThreeBarPlay | 4-10 | 53.8 | 26.9 — too slow |
| PDH_PDL_FailedBreak | 4-8 | 62.8 | 31.4 — too slow |
| Gap_Fade | 4-6 | 75.3 | 37.7 — **will never reach a verdict** |

**Two design consequences, both binding:**

1. **QQQ + SPY is mandatory, not a preference.** On QQQ alone only 3 of 13 setups reach a verdict
   inside 18 months. Adding SPY halves every timeline and brings 9 of 13 within reach. Section 9
   question 3 is therefore answered by the arithmetic, not by taste.
2. **Four setups are recorded but cannot conclude.** A6, A7, A4 and A9 will sit at `INSUFFICIENT`
   for 2-3+ years. They are **kept** — deleting them would be exactly the selective reporting this
   spec forbids — but they are labelled **`SLOW: verdict not expected before <date>`** in the
   dashboard from day one, so neither of us mistakes an early reading for evidence.

**The realistic first verdict is `MOMO_CHASE` at ~3.4 months, then `ORB_5min` at ~5.1.** Everything
else is a 2026-2027 answer.

---

# APPENDIX C — Build report and measured calibration (2026-08-27)

The lab is built at `trade_analysis/live_lab/` (~2,900 lines, 10 modules). It imports
nothing from `trade_analysis.live_trading` and contains **no order-placement code path and
no credentials** — verified by grep, not by assertion.

## C1. Two defects the replay gate caught before any data was collected

### C1.1 The probe itself was wrong (found on its first run)

The gate flagged `MOMO_CHASE` as inconsistent at 14:05. Cause: **1-minute bars carry the
vendor's OPEN stamp while 5-minute buckets are stamped at their CLOSE**, so a naive
truncation handed the probe one extra bar. Fixed with timeframe-aware truncation.

The probe's claim was also overstated and has been narrowed. It **cannot** prove "no
lookahead" by feeding a setup the future, because the architecture makes that impossible:
a `Context` is built from a buffer that *ends* at the evaluation bar, so future bars are
absent rather than merely unused. What it actually verifies is (a) that structural
invariant, asserted on every single evaluation, and (b) order-dependence and leaked state,
via a one-shot rebuild. That is a narrower claim and it is the true one.

### C1.2 Indicators were resetting at 09:30 — the serious one

Indicator state was being seeded from the current session only. **Real charts do not reset
a moving average at the open**, and a continuous chart is what the trader actually watches.

| setup | earliest it could fire | its window | consequence |
|---|---|---|---|
| `EMA_9_20_Pullback` | 11:35 | 09:45-11:00 | **structurally impossible — 0 signals in 40 symbol-days** |
| `MOMO_CHASE` | 12:25 | 10:30-14:30 | **could not fire at his 11:39 median entry** |
| `TTM_Squeeze` | 13:00 | to 15:00 | lost every morning |
| `ThreeBarPlay` | 11:20 | 09:35-15:00 | lost the session open |

The `MOMO_CHASE` case is the one that mattered: the reconstruction of his own trading could
not fire at the hour he trades. Nine months would have been spent measuring a distorted
version of him.

**Fix:** indicators are seeded from `PREFIX_SESSIONS = 2` completed prior sessions, while
everything session-scoped — VWAP and its sigma bands, opening range, gap, session
high/low — stays anchored to today. The VWAP-anchoring guarantee is therefore untouched.
5-minute buckets are aggregated **per session** so none straddles the overnight break.

**Accepted consequence, recorded as a modelling choice:** ATR and ADX now see the overnight
gap as one large true range on the session's first bars, exactly as they do on any
continuous chart. This is the standard convention, not an oversight.

Effect on `MOMO_CHASE` signal timing (8 QQQ sessions):

```
10:00   4        pre-fix: nothing was possible before 12:25
11:00  18        his median entry is 11:39
12:00  17
13:00  12
14:00   7
```

## C2. Measured signal frequency — replaces the Appendix B estimates

40 symbol-days (QQQ + SPY, 20 sessions), post-fix. **461 signals. Probe PASS throughout.**

| setup | signals | per sym-day | per month | **months to n=377 (2 symbols)** | Appendix B est. |
|---|---|---|---|---|---|
| **MOMO_CHASE** | 111 | 2.77 | 58.3 | **3.2** | 3.4 |
| PDH_PDL_Breakout | 63 | 1.57 | 33.1 | **5.7** | 17.1 |
| PDH_PDL_FailedBreak | 52 | 1.30 | 27.3 | **6.9** | 31.4 |
| TTM_Squeeze | 41 | 1.02 | 21.5 | **8.8** | 15.7 |
| ORB_15min | 37 | 0.93 | 19.4 | **9.7** | 9.4 |
| Crabel_Stretch | 37 | 0.93 | 19.4 | **9.7** | 9.0 |
| ORB_5min | 34 | 0.85 | 17.8 | **10.6** | 5.1 |
| IntradayMomentumBoundary | 33 | 0.82 | 17.3 | **10.9** | 9.0 |
| VWAP_Reclaim | 31 | 0.78 | 16.3 | **11.6** | 11.8 |
| Gap_Fade | 11 | 0.28 | 5.8 | 32.6 `[SLOW]` | 37.7 |
| EMA_9_20_Pullback | 10 | 0.25 | 5.2 | 35.9 `[SLOW]` | 20.9 |
| ThreeBarPlay | 1 | 0.03 | 0.5 | **359** `[DEAD]` | 26.9 |
| VWAP_2sigma_Fade | 0 | 0.00 | 0.0 | **never** `[DEAD]` | 13.5 |

**9 of 13 reach a verdict inside ~12 months** — better than Appendix B projected. The two
biggest estimate errors were in opposite directions: `PDH_PDL_Breakout` is 3x faster than
guessed, `ThreeBarPlay` 13x slower.

`MOMO_CHASE` firing at **2.77/symbol-day against his observed ~2.6/session** is independent
evidence the reconstruction behaves like him.

## C3. Reclassification — on frequency only, never performance

**`SLOW`** = `Gap_Fade`, `EMA_9_20_Pullback`. Recorded, but a verdict is 30+ months away;
labelled so an early reading is never mistaken for evidence.

**`DEAD`** = `VWAP_2sigma_Fade` (0 signals), `ThreeBarPlay` (1 signal / 40 symbol-days).

`PDH_PDL_FailedBreak` is **no longer SLOW** — measured 6.9 months against an estimated 31.4.

Both `DEAD` setups are dead because of a parameter **we invented**, not one a source gave:

| setup | binding constraint | origin |
|---|---|---|
| `VWAP_2sigma_Fade` | `RVOL >= 1.5` (funnel: 256 in-window -> 156 ADX-ok -> **12 RVOL-ok** -> 1 band touch -> 0 wick) | `[R]`, copied from `ORB_5min` |
| `ThreeBarPlay` | igniting range `>= 1.5 x ATR14` | `[R]` |

**Both could be made to fire by loosening a number we chose. Neither will be.** That is
parameter tuning, and tuning is how this project manufactured its earlier false positives.
If either is ever revisited it requires a dated amendment and a fresh `setup_id` with no
inherited history.

## C4. The family is NOT shrunk. m stays 13.

Two members provably cannot contribute evidence, and both stay in the Holm family anyway.
Removing them would lower the bar for all eleven survivors — the same move refused in
`construction_preregistration.md` when an API outage killed a pre-registered test and the
family was held at **m = 5** rather than relaxed to 4.

Cost of keeping them: ~2% on the required n (377 -> 372 if dropped). Cost of the precedent:
much higher.

## C5. Status

**Not frozen. Nothing committed.** Config hash prints via
`python -m trade_analysis.live_lab.runner --print-config`; it changes whenever a definition
changes, which is the fork mechanism.

Still blocking: **Option Data cancels 2026-09-05.** Without it the runner records signals
and the underlying twin, and every fill becomes `SKIPPED: feed`. Replay is unaffected — it
uses historical bars only and never touches the options entitlement.

Still open for the trader's decision: whether `MOMO_CHASE` resembles what he actually does.
It is the one input that cannot be derived from his fills, and amending it after data exists
forks its history.

---

# APPENDIX D — MOMO_CHASE threshold correction (2026-08-27, pre-freeze)

Prompted by the trader asking the right question: *is 0.20 sigma my decision threshold, or
merely the median?* It was neither, for two separate reasons.

## D1. Unit error — mine

The measured "+0.2029 sigma" figure was expressed in **daily-sigma** units. Section 2
specified the setup as normalising by **15-minute sigma**, which is smaller by
`sqrt(390/15) = 5.1x`. The threshold as written was therefore ~5x too loose.

Measured in the units the code actually uses, his trailing-15m move at entry is:

```
p25 0.62    p50 0.98    p75 1.49    p90 2.10        placebo median 0.46
```

His median entry sits at **2.1x** the momentum of a typical midday minute.

## D2. There is no threshold. It is a smooth preference.

Density ratio, his share of trades / same-session placebo share, 340 entries vs 2,720
random midday minutes:

| bucket | his | placebo | ratio |
|---|---|---|---|
| 0.0-0.2 | 6.5% | 23.6% | 0.27x |
| 0.2-0.4 | 5.9% | 21.1% | 0.28x |
| 0.4-0.6 | 11.8% | 16.7% | 0.71x |
| **0.6-0.8** | 13.8% | 12.1% | **1.14x** |
| 0.8-1.0 | 13.5% | 9.0% | 1.50x |
| 1.0-1.3 | 14.4% | 8.1% | 1.77x |
| 1.3-1.7 | 14.7% | 5.3% | 2.76x |
| 1.7-2.2 | 9.7% | 2.5% | 3.88x |
| 2.2-3.0 | 6.2% | 1.2% | 5.09x |
| 3.0+ | 3.5% | 0.3% | **10.67x** |

**Monotone, no cliff, no upper satiation.** He is 10x more likely than chance to trade a 3
sigma move and never stops preferring more. That is a graded preference, not a decision rule.

At the old value the rule was barely a filter: `z >= 0.20` captured 93.5% of his trades but
fired on **76.4% of random minutes** (lift 1.22x).

## D3. New value: 0.60

**0.60 is where his density ratio crosses 1.0** — the point he stops being *less* likely than
chance and starts being *more* likely. Non-arbitrary, and definable without reference to any
outcome.

```
z >= 0.60   captures 75.9% of his trades   fires on 38.6% of minutes   lift 1.96x
            ~1.97 trades/session   (his observed rate ~2.6)
```

A higher threshold would show a better lift (1.00 -> 2.77x) but discards half his trades, and
choosing it *because the number looks better* is where tuning starts.

**The distinction that licenses this change:** the threshold is fit to his **behaviour**,
which is the entire purpose of reconstructing him. It is **not** fit to his **outcomes** — no
P&L, return, or win rate entered the calculation. Behavioural fitting is the assignment;
outcome fitting is the failure mode.

## D4. Post-change verification

Replay, 6 QQQ sessions: probe **PASS**, `MOMO_CHASE` fires **2.33/symbol-day** (was 2.67 at
0.20), still close to his observed ~2.6.

**Notable:** raising the threshold 3x removed only ~13% of signals. The binding gate is the
**MACD/DMI/ADX alignment stack, not the momentum magnitude.** Worth remembering when reading
any eventual `MOMO_CHASE` result — it is mostly an indicator-alignment rule with a momentum
qualifier, not the reverse.

Change made **before freeze and before any live data exists**, so no history is forked.
