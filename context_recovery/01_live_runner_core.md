# 01 — The live runner: one trading day, from process start to the last record

Scope: the core path of the "live lab" in `trade_analysis/live_lab/` and the records it
writes under `live_lab_data/`. Launch and supervision, outage recovery, backfill and replay
belong to another report; they appear here only where the core path calls into them.

Read on 2026-10-13 from the worktree. Every code statement carries a `path:line` that was
read. Anything worked out rather than read is marked **(inferred)**. Numbers taken from data
files say which file. Nothing was run except read-only shell tools (`head`, `tail`, `grep`,
`wc`) and read-only `git log` inside the worktree.

Terms used throughout:

- **bp** — basis point, one hundredth of one percent. 5 bp on a $750 share is about 38 cents.
- **0DTE** — an option that expires the same day it is bought.
- **NBBO** — the best bid and best ask across all exchanges at one instant.
- **ATM** — "at the money": the listed strike closest to the share price.
- **RTH** — regular trading hours, 09:30 to 16:00 New York time (ET).
- **Arm** — one of the two independent paper-trading engines (options arm, shares arm).
- **Setup** — one mechanical entry-and-exit rule.
- **Holm** — a correction that raises the bar for "significant" when many rules are tested at once.

---

## 1. In one paragraph

The live lab is a paper-trading machine. It never places an order
(`trade_analysis/live_lab/__init__.py:5-7`, `:32`). Every trading day two separate programs
watch live prices from a local ThetaData terminal, apply the same 15 fixed rules to each
newly finished one-minute bar, and write down what each rule would have bought, at the real
ask, and what it would have sold for, at the real bid. One program prices the trades as
same-day options on QQQ and SPY (`runner.py`); the other prices them as $10,000 of shares in
15 tickers (`shares_runner.py`). It exists because every earlier number in the project was
computed looking backward, and a backward-looking number can be wrong without any error
showing. A record that is written before the outcome is known cannot be bent afterwards
(`trade_analysis/live_lab/README.md:11-19`).

## 2. The goal it serves, and how it changed

**The original goal (2026-08-27/28).** Settle, with new data, whether any mechanical setup
makes money on same-day options. The pre-registration states the question as: "Did the setup
correctly predict the underlying, and if so, did ATM/ATM+-1 convert that into money?"
(`research/forward_test_preregistration.md:33-36`). Those are two different failures with
opposite fixes — wrong direction cannot be rescued by any strike; right direction that still
loses points at the instrument (`forward_test_preregistration.md:38-43`). The stated
expectation, written before results, was that most or all setups fail
(`forward_test_preregistration.md:129-139`).

**The size it was designed to detect.** A mean option return of +10% per trade, with a
measured 52% per-trade spread of returns, corrected for 13 rules tested together, needs
about 377 finished trades per setup (`research/live_lab_specification.md:27-33`).

**Change 1 — shares arm added 2026-08-31, first session 2026-09-01.** The one rule in the
project with real backward evidence, `IntradayMomentumBoundary` (IMB), only worked on shares
(+$3.34 per $10,000 trade over 2,905 QQQ trades) and was not being forward-tested at all
(`shares_runner.py:3-16`; `research/shares_lab_preregistration.md:17-33`).

**Change 2 — shares universe widened to 15 tickers on 2026-09-05**, chosen on bid-ask spread
alone (`shares_runner.py:129-151`; `live_lab_data/shares/FREEZE.json:22-52`). A 60-session
minimum was added at the same moment because 15 correlated tickers produce trades much
faster than they produce information (`shares/FREEZE.json:69-75`).

**Change 3 — the trader's own "six lines" rule added 2026-09-25** as two setups
(`Six_Lines`, `Six_Lines_NoCap`), by a dated amendment. The family grew from 13 to 15
(`forward_test_preregistration.md:143-176`; `setups.py:619-647`).

**Operational goal that grew alongside:** keep the record unbroken on a laptop that travels
in a car. Most commits to the two runners since 2026-09-03 are about surviving network drops
and host sleep without writing false rows (`git log -- trade_analysis/live_lab/runner.py`:
`264068e`, `e7bf6de`, `88bae1e`, `1e7aeef`, `494c314`, `5d1ceb7`, `2d99165`).

### 2.1 What exactly is being forward-tested — the 15 setups

Both arms run **all 15** (`runner.py:379`; `shares_runner.py:540`). The options arm trades
them on QQQ and SPY as one bought call or put; the shares arm trades them on 15 tickers as
$10,000 long or short. "Cap" is the most signals per symbol per day. R means the distance
from entry to stop. All in `setups.py`.

| # | setup (bar size, cap) | what it claims | entry | exit | source | lines |
|---|---|---|---|---|---|---|
| 1 | `ORB_5min` (1m, 1) | a break of the first 5 minutes' range continues | 1-minute close beyond the range; range ≥ 0.15% of price; bar volume ≥ 1.5× the prior 10 bars; before 11:00 | stop at the far side; target 1.5R | the trader, from edgeful | `57-84` |
| 2 | `ORB_15min` (1m, 1) | same, 15-minute range | close beyond the range and on the right side of VWAP; before 10:30 | stop at mid-range; target 2R | the trader, from journalplus | `87-107` |
| 3 | `VWAP_Reclaim` (1m, 2) | a cross of VWAP with the trend continues | close crosses VWAP by 0.05% with the 20-bar average sloping the same way | stop at the prior 5 bars' extreme; target 1.5R | the trader | `110-139` |
| 4 | `MOMO_CHASE` (5m, 3) | the trader's own discretionary habit, rebuilt from his tape. "Expected to fail" | 10:30–14:30; 15-minute move ≥ 0.60 of its typical size; MACD, DMI aligned; ADX > 20 | 25-minute time exit only | reconstruction | `147-207` |
| 5 | `IntradayMomentumBoundary` (1m, 4) | a move beyond the usual noise for that time of day continues | on the hour and half hour, 10:00–15:30; close beyond the noise boundary and beyond VWAP | trailing: close back inside the boundary or VWAP | published paper, SSRN 4824172 | `215-254` |
| 6 | `TTM_Squeeze` (5m, 2) | a volatility squeeze releases in the direction of momentum | squeeze was on last bar, off now; momentum rising; before 15:00 | stop 1.5 × ATR; target 2R; exit after 9 bars | StockCharts | `257-311` |
| 7 | `PDH_PDL_Breakout` (5m, 2; 1 per direction) | a break of yesterday's high or low continues | 5-minute close beyond the level by 0.2 × ATR, body ≥ half the bar; before 15:00 | stop at that bar's low/high; target 2R | NetPicks | `314-342` |
| 8 | `PDH_PDL_FailedBreak` (5m, 2; 1 per direction) | a failed break of yesterday's high or low reverses | 09:45–15:00; a poke beyond the level in the last 12 bars, then a close back inside | stop at the session extreme; target 2R | TradeMomentum; mostly our numbers | `345-379` |
| 9 | `VWAP_2sigma_Fade` (5m, 3) **DEAD** | a rejection far from VWAP reverts to it | 10:00–11:30 and 13:30–14:30; wick through the 2-sigma band; volume ≥ 1.5×; not trending | stop a full ATR beyond; target VWAP | CrossTrade | `382-429` |
| 10 | `EMA_9_20_Pullback` (5m, 2; 1 per direction) **SLOW** | the first pullback in a trend resumes | 09:45–11:00; first touch of the 9-bar average on falling volume | stop the wider of bar low / 20-bar average; target 3R | Bulls on Wall Street | `432-493` |
| 11 | `ThreeBarPlay` (5m, 3) **DEAD** | a big bar, a rest, then continuation | big high-volume bar, 1–2 inside bars, close beyond them; 09:35–15:00 | stop at the last rest bar; target 2R | TradingSim / HowToTrade | `496-555` |
| 12 | `Crabel_Stretch` (1m, 1) | a move from the open beyond the usual early noise continues | bars stamped 09:31–11:00; close beyond open ± the 10-day "stretch" | stop at the opposite level; no target; 15:55 | Crabel | `558-582` |
| 13 | `Gap_Fade` (5m, 1) **SLOW** | an overnight gap of 0.5% or more fills | 09:45–12:00; close back through the open by 0.05% | stop at the session extreme; target yesterday's close | SharePlanner / QuantifiedStrategies | `585-616` |
| 14 | `Six_Lines` (10m bars judged each minute, 1) | the trader's own rule: the first break of one of his six lines continues | first 10-minute close beyond a line the open had not already passed | +20 bp target; a later 10-minute close back inside the line; 15:55 | the trader, 2026-09-20 | `684-735` |
| 15 | `Six_Lines_NoCap` (same, 1) | the same without the profit cap | same | same, no target | amendment 2026-09-25 | `738-744` |

The six lines are yesterday's premarket high and low, yesterday's market-hours high and low,
and today's premarket high and low (`six_lines.py:83-106`).

SLOW and DEAD are set from how often a rule fires, never from profit
(`setups.py:759-787`). A param name ending `[R]` marks a number the researchers chose
rather than a source (`setups.py:6-9`). The README warns these are "roughly three
independent bets, not thirteen" (`README.md:127-130`).

**Checkpoints.**

| arm | checkpoint | what may be said | where |
|---|---|---|---|
| options | 50 trades total | pipeline health only; no effect claims | `forward_test_preregistration.md:52-57` |
| options | 100 trades total | same | same |
| options | 200 per setup | first serious look; the four-part promotion bar goes live | same |
| options | 377 per setup | enough to detect +10% per trade | same |
| shares | 200 per setup | first look, stated in advance to be underpowered | `shares_lab_preregistration.md:95-100` |
| shares | 60 sessions | "the checkpoint that actually governs" | `shares/FREEZE.json:56`, `:60` |
| shares | 666 IMB trades | first n at which the backtest effect could show | `shares_lab_preregistration.md:100` |

A setup is "promising" only if all four hold: at least 200 finished trades, clears Holm
within the family, the effect is larger than the smallest effect the sample could detect,
and the first and second halves agree in sign (`forward_test_preregistration.md:65-70`;
`dashboard.py:390-400`).

---

## 3. How it works, end to end

### 3.1 Launch (other agent's slice — one paragraph for orientation)

Windows Task Scheduler runs `run_lab.cmd`, which changes into the **main checkout**
`C:\Users\chaitanyakharche\Documents\stock` and runs the pyenv Python 3.12.10 on
`trade_analysis.live_lab.autostart --symbols QQQ SPY --start-terminal`
(`run_lab.cmd:6-7`). `autostart` sleeps until 09:05 ET (`autostart.py:64`), runs a structural
preflight, then starts each arm as a child process (`autostart.py:855`). The ledger for
2026-10-12 shows both arms opened at 09:07:54 ET (`live_lab_data/session_ledger.jsonl`, last
two rows).

### 3.2 Process start

Options arm, `runner.main` (`runner.py:625-655`):

1. Take an operating-system file lock named `runner.lock` in the lab directory. If another
   runner holds it, exit with code 4 so the supervisor does not retry
   (`runner.py:644-648`; `lock.py:44-74`). The lock is released by the kernel if the process
   dies, so a stale lock cannot block the next day (`lock.py:7-10`).
2. Build the `LiveLab` object (`runner.py:129-152`): open the record store, compute the
   configuration hash and write `config/<hash>.json` if it is not already there
   (`runner.py:134-135`; `store.py:224-231`), open the feed, create the catch-up helper and
   the six-lines loader, and install a Ctrl-C handler that only sets a stop flag
   (`runner.py:152-156`).

Shares arm, `shares_runner.main` (`shares_runner.py:675-694`): the same, with lock name
`shares_runner.lock`, store root `live_lab_data/shares/` (`:678`) and default symbols
`BROAD_UNIVERSE` (`:677`, list at `:147-151`).

### 3.3 `run()` before the open

Both arms, same order (`runner.py:256-275`; `shares_runner.py:286-303`):

1. **Roll yesterday's signal rows** into `signals_archive/<date>.jsonl.gz`, byte for byte,
   so the active `signals.jsonl` holds today only (`store.py:167-220`). Written: event
   `signals_rolled`.
2. Write event `start` with the config hash.
3. **Warmup** (`runner.py:163-219`; `shares_runner.py:205-258`). For each symbol, ask for
   one-minute bars of the previous 25 weekdays (`runner.py:222-228`) and keep the sessions
   with at least 300 bars (`session.py:273-276`). From those, `build_warmup` computes
   (`session.py:267-344`):
   - average cumulative volume by minute, last 20 sessions (`:279-285`);
   - the IMB "noise" size: mean absolute move from the open to each minute, last 14 sessions (`:288-295`);
   - the Crabel "stretch": 10-day average of the smaller of (open − low, high − open) (`:298-305`);
   - 20-day average of the 09:30–10:30 range (`:308-313`);
   - yesterday's RTH open, high, low, close (`:315-325`);
   - the last 2 full sessions of bars, used only to seed moving averages (`:327-333`; `PREFIX_SESSIONS = 2` at `:27`).
   Then the six-lines loader fetches yesterday's bars from 04:00 to 16:00
   (`levels_live.py:63-76`). Written: event `warmup` per symbol.
   If the terminal answers but its upstream does not, warmup waits up to 180 s and retries
   once; otherwise it writes event `aborted` and the arm exits (`runner.py:174-205`, `:265-268`;
   `feed.py:50`, `:215-250`).
4. **Recover** (`runner.py:230-252`; `shares_runner.py:260-282`). Per-day caps and the set
   of decisions already taken are rebuilt from today's DECISION rows on disk
   (`store.py:365-417`). Open positions are reloaded from `positions_open.json` only if the
   file is stamped with today's date; otherwise the file is copied to `recovery_archive/`
   and dropped (`store.py:324-363`).
5. If a checkpoint from an earlier process today exists, note its last-managed minute so the
   missed minutes are replayed (`runner.py:272-274`; `catchup.py:57-71`).
6. Sleep in 15-second steps until 09:30 (`runner.py:281-283`).

On 2026-10-12 the options arm finished warmup at 09:08:41 ET (`live_lab_data/events.jsonl`,
seq 9811–9813).

### 3.4 The loop, once per poll

Poll interval is 5 s while any position is open and 15 s otherwise (`runner.py:130-131`,
`:308`; `shares_runner.py:172`, `:339`).

Each pass (`runner.py:277-308`):

- stop if the date changed or it is 16:00 or later (`:279-280`, `:296-297`);
- if more than 90 s have passed since the last good tick, **freeze** and replay the missed
  minutes from history before doing anything live (`:288-295`; `catchup.py:50`, `:92-144`);
- run `_tick`; on success note the time as "good" (`:299-300`);
- a feed failure writes an outage row of kind `tick` and the loop continues (`:301-305`);
- save open positions to `positions_open.json` by atomic replace (`:306-307`;
  `store.py:291-322`).

### 3.5 What a tick does — options arm (`runner.py:317-368`)

For each symbol, in order:

1. Six-lines loader: at the first tick at or after 09:31, fetch today's premarket bars and
   build the six lines (`runner.py:321`; `levels_live.py:78-102`). Written: event `levels`.
2. Pull today's one-minute bars and **admit only finished ones** (`runner.py:324`). A bar
   stamped T covers T to T+60 s and is admitted no earlier than T + 60 s + 1.5 s
   (`session.py:179-181`, `:196-197`; settle default `runner.py:130`). The bar still forming
   never reaches a setup.
3. If the admitted sequence skipped a minute, write an outage row `degraded_bar`
   (`runner.py:330-336`; detection `session.py:200-203`).
4. Fetch the live share quote. If it is more than 5 s old when received, or stamped more
   than 5 s in the future, discard it and write `stale_quote` or `future_quote`
   (`runner.py:337-350`; thresholds `:51-52`).
5. **Manage open positions first** (`runner.py:352`, body `:514-563`).
6. If new bars were admitted, evaluate every one-minute setup once per new bar
   (`:357-358`), then every five-minute setup once per newly completed five-minute bucket
   (`:363-368`). A five-minute bucket exists only when all five of its minutes are present
   (`session.py:46-49`).

### 3.6 The opening range and the two timestamp conventions

- One-minute bars carry the vendor's **open** stamp: the bar stamped 09:31 covers 09:31:00
  to 09:31:59 (`runner.py:64-66`; `dashboard.py:139-143`).
- Five-minute buckets are stamped at their **close**: minutes 09:30–09:34 become the bar
  stamped 09:35 (`session.py:30-44`).
- The opening range is the high and low of the first 5 and the first 15 one-minute bars,
  and it exists only once all of them are present (`session.py:227-240`).

Consequence **(inferred from the code, not stated anywhere)**: time windows on one-minute
setups are tested against open stamps, so each window is one minute later than it reads.
`ORB_5min` refuses any bar stamped at or before 09:35 (`setups.py:66`), so the first bar it
can act on is the one stamped 09:36, whose close is known at 09:37. `Crabel_Stretch`'s
window starts at the bar stamped 09:31 (`setups.py:568`); on 2026-10-12 its decision for
that bar was written at 09:32:14 (`live_lab_data/signals.jsonl`, seq 9816). The comment in
`session.py:67` calls `bar_ts` the "CLOSE timestamp", which is only true for five-minute bars.

### 3.7 When a signal fires — the order of checks (options arm)

`_evaluate` (`runner.py:372-418`) builds an immutable `Context` for the bar
(`session.py:242-261`) and calls each setup's `evaluate`. If a setup returns a signal:

| check | where | what is written if it fails |
|---|---|---|
| runner is frozen catching up | `runner.py:373-377` | nothing per signal; a count in event `caught_up` |
| bar older than 180 s | `:390-395` | SKIP row, reason `stale_bar` |
| setup's per-day cap reached | `:397-401` | SKIP `max_per_day` |
| per-direction cap reached | `:402-408` | SKIP `max_per_direction` |
| that setup already has a position in this symbol | `:409-412` | SKIP `already_open` |
| clock at or past 15:55 | `:413-416` | SKIP `past_eod_flat` |
| a DECISION for this setup, symbol and bar already exists | `:422-426` | SKIP `already_decided` |

### 3.8 How an entry is priced — and the guard against looking ahead

This is the centre of the design (`runner.py:420-499`; `store.py:245-267`):

1. **Write the DECISION row and force it to disk** (`fsync`) — setup, symbol, direction,
   bar stamp, and the setup's own state (`runner.py:428-437`).
2. **Only after that** ask for any option price (`runner.py:438` comment, `:443-448`).
3. If there is no usable share quote: FILL row `SKIPPED`, `no_underlying_quote` (`:440-442`).
4. If no option expiring today is listed: `SKIPPED`, `no_0dte` (`:443-446`).
5. Pull the whole option chain in one call (cached at most 2 s, `:501-510`) and pick the
   contract: call for a long signal, put for a short one; the strike nearest the live share
   midpoint, ties to the lower strike; plus the strike one step below ("ATM-1") and one step
   above ("ATM+1") (`options.py:89-123`). A contract is unusable if its bid is zero or its
   spread exceeds 25% of its midpoint (`options.py:90`, `:130-131`); an unusable ATM gives
   `SKIPPED`, `unusable_quote` (`runner.py:451-454`). An unusable neighbour is simply absent.
6. If the ATM quote's own timestamp is more than 120 s from the wall clock in either
   direction: `SKIPPED`, `stale_option_quote` (`runner.py:461-470`; threshold imported from
   `preflight.py:39`).
7. Work out implied volatility and delta from the midpoint — the vendor tier has no Greeks
   (`options.py:134-145`; `feed.py:13-15`) — fetch open interest, and open up to three
   positions that share one `signal_id` (`runner.py:472-484`; `positions.py:163-202`).
8. **Entry price is the ask** (`positions.py:188`). Write FILL row `OPEN` with the three
   strikes' bid, ask, spread, implied volatility and delta (`runner.py:490-497`).

Which quote, which minute: the live quote at the moment the bar is admitted. Real example,
2026-10-12: bar stamped 09:31 (closes 09:32:00), DECISION at 09:32:14.53, FILL at
09:32:14.90 (`signals.jsonl`, seq 9816–9817).

Other structural guards against lookahead:

- setups get tuples, not the live buffer (`session.py:249-252`);
- VWAP (volume-weighted average price) is always anchored at 09:30 today (`session.py:243`;
  `indicators.py:149-161`);
- the regression used by `TTM_Squeeze` is the endpoint form, not centred (`indicators.py:205-224`);
- no setup reads the live quote: `grep "ctx.quote" setups.py` finds nothing, and the shares
  arm passes `quote=None` into the context outright (`shares_runner.py:448`, `:452`);
- `Six_Lines` cuts the buffer at its own bar before doing anything (`setups.py:653-664`).

**One real gap in that guarantee.** `SessionState.context` hands a setup the whole admitted
buffer (`session.py:249`). When two or more bars are admitted in one tick, the evaluation
"for" the older bar sees the newer bar as its last bar. The code says so itself:
"on a batch admission ctx.bars_1m can hold bars AFTER the one being evaluated"
(`setups.py:656-658`), and `six_lines_setup_test.py:12-13` lists it as a danger. Only the two
Six_Lines setups defend against it. It is not future information relative to the fill —
the fill is at the current quote — but the DECISION row carries the older bar's stamp. The
exposure is bounded: a gap over 90 s now triggers a minute-by-minute replay
(`catchup.py:50`), so a live batch is at most two bars **(inferred)**.

### 3.9 How exits work — options arm

Every poll, for each open position (`runner.py:514-563`):

1. Skip everything if there is no usable share quote this poll (`:516`).
2. Look the contract up in the chain and take its **bid**; update best and worst excursion
   (`:531-534`; `positions.py:80-93`).
3. Count one-minute bars after the entry bar (`:535-536`).
4. Test, in this order:
   - **stop, then target** — against the high and low of the *latest admitted one-minute
     bar* (`runner.py:521`, `:539`; `positions.py:97-114`). Levels are share prices, never
     option prices.
   - **15:55 flatten** (`positions.py:117-118`), reason `eod`.
   - **time exit** — wall-clock minutes since entry (`positions.py:119-122`), reason `time`.
   - **bar exit** — bars held ≥ N (`positions.py:123-124`), reason `bars`.
   - **trailing** — the setup's own `manage` (`runner.py:544-551`); the reason is whatever
     it returns: `trail_imb` (`setups.py:251`, `:253`) or `failed` (`setups.py:734`).
5. If an exit fired but the contract has no bid in the chain, write outage `exit_no_quote`
   and keep the position (`runner.py:554-557`).
6. Otherwise close **at the bid** and append one trade row, fsynced (`:558-560`;
   `store.py:285-287`).

The trigger and the fill are deliberately different things: a stop is *detected* on a
finished bar's low and *filled* at the next observed option bid (`setups.py:11-14`;
`positions.py:100-102`). The first row of `trades.jsonl` shows what that means: a `stop`
exit where the share price at exit (772.64) was above the stop (772.17) and the option made
$1.92.

End of day: at the first poll at or after 15:55 every position exits with reason `eod`
(real example: `exit_ts` 15:55:03 on 2026-10-12, last rows of `trades.jsonl`). When the loop
ends at 16:00, anything still open is closed by `_flatten_all` (`runner.py:310`, `:565-593`).
After 16:00 it uses the last in-session bid and appends `_after_close_mark` to the reason,
because a post-close quote is not a price anyone could have traded (`:566-580`). Then the
daily summary is written (`:597-619`).

### 3.10 The shares arm tick (`shares_runner.py:370-455`)

Same bars, same setups, different mechanics:

1. Fetch bars for all 15 symbols at once, up to 8 in parallel (`:382`; `feed.py:414-462`).
   Eight workers is a measured optimum; more is slower (`feed.py:433-441`).
2. Admit finished bars per symbol (`:396-411`).
3. Fetch quotes only for symbols that admitted a bar (`:415`).
4. Then, strictly one symbol at a time (`:419`): discard a stale or future quote
   (`:428-441`), and **if there is no good quote, skip the symbol for this tick** (`:442-443`).
5. For each admitted bar: manage open positions, then look for entries (`:447-455`).

Managing (`:463-496`): only bars *after* the entry bar count (`:465-466`). Order: stop,
target (bar high/low), time (count of one-minute bars), bars (N × 5 for a five-minute
setup, `:481-482`), trailing (reason always `trail`, `:484-492`), then `eod` when a bar
stamped 15:55 or later is admitted (`:493-494`) — which happens at about 15:56 (real
`exit_ts` 15:56:03 on 2026-10-12).

Entering (`:525-609`): refuse while frozen (`:526-529`); refuse if the bar is over 180 s old
and write an **outage** row `stale_bar` (`:530-535`); per setup, check the per-day cap
*before* evaluating (`:543-547`); write DECISION, fsynced (`:573-576`); fill a long at the
ask and a short at the bid (`:578`); shares = 10,000 ÷ price (`:585`); write FILL `FILLED`
with the exit parameters (`:596-601`).

Closing (`:498-521`): a long sells the bid, a short buys the ask (`:499`). No commission is
subtracted (`:500-501`).

Half days: if no new bar has arrived for 12 minutes after 12:30 *and* the feed answered
within the last 120 s, the arm treats the market as closed and flattens
(`:348-368`; constants `:98-102`). The second condition exists because the first version
flattened on a dead WiFi link (`:351-359`).

End of day flatten never leaves a position unrecorded: live quote, else last known quote
flagged `_stale_mark`, else after 16:00 the last in-session quote flagged
`_after_close_mark` (`:611-655`).

### 3.11 What is written, step by step

| moment | file | row |
|---|---|---|
| runner start | `events.jsonl` | `signals_rolled`, `start` |
| warmup | `events.jsonl` | `warmup` per symbol |
| 09:31 | `events.jsonl` | `levels` per symbol, with all six prices |
| signal fires | `signals.jsonl` | `DECISION` (fsync) |
| price obtained | `signals.jsonl` | `FILL` (`OPEN`/`FILLED`/`SKIPPED`) (fsync) |
| signal refused | `signals.jsonl` | `SKIP` with reason (fsync) |
| every poll | `positions_open.json` | whole open book, atomic replace (not in git: `.gitignore:99-100`) |
| exit | `trades.jsonl` | one row per position (fsync) |
| anything wrong | `outages.jsonl` | kind + detail (fsync) |
| after a gap | `events.jsonl` | `caught_up` |
| 16:00 | `daily/<date>.json` | session summary |
| supervisor | `session_ledger.jsonl` | `OPENED`, then `COLLECTED`/`PARTIAL`/`ABORTED` |

All timestamps are New York wall-clock time with no timezone marker (`clock.py:27-29`;
`store.py:30-33`). One sequence counter is shared by events, signals, trades and outages
within an arm (`store.py:131-134`), which is why `events.jsonl` has 138 lines but sequence
numbers near 9,800.

### 3.12 Where the data comes from, and what happens when it is late, stale or missing

**Source.** One vendor, ThetaData. The lab talks only to the vendor's "Theta Terminal", a
Java program running on the same laptop, over plain HTTP at `http://127.0.0.1:25503/v3`
(`feed.py:35`). `feed.py` is deliberately standalone so a change to the research data client
cannot alter live behaviour (`feed.py:3-4`).

| need | endpoint | note |
|---|---|---|
| live share quote | `/stock/snapshot/quote` | served by the terminal itself (`feed.py:267-294`) |
| one-minute bars, today and prior days | `/stock/history/ohlc` | goes through the vendor's upstream (`feed.py:359-379`) |
| premarket bars | same, 04:00–20:00 window | never cached (`feed.py:381-410`) |
| whole option chain for one expiry | `/option/snapshot/quote` | one call, about 333 rows (`feed.py:487-515`) |
| listed expiries | `/option/list/expirations` | (`feed.py:473-485`) |
| open interest | `/option/snapshot/open_interest` | (`feed.py:517-528`) |

**Entitlement.** A stock data plan is needed for everything; an option data plan for every
contract price (`README.md:45-50`). The plan in use has no Greeks endpoint, which is why
implied volatility and delta are computed locally and suffixed `_derived` (`feed.py:13-15`).
Which option plan is active today is not recorded in the code; see §9 and §10.

**What the runner does when something is wrong.**

| situation | response | where |
|---|---|---|
| bar still forming | not admitted | `session.py:196-197` |
| vendor publishes a bar late | the cache refuses to answer until it has the bar, so every poll re-asks | `feed.py:339-342` |
| a minute is missing | outage `degraded_bar`; the 5-minute bar that needed it never exists; 1-minute setups still run | `runner.py:330-336`; `session.py:48-49` |
| a bar arrives out of order | dropped permanently | `session.py:198-199` |
| share quote older than 5 s, or 5 s in the future | discarded, outage row. Options: no exit management this poll, and a signal becomes DECISION + `SKIPPED no_underlying_quote`. Shares: the symbol is skipped for this tick | `runner.py:337-350`, `:516`, `:440-442`; `shares_runner.py:428-443` |
| signal bar older than 180 s | entry refused | `runner.py:390-395`; `shares_runner.py:530-535` |
| ATM option quote more than 120 s from the clock | `SKIPPED stale_option_quote` (entries only) | `runner.py:461-470` |
| ATM spread over 25% of midpoint, or zero bid | `SKIPPED unusable_quote` | `options.py:109-112` |
| no same-day expiry listed | `SKIPPED no_0dte` | `runner.py:443-446` |
| a call fails | 4 attempts with 1, 2 and 4 s pauses, then `FeedOutage`; outage row `tick`; loop continues | `feed.py:150-182`; `runner.py:301-302` |
| vendor answers "no data" (HTTP 472) | empty result, not an error | `feed.py:159-160` |
| more than 90 s without a good tick | freeze; replay the missed minutes from history; exits fire at their real minute; entries stay frozen | `catchup.py:16-30`, `:50` |
| upstream down during warmup | wait up to 180 s, retry once, else abort the session | `runner.py:174-198` |
| a position cannot be priced at the close | last known mark, flagged in the reason | `runner.py:571-589`; `shares_runner.py:627-649` |

The rule behind all of it: the feed "never fabricates a value; raises or returns None"
(`feed.py:129`).

### 3.13 How the two arms differ

| | options arm | shares arm |
|---|---|---|
| file, store | `runner.py`, `live_lab_data/` | `shares_runner.py`, `live_lab_data/shares/` |
| symbols | QQQ, SPY | 15 tickers (`shares_runner.py:147-151`) |
| instrument | one bought call or put per strike; three strikes per signal | $10,000 of shares, long or short |
| cost in the record | bid-ask spread plus $0.0808 fee per round trip | bid-ask spread only |
| exit check | every poll (5 s), against the latest finished bar | once per admitted bar, bars after entry only |
| time exit | wall-clock minutes (`positions.py:119-122`) | count of 1-minute bars (`shares_runner.py:478-480`) |
| bar exit | N one-minute bars (`positions.py:123`) | N bars of the setup's own size (`shares_runner.py:481-482`) |
| trailing exit name | `trail_imb` or `failed` | `trail` |
| end-of-day exit | first poll at or after 15:55 | when the bar stamped 15:55 is admitted, about 15:56 |
| half day | no handling | flattens when the tape stops (`shares_runner.py:348-368`) |
| underlying twin, excursions | recorded | not recorded |
| stale-bar refusal recorded as | SKIP row per signal | outage row per bar |
| `max_per_day` SKIP means | a real signal was capped | the cap was already full on this bar |
| fill status word | `OPEN` | `FILLED` |

**Why the shares arm was added.** In its own words: "the strategy with the best evidence in
the entire project was not being forward tested at all" (`shares_runner.py:15-16`). An ATM
same-day option needs roughly 5 bp of share movement just to pay for its spread and time
decay; QQQ shares cost about 0.5 bp round trip (`shares_runner.py:4-13`). The stated limit,
written before any data: this arm cannot confirm or refute IMB inside a year
(`shares_lab_preregistration.md:117-118`).

### 3.14 The dashboard — what it shows and to whom

It is a text report printed in a terminal for the owner. There is no web page and it writes
nothing (`dashboard.py:1`, `:403-495`).

    python -m trade_analysis.live_lab.dashboard                # options, ATM strike
    python -m trade_analysis.live_lab.dashboard --arm ATM+1
    python -m trade_analysis.live_lab.dashboard --today

For the shares arm it needs both `--arm SHARES` and `--lab-dir live_lab_data/shares`
**(inferred** from the defaults at `dashboard.py:405`, `:409-410` and the filter at `:312`).

Sections, in order:

1. **Forward test checkpoints** — freeze date, total trades, a tick box for 50, 100, 200 per
   setup and 377 per setup, the 60-session line when the freeze declares one, and the clock
   rule (`dashboard.py:219-255`).
2. **Signal reconciliation** — per session, decisions × strikes against trade rows
   recorded. Built after three sessions silently lost every position still open at the
   close (`dashboard.py:181-216`).
3. **Fill latency** — how late each fill was beyond its setup's normal lag, and the net
   with and without fills more than 3 minutes late (`dashboard.py:120-178`).
4. **Session coverage** — weekdays since the freeze: recorded, holiday, unaccounted
   (`dashboard.py:53-117`).
5. **The setup table** — signals, n, win rate, mean return, net dollars, worst drawdown,
   Holm-adjusted p-value and a verdict that reads `INSUFFICIENT (n=k/200)` until the bar is
   met; `[DEAD]` and `[SLOW]` tags (`dashboard.py:451-471`).
6. **Underlying twin** — option mean beside share mean, with a one-line reading such as
   "direction OK, option structure eats it" (`dashboard.py:473-488`).

How the numbers are made: only trades with an accepted hash and the chosen arm
(`dashboard.py:309-313`); intervals from a bootstrap that resamples whole days, 4,000
repetitions (`dashboard.py:261-289`); Holm across the full family of 15 (`:292-298`, `:375`).

---

## 4. Map

In scope and read in full:

| file | what it is for |
|---|---|
| `live_lab/README.md` | Purpose, the ordering guarantee, file list, ledger. Partly stale (see §9). |
| `live_lab/__init__.py` | Package statement: never places orders; `PLACES_ORDERS = False`. |
| `live_lab/run_lab.cmd` | Task Scheduler launcher: main checkout, pyenv Python, `autostart`. |
| `live_lab/runner.py` | Options arm engine. Config hash, loop, entry ordering, exits, daily summary. |
| `live_lab/shares_runner.py` | Shares arm engine. Universe, $10,000 clips, early-close detection. |
| `live_lab/session.py` | Bar admission (finished bars only), 5-minute buckets, `Context`, `Signal`, warmup baselines. |
| `live_lab/feed.py` | The only network client: local Theta Terminal over HTTP. Retries, outage classes, bar cache, parallel fetch. |
| `live_lab/clock.py` | The single source of "now", in New York time. |
| `live_lab/lock.py` | One-runner-per-arm OS file lock. |
| `live_lab/preflight.py` | Manual/supervisor health check: exposure, freshness, option chain, warmup bars. Owns the 120 s "delayed" threshold. |
| `live_lab/setups.py` | The 15 rules, the SLOW and DEAD labels, the registry whose order is hashed. |
| `live_lab/indicators.py` | Pure maths: EMA, MACD, ATR, ADX, VWAP, relative volume, regression endpoint. |
| `live_lab/six_lines.py` | The trader's six lines (definition, tally, and the research backtest `first_break_trade`). |
| `live_lab/levels_live.py` | Builds the six lines live: yesterday at warmup, today's premarket at 09:31. |
| `live_lab/breakout_levels.py` | The earlier 8-line level set. Research only; not imported by either runner. |
| `live_lab/orb_veto.py` | Day-level "is today tradeable" veto. Research only; the runners use just its `resample_10m`. |
| `live_lab/options.py` | Strike selection, usability test, Black-Scholes implied volatility and delta. |
| `live_lab/positions.py` | Option position record, excursions, exit tests, P&L arithmetic, fee. |
| `live_lab/store.py` | Append-only JSONL writer, fsync on decisions, signal archive roll, checkpoint, cap recovery. |
| `live_lab/ledger.py` | One line per session state change; coverage denominator; `reconcile`. |
| `live_lab/dashboard.py` | Read-only terminal report: checkpoints, leak check, latency, coverage, per-setup verdicts. |
| `live_lab/restart_terminal.py` | Manual tool: report or restart a wedged Theta Terminal without touching the arms. |
| `live_lab/catchup.py` | (Other slice; read because the loop calls it.) Freeze after 90 s away, replay missed minutes from history, entries stay frozen. |
| `research/live_lab_specification.md` | Design spec, 13 setups with sources, power calculation, build report. Status line still says "DRAFT". |
| `research/forward_test_preregistration.md` | The options freeze: what is fixed, checkpoints, clock rule, 2026-09-25 amendment. |
| `research/shares_lab_preregistration.md` | The shares freeze and its honest power statement. |
| `research/orb_veto_preregistration.md` | The veto experiment; §8 forbids applying it to the live lab. |
| `research/incident_2026-09-24_opening_window.md` | Why the runners now start before the open. |
| Tests | `six_lines_test.py`, `six_lines_setup_test.py`, `ledger_test.py`, `opening_window_test.py`, `options_runner_guards_test.py`, `orb_veto_test.py`, `foreign_options_gate_test.py`. `levels_test.py` is not a test: it is an exploratory study that also owns `get_ext` and `BUFFER`. |

Records sampled: both `FREEZE.json`, all nine `config/*.json`, `daily/2026-10-12.json` in
each arm, heads and tails of `trades`, `signals`, `events`, `session_ledger`, `outages`,
`trade_corrections`, `stranded_recovery`, `six_lines_QQQ.json`.

In the same directory but outside this slice (not read): `autostart.py`, `gap_recovery.py`,
`stranded.py`, `backfill*.py`, `replay*.py`, `archive.py`, `attribution.py`,
`checkpoint_diagnostic.py`, `reconstruct_eod.py`, `walkforward.py`, `sharewf.py`,
`day_charts.py`, `after_break.py`, `early_r1_hold.py`, `breakout_sweep.py`,
`orb_veto_backtest.py`.

---

## 5. Design decisions and the reason for each

| decision | reason, with evidence |
|---|---|
| Never place an order; no broker code at all | "A design boundary, not a phase-1 limitation" (`__init__.py:5-7`). The older `live_trading` package evaluated signals on the forming bar and is deliberately not imported (`README.md:132-137`). |
| Decision on disk before any option price is requested | So the record cannot be shaped by the price. "If the process dies in between, the signal still exists with no knowledge of any price" (`README.md:62-64`; `store.py:247-253`). |
| Only finished bars, plus 1.5 s settle | The project lost an apparent +16.46% option edge to a one-minute lookahead; with honest timing it was +0.70% (`levels_test.py:9-10`; memory note `live-lab-forward-test.md`). |
| Hash covers definitions only, not the git commit | The commit was first inside the hash, so any commit would have reset every setup's history — "precisely the 'reset when it looks bad' failure mode this experiment forbids" (`runner.py:115-118`). |
| Operational guards sit outside the hash | The hash says what a trade *means*; staleness thresholds say whether the feed is fit to act on (`runner.py:69-72`; `shares_runner.py:89-91`). Tested: `options_runner_guards_test.py:137`. |
| Dead and slow setups stay in the family | Removing a rule that cannot fire lowers the bar for the others. Labels are set from signal frequency, never from profit (`setups.py:759-787`). |
| No parameter is ever loosened to make a rule fire | "That is tuning, and tuning is how this project manufactured its earlier false positives" (`setups.py:780-781`). |
| Indicators seeded from 2 prior sessions; VWAP and opening range stay today-only | Seeding from today only made `EMA_9_20_Pullback` impossible and stopped `MOMO_CHASE` firing before 12:25 (`session.py:105-114`; spec Appendix C1.2, `live_lab_specification.md:525-548`). |
| Three strikes recorded, ATM is the only one that counts | Stops a later "actually ATM+1 was best all along" (`options.py:9-11`). |
| Entry at ask, exit at bid, real broker fee both sides | The fee, $0.0404 per contract per side, was measured from the trader's own tape (`positions.py:24`). |
| An "underlying twin" on every option trade | To tell "no directional edge" from "direction right, option structure eats it" (`positions.py:10-16`). |
| All 15 setups run on shares, not just IMB | IMB won a 13-way comparison; forward-testing only the winner would be selection on the outcome (`shares_runner.py:24-27`). |
| Shares universe chosen on spread alone, 11 of 15 are ETFs | The edge is 3.34 bp gross; 16 of 51 candidates had a median spread wider than that. ETFs have no earnings gaps (`shares_runner.py:129-142`). |
| Per-symbol caps, no shared capital | So a QQQ trade is identical whether the universe holds 2 names or 15, which is what allows pooling across the widening (`shares/FREEZE.json:12-19`). |
| Quote age measured at receipt, two-sided | A reference sampled before a slow batch made every quote look future-dated; preflight once printed "REAL-TIME (age -50s)" 15 times (`feed.py:281-288`). A one-sided test let future-dated quotes through (`runner.py:44-50`). |
| Stale-bar guard, 180 s | On 2026-09-09 a late start produced 11 chase entries for −$352.22 (`shares_runner.py:74-87`). Back-ported to options on 2026-09-24 after four late Crabel fills (`runner.py:53-62`). |
| Runners start before the open | From 2026-08-31 to 09-24 they started at 09:33–09:35 and missed or chased the opening signals (`incident_2026-09-24_opening_window.md:1-50`). |
| Caps rebuilt from DECISION rows, not from open positions | A 15:00 restart on 2026-09-02 re-entered two cap-1 setups; counting positions would triple-count the options arm (`store.py:371-382`). |
| One decision per (setup, symbol, bar), ever | Idempotency, as a broker gives with a unique order id (`store.py:403-408`). |
| Recovery file must carry today's date | Otherwise yesterday's expired contracts would be marked against today's quotes (`store.py:327-336`). |
| `managed_through` stored separately from `saved_at` | A frozen runner keeps saving while managing nothing; on 2026-10-02 that hid an hour of unchecked stops (`store.py:298-302`). |
| Never flatten at an after-hours quote | On 2026-09-25 the host slept through 15:55 and two XLP longs were closed 6.2% below entry at 16:19 (`shares_runner.py:622-626`). |
| A position is never left unrecorded | "A stale mark is a compromise; an unrecorded position is a hole" (`shares_runner.py:619-620`). |
| Signals archived one gzip per session | The shares arm writes 3–4.5 MB of signal rows a session; one file would pass the git host's 100 MB limit (`store.py:43-48`). |
| Bar cache serves only when provably identical to refetching | Removes about 12× wasted calls without changing any fill; stops answering within 60 s of an outage (`feed.py:321-343`). |
| Outages logged as episodes, not per failed call | On 2026-09-01 one fact was logged 202 times and buried the one different error (`feed.py:198-206`). |
| "Upstream unreachable" and "terminal unreachable" are different errors | One wants patience, the other a relaunch; treating them alike cost whole sessions (`feed.py:57-86`). |
| Six lines built by the research function itself | So live levels cannot drift from the backtest's definition (`levels_live.py:21-22`). Checked on 400 synthetic days (`six_lines_setup_test.py:5-8`). |
| Six lines: first break only, no second chance | "The first break", not "a break" — the distortion that let Crabel fire on the first bar a blind runner could see (`setups.py:633-636`). |
| Six-line premarket fetched at 09:31, not 09:30:01 | The 09:29 bar is published a moment after 09:30:00; a high set in the last minute could be missed (`levels_live.py:16-19`). |
| The veto is not applied live | Prohibited by its own pre-registration (`orb_veto_preregistration.md:172-174`). |
| Ledger holds coverage only, no P&L | "Mixing those in would make the denominator depend on the results" (`ledger.py:42-46`). |
| Dashboard ranks by expectancy with interval and n | "A big number from four trades is not evidence" (`dashboard.py:3-4`). |

---

## 6. Records, formats, invariants, frozen things

### 6.1 What "FROZEN" means concretely

**What is hashed.** SHA-256, first 16 hex characters, of a JSON blob.

- Options (`runner.py:104-121`): symbols, `settle_ms`, fee, primary arm, the three arm
  names, family size, the SLOW and DEAD lists, and each setup's `describe()`.
- Shares (`shares_runner.py:154-168`): arm name, symbols, notional, a fill-rule sentence,
  family size, SLOW and DEAD lists, each `describe()`, and the 15:55 flatten time.
- `describe()` is id, version, timeframe, the two caps and the hand-written `params`
  dictionary (`setups.py:46-49`). The list order matters (`setups.py:749-751`).

**What is not hashed — and this is the trap.** The hash fingerprints a *declaration*, not the
code. The body of every `evaluate` and `manage`, the indicator maths, bar admission, strike
selection (including the 25% spread limit, `options.py:90`), the exit tests in
`positions.py`, and the fee actually charged (`positions.py:24` — the hashed `0.0404` is a
separate literal at `runner.py:107`) can all change without the hash moving
**(read from the code; no document states it)**.

**Where the hash is checked.**

- `freeze_config` writes `config/<hash>.json` if absent (`store.py:224-231`). It never
  compares against anything.
- `FREEZE.json` lists the accepted hashes per arm. The dashboard keeps only trades whose
  hash is in that list (`dashboard.py:309-313`); so do `backfill.py:418-419`,
  `replay_audit.py:187-188` and `checkpoint_diagnostic.py:106`.
- A test pins the two current hashes and checks that the 13 original definitions are
  byte-identical to the frozen file (`six_lines_setup_test.py:261-281`).

**What happens on a mismatch.** Nothing stops. The runner silently writes a new config file
and stamps every trade with the new hash; the dashboard then silently drops those trades
because the hash is not accepted. Neither runner reads `FREEZE.json`
(`grep FREEZE runner.py shares_runner.py store.py` finds nothing). The test above is the only
alarm, and only if someone runs it.

**Accepted hashes today** (`live_lab_data/FREEZE.json:6-10`; `shares/FREEZE.json:7-11`), with
row counts from `grep` on each `trades.jsonl`:

| arm | hash | meaning | trade rows |
|---|---|---|---|
| options | `1d7c5a9fc71d67fa` | day 1, 2026-08-28; commit id was inside the hash | 79 |
| options | `1f7247d7839d9950` | 13 setups, definitions only | 732 |
| options | `ccd00ac71f8243e9` | from 2026-09-25, 15 setups | 798 |
| shares | `53229d6f1f24df10` | 2026-09-01 to 09-04, QQQ + SPY | 80 |
| shares | `b53ca8a58aa11718` | from 2026-09-05, 15 symbols | 1,704 |
| shares | `7c6414fc549315f3` | from 2026-09-25, 15 setups | 2,072 |

`live_lab_data/config/` also holds `1f176363aeff8f40` (QQQ only), `7b50d31a5dd61777` and
`8d4bb40056518b20`. None is accepted and no trade row carries one. Reason not recorded;
`runner --print-config` constructs a full `LiveLab` and therefore writes a config file
(`runner.py:635-639`, `:135`), which would explain strays **(inferred)**.

**What a person must never edit.**

1. Anything in `setups.py`: an id, a `params` entry, a cap, a timeframe, the list order — and
   equally the `evaluate`/`manage` bodies, which the hash does not protect.
2. The literals in `runner.build_config` and `shares_runner.build_config`.
3. `FREEZE.json` except by a dated amendment that *adds* a hash; the family size only grows
   (`forward_test_preregistration.md:170-171`).
4. Any existing line of `trades.jsonl`, `signals.jsonl`, the signal archives, `events.jsonl`,
   `outages.jsonl`, `session_ledger.jsonl`. Corrections are appended to
   `trade_corrections.jsonl`; the raw row stays (`gap_recovery.py:23-24`).
5. The clock rule: no session, setup, date, regime or symbol is ever excluded
   (`FREEZE.json:24`; `forward_test_preregistration.md:76-93`).

### 6.2 Option trade row — every field

From `positions.py:29-76`, `:140-160`, plus `seq` (`store.py:285-287`). Sample: first and
last lines of `live_lab_data/trades.jsonl`.

| field | meaning | trap |
|---|---|---|
| `seq` | counter shared with events, signals, outages | not a trade number |
| `position_id` | unique per row | the key for `trade_corrections.jsonl` in this arm |
| `signal_id` | shared by the up-to-three rows of one signal | join to DECISION/FILL on this |
| `config_hash` | definition identity | filter on the accepted list, all three |
| `setup_id`, `setup_version` | rule name, `"1"` | |
| `symbol` | QQQ or SPY | |
| `arm` | `ATM`, `ATM-1`, `ATM+1` | filter to `ATM` for the pre-registered result. `+1` means one strike *higher*, not further out of the money: for a put, ATM+1 is the in-the-money one (2026-10-12 row: QQQ put 748, delta −0.58) |
| `direction` | `long` / `short` | the view on the **shares**. The option is always bought: long → call, short → put (`positions.py:167`) |
| `right`, `strike`, `expiration`, `contracts` | contract; contracts is 1.0 | |
| `entry_ts` | wall clock of the fill | New York time, no zone |
| `entry_bar_ts` | stamp of the signal bar | open stamp for 1-minute setups, close stamp for 5-minute setups, so honest lag is ~61–76 s or ~2–15 s respectively (`dashboard.py:139-143`) |
| `entry_ask`, `entry_bid` | option quote at entry; filled at the ask | |
| `entry_spread_pct` | (ask − bid) ÷ midpoint | a **fraction**: 0.0069 means 0.69% |
| `entry_underlying` | share midpoint at entry | |
| `iv_derived`, `delta_derived` | computed, not vendor data | delta is **signed**: puts negative |
| `open_interest` | from the vendor snapshot, may be null | |
| `stop`, `target` | **share-price** levels | not option prices |
| `time_exit_min`, `bar_exit`, `trailing` | exit parameters from the signal | see §7.1 on `bar_exit` |
| `state` | the setup's own numbers at the signal | |
| `bars_held` | 1-minute bars after the entry bar | always 1-minute bars, whatever the setup's timeframe (`runner.py:535-536`) |
| `mfe_opt`, `mae_opt` (+ `_ts`) | best / worst option return seen at polls, bid ÷ entry ask − 1 | sampled every 5 s, not continuous; worst starts negative by the spread |
| `mfe_und`, `mae_und` | best / worst share move | already signed toward the trade |
| `last_bid`, `last_underlying` | last marks | |
| `exit_ts`, `exit_bid`, `exit_underlying` | the exit; filled at the bid | no `exit_ask` is recorded |
| `exit_reason` | `stop`, `target`, `eod`, `time`, `bars`, `trail_imb`, `failed`, `shutdown`, `eod_reconstructed`, or a `_after_close_mark` variant | `stop` does not mean the option lost money, and the stop may not have been breached at the fill |
| `trigger_level` | the stop or target that fired | null for clock and trailing exits |
| `pnl_gross` | (exit bid − entry ask) × contracts × 100 | dollars |
| `pnl_net` | gross − 2 × 0.0404 × contracts | 8 cents per round trip |
| `return_pct` | exit bid ÷ entry ask − 1 | a **fraction**, despite the name |
| `hold_minutes` | wall-clock minutes | |
| `underlying_return` | share move over the same holding period | **already signed toward the trade.** The correct "was direction right" test is simply `> 0`. Applying the sign again misclassifies every short (memory note `live-lab-record-traps.md`; code `positions.py:150-153`) |
| `fee_per_contract_side` | 0.0404 | |
| `exit_mode`, `caught_up_window` | only on exits rebuilt from history (`catchup.py:105-106`) | 6 `caught_up_from_history`, 42 `reconstructed_from_history` rows in the file |

There is no `date` field; select a session on `entry_ts[:10]` or `exit_ts[:10]`.

### 6.3 Shares trade row — every field

From `shares_runner.py:502-518`. Sample: `live_lab_data/shares/trades.jsonl`.

| field | meaning | trap |
|---|---|---|
| `seq` | shared counter (261,002 by 2026-10-12, because SKIP rows consume numbers) | |
| `arm` | always `SHARES` | |
| `config_hash`, `setup_id`, `symbol`, `direction` | | `short` here is a real short of shares |
| `entry_ts`, `exit_ts`, `entry_bar_ts` | as above | |
| `entry_px` | ask if long, bid if short | |
| `exit_px` | bid if long, ask if short | |
| `shares` | 10,000 ÷ entry price | fractional (13.37 shares) |
| `notional` | 10,000.0 | |
| `exit_reason` | `stop`, `target`, `time`, `bars`, `trail`, `eod`, `eod_stale_mark`, `eod_reconstructed`, `_after_close_mark` variants | `trail` covers both IMB's trailing stop and the Six_Lines "failed" exit, which the options arm names separately |
| `pnl_net` | shares × price change | **no fee or commission is subtracted**, despite "net" |
| `return_pct` | pnl ÷ 10,000 | fraction |
| `entry_spread_bp` | quoted spread at entry, in bp of the fill price | |
| `signal_id` | the correction key in this arm | |
| `hold_minutes` | | |
| `decision_lag_min` | fill time − signal bar stamp | absent on the earliest rows; ~1.1 is healthy (`shares_runner.py:512-514`) |

Stop, target and the other exit parameters are not in the trade row; they are in the FILL
row (`shares_runner.py:596-601`). There is no underlying twin and no excursion data.

### 6.4 Signal rows

`signals.jsonl` holds today only; earlier sessions are in `signals_archive/`. Always read
through `store.read_signals` (`store.py:75-100`).

**DECISION** (`store.py:255-266`): `seq`, `signal_id`, `phase`, `ts` (wall clock), `bar_ts`,
`setup_id`, `config_hash`, `symbol`, `direction`, `state`. The options arm adds to `state`:
`price_at_bar`, `vwap`, `bars_1m`, `bars_5m`, `dow`, `tf`, `slow_setup`, `dead_setup`
(`runner.py:428-433`). The shares arm stores the setup's state only
(`shares_runner.py:573-576`).

**FILL** (`store.py:269-273`), joined by `signal_id`; it carries no setup, symbol or hash in
the options arm.
- Options, opened: `status: "OPEN"`, `n_arms`, `atm_strike`, `strike_distance_pct` (a
  percent this time: `options.py:123`), `underlying_mid`, `expiration`, `arms` with each
  strike's `bid`, `ask`, `spread_pct_of_mid`, `iv_derived`, `delta_derived`.
- Options, not opened: `status: "SKIPPED"`, `skip_reason` one of `no_underlying_quote`,
  `no_0dte`, `unusable_quote: …`, `feed: …`, `option_quote_no_timestamp`,
  `stale_option_quote: …` (`runner.py:440-470`).
- Shares, opened: `status: "FILLED"`, `symbol`, `price`, `shares`, `spread_bp`, `stop`,
  `target`, `time_exit_min`, `bar_exit`, `trailing`, `timeframe`.
- Shares, not opened: `status: "SKIPPED"`, key **`reason`** (not `skip_reason`), value
  `bad_quote` (`shares_runner.py:580`).

**SKIP** (`store.py:275-283`): `seq`, a fresh unrelated `signal_id`, `phase`, `ts`, `bar_ts`,
`setup_id`, `config_hash`, `symbol`, `skip_reason`, plus `bar_age_min` for `stale_bar`
(options) or `direction` for `max_per_direction` (shares).

Traps in signal rows:

- **A SKIP means different things in the two arms.** Options: the setup really fired and was
  refused (`runner.py:383-401`). Shares `max_per_day`: written on every later bar once the
  cap is full, *without evaluating the setup* (`shares_runner.py:543-551`). On 2026-10-12
  that was 13,539 of the 14,016 rows in `shares/signals.jsonl`. They are not signals.
- **Stale-bar refusals live in different files.** Options: a SKIP row per signal. Shares: an
  outage row per bar (`shares_runner.py:530-535`); `shares/signals.jsonl` has zero
  `stale_bar` rows and `shares/outages.jsonl` has 4,495.
- **Status words differ:** `OPEN` (options) versus `FILLED` (shares).
- A feed failure while asking which expirations exist is outside the `try`
  (`runner.py:443` versus `:447`), so it would leave a DECISION with no FILL row and abandon
  the rest of that tick **(inferred)**.

### 6.5 Other files

- `daily/<date>.json` — options: `signals_decided`, `signals_skipped`, `fills_skipped`,
  `trades_closed` (all three strikes), `degraded_bars`, `atm_net`, `feed` counters
  (`runner.py:602-617`). Shares: `trades_closed`, `net`, `still_open`, `degraded_bars`,
  `feed` (`shares_runner.py:660-670`). 2026-10-12: options 21 decisions, 63 trade rows, ATM
  net −$144.70; shares 172 trades, net +$363.19.
- `session_ledger.jsonl` — `date`, `outcome`, `reason`, `attempts`, `is_session`,
  `recorded_at_et`, `utc_offset_hours`, plus extras (`ledger.py:151-161`).
- `six_lines_QQQ.json` — research output, 2,656 sessions, not a live record
  (`six_lines.py:392-394`).

### 6.6 Invariants

1. A DECISION is on disk before its FILL exists (`store.py:245-273`).
2. At most one DECISION per setup, symbol and bar (`store.py:400-417`).
3. The one-minute buffer is append-only and strictly ascending; a bar arriving behind the
   head is dropped for good (`session.py:198-199`).
4. A five-minute bar never spans two days and never has fewer than five minutes
   (`session.py:30-55`, `:328-333`).
5. One runner per arm (`lock.py`).
6. Nothing calls the machine clock directly (`clock.py:8-10`). Exceptions found:
   `lock.py:70` (a human-readable note) and `dashboard.py:72` (`date.today()`).

---

## 7. Failure modes, incidents, traps

### 7.1 Found during this read — not written down anywhere I could find

**A. On the options arm, `TTM_Squeeze` exits after about 10 minutes; on the shares arm, after
45.** The setup asks for an exit after 9 bars of its own five-minute timeframe
(`setups.py:306`; `session.py:146`). The shares arm multiplies by 5 (`shares_runner.py:481-482`),
as do both backtests (`walkforward.py:106`; `sharewf.py:146-147`). The options arm compares
against a count of one-minute bars with no multiplier (`positions.py:123`;
`runner.py:535-536`). The data agree: of the 44 ATM `TTM_Squeeze` rows in `trades.jsonl`, 38
exit on `bars`, and the first twelve of those hold 8.0 to 10.0 minutes (one holds 23.2); the
first twelve `bars` exits in `shares/trades.jsonl` all hold 45.0 minutes.
So the options record for this setup tests a different rule from the one specified
("time stop 9 bars", `live_lab_specification.md:345`). No commit message or research file
mentions it (`grep bar_exit` over the repo).

**B. On the options arm, a stop or target can fire on the very bar that produced the
signal.** `_manage_open` tests the latest admitted bar with no check that it came after the
entry (`runner.py:521`, `:539`). The shares arm skips bars at or before the entry bar
(`shares_runner.py:465-466`). In `trades.jsonl`, 14 of the 210 ATM rows that exit on `stop`
or `target` do so 6 to 8 seconds after entry — one poll, before any new bar could have been
admitted — so the bar that triggered them was at or before the signal. Example:
`VWAP_Reclaim` long, 2026-08-28 10:00, entry with shares at 721.02, stop 719.71, exit six
seconds later with shares at 720.89, reason `stop`. Those are spread-only losses caused by
the signal bar's own range.

**C. `PDH_PDL_Breakout` can produce a trade with no room at all.** Its stop is the signal
bar's low (or high) and nothing requires a minimum distance or that the bar points the same
way as the trade (`setups.py:329-341`). One row has stop and target both 708.12 (2026-09-01
09:50, short). Seven of the 14 rows in B are this setup. It is part of the frozen definition.

**D. A stale quote at the wrong moment silently drops that minute on the shares arm.** If
the quote is missing or stale on the tick that admits a bar, the symbol is skipped
(`shares_runner.py:442-443`) and that bar is never revisited — no stop check, no setup
evaluation, no SKIP row, only an outage row. The options arm writes a DECISION and a
`SKIPPED` fill in the same situation (`runner.py:440-442`).

**E. Half days are not handled on the options arm, and both arms mis-pick "yesterday" after
one (inferred).** The shares pre-registration says so for the first part: "The options arm
still has this defect; it is logged and not yet fixed" (`shares_lab_preregistration.md:89`);
`runner.py` has no early-close code. Separately, warmup drops any session with fewer than
300 bars (`session.py:275-276`) and the six-lines loader does the same
(`levels_live.py:71-74`), so on the day after a half day, "prior day high/low/close" and
"yesterday's" four lines would come from two sessions back. The `levels_live.py:23-25`
docstring says such a day would simply not trade; the code falls back instead. No half day
has occurred since 2026-08-28. The next is the day after Thanksgiving.

**F. `dashboard --today` looks broken (inferred, not run).** `load_open_positions` returns a
pair (`store.py:324`), but the dashboard takes `len()` of it and iterates it as if it were
the list (`dashboard.py:421-428`). It also prints `x['strike']`, which shares rows lack.

**G. The dashboard reads raw trades.** It never applies `trade_corrections.jsonl`
(`grep` finds no reference in `dashboard.py`), while the memory note says corrections must
always be applied and that raw totals were once off by about $2,030.

**H. The leak check assumes three rows per options decision** (`dashboard.py:205`, `:440`).
A skipped fill or a missing neighbour strike produces fewer, which would print as
"POSITIONS LOST" **(inferred)**.

**I. Sequence numbers can repeat after a restart.** Recovery scans events, signals and
trades but not `outages.jsonl` (`store.py:144-146`), which shares the counter.

### 7.2 Incidents already recorded, and the guard each left behind

| date | what happened | guard | evidence |
|---|---|---|---|
| pre-freeze | probe handed one extra bar: 1-minute open stamp vs 5-minute close stamp | timeframe-aware truncation | `README.md:88-90` |
| pre-freeze | indicators reset at 09:30 | 2-session prefix | `live_lab_specification.md:525-548` |
| 2026-08-28 | one upstream 503 during warmup threw away the session | wait 180 s and retry | `shares_runner.py:219-226` |
| 2026-09-01 → 03 | supervisor killed both arms at 15:55; every open position vanished | reconciliation table; reconstruction | `dashboard.py:184-191` |
| 2026-09-02 | 15:00 restart re-entered cap-1 setups | caps from DECISION rows | `shares_runner.py:261-265` |
| 2026-09-04 | feed down at close, three positions abandoned | fallback marks | `shares_runner.py:614-620` |
| 2026-09-04 | early-close detector fired on a dead link | require a healthy feed | `shares_runner.py:351-359` |
| 2026-09-08 | Labor Day counted as a prior session; preflight failed every symbol | exchange calendar in preflight | `preflight.py:271-281` |
| 2026-09-09 | 11 chase entries on a late start, −$352.22 | stale-bar guard | `shares_runner.py:74-80` |
| 2026-09-09 | quotes looked future-dated because of the reference point | `recv_ts` | `feed.py:281-288` |
| 2026-09-16 | coverage printed 100% with six sessions missing | calendar denominator | `ledger.py:333-336` |
| 2026-08-31 → 09-24 | both arms blind to the open | pre-open start; option-quote age check | `incident_2026-09-24_opening_window.md` |
| 2026-09-25 | host slept through 15:55; after-hours flatten | last in-session mark | `runner.py:566-570` |
| 2026-10-02 | restart resumed from `saved_at`, an hour of stops unchecked | `managed_through` | `store.py:298-302` |
| 2026-10-05/06 | network never returned; positions got no exit row | resolved later by `stranded.py` | `catchup.py:32-36`; `stranded_recovery.jsonl` |

### 7.3 Standing traps for whoever touches this next

- The lab runs whatever is checked out in the main checkout (`run_lab.cmd:6`; memory note
  `lab-runs-from-main-checkout.md`). An edit there is live code at the next import.
- The lab runs on the pyenv Python, not the repo's virtual environment (`run_lab.cmd:7`).
- Any change to setup logic changes what a trade means without changing the hash (§6.1).
- The two arms use different words and different files for the same events (§6.4).
- `underlying_return` is already signed; put deltas are negative; `*_pct` fields are fractions.
- Hit rate pooled across exit reasons is meaningless: a `stop` fires *because* the move was
  adverse (memory note `live-lab-record-traps.md`).
- Premarket stray prints set some of the six lines. On 2026-10-12 QQQ's S3 was 719.60
  ("yday premarket low") against a prior RTH low of 748.365 (`events.jsonl`, seq 9814).
  Whether those are bad ticks or real trades is unverified (memory note
  `premarket-stray-prints.md`).
- The runner still uses weekdays, not the exchange calendar, for its 25 prior days
  (`runner.py:222-228`). The 300-bar filter hides it; 2026-10-12 shows `sessions_used: 24`.
- Missing one-minute bars are recorded but do not stop one-minute setups from evaluating
  (`runner.py:325-336`). 2026-10-12 shares: XLY 2, XLV 1, XLP 3 gaps.
- Exits never check the age of the option quote; only entries do (`runner.py:461-470` versus
  `:531-533`).

---

## 8. Current state

**Done and running.** Last committed session 2026-10-12 (`git log`: `6f9c1ab`). Both arms
opened at 09:07:54 ET and the ledger reads `COLLECTED`. Options config `ccd00ac71f8243e9`,
shares `7c6414fc549315f3` (`daily/2026-10-12.json` in each arm).

**Record size** (`wc -l` and `grep -c` in the worktree, raw, corrections not applied):

- Options: 1,609 trade rows, of which 537 are ATM; 24 daily files.
- Shares: 3,856 trade rows; 26 daily files.

**Against the checkpoints.**

| arm | checkpoint | status |
|---|---|---|
| options | 50 and 100 total (diagnostic only) | passed (537 ATM rows) |
| options | 200 per setup | none there; the largest is `MOMO_CHASE` at 111, then IMB 77, `Crabel_Stretch` 50 |
| options | 377 per setup | none |
| shares | 200 per setup | reached by raw count for 9 setups (`MOMO_CHASE` 711, IMB 413, `VWAP_Reclaim` 402, …) |
| shares | 60 sessions — the one that governs | not reached (26 daily files) |

`ThreeBarPlay` has no ATM rows; `VWAP_2sigma_Fade` has 2 — consistent with their DEAD label.

**Half-done.**

- Options-arm early close (§7.1 E).
- Preflight's option-age check still samples its clock early
  (`incident_2026-09-24_opening_window.md:150-153`).
- The shares counterfactual for the blind-open period was listed as a follow-up
  (`incident_2026-09-24_opening_window.md:142-144`); whether it was done is outside this slice.

**Open.** Items A–I in §7.1; the premarket stray-print question; 2026-10-05, 10-06 and
10-07 have no daily file in either arm, although the ledger calls 10-05 and 10-06
`COLLECTED` (`ls live_lab_data/daily`; `session_ledger.jsonl`).

---

## 9. Contradictions — document or memory versus code

The code wins in every row.

| claim | where | what the code or data says |
|---|---|---|
| "A Context cannot hold a bar past its own instant" | `README.md:71` | It can, on a batch admission (`session.py:249`; `setups.py:656-658`) |
| `bar_ts` is the "CLOSE timestamp" | `session.py:67` | True for 5-minute bars only; 1-minute bars carry the open stamp (`runner.py:64-66`) |
| "The 13 frozen setups"; "family of 13" | `setups.py:1`; `__init__.py:18`; `README.md:116` | 15 since 2026-09-25 (`setups.py:752-757`) |
| Four setups labelled SLOW, including `ThreeBarPlay` and `PDH_PDL_FailedBreak` | `README.md:122-125` | SLOW is `Gap_Fade`, `EMA_9_20_Pullback`; DEAD is `VWAP_2sigma_Fade`, `ThreeBarPlay` (`setups.py:768`, `:787`) |
| Option subscription "cancels 2026-09-05"; ThetaData "lapsed to Options: FREE on 2026-09-08", nothing new can be measured | `README.md:48-50`; `CLAUDE.md` | "the entitlement came back on 2026-09-14" (`runner.py:54-56`); 63 option trade rows on 2026-10-12. Already flagged in `incident_2026-09-24_opening_window.md:154-157` |
| The clock uses "a fixed offset" from an MST machine | `runner.py:47-49` | `clock.py:19`, `:27-29` uses the `America/New_York` zone database |
| Timestamps "carry their offset" | `README.md:170`; `ledger.py:40` | `utc_offset_hours` is `null` on every ledger row sampled, because `now_et()` is zone-less (`ledger.py:158-159`) |
| Timestamps stored "as ET and as UTC epoch ms" | spec `:169` | New York wall clock only (`store.py:237`) |
| Underlying quote polled every 1 s | spec `:189-190` | One tick per 5 s or 15 s (`runner.py:308`) |
| A degraded bar means setups are skipped for it | spec `:199-200` | Only recorded (`runner.py:325-336`) |
| Per-signal record includes ema9, ema20, macd, adx, atr, rvol … | spec `:206-209` | Only the fields in §6.4 |
| Bootstrap 10,000 repetitions | spec `:242` | 4,000 (`dashboard.py:261`) |
| Split check: second half's interval must exclude zero | spec `:251-252` | Same sign only (`dashboard.py:397-399`); the later pre-registration also says sign only (`forward_test_preregistration.md:70`) |
| IMB "flat 15:59" | spec `:336` | 15:55 for everything (`positions.py:26`, `:117`) |
| "377 … Holm m=13" | `dashboard.py:31` | Family size is now 15 (`dashboard.py:32`) |
| Shares checkpoint is 666 IMB trades | `shares_lab_preregistration.md:100` | `shares/FREEZE.json:53-57` lists 200, 377 and 60 sessions; the dashboard shows 377 and 60 (`dashboard.py:31`, `:242-250`) |
| Shares arm "writes a SKIP row every time a capped setup fires again" | `store.py:43-45` | It writes before evaluating (`shares_runner.py:543-551`) |
| A holiday-shortened yesterday means no six-line trade | `levels_live.py:23-25` | It falls back to an earlier full day (`levels_live.py:65-74`) |
| Spec status "DRAFT … No code written" | spec `:3` | Built and frozen 2026-08-28 |
| Memory: autostart sleeps until 09:20 | `live-lab-forward-test.md` | 09:05 (`autostart.py:64`) |
| Memory: accepted hashes are two; shares-only is the steady state | `live-lab-forward-test.md` | Three per arm; options arm trading through 2026-10-12 |
| Memory: "`runner.py` also has no stale-bar entry gate at all" | `live-lab-record-traps.md` | It has had one since 2026-09-24 (`runner.py:73`, `:390-395`) |
| Ledger: 2026-10-05 and 10-06 `COLLECTED` | `session_ledger.jsonl` | No daily file for either; 21 and 18 option positions were resolved afterwards (`stranded_recovery.jsonl`) |

---

## 10. Questions only the owner can answer

1. **`TTM_Squeeze` on options exits after ~10 minutes instead of 45 (§7.1 A).** Intended?
   Fixing it changes what those trades mean without changing the hash, and 44 ATM rows
   already exist, 38 of them closed by the short rule.
2. **Options stops and targets can fire on the signal bar itself (§7.1 B).** Intended, or
   should the options arm wait for the next bar as the shares arm does? Same hash question.
3. **Which ThetaData options plan is active now, and until when?** The code says it returned
   on 2026-09-14; CLAUDE.md still says it lapsed.
4. **Do you count the 08:00–08:05 premarket spikes when you draw your lines?** They set
   live levels (QQQ S3 at 719.60 on 2026-10-12).
5. **Half days.** Before the day after Thanksgiving: should the options arm get the
   early-close rule, and what should "yesterday" be on the Monday after?
6. **Which shares checkpoint governs — 666 IMB trades (pre-registration) or 60 sessions
   (FREEZE.json)?** Both, or does the later one replace the earlier?
7. **`PDH_PDL_Breakout` zero-distance stops (§7.1 C).** Known and accepted as part of the
   frozen rule?
8. **Should the dashboard apply `trade_corrections.jsonl`?** Today it reports raw rows.
9. **The three unaccepted config files** (`1f176363aeff8f40`, `7b50d31a5dd61777`,
   `8d4bb40056518b20`). Pre-freeze leftovers that can be ignored?
10. **Shares `max_per_day` SKIP rows.** They are written whether or not the setup fired. Is
    that the "honest denominator" you wanted, or should they mean "a real signal was capped"
    as in the options arm?
