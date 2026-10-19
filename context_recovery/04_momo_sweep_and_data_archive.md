# Slice 04 - the MOMO_CHASE sweep (2.8M cells) and the data archive

Status: COMPLETE PASS (final). Sections 1-10 below follow the brief. The detail and the
answers to the seven numbered questions sit after them in "PART A" (sweep: Q1, Q2) and
"PART B" (lookahead Q3, inventory Q4, vendors/entitlement Q5, manifest Q6, stray code Q7).

Paths are relative to the repo root
`C:\Users\chaitanyakharche\Documents\stock\.claude\worktrees\blissful-kalam-999916`.
Everything is static reading; no project code, test, install or network call was run.
`path:line` = I read that line. **(inferred)** = I did not read it directly. **[LS]** = I listed a
directory read-only (file counts, no file content) on 2026-10-13. Two read attempts on
`Desktop\data\...\entitlements.json` and a copy of `manifest.sqlite` were blocked by the
permission layer; I did not route around that, so those two files are unread.

## 1. In one paragraph

This slice is (a) the pre-registered, 2,822,400-cell brute-force search that asked whether ANY
setting of the rule "MOMO_CHASE" (the trader's behaviour, reconstructed as a 5-minute
momentum-following rule) beats not trading on QQQ in 2016-2021 - run on the Northeastern
Discovery cluster as 500 array tasks and answered NO (Hansen SPA p = 0.8227; the best of 2.8M
cells is exactly what luck produces); and (b) the plumbing that bought, downloaded, verified and
stored the historical stock and option data every backtest rests on (a ThetaData bulk-download
package writing to `Desktop\data`, a resumable SQLite manifest, thin vendor clients, and the
audit scripts born from the project's defining mistake: a 5-minute "lookahead" that bought
options before the signal existed and manufactured 88% of a measured edge). The only option
archive that exists is SPY 0DTE, 769 sessions, 2020-01-03 to 2024-08-07, 1-minute bid/ask.

## 2. The goal it serves

Single purpose: stop the owner (and the AI helper) believing a confident number that measured
nothing, and bound what history can and cannot say about a rule he trades live.
How it changed, with dates:
- 2026-08-14: first option data = Massive free tier (traded prices, no bid/ask, ~2 years)
  (`massive_client.py:1-25`).
- by 2026-08-13/14: switched to ThetaData Options VALUE ($40/mo) for real bid/ask, then upgraded
  Stock to STANDARD ($80/mo) on 2026-08-15 for 1-minute stock bars back to 2016
  (`spy_qqq_0dte_quotes_backtest.py:2-4`; `thetadata_client.py:3-4,63-71`; $ figures from memory).
- 2026-08-16: the 5-minute lookahead found; the two studies it was bought for are void (B1).
- 2026-08-20: `bulk_download` package built and the archive pulled (`config.py:1-6,34-40`;
  `Desktop\data\reference` timestamps Aug 20, `manifest.sqlite` Aug 21 [LS]).
- 2026-08-20: three setup pre-registrations (24-setup sweep run; cross-check run; standalone NOT run).
- 2026-09-04/13: `momo_v2.py` pre-registered, then run (3 hand-picked variants, null).
- 2026-09-08: ThetaData options dropped to FREE; archive can no longer be extended (then);
  VRP work forced onto 5-minute bars because the 1-minute SPY 2024-01..07 hole could not be refetched.
- 2026-09-13/16: the 2.8M-cell sweep pre-registered, audited, run, collected (null).
- 2026-09-14: options entitlement apparently restored (B4) - docs mostly not updated.

## 3. How it works, end to end

Sweep (detail A2-A3): `pack.py:53-94` (per-session pickles -> float64 `.npz`, >=300 bars) ->
scp to `~/momo/data` -> `sweep.py --validate` (V0 must be n=8,598, ~+$0.03; `sweep.py:152-193`) ->
`stats_test.py` (6 checks) -> `audit.py` (lookahead + oracle power + direction placebo) ->
`sbatch submit_sweep.sbatch` (array 0-499, shard i = cells with index % 500 == i,
`sweep.py:204`; each shard reduces itself to a (10,000,) bootstrap-max vector, `sweep.py:239-270`)
-> `collect.py --expect 500` (refuses on mismatched `session_hash/spread/block/reps` or missing
shards, `collect.py:74-109`; element-wise max across shards = exact grid-wide Hansen SPA,
`collect.py:113-123`) -> decision text pre-committed in the pre-registration §8.
Data (detail B5): `probe_entitlements.py` (measure tier + floor) -> `universe.py` (verify symbols)
-> `download_reference.py` (per-symbol session lists, expirations) -> `download.py` (Layer ->
tasks clipped to real sessions, newest first, manifest-gated resume) -> `theta_v3.fetch_csv` (4-way
classification, `.tmp` then rename) -> `manifest.sqlite` -> `to_parquet.py` (typed, ET-localised).
Consumers: `trade_analysis/hpc/build_vrp_dataset.py` reads `Desktop\data\raw` directly;
the live lab reads `trading_days.py` only (`live_lab/feed.py:29-33`).

## 4. Map

`trade_analysis/momo_sweep/`: `__init__.py` empty (0 lines) - `grid.py` the frozen 13-axis universe,
counted before it is run - `engine.py` per-session evaluation of every cell using the proven
`Panel` indicators; returns gross P&L, trades, shares - `sweep.py` loader, `--validate`, shard
runner, per-shard bootstrap reduction - `pack.py` pickles -> npz - `stats.py` own Hansen SPA,
stationary bootstrap, MinBTL floor, effective-trials measure - `stats_test.py` 6 checks of stats.py -
`audit.py` pre-conditions 3-4 - `collect.py` reducer + verdict - `submit_sweep.sbatch` the Slurm job.
`trade_analysis/backtesting/momo_v2.py` the 3-variant precursor.
`research/`: `momo_sweep_preregistration.md` (design, frozen 09-13), `momo_sweep_precondition3.md`
(the audit rewrite, 09-15), `momo_sweep_results.md` (result 09-16), `momo_v2_results.md`,
`setup_sweep_*` (24 setups vs his trades), `setup_crosscheck_*` (10 standard setups, coverage
saturated), `setup_standalone_preregistration.md` (never run), `spread_survey_results.md`.
`trade_analysis/bulk_download/`: `config.py` paths + tier + gateway limits; `layers.py` the
dataset catalogue (one dict entry per dataset); `download.py` the engine; `download_reference.py`
small reference lists; `manifest.py` SQLite ledger; `probe_entitlements.py` tier/floor prober;
`status.py` report; `theta_v3.py` streaming client; `to_parquet.py`; `trading_days.py` calendar
(also used by the live lab and HF Space); `universe.py` SPY+QQQ+constituents; `live_recorder.py`
1-second snapshot recorder (polls `symbol=*` once per second; sample not stream);
`__init__.py` empty.
`trade_analysis/data_sources/`: `thetadata_client.py` interactive options/stock client with JSON
cache and a hard refusal of non-`/option/` `/stock/` paths; `massive_client.py` rate-limited
Polygon client (free tier); `unified_data_provider.py`, `collect_alt_data.py` news/reddit
collectors; `download_5min_alpaca.py`, `download_daily_yfinance.py`, `crypto_history.py`,
`crypto_live_polling.py`, `forex_signal_generator.py` dead earlier-app scripts (B6); `__init__.py` empty.
`trade_analysis/config.py` env-key loader that raises if Finnhub/Reddit keys absent; `paths.py`
filesystem anchors (`.cache`, `local_data`, `logs`, `data`).
`trade_analysis/backtesting/`: `decision_time_audit.py` honest-timing fix for the 5-minute bug;
`underlying_orb_longrun.py` the void decade test; `spy_qqq_0dte_quotes_backtest.py` (real NBBO)
and `spy_qqq_0dte_real_backtest.py` (Massive) - both inherit the bug via `find_entries`;
`vilkov_0dte_conditional_backtest.py` SPXW panel study; `spread_survey.py` 51-name spread measure.
`local_data/` 9 Finnhub-news JSON files (B6).

## 5. Design decisions and the reason for each

| decision | reason (evidence) |
|---|---|
| Enumerate the WHOLE grid and fix it before running | a data-snooping correction needs the size of the search; a hand-picked shortlist has none (`grid.py:5-20`, `momo_v2_results.md:57-64`) |
| Write own SPA instead of `arch.SPA` | `arch` 8.0.0 `studentize=True` is a no-op (40/40 identical p-values) (`stats.py:5-24`) |
| Sessions outer loop, cells inner, reuse `Panel` | speed ~grid size, and no re-implemented indicators to disagree with V0 (`engine.py:1-17`) |
| Return per-session sums, apply $0.01/share cost afterwards | session is the independent unit; gross can be re-costed; but "a gross number is not a result" (`engine.py:36-50`) |
| Cost per share, not per bp | QQQ went ~$100 -> ~$700; a $10k clip is 100 shares then, 14 now (`engine.py:132-146`, prereg §3) |
| Each shard reduces to a bootstrap-max vector | full matrix ~34 GB; max is associative (`sweep.py:225-231`, `stats.py:33-47`) |
| `.npz` not pickle | pickle is code; cluster interpreter differs (`pack.py:9-19`) |
| Discovery = QQQ 2016-2021 only; 2022-2026 holdout unread (`sweep.py:113-114`) | one-shot holdout, plus SPY adds a second instrument, entirely in holdout (prereg §1) |
| Two gates (SPA and MinBTL Sharpe floor) | SPA says "could luck produce this t"; MinBTL says "is the Sharpe big enough after a search this size" (`stats.py:174-226`, prereg §4b) |
| Include `direction=fade` and `gate_ref` as axes | testing only "chase" would assume the answer (`engine.py:31-34`, `grid.py` axes) |
| Pre-condition 3 amended before any shard ran | original rule arithmetically invalid for a negative baseline; first oracle wasn't one (`momo_sweep_precondition3.md:81-98`) |
| Drop `srun`, `exec` the interpreter | 15 tasks died on inherited CPU-bind (`submit_sweep.sbatch:164-183`) |
| Bulk archive outside the repo | "tens of GB would poison every git operation" (`bulk_download/config.py:10-13`) |
| Manifest, not file existence, drives resume | empty results leave no file (`manifest.py:1-13`) |
| Four-state response classification | a swallowed exception once dropped 40 of 256 sessions (`theta_v3.py:28-31`) |
| Layer registry as data | one code path for orchestrator, manifest, estimator, converter (`layers.py:1-6`) |
| Raw csv.gz kept exactly as returned; parquet is the working copy | "makes every later decision re-checkable" (`to_parquet.py:3-5`) |
| Hard refusal of `/index/` in `ThetaDataClient` | Index is FREE; calls would only produce 403 noise or imply an upgrade (`thetadata_client.py:55-60,110-116`) |
| Honest-timing as a separate module, source left buggy | repo rule: never edit proven code in place (memory `feedback-isolate-experimental-strategy-code`); consequence in B1 |
| Archive tidy tool is an allowlist | an age rule would have deleted 2.3 GB of irreplaceable data (`scripts/tidy.py:9-23`) |
| VRP work moved to 5-minute bars | 1m SPY 2024-01..07 unfillable; 5m is the realised-variance standard (`c1ea49a`, `vrp_baseline_results.md:110-120`) |

## 6. Records, formats, invariants, frozen things

- The sweep universe (`grid.py`, 2,822,400 cells, V0 = cell #928,056), `TOP_KEEP=200`, seed
  20260913, B=10,000, mean block 5 sessions, spread $0.01/share, discovery 2016-01-01..2021-12-31:
  pre-registration §9 forbids adding, removing or re-ranging a cell, re-running at a different
  cost, or touching the 2022-2026 holdout (it must not be: §8 row 1 fired).
- Cell ordering = `itertools.product` order with dedup (`grid.py:115-130`); changing an axis list
  or order silently re-labels every shard's `cell_index`.
- Shard integrity: `session_hash` (sha256 of session labels), `spread`, `block`, `reps` must match
  across all 500 shards (`collect.py:74-81`). Result session hash `ad6011cfb4c6128d`
  (`momo_sweep_results.md:32`).
- Archive: raw csv.gz is the master; the 769-session SPY 0DTE set, 2024 block = pre-registered
  held-out (VRP), "touched exactly once". `manifest.sqlite` is the only ledger of requests.
- ThetaData licence: personal use, destroy on termination, never commit raw vendor quotes (memory).
- `live_lab_data/` is the frozen forward test (CLAUDE.md); not in this slice's write set.
- Do NOT edit `find_entries` casually: the fix lives in `decision_time_audit.honest_entries` by design.
- Do NOT use `data_sources/*` from the live lab: it deliberately imports none (`live_lab/feed.py:3-4`).

## 7. Failure modes, incidents, traps

Sweep (all documented in file headers): V0 mismatch = a gate or session-admission difference, not
rounding (`sweep.py:186-188`); `sqrt(n)` inflated t to ~96 and silently disabled Hansen's recentring
(`stats.py:111-119`); a truncated shard looks like a good file (`submit_sweep.sbatch:46-51`);
`/scratch` is purged monthly (`:63-64`; destroyed the TFT source once); `conda create` on a login
node gets SIGKILLed (`:119-127`); `.gitattributes` must pin `*.sbatch` LF or `#SBATCH` lines become
comments (memory `momo-sweep-null-result`); the collector printed a rank as a percentile, reading as
the opposite - V0 is in the BOTTOM 6%, not "94.1th" (`collect.py:299-304`, commit 8860e1f); NaN != NaN
makes `cells.index(v0_cell())` miss V0 (`grid.py:143-148`); one-draw placebo gives NaN and a silent
PASS unless guarded (`audit.py:344-348`); a sweep over fewer shards is a valid test of a SMALLER
family, not a biased one (`collect.py:94-103`).
Data: NO_DATA vs NOT_ENTITLED vs ERROR must never be conflated (`theta_v3.py:14-30`); wrong strike
format "532000" silently returns 472 = "no such contract" and cost an hour of believing data was
unavailable (`theta_v3.py:242-244`); naive Eastern timestamps on a Mountain-time machine shift every
window by 2 hours (`thetadata_client.py:35-37`, `to_parquet.py:11-16`, `massive_client.py:19-23`);
`venue=utp_cta` is required for any range touching today (`config.py:72-75`); 16:00 stub bar and
09:30 `bid=0/ask=0` rows (`live_lab/feed.py:9`, `broker_entry_state_study.py:27-28`); zero-filled
holiday closes fabricated ~7% of entries; a halt bar with all-zero OHLC made `log(0)=-inf` and poisoned
170 sessions (`vrp_baseline_results.md:133-141`); native month files can be truncated (SPY_2020-02:
3 days vs 22 for March); the gateway returns "io exception" 500s under load so concurrency is 4 stock /
2 option (`config.py:55-59`); `qqq_constituents.csv` is today's membership = survivorship bias if used for
history (`universe.py:11-17`); ThetaData requires the local Java terminal on 127.0.0.1:25503
(`theta_v3.py:157-160`); `trade_analysis/config.py` raises ImportError without Finnhub/Reddit keys
(`config.py:12-13`), so anything importing `unified_data_provider` fails on the cluster or a clean venv.
The big one - the 5-minute lookahead - is B1.

## 8. Current state

- Sweep: DONE and CLOSED (null). Nothing further is allowed on it; holdout never read.
- `momo_v2`: done, null. 24-setup and cross-check: done, null. `setup_standalone`: pre-registered,
  never run, no results file (grep found no reference).
- Archive on disk [LS]: SPY 0DTE 769 sessions; 104-symbol stock layers (eod 47 MB, 1m 1.4 GB, 5m
  395 MB, quote-1m 760 MB); tiny 1-second and tick samples; reference lists; manifest dated Aug 21.
  Nothing new has been added since the 2026-09-08 lapse that I can see.
- Entitlement: UNKNOWN today. Live options arm records trades since 09-14 and a 09-29 replay pulled
  history for 09-08..09-11, so options access is evidently back at least at the level the live
  feed needs (B4). No document in the repo records the purchase or the tier.
- `decision_time_audit` re-ran 2 of 6 affected modules; 4 never re-run.
- HF Space and news files: `local_data/` is shipped to the Space; no research uses it.

## 9. Contradictions (doc or memory versus code; code wins)

1. `CLAUDE.md` settled-table row "MOMO_CHASE form, 2,822,400 cells, p=0.8227" points to
   `setup_sweep_results.md`. That file is the 24-setup sweep (23 setups, no p=0.8227). The sweep
   result is `research/momo_sweep_results.md`.
2. `CLAUDE.md:81` "A 1-minute lookahead once supplied 88-95.7%": two incidents merged. 88% is the
   5-minute label bug on QQQ's underlying; 95.7% is the 1-minute quote-stamp bug on IMB option
   return. `PROJECT_REPORT.md:90` and `huggingface_space/tools/build_research_snapshot.py:178` call
   the range the 5-minute bug alone.
3. `CLAUDE.md:73`, `PROGRAMME_REVIEW_2026-09-24.md:55-56,350`, `scripts/tidy.py:16-17`,
   `breakout_sweep.py:16`, `breakout_options_preregistration.md:35`, `hpc/README.md:40`,
   `vrp_preregistration.md:42`: ThetaData options "lapsed to FREE on 2026-09-08", "cannot be
   rebuilt / nothing new". Versus `live_lab/runner.py:53-57` ("entitlement came back on
   2026-09-14"), the live options arm trading since 09-14, and `backfill.py`/`replay_22_sessions.md`
   which pulled history quotes for the blocked days. Same-day (09-24) documents disagree.
4. `collect.py:170-171,203-204` still say a partial reduction "is biased DOWN"; its own comment
   (`:94-103`, commit a782478) says that is wrong (scope defect, not bias).
5. CPU-hours: ~700 (cluster, `submit_sweep.sbatch:45-47`), ~177 (laptop, `precondition3.md:142-145`),
   ~756 (`grid.py:154`, stale 3,731 cells/CPU-h). `stats.py:35-36` still cites 3,815 sessions x
   1,038,800 cells (current: 1,497 x 2,822,400). `stats_test.py:5` says "Four properties"; there are six.
6. `momo_sweep_results.md:60-63` "within 2%": 1.688 vs 1.65 is +2.3%.
7. `thetadata_client.py:270` says "Stock Value tier" in a body comment while its header says Stock
   STANDARD since 2026-08-15 (stale remnant).
8. `underlying_orb_longrun.py:24-26` still prints the void "gets 80.3%"; no banner in the file.
9. "SPY 0DTE 2020-2024 769 sessions" (CLAUDE.md, hpc README): true count, but the span is
   2020-01-03..2024-08-07 [LS], 2021 has only 39 sessions, and "the 2024 block" is Jan 2-Aug 7 only.
10. `hpc/README.md:37` says `stock_ohlc_1m/SPY` is "continuous 2016-2026"; `vrp_baseline_results.md:110`
    says it is missing 2024-01..07. And `bars_cache` SPY starts 2022-01-03 (`momo_sweep_preregistration.md:53`)
    while `download.py:78-82` says SPY stock history is "unusually shallow ... not on the UTP tape".
11. `vrp_baseline_results.md:110-113`: `stock.ohlc.1m` NOT ENTITLED "despite the terminal reporting Stock:
    STANDARD" - unexplained, while the shares arm used `/stock/history/ohlc` throughout.
12. `momo_sweep_preregistration.md:13-14` says the standalone-setup project "was never authorised"; but
    `setup_standalone_preregistration.md` (2026-08-20) exists and authorises exactly that, status "NOT
    YET RUN". Never run, never withdrawn.

## 10. Questions only the owner can answer

1. Did you re-buy ThetaData Options (Value $40?) around 09-12..09-14? Which tier today, what billing
   date, is it current? (Nothing in the repo records it.)
2. Was the historical floor ever re-probed after 09-14? Is it 2020-01-01 (docs table) or a rolling
   ~4 years (marketing)? This decides whether 2020-2022 option history is re-downloadable at all.
3. Why does the SPY 0DTE archive stop at 2024-08-07 and hold only 39 sessions in 2021? Deliberate
   `--end`/gap, or interrupted? Is any of 2025-2026 SPY or any QQQ 0DTE wanted now that history may work?
4. Why did `stock.ohlc.1m` return NOT ENTITLED for SPY 2024-01..07 on a Stock STANDARD account?
5. Was the Databento $125 credit ever used; was `cbbo-1m` ever priced?
6. Is `setup_standalone_preregistration.md` abandoned? (Hygiene: mark it so.)
7. Are the crypto/forex/alpaca/yfinance scripts and `local_data/` safe to delete or archive?
8. Should the `CLAUDE.md` "cannot be measured" list be reopened for QQQ options if history works?
   That is your call; I only report that its stated reason may be stale.

---------------------------------------------------------------------------------------------
Everything below is the supporting detail and the question-by-question answers.
---------------------------------------------------------------------------------------------

## PART A - THE MOMENTUM SWEEP

### A1. The sweep in plain words (answers question 1)

**What MOMO_CHASE is.** It is one of the live lab's setups: a rule written down by watching what
the trader does, not by looking at what makes money. On a 5-minute chart, between 10:30 and 14:30
ET, if the stock's last 15 minutes moved by at least 0.6 "sigmas" (sigma = the usual size of a
1-minute move that day, scaled to 15 minutes) and three trend indicators agree with the move's
direction (MACD histogram, DMI, ADX above 20), then **buy if it moved up, sell short if it moved
down, and get out 25 minutes later**. No stop, no profit target. At most 3 trades a day.
`trade_analysis/live_lab/setups.py:147-207` (class `MomoChase`; thresholds at :158-169, rule at
:171-207). The docstring says "Reconstructed from measured behaviour only. Expected to fail."
(:148). Its numbers came from the journal study of his behaviour, "never to his outcomes"
(:163). "Chase" = trade in the direction of the move.

**What was varied - how you reach 2,822,400.** Thirteen settings ("axes") each given a short
list of values, in `trade_analysis/momo_sweep/grid.py:51-91`:

| axis | values | count |
|---|---|---|
| lookback minutes `trail_min` | 5,10,15,20,30,45,60 | 7 |
| entry threshold in sigmas `sigma_mult` | 0.2,0.4,0.6,0.8,1.0,1.25,1.5,2.0 | 8 |
| hold minutes `time_exit` | 5,15,25,40,60,90 | 6 |
| ADX minimum | off,20,25,30 | 4 |
| DMI timeframe | off,1m,5m | 3 |
| MACD gate | off,on | 2 |
| time-of-day window | whole day; 09:30-10:30; 10:00-13:00; 10:30-14:30 (the frozen one); 13:00-15:55 | 5 |
| direction | chase, **fade** (do the opposite) | 2 |
| EMA9-distance gate | off, 0.71 ATR | 2 |
| max trades/day | cooldown only, 3 | 2 |
| `gate_ref` (do gates confirm the trade sign or the momentum sign) | trade, momentum | 2 |
| `sig_norm` (threshold in whole-session sigma, or sigma for that minute of day over 14 past sessions) | session, tod | 2 |
| `vol_regime` (prior-day range tercile) | off, high, low | 3 |

Product = 7x8x6x4x3x2x5x2x2x2x2x2x3 = **3,870,720** nominal. `build_grid()`
(`grid.py:104-130`) drops cells that are bit-identical to another: `gate_ref="momentum"` is
meaningless for `chase` (same sign) or when no sign-sensitive gate (MACD/DMI/EMA9) is on
(`grid.py:118-120`). I checked the arithmetic by hand (not by running it): `trade` half =
1,935,360; `momentum` cells = 1,935,360 x 1/2 (fade only) x 11/12 (some sign gate on) = 887,040;
sum = **2,822,400**, matching `research/momo_sweep_preregistration.md:68-70` and
`research/momo_sweep_results.md:7-8,34`. NOTE: the `adx_min` gate is *not* counted as a "gate on"
in the dedup test (`grid.py:118`), but that does not change the count I verified.

**Deliberately not swept:** MACD (9,17,9) and DMI length 14 - "calibrated to the trader's
BEHAVIOUR ... never to P&L" so sweeping them would turn a measurement into a fitted value
(`grid.py:35-40`, `engine.py:24-29`). No stop/target axis: the engine is time-exit-only
(`momo_sweep_preregistration.md:109-114`, reason: literature holds the window with no stop). No
option expression: shares are "the generous case" (`:112-114`).

**Why sweeping this many cells is dangerous.** If you try 2.8 million variants of a rule that has
no real edge, the best one will look wonderful purely by luck - like asking 2.8 million people to
flip a coin ten times and being impressed that one got ten heads. Here the winner looked like a
discovery: net +$2.32 per session, annualised Sharpe 1.688 (Sharpe = average return divided by its
day-to-day wobble, scaled to a year; ~1 is "good"), and its breakeven cost was $0.155 per share,
15x the penny actually charged. 31.1% of all cells made money (`momo_sweep_results.md:44-50,
103`).

**What test corrects for it.**
- *White's Reality Check*: asks "is the best of this fixed list really better than doing nothing,
  or is it just the luckiest of the list?" by re-shuffling the data many times and seeing how good
  the *best* cell looks by luck alone.
- *Hansen's SPA (Superior Predictive Ability)*: the same idea, but it ignores obviously-hopeless
  cells so a grid full of junk does not drown a real winner. "Studentized" = each cell's mean is
  divided by its own noise level so thin-trading cells cannot win on volatility.
This repo implements SPA itself (`momo_sweep/stats.py`) because `arch` 8.0.0's
`SPA(studentize=True)` silently ignores the flag (40 of 40 identical p-values)
(`stats.py:5-24`); the pre-registration says that costs "about an order of magnitude of power;
measured here at 0.57 vs 0.00 on Hansen's own design" (`momo_sweep_preregistration.md:147-151`).

**Why a huge grid is still OK.** A correction needs the *size of the search* known in advance. A
hand-picked shortlist (what `momo_v2.py` did - three variants chosen after looking at an AUC table)
has an unknowable size; a fully enumerated grid has an exact one (`grid.py:5-20`;
`momo_v2_results.md:57-64`). Neighbouring cells are ~0.99 correlated so the bootstrap charges them
almost nothing (`grid.py:22-33`); structural changes (fade, different window) cost nearly a full
trial.

**The pre-registered decision rule** (`momo_sweep_preregistration.md:262-269`, written 2026-09-13
before any run):
- SPA p >= 0.05 on the 2016-2021 QQQ discovery set, net of cost -> **MOMO_CHASE is closed as a
  standalone strategy**; the 2022-2026 holdout is never touched.
- SPA p < 0.05 -> only "a hypothesis"; survivors (found by Romano-Wolf StepM) go to the holdout
  once, QQQ *and* SPY; even a pass earns only a new forward-test pre-registration.
- significant gross but not net -> "a cost failure, not an edge".
- significant on QQQ but not SPY in holdout -> failed.
A second gate was added 2026-09-14 (`:157-198`): the winner's Sharpe must also clear a floor
derived from the search size (MinBTL, using the *measured* effective number of independent
trials). Both gates required.

**The result, p = 0.8227, for a trader.** `momo_sweep_results.md:7-23`: SPA p(consistent) = 0.8227
(lower 0.7276, upper 0.8500), observed max t = +4.0962. In plain words: if the rule had *no* edge
at all, searching 2.8M variants would produce a best cell at least this impressive about **82% of
the time**. So the best cell is exactly what luck gives you. Nothing in the family - chase or fade -
beats not trading after a penny-per-share round-trip cost. A trader should read it as: there is
no setting of this momentum rule, on QQQ in 2016-2021, worth trading; the apparent winners are
noise.

**"The best cell's Sharpe is what pure noise would give."** Before the run, section 5 of the
pre-registration said the expected best Sharpe from pure luck is "≈1.65 annualised at 5,000
independent trials" (`momo_sweep_preregistration.md:200-206`). The observed winner was 1.688
(`momo_sweep_results.md:60-63` - the doc says "within 2%"; 1.688/1.65 is +2.3%). The measured
effective number of independent bets was 31.4 (`:74`), so the real "how many tries" is far below
2.8M, but even so the winner is unremarkable. The winner is a **fade** (afternoon window, 1.25
sigma, ADX>=30, other gates off) (`:65-68`), so the null covers mean-reversion too.

**The frozen V0 is bad even inside its own family.** The exact live-lab parameterisation is cell
#928,056; it ranks 2,656,847 of 2,822,400 (beats 5.9%), net -$1.9352 per session, breakeven
-$0.0058 per share (negative = loses money *before* cost) (`momo_sweep_results.md:107-117`). Four
caveats in the doc: QQQ-only; 2016-2021; sweep V0 uses `max_per_day=0` not the live 3/day;
`--validate` over QQQ+SPY all years still gives +$0.0339/trade gross at n=8,598 (`:119-127`). It
"weakens the prior substantially without settling the live test" (`:129`).

**Operational consequence: none, by design.** The live lab is a frozen forward test; pulling a
setup because a retrospective search disliked it would turn a prospective test into a
retrospective one (`:131-141`; preregistration §9 `:271-278`).

### A2. How it ran on the cluster (answers question 2)

Order, as written in the header of `momo_sweep/submit_sweep.sbatch:17-25`:

1. **Pack** the 1-minute bars: `python -m trade_analysis.momo_sweep.pack --out $HOME/momo/data`.
   `pack.py:53-94` reads `live_lab_data/bars_cache/{SYM}_*.pkl` (one pickle per session), keeps
   sessions with >= 300 one-minute bars (drops half-days; copied from
   `journal_zone_wf.sessions()`; `pack.py:50,67`), and writes plain-number `.npz` arrays at
   float64 (`pack.py:86-93`). Why: pickles are code and the cluster runs a different interpreter;
   `.npz` is ~18 MB (`pack.py:9-19`, `submit_sweep.sbatch:55`). Failed pickles are counted, not
   silently skipped (`pack.py:34-35,77-80`). Copied to Discovery with `scp` to `~/momo/data`
   (`submit_sweep.sbatch:56-64`) - **not** `/scratch`, which is wiped monthly (`:63-64`).
2. **Precondition 1 - `sweep.py --validate`** (`sweep.py:152-193`): run only the frozen cell and
   require n = 8,598 trades and mean ≈ +$0.03, the numbers `momo_v2.py` printed on 2026-09-13
   (`sweep.py:38-40,179-181`). Fails -> do not run. Data = QQQ+SPY, all years.
3. **Precondition 2 - `stats_test.py`** (`stats_test.py:26-177`): six tests - shard reduction equals
   whole-grid max, studentization power on Hansen's Example 4, block structure, effective trials,
   t-statistic scale (a `sqrt(n)` bug had inflated t to ~96; `stats.py:103-122`), MinBTL calibration
   to Bailey et al.'s worked example. Test 2 SKIPs where `arch` is absent
   (`momo_sweep_precondition3.md:106`).
4. **Preconditions 3 and 4 - `audit.py`**. See A3.
5. **Submit**: Slurm array `--array=0-499`, 500 tasks, 2 CPU, 8 GB, 6 h limit, `short` partition,
   50 concurrent (`submit_sweep.sbatch:2-9,32-34`). Shard `i` takes cells `j` with `j % 500 == i`
   (`sweep.py:204`), about 5,645 cells each. Header refuses to run if grid cells < tasks
   (`submit_sweep.sbatch:131-145`) and the sbatch **asks the code for the grid size** rather than
   trusting the hand-written `--array` (`:131-139`). Window passed explicitly:
   `--symbols QQQ --start 2016-01-01 --end 2021-12-31` (`:188-195`); the holdout is never read
   (`sweep.py:113-114`).
6. **Inside a task** (`sweep.py:199-273`): sessions are the outer loop and cells the inner loop,
   so the expensive indicator build (`Panel`) happens once per session and is reused by all
   cells (`engine.py:1-17`) - and the indicators are the *proven* code, not a re-implementation,
   which is what makes V0 reproduce. Returns gross P&L, trade count and shares per session
   (`engine.py:132-266`); cost = shares x $0.01 applied afterwards (`sweep.py:237`). Each task
   **reduces on the spot**: it runs its own stationary bootstrap (`stats.boot_indices`, fixed seed
   20260913, 10,000 resamples, mean block 5 sessions; `stats.py:74-100`, `sweep.py:239-240`) and
   writes only a (10,000,) vector of the per-resample maximum over its cells, plus per-cell
   summaries and the series of its best 200 cells (`TOP_KEEP`; `sweep.py:196,247-270`). Reason:
   the full matrix would be ~34 GB (`sweep.py:225-231`). A maximum can be combined across shards
   exactly, because max is associative (`stats.py:33-47`).
7. **Collect** - `collect.py --results $HOME/momo/results --expect 500` (`collect.py:164-176`):
   - refuses if shards disagree on `session_hash`, `spread`, `block` or `reps`
     (`collect.py:74-81`) - the load-bearing check, since a different bootstrap in one shard makes
     the combined max meaningless;
   - refuses if any of the 500 shards is missing unless `--allow-missing` (`:104-109,193-202`).
     The comment explains this is a *scope* defect (valid test of a smaller family), not a
     downward bias (`:94-103`); commit `a782478`. **Leftover inconsistency:** the CLI help at
     `collect.py:170-171` and the print at `:203-204` still say the verdict "is biased DOWN"
     - the comment above them says that is wrong.
   - reduces: `max` of observed t across shards, element-wise `max` of boot vectors, p = share of
     the null max above the observed (`:113-123`, `stats.py:169-171`);
   - second gate (`:252-277`), distribution of all cells (`:279-293`), V0's rank (`:295-311`),
     StepM only if SPA rejects (`:313-336`), decision text (`:338-365`).
8. The run: array job **10374118**, 500 tasks, 2026-09-15/16, ~700 CPU-hours
   (`momo_sweep_results.md:3-4,151-153`). An earlier array **10373735** "failed 15 tasks in a row"
   on `srun: CPU binding outside of job step allocation` because it was submitted from inside an
   `srun --pty` session; fix: drop `srun`, `exec` the interpreter (`submit_sweep.sbatch:164-195`).
   Other sbatch fixes recorded in the header: a conda env named `vrp` that does not exist on
   Discovery (`:84-96`); do not `conda create ... numpy` on a login node (`:119-127`); absolute
   `--output` path so logs don't land in the repo (`:10-14`); `--time` generous because a
   wall-clock-killed task leaves a truncated shard that looks fine (`:46-51`).
   Sizing: laptop benchmark 0.151 ms per cell-run implied ~177 CPU-h; measured on a compute node
   ~0.51 ms per cell-run, so ~700 CPU-h (`:36-47`). `grid.py:154` still prints a stale
   "3,731 cells/CPU-hour" estimate (~756 CPU-h) from an even earlier benchmark; and
   `stats.py:35-36` still quotes "3,815 sessions x 1,038,800 cells" - old grid/session counts
   (current: 1,497 sessions, 2,822,400 cells). Cosmetic, but they are exactly the "stale number
   in a comment" shape.

### A3. The audit step (preconditions 3 and 4) - `momo_sweep/audit.py`

Purpose: stop a finished 700-CPU-hour job being submitted if the harness cheats. Two tests aimed
at two specific past failures (`audit.py:11-35`).

- **Lookahead test.** Shift the signal (`Prep.close`) one bar into the *future*
  (`shift_signal`, `audit.py:104-111`; normaliser left honest). If peeking does not make results
  materially better, the honest path already contains the information and every number is void.
- **Direction placebo.** Randomise the traded sign *after* every gate (`flip` hook,
  `engine.py:252`) so the placebo takes the same trades at the same times; a coin flip must earn
  ~0 (|z| <= 3 across 4 draws) or the fill/exit convention manufactures profit
  (`audit.py:29-34,332-361`).
- **Power check ("oracle") 3b.** Hand each cell the sign of its own trade's outcome
  (`force_dir`, `engine.py:260-261`; `audit.py:150-174`); an honest engine must profit on >=95% of
  trading cells, otherwise the audit is blind (`audit.py:266-304`).

The one-bar lookahead criterion **failed as first written** and was amended 2026-09-15 before any
shard ran (`momo_sweep_precondition3.md`; `momo_sweep_preregistration.md:226-251`). First rule
(`peek > 3x |honest|`) was arithmetically invalid for a negative baseline
(`audit.py:48-62`). Honest vs peek: V0 -0.7118 -> +2.2013 (paired +2.913); only 48% of sampled
cells improved because most cells trade too thinly for one bar to register; improvement is
monotone in trade count (table in `momo_sweep_precondition3.md:46-55`). A first "oracle" shifted
both ends of the momentum window and scored 53% - "not an oracle" (`:88-93`). Fixed with
`force_dir`; oracle then 100% (113/113). **Amended gate** = V0 anchor improves under a 1-bar peek
AND oracle >= 95% (`audit.py:40-46,306-329`). Placebo z = -0.10 across 4 draws
(`momo_sweep_precondition3.md:107`). The author explicitly declines to close the criterion alone
(`:128-130`): "a pre-condition rewritten by the person who wants it to pass is the exact pattern a
pre-registration exists to prevent"; the defence is that each defect was demonstrable from the
algebra before the result was seen.

### A4. The three earlier steps that led to the sweep

1. **24-setup sweep** (`research/setup_sweep_preregistration.md`, results
   `setup_sweep_results.md`, run 2026-08-20): tested whether any of 23 setups separates *his own*
   trades (418-trade frame, 349 trades / 102 dates in discovery). Expected false positives at raw
   alpha 0.05 = 1.2; observed 0 at raw p < 0.05; 0 survive Holm or BH. Power: can only see effects
   above ~22-24 percentage points of return (`setup_sweep_results.md:64-68`). Explicitly *not* a
   standalone backtest (`setup_sweep_results.md:74-77`; `setup_sweep_preregistration.md:103-107`).
2. **Setup cross-check** (`setup_crosscheck_*`, 2026-08-20): do his entries coincide with 10
   standard setups? Coverage 91.4% match >=1 setup -> "SATURATED" so a match carries no
   information; 0 of 7 testable setups clear Holm (`setup_crosscheck_results.md:9-28,57`). Fade
   1.5%, flag 0.7%, late-day breakout 0% (he chases, never waits).
3. **Setup standalone** (`setup_standalone_preregistration.md`): ten setups as standalone rules on
   QQQ+SPY 2020-2026, fixed 25-min exit, Holm over 10, advance only if mean >= 0.05 sigma and
   Holm p < 0.05. File header says **"PRE-REGISTERED, NOT YET RUN"** (`:3`). I found no
   `setup_standalone_results.md` and no other file referencing it (grep returned nothing); the
   momo prereg says "That project was never authorised since. This is it."
   (`momo_sweep_preregistration.md:13-14`). **(inferred)** it was superseded by the sweep and
   never executed. Owner should confirm.
4. **momo_v2.py** (`trade_analysis/backtesting/momo_v2.py`, pre-registered 2026-09-04, run
   2026-09-13): three variants - V0 as frozen, V1 + `px_vs_ema9 >= 0.71 ATR`, V2 with 1-minute DMI.
   Falsifier: beat +$1.00/trade and clear Holm at 0.05 (`:23-25`). Result V0 +0.03 (n=8,598), V1
   -0.03 (8,104), V2 -0.09 (13,255); all p_Holm = 1.0 (`momo_v2_results.md:20-28`). Making the
   rule more faithful to his behaviour made it *worse* in both cases (`:47-51`). The
   pre-registration "sat unexecuted for nine days", recorded openly (`:7-9`). Execution convention
   (shared with the sweep): signal from a closed 5-minute bar, fill at next 1-minute open, exit at
   open 25 minutes later, $10,000 notional (`momo_v2.py:34-35,115-120`).

### A5. Part-A traps and contradictions found

- **CLAUDE.md points to the wrong file.** The "WHAT IS SETTLED" table row "MOMO_CHASE form,
  2,822,400 cells, p=0.8227" cites `setup_sweep_results.md`. That file is the 24-setup sweep (23
  setups, no p=0.8227). The sweep result is `research/momo_sweep_results.md`.
- **CPU-hours** quoted three ways: ~700 (cluster-measured; sbatch header, results doc), ~177
  (laptop; `momo_sweep_precondition3.md:142-145`), ~756 (`grid.py:154`, stale). The cluster number
  is the real one.
- **Sharpe "within 2%"**: arithmetic gives +2.3% (1.688 vs 1.65). Immaterial.
- **V0 on the sweep window is negative gross** (-$0.7118/session gross, `momo_sweep_precondition3.md:21`;
  -$1.9352 net, `momo_sweep_results.md:112`), whereas `--validate` over QQQ+SPY all years is
  +$0.0339/trade gross. Different samples; both stated.
- Cell index of V0 (#928,056) is quoted in the docs; I did not recompute it.

---------------------------------------------------------------------------------------------

## PART B - DATA ARCHIVE, VENDORS, LOOKAHEAD, MANIFEST, STRAY CODE

(Supporting detail for sections 1-10 above; each item is sourced. Q3 = B1, Q4 = B2,
Q5 = B3 + B4, Q6 = B5, Q7 = B6.)

### B1. The lookahead story (question 3)

There are TWO separate incidents that the repo's headline sentence merges. CLAUDE.md:81 says
"A 1-minute lookahead once supplied 88-95.7% of a measured edge"; `research/PROJECT_REPORT.md:90`
and `huggingface_space/tools/build_research_snapshot.py:178` attach the same 88-95.7% to the
5-minute label bug. Reading the primary documents, the two numbers belong to different bugs:

| | incident 1: 5-minute label | incident 2: 1-minute quote stamp |
|---|---|---|
| found | 2026-08-16 (memory note `underlying-orb-decade-test`, `0dte-real-data-findings`) | 2026-08-28 walk-forward (`research/walkforward_results.md:3,56`) |
| where | `trade_analysis/backtesting/options_premium_backtest.py:152-159` `find_entries`: loops 5-minute bars, tests `bar['Close']`, records the bar's LABEL (`ts`) as `entry_time` (:155). `pandas.resample("5min")` labels left, so the bar labelled 09:30 spans [09:30,09:35) and its close is unknowable until 09:35. `spy_qqq_0dte_quotes_backtest.py` then filled at `timestamp >= entry_time` (memory note cites line 116). | `trade_analysis/live_lab/walkforward.py` first option-pricing pass: used the option quote stamped T, the START of the 1-minute bar whose close triggers the signal (`walkforward_results.md:58-61`) |
| size | QQQ 25%-target option: +6.44% as published -> -8.34% honest (bias +14.79 pts, paired t +6.67); SPY -0.26% -> -11.33% (bias +11.08, t +5.77). Underlying QQQ: t=3.55 -> t=0.44; the free 5 minutes carried +0.0875% of +0.0992% = **88%** of the "signal" (`decision_time_audit.py:13-19`; memory) | IMB option return +16.46% (p=0.0003) -> +0.70% (p=0.79); (16.46-0.70)/16.46 = **95.7%** of the edge (`walkforward_results.md:63-71`) |
| consequence | "two whole result sets void" (PROJECT_REPORT:90): the 2021-2026 "QQQ works, SPY does not" 0DTE study and the 2016-2026 underlying decade test ("the signal is real, t=3.55"), which had partly justified the $80 Stock STANDARD purchase (memory `underlying-orb-decade-test`) | IntradayMomentumBoundary, the only setup to clear Holm in stage 1, dies under honest timing; live runner unaffected because it waits `bar_close + 1500 ms` (`walkforward_results.md:91-93`) |

Mechanism of incident 1, in plain words: the backtest bought the option five minutes before the
signal that triggered it existed, and during those five minutes the stock drifts ~0.09% in the
signal's direction - about the 0.2% move that IS a +25% option gain. So it was "handed the profit
target before the position opened" (`decision_time_audit.py:21-24`). The entry ask was unchanged
($1.249 vs $1.238, memory), so it was free drift, not a pricing artefact.

**The bug was NOT fixed in the source.** `options_premium_backtest.py:152-159` still records the
label. The fix is a wrapper that lives only in `decision_time_audit.py:81-100` (`honest_entries`
moves `entry_time` to label + 5 minutes and re-slices `today_bars`). That follows the repo rule to
isolate experimental code rather than edit proven code in place (memory
`feedback-isolate-experimental-strategy-code`). Consequence: anything that still imports
`find_entries` directly is still contaminated. `decision_time_audit.py:71-78` lists six modules
that inherit the bug; only two (`spy_qqq_0dte_quotes_backtest`, `underlying_orb_longrun`) were
re-run. Four ("not re-run", read yfinance/Massive): `spy_qqq_0dte_real_backtest`,
`retest_vs_gapandgo_backtest`, `options_premium_backtest_scalp`, `options_premium_backtest_confluence`.
`underlying_orb_longrun.py:24-26` (docstring) still print the pre-correction numbers
("QQQ needs 75.3% (gets 80.3%)") with no VOID banner in the file (I grepped; no "void" string).

**What `decision_time_audit.py` checks** (read in full): for each entry it prices the option twice,
at the label (`asis_*`) and at the bar close (`hon_*`), paired trade-for-trade
(:113-141); for the underlying it measures forward return and MFE from label vs close (:144-176);
two readings of the ambiguous 09:45 bar - `strict` (decide by 09:45, only 09:30/09:35/09:40
bars qualify, :91-92) and `--loose` (:232-233); it prints the share of the as-is result that was
"drift handed over" (:213-217) and flags any cell below the 75.3% breakeven win rate (:195).
`BREAKEVEN_WIN_RATE = 75.3` (:66) applies to the 25% target ONLY - the 2026-08-18 audit
correction in memory `0dte-real-data-findings` says the hold-to-close cell breaks even at 35.9%.
Locked honest baseline (memory): all 8 option cells below breakeven, 7 of 8 significantly
negative; underlying QQQ t=0.47, SPY t=-0.09.

**A third, smaller fabrication found in the same audit:** ~7% of entries were post-holiday
fabrications - `find_entries` inherits `yesterday_high = 0` from zero-filled closed days (105 QQQ,
98 SPY, all UP, all at the 09:30 label), SPY t25 -11.67% -> -9.89% after removal (memory
`0dte-real-data-findings`, audit correction 3). Related silent data defect: 40 of 256 SPY sessions
vanished from a sample because a transport exception was swallowed as "no data" and the run
reported n=216 (`thetadata_client.py:124-130`, `theta_v3.py:28-31`).

**How the lesson was institutionalised** (every later experiment carries a one-bar-shift test):
`momo_sweep/audit.py` (A3 above), `hpc/har_baseline.py --audit` (shifts every feature one minute
forward; honest QLIKE 0.541358 vs audit 0.530811, "Audit better by 1.95% -> PASS",
`vrp_baseline_results.md:44-58`), and the rule "`_features_at` never receives a frame that extends
past t" (`hpc/build_vrp_dataset.py:5-11`). The live lab's own decision-on-disk-before-price rule
is the same lesson applied at runtime.

### B2. The data inventory (question 4)

Counts marked [LS] I measured by listing the directory read-only on 2026-10-13 (file counts only,
no file content opened). Everything else is quoted from code/docs.

| vendor / product | symbols | date range | resolution | where stored | how verified | status today |
|---|---|---|---|---|---|---|
| ThetaData, Options, `option.quote.1m.0dte` (bulk layer, `layers.py:150-154`) | **SPY only** | first file SPY_2020-01-03, last SPY_2024-08-07 [LS]; 2020:159, 2021:39, 2022:170, 2023:250, 2024:151 = **769** sessions [LS] | 1-minute NBBO bid/ask for the whole 0DTE chain | `C:\Users\chaitanyakharche\Desktop\data\raw\option_quote_1m_0dte\SPY\<year>\SPY_<date>.csv.gz`, 682,604 kB [LS], outside the repo (`bulk_download/config.py:11-17`) | 769 = sum of per-year file counts [LS] and matches `vrp_preregistration.md:34-37`; no 2025/2026 directory exists [LS]; an empty 2019 directory exists [LS] (fetch_csv makes the directory before requesting, `theta_v3.py:109`) | exists, irreplaceable per docs; the 2024 block (Jan 2 - Aug 7, 151 sessions) is the pre-registered held-out block "touched exactly once" (`vrp_preregistration.md:48-54`) |
| ThetaData, Stock, `stock.ohlc.1m` | 104 symbols [LS] (SPY, QQQ + QQQ constituents, `universe.py`) | to 2016 (`thetadata_client.py:69`) | 1-minute OHLCV+vwap | `raw\stock_ohlc_1m`, 12,539 files, 1.41 GB [LS] | **SPY is missing 2024-01..2024-07** and cannot be refetched ("NOT ENTITLED", `vrp_baseline_results.md:110-113`) | usable except that hole |
| ThetaData, Stock, `stock.ohlc.5m` | 104 [LS] | "complete 2016-2026" (`build_vrp_dataset.py:43`) | 5-minute | `raw\stock_ohlc_5m`, 13,293 files, 395 MB [LS] | SPY_2020-02 native file held 3 days vs 22 for March; repaired by falling back per day to the 1m archive; equivalence checked on 2020-02-04, max abs diff 0.000000 over 79 bars (`vrp_baseline_results.md:123-131`) | used by the VRP work |
| ThetaData, Stock, `stock.quote.1m` | 104 [LS] | SPY 2017-2026 (`hpc/README.md:38`) | 1-minute NBBO | `raw\stock_quote_1m`, 10,656 files, 760 MB [LS] | not independently verified | used for spread/fill realism |
| ThetaData, Stock, `stock.eod` | 104 [LS] | whole history | daily + closing NBBO | `raw\stock_eod`, 10,656 files, 47 MB [LS] | - | exists |
| ThetaData, Stock, second layers `stock.ohlc.1s`, `stock.quote.1s` | AAPL, KHC, SPY only [LS] | 15 files each [LS] | 1-second | `raw\stock_ohlc_1s`, `raw\stock_quote_1s` | - | tiny sample, not a dataset |
| ThetaData, Stock, `stock.trade_quote.tick` | SPY only [LS] | **99 sessions** [LS] (`layers.py:135` says "measured: 99 SPY sessions") | every print with prevailing NBBO | `raw\stock_trade_quote_tick`, 540,720 kB [LS] | - | sample only |
| ThetaData reference | - | pulled 2026-08-20 [LS timestamps] | - | `Desktop\data\reference\` (available_dates, expirations, holidays, universe.json, qqq_constituents.csv, entitlements.json, OpenAPI yaml) | - | `qqq_constituents.csv` is TODAY's membership - survivorship trap flagged in `universe.py:11-17` |
| ThetaData manifest | - | `Desktop\data\state\manifest.sqlite` 11 MB, last modified Aug 21 [LS] | - | - | I did NOT open it (see note) | the only record of what was requested and what came back |
| **Not on disk** [LS: no such directories under `raw`] | QQQ options (any expiry); SPY/any 1DTE or weekly; option EOD; option open interest; option 1-second | - | - | - | - | layers are *defined* (`layers.py`: `option.eod`, `option.open_interest`, `option.quote.1m.week`, `option.quote.1s.0dte`) but there is no raw directory for them, so they were never downloaded |
| ThetaData via `ThetaDataClient` JSON cache | SPY, QQQ options | the 1,465-session 2021-01-01..2026-08-13 study and the 2016-2026 underlying study | 1-minute quotes | `<repo>\.cache\thetadata\*.json.gz` (`thetadata_client.py:52`, `paths.py:12`) | not listed by me | **cache of the VOID studies' inputs**; data fine, analysis void (memory `underlying-orb-decade-test`: "the defect was in the analysis, not the acquisition") |
| Live-lab bar cache | QQQ 2,657 sessions 2016-01-04..2026-08-27; SPY 1,158 sessions 2022-01-03..2026-08-27 (`momo_sweep_preregistration.md:50-53`) | - | 1-minute bars, one pickle per session | `live_lab_data/bars_cache/` (gitignored, `.gitignore:88`; main checkout only - I did not touch it) | `pack.py:29-31`: QQQ has 2,779 files, 122 under 300 bars dropped | feeds `momo_sweep/pack.py` -> `data/momo/*.npz` (~18 MB) |
| Vilkov replication panel (free academic) | SPXW (S&P 500 index 0DTE) | 1,397 sessions 2016-09-02..2024-05-01 (memory; README claim of ~3.5M rows/Jan 2026 is wrong) | 30-minute bars, 10:00-15:30 ET only | `C:\Users\chaitanyakharche\Documents\research_data\0dte-strategies` outside repo | memory: verified by direct inspection; one corrupt session 2020-03-06 (`mid=0`, `IV=0.89`) | licence "academic replication only", never copy into repo; upstream cost-scale bug (spread charged at 1/100) voids its *paper* net numbers but not the parquet |
| Massive (ex-Polygon) free tier | SPY/QQQ stock + option bars | ~2 years back (~2024-08 onward) | 1-minute **traded** bars, no bid/ask | `<repo>\.cache\massive\*.json.gz` (`massive_client.py:47`) | - | superseded; first-generation study only (`spy_qqq_0dte_real_backtest.py:1-48`) |
| yfinance / Alpaca / Finnhub / Reddit | daily bars, 5-min bars, news, reddit | - | - | `trade_analysis/data/`, `local_data/*.json` | - | HF-app support data (see B6) |

Note on what I did NOT read: two read attempts at `Desktop\data\reference\entitlements.json` and a
copy of `manifest.sqlite` were blocked by the permission layer (flagged as PII / credential
handling). I did not try to route around that. Both exist and would answer "what tier did the
probe see on 2026-08-20 and what did every request return" - owner can open them.

### B3. The vendor decision (question 5, vendors and costs)

All from `memory/options-data-vendor-options.md` (survey of 2026-09-11, 31 days old) unless noted;
the memory note itself says some prices could not be verified.

| vendor | considered for | outcome | reason |
|---|---|---|---|
| **ThetaData Options VALUE, $40/mo** | option NBBO history + live chain | **chosen**, bought on or before 2026-08-13 (the real-quote study ran 2021-01-01..2026-08-13, memory `0dte-real-data-findings`; exact purchase date not recorded) | "the run the $40 ThetaData Value subscription was bought for" (`spy_qqq_0dte_quotes_backtest.py:2-4`); 1-minute real bid/ask, whole chain in one call, first access 2020-01-01 per docs table |
| **ThetaData Stock STANDARD, $80/mo** | 1-minute stock bars from 2016, real time | **chosen**, upgraded 2026-08-15 (`thetadata_client.py:3-4,63-71`; ~"$80" from memory) | Stock VALUE is 15-minute delayed and starts 2021-01-01; STANDARD starts 2016. The upgrade was partly justified by the underlying decade test that was later voided |
| ThetaData Index | SPX/VIX intraday | **never bought** | FREE tier only, intraday 403 (`config.py:34-40`, `thetadata_client.py:55-60`); the client hard-refuses `/index/` paths (`:60,110-116`) |
| Massive (ex-Polygon) | first option source | **used on free tier, rejected** | 5 calls/min, ~2 years, **no bid/ask** (NBBO is $199/mo Advanced) (`massive_client.py:1-25`); traded-price fills needed an assumed spread, "the weakest link" |
| Alpaca ($99/mo Algo Trader Plus) | historical option archive | **disqualified for archive** | options history only since Feb 2024, no historical option *quotes* at all, no index data; "mid-only data is worthless" because the project turns on gross +$0.0205 vs net -$0.0171. Strong for the *live* lab. Free tier historical bars are IEX-only (biases realised variance) |
| Databento OPRA `cbbo-1m` | extend archive to 2013 | **proposed, no sign it was bought** | history from 2013-04-01, 1-minute consolidated NBBO, pay-per-GB, $125 free credit; would take 769 sessions to ~3,200 and rebuild SPXW at 1-minute. It would be a *replication*, not a reproduction (different sampling rule). No code in the repo references Databento (I grepped) |
| Cboe DataShop "Option Quotes" | one-off archive purchase | **dark horse, not pursued** | 1-minute NBBO with sizes from Jan 2012, SPX incl., one-off purchase; price not extractable |
| Vilkov replication package | free real SPXW quotes | **used** | free, real bid/ask, but 30-minute bars, no data before 10:00 and ends 2024-05-01 |

Licensing constraint (memory): ThetaData personal licence forbids redistribution/business use and
requires destroying all copies within 30 days of termination; "a public demo arguably reads as
business use, so keeping the Space DERIVED-ONLY is what keeps us inside the cheap tier." OPRA
professional-classification fee $31.50/mo is a +79% hidden cost if triggered.

The memory note's recommended call (2026-09-11): "pay $40 (Options Value) now to unbreak the live
lab, and burn the $125 Databento credit this week to price a 2013-2026 `cbbo-1m` pull." The first
half appears to have happened (see B4); there is no evidence of the second half.

### B4. THE DISCREPANCY: did ThetaData options "lapse", and what does the live feed need? (question 5)

What the files say, in time order:

1. **2026-08-20** - `bulk_download/config.py:34-40`: probed tiers "Stock STANDARD, Option VALUE,
   Index FREE". The 769-session SPY archive was built under this.
2. **2026-09-08** - subscription dropped. `autostart.py:414-419`: "On 2026-09-08 the ThetaData
   subscription dropped to Stock: STANDARD / Options: FREE; four 403s from the options chain
   aborted BOTH arms". `ledger.py:15-16` and `research/replay_22_sessions.md:50-56` record options
   "blocked" on 09-08, 09-09, 09-10, 09-11 (4 sessions).
3. **2026-09-09** - `build_vrp_dataset.py:40-41` and `vrp_baseline_results.md:110-113`: refetching
   SPY `stock.ohlc.1m` returns NOT ENTITLED "despite the terminal reporting Stock: STANDARD".
4. **2026-09-11** - vendor survey recommends paying $40 for Options Value to "unbreak the live lab".
5. **2026-09-14** - `live_lab/runner.py:53-57` (comment, written when the stale-bar guard was
   back-ported 2026-09-24): "the lapsed options entitlement ... **the entitlement came back on
   2026-09-14** and this arm resumed". `replay_22_sessions.md:57-58` shows options trades live
   from 09-14 (+857).
6. **2026-09-11 to 09-24** - but `scripts/tidy.py:16-17` (09-11), `breakout_options_preregistration.md:35`
   and `live_lab/breakout_sweep.py:16` (both 2026-09-18/19), `research/PROGRAMME_REVIEW_2026-09-24.md:55-56,350`
   and `CLAUDE.md:73` all still say the archive "cannot be rebuilt / nothing new from ThetaData:
   Options: FREE since 2026-09-08" - as a permanent present-tense fact.
7. **2026-09-24 to 09-29** - `live_lab/backfill.py` (commit e423c3a, 09-24; run 09-29) replays all
   22 sessions 2026-08-28..09-29 using `/option/history/quote` at 1-minute for the full 0DTE chain
   (`backfill.py:200`; strike list `:192`) and `/stock/history/quote` (`:159`). The replay table gives clean
   options results **for the four "blocked" days 09-08..09-11** (-374, -727, -1,177, -1,474; total
   -$3,752, `replay_22_sessions.md:53-56,88-89`). That could only be pulled if the history-quote
   entitlement for those dates was working after the renewal. (Inferred - I could not see the cache
   provenance of those chain files.)

**What the LIVE feed needs.** `live_lab/feed.py:7-12` (and :474,:492,:518): `/v3/option/snapshot/quote`
(real-time whole chain), `/v3/option/list/expirations`, `/v3/option/snapshot/open_interest`, plus
stock `snapshot/quote` and `history/ohlc`. `preflight.py:177-191`'s options check fails with
"chain snapshot failed -- entitlement? (403)" if the snapshot 403s. Per memory
`options-data-vendor-options` Options FREE has "no real-time Quote endpoint - this is exactly the
403 on `/option/snapshot/quote`"; Options VALUE has real-time snapshot + historical quote +
bulk quote. So the live options arm recording since 09-14 implies **Options is at VALUE or higher**.
`feed.py:13-14`: greeks/IV endpoints "do NOT exist on this subscription tier (all 404)", which fits
VALUE (greeks need STANDARD per `layers.py:176-181`).

**What the HISTORICAL archive download needs.** `layers.py:150-167`: `/option/history/quote` at 1m
(and even 1s) is served on Value tier ("Value tier DOES serve this - only /option/history/trade
needs Standard", `layers.py:166-167`). So the same entitlement that the live feed needs also covers
the historical quote endpoint the archive was built from.

**Conclusion (evidence-weighted, not proven):** CLAUDE.md's "Anything new from ThetaData. Lapsed to
Options: FREE on 2026-09-08" was true for 09-08..09-13 and is **stale since 2026-09-14**. The
owner very probably re-bought Options VALUE (no document records the purchase - **reason not
recorded**). That means the claims "cannot be measured: QQQ options any expiry / anything new from
ThetaData" are **not established**. They rest on an entitlement state that changed six days after
it was observed and was never re-probed for the archive.

**What is still unknown** (needs the owner or one read-only probe):
- the tier actually active today (VALUE vs STANDARD) and whether billing is current;
- whether the historical floor is **2020-01-01** (docs table) or a **rolling ~4 years** (marketing)
  - the vendor note flagged this as "THE ONE QUESTION TO EMAIL SUPPORT BEFORE PAYING" and no
  answer is recorded. If rolling, 2020-2022 history (the bulk of the SPY archive, and all of
  QQQ's early-weekly era) would not be re-downloadable even now;
- why `stock.ohlc.1m` was NOT ENTITLED for SPY on 09-08/09 while Stock was reported STANDARD, and
  whether that is fixed;
- whether `probe_entitlements.py` was re-run after 09-14 (it writes `reference/entitlements.json`
  and the manifest; the file I could see is dated Aug 20 [LS]). The way to settle this is to run
  `python -m trade_analysis.bulk_download.probe_entitlements --layers option.quote.1m` (one
  status-only probe set; reads kilobytes) - **I did not run it** (rules).
- whether QQQ 0DTE history is worth pulling at all: `walkforward_results.md:40` says QQQ daily
  expiries began in 2023 and `spy_qqq_0dte_quotes_backtest.py:32-37` says Tue/Thu expiries began 2022
  (QQQ was Friday-only for 0DTE in 2020 per memory).

### B5. The manifest / coverage mechanism (question 6)

**Bulk download completeness** (`bulk_download/`, built 2026-08-20; first committed 2026-08-31):
1. `universe.py` verifies each symbol against the live `/stock/list/symbols` and
   `/option/list/symbols` feeds up front and records misses (`:82-106`).
2. `download_reference.py` pulls `/stock/list/dates/{trade,quote}` per symbol = the exact sessions
   the feed holds, so task lists are clipped to reality (`download.py:73-77,325-332`).
3. `trading_days.py` supplies the calendar (pandas_market_calendars if installed, else a hand table
   2004-2026 with Sandy, 9/11, Carter 2025 etc.). Its stated purpose is to keep "missing"
   meaningful: a holiday returns NO_DATA, "indistinguishable from a genuine coverage gap once it is
   written to the manifest" (`trading_days.py:5-9`).
4. `Layer` objects (`layers.py`) declare chunking (monthly >=1m intervals, daily for sub-minute,
   because the gateway enforces "no more than 1 month" and "at least 1 minute" bulk limits,
   `config.py:42-53`).
5. `theta_v3.fetch_csv` classifies every response into **four** states - OK / NO_DATA (472, 404, or
   200 with no rows) / NOT_ENTITLED (403) / ERROR (5xx, timeouts after 5 attempts) - and never
   conflates them (`theta_v3.py:14-30,98-168`). NOT_ENTITLED must never be recorded as NO_DATA "or
   the archive will look complete when it has a tier-shaped hole in it" (`:21-24`). A 404 with HTML
   is a wrong path, returned as ERROR (`:117-122`). Writes go to `.tmp` and rename on success so an
   interrupted run cannot leave a half file (`:101-110,151`).
6. `manifest.py`: SQLite `downloads(layer, symbol, key, status, rows, bytes, path, detail)` with
   PK (layer, symbol, key). Resumption is driven by the manifest, NOT file existence, because empty
   results leave no file (`:1-13`). `done_keys()` treats OK, NO_DATA, NOT_ENTITLED as settled and
   ERROR as retryable (`:102-115`). `status.py` reports by layer/status and "TRANSIENT FAILURES".
7. `probe_entitlements.py` binary-searches the true first available date per layer (~12 probes) and
   writes `reference/entitlements.json` + a manifest table; downloaders start from the *measured*
   floor, never a guess (`probe_entitlements.py:1-22,111-162`; `download.py:343-354`). Layers that
   403 on a known-good date (SPY 2024-08-07) are marked unavailable.
8. `download.py`: newest-first (`:113-115`), disk floor of 15 GB (`:47`), concurrency 4 stock / 2
   option (`config.py:59`), aborts a layer on first NOT_ENTITLED (`:232-239`), `--dry-run` for
   request counts and size estimate.
9. `to_parquet.py` converts raw csv.gz to typed zstd parquet, localising naive timestamps to
   America/New_York explicitly (`:11-16,63-73`); raw stays "exactly as the API returned it".

**Gap in this mechanism worth knowing:** the manifest proves each *request* settled; it does not
prove the archive covers every session of the calendar. NO_DATA is terminal, so a session the feed
transiently answered 472 for would never be retried. The only completeness check is after-the-fact
counting (`ls` per year) - which is how the SPY 2021 count of 39 stands unexplained.

**The "coverage tool reported 100% with six sessions missing" incident is NOT in the bulk
download.** It is the **live lab ledger** (`trade_analysis/live_lab/ledger.py:324-361`): `coverage()`
built its day set from ledger records alone, so a day with no record was missing from both
numerator and denominator. On **2026-09-16** it printed `usable 7/7 = 100.0%` with six consecutive
sessions absent (`ledger.py:332-335`). Fixed (commits `9f1691f`, `df25c2f` per
`PROGRAMME_REVIEW_2026-09-24.md:101-107`) by walking the exchange calendar (`trading_days.py`'s
`is_trading_day`, imported at `ledger.py:59`) from the lab's first evidence to the last closed
session, tagging unrecorded days `UNRECORDED`; the same period then read 7/13 = 53.8%. The fix
reuses this slice's `trading_days.py`. Root cause of the six missing sessions: committing the
record by hand; fixed by `archive.py` (auto-commit at 16:00). Slice 01/02 own the detail.

**The momo sweep has its own completeness guards** (A2 above): packed-session filter reproduced,
V0 must give n=8,598, `session_hash` must match in every shard, `--expect 500` and refusal on
missing shards, grid size cross-checked in the sbatch.

### B6. The stray non-equity and support code (question 7)

| file | what it is | imported by anything? |
|---|---|---|
| `data_sources/crypto_history.py` (422 lines) | `CryptoEventAnalyzer`: crypto event-study on 730-day history, writes under `data/crypto_historical_data` | **No.** My grep for imports across `*.py/*.md/*.sbatch` found none |
| `data_sources/crypto_live_polling.py` (297) | `CryptoMarketPredictor`: Alpaca crypto websocket live predictor (BTC/ETH/SOL) | **No** |
| `data_sources/forex_signal_generator.py` (221) | `ForexSignalGenerator`: EUR/USD, GBP/JPY, USD/JPY RSI/momentum signals via Alpaca forex rates; log to `forex_signals.log` | **No** |
| `data_sources/download_5min_alpaca.py` (144) | pulls 5-minute Alpaca bars for 7 symbols from 2024-01-01; **exits at import if ALPACA keys unset** (`:13-15`) | **No** |
| `data_sources/download_daily_yfinance.py` (196) | 20-year daily yfinance downloader with log file | **No** |
| `data_sources/collect_alt_data.py` (62) | fetches Finnhub news + Reddit and writes `local_data/<SYM>_external_data.json` | Only by the README (`README.md:188-190`); it is the producer of `local_data/` |
| `data_sources/unified_data_provider.py` (135) | yfinance multi-timeframe + local-file-first news/reddit; `get_alternative_data` returns **hard-coded** `put_call_ratio: 0.85`, `iv_rank: 45.5` (and `20.0/"Error"/1.0/50.0` on failure) (`:128-136`) | Only by `collect_alt_data.py` |
| `local_data/*_external_data.json` (9 files, Sep 14 2026) | Finnhub news (`news_data`: headline, summary, source, datetime epoch) per symbol - AAPL, AMD, AMZN, META, MSFT, NVDA, QQQ, SPY, TSLA; reddit data key also supported (`unified_data_provider.py:99-104`). A byte-identical copy lives in `huggingface_space/local_data/` | Read by the unified provider and shipped to the HF Space (`huggingface_space/README.md:478`). **Not used by any research result** - the news-tagging idea was rejected 2026-08-15 (memory `news-event-study-assessment`: flag is constant, vendor sentiment not backfilled, headlines reverse-caused, LLM scoring leaks hindsight) |
| `trade_analysis/config.py` | loads `.env`; **raises ImportError at import if `FINNHUB_KEY` or `REDDIT_CLIENT_ID` is unset** (`:12-13`) | Imported by `unified_data_provider.py`; is the "config.py raises on import" trap (memory `python-environment-split`). `trade_analysis/__init__.py` has no imports so the sweep avoids it |

Verdict: the crypto/forex/alpaca/yfinance scripts are **dead code from the earlier "stock
opinion analyzer" app** (README.md structure, line 26 "data_sources/ # Data ingestion"), not part of
the 0DTE research or the live lab. The live lab explicitly imports nothing from `data_sources`
(`live_lab/feed.py:3-4`, `live_lab/__init__.py:11`). Only `thetadata_client.py` (8 backtests,
including `decision_time_audit.py`) and `massive_client.py` (2 backtests) are live dependencies of
research scripts. Caveat: grep cannot see dynamic or string-based loading; "dead" = no static import.

### B7. Other scripts in scope (short)

- `backtesting/spread_survey.py` + `research/spread_survey_results.md` (2026-09-02): measured median
  round-trip spread in bp for 51 candidate names (5 sessions 2026-08-27..09-02, 09:45-15:45 only)
  against the IMB shares edge of +3.34 bp gross. 16 of 51 untradeable (COIN 9.56 bp worst);
  tightest SPY 0.26, IWM 0.34, QQQ 0.41, DIA 0.56. Picked the 15-name shares universe on spread
  alone. Contains a self-corrected false claim (spread vs QQQ-correlation -0.13, not the
  trade-off asserted) and the corrected hash-fork claim (`:115-127`). Scratch path hard-coded to a
  per-session temp folder (`spread_survey.py:~44-46`). **(inferred)** not reusable as-is.
- `spy_qqq_0dte_quotes_backtest.py`: real-NBBO version (buy ask, sell bid), `STRIKE_RANGE=15`,
  skips Tue/Thu before 2022 as legitimate `chain_empty`; **inherits the 5-minute bug through
  `find_entries`**. `spy_qqq_0dte_real_backtest.py`: Massive traded-price version with an assumed
  spread; same inheritance, "not re-run". `vilkov_0dte_conditional_backtest.py`: SPXW panel, entry
  forced to 10:00 ET, shift(1) on prior ATR (`:152`).
