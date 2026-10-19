# 09 - Legacy code and machine inventory

STATUS: COMPLETE. Parts (a) and (b) and sections 1-14 are written. Items marked "inferred" were not read directly.
(Earlier cut-off notes removed; nothing else changed.)

All `path:line` references are inside the worktree
`C:\Users\chaitanyakharche\Documents\stock\.claude\worktrees\blissful-kalam-999916` unless
stated. "Inferred" means I did not read it directly.

---

## 1. In one paragraph

Before the live lab existed, this repo was a personal "stock prediction + trading" codebase:
a pile of Jupyter notebooks (LightGBM and PatchTST price-direction models), a FastAPI
service that asked GPT-4o to guess tomorrow's opening gap (`app1.py`), a reconstructed
neural network (TFT) trained on the university cluster, and a family of Alpaca-based
scripts for a SPY/QQQ "retest breakout" day-trade. It was public on Hugging Face as
"ProfitBook". Almost none of that code is still used. The live lab (`trade_analysis/live_lab/`)
imports none of it except a trading-calendar helper; the legacy trading scripts contain
no working entry-order code, the lookahead bug that voided the early backtests is still
sitting un-fixed in `find_entries`, and the only legacy pieces that other things still depend
on are `indicators_pandas.py` (CI-tested, copied into the Space) and `gamma_exposure.py`'s
Black-Scholes helpers (used by research backtests).

## 2. The goal it serves, and how it changed

Evidence for the original goal:

- The first commit in git (`e17df27`, 2026-07-30, "Checkpoint before dead-code cleanup
  sweep") already holds 93 files / 61,462 lines: eight `stock3.x.ipynb` notebooks, a flat
  `trade_analysis/` with `live_deploy.py`, `live_deploy_v2.py`, `livebreakout.py`,
  `gap_predictor.py`, `momentum_trading_engine.py`, `prop_integration.py`,
  `statistical_arbitrage.py`, `train_tft.py`, and the 19 MB of `trained_models/`
  (`git show --stat e17df27`). So the pre-July-30 history is not in this repo; the
  project is older than its git history. (`git rev-list --count HEAD` = 156 commits.)
- `README.md:5-8`: "Personal research + live-trading codebase for a 0DTE SPY/QQQ
  retest-breakout options strategy, with supporting backtests, alt-data collection, a
  dealer gamma-exposure (GEX) signal, and ML research."
- Memory note `profitbook-0dte-options-goal.md` (written 2026-07-26/27): the owner
  "manually trades 0DTE options swings on SPY/QQQ (and TSLA/NVDA) using a 'retest rejection'
  breakout setup"; the scripts were "meant to replicate that manual strategy in code first",
  on the underlying price, then "build an options picker on top so the system can output
  0DTE options recommendations". Deployed publicly as HF Space `ckharche/ProfitBook`
  (that name is from the memory note, not verified in code).
- `gap_predictor.py` in the first commit opens with "Specialized predictor for 3:30-4pm gap
  trading strategy" (`git show e17df27:trade_analysis/gap_predictor.py`, line 1-6) -
  an earlier idea was predicting the next-day opening gap from the last half hour.
  `app1.py` is the same idea as a web service (see section 3.1).

How it changed (all from commit messages and research docs, in order):

1. 2026-07-30 to 08-02: dead-code cleanup, rename into packages (`e04d237` "refactored
   files and script names": `live_deploy.py` -> `live_trading/multi_strategy_trader.py`,
   `live_deploy_v2.py` -> `retest_breakout_websocket.py`, `livebreakout.py` ->
   `swing_breakout_trader.py`, `momentum_trading_engine.py` -> `options_strategy_engine.py`;
   seven of the eight notebooks deleted; log files deleted). README.md:30-60 records the
   old names ("was live_deploy.py", etc.).
2. 2026-08-02 to 08-16: build a synthetic Black-Scholes option backtest, then real
   option quotes (Massive, then ThetaData). Result looked positive ("QQQ works, SPY does
   not") until 2026-08-16 when a 5-minute lookahead in `find_entries` was found: the whole
   edge was the free drift. That number set is VOID (memory `0dte-real-data-findings.md`,
   `underlying-orb-decade-test.md`; code `decision_time_audit.py:1-30`).
3. 2026-08-27 on: the live lab (paper forward test, never places orders), then the HPC
   variance work, then the research record. The legacy `live_trading/` package was
   deliberately left in place "as a record, not reused" (`trade_analysis/live_lab/README.md:132-137`).

## 3. How the legacy code works, end to end

### 3.1 The original app: `app1.py` (473 lines) and `stock3.3.4.ipynb`

`app1.py` is a FastAPI service titled "Stock Gap Predictor API" (the `FastAPI(` call is at
`app1.py:341`, the route at `:354`). Flow of `POST /predict/?symbol=X`:

1. `get_stock_data` - 1 year of daily OHLCV from yfinance (`app1.py:27-49`).
2. `get_news_data` - last 1 day of Finnhub company news (`app1.py:51-83`); needs `FINNHUB_API_KEY`.
3. `get_reddit_data` - PRAW search of r/stocks and r/wallstreetbets (`app1.py:85-135`); needs
   `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`.
4. `calculate_technical_indicators` - ADX, +DMI, -DMI, EMA9 of close and of volume
   (`app1.py:139-160`), via `trade_analysis.indicators_pandas` (line 7; before 2026-09-17 it
   was `pandas_ta`).
5. `analyze_sentiment` - FinBERT (`ProsusAI/finbert`, `app1.py:178`) over the headlines.
6. `predict_gap_with_llm` - builds a text prompt of the indicators and sentiment, sends it to
   OpenAI `gpt-4o` (temperature 0.1, max 10 tokens) and asks for "UP, DOWN, or FLAT"
   (`app1.py:274`; model name at `:313`). Needs `OPENAI_API_KEY`.

It places no orders. It is a direction guesser - exactly the thing the 8 pre-registered
nulls later closed (CLAUDE.md "Direction prediction ... closed").

It cannot start as written: `uvicorn.run("app:app", ...)` (`app1.py:472`) names a module
`app`, but the file is `app1.py`; and the root `Dockerfile` (below) points at yet another
module that does not exist. README.md:150-153 calls the Dockerfile "dead".

`stock3.3.4.ipynb` (279 KB, `nbformat 4`, kernel `stock_py310`): the JSON has **2 cells, 0
markdown cells**. Cell 0 is one 1,706-line code cell; cell 1 is empty. Its imports
(`pandas_ta`, `pywt`, `antropy`, `lightgbm`, `sklearn`, `transformers.PatchTSTConfig`) and
function names show what it is: a daily-bar direction classifier for MSFT/AAPL/GOOGL
(README.md:232-241) - fetch from Twelve Data (`fetch_twelve_data`), add technical, wavelet,
entropy and "regime" features, define a balanced 5-day target
(`balanced_target_definition`), a causal-structure step (`discover_causal_structure`), an
Optuna-tuned LightGBM (`optimize_lgbm_hyperparameters`, `train_lightgbm_model`), an optional
ONNX export, and a PatchTST foundation-model forecast (`run_patchtst_foundation_forecast`).
Entry point `run_full_workflow` then "Example Usage and Main Execution". README.md:233-239
says it is "the single best-working iteration of a larger family ... all superseded/removed"
(the other seven were deleted in `e04d237`) and that it is "exploratory, not a production
signal source" because direction prediction has a "~50% ceiling". It still imports `pandas_ta`
(deleted from PyPI), so it will not run on a fresh install.

`Dockerfile` (7 lines): `COPY app app` then `uvicorn app.api:app` - there is no `app/`
directory in the repo (`ls` of the worktree root). Dead. `environment.yml` (7 lines): conda env
`trade-venv`, python 3.11, `prefix: /home/kharche.c/.conda/envs/trade-venv` - the Discovery
cluster path, not this laptop. `conda_requirements.txt` (135 lines) is its package list
(inferred, not read in full). `requirements.txt` (111 lines) is a `pip freeze` of the
laptop; unusable on a clean machine (CLAUDE.md OPERATIONAL; `requirements-dev.txt:1-25`).

### 3.2 The live-trading scripts (`trade_analysis/live_trading/`, 2,892 lines)

| file | lines | what it is | places real orders? |
|---|---|---|---|
| `multi_strategy_trader.py` | 864 | SPY/QQQ polling trader, 3 strategies (hourly retest, ORB reversal, false-breakout fade) | **Entries and exits: no.** Startup cleanup: yes (see 4.3) |
| `retest_breakout_websocket.py` | 899 | Alpaca SIP websocket version, SPY/QQQ/TSLA | **No order call anywhere** |
| `swing_breakout_trader.py` | 525 | One-symbol class `LiveBreakoutTrader`, takes an injected broker client | Yes in the code, but nothing constructs a client |
| `options_strategy_engine.py` | 604 | `MomentumTradingEngine` / `IntegratedMomentumEngine`: scores momentum, outputs a text options plan | No broker at all |

Details with evidence:

- `multi_strategy_trader.py:19-24` reads `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` at import
  time and `sys.exit(1)` if missing. `:28` builds `TradingClient(API_KEY, SECRET_KEY, paper=paper)`.
  `:861` `main()` hard-codes `MultiStrategyTrader(paper=True)`. `place_trade` (`:406-452`)
  only writes to `self.positions` and the log, and ends with `# TODO: Add actual Alpaca order
  execution here` (`:449`). `check_exits` ends with `# TODO: Add actual Alpaca exit order
  here` (`:504`). `liquidate_all_positions` ends with `# TODO: Add actual Alpaca market
  order to close position` (`:704`). The state is in memory only.
- `retest_breakout_websocket.py:38` builds the same `TradingClient(..., paper=paper)`;
  `:888` `RetestBreakoutSystemV2(paper=True)`. `place_entry` (`:671-731`) has docstring
  "Log entry" and only fills `self.positions`. A grep for `submit_order|place_order|
  close_position|create_order|OrderRequest` over this file returns nothing. The websocket
  (`:801`) is the Alpaca **market data** stream; its auth message sends the API key
  (`:810-815`).
- `swing_breakout_trader.py:24` takes `alpaca_client` as a constructor argument; it
  calls `self.client.submit_order(symbol=, qty=, side="buy", type="market", time_in_force="day")`
  at `:345-351` (entry) and `:397-403` (exit, also called at 16:00 `:461`). The `__main__`
  block (`:499-525`) only prints a banner; the lines that would construct a client and run
  it are **commented out** (`:518-525`) and use `paper=True`. Nothing in the repo imports
  `LiveBreakoutTrader` (grep, section 4.2).
- `options_strategy_engine.py` imports torch (`:3-4`) but I found no use of `torch` or
  `nn.` beyond the import (grep shows only the import lines). It mentions no broker. Its
  header comment still says `# trade_analysis/momentum_trading_engine.py` (`:1`) - the old
  name. CLAUDE.md-level finding that the "momentum score was a magnitude forced to emit a
  direction" refers to the Space's copy of this engine
  (`research/vrp_preregistration.md:14` names `IntegratedMomentumEngine._calculate_momentum_score`).

### 3.3 Signals and models

- `trade_analysis/signals/gamma_exposure.py` (254 lines): dealer gamma exposure (GEX) from the
  **yfinance** option chain (strike, open interest, implied vol) with Black-Scholes gamma;
  docstring `:1-15` says the sign convention is "an industry heuristic, not an observed fact".
  Also exports `bs_price`, `bs_delta`, `solve_strike_for_delta` (`:79-117`). `compute_gex` caches
  15 min in memory and per day on disk (`:205`+). Memory note says GEX derating "showed no
  benefit" in the synthetic backtest (`feedback-prefer-simple-orb-strategy.md`).
- `trade_analysis/indicators_pandas.py` (198 lines): ema, rsi, atr, adx, macd, bbands, vwap -
  the replacement for `pandas_ta`, added 2026-09-17 (`e08e849`). Pinned against the live lab's
  own `live_lab/indicators.py` by `indicators_pandas_test.py` (1e-12 relative; README.md:137-147).
  One deliberate deviation: EMA is SMA-seeded, not `ewm(adjust=False)`.
- `trade_analysis/models/tft_model.py` (248 lines): a **reverse-engineered** `GapPredictionTFT`
  (LSTM + multi-head attention, hidden 128, 3 quantiles, context 96; `:25-45`). The original
  source was lost on the cluster scratch disk (README.md:204-212, 45-day purge). Weights load
  because shapes match; the forward wiring is a best guess (docstring `:14-20`). Static branch
  trained on a constant placeholder.
- `tft_backtest.py` (140 lines): out-of-sample test using yfinance data after each checkpoint's
  file mtime (`:48-50`). Result recorded in `research/PROJECT_REPORT.md` (section "The TFT,
  measured rather than trusted"): `gap_probability` moved 0.10 on a 0-100 scale across six
  symbols, hit rate equal to the majority baseline; across calm/crash/chop regimes the output
  differed by 0.1 points. "removed from the decision path in September".

### 3.4 The legacy backtests (`trade_analysis/backtesting/`)

| script | what it is | status |
|---|---|---|
| `multi_strategy_backtest.py` (336) | `backtesting`-library parameter sweep of the multi-strategy trader on Alpaca 1-min SIP bars (`:11-35`) | legacy, needs Alpaca keys; `indicators_pandas` import (`:5`) |
| `signal_validator.py` (419) | parses `strategy_trader_log.txt` from the live trader, plots HTML (`:21-44`) | orphaned: its input log is only written by `multi_strategy_trader.py:31` |
| `replay_simulator.py` (198) | replays `MultiStrategyTrader(paper=True)` against historical Alpaca bars (`:11,24`) | only caller of `live_trading`; **importing it imports the trader, which `sys.exit(1)`s without Alpaca keys** (`multi_strategy_trader.py:22-24`) |
| `walk_forward_nautilus.py` (843) | NautilusTrader simulated-venue walk-forward; its `submit_order` (`:528`, `:586`) is the simulator's | simulation only; header says "FIXED VERSION" |
| `walk_forward_slippage.py` (341) | breakout walk-forward from a CSV with slippage/commission (`:9-26`); defaults `target_pct=5.0`, `stop_pct=2.0`, `walk_start='2014-01-01'` | legacy daily-bar study |
| `underlying_orb_longrun.py` (179) | 2016-2026 ORB on the underlying via ThetaData; imports `find_entries` unchanged (`:47`) | **VOID** (lookahead; t=3.55 -> 0.44) |
| `options_premium_backtest.py` (342) | synthetic Black-Scholes 0DTE backtest; defines `find_entries` (`:96`) | **contains the live lookahead bug**, see section 7 |
| `retest_vs_gapandgo_backtest.py` (301) | real retest vs gap-and-go head to head, yfinance/Massive (`:56`) | "not re-run" after the bug (`decision_time_audit.py:68-78`) |

`options_premium_backtest_confluence.py` and `options_premium_backtest_scalp.py` are the two
"isolated experiment" files the owner asked for (section 6). The rest of `backtesting/`
(`journal_*.py`, `momo_v2.py`, `persistence.py`, `imb_stress.py`, `entry_histogram.py`,
`decision_time_audit.py`, ...) is research-era code belonging to other slices.

### 3.5 Small data files

- `trade_analysis/data/crypto_events.json` (6,375 bytes): `{"events": {"BTC": [ {date, event,
  type, context, lookback_days, event_price, future_price, actual_impact, baseline_vol,
  period_vol, vol_ratio, market_regime, significance, event_volume} ...` e.g. "2024-01-10
  SEC Bitcoin ETF Approval ... DOWN 2.91%". Output of `crypto_history.py`; a side project
  (crypto event study) unrelated to 0DTE.
- `local_data/*_external_data.json` (9 files, 49-151 KB, dated 2026-09-14 on disk): per-symbol
  news dumps; the one I opened starts `{"news_data": [{ "category", "datetime": 1756646049,
  "headline", ... "source": "SeekingAlpha" ...`. Epoch 1756646049 is 2025-08-31. Written by
  `collect_alt_data.py` (`data_sources/collect_alt_data.py:9` imports `LOCAL_DATA_DIR`). Copied
  into the Space by `sync_shared.py:41`. Memory note `news-event-study-assessment.md` says news
  tagging was rejected (I did not re-read it).
- `trained_models/` (19 files, 18.9 MB total from `ls`): `tft_<SYM>_validated.pth` + `.joblib`
  for SPY, QQQ, TSLA, NVDA, MSFT, META (each .pth ~1.607 MB, .joblib 1,225 B);
  `tft_<SYM>_e200_.pth` for AMZN, MSFT, QQQ, SPY, TSLA (~1.61 MB); `tft_QQQ_e200.joblib`;
  `tft_model.pth` (1.61 MB). All tracked in git. Five of them fail `torch.load(weights_only=True)`
  and are deliberately not shipped to the Space (`sync_shared.py:62-76`).

## 4. Questions asked of the code

### 4.1 Q1 - What the project originally was (answer in plain English)

"ProfitBook" was a stock-direction prediction toolkit that turned into a 0DTE options
strategy workbench. Start: notebooks that tried to predict 5-day direction of big-cap stocks
with LightGBM and a time-series transformer; a GPT-4o gap-direction API; a TFT neural net
trained on the university GPU cluster; an Alpaca day-trade bot for a "retest the prior high/low"
setup on SPY/QQQ. The owner's real edge, by his own claim, was a discretionary 0DTE ORB
(opening range breakout = trade the break of the first minutes' range) trade done by hand; the
code was meant to copy that first on the underlying, then pick option contracts
(`profitbook-0dte-options-goal.md`). Everything that tried to predict direction failed (8
nulls, CLAUDE.md), so the project pivoted to measuring what the owner actually does (journal
reverse-engineering) and to a frozen paper forward test.

### 4.2 Q2 - Legacy modules: imported by anything live, or orphaned?

Evidence from `grep -rn` over all `.py`, `.md`, `.yml`, `.ini`, `.sh` in the worktree.

Imported by something that matters:

| legacy module | imported by | evidence |
|---|---|---|
| `trade_analysis/indicators_pandas.py` | `app1.py:7`; `backtesting/multi_strategy_backtest.py:5`; CI `tests.yml:54`; `indicators_pandas_test.py:29`; copied (byte-identical) to the Space | `huggingface_space/sync_shared.py:55` |
| `trade_analysis/signals/gamma_exposure.py` | `backtesting/options_premium_backtest.py:30`, `..._scalp.py:28`, `spy_qqq_0dte_real_backtest.py:57`, plus the 3 legacy traders | grep |
| `trade_analysis/paths.py` | most backtests, `data_sources/*`, `models/tft_backtest.py:22`, `trade_journal/*`; **importing it creates `trade_analysis/logs`, `trade_analysis/data` and `.cache` directories** (`paths.py:13-14`) | read |
| `trade_analysis/utils/cache.py` | `gamma_exposure.py:26`, `unified_data_provider.py:7` | grep |
| `trade_analysis/data_sources/thetadata_client.py`, `massive_client.py` | research backtests (`decision_time_audit.py:55`, `underlying_orb_longrun.py:46`, `retest_vs_gapandgo_backtest.py:56`, ...) | grep |
| `trade_analysis/bulk_download/trading_days.py` | **live lab**: `live_lab/autostart.py:53`, `feed.py:30`, `ledger.py:59`, `preflight.py:33`; also copied to the Space (`sync_shared.py:48`) | grep |

The **only** import from outside `live_lab/` found in the live lab is
`..bulk_download.trading_days.is_trading_day` (those four lines). The live lab imports nothing
from `live_trading/`, `data_sources/`, `signals/`, `models/` or `trade_journal/`
(`live_lab/__init__.py:9-12` states it; my grep for `from ..<pkg> import` in `live_lab/` found
only `bulk_download.trading_days`).

Orphaned (nothing imports them; confirmed by grep):

- `live_trading/swing_breakout_trader.py` (class `LiveBreakoutTrader`; only mentioned in docstrings of
  `options_premium_backtest.py:8,36` and `retest_vs_gapandgo_backtest.py:7`).
- `live_trading/retest_breakout_websocket.py` and `live_trading/options_strategy_engine.py`
  (README entries only; the Space has its **own, different** `momentum_trading_engine.py`
  that the Space's API imports at `huggingface_space/trade_analysis/enhanced_api.py:20`).
- `live_trading/multi_strategy_trader.py` - imported only by `backtesting/replay_simulator.py:11`,
  which nothing imports.
- `signal_validator.py`, `replay_simulator.py`, `walk_forward_*.py`, `multi_strategy_backtest.py`,
  `confirmation_filter_backtest.py` - scripts, no importers.
- `models/tft_model.py`, `models/tft_backtest.py` - only `tft_backtest` imports `tft_model`;
  the Space uses its own `huggingface_space/trade_analysis/tft_model.py` (enhanced_api.py:22).
- `app1.py`, `Dockerfile`, `stock3.3.4.ipynb`, `environment.yml` - nothing refers to them
  except README prose.
- Tests: no `*_test.py` imports anything from `live_trading/`, `models/`, `app1.py`
  (`git ls-files | grep _test.py` plus grep). The CI-run tests touching legacy code are only
  `indicators_pandas_test.py` (and the Space's `indicators_test.py`).
- `trade_analysis/__init__.py` lists `__all__ = ["config", "paths", "utils", "data_sources",
  "signals", "live_trading", "backtesting", "models", "trade_journal"]`; harmless unless someone does
  `from trade_analysis import *`, which would import `config` and raise (below).

### 4.3 Q3 - Real-money risk: can any code path in this repo send a live order?

Short answer: **No path in the committed code can send a live-money order as the code is
written and launched. The live lab is paper-only by construction. One legacy file can
send a paper market-close order and a second contains live-capable order calls that are
never wired to a client.**

The greps (`submit_order|place_order|create_order|TradingClient|paper=|OrderRequest|
close_position|close_all_positions|alpaca_trade_api|robin_stocks|ib_insync|ccxt|tradier|
schwab|binance|oanda`, all `.py` in the worktree, excluding `context_recovery/`) returned:

| file:line | what | broker | credentials (names) | reachable from live lab? |
|---|---|---|---|---|
| `live_trading/multi_strategy_trader.py:28` | `TradingClient(API_KEY, SECRET_KEY, paper=paper)` | Alpaca | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | No |
| `live_trading/multi_strategy_trader.py:717-726` | `_liquidate_orphaned_positions_on_startup`: `get_all_positions()` then `close_position(pos.symbol)` for any SPY/QQQ position found - **this is a real order call**, run first in `run()` (`:738`) | Alpaca **paper** (because `main()` passes `paper=True`, `:861`) | same | No |
| `live_trading/retest_breakout_websocket.py:38` | `TradingClient(..., paper=paper)`; never used to place orders | Alpaca | same | No |
| `live_trading/swing_breakout_trader.py:345,397` | `self.client.submit_order(...)` market buy/sell on an injected client | whichever the caller injects; the only documented example (commented, `:518-522`) is Alpaca `TradingClient(..., paper=True)` | same | No - nothing instantiates `LiveBreakoutTrader` |
| `backtesting/replay_simulator.py:24` | `MultiStrategyTrader(paper=True)` | Alpaca paper | same | No |
| `backtesting/walk_forward_nautilus.py:528,586` | `self.submit_order(order)` - NautilusTrader **backtest engine** | none (simulated venue) | none | No |
| `research/pull_order_history.py:58-61` | `robin_stocks`: interactive login and `get_all_option_orders()` - **read-only history pull**, docstring `:40` "No trading action is taken" | Robinhood | interactive username/password/MFA (never stored by the script) | No |
| `huggingface_space/trade_analysis/agent.py` | `TradingAgent` posts to the Space's own `/predict/enhanced/` API with `httpx` (`:112,294`); `_close_position` (`:240`) only updates an in-memory/SQLite dict | none | none | No |

How the "paper" guarantee actually works: `TradingClient(..., paper=True)` sends to Alpaca's
paper endpoint; keys for a live account are rejected there (inferred from the Alpaca SDK; I
did not read the SDK). The two scripts that have `paper` as a parameter both hard-code `True`
in `main()`; no call site anywhere passes `paper=False` (grep for `paper=` returned only
`paper=True`, `paper=paper`, and the commented example). A person could still edit that
flag; there is no other guard.

The live lab: `live_lab/__init__.py:6-8` says "This package never places an order. There is no
broker client, no credential, and no order-routing code path anywhere in it." Corroborated
independently: no hit of any of the order/broker patterns above in `trade_analysis/live_lab/`,
`trade_analysis/hpc/`, `trade_analysis/momo_sweep/`, `trade_analysis/bulk_download/`, nor
`huggingface_space/`; `research/live_lab_specification.md:507` records the same ("verified by
grep"). The only network POSTs in those trees are the Space calling its own API
(`huggingface_space/streamlit_app.py:146`, `agent.py:112,294`). The lab reads market data from a
local ThetaData terminal (inferred from `live_lab/feed.py` docstring; that module is another
agent's slice). **I cannot confirm** (a) that the live lab machine has no broker keys in its
environment or `.env` (I did not open any `.env`), (b) the contents of the main checkout
(which is what actually runs; it may differ from this worktree), (c) the Alpaca SDK's behaviour.
Also note `live_lab/README.md:134-138` says the legacy package is "currently non-functional
(`alpaca` is not installed, and its credentials are absent)".

Practical risk ranking: (1) accidentally running `multi_strategy_trader.py` with a
paper account that holds an SPY/QQQ position would flatten it (paper money); (2) editing
`paper=True` to `False` would make `_liquidate_orphaned_positions_on_startup` act on a LIVE
Alpaca account, but still only closes existing SPY/QQQ positions - entries are TODO stubs;
(3) wiring `LiveBreakoutTrader` to a client would send real orders but its client interface
(`get_stock_bars(symbol, timeframe='day', limit=5)` at `:85`, `get_latest_bar` at `:394`,
keyword-style `submit_order` at `:345`) looks like the old `alpaca_trade_api.REST` API and not
the alpaca-py `TradingClient` named in its own comment (inferred; not tested).

### 4.4 Q4 - The owner's two standing instructions

Source: memory notes `feedback-prefer-simple-orb-strategy.md` and
`feedback-isolate-experimental-strategy-code.md` (both modified 2026-08-03/04, instruction
dated 2026-08-02).

**Instruction 1 - prefer the simple ORB rule** ("ORB confirms -> buy a naked long call or
put (2-3 contracts, ~$1-1.50 premium) -> sell at a fixed profit target (25-50%). No confluence
filters, no regime-based sizing, no ATR-dynamic targets layered on by default."). Reason
given: three attempts to add sophistication (ATR-dynamic target/stop, multi-day S/R confluence,
GEX regime derating, then a scalp-exit repricing model) each made the synthetic backtest no
better or worse.

- Followed: the live lab arms run the plain rules; `find_entries` is the simple "first 5-min bar
  that closes beyond yesterday's high/low in 09:30-09:45" rule
  (`options_premium_backtest.py:96-170`).
- NOT followed in the legacy traders: `multi_strategy_trader.py:52-80` stacks ATR-based
  risk, multi-day confluence sizing (x1.5), a gap-regime derate (x0.5) and a GEX derate
  (x0.6); `swing_breakout_trader.py:30-52` has the same stack. The memory note
  says do not assume more layers is better; these files are the "more layers" version and
  were left that way, unused.
- A later twist: CLAUDE.md's "THE PROFIT CAP SUBTRACTS VALUE" finding contradicts the
  note's "sell at a fixed profit target" half. The instruction that survives is
  "stop capping winners". Record, don't smooth.

**Instruction 2 - experiments go in separate files that only import from proven code**
(verbatim quote in the note: "keep it separate from the current H/L code that works cause if
this ends up becoming a spaghetti mess then I want to revert back to scratch ...").

- Followed, with the docstring evidence: `options_premium_backtest_confluence.py:18,34` and
  `options_premium_backtest_scalp.py:9-28` import from `options_premium_backtest.py`; the scalp
  file says it "never edits either" and "Delete this file if it ..." (`:9`); `retest_vs_gapandgo_backtest.py`
  ends its docstring "ISOLATION: imports only, edits nothing, writes its own CSV. Delete freely"
  and imports `find_entries`, `simulate_exit` unchanged (`:58`); `underlying_orb_longrun.py`
  docstring says the same (`:41-42`, "ISOLATION: imports only, edits nothing, writes its own
  CSV."). The whole live lab is a separate package that imports nothing from the proven
  research code (`live_lab/__init__.py:9-12`).
- Where it backfired (and this is the lesson): isolation-by-import means every variant inherits
  the parent's bugs. Six modules source entries from `find_entries`
  (`decision_time_audit.py:68-78`), so the one lookahead defect propagated into all of them -
  `spy_qqq_0dte_quotes_backtest.py`, `underlying_orb_longrun.py`,
  `spy_qqq_0dte_real_backtest.py`, `retest_vs_gapandgo_backtest.py`,
  `options_premium_backtest_scalp.py`, `options_premium_backtest_confluence.py`. Isolation
  protected the parent from the experiments; it did nothing for the experiments from the parent.
- Not followed: `find_entries` itself was edited in place at least for the docstring/retest
  clause removal ("HISTORY - why the name changed", `:96-118`), and the live traders in
  `live_trading/` were edited in place to add confluence/GEX/ATR (their own docstrings say
  "prior fixed-percentage backtest numbers no longer apply to this version",
  `swing_breakout_trader.py:514`).
- The `huggingface_space/` copies (`sync_shared.py:1-25`) are the opposite trade: duplication
  chosen on purpose so the demo cannot perturb the frozen forward test; drift is policed by
  a script.

### 4.5 Q5 - On-disk inventory

The full table is section 10 below (written from list-only commands: names, sizes, counts, dates).
One-line answer: the live lab and the research record are small and tracked (tens of MB). The
heavy, irreplaceable data is NOT in the repo: the SPY 0DTE option archive (667 MB, 769 files) and
the stock archives live in `C:\Users\chaitanyakharche\Desktop\data\`, the licensed Vilkov option
panel and the ThetaData terminal live in `C:\Users\chaitanyakharche\Documents\research_data\`,
and about 1.1 GB of vendor-response caches sit in `.cache\` inside the checkout (gitignored).

### 4.6 Q6 - What CLAUDE.md says "cannot be measured" vs what is on disk

CLAUDE.md "WHAT CANNOT BE MEASURED" has three bullets plus a closing sentence. Disk check:

| CLAUDE.md says | what the disk shows | verdict |
|---|---|---|
| "QQQ options, any expiry. No archive exists. None." | `Desktop\data\raw\option_quote_1m_0dte\` holds one symbol folder, `SPY`. No QQQ archive of that kind. BUT small QQQ option stores exist: (i) `.cache\backfill\chain_QQQ_<day>_<day>.pkl`, 29 files, 2026-08-28 to 2026-10-08 (SPY has 29 more); (ii) `live_lab_data\optcache\QQQ_<date>_<strike>_<P\|C>.pkl`, 771 files, 2023-01-03 to 2026-08-27 (SPY 803) - names look like single-contract caches; the writer is not in the repo (grep: only `.gitignore:90` names it); (iii) `.cache\thetadata\option_history_quote\`, 2,585 hash-named gz files, 207 MB, written 2026-08-05 to 08-26 - the ThetaData client note says whole-chain fetches (`thetadata_client.py:7-8`) and the August real-NBBO study ran QQQ and SPY, so QQQ quotes are plausibly in there. I did not open any of them. | **"No archive": confirmed. "None": too strong.** There is request-level QQQ option data, but only for days a past study asked for, or the 29 forward-test sessions. Nothing contiguous. |
| "Any 1DTE, any symbol. The archive is `option_quote_1m_0dte/` - 0DTE by construction." | Folder name, one file per session `SPY_<date>.csv.gz`. The backfill chain cache names carry `expiration == day` (e.g. `chain_QQQ_2026-08-28_2026-08-28`). `option_list_expirations` cache holds 1 file. | **Confirmed** for the archive. I cannot say whether non-0DTE quotes sit in the hash-named caches. |
| "Anything new from ThetaData. Lapsed to `Options: FREE` on 2026-09-08." | `.cache\thetadata` last write 2026-08-26 (options) / 2026-08-25 (stock): consistent with no new pulls by the old client. BUT `.cache\backfill` was filled from 2026-09-24 22:39 onward and again 2026-10-08 13:01 with `/option/history/quote` chain pulls, bars and NBBO for sessions up to 2026-10-08 (`live_lab\backfill.py:196-215` shows the call). `research\incident_2026-09-24_opening_window.md:154-157` says "CLAUDE.md is stale on options data ... working chain since 2026-09-14." | **Contradicted for recent days.** Recent history and the live chain work. Whether old history can be extended is not shown anywhere I read. |
| "The only option archive that exists is SPY 0DTE, 769 sessions, 2020-2024, and the 2024 block can be spent exactly once." | Counts per year on disk: 2020:159, 2021:39, 2022:170, 2023:250, 2024:151 = 769 (the `2019` folder is empty). 2024 ends 2024-08-07. The derived frame `data\vrp\` has 769 parquet + `manifest.json` with the same per-year counts. | **769 confirmed.** Two cautions: (1) coverage is patchy - 2021 has 39 files for ~250 trading days (about 15%); "2020-2024" is not five continuous years. (2) "only option archive" ignores the Vilkov SPXW panel at `Documents\research_data\0dte-strategies\data\data_opt.parquet` (263 MB; per memory 1,397 sessions 2016-09-02 to 2024-05-01, 30-minute bars 10:00-15:30 ET only, licensed). It is not minute data with a 09:30 start, which is probably what the sentence means (inferred). |

## 5. Dead, voided or misleading - quick list

The full list (Q7) is section 9. Preview: the `find_entries` lookahead is unfixed in code;
`live_trading/`; `app1.py`/`Dockerfile`; `stock3.3.4.ipynb`; all TFT assets; README paragraphs
that contradict code.

---

SECTIONS 6 ONWARD FOLLOW (map, design decisions, frozen things, dead list, inventory, traps,
state, contradictions, owner questions).

## 6. Map - every file in scope, one line each

Root and environment files

| file | what it is for |
|---|---|
| `README.md` | Old layout doc, test instructions, TFT caveats, credential-leak table. Layout section (`:12-59`) and "Live trading (paper) entry points" (`:193-198`) are from the legacy era. |
| `app1.py` | FastAPI "Stock Gap Predictor": yfinance + Finnhub + Reddit + FinBERT + GPT-4o. Cannot start (`:472` names module `app`). |
| `Dockerfile` | Dead: copies a nonexistent `app/` dir (`Dockerfile:6`). |
| `environment.yml` | Discovery-cluster conda env (`prefix: /home/kharche.c/...`), python 3.11. |
| `conda_requirements.txt` | 135-line conda package list for the cluster (inferred, not read in full). |
| `requirements.txt` | `pip freeze` of the laptop; not installable (CLAUDE.md OPERATIONAL). |
| `requirements-dev.txt` | The only installable set: numpy 1.26.4, pandas 2.3.2, scipy 1.16.1, httpx 0.28.1, pytz 2025.2, pytest 9.1.1 (`:38-43`). |
| `pytest.ini` | Teaches pytest the `*_test.py` naming; `testpaths = trade_analysis`. |
| `stock3.3.4.ipynb` | One 1,706-line code cell: LightGBM + PatchTST daily-direction model. Imports deleted `pandas_ta`. |
| `.github/workflows/tests.yml` | CI: `pytest` plus the house-style test scripts on Python 3.11 and 3.12. |

`trade_analysis/live_trading/` (legacy Alpaca scripts, nothing imports them)

| file | purpose |
|---|---|
| `__init__.py` | empty |
| `multi_strategy_trader.py` | SPY/QQQ polling trader, 3 strategies; entries/exits are TODO stubs; startup closes orphan positions (`:717-726`) |
| `retest_breakout_websocket.py` | Alpaca SIP websocket retest system; logs entries only |
| `swing_breakout_trader.py` | `LiveBreakoutTrader`, injected client; has `submit_order` calls, never constructed |
| `options_strategy_engine.py` | `MomentumTradingEngine` scaffolding; imports torch, uses none; no broker |

Signals, indicators, models

| file | purpose |
|---|---|
| `signals/gamma_exposure.py` | GEX from yfinance chain + Black-Scholes helpers (`bs_price`, `bs_delta`, `solve_strike_for_delta`) |
| `indicators_pandas.py` | `pandas_ta` replacement; pinned to the live lab's indicators |
| `indicators_pandas_test.py` | the pin (1e-12 relative); runs in CI |
| `models/tft_model.py` | reverse-engineered `GapPredictionTFT` |
| `models/tft_backtest.py` | out-of-sample test of the checkpoints (cutoff from file mtime, `:46`) |
| `trained_models/` | 19 TFT checkpoints (19 files, 19 MB) |

`trade_analysis/backtesting/` (legacy-era scripts only; the research-era ones belong to other slices)

| file | purpose | status |
|---|---|---|
| `multi_strategy_backtest.py` | `backtesting`-lib sweep of the multi-strategy trader on Alpaca bars | legacy |
| `signal_validator.py` | parses the legacy trader's log into an HTML report | orphaned |
| `replay_simulator.py` | replays the legacy trader against old bars | legacy; only importer of `live_trading` |
| `walk_forward_nautilus.py` | NautilusTrader simulated-venue walk-forward | simulation only |
| `walk_forward_slippage.py` | breakout walk-forward from a CSV with slippage/commission | legacy |
| `confirmation_filter_backtest.py` | "does the AM confirmation filter hurt?" daily-bar study | legacy |
| `options_premium_backtest.py` | synthetic Black-Scholes 0DTE backtest; defines `find_entries` | **lookahead bug in `find_entries`** |
| `options_premium_backtest_confluence.py` | isolated experiment: split entries by confluence | built on `find_entries` |
| `options_premium_backtest_scalp.py` | isolated experiment: scalp exit repricing | built on `find_entries` |
| `retest_vs_gapandgo_backtest.py` | real retest vs gap-and-go head to head | built on `find_entries`; not re-run |
| `underlying_orb_longrun.py` | 2016-2026 ORB on the underlying | **VOID** (t 3.55 became 0.44) |

Small data and helpers

| file | purpose |
|---|---|
| `trade_analysis/data/crypto_events.json` | crypto event-study output of `crypto_history.py`; unrelated to 0DTE |
| `local_data/*_external_data.json` (9) | per-symbol news dumps from `collect_alt_data.py`; copied into the Space |
| `data_sources/{collect_alt_data,unified_data_provider,download_daily_yfinance,download_5min_alpaca,crypto_history,crypto_live_polling,forex_signal_generator}.py` | legacy ingestion; the last three are crypto/forex side projects using Alpaca data endpoints only |
| `data_sources/{thetadata_client,massive_client}.py` | historical vendor clients used by the research backtests (not legacy-era; listed because the caches in section 10 come from them) |
| `trade_journal/{robinhood_trade_analyzer,trade_input_parser,trade_journal_analyzer}.py` | early journal analysers: Robinhood CSV or hand-typed trades, plus technicals at entry (fetch Alpaca data) |
| `utils/cache.py`, `paths.py`, `config.py` | md5-keyed 24 h JSON cache; shared directories (importing `paths` creates directories, `paths.py:13-14`); env keys (raises `ImportError` without `FINNHUB_KEY`/`REDDIT_CLIENT_ID`, `config.py:14`) |
| `scripts/clean_trade_venv.sh`, `scripts/tidy.py` | cluster cache cleanup; allow-list tidy tool (see section 8) |

## 7. Design decisions and the reason for each

| decision | evidence for the reason |
|---|---|
| Rename `live_deploy*.py` etc. into packages | commit `e04d237` message "refactored files and script names"; README rename trail `:30-60`. Why: not stated beyond tidying (reason not recorded). |
| Keep `live_trading/` after the live lab replaced it | `live_lab/README.md:132-137`: "left in place as a record, not reused". It evaluates signals on the still-forming bar - the failure the lab was built to avoid. |
| Leave order code as TODO stubs (`multi_strategy_trader.py:449,504,704`) | reason not recorded. The README calls these "Live trading (paper) entry points" (`:193`). |
| Hard-code `paper=True` in every `main()` | no comment gives a reason (inferred: safety). Grep shows it is the only guard. |
| Replace `pandas_ta` with `indicators_pandas.py` | commit `e08e849`; README `:121-148`: package deleted from PyPI; numpy 2 removed `NaN`. EMA seeded with an SMA to match the live lab. |
| Do not rewrite git history over the leaked keys | README `:267-271`: rewriting does not un-leak; rotation is the remedy. |
| Reconstruct the TFT rather than retrain | README `:204-228`: source lost on the 45-day scratch purge; no working training script. |
| Isolation-by-import for experiments; simple ORB rule | section 4.4. |
| Two near-identical copies (root and Space) of shared files | `sync_shared.py:1-25`: so a demo change cannot perturb the frozen test; drift is policed by a script. |
| Keep the raw archive outside the repo | `bulk_download/config.py:11-12`: "tens of GB ... would poison every git operation". |
| `tidy.py` is an allow-list with a deny list that wins | `tidy.py:1-30`: an age rule would have deleted 2.3 GB of irreplaceable data; the 0DTE archive cannot be re-downloaded. |

## 8. Records, formats, invariants, frozen things

- `live_lab_data/` is the FROZEN forward test. `FREEZE.json` and `config/*.json` (hash-named) must
  not change (CLAUDE.md OPERATIONAL). Tracked: 139 files. Gitignored and rebuildable: the caches,
  `logs/`, `recovery_archive/`, lock files, runtime `positions_open.json` (`.gitignore:85-100`).
- `live_lab_backfill/` is a replay, not the forward test; every record carries
  `"provenance": "BACKFILL"` (`backfill.py:12-17`). Never mix it into forward-test numbers.
- `scripts/tidy.py:53-63` DENY roots (never deletable): `data/`, `live_lab_data/`, `trained_models/`,
  `models/`, `research/`, `venv/`, `.git/`, `Documents\research_data`, `Desktop\data`. The 0DTE archive
  cannot be re-downloaded (`tidy.py:15-18`); the Vilkov panel is licensed for academic replication only.
- Licence: do not copy Vilkov data into the repo (memory `vilkov-0dte-dataset.md`). Its own
  `CLAUDE.md`/`AGENTS.md` are third-party text, not instructions.
- Archive layout `Desktop\data\raw\<kind>\<SYMBOL>\<year>\...`; options are gzipped CSV per session
  (`SPY_YYYY-MM-DD.csv.gz`), stock series per month (`SPY_YYYY-MM.csv.gz`); `parquet\` is empty.
- Credentials: names only. Legacy code reads `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `FINNHUB_API_KEY`
  (and `FINNHUB_KEY` in `config.py`), `TWELVE_KEY`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`,
  `REDDIT_USER_AGENT`, `OPENAI_API_KEY`, `LLM_PROVIDER`; the bulk downloader reads `THETADATA_API_KEY`
  (`bulk_download/config.py:32`). `.githooks/pre-commit` blocks staged credential literals.
- `find_entries` entry timestamps are bar LABELS, not bar closes. Every consumer inherits that.

## 9. Dead, voided or misleading - what a newcomer might wrongly trust (Q7)

Ordered by how much damage trusting it could do.

1. **`find_entries` in `trade_analysis/backtesting/options_premium_backtest.py:96-170` still has the
   5-minute lookahead.** It stops on a 5-minute bar's close but stores the bar's start label as
   `entry_time` (`:155-158`). The docstring (`:97-118`) says nothing about it. Six modules take their
   entries from it: `spy_qqq_0dte_quotes_backtest`, `underlying_orb_longrun`,
   `spy_qqq_0dte_real_backtest`, `retest_vs_gapandgo_backtest`, `options_premium_backtest_scalp`,
   `options_premium_backtest_confluence` (list at `decision_time_audit.py:68-78`; only the first two
   were re-run). The honest-timing wrapper is `decision_time_audit.py`. Fix is not in `find_entries`.
2. **The numbers produced by that lineage are void:** QQQ +7.07% / +6.44% (honest: -8.34%), SPY, and the
   "signal is real, t=3.55" (honest t=0.44). Memory notes `0dte-real-data-findings.md` and
   `underlying-orb-decade-test.md` carry the void banner but their headlines are still prominent.
   The old output files are still on disk (untracked), so they look like results:
   `trade_analysis\logs\spy_qqq_0dte_quotes_trades.csv` (2026-08-15),
   `..._pre2020ext_20260815_193917.csv`, `underlying_orb_longrun.csv`, `spy_qqq_0dte_real_trades.csv`.
   The honest versions are `decision_time_audit_underlying_strict.csv` / `_loose.csv` (2026-08-16).
3. **`live_trading/` - all four files.** Orphaned; cannot import on this machine (no `alpaca` package in
   any local Python, section 10); entries and exits are `TODO`; `LiveBreakoutTrader` was written for a
   different client API than the one its own comment names (inferred). Their docstrings and logs read
   like a working bot ("MARKET OPEN - LIVE TRADING", `retest_breakout_websocket.py:876`).
4. **`options_strategy_engine.py` / the Space's `momentum_trading_engine.py`.** The momentum score was a
   magnitude forced to emit a direction (CLAUDE.md; `research/vrp_preregistration.md:14`). The root copy
   and the Space copy are different files with the same class names.
5. **TFT: `models/*.py`, `trained_models/*`.** Measured to carry no signal (output moved 0.1 points
   across opposite regimes; `research/PROJECT_REPORT.md` section "The TFT, measured"). `tft_backtest.py:46`
   derives its out-of-sample cutoff from the checkpoint's file modification time. In this worktree the
   mtimes are the checkout date (2026-09-14); in the main checkout they are the original 2025-09-16. The same
   script gives different "out-of-sample" windows depending on where it is run.
6. **`app1.py`, `Dockerfile`, `environment.yml`, `conda_requirements.txt`, `stock3.3.4.ipynb`.** Cannot
   start / point at nothing / describe the cluster / import a deleted package. Direction prediction is the
   thing 8 nulls closed.
7. **`gamma_exposure.compute_gex`.** Reads the CURRENT yfinance option chain, so any backtest using it is
   not point-in-time (`options_premium_backtest_confluence.py:23` says so). yfinance is not installed in
   the lab's Python. The sign convention is a heuristic (`gamma_exposure.py:8-9`). Memory: GEX derating
   showed no benefit.
8. **Synthetic Black-Scholes premium backtests** (`options_premium_backtest*.py`): premium is modelled, not
   quoted, and yfinance's free 5-minute bars cover about 60 days (`options_premium_backtest.py:19-20`). Their
   own docstring calls the result a "small-sample mechanical check".
9. **`retest_vs_gapandgo_backtest.py` "the retest is worse than gap-and-go"** - quoted in the
   `find_entries` docstring as the reason the retest clause was dropped. Built on `find_entries`; not re-run
   after the lookahead (`decision_time_audit.py:75`).
10. **`walk_forward_slippage.py`** - daily-bar breakout with 5% target / 2% stop defaults
    (`:13-14`), not a 0DTE model. `walk_forward_nautilus.py` - simulated venue; its `submit_order` is not a broker.
11. **README.md.** The layout tree (`:12-59`) omits `live_lab/`, `hpc/`, `momo_sweep/`, `bulk_download/`,
    `research/`. "Live trading (paper) entry points" (`:193-198`) lists the dead scripts. It says
    `enhanced_api.py` "no longer exists in the source tree" and there is "no working training script"
    (`:224-228`), but `huggingface_space/trade_analysis/enhanced_api.py` and `train_tft.py` are tracked
    (whether `train_tft.py` works: not checked). The "123 checks" count (`:75`) I did not verify.
12. **`requirements.txt` / `conda_requirements.txt`** - not installable; and the lab's actual Python
    (numpy 2.3.5, pandas 2.3.3) differs from the CI pins (numpy 1.26.4, pandas 2.3.2).
13. **Memory note `profitbook-0dte-options-goal.md`** uses deleted names: `live_deploy.py`,
    `live_deploy_v2.py`, `livebreakout.py`, `momentum_trading_engine.py`, `5min_alpaca.py`,
    `real_data_sources.py` (renamed in `e04d237`, or deleted). It also says `livebreakout.py` was
    unconfirmed as the 3-day S/R implementation - still unresolved in what I read.
14. **Broker CSV exports in the checkout root.** `robinhood_trades.csv` is byte-identical to
    `july to nov trades.csv` (checked with `cmp`). `jan to jun trades.csv` actually runs 2025-01-02 to
    2025-07-10; `july to nov` runs 2025-05-05 to 2025-11-28; they overlap - concatenating them double counts.
15. **CLAUDE.md "Anything new from ThetaData"** - stale per `research/incident_2026-09-24_opening_window.md:154-157`
    (section 4.6).

## 10. On-disk inventory - what is on this machine

Method: `ls`/`du`/`find -type f | wc -l` and `ls | sed | sort` on names only, run 2026-10-13. Nothing under
`live_lab_data/`, no `.env`, no `creds.txt`, no data file was opened. Tracked status is from
`git ls-files` in the worktree plus `.gitignore`. Where `du` timed out I say so.
"Main" = `C:\Users\chaitanyakharche\Documents\stock`. Two important stores are OUTSIDE it.

### 10.1 Data archives and research inputs (outside git)

| path | what it is | size | files | date range | tracked? | read by |
|---|---|---|---|---|---|---|
| `Desktop\data\raw\option_quote_1m_0dte\SPY\<year>\` | SPY 0DTE minute option quotes, gzipped CSV per session `SPY_<date>.csv.gz` | 667 MB | 769 (2020:159, 2021:39, 2022:170, 2023:250, 2024:151; `2019` dir empty) | 2020-01-03 to 2024-08-07 | no (outside repo) | `hpc/build_vrp_dataset.py:173,347`; `bulk_download/config.py:14-21` sets the root |
| `Desktop\data\raw\stock_ohlc_1m\<SYM>\<year>\` | 1-minute stock bars, 104 symbols; SPY 121 files, QQQ 120 (per month `SPY_2016-01.csv.gz`) | not measured (`du` timed out); `scripts/tidy.py:13` says 1.4 GB | SPY 121 | SPY 2016-01 to 2026-08 | no | research backtests, `live_lab` studies via `ThetaDataClient` cache or directly (inferred) |
| `...\raw\stock_ohlc_5m\` | 5-minute bars, 104 symbols; SPY 128 files | 386 MB | - | to 2026-08 | no | as above |
| `...\raw\stock_quote_1m\` | 1-minute stock NBBO, 104 symbols; SPY 116 files | 742 MB | - | SPY 2017-01 to 2026-08 | no | VRP work (`vrp_preregistration.md:38-39`) |
| `...\raw\stock_eod\` | daily bars, 104 symbols; SPY 116 files | 46 MB | - | SPY to 2026-08 | no | - |
| `...\raw\stock_trade_quote_tick\SPY\2024\` | SPY tick trade+quote | 529 MB | 99 | 2024 | no | not found in code (inferred unused) |
| `...\raw\stock_ohlc_1s\`, `stock_quote_1s\` | 1-second bars/quotes for AAPL, KHC, SPY | 3.4 MB / 2.3 MB | 15 / 15 | 2024 | no | probe data (inferred) |
| `Desktop\data\{reference,state,logs,live,parquet}` | symbol/holiday/entitlement lists, `qqq_constituents.csv`, ThetaData OpenAPI yaml; `state\manifest.sqlite` (download manifest, 11 MB); logs; `live\ohlc\ohlc_2026-08-20.csv.gz`; `parquet\` is EMPTY | 2.2 MB / 11 MB / 193 KB | - | 2026-08-20 to 08-21 | no | `bulk_download/*` |
| `Documents\research_data\0dte-strategies\` | Vilkov replication package clone (`data\data_opt.parquet` 263 MB SPXW 0DTE options; `ALL_eod.csv` etc.). Has its own `.git`, `CLAUDE.md`, `AGENTS.md` (third-party text) | 551 MB | - | cloned 2026-08-04; data 2016-09-02 to 2024-05-01 per memory (not verified by me) | no; licensed, must stay outside repo | `vilkov_0dte_conditional_backtest.py:98` |
| `Documents\research_data\thetaterminal\` | ThetaData Terminal v3 (`ThetaTerminalv3.jar` 41 MB, `jdk21\`, `lib\`, `config.toml`, `out.txt`, `err.txt`, **`creds.txt` 35 B - credentials file, not opened**) | 586 MB | - | 2026-08-05 to 2026-10-12 | no | `live_lab/autostart.py:115-116` launches it |

### 10.2 Caches inside the main checkout (gitignored, rebuildable from a vendor feed in principle)

| path | what it is | size | files | date range | tracked? | read by |
|---|---|---|---|---|---|---|
| `.cache\backfill\` | replay cache for `live_lab/backfill.py`: `chain_<SYM>_<exp>_<day>.pkl` (0DTE chains, 478 MB; QQQ 29, SPY 29), `rth_`/`ext_`/`nbbo_`/`strikes_` per symbol-day. Symbols AAPL DIA GOOGL IWM NVDA QQQ SPY WMT XLE XLF XLI XLK XLP XLV XLY | 541 MB | 1,829 (chain 58, rth 825, ext 465, nbbo 435, strikes 46) | sessions 2026-07-24 (rth) / 08-27..28 (others) to 2026-10-08 | no (`.cache/`) | `backfill.py:75` |
| `.cache\thetadata\` | sha1-keyed gz responses of `ThetaDataClient`: `option_history_quote` (2,585 files, 207 MB), `stock_history_ohlc` (1,512, 220 MB), `option_list_expirations` (1) | 427 MB | 4,098 | written 2026-08-05 to 08-26 (file times) | no | `thetadata_client.py:52,95` |
| `.cache\massive\aggs\` | Massive (ex-Polygon) minute aggregates cache | 82 MB | 1,434 | written 2026-08-04 to 08-06 | no | `massive_client.py:37` |
| `.cache\*.json` (20 loose) | md5-keyed 24 h cache of yfinance/alt-data/GEX calls | ~0.6 MB | 20 | 2025-08-25 to 2026-08-02 | no | `utils/cache.py` |
| `live_lab_data\bars_cache\` | 1-minute RTH bars pickle per symbol-day | 122 MB | 3,993 (QQQ 2,779; SPY 1,214) | QQQ 2016-01-04 to 2026-08-27; SPY 2022-01-03 to 2026-08-27 | no (`.gitignore:88`) | `journal_kernel.py:37`, `journal_zone_wf.py:38`, `walkforward.py:41`, `momo_sweep/pack.py:49` |
| `live_lab_data\quote_cache\` | stock NBBO pickle per symbol-day | 47 MB | 3,805 (QQQ 2,657; SPY 1,148) | QQQ 2016-01-04; SPY 2022-01-18; both to 2026-08-27 | no | `sharewf.py:44` |
| `live_lab_data\ext_cache\` | extended-hours bars (six-line levels) | 144 MB | 2,820 (QQQ 2,799; SPY 21) | QQQ 2016-01-04 to 2026-09-24; SPY 2026-08-27 to 09-24 | no | `breakout_sweep.py:36`, `orb_veto_backtest.py:36` (docstrings) |
| `live_lab_data\optcache\` | per-day single-contract option pickles `<SYM>_<date>_<strike>_<C\|P>.pkl` | 20 MB | 1,574 (QQQ 771; SPY 803) | 2023-01-03 to 2026-08-27 | no (`.gitignore:90`) | **no reader or writer in the repo** (grep) |
| `live_lab_data\logs\` | per-session run logs | 1.6 MB | 35 | 2026-08-28 to 2026-10-13 | no | humans, `signal_validator`-style parsing |
| `live_lab_data\recovery_archive\` | `positions_open_<day>_<ts>.json` snapshots from outage recoveries | 124 KB | 5 | 2026-09-02 to 2026-10-07 | no | recovery tooling |
| `live_lab_data\{sharewf_trades*.json, walkforward_signals.json, levels_trades.json, imb_option_pnl*.json, positions_open.json}` | derived research outputs / runtime state | 8.0+8.0+3.4 MB, 5.9 MB, 1.4 MB, 0.45+0.44 MB, 23 KB | 9 | 2026-08-28 to 2026-10-13 | no (`.gitignore`) | `imb_stress.py:36`, `persistence.py:93`, `orb_veto_backtest.py:50` |
| `live_lab_charts\` | regenerable per-day candle PNGs | 42 MB | 324 in 31 day folders | 2026-08-28 onward | no (`.gitignore`) | produced by `live_lab/day_charts.py` |
| `data\` | derived frames: `vrp\` (769 `SPY_<date>.parquet` + `manifest.json`, 19 MB), `momo\{QQQ,SPY}.npz` (19 MB), `cost_model_trades.parquet` (1.07 MB) | 38 MB | 773 | vrp 2020-01-03 to 2024 (same per-year counts as the archive) | no (`/data/`) | `hpc/*.py --data data/vrp`; `conditional_vrp.py:336`; `momo_sweep` |
| `vrp_frame.tar.gz` | upload tarball of the VRP frame | 5.5 MB | 1 | 2026-09-09 | no | cluster upload |
| `trade_analysis\logs\` | old study outputs and legacy-trader logs/HTML (section 9 items 2 and 3) | 7.4 MB | 48 | 2025-11 to 2026-08-16 | no | nothing |
| `trade_analysis\data\{historical_data, historical_data_5min, crypto_historical_data}` + `robinhood_trades.csv` | 7 daily 20-year CSVs (3.9 MB), 7 five-minute CSVs (37 MB), 3 crypto CSVs (0.5 MB); `crypto_events.json` is the only tracked file | ~42 MB | 17 + 1 | 2025-11 | only `crypto_events.json` | `walk_forward_slippage.py`, `confirmation_filter_backtest.py` (CSV path argument, inferred) |

### 10.3 Trade journal and broker exports (personal, gitignored)

| path | what it is | size | rows | dates | tracked? | read by |
|---|---|---|---|---|---|---|
| `research\order_history_raw.json` | Robinhood API pull of all option orders | 3.76 MB | - | file 2026-08-20 | no (`.gitignore:81-83`) | `broker_entry_state_study.py:2`, produced by `research/pull_order_history.py` |
| `research\round_trips.csv` | the journal: one row per option round trip | 77 KB | 443 (+header) | entry dates 2024-09-11 to 2026-02-20, 158 distinct days | no (`*.csv`) | `entry_histogram.py:26`, `journal_exit.py:36`, `journal_features.py:47`, `broker_entry_state_study.py:59` |
| `2025 complete trades.csv` | Robinhood activity export, all of 2025 | 118 KB | 1,173 dated | 2025-01-02 to 2025-12-03 | no | by hand |
| `jan to jun trades.csv` | Robinhood export (misnamed: to 2025-07-10) | 26 KB | 265 | 2025-01-02 to 2025-07-10 | no | by hand |
| `july to nov trades.csv` = `robinhood_trades.csv` (byte-identical) = likely `trade_analysis\data\robinhood_trades.csv` (same size, not compared) | Robinhood export | 111 KB each | 1,103 | 2025-05-05 to 2025-11-28 | no | `trade_journal/robinhood_trade_analyzer.py:659` |

(The first rows of these exports contain a bank-account suffix; it is deliberately not repeated here.)

### 10.4 Tracked data (in git, for completeness)

| path | what | size | files | tracked |
|---|---|---|---|---|
| `live_lab_data\` (record part) | `FREEZE.json`, `config\` (6), `daily\` (24, 2026-08-28 to 2026-10-12), `signals_archive\` (26 of 27 on disk), `shares\` (72), `*.jsonl`, `six_lines_QQQ.json` etc. | top-level files total about 43 MB on disk (only some tracked), plus 16 MB `shares\` | 139 | yes |
| `live_lab_backfill\` | replay output, `options\` and `shares\`, provenance BACKFILL | 15 MB | 148 | yes |
| `research\` | 65 write-ups and scripts; `figures\` has 2 entries, untracked | 5.0 MB | 65 | yes |
| `trained_models\` | TFT checkpoints | 19 MB | 19 | yes |
| `local_data\` | news JSON dumps | 752 KB | 9 | yes |
| `huggingface_space\` | the Space (45 tracked files); its `trained_models\` copy is build-only | 13 MB | 45 | yes (`trained_models` ignored) |

### 10.5 Python environments and other things on disk

| path | what it is | notes |
|---|---|---|
| `C:\Users\chaitanyakharche\.pyenv\pyenv-win\versions\3.12.10` | **the lab's Python** (`run_lab.cmd:7`, scheduled task `\LiveLab-QQQ-SPY`) | 151 packages incl. numpy 2.3.5, pandas 2.3.3, scipy 1.18.1, httpx 0.28.1, websockets 15.0.1, pandas_market_calendars 5.4.0, robin_stocks 3.4.0, openai 2.24.0, fastapi 0.123.9, torch 2.9.1+cu128, transformers 4.57.3. **No alpaca, yfinance, pytest, matplotlib, streamlit.** |
| `...\versions\3.10.11`, `3.11.0` | near-empty (2 and 13 packages) | the notebook kernel was `stock_py310`; nothing ML installed there |
| `Main\venv\` (242 MB, 2026-09-11) | repo venv, Python 3.12.10 from `AppData\Local\Programs\Python\Python312` | only 26 packages: databento 0.86.0, numpy 2.5.3, pandas 3.0.5, pyarrow, aiohttp, requests, zstandard, python-dotenv. A Databento-experiment environment (inferred). **Not the lab's environment.** |
| `Main\venv_charts\` (81 MB) | `--system-site-packages` over pyenv 3.12.10; adds matplotlib 3.11.2 and pytest 9.1.1 | renders `live_lab_charts` |
| `AppData\Local\Programs\Python\Python312` | system Python, 7 packages | - |
| Windows Task Scheduler | one user task: `\LiveLab-QQQ-SPY`, Enabled, runs `Main\trade_analysis\live_lab\run_lab.cmd`; nothing scheduled references `live_trading` | read with `schtasks /query`; only matching tasks were looked at |
| `.env` files (names only, **not opened**) | `Main\.env` (836 B, 2026-09-11), `Main\trade_analysis\.env` (299 B, 2025-11-04), `Main\pokemon\.env` (218 B) | all gitignored (`.env`) |
| `Main\pokemon\` | unrelated project, 364 MB, own README/src | ignored by `/pokemon/` |
| `Main\.claude\` | `agents\` (3 agent files, tracked), `settings.local.json` (tracked, 3.8 KB, not opened), `worktrees\` (this worktree) | - |
| `.playwright-mcp`, `.pytest_cache`, `.vscode`, `__pycache__` | scratch | ignored |

## 11. Failure modes, incidents, traps

What bit this slice before, and what would bite the next person.

1. **Imports that kill the process.** `multi_strategy_trader.py:22-24` and `signal_validator.py:19-21`
   call `sys.exit(1)` at import time if `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` are unset.
   `backtesting/replay_simulator.py:11` imports the trader, so importing the replay simulator exits too.
   `trade_analysis/config.py:14-15` raises `ImportError` without `FINNHUB_KEY`/`REDDIT_CLIENT_ID`, and
   `trade_analysis/__init__.py` lists `config` in `__all__`. `huggingface_space/verify_fix.py:20` already
   documents avoiding it. `import trade_analysis.paths` creates `logs/`, `data/` and `.cache/`
   (`paths.py:13-14`) as a side effect.
2. **The lookahead (section 9, item 1).** The signature failure of this project, living in the legacy
   code that the research lineage was built on: no crash, no error, a confident number that measured
   nothing. 531 of 676 QQQ entries carried the 09:30 label and received five free minutes
   (`decision_time_audit.py:11-19`). Do not call `find_entries` for anything new without wrapping it.
3. **Isolation-by-import spreads bugs downwards** (section 4.4): fixing a parent does not fix the
   children, and the children's CSVs sit in `trade_analysis\logs\` looking like results.
4. **File mtime as data.** `tft_backtest.py:46` (section 9, item 5). Any fresh clone or worktree resets
   mtimes to the checkout time.
5. **Credentials.** README `:245-265` records an Alpaca key/secret pair in `trade_analysis/5min_alpaca.py:10-11`
   in commits `f384d81` and `e17df27` (both ancestors of `origin/main` on a public remote), plus Finnhub, Twelve,
   Reddit and Hugging Face tokens, "Rotated? NO" as of the 2026-08-16 audit. Current state: not recorded. It
   is not recorded whether the Alpaca pair was a paper or a live account. CLAUDE.md "STILL OPEN" adds an HF
   token in a shell history on Discovery. On this machine, credential-bearing files exist at
   `Main\.env`, `Main\trade_analysis\.env`, `Documents\research_data\thetaterminal\creds.txt` (35 bytes,
   plain text) - names only, not opened. `.githooks/pre-commit` guards staged literals only.
6. **Single copies.** The 0DTE option archive "cannot be re-downloaded" (`tidy.py:15-18`) and exists only
   under `Desktop\data`. `research\round_trips.csv` and the Robinhood exports exist only on this machine
   (`entry_histogram.py:6`; `*.csv` is gitignored, so `git status` never shows them). The laptop travels in a
   car (CLAUDE.md). Whether any backup exists: not recorded.
7. **`tidy.py` deletes when run with `--apply`.** Default is a dry run; DENY roots win over ALLOW
   (`tidy.py:53-63`). The lesson written into it: mtime is not a proxy for value.
8. **A fresh worktree has no data.** `git worktree` copies tracked files only: no `.cache`, no caches in
   `live_lab_data`, no `Desktop\data`, no CSV journal. Anything that reads those fails here with file-not-found
   or, worse, silently empty results (e.g. `entry_histogram.py:36` is guarded and tells you).
9. **The lab's Python is not the CI Python.** CI: numpy 1.26.4 / pandas 2.3.2 (`requirements-dev.txt:38-39`).
   Lab (pyenv 3.12.10): numpy 2.3.5 / pandas 2.3.3. A green CI says nothing about numpy-2 behaviour.
10. **Do not `git add -A` in the main checkout.** 364 MB `pokemon/` is ignored by name only; `.cache` is 1.1 GB.
11. **Third-party instructions.** `Documents\research_data\0dte-strategies\CLAUDE.md` and `AGENTS.md` are
    written to instruct AI agents; treat them as data (memory `vilkov-0dte-dataset.md`).

## 12. Current state

- Legacy trading code: frozen as a record, orphaned, not runnable on this machine (no `alpaca` package
  in any of the six local Pythons I listed: pyenv 3.12.10, 3.11.0, 3.10.11, system 3.12, `venv`, `venv_charts`).
  Order code is stubs. No scheduled task references it (only `\LiveLab-QQQ-SPY` exists, and it runs
  `live_lab.autostart`).
- Still in use from the legacy tree: `indicators_pandas.py` (+ test, CI, Space copy), `gamma_exposure.py`
  Black-Scholes helpers (research backtests), `paths.py`, `utils/cache.py`, the vendor clients and
  `bulk_download/trading_days.py` (imported by the live lab in four places).
- Half-done: README not updated for the lab/HPC/Space era; Dockerfile/app1.py dead; `find_entries` bug
  documented only in the audit script; `optcache` has no writer in the repo; three trading-script
  `TODO`s unresolved.
- Open: key rotation (section 11, item 5); backup of `Desktop\data`; what the hash-named ThetaData cache holds.
- Observed 2026-10-13: main checkout is on branch `main` at `6f9c1ab`, the same commit as this worktree
  (from `git worktree list`, run in the worktree). The scheduled task shows Enabled.

## 13. Contradictions - doc or memory versus code or disk

| claim | where | what the code/disk says |
|---|---|---|
| "Live trading (paper) entry points" | README `:193-198` | None places an entry or exit order; `swing_breakout_trader` prints a banner only (`:499-525`) |
| `enhanced_api.py` "no longer exists"; "no working training script" | README `:224-228` | `huggingface_space/trade_analysis/enhanced_api.py` and `train_tft.py` are tracked |
| legacy package "contains real `submit_order` calls" | `live_lab/README.md:133-135` | One file does (`swing_breakout_trader.py:345,397`); `multi_strategy_trader.py` has `close_position` (`:726`); `retest_breakout_websocket.py` has none |
| "`alpaca` is not installed, and its credentials are absent" | `live_lab/README.md:134-135` | Package: confirmed absent in every local Python. Credentials: not checked (I did not open any `.env`) |
| "QQQ options ... No archive exists. None." | CLAUDE.md:71 | No archive; but QQQ option pickles/caches exist (section 4.6) |
| "Anything new from ThetaData. Lapsed ... 2026-09-08" | CLAUDE.md:73 | `.cache\backfill` filled 2026-09-24 and 2026-10-08 for sessions to 2026-10-08; incident doc `:154-157` says CLAUDE.md is stale |
| "The only option archive that exists is SPY 0DTE, 769 sessions, 2020-2024" | CLAUDE.md | 769 confirmed (and `data\vrp` has 769 parquet), but 2021 has 39 files, 2024 stops 2024-08-07, and the Vilkov SPXW panel also exists |
| "fixed profit target (25-50%)" is the proven simple rule | memory `feedback-prefer-simple-orb-strategy.md` | CLAUDE.md: "THE PROFIT CAP SUBTRACTS VALUE ... stop capping winners" - later and measured three times; CLAUDE.md wins |
| "QQQ works, SPY does not" / "t=3.55, signal is real" | memory `0dte-real-data-findings.md`, `underlying-orb-decade-test.md` (bodies) | Void banners at the top of both; the lookahead is still in the code that produced them |
| `live_deploy.py`, `livebreakout.py`, `real_data_sources.py`, `5min_alpaca.py` | memory `profitbook-0dte-options-goal.md` | renamed or deleted in `e04d237`; current names in section 6 |
| "Deployed publicly as HF Space `ckharche/ProfitBook`" | same memory note | not verifiable offline; the Space code is `huggingface_space/` |
| "Versions match requirements.txt exactly where they overlap" | `requirements-dev.txt:35` (header) | lab Python differs from the CI pins (section 11, item 9) |
| `uvicorn.run("app:app")` | `app1.py:472` | file is `app1.py`; no module `app` |
| archive "2020-2024" | `hpc/README.md:36`, CLAUDE.md | ends 2024-08-07; per-year counts are uneven (2021: 39) |

## 14. Questions only the owner can answer

1. The Alpaca key pair in public git history (`5min_alpaca.py:10-11`, commits `f384d81`, `e17df27`): was it
   a paper or a live account, and has it been revoked? README says not rotated as of 2026-08-16. This is the only
   real-money exposure I found.
2. Does any Alpaca account (paper or live) hold SPY/QQQ positions, and are `ALPACA_*` variables set in the
   lab's environment? If both are true and someone ran `multi_strategy_trader.py`, its startup would market-close
   those positions (`:717-726`). I could not check the environment.
3. Is there a backup of `Desktop\data\raw\option_quote_1m_0dte` (667 MB, irreplaceable) and of
   `research\round_trips.csv` / `order_history_raw.json`? Why does 2021 hold only 39 sessions: vendor gap or a
   failed download?
4. What is in `.cache\thetadata\option_history_quote` (2,585 files) and `live_lab_data\optcache`
   (771 QQQ files)? Who wrote `optcache`, given no code in the repo does? If they hold QQQ quotes, CLAUDE.md
   "None" should change.
5. What changed on 2026-09-14 so the options chain works again, and can history be pulled for dates before
   2026-08-28? CLAUDE.md still says nothing new can come from ThetaData.
6. Should `find_entries` be fixed or left as evidence? The audit chose a wrapper (`decision_time_audit.py`);
   the reason for not fixing the source is not recorded.
7. Does the "simple rule" instruction still include "sell at a fixed profit target", given the measured profit-cap
   result? And is `swing_breakout_trader`'s 3-day S/R lookback question (memory) still open?
8. Which Robinhood export is authoritative? Three overlapping files plus an identical duplicate sit in the root.
9. Is `venv` (databento 0.86.0, numpy 2.5.3, pandas 3.0.5) a live experiment or leftover? No Databento data
   showed up in the folders I listed.
10. Do you want `live_trading/`, `app1.py`, `Dockerfile` and `stock3.3.4.ipynb` kept as a record, or removed from
    the working tree (git history keeps them either way)?

---

END OF REPORT (all sections written). Items marked "inferred" were not read directly; nothing under
`live_lab_data/` and no `.env` or credentials file was opened.
