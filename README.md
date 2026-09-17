# Quantitative Finance Trading System

## Overview

Personal research + live-trading codebase for a 0DTE SPY/QQQ retest-breakout
options strategy, with supporting backtests, alt-data collection, a dealer
gamma-exposure (GEX) signal, and ML research (LightGBM/PatchTST notebook,
reconstructed TFT checkpoints).

---

## Project layout

```
stock/
├── app1.py                    # Standalone FastAPI/data-fetch service (news, sentiment, OHLCV)
├── stock3.3.4.ipynb           # ML research notebook (LightGBM + PatchTST, MSFT/AAPL/GOOGL)
├── trained_models/            # Trained TFT checkpoints (.pth/.joblib) - see caveats below
├── local_data/                # Collected alt-data JSON per symbol (news/reddit)
├── .cache/                    # Shared disk cache (trade_analysis/utils/cache.py)
└── trade_analysis/
    ├── config.py               # API keys / env config
    ├── paths.py                # Shared LOGS_DIR/DATA_DIR/CACHE_DIR/TRAINED_MODELS_DIR constants
    ├── utils/
    │   └── cache.py             # Generic disk cache (used by data_sources + signals)
    ├── data_sources/            # Data ingestion
    │   ├── unified_data_provider.py    # Alt-data (news/reddit/VIX/sector), was data.py
    │   ├── collect_alt_data.py         # CLI: dump alt-data to local_data/, was collect_data.py
    │   ├── download_5min_alpaca.py     # Alpaca 5-min OHLCV bulk download, was 5min_alpaca.py
    │   ├── download_daily_yfinance.py  # yfinance daily OHLCV bulk download, was download_historical_data.py
    │   ├── crypto_history.py           # Crypto event/history analyzer
    │   ├── crypto_live_polling.py      # Live crypto WS predictor, was crypto_trading_polling.py
    │   └── forex_signal_generator.py   # EUR/USD, GBP/JPY, USD/JPY signals, was crypto_live_monitoring.py (misnamed)
    ├── signals/
    │   └── gamma_exposure.py    # Dealer GEX regime + gamma-strike levels (yfinance option chain)
    ├── live_trading/             # Production/live entry points
    │   ├── swing_breakout_trader.py       # Single-symbol swing validation layer, was livebreakout.py
    │   ├── multi_strategy_trader.py       # SPY/QQQ multi-strategy live trader, was live_deploy.py
    │   ├── retest_breakout_websocket.py   # WebSocket retest system, was live_deploy_v2.py
    │   └── options_strategy_engine.py     # Options strategy scaffolding, was momentum_trading_engine.py
    ├── backtesting/
    │   ├── multi_strategy_backtest.py     # `backtesting` lib param sweep, was 5min_backtest.py
    │   ├── walk_forward_nautilus.py       # nautilus_trader walk-forward engine
    │   ├── walk_forward_slippage.py       # Walk-forward with slippage/commission modeling, was walkforwardlive.py
    │   ├── confirmation_filter_backtest.py # Proves the AM confirmation filter doesn't hurt edge, was confirmation.py
    │   ├── signal_validator.py            # Post-mortem log parser -> HTML report
    │   └── replay_simulator.py            # Replays multi_strategy_trader against a historical day
    ├── models/
    │   ├── tft_model.py          # Reconstructed GapPredictionTFT architecture (see caveats)
    │   └── tft_backtest.py       # Out-of-sample backtest of the trained_models/ checkpoints
    ├── trade_journal/
    │   ├── robinhood_trade_analyzer.py    # Parses Robinhood CSV exports
    │   ├── trade_input_parser.py          # Converts a simple trade log into trade_journal_analyzer's format
    │   └── trade_journal_analyzer.py      # Manual TRADES list -> technicals-at-entry report
    ├── logs/                     # Generated logs, trade CSVs, HTML reports (gitignored patterns apply)
    ├── data/                     # Cached historical/crypto data pulled by data_sources scripts
    └── scripts/
        └── clean_trade_venv.sh   # HPC conda env cache cleanup
```

---

## Running the tests

You do not need the trading stack to run the tests, and you should not install it just to
run them. `requirements.txt` is a full `pip freeze` of the trading machine — torch,
triton, the entire nvidia-cu12 CUDA stack, several GB of wheels — so it used to be that
the only host where any test could run was the same Windows box that has to collect a
live session every weekday. On a fresh clone every test failed identically with
`ModuleNotFoundError: No module named 'httpx'`, which looks like the suite is broken and
actually means nothing is installed.

```
pip install -r requirements-dev.txt     # ~150 MB, no CUDA, any OS
pytest                                  # trade_analysis/, offline, 76 checks
```

`pytest.ini` teaches discovery this repo's `<thing>_test.py` naming — without it, bare
`pytest` collected zero tests and exited 5, which at a glance is indistinguishable from a
suite that ran clean. The same set runs on every push via `.github/workflows/tests.yml`.

The house-style tests also run standalone and print their own PASS/FAIL report:

```
python -m trade_analysis.live_lab.ledger_test           # coverage denominator
python -m trade_analysis.live_lab.preflight_gate_test   # preflight <-> autostart contract
python -m trade_analysis.live_lab.bar_cache_test        # cache cannot change a fill
python -m trade_analysis.indicators_pandas_test         # indicators match the live lab
python -m trade_analysis.live_lab.feed_resilience_test  # WiFi <-> hotspot transitions
python -m trade_analysis.live_lab.archive_test          # end-of-session commit
python -m trade_analysis.live_lab.orb_veto_test         # range-expansion veto
python -m trade_analysis.live_lab.orb_veto_backtest_test
```

**Two exclusions, both deliberate.** `huggingface_space/*_test.py` fetch live quotes
through yfinance, so they are integration checks against a third party — run them by
hand, because a network flake must never read as a code failure.
`trade_analysis/backtesting/*_backtest.py` are research scripts needing the local
ThetaData archive; they are not tests despite the filenames.

**The live lab archives itself.** `autostart` commits `live_lab_data/` at the end of
every session and pushes it, on by default (`--no-archive` / `--no-push` to opt out). It
stages an explicit pathspec, never `git add -A`, so it cannot commit code or `.env`; it
makes no commit when there is nothing to say; and a rejected push is left rejected rather
than rebased — the commit is local and the next successful push carries it. This exists
because committing by hand was the plan and the plan produced a six-session hole
(`research/live_lab_coverage_audit.md`). To backfill one by hand:
`python -m trade_analysis.live_lab.archive --day 2026-09-17`.

**A platform-gated check is SKIP, never FAIL.** `preflight_gate_test` verifies a Windows
Firewall contract; off Windows it skips that one assertion and still exits 0. Scoring an
unrunnable check as a failure is how "all the tests fail" becomes the normal state and
stops meaning anything.

## Setup

> **`pandas_ta` has been dropped** (2026-09-17), because it was **deleted from PyPI** —
> the entire 0.3.x release history was withdrawn and the package changed maintainer, so
> `pip install -r requirements.txt` failed outright:
>
> ```
> $ pip download --no-deps 'pandas_ta==0.3.14b0'
> ERROR: Could not find a version that satisfies the requirement pandas_ta==0.3.14b0
>        (from versions: none)
> ```
>
> The only remaining releases need Python >= 3.12 and numpy >= 2.2.6, conflicting with
> this repo's `numpy==1.26.4`; and 20 modules of the old version do `from numpy import
> NaN`, an alias numpy 2.0 expired. No available version worked. **This is a large part
> of why a working checkout could not be reproduced anywhere** but the one machine whose
> venv predated the deletion.
>
> Replaced by **`trade_analysis/indicators_pandas.py`** — ema, rsi, atr, adx, macd,
> bbands, vwap, ported from the Wilder implementations already in
> `huggingface_space/trade_analysis/indicators.py`. Its numbers are pinned against
> `trade_analysis/live_lab/indicators.py` (the dependency-free code the live lab actually
> trades on) by `indicators_pandas_test.py`, at 1e-12 relative.
>
> **One deliberate deviation from pandas_ta:** `ema` is seeded with an SMA of the first
> `length` values, matching the live lab, not from the first value as
> `ewm(adjust=False)` does. The live lab's definition wins because `replay.py` proves
> live and batch agree on it and every trade in `live_lab_data/` was taken under it.
> Keeping two EMAs in one repo to match a library that no longer exists is the worse
> trade.

> The root **`Dockerfile` is dead** and separate from the above: it does `COPY app app`
> and runs `uvicorn app.api:app`, but there is no `app/` directory in this repo. The
> deployed Space builds from `huggingface_space/`, which has its own Dockerfile and
> requirements. Delete it or point it somewhere real.

### Windows (Local Development)

1. **Prerequisites**
   - Python 3.11+
   - Miniconda/Anaconda

2. **Environment Setup**
```
cd C:\path\to\your\repo
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. **Fix Windows Encoding (REQUIRED)**
```
chcp 65001
$env:PYTHONUTF8="1"
```

4. **Environment Variables**
```
$env:FINNHUB_API_KEY="your-finnhub-key"
$env:TWELVE_KEY="your-twelve-data-key"
$env:REDDIT_CLIENT_ID="your-reddit-client-id"
$env:REDDIT_CLIENT_SECRET="your-reddit-secret"
$env:REDDIT_USER_AGENT="script:stock-opinion-analyzer:v1.0 (by /u/YourUsername)"
$env:ALPACA_API_KEY="your-alpaca-key"
$env:ALPACA_SECRET_KEY="your-alpaca-secret"
```

5. **Data Collection**
```
python -m trade_analysis.data_sources.collect_alt_data --symbol MSFT
python -m trade_analysis.data_sources.collect_alt_data --symbol QQQ
python -m trade_analysis.data_sources.collect_alt_data --symbol SPY
```

6. **Live trading (paper) entry points**
```
python -m trade_analysis.live_trading.swing_breakout_trader
python -m trade_analysis.live_trading.multi_strategy_trader
python -m trade_analysis.live_trading.retest_breakout_websocket
```

---

## TFT models (`trained_models/`) — important caveats

`trained_models/*.pth` / `*.joblib` are checkpoints from an earlier HPC training
run (`srun --gres=gpu:h100:1`) whose original model-definition source file was
lost. **Confirmed permanently unrecoverable on 2026-09-09:** it lived under
`/scratch/kharche.c/` on Discovery, which purges after 45 days of no access, and
that tree no longer exists. The checkpoints survive only because they were copied
into this repo. It costs little -- the model was measured and found degenerate --
but it is the reason nothing that matters is left on scratch again. `trade_analysis/models/tft_model.py` is a **reverse-engineered
reconstruction** of that architecture from the checkpoints' tensor shapes and
embedded config — the weights load correctly, but:

- The "static" branch (market cap/beta/sector/VIX/liquidity) was trained on a
  **constant placeholder**, not real varying data — it carries no signal.
- An out-of-sample backtest (`python -m trade_analysis.models.tft_backtest`)
  shows **no real directional edge** (~50% hit rate, gap-classifier accuracy
  at or below a trivial majority-class baseline on every symbol tested).

These checkpoints are kept for reference/reconstruction purposes, not as a
production-ready model. See `trade_analysis/models/tft_model.py`'s docstring
for the full reconstruction methodology.

There is currently **no working training script** for these models — the
original `train_tft.py` depended on the same lost model-definition file and
was removed as non-functional. `enhanced_api.py` (a FastAPI serving layer
referenced in earlier versions of this README) no longer exists in the
source tree either.

---

## `stock3.3.4.ipynb`

A LightGBM + PatchTST research notebook (MSFT/AAPL/GOOGL, daily bars, 5-day
direction classification). Kept as the single best-working iteration of a
larger family of near-identical notebook drafts (all superseded/removed).
Per literature review, raw price/technical-indicator direction prediction
has a well-documented ~50% ceiling — this notebook is exploratory, not a
production signal source. See the GEX-based signal in
`trade_analysis/signals/gamma_exposure.py` for the more evidence-backed
direction this project moved toward instead.

---

## Known outstanding issues

- **Credentials committed to PUBLIC git history — rotation still outstanding.**
  Audited 2026-08-16. The working tree is clean; the history is not, and every
  affected commit is an ancestor of `origin/main` on the public remote, so
  "local git history" in the earlier note was wrong.

  | Credential | Where in history | Rotated? |
  |---|---|---|
  | `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | `trade_analysis/5min_alpaca.py:10-11` in `f384d81`, `e17df27` | **NO — do this first, it is a brokerage credential** |
  | `FINNHUB_API_KEY` | `README.md`, `clean_trade_venv.sh` (14 occurrences) | NO |
  | `TWELVE_KEY` | same | NO |
  | `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | same | NO |
  | `HF_TOKEN` | `README.md` in `f384d81`, `e17df27` | NO |

  Also present in history: `trade_analysis/slurm-1612669.out`, a job log that
  captured exported env vars. Both that file and `5min_alpaca.py` are deleted
  from the tree but remain in history.

  Verified clean, no action needed: `.env` was never tracked, and no ThetaData
  (`td1_prod_*`) or Massive key literal appears anywhere in history.

  History was **deliberately not rewritten**. Once a key is published, rotation
  is the only real remediation — a rewrite does not un-leak anything already
  scraped, and GitHub retains old objects in cached views until Support purges
  them. Current source correctly reads from env vars
  (`download_5min_alpaca.py:12-13`).

  Prevention is in place: `.githooks/pre-commit` blocks staged credential
  literals. Enable per clone with `git config core.hooksPath .githooks`.
- Several live-trading scripts (`live_trading/`, `backtesting/replay_simulator.py`,
  `trade_journal/*`) depend on `alpaca-py`, `ta`, and other packages not
  necessarily installed in every environment — check `requirements.txt`.
