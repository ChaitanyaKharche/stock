# 05 — The Hugging Face Space ("ProfitBook")

Written 2026-10-13 from a static read of the worktree at commit `6f9c1ab`. Nothing was run,
nothing was fetched from Hugging Face, and the main checkout was not touched. Every claim
about code carries a `path:line` that was read. Anything worked out rather than read is
marked **(inferred)**. Paths are relative to the repo root unless they start with a drive.

Where the eight slice questions are answered:

| question | section |
|---|---|
| 1. What a visitor sees, and what each number comes from | 3.2 |
| 2. What the Space is for, and how the message changed | 2 |
| 3. Architecture (entry point, port, data, keys, cache, LLM, sentiment, TFT, HAR) | 3.1, 3.3 |
| 4. Deployment (deploy.py, sync_shared.py, copies, correct sequence) | 3.4, 5 |
| 5. The silent-failure bugs, one by one | 7.1 |
| 6. What the calibration tools did and the thresholds they set | 7.2 |
| 7. Secrets hygiene and the exposed HF token | 7.4 |
| 8. Dead or misleading code | 7.3 |

---

## 1. In one paragraph

`huggingface_space/` is a self-contained web app, published as a public Hugging Face
"Space" (a hosted container) at `https://huggingface.co/spaces/ckharche/ProfitBook`
(`research/PROJECT_REPORT.md:271`). It began as a signal generator: type a ticker, get
`CALLS` / `PUTS` / `HOLD` with a confidence number. In September 2026 that app was found to
be returning the same constant for every ticker, was repaired defect by defect, and then —
once the research programme concluded that intraday direction cannot be predicted with
these tools — was turned around: it now opens on what the research found, shows the
paper-trading forward test and a gallery of the bugs, and keeps the original signal
generator as the last tab, labelled a demo (`huggingface_space/streamlit_app.py:22-26`,
`:686-687`). It shares no imports with the live lab; five paths are held as checked copies
instead (`huggingface_space/sync_shared.py:40-56`).

## 2. The goal it serves, and how it changed

**Three stages, each visible in the code.**

1. **Original (2025, inferred).** A "SOTA Financial AI" that predicts direction from
   multi-timeframe momentum, news and Reddit sentiment, a transformer model and a
   large-language-model ensemble. The self-description is still in the API metadata:
   `huggingface_space/trade_analysis/enhanced_api.py:74-78` and the feature list at
   `:150-157`. The fallback model directories are university paths
   (`huggingface_space/trade_analysis/tft_model.py:30-33`, one under a `CS6120` course
   folder) and the bundled news snapshots date from August–September 2025 (section 6), so
   this was a coursework-era project run on the cluster **(inferred)**. The repo root's
   `app1.py` is an even earlier prototype of the same idea (section 3.5).

2. **Engineering showcase (2026-09-06 → 09-11).** The stated purpose became "demonstrating
   engineering skill, not research" (`research/PROJECT_REPORT.md:272`). The memory note
   says more specifically "showing SDE skill to recruiters" — the word "recruiter" appears
   nowhere in the repo, only in that note. In this stage the app was audited and each
   silent defect fixed and written up in `huggingface_space/README.md:83-444`.

3. **Research showcase (2026-09-24, commit `25cd841`).** The commit message gives the
   reason: "The Space opened on a BUY/HOLD generator -- the one thing a year of
   pre-registered work found no edge in. It now opens on the findings and keeps the engine
   as the last tab, labelled as the engineering demo it is." The page caption now reads
   "A research demo — not investment advice, and nothing here places an order"
   (`huggingface_space/streamlit_app.py:205-207`).

**Audience.** "Someone" who should understand the work "in minutes"
(`huggingface_space/streamlit_app.py:245`). Recruiters and interviewers, per the memory
note only.

**Does it still claim to predict anything?** The framing no longer does. The last tab says
outright that it "calls direction, which the research above found no edge in"
(`streamlit_app.py:446-449`). But three residues still read as prediction or advice:

- the analyzer still prints a "Suggested options structure" with contract counts, profit
  targets and stops (`streamlit_app.py:611-616`, built at `enhanced_api.py:762-815`) —
  including **spreads** for the 4h and 1d settings (`enhanced_api.py:798-815`) and a fixed
  profit target on every structure, which sits beside the research tab's own headline that
  capping winners lost money every time (`tools/build_research_snapshot.py:164-169`);
- an "Allocation … % of capital" tile of up to 50% when a signal fires
  (`enhanced_api.py:660`, `streamlit_app.py:591`);
- the API still describes itself as "SOTA Financial AI with multi-LLM ensemble"
  (`enhanced_api.py:77`). On the deployed CPU container no language model is loaded at all
  (section 3.3).

**The Space also fed the research.** The discovery that the momentum score measured size,
not direction, was made "by accident on 2026-09-08 while fixing the showcase app" and is
the stated origin of the variance experiment (`research/vrp_preregistration.md:13-22`;
diagram at `research/PROJECT_REPORT.md:341-364`: "the entire HPC experiment came from a
bug in the demo app").

## 3. How it works, end to end

### 3.1 Container start

1. Hugging Face reads the front-matter of `huggingface_space/README.md:1-9`: `sdk: docker`,
   `app_port: 8501`.
2. `huggingface_space/Dockerfile:1-23`: `python:3.12-slim`, model caches pointed at `/tmp`,
   `pip install -r requirements.txt`, copy the whole folder, run `./run.sh`.
3. `huggingface_space/run.sh:22` starts the API (FastAPI under uvicorn) on port **7860**,
   in the background. `run.sh:28-50` polls it for up to 45 × 2 s. If the process dies it
   writes the reason plus the last 25 log lines to `/tmp/api_status` (`run.sh:29-41`); if
   it answers it writes `ready` (`run.sh:43-48`). `run.sh:64` then starts the user
   interface (Streamlit) on port **8501**, whatever happened.
   *Why:* the old two-line script let a missing secret kill the API while the Space still
   showed "Running" and every click said "Failed to connect" (`run.sh:4-14`).
4. Importing the API imports `config.py`, which **raises** if `FINNHUB_KEY` or
   `REDDIT_CLIENT_ID` is unset (`trade_analysis/config.py:14-15`). So with no keys the API
   never starts, the status file says why, and the analyzer tab shows that text
   (`streamlit_app.py:450-454`). The four research tabs still work — they read one bundled
   file and need no key and no network.
5. API startup (`enhanced_api.py:87-133`): detect CPU/GPU (`:93-94`), load the sentiment
   model (`:98`), **skip** the language models on CPU (`:101-105`), load transformer
   checkpoints for six symbols (`:113-125`), and launch a background "trading agent"
   (`:127-131`; see 7.3).

Only port 8501 is exposed; the UI calls the API on `localhost:7860`
(`streamlit_app.py:42-44`). The memory note says the API is not reachable from outside
(external calls return 403) — not verifiable from code.

### 3.2 What a visitor sees, tab by tab

Title, caption, then five tabs (`streamlit_app.py:204-207`, `:686-697`). **Tabs 1–4 are a
frozen snapshot**: they read `huggingface_space/research_snapshot.json`
(`streamlit_app.py:53`, `:125-130`), generated `2026-09-24T22:53:01`
(`research_snapshot.json:2`). Nothing in them updates until the file is rebuilt and
re-uploaded. **Tab 5 is live**: it calls the API, which fetches market data on request.

**Tab 1 — "What I found"** (`streamlit_app.py:213-286`)

| what is on screen | where it comes from |
|---|---|
| Five headline cards (direction closed; mid-price backtests can have the wrong sign; one survivor on shares; capping winners loses; the big model was a constant) | Hand-written text and numbers in `tools/build_research_snapshot.py:148-175`, each with a source path. Not computed. |
| "How it is organised" — HPC / live lab / this demo | Static text, `streamlit_app.py:239-246` |
| Scatter of QQQ days coloured by "day type", plus a table of the trader's own journal results per type | `research_snapshot.json:1570` onward: 2,631 points and a 3-row summary, copied from `research/figures/day_clusters*.csv` (`build_research_snapshot.py:123-143`). The stability figures (0.996, 0.830, silhouette 0.236) are literals in that script (`:138-140`). The chart drops days whose open-to-close move exceeds 4% (`streamlit_app.py:256`) without saying so. |

**Tab 2 — "Live forward test"** (`streamlit_app.py:290-376`)

| what is on screen | where it comes from |
|---|---|
| Running since / option sessions / option trades / share trades | `forward_test` block, built from `live_lab_data/FREEZE.json`, `trades.jsonl` and `shares/trades.jsonl`, keeping only trades whose `config_hash` is in the accepted list and, for options, `arm == "ATM"` (`build_research_snapshot.py:80-106`) |
| "Not evidence yet" warning | Static text plus a note hard-coded in the builder (`build_research_snapshot.py:102-104`) |
| Bar chart of net dollars per session, options vs shares | `forward_test.daily`, 19 sessions 2026-08-28 → 2026-09-24 (`research_snapshot.json:512-608`) |
| Per-setup tables with `INSUFFICIENT` | `_per_setup` (`build_research_snapshot.py:60-77`): needs 200 trades, and for shares also a minimum session count, because "15 correlated names reach 200 raw trades in weeks" (`:61-63`). In the snapshot the shares arm's MOMO_CHASE has 367 trades and still reads `INSUFFICIENT`. |
| Six-line backfill table and five totals | `backfill` block from `live_lab_backfill/options/report.json` and `BACKFILL.json` (`build_research_snapshot.py:109-118`). The totals are summed in the UI over **only** the sessions the live options arm traded, so both sides cover the same days (`streamlit_app.py:349-365`). |

**Tab 3 — "Fill-price explorer"** (`streamlit_app.py:380-420`)

A slider from 0 (filled at the mid price) to 1 (pay the whole bid/ask spread). For each of
21 positions the builder computed `gross − f × spread_cost` over 37,517 short SPY 0DTE
("zero days to expiry") straddles in 618 sessions (`build_research_snapshot.py:34-55`;
`research_snapshot.json:116-251`). The tiles show mean dollars per straddle, a t-statistic
computed on each session's mean trade, and win rate. Endpoints: t = 7.52 at f = 0 and
−5.38 at f = 1 (`research_snapshot.json:127`, `:247`), matching
`research/vrp_cost_model_results.md:18-19`. The sentence about where significance ends and
where the mean turns negative is computed in the UI from the grid
(`streamlit_app.py:412-417`). The "worst 1% … 94% of the loss" line is a literal
(`streamlit_app.py:418-420`), true to `research/vrp_cost_model_results.md:53`.

**Tab 4 — "Silent-bug gallery"** (`streamlit_app.py:424-441`)

A ten-row table of defects and what each cost, and a 3 × 4 table of the transformer's
output in three opposite market regimes. Both are literals in the builder
(`build_research_snapshot.py:177-195`), sourced to `research/PROJECT_REPORT.md` §3.3
(`:296-303` there).

**Tab 5 — "Signal analyzer (demo)"** (`streamlit_app.py:445-683`)

Controls: a symbol box searching 285 bundled ticker/company names
(`streamlit_app.py:59-66`; `trade_analysis/symbols.json`), a timeframe (15m / 1h / 4h /
1d) and a strategy (momentum / gap / reversal) (`:73-79`). Results are cached in the UI
for five minutes per (symbol, timeframe, strategy) (`:139-150`).

| what is on screen | computed from | live / frozen / model |
|---|---|---|
| Market-open banner | Clock plus the bundled holiday table; drawn before any request (`streamlit_app.py:456-470`; `trade_analysis/market_session.py:64-119`) | live clock, bundled calendar |
| `CALLS` / `PUTS` / `HOLD` | `_generate_master_signal` (`enhanced_api.py:447-737`); see 3.3 | rule output on live bars |
| Price, % change, day range | Last 15-minute close; change against the previous daily close; high/low of the newest daily bar (`enhanced_api.py:307-321`) | live (yfinance) |
| "session … · newest bar …" and any lag warning | Calendar's expected session compared with the newest bar actually fetched (`enhanced_api.py:288-302`) | live |
| Confidence and "vs N gate" | Weighted mean of components against the per-timeframe gate (`enhanced_api.py:596-616`, `:420-445`) | rule output |
| Score and "needs ±x" | Signed score against the timeframe threshold (`enhanced_api.py:529-576`, `:658-667`) | rule output |
| Timeframe agreement | Share of timeframe weight on the same side as the average direction (`momentum_trading_engine.py:437-442`) | rule output |
| Blue "why it held" line | The one condition that actually blocked (`enhanced_api.py:673-701`) | rule output |
| "What each timeframe contributed" table | Per bar set: direction score, momentum score, RSI, ADX, trend, volatility regime (`streamlit_app.py:99-122`) | computed on live bars |
| Suggested options structure | A template picked by timeframe; quantity is confidence divided by 8, 12 or 15 (`enhanced_api.py:762-815`) | canned template |
| VIX | Last daily close of `^VIX` (`data.py:316-321`) | live |
| Implied vol | Backed out of the last traded prices of the at-the-money call and put on the nearest expiry (`data.py:373-397`) | live, from trade prices — no bid/ask |
| Put/call | Put open interest ÷ call open interest, nearest expiry only (`data.py:337-342`) | live |
| Sentiment score and its confidence label | One news-sentiment model over headlines and Reddit titles (`enhanced_sentiment.py:53-58`, `:392-431`) | **model output on frozen text for nine symbols** — see section 6 |
| HAR forecast, last realised, variance risk premium | A small regression on this symbol's own ~2 years of daily bars (`har_forecast.py:105-169`); premium = implied − forecast (`:159-162`) | model output on live bars |
| "Headlines the sentiment model read" | Up to five strongest-scoring texts (`enhanced_sentiment.py:433-459`) | as sentiment |
| "Gap model — … TFT that collapsed (does not vote)" | Raw transformer output, shown so the failure is visible (`streamlit_app.py:673-680`) | model output, known degenerate |
| Raw response | The whole API reply (`streamlit_app.py:682-683`) | — |

Two captions in this tab carry research numbers as literals: "618 sessions … t=+7.5 gross
… 1.84x too small … t=-5.4 net" (`streamlit_app.py:660-662`).

### 3.3 One analyzer request, in order

`POST /predict/enhanced/` (`enhanced_api.py:161-376`).

1. **Bars.** yfinance (a free Yahoo Finance client, no key) is asked for 60 days of
   15-minute bars, 180 days of hourly bars and 2 years of daily bars, in parallel threads
   (`data.py:27-31`, `:179-191`). Four-hour bars are not fetched — yfinance has no such
   interval — they are folded from hourly bars one session at a time so a bucket never
   spans the overnight gap (`data.py:36-61`, `:193-198`). A timeframe that comes back empty
   is left out, not stubbed (`data.py:187-191`). Only if everything fails does it fall back
   to a single Finnhub quote row, and it says so (`data.py:200-208`).
2. **News and Reddit.** If `local_data/<SYMBOL>_external_data.json` exists it is returned
   immediately, before the cache and before any API call (`data.py:211-215`, `:252-256`).
   Otherwise: Finnhub company news for the last 3 days using `FINNHUB_KEY`
   (`data.py:222-240`), and a Reddit search of three subreddits using `REDDIT_CLIENT_ID`,
   `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` (`data.py:242-281`). Results are cached on
   disk under `/tmp/.cache` (`cache.py:2`).
3. **Other market context** (`data.py:283-402`): VIX, sector (Finnhub profile), put/call
   ratio and implied volatility from the option chain. Anything unavailable is returned as
   `None` and noted in `data_quality`, "never as a plausible-looking number"
   (`data.py:309-310`).
4. **Indicators** per bar set (`enhanced_api.py:190-195`): `enrich_with_indicators` and
   `identify_current_setup` (`indicators.py:91-340`). ADX, ATR and RSI use Wilder's
   smoothing written in-house (`indicators.py:30-89`); the `import pandas_ta` at
   `indicators.py:115` is expected to fail and the `except ImportError` branch is the real
   implementation (`indicators.py:175-194`).
5. **Sentiment** (`enhanced_api.py:199-204`): one model,
   `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`; the four others in
   the config are commented out (`enhanced_sentiment.py:33-67`). Each headline is scored
   positive minus negative; a composite mixes news, Reddit and a keyword count
   (`enhanced_sentiment.py:392-408`); the label is HIGH / MEDIUM / LOW by how tightly the
   scores cluster (`:410-431`).
6. **Momentum engine** (`momentum_trading_engine.py:93-138`). For each bar set it computes
   a **magnitude** score — how much is happening, every term non-negative (`:298-340`) —
   and a separate **signed direction** score in [−1, +1] (`:274-296`). The selected
   timeframe picks a weight vector over the four bar sets (`:399-405`). Direction counts
   only if the weighted average clears a deadband of 0.40 **and** at least 75% of the
   weight agrees (`:44-45`, `:437-447`). Magnitude decides the band: above 0.6 strong,
   0.4 moderate, 0.25 weak, otherwise none (`:456-499`).
7. **Transformer (TFT)** (`enhanced_api.py:214-238`): computed if a checkpoint was loaded
   and there are at least 96 daily rows; never trained inside a request (`:219-224`). It
   does not vote (step 9).
8. **"LLM" layer** (`enhanced_api.py:240-256`). On a CPU machine no language model is
   loaded (`enhanced_api.py:101-105`), so `generate_enhanced_trading_signal` takes its
   no-models branch and calls the rule-based fallback with every condition except VIX
   **hard-coded False** (`enhanced_llm.py:123-132`). That fallback then always returns
   `HOLD` with confidence 40, or 30 when VIX is above 25 (`enhanced_llm.py:357-358`,
   `:386-389`). So on the Space this layer is a near-constant. See 7.3.
9. **Master signal** (`enhanced_api.py:447-737`).
   - Take a **copy** of the per-timeframe config (`:502`).
   - Gap mode lowers the confidence gate by 10 if the measured overnight gap is at least
     0.5% (`:513-526`); the gap is yesterday's close to today's open from daily bars
     (`:265-270`).
   - If momentum is strong or moderate: score = magnitude × direction sign, confidence =
     the magnitude as a percentage (`:529-538`).
   - Otherwise: score = momentum vote × 0.4706 + LLM vote × 0.2941 + sentiment × 0.2353
     (`:564-576`), and confidence = weighted mean of momentum (0.4), LLM (0.3) and
     sentiment (0.3), where a LOW sentiment reading is **left out and the rest
     renormalised** rather than counted as zero (`:596-616`).
   - Reversal mode flips the score when RSI is ≥ 70 or ≤ 30 in the direction of the move,
     and zeroes it otherwise, saying so (`:631-655`).
   - Fire `CALLS`/`PUTS` only if |score| beats the threshold and confidence beats the gate
     (`:658-667`); otherwise `HOLD` with a reason (`:673-701`).
10. **Options structure** (`enhanced_api.py:739-842`), **session context** (`:288-302`),
    **quote** (`:307-321`), **HAR** (`:332`).

**HAR**, the volatility forecast (`har_forecast.py`). HAR means "heterogeneous
autoregressive": tomorrow's variance is regressed on yesterday's, the last week's average
and the last month's average. The variance measure is Garman-Klass (uses the day's open,
high, low and close) plus the squared overnight gap (`har_forecast.py:57-82`). The fit is
in logs, on this symbol's own daily bars, at least 60 of them (`:53`, `:119-128`), with a
lognormal correction because exponentiating a log fit gives the median, not the mean
(`:130-139`). It reports in-sample fit against "tomorrow equals today" (`:141-158`) and the
gap to implied volatility (`:159-166`). It "replaces nothing and votes on nothing"
(`har_forecast.py:3`).

**TFT**, the transformer (`tft_model.py`). 96 daily bars of 10 features plus 5 "static"
features that are constants (`tft_model.py:215-233`, `:344-358`). Weights are loaded with
`weights_only=True` (`:261`), scalers from a `.joblib` beside them (`:266-270`).
`gap_probability` is simply (P(up) + P(down)) × 100 (`:559`), which is why a model sitting
at one-third per class reports about 67 for everything **(inferred)**.

### 3.4 Deployment

**There is no deploy script in this repository.** `trade_analysis/deploy.py` is *not* a
deployer despite the name: it is a hardware detector (`DeploymentConfig.auto_detect`,
`deploy.py:29-79`) and the only thing the app uses from it is that one call
(`enhanced_api.py:93-94`). The rest of the file is dead (7.3).

The Space is a separate repository on Hugging Face. This git repo has a single remote,
GitHub (`git remote -v`: `origin https://github.com/ChaitanyaKharche/stock.git`), no
submodule, and CI only runs tests (`.github/workflows/tests.yml`). The upload step itself —
`hf auth login`, then `HfApi().upload_folder(...)` — is recorded **only in the memory
note**, not in any file in the repo.

`huggingface_space/sync_shared.py` is the pre-upload gate:

- It compares five paths between the repo root and the Space folder
  (`sync_shared.py:40-56`): `local_data/`, `trained_models/`,
  `trade_analysis/live_lab/indicators.py` ↔ `trade_analysis/lab_indicators.py`,
  `trade_analysis/bulk_download/trading_days.py` ↔ `trade_analysis/trading_days.py`, and
  `trade_analysis/indicators_pandas.py`.
- Directories are compared by content hash, not size and timestamp (`:100-116`).
  `lab_indicators.py` is compared on the code below its docstring because it carries a
  "do not edit" banner (`:77-79`, `:86-90`, `:118-119`).
- Five checkpoints the root keeps are deliberately not shipped (`:61-75`).
- No flag = check only. `--push` copies root → Space, `--pull` copies Space → root; the
  direction is never guessed (`:18-19`, `:143-145`). The banner-exempt file is skipped by
  both (`:126-128`).
- It exits non-zero on drift, or if `trained_models`, `local_data` or
  `lab_indicators.py` is missing on the Space side (`:59`, `:174-189`).

**Why copies and not imports** (`sync_shared.py:10-13`, `README.md:492-494`): the Space
must build from its own small `requirements.txt` with nothing from the parent repo, and
"nothing done to make a demo look good may reach back and perturb a frozen forward test".

**A correct deploy sequence, assembled from what the repo records.** Steps 1–5 are each
documented; step 6 is from the memory note only.

1. If the research tabs should be refreshed: `python
   huggingface_space/tools/build_research_snapshot.py` from the repo root
   (`build_research_snapshot.py:3`, `:8`). It needs `data/cost_model_trades.parquet` and
   `research/figures/day_clusters*.csv`, which are git-ignored (`.gitignore:66`, `:139`)
   and absent from this worktree — so this runs only on the machine that holds them.
2. `python huggingface_space/sync_shared.py --push`, to create the git-ignored
   `huggingface_space/trained_models/` and refresh the copies (`.gitignore:106-113`).
3. `python huggingface_space/sync_shared.py` and require exit 0 (`README.md:505`).
4. If any dependency changed: `pip install --dry-run --python-version 3.12
   --only-binary=:all: -r requirements.txt` (`README.md:189-197`).
5. The three hand-run checks and `verify_fix.py` (`README.md:446-451`; commit `6aa6c85`
   lists exactly this set as its verification). They need the network (yfinance) and are
   deliberately outside `pytest` (`pytest.ini:11-14`).
6. **Needs a token.** Log in to Hugging Face and upload the folder. The token needs write
   access to the Space. Memory note only: use `venv\Scripts\python.exe -m …` because the
   venv's `.exe` shims are dead, and set `PYTHONUTF8=1`.
7. **Needs the four Space secrets** to be set in the Space's settings, not in files:
   `FINNHUB_KEY`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`
   (`README.md:70-72`, `run.sh:33-35`).
8. Check through the page, not with curl (memory note: the API is not publicly routed).

### 3.5 How the repo-root app relates to the Space

- **`app1.py` is an older ancestor, not a duplicate, and it cannot run.** It is a "Stock
  Gap Predictor API" v0.1.1 (`app1.py:341-345`): daily bars, FinBERT sentiment
  (`app1.py:178`), and OpenAI `gpt-4o` asked to answer UP, DOWN or FLAT for the next open
  (`app1.py:296-321`). It shares no code with the Space. Nothing imports it (only
  `README.md:16` mentions it). Its imports `finnhub`, `praw` and `openai` are in none of
  the requirements files, and its launcher names a module `app` that does not exist
  (`app1.py:472`). It reads different variable names (`FINNHUB_API_KEY`,
  `OPENAI_API_KEY`, `app1.py:19-23`). Its only change since the first commit was the
  `pandas_ta` swap (`git log -- app1.py`: `e17df27`, `e08e849`).
- **Root `Dockerfile` is dead.** It copies an `app/` directory that does not exist
  (`Dockerfile:6-7`); the root README says so (`README.md:150-153`).
- **`environment.yml` and `conda_requirements.txt`** describe the cluster's conda
  environment (`environment.yml:7`: a `/home/kharche.c/` prefix; `conda_requirements.txt:3`:
  linux-64), including `bitsandbytes` and `accelerate` — the GPU language-model stack the
  Space does not install.
- **Where the Space's code came from.** `enhanced_api.py`, `agent.py`, `deploy.py`,
  `enhanced_llm.py`, `enhanced_sentiment.py`, `live_signals.py`, `tft_model.py`,
  `streamlit_app.py` and `run.sh` were never in the root tree; they first appear in git
  under `huggingface_space/` in the single commit `c0c869d` (2026-09-08). The root had
  flat-layout `data.py` and `momentum_trading_engine.py` until `e04d237` (2026-08-02)
  renamed them. So the Space folder is the old flat app brought into the repo from outside
  — presumably from the Space itself **(inferred; reason not recorded)**.

## 4. Map

`huggingface_space/`

| file | what it is for |
|---|---|
| `README.md` | Space card front-matter (`:1-9`) plus a long, **chronological** fix log. Earlier sections describe states later sections supersede. |
| `Dockerfile` | Image definition; runs `run.sh` |
| `run.sh` | Starts API then UI; writes `/tmp/api_status` |
| `requirements.txt` | Minimal dependencies, with the reason for each pin in comments |
| `streamlit_app.py` | The whole UI: five tabs |
| `research_snapshot.json` | Frozen research numbers for tabs 1–4 (290 KB; 2,631 day points) |
| `local_data/*.json` | Nine symbols' news and Reddit, collected Aug–Sep 2025; a copy of root `local_data/` |
| `sync_shared.py` | Drift check and pre-upload gate |
| `verify_fix.py` | Network check that real bars reach the indicators and symbols differ |
| `indicators_test.py` | Offline: vectorised Wilder ADX/ATR against the live lab's copy; ADX not constant, not 25.0 |
| `confidence_spread_test.py` | Network: confidence responds to its inputs; TFT does not vote; shared gate is not mutated |
| `signal_reachability_test.py` | Offline, synthetic bars: non-HOLD reachable, PUTS reachable, selectors do something, calendar, 4h fold |
| `tools/build_research_snapshot.py` | Builds `research_snapshot.json` from the parent repo's data |
| `tools/build_symbol_table.py` | Builds `symbols.json`; hard-coded absolute paths (7.5) |
| `tools/calibrate_confidence_gate.py` | Measures the confidence distribution and the gate's fire rate on 20 symbols |
| `tools/calibrate_direction_gate.py` | Sweeps the direction gates against shuffled bars |

`huggingface_space/trade_analysis/`

| file | what it is for |
|---|---|
| `__init__.py` | Stale: `__all__` names a `backtest` module that does not exist (`:2`) |
| `enhanced_api.py` | The API: request handler, master signal, options template, gates |
| `data.py` | Bars, news, Reddit, VIX, option chain; `fold_to_4h` |
| `config.py` | Reads keys from the environment; raises if two are missing |
| `cache.py` | Disk cache in `/tmp/.cache`; the 24 h lifetime is declared but never enforced (7.3) |
| `indicators.py` | Indicators and per-timeframe "setup" summary |
| `lab_indicators.py` | Copy of the live lab's indicators; the test oracle. Do not edit. |
| `indicators_pandas.py` | Byte-identical copy of the root's `pandas_ta` replacement; used only by `train_tft.py` |
| `momentum_trading_engine.py` | Magnitude and direction scores, direction gates, engine-level options text |
| `enhanced_sentiment.py` | One sentiment model and the composite |
| `enhanced_llm.py` | Language-model ensemble for GPUs, plus the rule fallback that actually runs |
| `har_forecast.py` | HAR volatility forecast |
| `tft_model.py` | Reconstructed transformer; loader, predictor, trainer |
| `market_session.py` | Open/closed/holiday and which session the bars belong to |
| `trading_days.py` | Holiday table to 2026; byte-identical copy of `trade_analysis/bulk_download/trading_days.py` |
| `symbols.json` | 285 tickers → company names |
| `deploy.py` | Hardware detection; otherwise dead |
| `agent.py` | Background loop that polls the API; cannot trade (7.3) |
| `live_signals.py` | Not imported by anything |
| `collect_data.py` | Would write a `local_data` file; broken (7.3) |
| `train_tft.py` | TFT training script; not used by the app |

Repo root: `app1.py`, `Dockerfile`, `environment.yml`, `conda_requirements.txt` — see 3.5.

## 5. Design decisions and the reason for each

| decision | reason, with evidence |
|---|---|
| Copies, not imports, for the five shared paths | Independence in both directions: `sync_shared.py:10-13`, `README.md:492-494`, `lab_indicators.py:3-7` |
| `trained_models/` git-ignored on the Space side only | The same 19 MB is already tracked at the root and git keeps blobs forever; "only free before the first commit": `.gitignore:106-113`, `README.md:482-486` |
| `sync_shared.py` exits non-zero rather than warning | So a Space without weights is never uploaded and silently falls back: `sync_shared.py:169-173` |
| Five checkpoints not shipped | They are full pickles that fail `weights_only=True` and are what Hugging Face flags as unsafe; the root keeps them because `trade_analysis/models/tft_backtest.py:39` reads one: `sync_shared.py:61-75` |
| Research tabs read one bundled JSON | "The Space must build with nothing from the parent repository, and it must never import the live lab": `tools/build_research_snapshot.py:5-8` |
| Analyzer kept, moved last, labelled | It calls direction, the thing found to have no edge: `README.md:24-27`, commit `25cd841` |
| yfinance for bars | Finnhub's free tier has no history; yfinance was already installed and needs no key: `data.py:169-172` |
| Empty timeframe omitted, not stubbed | So "Insufficient data" keeps meaning what it says: `data.py:176-177` |
| `pandas_ta` removed outright | Abandoned, breaks on numpy 2, and later deleted from PyPI: `requirements.txt:20-26`, commit `e08e849` |
| Floors, not ranges, in requirements; `pandas<3` and `scikit-learn<1.8` as justified ceilings | An upper bound once made pip back-track to a pandas with no Python 3.12 wheel and the build died; pandas 3 would have built and then failed on every request: `requirements.txt:4-18`, `:39-52` |
| Unavailable data is `None`, never a default | `put_call_ratio 0.85`, `iv_rank 45.5`, `vix 20` used to be literals shown as measurements: `data.py:286-310` |
| Implied vol from the traded straddle | yfinance's own field is a placeholder (`0.00001` at the money): `data.py:355-372`, commit `5565a71` |
| TFT computed but does not vote | Measured as a constant per symbol; a constant direction was a fixed bias: `enhanced_api.py:540-563` |
| Never train inside a request | `enhanced_api.py:219-224` |
| HAR shown instead of the TFT | "a small model that demonstrably responds to its input is a better thing to display than a large one that does not": `enhanced_api.py:328-331` |
| HAR fitted per symbol | Lag structure transfers between assets, coefficients do not: `har_forecast.py:108-110` |
| Garman-Klass plus overnight gap | The bare estimator is blind to the gap, which would inflate the apparent premium: `har_forecast.py:60-66` |
| A component with no reading abstains | Counting it as zero spent 30% of the confidence on a constant: `enhanced_api.py:600-608` |
| Component weights left alone | "Widening the visible spread by reweighting would be making the demo look livelier, not making it correct": `enhanced_api.py:594-595` |
| Confidence does not change with strategy | It measures evidence quality; the strategy changes score and gate: `README.md:388-406` |
| Gates not retuned after the 2.5% finding | "retuning a gate is not a bug fix" — a product decision: `enhanced_api.py:491-494` |
| Gates live in one module-level dict the calibrator imports | The calibrator's own copy had drifted: `enhanced_api.py:409-419`, commit `6aa6c85` |
| `scalp` removed, `reversal` added | Scalp was byte-identical to momentum in all 36 measured rows; reversal can take the other side: `enhanced_api.py:505-510`, `:618-630` |
| 1m and 5m removed | yfinance gives 7 days of 1-minute bars, too few to warm a 20-period indicator: `data.py:19-22` |
| 4h folded per session | A wall-clock resample would splice the overnight gap into an intraday bar: `data.py:39-42` |
| Calendar, not weekday check | Labor Day 2026-09-07 is a Monday: `market_session.py:3-7` |
| Space tests excluded from `pytest` | They hit yfinance; "a network flake must never read as a code failure": `pytest.ini:11-14` |
| Symbol names bundled | Search must work with no network and no vendor entitlement: `streamlit_app.py:55-58` |
| `return`, not `st.stop()`, inside tabs | `st.stop()` would blank every later tab: `streamlit_app.py:28-30` |
| Emoji, not `:material/…:` icons; icon font exempt from the font override | Icons rendered as literal text: `streamlit_app.py:15-17`, `:176-186` |
| Per-request HTTP client removed | It existed only to be passed into a bug: `enhanced_api.py:173-179` |

Reason **not recorded**: the score thresholds 0.30 / 0.35 / 0.38 / 0.42, the gap-mode −10,
the confirmed-reversal −8, the 0.5% gap floor beyond one sentence
(`enhanced_api.py:516`), and every weight in `TIMEFRAME_WEIGHTS`
(`momentum_trading_engine.py:399-404`).

## 6. Records, formats, invariants, frozen things

**Must not change without the stated care**

- `trade_analysis/lab_indicators.py` — never edit; re-copy wholesale from the lab and
  re-run `indicators_test.py` (`lab_indicators.py:1-7`). Verified today: `diff` against
  `trade_analysis/live_lab/indicators.py` shows only the docstring banner.
- `trade_analysis/indicators_pandas.py` and `trade_analysis/trading_days.py` —
  byte-identical to their root sources (same git blob hashes today). Holidays are appended
  at the root each year; the table currently ends at 2026 (`trading_days.py:94-96`).
- `local_data/` — all nine files identical to root `local_data/` today.
- `TIMEFRAME_CONFIGS` — callers must copy it; `dict()` at `enhanced_api.py:502` is
  "load-bearing" because strategy branches mutate the copy (`enhanced_api.py:416-419`).
- `TIMEFRAME_SPEC` keys must stay `15m` / `hourly` / `daily` (`data.py:15-17`).
- The TFT must stay out of `weights`; the three remaining weights sum to 1
  (`enhanced_api.py:564-568`).
- The Space must not import anything from `trade_analysis/live_lab/` or the repo root.
- The root's five excluded checkpoints must not be deleted to "fix" the asymmetry
  (`sync_shared.py:68-69`).
- `huggingface_space/trained_models/` must never be committed (`.gitignore:106-113`).
- `tools/build_research_snapshot.py` only **reads** `live_lab_data/`
  (`build_research_snapshot.py:80-87`). Keep it that way; that directory is the frozen
  forward test.

**Formats**

- API reply: `EnhancedSignalResponse` (`enhanced_api.py:47-71`) — signal, confidence,
  reasoning, `position_size` (a fraction of capital, not a share count), `details`,
  `momentum_analysis`, `llm_ensemble`, `options_strategy`, `verdict`, `quote`,
  `volatility`, `market`.
- `research_snapshot.json` top-level keys: `generated_at`, `headlines`, `silent_bugs`,
  `tft_regimes`, `fill_explorer`, `forward_test`, `backfill`, `day_clusters`
  (`research_snapshot.json:2`, `:3`, `:35`, `:87`, `:116`, `:253`, `:619`, `:1570`).
- `local_data/<SYM>_external_data.json`: `news_data` (Finnhub company-news objects) and
  `reddit_data` (title, score, url, created_utc, subreddit; `data.py:272-276`). News
  timestamps in the nine files run from epoch 1755823031 to 1757342441; converting two of
  them with `date` gives 2025-08-28 and 2025-08-31, and one headline is date-lined "Aug.
  30, 2025" (`local_data/NVDA_external_data.json:44`).
- `/tmp/api_status`: empty, `ready`, or a diagnosis (`run.sh:17-19`).
- Checkpoint lookup order per symbol: `tft_<SYM>_validated.pth`, `tft_<SYM>_e200_.pth`,
  `tft_<SYM>.pth`, `tft_model.pth` (`tft_model.py:60-70`).

## 7. Failure modes, incidents, traps

### 7.1 The silent-failure bug class — each instance

"Gate" below means a threshold a value must clear before the app acts.

| # | defect | symptom | fix | what pins it now |
|---|---|---|---|---|
| 1 | The data layer fetched a one-row quote, not history | `HOLD` at exactly 15% for every ticker: `0×0.4 + 50×0.3 + 0×0.3` | Real bars from yfinance (`data.py:153-209`) | `verify_fix.py` (network) |
| 2 | An HTTP client was passed where `days` was expected | HTTP 500 for any symbol without a `local_data` file; hidden because the nine demo symbols return earlier | Argument removed (`enhanced_api.py:173-183`) | **Nothing.** `README.md:460` says the AVGO row "exercises the news fix", but `verify_fix.py` never calls `fetch_news` (`verify_fix.py:34`, `:41-54`) |
| 3 | `ADX_9 = 25.0` constant, tested with `adx > 25` | Trend filter permanently off | Wilder ADX (`indicators.py:75-89`, `:185-187`) | `indicators_test.py:89-109` |
| 4 | `pandas_ta` could not be imported, silently | Every build took the constant fallback above | Dependency removed | `indicators_test.py` |
| 5 | `put_call_ratio`, `iv_rank`, `vix_level` were literals | Shown as measurements | Real values or `None` (`data.py:312-402`) | none |
| 6 | `iv_rank` divided by 100 and used as a volatility | A category error that produced a plausible number | Renamed `implied_vol_pct` (`momentum_trading_engine.py:670-674`) | none |
| 7 | `scikit-learn` never listed; TFT weights never loaded; a training loop in the request handler; `resolve_tft_path` and `MODEL_DIR` each defined twice | None — the code had never run, because the ≥96-row test never passed | `requirements.txt:39-52`, `tft_model.py:15-71`, `enhanced_api.py:107-125`, `:219-238` | none |
| 8 | Confidence read `conviction` (never set → default 50) and needed sentiment `HIGH` (effectively never) | 60% of the weight fixed; every ticker in a 25–27 band | Read the real key, grade MEDIUM (`enhanced_api.py:596-613`) | `confidence_spread_test.py:96-116` |
| 9 | The TFT emitted one constant direction per symbol | A fixed ±0.105 bias on the score | Removed from the weights (`enhanced_api.py:540-574`) | `confidence_spread_test.py:123-134` |
| 10 | Confidence gate 70/65/60/55 against a quantity whose maximum was 47 | Fired on 0 of 80; the app could only say `HOLD` | Recalibrated (7.2) | `signal_reachability_test.py:98-103` |
| 11 | A component with no reading scored as zero confidence | Output pinned near 25 | Abstain and renormalise (`enhanced_api.py:609-616`) | `confidence_spread_test.py` |
| 12 | Momentum score is a magnitude; the converter hard-coded `return 'CALLS'` | `PUTS` unreachable; a sell-off read as bullish | Signed direction score (`momentum_trading_engine.py:274-296`, `:702-715`; `enhanced_api.py:529-538`) | `signal_reachability_test.py:105-116` |
| 13 | The timeframe never reached the engine; a loop variable also shadowed the parameter | The dropdown only moved a threshold | `TIMEFRAME_WEIGHTS` (`momentum_trading_engine.py:399-405`, `:100-104`) | `signal_reachability_test.py:118-130` |
| 14 | Gap mode gated on TFT `gap_probability > 70`, which sits near 67 | Gap mode did nothing, silently | Measured overnight gap; says so when none (`enhanced_api.py:513-526`) | `signal_reachability_test.py:132-140` |
| 15 | `scalp` was a 0.8× threshold multiplier | Identical to momentum in every measured row | Removed; `reversal` added (`enhanced_api.py:631-655`) | `signal_reachability_test.py:160-193` |
| 16 | Options template gated on `confidence > 70` and on removed options | Dead branches; 4h and 1d fell through | `enhanced_api.py:756-815` | none |
| 17 | A closed market looked like a stale feed (Labor Day) | Friday's bars shown as current | `market_session.py`, cross-check with the newest bar (`enhanced_api.py:288-302`) | `signal_reachability_test.py:195-217` |
| 18 | **The calibrator carried its own copy of the gate**, 36/34/33/32, while production ran 36/36/37/38 | The published "fires on 8.8%" described a gate never shipped; the real rate was 2.5% | Calibrator imports `TIMEFRAME_CONFIGS` and asserts the timeframes match (`tools/calibrate_confidence_gate.py:28-42`) | The import itself, plus `confidence_spread_test.py:144-157` for the mutation hazard the fix created |
| 19 | `run.sh` never checked the API came up | Space "Running", every click "Failed to connect" | Status file (`run.sh:27-61`) | none |
| 20 | pandas ceiling → source build with no compiler; pandas 3 → `fillna(method=…)` fails per request | Build failure; a would-be request-time failure | `requirements.txt:4-18`; `.bfill()/.ffill()` (`indicators.py:228-234`) | The `pip --dry-run` command |
| 21 | HAR without the lognormal factor | Forecast about 40% low; found while porting it into the app | `har_forecast.py:130-139`; commit `cad54eb` | none in the Space |
| 22 | yfinance `impliedVolatility` is a placeholder | "Implied vol" of 90.63 then 12.5 for NVDA in one session | Straddle inversion (`data.py:355-397`) | none |
| 23 | Bollinger upper and lower bands read by position, backwards | The TFT was trained on an inverted feature | Read by name (`train_tft.py:64-72`); commit `e08e849` | none |

### 7.2 What the calibration tools did and the numbers they set

**`tools/calibrate_confidence_gate.py`** pulls real bars for 20 symbols, runs the real
momentum engine and the real `_generate_master_signal` for each of four timeframes with
sentiment, VIX and the LLM input held at fixed values (`:44-48`, `:67-84`), and prints the
distribution of confidence and how often the shipped gate fires (`:89-100`).

- Before: gates 70 / 65 / 60 / 55; measured min 17, median 33, max 47; fired on 0 of 80
  (`README.md:263-266`).
- First recalibration: 40 / 38 / 36 / 34, "fires ~9%" (`README.md:266-267`).
- For the 15m / 1h / 4h / 1d set: **36 / 36 / 37 / 38**, graded upward because slower
  timeframes carry higher confidence (`enhanced_api.py:473-476`). These are the shipped
  values (`enhanced_api.py:420-445`).
- Corrected 2026-09-11: with the real gates the fire rate is **2 of 80 = 2.5%**; the 8.8%
  belonged to the drifted copy. That day's distribution: min 17, p50 31, p75 34, p90 37,
  max 47 (`enhanced_api.py:478-489`).

**`tools/calibrate_direction_gate.py`** uses a **permutation test**: shuffle a real
series' returns so any trend is destroyed while volume and volatility are kept; anything
the engine emits on the shuffled series is a false positive, and "lift" is the real firing
rate divided by the shuffled firing rate (`:5-10`, `:80-92`). It sweeps a grid of deadband
and agreement values (`:47-48`, `:113-128`).

- Original behaviour (deadband 0.15): lift 0.70 — it fired more on shuffled data than on
  real data.
- Chosen: **deadband 0.40, agreement 0.75**, lift 1.83 (`momentum_trading_engine.py:21-28`,
  `:44-45`), "a MODEST edge … NOT enough to claim a tradeable edge" (`:40-43`).

Three cautions about these numbers:

1. The lift table records a real firing rate of 8.8%. The tool fires through the same
   confidence gate, and the repo does not record a re-run after the 2.5% correction
   (commit `6aa6c85` lists the tests it re-ran; this tool is not among them).
2. `permute` is applied to each timeframe's frame separately
   (`calibrate_direction_gate.py:119`), so the shuffled 15m, hourly, 4h and daily series
   are no longer views of one price path. The tool's own docstring calls independent
   timeframes a mistake because it "flatters the multi-timeframe agreement test"
   (`:14-16`). The shuffled side may therefore under-fire and the lift be overstated
   **(inferred from reading; not run)**.
3. The per-strategy table in `README.md:314-319` (momentum 3.18, gap 3.53, reversal 5.00)
   has **no script in the repo that produces it**: neither tool mentions `reversal` or
   `gap`, and `calibrate_direction_gate.py:100` hard-codes `"momentum"`.

### 7.3 Dead or misleading code a newcomer might trust

Still-live instances of the house bug class, found in this read:

- **The "LLM ensemble" is a constant on the Space.** No model loads on CPU
  (`enhanced_api.py:101-105`); the no-models branch passes all-False conditions
  (`enhanced_llm.py:123-132`); the result is always `HOLD`, confidence 40 or 30
  (`enhanced_llm.py:357-358`, `:386-389`). So the LLM's 29% of the score is always zero
  and its 30% of the confidence is one of two numbers. The fix in defect 8 replaced a
  constant 50 with a near-constant 40. The richer fallback at `enhanced_api.py:246-256`
  runs only if the first call raises. The calibrators hard-code the same value
  (`calibrate_confidence_gate.py:48`). It follows that a no-momentum reading on a quiet
  VIX is always (0 × 0.4 + 40 × 0.3) / 0.7 ≈ 17 **(inferred)** — the "min 17" in every
  measured distribution, and the `SPY 1h HOLD conf 17/36` in `README.md:429`.
  The model ids, if a GPU were present, are `mistralai/Mistral-Nemo-Instruct-2407` and
  `microsoft/Phi-3-medium-4k-instruct`, the latter under the key `phi4_mini_instruct`
  (`enhanced_llm.py:47-64`); `bitsandbytes` is not in `requirements.txt`, so they could
  not load there either.
- **The background agent cannot act.** It starts with the API (`enhanced_api.py:127-131`),
  wakes every 900 s (`agent.py:96`), asks for QQQ and NVDA at timeframe `5m` — not a
  shipped timeframe, so it silently gets the 15m settings (`agent.py:105-118`;
  `enhanced_api.py:502`) — and requires confidence ≥ 75 (`agent.py:30`, `:129`). The
  highest confidence ever recorded is 52 (`enhanced_api.py:474`). A gate above the
  attainable range, on a loop that still costs two full analyses every 15 minutes.
  `analyze_agent_performance` opens a different database file from the one the agent
  writes (`agent.py:23` vs `:354`), and `compare_strategies` calls an endpoint that does
  not exist (`agent.py:294-297`).
- **The cache never expires.** `TTL = 60*60*24` is declared (`cache.py:3`) and `get` never
  looks at file age (`cache.py:12-21`). For a symbol without a `local_data` file, news is
  fetched once and reused until the container restarts. `put` is also defined twice
  (`cache.py:6`, `:24`).
- **Sentiment for the nine bundled symbols is from August–September 2025.** `local_data`
  wins over cache and API (`data.py:213-215`). NVDA is the default symbol
  (`streamlit_app.py:67-68`). `README.md:520-522` says the files are static but not how
  old. The label on screen is just "Sentiment".
- **`collect_data.py` cannot refresh them.** It still passes an HTTP client as `days`
  (`collect_data.py:29`) — defect 2, unfixed here — and for a symbol that already has a
  file, `fetch_news` returns that file, so it would rewrite the same stale content.
- **The sentiment model is loaded twice.** Once at startup into `enhanced_api`'s instance
  (`enhanced_api.py:82`, `:98`), used only by the health endpoint; again on first request
  into a separate module-level instance (`enhanced_sentiment.py:541-549`, `:554`).
- **A non-negative term is added to a signed score.** `economic_impact` is a keyword count
  (`enhanced_sentiment.py:355-390`) added into the sentiment composite
  (`enhanced_sentiment.py:400-402`). Headlines that mention earnings or the Fed push the
  composite up whatever they say. Same shape as defect 12, smaller.
- **Engine-level options text is always bullish.** `_generate_momentum_strategy` returns a
  `CALL` and `_generate_conservative_strategy` a `BULL_CALL_SPREAD` regardless of
  direction (`momentum_trading_engine.py:578-627`); the scalp version is a long straddle
  (`:542-576`). Visible only under "Raw response".
- **A missing VIX makes the request fail.** `vix_level` is present with value `None` when
  the fetch fails (`data.py:312`), so `.get('vix_level', 20) - 25` raises
  (`momentum_trading_engine.py:451`) and the request returns 500. Loud, not silent.
- `POST /backtest/enhanced/` returns `"status": "success"` and a feature list and does
  nothing (`enhanced_api.py:844-866`).
- `/health/detailed` reports a module-level `tft_predictor` that is never loaded
  (`enhanced_api.py:85`, `:904-906`).
- The request's default timeframe is `"5m"` (`enhanced_api.py:164`), which is not in
  `TIMEFRAME_CONFIGS`.
- `deploy.py`: `ScalableModels` and `ProductionAgent` are unused; `ProductionAgent` uses
  `asyncio` and `datetime` without importing them (`deploy.py:220`, `:236`).
- `live_signals.py` is imported by nothing. `train_tft.py` is not used by the app.
- `indicators.py:113-173` (the `pandas_ta` branch) still contains `ADX_9 = 25.0`
  (`:124`, `:126`); unreachable because the package cannot be installed.
- `tft_model.py:533-535` still trains if called on an unloaded model; only the caller's
  check prevents it.
- Names lie about units: `price_change_5min`, `velocity_1min` are "bars", not minutes, on
  every timeframe (`momentum_trading_engine.py:151-160`). Template text says "max 5
  minutes" for 15m and "15-minute momentum" for 1h (`enhanced_api.py:776`, `:783`).
- Stale comments: `enhanced_api.py:632` (UI timeframes "1m/5m/15m/1h");
  `momentum_trading_engine.py:431-436` ("picking 1m", "Requiring 0.60" — the constant is
  0.75); `verify_fix.py:15` cites `enhanced_api.py:381`.
- `FastAPI` title, description and the root endpoint's feature list
  (`enhanced_api.py:74-78`, `:150-157`): "Multi-LLM Ensemble", "10+ models", "Autonomous
  Trading Agent". One sentiment model, no language model, an agent that cannot trade.

### 7.4 Secrets hygiene

- **Where keys are expected:** environment variables only. Space: `FINNHUB_KEY`,
  `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`; read but unused:
  `TWELVE_KEY`, `LLM_PROVIDER` (`config.py:7-12`); optional: `TFT_MODEL_DIR`
  (`tft_model.py:38`), `TRADE_LOCAL_DATA` (`data.py:67-70`). `config.py:4` calls
  `load_dotenv()`, so a local `.env` also works.
- **Tracked tree at HEAD:** no `.env` is tracked. No token-shaped literal anywhere under
  `huggingface_space/` (pattern search, counts only). The only credential-shaped
  assignments in the whole tree are placeholders, at `README.md:177-183` and
  `trade_analysis/scripts/clean_trade_venv.sh:112`, `:113`, `:115` (classified by
  pattern; values not printed).
- **History:** the root README says credentials were committed to public history in
  `f384d81` and `e17df27` and are not rotated (`README.md:247-271`). I checked only the
  Hugging Face row. At both commits, `README.md:41` and `:82` assign `HF_TOKEN` a
  22-character value that does not start with `hf_` and matches a placeholder pattern, and
  `git log -G` over all 156 commits on this branch finds no commit that adds or removes an
  `hf_`-shaped string of 25 or more characters. So that row looks like a false positive
  **(pattern-based; other branches not scanned; the Alpaca, Finnhub, Twelve Data and
  Reddit rows were not checked)**.
- **The exposed Hugging Face token.** Every mention in the repo says the same thing: in
  `~/.bash_history` on the shared Discovery filesystem, revocation unconfirmed
  (`CLAUDE.md:116-117`, `research/PROJECT_REPORT.md:391-392`,
  `research/PROGRAMME_REVIEW_2026-09-24.md:168-169`). Nothing in the repo records it being
  revoked. It cannot be checked from here.
- **The upload token.** Memory note only: a pasted token was rejected as invalid; the
  cached one is fine-grained, named "RAG + NLP Project", with account-wide `repo.write`.
- **Upload hazard.** There is no `.gitignore` or `.dockerignore` inside
  `huggingface_space/`, the Dockerfile copies everything (`Dockerfile:17`), and the upload
  command is not in the repo, so whether it excludes anything is unknown. A `.env` placed
  in that folder would be published to a public Space **(inferred)**.
- **What the public snapshot contains.** The trader's own journal totals by day type
  (`research_snapshot.json:17359-17398`) and every six-line replay trade with prices
  (`research_snapshot.json:936` onward). Deliberate, by the look of the UI — but it is
  personal P&L on a public page.

### 7.5 Traps for whoever touches this next

- **The Space README is a diary, not a description.** `README.md:266-267` still says the
  gates are 40/38/36/34 and fire about 9%; `README.md:283-289` still describes 5m bars
  that `README.md:375-379` then removes; the layout block (`README.md:36-66`) omits
  `research_snapshot.json`, `har_forecast.py`, `indicators_pandas.py` and two tools.
- **`tools/build_symbol_table.py` writes into the main checkout** by absolute path
  (`build_symbol_table.py:16`) and reads a Desktop file (`:17`). The live lab runs from
  that checkout.
- **The snapshot is stale by design.** It ends 2026-09-24 (`research_snapshot.json:255`);
  the lab has recorded through 2026-10-12 (`git log`). The note inside it says "Fixed
  2026-09-25" (`research_snapshot.json:616`).
- **Rebuilding the snapshot elsewhere degrades silently in one place.** Without the
  parquet, `fill_grid` raises (`build_research_snapshot.py:43`) — loud. Without the
  day-cluster CSVs, `day_clusters` returns `None` (`:125-126`) and the section simply
  disappears from the page (`streamlit_app.py:248-249`).
- **This worktree is not deployable as it stands.** `huggingface_space/trained_models/`
  does not exist here, so `sync_shared.py` would report it missing and exit 1
  (`sync_shared.py:97-98`, `:175-189`). Correct behaviour.
- **`trading_days.py` runs out at the end of 2026.** After that the fallback treats every
  weekday as a session (`trading_days.py:147-151`); `pandas_market_calendars` is not in
  the Space's requirements.
- **Two checks are weaker than they read (inferred; not run).**
  `signal_reachability_test.py:220-232` builds 140 hourly bars with no overnight gap
  (`:223`), on which a plain wall-clock resample gives the same bucket counts as
  `fold_to_4h`; and `:225` compares a value with itself. The project rule is that a test
  must fail on the broken version.
- **`README.md:157-158`'s example outputs (6 vs 27, 24 vs 12) predate the abstain fix**;
  under the current formula a LOW-sentiment row no longer gives 12 **(inferred)**.
- **`streamlit` is unpinned** (`requirements.txt:59`); the UI probes for a feature rather
  than assuming it (`streamlit_app.py:69-71`).
- **Implied vol uses a fixed half-session time to expiry for a same-day chain**
  (`data.py:380-382`), and compares a nearest-expiry number with a next-session forecast.
  The on-screen caveat covers the missing bid/ask, not the maturity mismatch.
- The memory note warns the Streamlit dropdowns cannot be driven by browser automation.

## 8. Current state

**Done**

- All fixes in 7.1; the research-first layout (`25cd841`, 2026-09-24).
- UI-level five-minute cache (`streamlit_app.py:139-150`).
- Shared copies in sync today for the four tracked paths.

**Half-done**

- Backend still refetches and refits on every uncached request.
  `research/PROJECT_REPORT.md:320-321` says "no caching anywhere"; that is now out of date
  for the UI only.
- TFT output is still displayed, now inside a labelled expander
  (`streamlit_app.py:673-680`); `research/PROJECT_REPORT.md:387` lists removing it as
  remaining work.
- The LLM layer, agent, cache lifetime and stale `local_data` (7.3) are untouched.

**Open**

- Whether commit `25cd841` was ever uploaded. Its message records only "Rendered and
  checked locally, all five tabs". The last recorded health check of the live Space is
  2026-09-11 (commit `6aa6c85`: "the Space is Running").
- The upload procedure exists only in a memory note.
- The Hugging Face token on Discovery: revocation unconfirmed.
- No offline CI covers the Space (`pytest.ini:17`, `.github/workflows/tests.yml`).

## 9. Contradictions

| source says | code says |
|---|---|
| Memory note and `README.md:473`, `research/PROJECT_REPORT.md:274`: **three** shared paths | **Five** (`sync_shared.py:40-56`) |
| Memory note: purpose is recruiter-facing SDE showcase, "not research" | The Space now opens on research results (`streamlit_app.py:22-26`); the note predates `25cd841` (file last modified 2026-09-11) |
| Memory note: nothing about HAR, the snapshot, the five tabs | All present |
| `README.md:266-267`: gates 40/38/36/34, fires ~9% | 36/36/37/38, fires 2.5% (`enhanced_api.py:420-445`, `:486`) |
| `README.md:60`: "FinBERT-family sentiment ensemble" | One DistilRoBERTa model (`enhanced_sentiment.py:53-58`) |
| `README.md:144-146`: fallback gives "40 base, 65 on momentum, −10 on high VIX" | The 65 path is unreachable on CPU (`enhanced_llm.py:123-132`) |
| `README.md:460`: AVGO "exercises the news fix" | `verify_fix.py` never calls `fetch_news` |
| "six-parameter" HAR (`streamlit_app.py:439`, `enhanced_api.py:323`, `har_forecast.py:5`) | Four fitted coefficients (`har_forecast.py:97`, `:128`); the same file says "only four parameters" (`:23`, `:53`) |
| `research/PROJECT_REPORT.md:349`: HAR beats persistence at t=+7.82 | `har_forecast.py:35`, `:43-44`: t=+8.27, and "The t=+7.82 does not match the recorded result" |
| Root `README.md:224-228`: `train_tft.py` removed, `enhanced_api.py` "no longer exists in the source tree" | Both exist under `huggingface_space/trade_analysis/` |
| Root `README.md:16`: `app1.py` is a "data-fetch service" | It is a gap predictor that asks `gpt-4o` (`app1.py:274-338`) and cannot start (3.5) |
| Root `README.md:258`: `HF_TOKEN` leaked in `README.md` at two commits | Placeholder-shaped at those lines (7.4) |
| `.githooks/pre-commit` header: Alpaca "Rotation fixed the key" | Root `README.md:254`: rotated? "NO" |
| `Dockerfile:6-7` comment: `TRANSFORMERS_CACHE` is deprecated | `Dockerfile:9` still sets it |
| `sync_shared.py:1`: "byte-identical" | One of the five is compared below its banner (`:77-79`) |
| `tools/calibrate_direction_gate.py:14-16`: independent timeframes are a mistake | `:119` shuffles each timeframe independently (7.2) |
| `CLAUDE.md`: ThetaData lapsed to `Options: FREE` on 2026-09-08 | The snapshot shows live option trades on every session from 2026-09-14 to 09-24 (`research_snapshot.json:512-608`). Outside this slice — for whoever owns the live lab. |

## 10. Questions only the owner can answer

1. Is the live Space running commit `25cd841`, or the older signal-first version? Is it
   running at all?
2. What exact command uploads it, and does it exclude anything (`__pycache__`, a stray
   `.env`)? It should be written into the repo.
3. Has the Hugging Face token in Discovery's `~/.bash_history` been revoked? Is it the same
   token as the cached "RAG + NLP Project" one?
4. Are the four Space secrets the same Finnhub and Reddit keys the root README lists as
   committed to public history and not rotated?
5. Is publishing your own journal totals by day type on a public page intended?
6. Should the snapshot be refreshed on a schedule, or is a dated snapshot the point?
7. The nine `local_data` files are from Aug–Sep 2025. Delete them so the Space fetches live
   news, refresh them, or label the date on screen?
8. Should the background agent, the unused LLM code and the "SOTA … multi-LLM" wording
   stay, given the Space's new message?
9. Should the analyzer keep printing suggested structures with profit targets and spreads,
   given "stop capping winners" and "naked calls and puts only"?
10. Was the direction-gate lift table re-measured after the 2.5% correction, and where is
    the script behind the per-strategy lift table?
11. Is "six-parameter" a slip, or does it count something the Space's HAR does not fit?
12. Where did the `huggingface_space/` code come from on 2026-09-08 — pulled down from the
    Space?
13. `app1.py`, the root `Dockerfile`, `environment.yml`, `conda_requirements.txt`: keep as
    history or delete?
