---
title: Trade Analysis Agent
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
---

# ProfitBook — Trading Analysis Engine

FastAPI + Streamlit service that turns multi-timeframe price history, news/social
sentiment and an options chain into a directional signal with a position size.

**This folder is the deployable Space.** It is self-contained: `docker build` here needs
nothing from the parent repository.

---

## Layout

```
huggingface_space/
├── Dockerfile               # python:3.12-slim; runs run.sh
├── run.sh                   # uvicorn :7860 (API) + streamlit :8501 (UI)
├── requirements.txt         # minimal, pinned where a version actually matters
├── streamlit_app.py         # UI; talks to the API on localhost:7860
├── verify_fix.py            # proves per-symbol analysis works (see below)
├── indicators_test.py       # checks the vectorised indicators against the live lab's
├── confidence_spread_test.py # asserts the score responds to its inputs
├── signal_reachability_test.py # asserts the app can emit something other than HOLD,
│                            #   and that momentum / gap / reversal really differ
├── sync_shared.py           # drift check + pre-upload gate (exits non-zero)
│   trade_analysis/market_session.py  # open/closed + which session the bars are from
│   trade_analysis/trading_days.py    # US exchange calendar  [COPY — see Shared files]
├── tools/
│   ├── calibrate_confidence_gate.py  # min_confidence, from the real distribution
│   └── calibrate_direction_gate.py   # direction gates, by permutation test
├── local_data/              # 9 symbols of collected news/Reddit  [COPY — see Shared files]
├── trained_models/          # TFT checkpoints (diagnostic only; see below)
└── trade_analysis/
    ├── data.py                     # market data + alt data
    ├── indicators.py               # technical indicators
    ├── lab_indicators.py           # [COPY of the live lab's indicators]
    ├── enhanced_api.py             # FastAPI app, /predict/enhanced/
    ├── enhanced_sentiment.py       # FinBERT-family sentiment ensemble
    ├── enhanced_llm.py             # LLM ensemble layer
    ├── momentum_trading_engine.py  # momentum setups + options strategy
    ├── tft_model.py                # gap-prediction TFT
    ├── agent.py / deploy.py / live_signals.py / collect_data.py / train_tft.py
    ├── cache.py / config.py
```

## Running it

Secrets required as Space variables: `FINNHUB_KEY`, `REDDIT_CLIENT_ID`,
`REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`. `config.py` raises on startup if the first
two are missing, so a blank Space will fail fast rather than serve nonsense.

Locally:

```bash
pip install -r requirements.txt
./run.sh          # UI on :8501, API docs on :7860/docs
```

---

## What was wrong, and what fixed it

The deployed app returned **`HOLD` at exactly 15% confidence for every ticker**. That was
not a cautious model. It was a constant, and it is worth writing down because the failure
was silent at every layer.

**Root cause — `data.py` never fetched history.** `fetch_multi_timeframe_stock_data`
called Finnhub's `/quote` endpoint, commented *"historical candle data is a premium
feature"*, and returned a **one-row** DataFrame with `Volume` hardcoded to `0`. That was
the only market data the engine ever saw. Then:

- `indicators.identify_current_setup` opens with `if df.empty or len(df) < 2:` and returns
  `{"adx": 0, "rsi": 50, "error": "Insufficient data"}`. One row is always `< 2`, so that
  branch fired **100% of the time, for every symbol**.
- The momentum engine consequently reported `confidence: 0`.
- `enhanced_api.py` then computed
  `weighted_confidence = 0*0.4 + 50*0.3 + 0*0.3 = 15.0` — where the `50` is the *default*
  from `llm_analysis.get("conviction", 50)`, a key that layer never sets.
- The TFT gate needs ≥96 daily rows and never once received them.

`yfinance` was already in `requirements.txt` and already imported by three other modules —
just not by the one that decided every signal. It serves the intraday history Finnhub's
free tier withholds, without an API key. Timeframes are now fetched concurrently in
threads, and an empty timeframe is **omitted** rather than stubbed, so "Insufficient data"
keeps meaning what it says.

**Five further defects, all found by following the first:**

| | |
|---|---|
| `fetch_news(symbol, client)` bound an `httpx.AsyncClient` to the `days` parameter | any symbol *without* a `local_data/` snapshot hit `timedelta(days=<AsyncClient>)` → HTTP 500. Hidden because the nine cached symbols return before that line. |
| `ADX_9 = 25.0` hardcoded in the manual fallback | ADX is gated three times with `adx > 25` / `adx > 20`. At exactly 25.0 the first is permanently False and the second permanently True — trend filtering stopped existing instead of failing. |
| `pandas_ta` silently unimportable | 0.3.14b0 still does `from numpy import NaN`, removed in numpy 2.0, so a fresh build always took the constant-valued fallback above. **Dependency removed entirely** rather than pinning numpy back to keep an abandoned 2021 beta alive — the verified Wilder implementations replace everything it provided. |
| `put_call_ratio: 0.85`, `iv_rank: 45.5`, `vix_level: 20` | literals rendered in the API response as though measured. Now computed from a real yfinance option chain and real `^VIX`; anything genuinely unavailable returns `None` and is flagged in `data_quality` rather than given a plausible-looking number. |
| `iv_rank` was misnamed against its own consumer | IV *rank* means where current IV sits in its 52-week range (0–100). The field's only reader was `implied_vol = alternative_data.get('iv_rank', 50) / 100.0` — dividing by 100 and using the result as a volatility, which only type-checks if the value is an IV **percentage**. Renamed `implied_vol_pct`, which is what the consumer always wanted and what the chain actually provides. A true rank needs IV history this tier cannot reach, so it is not offered at all rather than approximated under its own name. |
| a per-request `async with httpx.AsyncClient()` that nothing used | `UnifiedDataProvider` already owns a long-lived client. This second one existed only to be passed to `fetch_news`, i.e. only to cause the bug above, and cost a TCP/TLS handshake per call. Removed, along with the import it left dangling. |

`yfinance==0.2.28` was also pinned to an Oct-2023 release that no longer speaks to Yahoo's
current endpoints; it is now `0.2.65`, the version the fix was verified against.

The through-line in all six: **every one failed silently.** A missing key fell back to a
default, an unimportable package fell back to a constant, a bad argument only crashed on
inputs nobody demoed. None of them logged anything. The reason the app looked like a
working model returning a cautious `HOLD` is that each layer politely absorbed the failure
below it — so the fixes above are paired with things that now fail loudly instead:
`sync_shared.py` exits non-zero, unavailable data returns `None` rather than a plausible
number, and an empty timeframe is omitted rather than stubbed.

### "Why is it the same number for every stock?" — a fair complaint, and a real bug

After the data fix, symbols returned 25 / 25 / 27 / 27. Not a constant any more, but close
enough to be a fair question. The underlying analysis was in fact varying by 3x — measured
TSLA daily momentum 0.638 against SPY 0.183 — and the compression was entirely downstream:

```
weighted_confidence = momentum_conf*0.4 + llm.get('conviction',50)*0.3 + (sent=='HIGH')*80*0.3
                    = momentum_conf*0.4 +            15                +           0
```

**60% of the weight was pinned.**

  * On CPU the LLM ensemble is skipped and the rule-based fallback returns **`confidence`**,
    not `conviction` — 40 base, 65 on momentum, −10 on high VIX. `.get("conviction", 50)`
    never found the key, so all of that was discarded for the literal default: a constant 15.
  * Sentiment `HIGH` requires `std_dev < 0.2 AND mean_abs > 0.3`. Headline scores cluster
    near 0 and 1, so that gate effectively never passes: a constant 0.

Both are now read honestly, and `MEDIUM` is graded instead of thrown away.
**The weights are deliberately unchanged** — widening the visible spread by reweighting
would make the demo look livelier without making it more correct.

`confidence_spread_test.py` guards it, driving the real `_generate_master_signal`:

```
llm confidence 20 vs 90   ->  6 vs 27    (reads the real key)
sentiment MEDIUM vs LOW   -> 24 vs 12    (graded)
```

**What this does not fix, and should not:** momentum conviction genuinely clusters, because
most symbols sit in `WEAK_MOMENTUM` most of the time and the strategy's own thresholds
(0.25 / 0.4 / 0.6) put them there. That is the model having little to say, which is the
correct output for a quiet tape — not something to paper over with a wider scale.

### The first deploy of this fix failed, and that is in here too

Pinning `pandas>=2.0,<2.3` to protect `pandas_ta` made the Space build fail:

```
Project version: 2.1.0
ERROR: Unknown compiler(s): [['cc'], ['gcc'], ['clang'], ...]
```

An upper bound let pip backtrack to **pandas 2.1.0 (Aug 2023), which predates Python 3.12
and ships no cp312 wheels**, so it tried to compile from source — and `python:3.12-slim`
has no C compiler. Bare `pandas` had always resolved to a wheel, which is why this only
appeared once a ceiling was added.

Removing `pandas_ta` removed the reason for the ceiling. But dropping it entirely then
resolved to **pandas 3.0.5**, which removed `DataFrame.fillna(method=...)` — called on
every request in `enrich_with_indicators`. That image would have built perfectly and
raised `TypeError` on every analyze: a failure that appears only at request time.

So: the calls are now `.bfill()`/`.ffill()` (valid on both), *and* `pandas<3` stays as a
**justified** ceiling — pandas 3 also flips copy-on-write and the default string dtype,
none of it validated here.

Both were caught before redeploying, by resolving the requirements the way the image does
rather than pushing and waiting:

```bash
pip install --dry-run --python-version 3.12 --only-binary=:all: -r requirements.txt
```

`--only-binary=:all:` fails on exactly the condition that broke the build — any package
with no matching wheel. Worth running before any dependency change.

### The TFT was measured, found degenerate, and removed from the decision path

The checkpoints load and are structurally intact. They carry no signal, and this was
established by measurement rather than assumed.

**It does not respond to its input.** Feeding the NVDA checkpoint six different symbols'
entire price histories:

```
fed      gap_prob   dir    P(UP)   P(DOWN)  P(FLAT)
QQQ        67.10   DOWN   0.3320   0.3390   0.3290
SPY        67.20   DOWN   0.3320   0.3390   0.3280
TSLA       67.10   DOWN   0.3310   0.3400   0.3290
META       67.10   DOWN   0.3330   0.3380   0.3290
```

gap_probability spread **0.10 on a 0–100 scale**; no class probability deviates from ⅓ by
more than **0.0077**. It has collapsed to the prior.

**Out of sample it is exactly the base rate.** Walking the last 120 sessions, predicting on
data up to *t* and scoring against *t+1*:

| | hit rate | majority baseline | predictions |
|---|---|---|---|
| NVDA | 48.7% | 51.3% | DOWN 119/119 |
| SPY | 54.6% | 54.6% | UP 119/119 |
| QQQ | 54.6% | 54.6% | UP 119/119 |

One constant direction per symbol, matching the baseline to the decimal — which is what a
constant predictor gives you.

**The checkpoint says why.** `scaler_static` has every `scale_` equal to 1.0, so the static
branch saw zero variance during training. That is the repo's "trained on a constant
placeholder" note verified from the artifact.

Because `signal_scores["tft"]` was ±0.7 by direction, a constant direction made it a **fixed
±0.105 per-symbol bias** on `weighted_score` — the same defect as the two dead confidence
terms, except the dead thing was a model. It has been removed from `weights`, which are
renormalised to sum to 1.0 so the 0.3 threshold still means what it meant.
`confidence_spread_test.py` asserts UP/DOWN/FLAT now produce identical output.

`tft_prediction` is still computed and returned — it is honest diagnostic output — it simply
no longer votes.

**Five checkpoints were also dropped from this folder.** `tft_AMZN_e200_`, `tft_MSFT_e200_`,
`tft_SPY_e200_`, `tft_TSLA_e200_` and `tft_model.pth` all fail
`torch.load(weights_only=True)` — they are full pickles rather than plain tensors, so
`tft_model.py` could never load them, and they are exactly the files Hugging Face flags
without a "Safe" badge. Shipping 8 MB of unloadable pickles as the only security warnings on
a public repo is worse than not shipping them. The **root repo keeps them**, because
`trade_analysis/models/tft_backtest.py` reads `tft_AMZN_e200_.pth` directly; `sync_shared.py`
knows the omission is deliberate and does not report it as drift.

**On provenance:** the repo README records `srun --gres=gpu:h100:1`. The artifact is 36
tensors / **398,854 parameters** / 1.61 MB of float32 — a model that size trains in minutes
on a CPU, so the checkpoint is not evidence of large-scale compute regardless of which node
it ran on.

### It could only ever answer HOLD

Reported as "it can't say anything other than hold and gives the same 25% or 27%". The
audit was worse than the complaint: **all 12 timeframe x strategy combinations returned one
identical answer.** Four independent defects, each sufficient on its own.

**1. The confidence gate was unreachable.** `min_confidence` was 70/65/60/55, tested
against a weighted mean of components whose own ceilings sit far below 100. Measured over
80 observations (20 symbols x 4 timeframes): **min 17, median 33, max 47.** The gate fired
on **0 of 80**. Not a strict threshold -- a broken one. Now 40/38/36/34, each a percentile
of that measured distribution; fires ~9%, so HOLD stays the common answer.

**2. A component with no reading was scored as zero confidence.** Sentiment confidence is
`LOW` unless `std_dev < 0.2 AND mean_abs > 0.3`, which headline sentiment essentially never
satisfies. Averaging that in as a 0 spent 30% of the budget on a constant, capping the
result at 70 and pinning it near 25. Absent components now **abstain and the remaining
weights renormalise** -- the same correction already applied to the TFT.

**3. `PUTS` was unreachable for any input.** `_calculate_momentum_score` is a pure
MAGNITUDE: every term is `abs()`-wrapped or non-negative by construction. It measures how
much is happening, never which way. `_convert_signal_format` therefore hardcoded
`return 'CALLS'` with the comment *"momentum typically bullish for options"*, and
`weighted_score = conviction` could not go negative. **A stock breaking down scored high
momentum and was reported as a bullish setup.** The direction was available all along --
`directional_bias` is already normalised to [-1, +1] and was being thrown away by `abs()`.

**4. The timeframe selector never reached the analysis.** The momentum engine hardcoded
`{15m: 0.5, hourly: 0.3, daily: 0.2}` and never read the user's choice, so every timeframe
analysed identical data and the dropdown only moved a threshold. It now selects a weight
vector, and `5m` bars were added so the fast settings have real data to weight. (`1m` is an
emphasis setting over 5m bars -- yfinance caps 1m history at 7 days, too few for the
20-period indicators to warm up, so minute bars are deliberately not fetched rather than
faked.)

`strategy_mode="gap"` was also dead: its only effect was gated on the TFT's
`gap_probability > 70`, and that model returns ~67.1 for every symbol. It is now gated on a
**measured** overnight gap from the daily bars, and says so in the reasoning when no gap
qualifies -- the original failure was that it did nothing *silently*.

### The strategy dropdown: scalp out, reversal in

Fixing `gap` left `momentum` and `scalp` differing only by `threshold *= 0.8`. Measured
across 36 symbol x timeframe rows they were **byte-identical in every one** -- the score is
either far past the threshold or nowhere near it, so an 0.8x multiplier never changed an
outcome. `scalp` is also not a strategy but a hold-time preference. It was removed; an API
caller who still passes it falls through to plain momentum.

**`reversal`** replaces it, and is the reason the dropdown is now worth having: it FADES an
extended move rather than following it, so on the same bars it can take the opposite side.
It fires only when there is something to fade -- an RSI extreme the move is still running
into -- and when there is not, it **says so** and stands down rather than silently
degrading into momentum, which is precisely how `gap` hid for so long. RSI,
`volume_exhaustion` and `near_resistance`/`near_support` were all already computed by
`identify_current_setup` and discarded, the same as `directional_bias` was.

Measured over 16 symbols x 4 timeframes, against 5 permutations each:

```
strategy      fire(real)   FPR(perm)    lift
momentum           10.9%        3.4%    3.18
gap                18.8%        5.3%    3.53
reversal           14.1%        2.8%    5.00
```

Reversal disagrees with momentum on **19%** of symbol-timeframe pairs -- on NFLX momentum
said `PUTS` while reversal said `CALLS`, and on BA the three modes gave three different
answers on one tape. That is a control doing work.

**Caveat, stated plainly:** n=64 on the real side, so 14.1% is nine firings and a lift
computed from nine events moves around. These numbers say the modes are not reading noise
and are not each other. They are not evidence of a tradeable edge, and none of this has
been tested against forward returns.

**The direction gates were set by permutation test, not by eye.** Shuffling a real series'
returns destroys the trend while preserving volume, volatility and bar geometry, so
anything emitted on the shuffled version is a false positive:

```
 dead   agr   FPR(perm)  fire(real)   lift
 0.15   0.60      14.4%       10.0%   0.70   <- the original behaviour
 0.35   0.60       7.3%        8.8%   1.20
 0.40   0.75       4.8%        8.8%   1.83   <- chosen
 0.50   0.75       1.2%        3.8%   3.00
```

Read the first row: at the original settings the engine fired **more often on shuffled data
than on real data**. A lift below 1.0 is worse than useless. Lift only crosses 1.0 around a
0.35 deadband.

**1.83x is a modest edge and is documented as one.** The sample is n=80 real observations
on a single quiet tape, and nothing here has been tested against forward returns. It is
enough to say the engine is not reading noise. It is not a claim of a tradeable edge.

### Which session is this, anyway?

Reported at 22:22 ET on 2026-09-07: the app was serving stale numbers. It was --
**2026-09-07 was Labor Day**, a full closure, so the newest bar was from Friday the 4th.
Nothing on the page said so, which makes a closed exchange look like a broken feed.

Note what a naive fix would have missed: 7 September 2026 is a *Monday*. A
`weekday() < 5` check calls it a session. The exchange calendar is the only thing that
knows otherwise, so `trading_days.py` (the repo's maintained table, back to 2004, including
the irregular closures) is copied into the Space and held against drift by `sync_shared.py`
-- it is the file most likely to rot, because holidays get appended to the root copy each
year and nothing would otherwise notice the Space's copy ageing.

`market_session.market_status()` returns the state (`open` / `closed` / `premarket` /
`weekend` / `holiday`), the reason, and the **reference session** -- the day whose bars a
signal computed right now actually reflects. The banner renders before any request, from
the calendar alone, so it still appears when the backend is down. Early closes are handled
(the Friday after Thanksgiving shuts at 13:00 ET); an unavailable timezone database
degrades to `unknown` rather than being reported as "closed".

The claim is then **cross-checked against the data**: the newest bar actually fetched is
compared with the session the calendar predicted, and a mismatch surfaces as a warning
rather than being quietly asserted. A calendar can be out of date; the bars cannot lie
about their own timestamp.

### Timeframes: 15m / 1h / 4h / 1d

1m and 5m were removed. yfinance caps 1m history at 7 days -- far too few bars to warm up a
20-period indicator -- so those settings could never be more than a relabelling of coarser
data, which is the same class of error as a dropdown that does nothing.

4h is **not fetched**; yfinance has no 4h interval. It is folded from the hourly bars one
session at a time, because a plain `resample("4h")` buckets by wall clock and would splice
the overnight gap into an intraday range.

Every timeframe still reads **all four** bar sets -- the selection changes emphasis, not
which data is consulted, so a daily view still sees intraday deterioration.

### Why confidence does not move when you change strategy

A fair question, asked of exactly this: NVDA at 1h reads 34% under `momentum` and 34% under
`reversal`. That is deliberate. `weighted_confidence` measures **the quality of the
evidence** -- momentum conviction, the LLM layer, sentiment -- and the same bars are the
same bars however you choose to read them. `strategy_mode` never enters that formula. What
it changes is the **score** and the **gate**:

```
NVDA @ 1h        signal   conf   score     strategy note
  momentum       HOLD     33     +0.0118   -
  reversal       HOLD     33     +0.0000   Reversal: RSI 60, no extension to fade
  gap            HOLD     33     +0.0118   Gap mode: no gap measurable
```

The score genuinely differs. Both still land on HOLD, so the headline looked frozen. The
fix is legibility, not arithmetic: the strategy's own verdict is now surfaced on the result
card instead of being buried in a collapsed expander. Making confidence move with strategy
would mean inventing a number, and this codebase has enough of those in its history.

### The result card said nothing

Three separate complaints, all fair, all about the same thing: the UI reported a verdict
and withheld the reason.

**"Position Size 0.00" was noise.** It is a FRACTION OF CAPITAL, not a share count, and
the same name carries three different values at three nesting levels of the response --
master signal, momentum arm, per-timeframe setup. A unitless `0.00` on every HOLD,
duplicating what `options_strategy.contracts` already says, is not information. It now
appears only when there IS a position, labelled "% of capital".

**The verdict had no reason.** `HOLD, 17%` cannot distinguish *nothing is happening* from
*the timeframes disagree* from *confidence missed the gate by two points* -- and those
mean completely different things to anyone deciding whether to look again in an hour. The
API now returns `verdict.blocking_reason` plus the gates it measured against, so the card
shows the margin:

```
NVDA 1d   HOLD  conf 34/38   Movement, but no agreed direction: direction +0.27, needs ±0.40.
TSLA 1h   HOLD  conf 34/36   Direction is bullish, but confidence 34 is under the 36 gate.
NFLX 1d   PUTS  conf 39/38   (fired)
SPY  1h   HOLD  conf 17/36   No timeframe showed enough movement to act on.
```

TSLA there is **two points from firing**, which was previously invisible. The reason names
only the condition that actually blocked -- an earlier version listed tests that had
passed alongside the one that failed, which reads as if both were problems.

**The detail dump was ~200 lines of raw JSON.** It is still available, but the parts a
reader needs -- per-timeframe direction, momentum, RSI, ADX, trend, vol regime -- are
pulled into a table above it.

Two dead branches in `_generate_options_strategy` were found while doing this: one gated
on `timeframe in ["1m","5m"] and strategy_mode == "scalp"` (all three removed earlier that
day, so unreachable) and one on `confidence > 70` against a value whose measured ceiling is
52 -- the same unreachable-gate defect fixed in the signal path and missed here. 4h and 1d
had no branch at all and fell through to `CONSERVATIVE`.

### Verification

```bash
python verify_fix.py        # real bars reach the indicators; symbols differ
python indicators_test.py   # vectorised Wilder ADX/ATR/RSI == the live lab's
```

`verify_fix.py` runs the real path and asserts distinct per-symbol readings:

```
  symbol   timeframes      rsi      adx   direction
  NVDA              3     60.4     25.0          up      (24.9687 — computed, not the old constant)
  SPY               3     55.6     17.1        down
  TSLA              3     50.9     35.8        down
  AVGO              3     38.0     32.4          up      (no local_data snapshot — exercises the news fix)
```

1560 × 15m, 1256 × hourly, 502 × daily bars per symbol; the TFT's ≥96-row gate is now
satisfied. `indicators_test.py` checks the vectorised indicators against
`lab_indicators.py` — the live lab's dependency-free implementation, exercised every
session by a running forward test — and agrees to ~1e-14 across three seeds and two
periods.

---

## Shared files

Three paths exist in **both** this folder and the parent repository, held as **byte-identical
copies rather than imports or symlinks**:

| path here | path in the repo | also used by | versioned here? |
|---|---|---|---|
| `local_data/` | `local_data/` | `data_sources/unified_data_provider.py` | yes (752 KB) |
| `trained_models/` | `trained_models/` | `models/tft_backtest.py` | **no — build artifact** |
| `trade_analysis/lab_indicators.py` | `trade_analysis/live_lab/indicators.py` | the live forward test | yes |

`trained_models/` is git-ignored **on this side only**. The same 19 MB of weights is
already tracked at the repo root, and git keeps blobs permanently — committing them a
second time would double them in every clone forever, and no later `git rm` would undo it.
It is treated as what it is: a build artifact with a deterministic generator. That choice
is only free before the first commit, which is why it was made then.

The obvious failure mode of that decision is uploading a Space whose weights were never
generated, so `sync_shared.py` **exits non-zero** until they exist, rather than printing a
warning nobody reads.

Copies, because the two deployments must stay independent in both directions: this Space
installs from its own minimal `requirements.txt` and must not drag in the lab, and nothing
done to make a demo presentable may reach back and perturb a frozen forward test.

The cost of copies is drift, so:

```bash
python sync_shared.py          # report drift, change nothing
python sync_shared.py --push   # repo root  -> huggingface_space
python sync_shared.py --pull   # huggingface_space -> repo root
```

Direction is always explicit — guessing which side is authoritative is how one gets
silently overwritten. **Run the check before uploading.**

`lab_indicators.py` differs from its source by a provenance banner only, so it is compared
on code rather than bytes, and `--push`/`--pull` deliberately skip it.

---

## Known limitations

- `implied_vol_pct` is the **median IV of the nearest expiration's chain**, not an ATM
  IV surface and not a 52-week rank. It is named for exactly what it is.
- The TFT checkpoints in `trained_models/` are from an earlier training run whose model
  definition was lost; `tft_model.py` is a reconstruction from tensor shapes. Out-of-sample
  they show **no directional edge** (~50% hit rate). They are kept for reconstruction
  reference, not as a production signal.
- `local_data/*.json` are static snapshots checked into the repo, and `data.py` reads them
  **before** the cache and the API — so for those nine symbols news and Reddit are frozen
  at collection time. Delete a file to force live fetching for that symbol.
- Signals are analysis output. Nothing here places orders, and there is no broker path.
