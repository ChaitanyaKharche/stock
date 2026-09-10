# ProfitBook — where this project has got to

*Written 2026-09-10. Every number here is measured and traceable to a file in this repo.
Where a number was published and later found wrong, both are shown.*

---

## 0. What this project actually is

It started as one question: **can a discretionary 0DTE options strategy be automated
profitably?** The strategy was the user's own — an opening-range breakout on QQQ/SPY, buy a
cheap same-day-expiry option, take profit around +25%.

It is now three things running in parallel, because that one question turned out to need
three different kinds of evidence, on three different clocks.

| branch | question it answers | clock |
|---|---|---|
| **HPC** | Can this be settled with data we already have? | hours |
| **Live lab** | Can it be settled with data we don't have yet? | months |
| **HF Space** | Can it be *shown* to a stranger in 90 seconds? | minutes |

They are not three projects. They are one question, split by what kind of answer each part
can produce.

---

## 1. Phase one: the original question, answered

**Prediction is closed.** Eight pre-registered nulls, all on *direction*:

strike distance · position size · within-day ordinal · entry-timing percentile · a
19-feature ceiling model (OOS R² **−0.078**) · his own declared indicator stack (OOS R²
**−0.062**) · ten external published setups (**0 of 10** advance on 1,542 sessions × 2
symbols — the one well-powered test) · entry timing vs a same-contract control (null at
5/10/15/25 min).

`pullback_9ema`, the indicator he trusted most, measures **−0.0017 σ**.

### What the strategy actually was

He believes he trades opening-range retests. Measured against his own broker export, he is
a **midday momentum chaser**: median entry **11:39 ET**, and the prior 15 minutes moved
**+0.2029 σ in his direction 89.3% of the time**. Only **5.7%** of his entries fall in the
09:30–09:45 window the repo had been backtesting for months.

So the code was testing a strategy nobody was running.

### The one thing that survived

`IntradayMomentumBoundary` (Zarattini/Aziz/Barbon, SSRN 4824172) — on **shares**, not
options. 2,905 QQQ trades, 2016–2026, real NBBO:

- **+$3.34/trade, Holm p=0.0043**
- split-half both independently significant · 10 of 11 years positive · both directions
- survives slippage to 2¢/share/side · Sharpe 1.39
- **parameters are published and untuned** — the only setup here needing no invented number

And the cost of believing it:

- **top 1% of trades = 73.9% of all P&L**; exclude them and p = 0.1764
- win rate **27.2%**, median trade **−$6.81**
- **796 days (26 months) underwater** 2018-12 → 2021-03
- correctly annualised on *calendar* time: **9.2%/yr on $10k = $18/week**

### The verdict that closed phase one (2026-08-26)

His discretionary exits beat every mechanical rule tested — in return space by 8.97pp
(Holm p=0.027), but in **dollar** space on 422 trades the sign flips (+$4.22, p=0.487).
A +25% target raises his win rate to 71.6% **and still loses money**, because that
structure breaks even at 75.3%. His realised structure (avg win +48.1%, avg loss −37.0%)
breaks even at 43.5% and he runs 45.8%. **Capping winners destroys the asymmetry that makes
his record positive.**

The decisive fact: **more analysis cannot help.** QQQ per-trade SE is ~$8.57 against a mean
of $8.72 — t ≈ 1.0. Reaching t = 2 needs ~4× the observations: about **356 more QQQ trading
days, roughly 1.5 years**. The sample is the binding constraint, not the method.

That is why the live lab exists. It is the only instrument that produces new information.

---

## 2. The spine running through everything: one bug class

Every serious defect in all three branches has been the same shape — **no crash, no error
message, a confident-looking number that measured nothing.**

| where | the defect | what it cost |
|---|---|---|
| backtest | `find_entries` recorded a 5-min bar's *label*, filling 5 min before the signal existed | **88–95.7% of the entire measured edge**; two whole result sets void |
| backtest | zero-filled holiday closes gave `yesterday_high = 0` | ~7% of entries were fabrications that survived the timing fix |
| HF app | `llm_analysis.get("conviction", 50)` — a key that layer never sets | 60% of the confidence formula was structurally dead; **every ticker returned 15%** |
| HF app | `_convert_signal_format` hardcoded `return 'CALLS'` | PUTS was unreachable for any input; a selloff read as bullish |
| HF app | `min_confidence` 70/65/60/55 against a quantity whose measured max was 47 | fired on **0 of 80** observations |
| HPC | a halt bar with all-zero OHLC → `log(0) = −inf` | poisoned 170 sessions; `dropna` silently removed **193 of 769** |
| HPC | `exp(X @ beta)` with no `exp(s²/2)` | turned "IV beats HAR p=0.023" into **p=0.583** |
| live lab | no stale-**bar** guard to match the stale-**quote** guard | **21% of all trades** were signals up to 325 minutes old |
| live lab | `now` sampled before two network calls | preflight printed 15× `[OK] REAL-TIME (age −49.0s)` on future-dated quotes |

**The lesson that generalises:** a gate above the attainable range is *broken*, not strict.
A control that changes a constant rather than the logic is decoration. And a test that
cannot fail loudly is not a test.

The permutation method now used everywhere came out of this: shuffle a real series' returns
— killing trend while keeping volume, volatility and geometry — and anything still emitted
is a false positive. At its original settings the HF momentum engine fired **more** on
shuffled data than real (lift 0.70). It now runs 3.18× / 3.53× / 5.00× across its three
modes. Modest, and documented as modest.

---

## 3. The three branches

### 3.1 HPC — *what can be settled with data we already have*

**Northeastern Discovery.** `~/vrp/rep` (the clone), `~/vrp/data/vrp` (the frame),
`~/vrp/results`. Nothing lives on `/scratch` — it purges after 45 days and has already
destroyed the original TFT model-definition source on this account.

#### The question

All 8 nulls tested **direction**. This tests **variance** — and the distinction is not
cosmetic. A 0DTE straddle held to expiry pays realised variance against the variance
implied by the price paid. **No directional call is required at any point.**

The hint came from a bug in the *HF app*: `_calculate_momentum_score` is a pure magnitude
(every term `abs()`-wrapped) and the code was forcing that magnitude to emit a directional
CALL/PUT. The part of the system measuring something real was measuring **size, not sign** —
and no experiment had ever tested it on its own terms.

#### The answer (2026-09-09): the premium is real and 1.84× too small to trade

37,517 trades over 618 sessions. Sell the ATM 0DTE straddle, hold 30 minutes:

| | mean/trade | win rate | session-clustered t |
|---|---|---|---|
| GROSS (mid to mid) | **+$0.0205** | 68.6% | **+7.52** |
| NET (bid to ask) | **−$0.0171** | 61.0% | **−5.38** |

Spread cost $0.0376 against a $0.0205 edge. **The mid-fill assumption is the difference
between t=+7.52 and t=−5.38** — between a strategy and its opposite. That is the single
most valuable number in this project.

And the trap: **61% of trades win, median +$0.04, mean still negative, because the worst 1%
of trades carry 94% of the loss.** Short gamma in pure form. It would show a rising equity
curve and a 61% hit rate for months and then erase all of it.

Note the symmetry with IMB, which has 74% of its P&L in the *top* 1%. **In this project,
every apparent edge so far has lived in the tail.**

#### The forecasting benchmark (reproduced on the cluster, 2026-09-09)

769 sessions, 5-minute bars, target `rv_fwd_30`, validation = 2023:

| model | QLIKE | RMSE |
|---|---|---|
| implied_variance | 0.411136 | 0.016123 |
| **HAR-RV** | **0.430642** | 0.013771 |
| HAR-RV-J | 0.430788 | 0.013765 |
| persistence | 0.803472 | 0.016252 |

Diebold–Mariano vs HAR, **clustered by session** (n=250, not n=14,940 — minute origins are
~0.9 correlated and naive n inflates significance ~300×): implied variance vs HAR is
**t=−0.55, p=0.5832**. Statistically indistinguishable. HAR beats naive persistence at
t=+8.27.

Lookahead audit: deliberately inject one minute of future → QLIKE **improves 1.91%**.
That is a PASS, and it also calibrates: one minute of genuine future is worth ~1.9% here,
so any model claiming a much larger margin is claiming more than cheating would buy.

#### Running now: the zero-shot arm

Before training anything, the 2026 discipline is to check whether a pretrained forecaster
already wins. Chronos-Bolt-base, same frame, same scoring code:

| | QLIKE | vs HAR |
|---|---|---|
| chronos, 512 origins (8.5 sessions) | 0.792147 | t=+5.25, p=3.2e-07 |
| chronos, 1320 origins (22 sessions) | 0.793359 | t=+5.00, p=1.1e-06 |
| persistence | 0.803472 | t=+8.27 |

It **beats naive persistence**, so it is not emitting noise. It loses to six OLS
coefficients by a wide, session-clustered margin. Giving it 22 sessions of history —
matching HAR's `rv_prev_22` reach — changed nothing, so it was not under-informed.

Three things had to be fixed to get that number, each of which would have produced a
plausible-looking wrong answer:

1. **The horizon is 7 steps, not 6.** `build_vrp_dataset` leaves a one-bar hole on purpose
   (the bar stamped *t* contains post-origin information). Verified by scan: k=7 matches on
   **40,541 of 40,541** pairs, k=6 on **3 of 41,310**.
2. **chronos-bolt's `mean` is its median** — measured identical on 100.0% of origins. QLIKE
   scores a *mean*. Fixed by fitting a lognormal to the model's own returned quantiles.
3. **That fit exploded on negative quantiles.** Chronos is not constrained positive; one
   clamped to `EPS=1e-12` entered as `log = −27.6` and produced means of 3.7e+05. RMSE
   6.7e+11 caught what QLIKE could not.

The positive-control audit is on its third design and still being settled — see §5.

---

### 3.2 Live lab — *what can only be settled by waiting*

**Frozen 2026-08-28.** `research/forward_test_preregistration.md`, machine-readable
`live_lab_data/FREEZE.json`. ~3,200 lines across 13 modules with **no order-placement path
and no credentials** — it cannot trade even by accident.

**The clock never resets.** No session, setup, date, regime or symbol may ever be excluded.
A setup performing badly is a **result**, not a reason to restart its count.

#### Why it exists

IMB is the only thing that ever cleared Holm on a large sample *and* survived honest fills —
and it was **not being forward tested at all**, because it lives on shares and dies on
options. All 13 setups run, not just IMB: forward-testing the winner of a 13-way sweep alone
would be selection on the outcome.

#### State as of 2026-09-10

| session | trades | net | clean only |
|---|---|---|---|
| 2026-09-01 | 16 | −278.81 | −113.53 |
| 2026-09-02 | 22 | +5.77 | +0.97 |
| 2026-09-03 | 22 | +214.65 | +214.65 |
| 2026-09-04 | 20 | −103.84 | −103.84 |
| 2026-09-08 | 65 | −373.14 | −302.01 |
| 2026-09-09 | 130 | −1,339.77 | −988.11 |
| **total** | **275** | **−$1,875.15** | **−$1,291.87** |

**6 sessions of 60 required.** None of this is evidence yet, and the pre-registration says
so explicitly: confidence-interval width is driven by **sessions**, not trades, because
every bootstrap resamples dates and 15 correlated names collapse into one cluster.

The one number under test:

```
IntradayMomentumBoundary   n=39 clean   −$4.54/trade   vs backtest +$3.34
```

Wrong sign, nowhere near powered. The honest expectation for a verdict is **3–4 months**.

#### The options arm is dead

ThetaData lapsed to `Options: FREE` and the subscription cannot be renewed. Every 0DTE
session that will ever exist for this project already exists. Shares-only is the steady
state.

#### What the last week of work actually bought

Nine days, six of them with a defect that would have silently corrupted the record:

- a zombie Theta Terminal from 09-03 holding port 25503 while unresponsive → now evicted
- preflight warmup using `weekday() < 5`, so **Labor Day** (a Monday) passed as a session
  → now uses the exchange calendar
- `terminal_up()` returning True on a quote before the history service connected → now waits
- options-entitlement 403s aborting the **shares** arm too → now tagged and degrades
- my own `NameError` in an argument list, invisible because stderr wasn't teed → cost 23 min
- **no stale-bar guard** → 43 of 208 trades (21%) were signals up to 325 minutes old
- `now` sampled before two network calls → preflight green-lit future-dated quotes

Plus the **session ledger** (`live_lab_data/session_ledger.jsonl`): one durable line per
state change, written locally with no feed and no network, so it survives the failures it
documents. A missing daily file used to be ambiguous five ways — holiday, no trades,
aborted, machine off, crashed. Yesterday it wrote `PARTIAL … supervisor_rc: 0` at 16:00:03,
correctly recording a degraded session rather than letting a late start look clean.

---

### 3.3 HF Space — *what can be shown*

**https://huggingface.co/spaces/ckharche/ProfitBook** — Docker Space, FastAPI on :7860
internal, Streamlit on :8501 exposed. Purpose is **demonstrating engineering skill**, not
research. Deliberately shares **no code** with the live lab: nothing done to make a demo
look good may perturb a frozen forward test. Three paths are byte-identical copies with a
drift detector (`sync_shared.py`) that gates uploads.

#### What it was

It returned **HOLD at exactly 15% for every ticker**, forever. `data.py` fetched a one-row
frame, `identify_current_setup` bailed with defaults, and `0×0.4 + 50×0.3 + 0×0.3 = 15.0`.
A constant presented as a cautious model.

Fixing the data layer then exposed defects on code paths that had **never executed in
production**: `sklearn` missing from requirements; weight loading left as the comment
`# (The rest of your TFT loading logic...)` so the else-branch trained a neural network
inside an HTTP request handler; `resolve_tft_path` defined twice with the first a silent
stub; `ADX_9 = 25.0` hardcoded while gated on `adx > 25`.

#### The TFT, measured rather than trusted

398,854 parameters. Fed six different symbols' full histories, `gap_probability` moved
**0.10 on a 0–100 scale**. Out of sample over 120 sessions it emitted one constant direction
per symbol with hit rates equal to the majority baseline to the decimal.
`scaler_static` has every `scale_ == 1.0` — the static branch saw zero variance in training.

Re-measured 2026-09-09 across three deliberately opposite market regimes:

```
                  QQQ     NVDA    MSFT    META
calm_uptrend     68.4    67.0    69.7    63.1
violent_crash    68.3    66.9    69.7    63.1
flat_chop        68.4    67.0    69.7    63.1
```

A violent crash and flat chop differ by **0.1 percentage points**. MSFT and META do not move
at all. It was removed from the decision path in September and replaced by a six-parameter
HAR volatility forecast that demonstrably responds to its input — **a small model that works
is a better thing to show than a large one that does not.**

#### Current state

Three real strategy modes (momentum / gap / reversal, permutation lift 3.18/3.53/5.00×) ·
timeframes 15m/1h/4h/1d with 4h folded per session so the overnight gap is not spliced into
an intraday range · exchange-calendar session awareness · 285-symbol company-name
autocomplete · HAR forecast against implied vol from the **traded straddle** (yfinance's
`impliedVolatility` field is a placeholder — it read `0.00001` at ATM on a strike that
traded 38,104 contracts) · verdicts that explain themselves ("confidence 34 is under the 36
gate" — TSLA was two points from firing, previously invisible).

Known and not yet fixed: **no caching anywhere**, so every dropdown change refetches and
refits; and the dead TFT's output is still dumped in the UI.

---

## 4. How the three tie together

### They share one adversary

Not the market — **silent failure**. Every branch has produced a confident number that
measured nothing, and the countermeasures developed in one keep transferring:

- permutation testing was built for the **HF app** and is now the standard here
- session-clustered inference was built for the **HPC** arm and is why the live lab counts
  sessions rather than trades
- the lookahead audit was built for the **backtest** and is now run on every HPC arm
- the stale-bar guard found in the **live lab** is the same defect class as the HPC frame's
  halt-bar poisoning

### They feed each other concretely

```
   HF app bug ──────────────► HPC hypothesis
   (momentum score is a       (test VARIANCE, not
    magnitude, forced to       direction — the one
    emit a direction)          class 8 nulls missed)

   HPC result ──────────────► HF app content
   (HAR beats persistence     (ship the 6-parameter
    at t=+7.82; TFT is         model, delete the
    a constant)                398,854-param one)

   Phase-1 survivor ────────► Live lab existence
   (IMB clears Holm on        (the only thing worth
    shares, untested          forward testing, and
    forward)                   it was untested)

   Live lab reality ────────► everything else
   (−$1,875 over 6 sessions   (the only source of
    and 21% contamination      genuinely new data
    found in its own record)   in the project)
```

The nicest link in that diagram is the first one: **the entire HPC experiment came from a
bug in the demo app.** Nobody planned that.

### They differ in what they can prove

- **HPC** can rule things out fast and cheaply. It cannot produce new data — the 0DTE
  archive is closed forever, and the 2024 held-out block may be spent exactly once.
- **Live lab** produces the only new information, at one session per day, and cannot be
  hurried. 6 of 60 done.
- **HF Space** proves nothing about markets and is not trying to. It is the artefact that
  makes the rest legible to someone who will not read a pre-registration.

---

## 5. Where each stands right now

| | status | blocking on |
|---|---|---|
| **Phase 1 research** | ✅ closed, 8 nulls + 1 fragile survivor | nothing — do not reopen |
| **VRP trading question** | ✅ answered, decisive null | nothing |
| **VRP forecasting question** | 🔄 zero-shot arm running | the positive-control audit |
| **GPU sweep** | ⏸ not started | zero-shot result first |
| **Live lab shares arm** | 🔄 6 of 60 sessions | time, ~3 months |
| **Live lab options arm** | ⛔ dead | subscription, which will not be renewed |
| **HF Space** | 🟡 live and working | caching; removing the dead TFT display |

### Open items that are not code

- The HF token exposed in `~/.bash_history` on the shared cluster filesystem —
  **revocation unconfirmed.**
- Two inbound firewall ALLOW rules exposing Theta Terminal's port 25503 on a **Public**
  network profile. Needs an Administrator shell.
- Windows Time service is **not running** and NTP is unreachable; the clock is currently
  accurate but unverifiable.

---

## 6. What happens next

**Immediate (HPC).** Finish the zero-shot audit. The honest numbers — 0.792 at 512 origins,
0.793 at 1320 — have been stable across every rerun; three broken audit designs in a row
have been about whether we are *allowed to believe* them, not about what they say. Once the
control passes, this becomes a reportable null: **a 205M-parameter pretrained forecaster,
given matched history, loses to six OLS coefficients.**

That result then sets the bar for the GPU sweep. If Chronos cannot beat HAR, a 20k-parameter
MLP has a hard target — which is exactly the point of running the cheap arm first.

**Ongoing (live lab).** Do not touch it. Collect sessions. The first meaningful read is 60
sessions away, and the pre-registration forbids reading results before then. The only
legitimate work here is preventing lost sessions, which is what the last nine days were.

**Next (HF).** Add caching, and either delete the TFT display or relabel it honestly. The
second option is arguably the better demo: *"here is a 398,854-parameter model that
collapsed to a constant, here is the six-parameter one that didn't, and here is how I
measured the difference"* is a more interesting story than hiding it.

---

## 7. The honest summary

After roughly a year of work this project has produced **one candidate edge**
(IntradayMomentumBoundary, 9.2%/yr, 74% of P&L in the top 1% of trades, currently running
at the wrong sign in live forward test) and a **large, well-documented set of negative
results**.

That reads like failure and mostly is not. The negatives are load-bearing: eight
pre-registered nulls, a decisive cost-model null on the variance premium, and a demonstrated
measurement of what mid-fill assumptions are worth (**t=+7.52 versus t=−5.38 on the same
trades**). Each one closes a direction that would otherwise have absorbed months.

The single most transferable thing here is not a strategy. It is the habit that produced
this line in the record more than once:

> *Every number reported before 2026-09-09 on the 1-minute frame — including HAR-RV
> 0.187365 — was computed with defect (c) present and is superseded. They are not deleted:
> a record that quietly removes its own wrong numbers is not a record.*
