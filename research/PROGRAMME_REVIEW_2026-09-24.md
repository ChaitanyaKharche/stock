# Programme review — 2026-08-28 to 2026-09-24

Everything changed in the code, every finding, every defect, and where the three arms
stand. Written 2026-09-24. Numbers recomputed from the repository for this document, not
quoted from memory.

---

# PART 1 — THE LIVE FORWARD TEST, EVERY SESSION

**ATM arm only, both legs, paper.** 249 option trades over 14 sessions; 16 shares
sessions.

| date | options $ | trd | shares $ | trd | combined | note |
|---|---:|---:|---:|---:|---:|---|
| 2026-08-28 | −1,201.18 | 27 | no record | — | −1,201.18 | shares arm not yet live |
| 2026-08-31 | −1,012.21 | 15 | no record | — | −1,012.21 | |
| 2026-09-01 | −614.29 | 16 | −278.81 | 16 | −893.10 | options backfilled |
| 2026-09-02 | −126.53 | 19 | +5.77 | 22 | −120.76 | options backfilled |
| 2026-09-03 | **+1,246.23** | 22 | +214.65 | 22 | **+1,460.88** | options backfilled |
| 2026-09-04 | −466.62 | 20 | −92.82 | 17 | −559.44 | |
| 2026-09-08 | no record | — | −373.14 | 65 | −373.14 | **options record lost** |
| 2026-09-09 | no record | — | −1,339.77 | 130 | −1,339.77 | **options record lost** |
| 2026-09-10 | no record | — | −1,734.28 | 106 | −1,734.28 | **options record lost** |
| 2026-09-11 | no record | — | −1,647.21 | 135 | −1,647.21 | **options record lost** |
| 2026-09-14 | +856.79 | 15 | +1.71 | 143 | +858.50 | |
| 2026-09-15 | +553.38 | 20 | +395.71 | 139 | +949.09 | |
| 2026-09-16 | +88.46 | 19 | −1,316.51 | 155 | −1,228.05 | |
| 2026-09-17 | −789.05 | 13 | −1,311.19 | 134 | −2,100.24 | worst day |
| 2026-09-18 | −506.13 | 14 | −931.43 | 118 | −1,437.56 | |
| 2026-09-21 | **+3,280.95** | 13 | +552.01 | 131 | **+3,832.96** | best day |
| 2026-09-22 | +184.71 | 16 | +49.45 | 151 | +234.16 | |
| 2026-09-23 | +902.38 | 20 | +993.83 | 143 | +1,896.21 | |
| **TOTAL** | **+2,396.89** | **249** | **−6,812.03** | | **−4,415.14** | |

## What this table actually says

**The forward test is down about $4,400.** The options leg is up ~$2,400; the shares leg
is down ~$6,800 and is the entire loss.

**Both legs are tail-driven, and the tails are a handful of days.** Two sessions (09-03,
09-21) supply +$4,527 of the options leg's +$2,397 — without them it is roughly
−$2,100. Four sessions (09-09 to 09-11, 09-16) supply −$5,738 of the shares leg's
−$6,812. This matches what the research already found: 73.9% of P&L sits in the top 1%
of trades. Reading any single day as evidence is a mistake, in either direction.

**The last three days are green on both legs, but that is not new** and it is not a
turn. Both-green has happened on 5 of the 9 days where both legs have a record
(09-14, 09-15, 09-21, 09-22, 09-23) — it is the most common outcome, not a rarity.
On 2026-09-22 I accepted the claim that it had never happened and built an explanation
on top of it without checking the record. It had happened a week earlier.

**Four options sessions (09-08 to 09-11) have no record and never will.** The shares leg
ran; the options daily files do not exist. That is 4 of 18 sessions, against a
pre-registered gate of 60, and the options archive cannot be rebuilt because ThetaData
lapsed to `Options: FREE` on 2026-09-08.

## A defect in the record found while writing this

**Three daily files use a field name that means something different from every other
day.** The backfilled files for 09-01, 09-02 and 09-03 write `net`; every other day
writes `atm_net`. `atm_net` is the ATM strike arm alone. `net` is **all three strike arms
summed** — ATM, ATM−1, ATM+1 — so it is roughly three times larger for the same trading.

Read naively, 09-03 reads +$3,659.68 when the comparable figure is **+$1,246.23**, and
the whole-experiment total reads −$3,534 instead of −$4,415. The table above uses ATM
only throughout, recomputed from `trades.jsonl`.

**This is unfixed.** It is this project's standard bug class living in the record itself:
no crash, no error, two fields with the same apparent meaning and a 3× difference.
Anything that reads `daily/*.json` without handling both keys is wrong today.

---

# PART 2 — THE THREE ARMS, AND WHAT CHANGED IN EACH

## 2A. LIVE RUNNER — almost all the work

### Reliability: the machine stopped losing sessions

**`feed.py` — the hotspot problem (`e7bf6de`).** The lab laptop travels in a car and
switches WiFi ↔ mobile hotspot. 282 of 284 recorded outages were one error:
`HTTP 503: Unable to resolve host mdds-01.thetadata.us`. The upstream was down while
the local Theta Terminal was still answering, so `terminal_up()` could not see it and
the runner kept trading against a dead feed.

- Added `UpstreamUnreachable` and `TerminalUnreachable`, both under `FeedOutage`
- `_classify()` separates upstream-DNS failures from transient local ones
- `wait_for_upstream()` with `UPSTREAM_WAIT_SEC = 180`
- `_default_probe()` probes `/stock/history/ohlc`, **not** the quote endpoint — the quote
  endpoint answers from cache and would report healthy through an outage
- Outage-episode collapsing so one network switch is one episode, not 400 log lines

**`autostart.py`.** `_wait_for_history` previously only ran from `start_terminal()`, so a
mid-session outage was never waited out. Now runs on the already-up branch too, uses a
suspend-aware sleep, releases the system-awake lock after the archive push, and takes
`--no-archive` / `--no-push`.

**Result: 09-22 and 09-23 both ran with `outage_episodes: 0`.** 09-21 had 1.

### The record: it stopped being able to silently lose days

**`ledger.py` (`9f1691f`, `df25c2f`).** `coverage()` reported `usable 7/7 = 100.0%` while
six sessions were unrecorded, because it built its denominator from the ledger — it
divided the record by itself. Fixed with a calendar-walked denominator (`UNRECORDED: -1`
rank, `_last_closed_session()`); the same period then reported **7/13 = 53.8%**.

A second defect in the same file: `reconcile()` consulted the wall clock even when
`today` was injected, so the same arguments produced different records depending on the
hour they ran.

**`archive.py` (`47c6835`, `c17aaa0`) — new.** Committing by hand was the plan, and it
produced the six-session hole. Commits `live_lab_data/` at 16:00. Deliberately narrow:
never raises, explicit pathspec (never `git add -A`, because `.env` holds API keys),
never force/rebase/amend, refuses detached HEAD, never commits an empty change.

Extended 2026-09-24 after the push failed non-fast-forward on four consecutive sessions:
one recovery — fetch, **merge**, push once — and only when git says the remote moved
ahead. Refuses on a dirty tree, aborts on conflict, one attempt.

### Research tooling built (all new, all tested)

| module | lines | what it does |
|---|---:|---|
| `six_lines.py` | 400 | the trader's actual 6 levels, rolling; tally + one trade/session |
| `after_break.py` | 280 | what happens after R1/S1 breaks early |
| `early_r1_hold.py` | 280 | A/B/C arms for removing the close-back-inside stop |
| `entry_histogram.py` | 235 | discretionary journal entry-time shape |
| `orb_veto.py` + backtest | — | close-based range-expansion veto |
| `breakout_levels.py` / `breakout_sweep.py` | — | the (superseded) 8-line sweep |

### Housekeeping

- `requirements.txt` had not been installable since PyPI deleted `pandas_ta==0.3.14b0`
- `indicators_pandas.py` (198 lines) replaces it — `ema` SMA-seeded to match the live lab
- `pytest.ini`, `requirements-dev.txt`, `.github/workflows/tests.yml` (3.11 + 3.12)
- **Test suite went from "everything fails" to 185 passing**, offline, any OS

## 2B. HUGGING FACE SPACE — three files, one real bug

Touched only where `pandas_ta` removal reached it:

- `huggingface_space/trade_analysis/indicators_pandas.py` — the replacement, synced
- `huggingface_space/sync_shared.py` — sync path updated
- **`huggingface_space/trade_analysis/train_tft.py` — a real bug.** It read
  `bbands(...).iloc[:, 0]` as the **upper** Bollinger band. `bbands` returns
  `(BBL, BBM, BBU)` — column 0 is the **lower** band. The model was trained with upper
  and lower inverted. Now read by name.

**Nothing else in the HF arm was changed, and no caching work was done.** Offered, not
taken.

## 2C. HPC ARM — untouched in this block

`git log --since=2026-09-16 -- trade_analysis/hpc/` returns **nothing**. The HPC work
finished immediately before this block:

- `65afbb9` **MOMO_CHASE sweep: null at p=0.8227 over 2,822,400 cells. Form closed.**
- `3f6179e` dropped `srun` after array 10373735 failed 15 tasks on an inherited CPU-bind
- `7c9fbff` re-sized from a compute-node benchmark; the laptop estimate was 3.5× optimistic
- `881171c` three sbatch bugs found by the login node killing conda
- `8860e1f` the V0 percentile label read as its own opposite

**`trade_analysis/hpc/` is 18 files with no test file at all** — `har_baseline.py`,
`build_vrp_dataset.py`, `train_vrp.py`, `conditional_vrp.py`, `cost_model.py` and the
sbatch submitters. That gap was flagged and never closed.

**Still open and not code: the HF token exposed in `~/.bash_history` on the shared
Discovery filesystem. Revocation unconfirmed. Highest-priority item in the project.**

---

# PART 3 — WHAT WE FOUND

## Closed, with numbers

| finding | evidence | verdict |
|---|---|---|
| Direction prediction, 8 pre-registered nulls | `PROJECT_REPORT.md` | null |
| MOMO_CHASE form, 2,822,400 cells | p=0.8227 | null |
| Unconditional short 0DTE straddle | real premium, 1.84× too small | untradeable |
| 8-line breakout, QQQ, 7,938 signals | +1.44 bp, CI [−0.97, +3.91] | null (wrong level set) |
| **6-line breakout, his exact set, 2,438 trades** | +0.54 bp, CI [−1.70, +2.87], p=0.61 | **null** |
| R1/S1 near round numbers (1, 5, 10, 25) | ratio ≈ 1.00, both price eras | **null** |
| S1 downside early break | +4.2 pp, p=0.242 | null |
| Removing the close-back-inside stop | +1.84 bp, CI [−13.02, +16.81] | **null, all 3 gates failed** |

## The one real conditional signal

**An early R1 break predicts continuation.** Break R1 in the first two 10-minute bars and
price reaches R2 **70.7%** of the time against **59.3%** for a later break — and it
survives a time-matched control (both groups given the same 60 minutes): **+11.3 pp,
p=0.002**. 93% of the sessions that reach R2 close through it; 46% carry on through R3.

**It does not turn into money.** Terminal return is unchanged, which is the fourth
independent arrival at the same sentence.

## MAGNITUDE, NOT SIGN — four independent measurements

| where | what |
|---|---|
| `vrp_preregistration.md` §0 | the momentum score was a magnitude forced to emit a direction |
| `breakout_options_results.md` §4 | MFE +34.26 bp, terminal drift indistinguishable from zero |
| `six_lines_results.md` | median MFE far above median terminal move |
| `early_r1_hold_results.md` §3 | early R1 reaches R2 more often, terminal return unchanged |

**These levels predict how far price travels, never which way it ends up.** That is why a
naked call or put keeps failing on them: you can be right about the travel, pay the
variance risk premium for the convexity, and be paid nothing for the direction.

## THE PROFIT CAP SUBTRACTS — three independent datasets

| sample | result |
|---|---|
| journal, 422 discretionary trades, dollars | +25% target → 71.6% win rate, still loses (breakeven 75.3%) |
| 8-line mechanical, 7,938 signals | cap worth **−1.17 bp** |
| 6-line, his exact set, 2,438 trades | cap worth **−0.72 bp** |

**The most robust positive instruction the project has produced: stop capping winners.**

## Descriptive facts worth keeping

- **92.0% of sessions break at least one of the six lines**, median first break 09:40 —
  a break is not a selective event
- Break rates are **perfectly monotone in distance** (R 55.4/42.8/38.1, S 46.7/33.3/27.0),
  which is what any six lines at those distances would produce
- Median RTH range **125 bp** against the ~20 bp a +50% ATM gain needs — the move is
  there six times over; the direction is not
- Journal: **2.80 trades/day** (443 round trips / 158 days), busiest bucket **09:45**,
  median 11:44 — right-skewed, one peak plus a noisy tail

## Open and explicitly not yet a finding

His journal P&L alternates sign by time of day (+$2,806 in the first 15 min; −$3,183 from
10:00–11:30; +$3,514 from 11:45–13:30; −$1,635 after 13:45, on a whole book of +$1,502).
**Not tradeable yet:** the window edges were chosen after seeing the histogram, there is
no day-clustering, and the P&L is tail-dominated.

---

# PART 4 — WHAT WENT WRONG

Every serious defect in this project has the same shape: **no crash, no error, a
confident number that measured nothing.**

## In the measurement code

1. **The profit cap was not implemented at all.** `evaluate_signal` coded exits 2 and 3
   and omitted exit 1 — the cap the strategy is built around — and reported a −13.79 bp
   median as the strategy. It measured "hold until the breakout fails", nearly the
   opposite. Found because the trader pushed back on the result, not by any test.
2. **Fixing (1) introduced lookahead.** `if cap_px is not None: break` let the cap
   register on a bar after the position had already exited. Caught because two runs over
   identical data disagreed (MFE 34.26 → 40.70, cap-hit 64.8% → 74.5%). The inflated run
   reported **+4.98 bp, p=0.0003** — published and withdrawn, kept in the record as VOID.
3. **The wrong level set.** The first sweep used three prior sessions with premarket
   merged into RTH, which deletes yesterday's premarket lines on most days. A null about
   levels he does not draw.
4. **The coverage tool divided the ledger by itself** and printed 100% with six sessions
   missing.
5. **`ledger.reconcile` consulted the wall clock** even with `today` injected — same
   inputs, different output by hour.
6. **Progress counting counted successes, not attempts**, so a sweep that was skipping
   thin days looked frozen for two hours.
7. **`bar_cache_test.test_fan_out` returned a bool.** pytest ignores return values, so
   `return False` passed.

## In the analysis, not the code

8. **"He takes 1–2 trades around 11:39."** Both numbers wrong. The journal is 2.90/day
   (now 2.80), and 11:39 is a **median, not a cluster** — quoting it as the hour he
   trades assumes a single-peaked distribution nobody had checked. §5 of
   `breakout_options_results.md` was withdrawn: the frequency gap it was built on
   (claimed ~3×, actually 1.17×) does not exist.
9. **A true number supporting a false conclusion.** "54.3% of stopped-out sessions later
   closed beyond R2" is correct. "Therefore the stop is throwing away winners" is not —
   a trade held to 16:00 does not exit when R2 breaks at 11:20. **Never read a statement
   about the PATH as a statement about the ENDPOINT.** Caught only because the claim was
   turned into a pre-registered experiment with the decision rule fixed in advance.
10. **Accepting a premise without checking it.** Told that both arms green had never
    happened, I explained why it was rare instead of opening `live_lab_data/daily/`. It
    had happened twice the previous week; it is the most common outcome.
11. **Nine pre-registrations still said "NOT YET RUN"** after their results had landed,
    which made the repository lie about its own state to the next reader.

## In the ported indicators

12. **`train_tft.py` read the lower Bollinger band as the upper one** for the whole
    training history.
13. My own `indicators_pandas` port had two bugs the tests caught (MACD signal seeding,
    `ema` using `ewm`), and **a 1e-9 tolerance let a 2%-wrong EMA pass** — tightened to
    1e-12.

## In the tests themselves

14. **A `--force` mutant passed every behavioural test in `archive_test.py`.** By the
    time the recovery pushes, the merge already contains the remote's work, so
    force-pushing a merge commit destroys nothing *in that scenario*. Fixed by asserting
    on the git argv directly.
15. **An is-ancestor check that a rebase would have survived** — rebase preserves the
    other side's sha and rewrites yours; I was checking the wrong one.
16. Several of my own test *premises* were wrong while the code was right: 20 one-minute
    bars is 2 ten-minute bars not 3; an unbalanced bimodal sample puts the median inside
    a cluster. Each time the fix was to the test, not the code.

## The practice that caught most of it

- Pre-registration with the decision rule fixed **before** the run — this is what killed
  (9), which no amount of staring would have
- **Two runs disagreeing on identical data** — the cheapest bug detector available; it
  caught (2)
- **Mutation testing every new test** — a test verified to fail against the broken
  version. This caught (14) and (15), both of which were decorative until then
- The trader pushing back on a result that did not match his experience — caught (1)

---

# PART 5 — HOW WE PIVOTED

| from | to | why |
|---|---|---|
| 8 lines, 3 prior sessions, premarket merged | **6 lines, his exact spec** | the first set was not what he draws |
| touch-based breaks | **close-based** | a wick is not expansion |
| every break (3.4/day) | **first break only, 1/day** | matches what he does |
| assume the cap helps | **measure it** → it subtracts, 3 datasets | |
| assume the entry carries the edge | **the exits carry it** | every entry test is null |
| chase direction | **accept it is magnitude** | 4 independent measurements |
| commit the record by hand | **`archive.py` at 16:00** | by hand produced a 6-session hole |
| trust `terminal_up()` | **probe the historical upstream** | the terminal answers while upstream is dead |

**The largest pivot is the one that has not been acted on.** Every entry measurement is
null and every exit measurement says the exits are where his record lives — but the
selection rule he actually uses has never been written down, and no experiment has been
built to capture it.

---

# PART 6 — WHERE WE STAND

## Closed. Do not re-run.

**The six-line family is finished on all four of its parts** — entry timing, level set,
profit cap, and stop. Plus the 8 pre-registered direction nulls, the MOMO_CHASE form, the
unconditional short straddle, and round numbers.

## Cannot be measured. Stop offering.

- **QQQ options, any expiry** — no archive exists
- **Any 1DTE, any symbol** — the archive is `option_quote_1m_0dte/`, 0DTE by construction
- **Anything new from ThetaData** — `Options: FREE` since 2026-09-08

The only option archive that exists is **SPY 0DTE, 769 sessions, 2020–2024**, and the
2024 block can be spent exactly once.

## Open, in priority order

1. **HF token in `~/.bash_history` on the shared Discovery filesystem. Revocation
   unconfirmed.** Not code. Highest priority in the project.
2. **Two inbound firewall ALLOW rules expose Theta Terminal port 25503 on a Public
   network.** Warned on every preflight for at least four sessions. Needs an
   Administrator shell.
3. **`net` vs `atm_net` in the daily files** (Part 1). Unfixed; anything reading those
   files is wrong today.
4. **Four options sessions (09-08 to 09-11) permanently missing**, against a 60-session
   gate.
5. **`trade_analysis/hpc/` has 18 files and zero tests.**
6. **The unwritten selection rule** — the last unexamined part of his process.
7. The time-of-day P&L pattern — exploratory, needs pre-registration and day-clustering.

## The honest summary

The engineering is in good shape: 185 tests, the feed survives the car, the record
commits and now pushes itself, and the repository no longer lies about what has been run.

**The research has produced one real conditional signal and no tradeable edge.** The
forward test is down $4,415 on paper. Every direction hypothesis tested has come back
null, and four separate measurements agree on why: these levels describe how far price
moves, not which way it ends up.

The one instruction the data supports without qualification is **stop capping winners**.
