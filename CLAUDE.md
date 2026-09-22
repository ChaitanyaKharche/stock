# Working notes for Claude on this repo

## HOW TO TALK TO ME — read this first

**Plain English before jargon. Always.** Give the answer in normal words first, then the
numbers if they're needed. I have told you more than once that you make things harder to
understand than they need to be. That is a defect in the explanation, not in me.

Specifically:

- **Lead with the conclusion in one sentence.** Not a table, not a preamble.
- **Use everyday comparisons.** "0.54 basis points on a $700 stock is about 4 cents" beats
  "+0.54 bp, CI [−1.70, +2.87]" every time. Put the cents first, the bp second.
- **Define a term the first time you use it in a reply.** bp, MFE, drift, CI, day-clustered.
  Do not assume the last explanation stuck.
- **Short.** If the point fits in five lines, use five lines. Long answers are not more
  rigorous, they are just longer, and they bury the thing I actually need.
- **Don't relitigate.** If something was measured and written up, cite it in one line and
  move on. Do not re-explain the whole chain every time.
- **Don't keep proposing new instruments.** I trade naked calls and puts. Not straddles,
  not spreads. If the data says calls and puts won't work, say that — don't offer a
  different structure as a consolation prize.
- **Metric units. No apologies, no fluff, no "great question".**

## WHAT IS SETTLED — do not re-run these

Cite these and move on. Every one is written up in `research/` with the numbers.

| finding | where | status |
|---|---|---|
| Direction prediction, 8 pre-registered nulls | `PROJECT_REPORT.md` | closed |
| MOMO_CHASE form, 2,822,400 cells, p=0.8227 | `setup_sweep_results.md` | closed |
| Unconditional short 0DTE straddle: real premium, 1.84× too small to trade | `vrp_cost_model_results.md` | closed |
| 8-line breakout on QQQ | `breakout_options_results.md` | null (wrong level set — superseded) |
| **His actual 6-line breakout on QQQ, 2,438 trades** | `six_lines_results.md` | **null** |

**THE PROFIT CAP SUBTRACTS VALUE. Measured three separate times:**

| sample | result |
|---|---|
| journal, 422 discretionary trades, dollars | +25% target → 71.6% win rate, still loses (breakeven needs 75.3%) |
| 8-line mechanical, 7,938 signals | cap worth **−1.17 bp** |
| 6-line, his exact set, 2,438 trades | cap worth **−0.72 bp** |

Three datasets, three level definitions, one answer. This is the most robust positive
*instruction* the project has produced: **stop capping winners.** His win rate goes up
and his money goes down, everywhere it has been measured.

## WHAT CANNOT BE MEASURED — stop offering

- **QQQ options, any expiry.** No archive exists. None.
- **Any 1DTE, any symbol.** The archive is `option_quote_1m_0dte/` — 0DTE by construction.
- **Anything new from ThetaData.** Lapsed to `Options: FREE` on 2026-09-08.

The only option archive that exists is **SPY 0DTE, 769 sessions, 2020–2024**, and the
2024 block can be spent exactly once.

## HOW THIS PROJECT FAILS — the one bug class

Every serious defect here has been the same shape: **no crash, no error, a confident
number that measured nothing.** A 1-minute lookahead once supplied 88–95.7% of a measured
edge. A coverage tool reported 100% with six sessions missing. A profit cap filled after
the position had already exited.

So:

- Before believing a number, ask what would make it wrong, and test that.
- **A number that changes when nothing changed is the cheapest bug detector there is.**
  Two runs disagreeing on identical data has caught two separate defects here.
- Every test must be verified to FAIL against the broken version. A test that passes on
  the bug is decoration.
- A gate that cannot pass in the environment it runs in is **broken, not strict**. Platform
  checks are SKIP, never FAIL.
- Wrong numbers get corrected in place with the reason, never quietly deleted. A record
  that removes its own mistakes is not a record.

## OPERATIONAL

- Tests: `pip install -r requirements-dev.txt && pytest` — offline, any OS, ~123 checks.
  Do NOT use `requirements.txt`; it is a pip freeze of the trading machine and no longer
  installable (`pandas_ta` was deleted from PyPI).
- The lab runs on his Windows box under pyenv 3.12.10, **not** the repo venv.
- `live_lab_data/` is a FROZEN forward test. Do not touch config hashes. `autostart`
  commits and pushes it at the end of every session.
- The live lab machine travels in a car. Network drops and host suspends are expected and
  are guarded, not bugs.

## STILL OPEN

- **HF token exposed in `~/.bash_history` on the shared Discovery filesystem —
  revocation unconfirmed.** Highest-priority item in the project. Not code.
- The unwritten selection rule: 92% of sessions break a line by 09:40. Whatever picks
  *which* break he takes has never been written down, and every measurement says the entry
  is null while the exits carry his record.

  **Do NOT say "he takes 1–2 trades around 11:39." Both numbers were wrong and he caught
  it.** The journal is **2.90 trades/day** (357 round trips / 123 days,
  `entry_timing_results.md:17-21`). And **11:39 is a median, not a cluster** — quoting it
  as the hour he trades assumes a single-peaked distribution that has never been checked.
  If his entries are bimodal, the median sits in the gap where he trades *least*. Corrected
  in `breakout_options_results.md` §5 and `six_lines_results.md` §6.

  **General rule this exposed: never quote a median as a location without the histogram.**
  A median of a two-humped distribution describes an hour that may hold no trades at all.
  This is the project's standard bug class wearing a new hat — a confident number that
  measured nothing.

- **The journal entry-time histogram HAS now been run** — 2026-09-22, on the lab machine,
  443 round trips over 158 days = **2.80 trades/day**. (The 2.90 figure above came from
  `entry_timing_results.md`, 357/123; both are right for their snapshot, the CSV has grown.)

  **Measured shape: busiest bucket 09:45 (41 trades). Median 11:44, roughly two hours
  later.** Right-skewed — a long thin afternoon tail drags the median well past the
  quarter-hour he actually trades in. `peaks()` returns two modes (09:45, 11:00) but the
  split clears by only **2.04 sigma** of counting noise, so **describe it as one peak plus
  a noisy declining plateau, not as two humps.**

- **OPEN and exploratory: his P&L alternates sign by time of day.** From the same run,
  whole book **+$1,502 over 442 trades = +$3.40/trade**:

  | window | n | total | per trade |
  |---|---|---|---|
  | 09:30–09:45 | 66 | **+$2,806** | +$42.52 |
  | 10:00–11:30 | 157 | **−$3,183** | −$20.28 |
  | 11:45–13:30 | 130 | **+$3,514** | +$27.03 |
  | 13:45–15:45 | 89 | **−$1,635** | −$18.37 |

  **Do NOT treat this as a finding yet, and do not let him trade it yet.** Three reasons,
  all of them this project's own settled lessons: (a) the window boundaries were chosen
  *after* seeing the histogram — 26 buckets with free choice of contiguous groupings is
  the garden of forking paths that `setup_sweep_results.md` spent 2,822,400 cells
  demonstrating; (b) no day-clustering, and 442 trades over 158 days are not independent;
  (c) P&L here is tail-dominated (73.9% in the top 1% of trades), so a bucket mean can be
  one fill. The `med $` and `top1 %` columns were added to the histogram for exactly (c) —
  a bucket whose mean and median disagree in sign is one trade, not an hour.
