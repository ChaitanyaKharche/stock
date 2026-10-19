# START HERE — recovered project context

Written 2026-10-13 by the lead session, from seventeen agent reports in this folder. The
owner lost the Claude account that held three months of working sessions on this project.
This folder rebuilds that context from four sources: the code, the research write-ups, the
git history, and the old chat logs, which were still on the laptop.

**If you are a new assistant: read this file, then `CLAUDE.md`, then report 16 (where the
old chats stopped). Open the other reports when you need the detail.**

How far to trust each line here:

- **[checked]** — the lead verified it against a file or log on 2026-10-13.
- **[report NN]** — an agent read the code or the chats and said so; the lead did not re-check.
  Each report cites `path:line` for its claims.
- **[chat]** — it comes from an old conversation. The logs were condensed (tool output dropped,
  long answers clipped at 7,000 characters), so numbers from chats are the weakest evidence here.

---

## 1. The project in one paragraph

The owner trades same-day-expiry (0DTE) calls and puts on QQQ and SPY by hand, buying when
price breaks one of six lines he draws before the open. The project began in July 2026 as
"ProfitBook": turn that manual method into code, then have it recommend contracts. On
2026-08-16 a 5-minute lookahead bug was found to have manufactured almost the whole measured
edge. From then on the question changed from "automate my edge" to "is there an edge at all",
answered by pre-registered experiments — the decision rule is written down before the data is
looked at. The answer so far, many times over: **no setup predicts direction.** What is left
running is a frozen paper-trading forward test, whose job is to settle the question with new
data instead of more analysis.

## 2. The three legs

### Leg 1 — the live runner ("live lab"), on the Windows laptop

- **What it is.** An unattended paper-trading forward test. It never places an order
  [report 01]. Two programs trade the same 15 fixed rules: an options arm (0DTE calls and puts
  on QQQ and SPY, three strikes per signal) and a shares arm ($10,000 per trade in 15 tickers).
- **Why it exists.** Backtests kept producing confident numbers that turned out to be bugs. A
  forward test cannot look ahead. Rules were frozen on 2026-08-28; the clock never resets and
  bad days count [report 12, chat].
- **The core guarantee.** Each decision is written to disk before any option price is
  requested, so no price can influence the decision that used it [report 01].
- **A normal day** [checked against the 2026-10-13 log; code references in report 02]:

  | laptop time (MST) | New York (ET) | what happens |
  |---|---|---|
  | 05:30 | 08:30 | Windows Task Scheduler task `LiveLab-QQQ-SPY` runs `run_lab.cmd` from the **main checkout**, on pyenv Python 3.12.10 |
  | 06:05 | 09:05 | supervisor (`autostart.py`) wakes, starts the ThetaData terminal, runs a preflight on 15 symbols |
  | 06:07 | 09:07 | both arms start and warm up on 25 prior sessions |
  | 06:30 | 09:30 | market opens; first signals within two minutes |
  | 06:33 | 09:33 | second preflight proves quotes are real-time |
  | 12:56 | 15:56 | everything still open is closed |
  | 13:00 | 16:00 | arms write a daily summary and exit; supervisor commits the day's record and pushes it to `main` |
  | 13:30 | 16:30 | last scheduled repeat: a clean-up pass for positions that never got an exit |

  Arizona has no daylight saving, so the MST column shifts by an hour on 2026-11-01.
- **Where the record lives.** `live_lab_data/` (options) and `live_lab_data/shares/`.
  Append-only. Mistakes are corrected by appended lines, never by editing rows.
- **Today, 2026-10-13** [checked]: ran clean, no restart. Options arm 57 trade rows,
  at-the-money net −$548.54. Shares arm 167 trades, net +$11.50. Committed as `48a76f2` and
  pushed at 16:00:25 ET. Detail in report 17.
- **Running total** [chat, report 16]: about −$16,239 through 2026-10-08 — options −$1,012,
  shares −$15,227. About 97% of the shares loss is the 13 symbols added on 2026-09-05; QQQ and
  SPY are roughly flat.
- Reports: **01** (the trading path), **02** (launch, supervision, outage recovery), **17**
  (watched live today).

### Leg 2 — the HPC cluster (Northeastern Discovery, Slurm)

- **What it is.** The place the heavy experiments ran: a volatility-forecasting contest and a
  2.8-million-combination parameter sweep.
- **Why it exists.** While fixing the demo app on 2026-09-08, the momentum score turned out to
  measure how *far* price moves, never *which way*. That raised a new question: is there money
  in predicting size instead of direction? [reports 03, 05]
- **What came out.**
  - Selling a 0DTE straddle does collect a real premium: about $2 per straddle before costs.
    The bid-ask spread costs about $3.76. Net loss. "Real, and 1.84× too small to trade."
    [report 03, quoting `research/vrp_cost_model_results.md`]
  - A simple statistical forecast (HAR, a regression on recent volatility) matched
    option-implied volatility and beat a pretrained transformer. No neural model was ever
    trained for it, by decision. [report 03]
  - The momentum sweep: 2,822,400 combinations, p = 0.8227 — the best one is no better than
    the best of that many coin-flips would be. Closed. [report 04]
  - The older Temporal Fusion Transformer (TFT) outputs a constant. Its 19 checkpoint files in
    `trained_models/` are leftovers. [report 03]
- **Traps.** `/scratch` is wiped in full on the first Tuesday of each month (several docs still
  say "45 days"). The 2024 block of option data (151 sessions) is held out and can be used
  once; nothing but convention protects it. [report 03]
- Reports: **03** (variance work, the cluster itself), **04** (the sweep, the data archive).

### Leg 3 — the Hugging Face Space (public demo)

- **What it is.** A Streamlit web app at `huggingface.co/spaces/ckharche/ProfitBook`. Separate
  from the lab: it shares no imports, only five files kept as checked copies by
  `huggingface_space/sync_shared.py`. [report 05]
- **Why it exists, in three stages.** A coursework-era "AI signal generator"; then, in
  September 2026, an engineering showcase after it was found to return the same constant for
  every ticker and was repaired bug by bug; then, from 2026-09-24, a research showcase that
  opens on what the research found and keeps the signal generator as a labelled demo. [report 05]
- **State.** The rebuilt version was never uploaded — the token was invalid [chat, report 16].
  The upload procedure exists only in a memory note [report 05].
- Report: **05**.

## 3. What the research settled

`CLAUDE.md` holds the table. In plain words:

- **Direction cannot be predicted with these tools.** Eight pre-registered tests, all null.
- **His six lines predict how far price travels, not which way it ends up.** That is why
  buying calls and puts on a break keeps failing: you can be right about the move and still
  collect nothing for direction. [report 06]
- **His own entry habit is recoverable and does not pay.** A model can tell where he would
  enter (he chases momentum, about 1.1 ATR above the 5-minute 9-EMA); traded mechanically it
  earns about nothing. [report 07]
- **One real conditional signal:** an early break of the first resistance line reaches the
  second more often (70.7% against 59.3%). It does not turn into money. [report 06]
- **The programme's own verdict:** prediction is closed; more analysis cannot settle whether
  his discretionary trading has an edge, only about 1.5 more years of trading can. [report 07]

Do not re-run any of it. `CLAUDE.md` says so, and so did he.

## 4. Where `CLAUDE.md` is wrong or stale

Nobody has edited `CLAUDE.md`. These are for the owner to correct or approve.

| `CLAUDE.md` says | what the files say | status |
|---|---|---|
| ThetaData options access lapsed on 2026-09-08; nothing new can be measured | The options arm has traded every session since 2026-09-14, including today. A comment at `runner.py:53-57` says the entitlement came back that day. Tier and reach unknown. | **[checked]** for today's trades; rest [reports 01, 02, 04] |
| Profit cap subtracts value: "journal, 422 trades, +25% target → 71.6% win rate, still loses" | 71.6% and 75.3% come from a **296-trade** test (`construction_results.md:56,82`). On the 422 trades in dollars the same target **gains** $4.22 a trade, p = 0.487 (`journal_reverse_engineering_results.md:235`). The two mechanical measurements (−1.17 bp, −0.72 bp) stand; the journal one is fragile. | **[checked]** |
| "73.9% in the top 1% of trades" describes his journal P&L | That figure is the IMB shares strategy (`imb_stress_results.md:69`). | **[checked]** |
| The 2,822,400-cell result is in `setup_sweep_results.md` | It is in `momo_sweep_results.md`. | **[checked]** |
| "QQQ options, any expiry. No archive exists. None." | No QQQ archive like the SPY one exists, but QQQ option caches are on disk: 771 QQQ files in `optcache`, 29 QQQ chains in `.cache/backfill`, plus a hash-named ThetaData cache. Contents not inspected. "None" is too strong. | [report 09] |
| SPY 0DTE archive: "769 sessions, 2020–2024" | 769 files, 2020-01-03 to 2024-08-07. 2021 has only 39 sessions; 2024 stops in August. | [reports 04, 09] |
| Tests: "~123 checks" | 327 test functions in 25 files. | [report 08] |
| One lookahead incident ("88–95.7% of a measured edge") | Two different bugs: a 5-minute label bug (88% of the QQQ signal) and a 1-minute quote-stamp bug (95.7% of the IMB option return). | [report 04] |
| No mention that Six_Lines trades live | Two six-line setups were added to the lab by dated amendment on 2026-09-25; the family went from 13 to 15. | [reports 01, 06] |

## 5. Security — the owner's to do, and first

None of this is code. All of it is from reports; the lead did not open any key.

- **Keys in git history, rotation recorded as "NO":** an Alpaca key and secret
  (`trade_analysis/5min_alpaca.py:10-11` in commits `e17df27`, `f384d81`); Finnhub, Twelve Data
  and Reddit keys (`clean_trade_venv.sh:112-115` until `9e52f44`, and `slurm-1612669.out:43-46`).
  Deleting a file does not remove it from history. If the GitHub repo is public these are
  public. [report 08]
- **Pasted into old chats,** which are plain files under `C:\Users\chaitanyakharche\.claude\projects\`:
  a Hugging Face token, the university/cluster password, a Databento key, a ThetaData key, a
  Robinhood password, and API keys for an unrelated side project. [reports 11, 12, 14, 15]
- **Hugging Face token in shell history on the shared cluster filesystem:** revocation still
  unconfirmed. Already `CLAUDE.md`'s top open item.
- **The paid data feed is reachable from the local network.** The ThetaData terminal listens on
  every interface (ports 25503 and 25520) and Windows has two inbound allow rules for it on the
  Public profile. The lab prints this warning every morning. [reports 02, 17] He has asked
  not to be nagged about it [chat, report 13]; it is listed here once, for completeness.
- **A Robinhood session file** (`~/.tokens/robinhood.pickle`) was to be deleted; not recorded as
  done. [report 11]
- **A plaintext `creds.txt`** sits in `Documents\research_data\thetaterminal`. Not opened by
  anyone during this rebuild. [report 09]

## 6. Things that could bite, found during this rebuild

Reported by agents, not re-checked unless marked. None is a reason to touch the frozen lab
without the owner deciding.

1. **2026-11-02, the first session after daylight saving ends.** The scheduled task repeats for
   8 hours from 05:30 local with `StopAtDurationEnd = True`. In winter that window ends at
   15:30 ET, 25 minutes before the lab closes its positions. Whether Task Scheduler would kill
   the lab then is **untested**. The 16:30 ET clean-up pass would also disappear. [report 02 §3.1]
   A second calendar item: **2026-11-27 is a half day** (13:00 ET close). The shares arm has
   early-close handling; the options arm has none. [report 01; chat, report 13]
2. **"Frozen" is a hash of declared parameters, not of code, and nothing enforces it.** A
   changed config silently gets a new hash; the dashboard then silently drops those trades.
   Code can change with the hash unchanged. [report 01]
3. **Options stops and targets can fire on the entry bar.** 14 of 210 such exits happened 6–8
   seconds after entry; the shares arm skips those bars. Seen live today: an options trade
   stopped out 5.7 seconds after entry at 09:40:08 ET. [report 01; report 17]
4. **One rule exits at different times in the two arms.** `TTM_Squeeze` closes after about 10
   minutes in options and 45 in shares, because `positions.py:123` lacks a ×5. [report 01]
5. **`trade_corrections.jsonl` must be applied when reading the record.** The dashboard does
   not apply it. [report 01]
6. **A `COLLECTED` ledger row does not mean a clean day.** 2026-10-05 and 10-06 are marked
   `COLLECTED` after losing hours to a dead network; 10-05, 10-06 and 10-07 have no daily
   summary. Today had an 8-second outage and a burst of stale quotes that never reached the
   log. [report 02; report 17]
7. **51 shares positions from 2026-09-08 and 09-10 have no exit** and are reported three times
   a day as "left for a human". [checked in today's log]
8. **The 5-minute lookahead bug is still in the old backtest source**
   (`options_premium_backtest.py:152-159`). The fix lives only in `decision_time_audit.py`.
   Four of six affected modules were never re-run. Their numbers are void. [report 04]
9. **Legacy "live trading" scripts exist** in `trade_analysis/live_trading/`, written for
   Alpaca. Report 09 found no path in committed code that can send a real-money order: one
   script is stubs with `paper=True` hard-coded, one has no order call, one has order calls but
   nothing builds a client for it, and the Alpaca package is not installed in any Python on
   this machine. The folder is orphaned — nothing live imports it. Leave it that way. [report 09]
10. **Some data exists in one copy only,** gitignored, with no backup recorded: the 667 MB
    SPY 0DTE archive in `Desktop\data`, `research\round_trips.csv` (his round trips), and the
    Robinhood exports. Losing the laptop loses them. [report 09]
11. **The lab's Python is not the test environment.** Lab: pyenv 3.12.10 with numpy 2.3.5.
    `requirements-dev.txt` and CI pin numpy 1.26.4. Never install either requirements file
    into the pyenv interpreter. The repo `venv` is a Databento environment, not the lab's.
    [reports 08, 09]

## 7. How the owner wants to be worked with

Collected from `CLAUDE.md` and every chat digest. Where two disagree, the later one is listed.

- **Conclusion first, in plain English, short.** One line when the question is simple.
  Everyday units before statistics (cents before basis points). Define each term.
- **Go or no-go.** A verdict, then the reason.
- **He buys calls and puts. Nothing else.** Never propose selling a leg, spreads or straddles.
- **Do not re-run or re-argue settled work.** Cite it in one line.
- **Measure before declaring something dead.** Do not rule things out from priors.
- **Pre-register first** when testing anything new: decision rule fixed before looking.
- **Exact bookkeeping.** He catches miscounts.
- **Never overwrite the record. Never change frozen rules because of early results.
  Read-only means read-only. Do not break the live lab.**
- **New experiments go in new files** that import from proven code; never edit proven code.
- **Simple beats layered.**
- **Label every command** as laptop PowerShell or cluster bash, and say where a job ran.
- **Do not push him to spend money.** Do not write code that needs a data tier he has not
  bought. Cluster time is free — do not argue from compute cost.
- **Finish work unattended; deliver findings without being asked twice.**
- **Never print secrets.**
- **Git:** in August, "no commits or pushes without approval" [report 11]. By late September,
  "work on `main`, no worktrees, commit and push" [report 16]. The later one stands for his
  own sessions — but the lab runs from the main checkout, so anything done there is live at
  05:30 the next morning.

## 8. Open questions only the owner can answer

1. **ThetaData:** it lapsed on 2026-09-08 because of cost [chat, report 13] and was working
   again by 09-14. Did you renew, at what tier, and how far back does it reach? This decides
   whether "QQQ options cannot be measured" is still true.
2. **Premarket prints around 08:00–08:05 ET:** when one stray print sets a line, do you count
   it? This reaches back into the six-line results.
3. **Which keys have been rotated?** Is the GitHub repo public?
4. **Daylight saving on 2026-11-01:** change the scheduled task before then?
5. **The 51 stranded shares positions** from 2026-09-08 and 09-10: close them out by rule, or
   leave them marked?
6. **The 13 share symbols added 2026-09-05** carry almost all the shares loss. A backtest was
   offered and never approved.
7. **Checkpoints:** the chats say the first 200-trade checkpoint was crossed with no formal
   evaluation [report 16]; the code reader says no single setup has reached 200, the largest
   being 111 [report 01]. Which unit is the checkpoint — total trades or per setup?
8. **Entry-bar exits and the `TTM_Squeeze` timing** in the options arm: intended, or a defect
   to record by dated amendment?
9. **The Hugging Face Space:** upload the rebuilt version?
10. **These reports:** leave them in this worktree, or commit them to `main`?

## 9. Still open from the research itself

From `CLAUDE.md` and reports 06 and 07 — unchanged by this rebuild:

- The unwritten rule for **which** break he takes. 92% of sessions break a line by 09:40.
- His P&L alternating by time of day — exploratory, windows chosen after seeing the data, not
  a finding, not to be traded.
- A matched comparison of the options arm with and without its hard stops (the loss sat
  there: −$3,547.64 over 45 trades as of mid-September) [chat, report 15].

## 10. The reports

| # | file | what it covers | source |
|---|---|---|---|
| 01 | `01_live_runner_core.md` | one trading day inside the lab; the 15 rules; record formats | code |
| 02 | `02_live_runner_resilience_ops.md` | launch, supervision, outages, recovery, the daily commit | code + logs |
| 03 | `03_hpc_vrp_cluster.md` | the variance experiment, the cluster, the dead TFT | code + research |
| 04 | `04_momo_sweep_and_data_archive.md` | the 2.8M-cell sweep; lookahead bugs; what data exists | code + research |
| 05 | `05_huggingface_space.md` | the public demo, its bugs, deployment | code |
| 06 | `06_research_levels_breakouts.md` | the six lines and every study around them | research + code |
| 07 | `07_research_programme_and_journal.md` | the whole programme; his own trading record | research + code |
| 08 | `08_git_history_and_repo_ops.md` | 156 commits as a story; tests; environments; secrets | git |
| 09 | `09_legacy_code_and_machine_inventory.md` | pre-lab code; what is on disk but not in git | code + disk |
| 10–16 | `10_chat_part1…` to `16_chat_part7…` | the old conversations, in date order, 2026-07-27 to 10-11 | chat logs |
| 17 | `17_live_observation_2026-10-13.md` | the lab watched while running; the close from the log | live + logs |

**Read 16 first among the chat digests** — its section 9 is the state of play on the day the
old conversations stopped.

## 11. How this was built, and its limits

- Thirteen prior sessions (2026-07-27 to 2026-10-11) were found as plain files on the laptop.
  They were de-duplicated, stripped of tool output, pattern-scrubbed for keys, and split into
  seven chunks. A bare password would not have matched the scrub patterns.
- Code-reading agents worked read-only in a git worktree at commit `6f9c1ab`. They ran no
  project code and no tests. The main checkout moved to `48a76f2` at today's close.
- The job hit the account's usage limit three times. Reports 01, 02, 03, 05 and 16 were written
  by Opus; the rest by Sonnet. The live observer was cut off twice and never saw the close;
  the lead reconstructed it from the log.
- Agents disagree with each other in at least one place (question 7 above). Where a report and
  the code disagree, the code wins.
- The original sessions may still be openable: `claude --resume` from the project folder reads
  the same local files. Untested under the new account.
