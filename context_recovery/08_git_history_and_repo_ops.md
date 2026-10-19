# 08 - Git history and repo machinery

Slice: the whole git history (156 commits, 2026-07-30 to 2026-10-12), the branches, and the
repo's own plumbing (tests, CI, pre-commit hook, environment files, .gitignore, tidy tool).
Method: read-only git inside the worktree, plus reading the files named in the brief. Nothing
was run. Hashes below are short commit ids. `path:line` means I read that line. **(inferred)**
marks anything I deduced rather than read. Secrets are never quoted; where a value would
appear I give file, line, commit and length only.

---

## 1. In one paragraph

The repo is ten weeks old in git terms (first commit 2026-07-30, last 2026-10-12) but the
project is older: the first commit is a single 93-file, 61,462-line snapshot of earlier work,
so *how the original app, notebooks and first backtests were built is not in git*. After that
the log reads as a lab notebook. Of 156 commits, 15 are one-per-day data commits written by a
robot (`live-lab-archive`); the other 141 are hand or AI-assisted commits whose messages are
unusually long incident reports: what broke, the measurement that proved it, the fix, and very
often a correction of the author's own earlier claim. The repo's machinery (tests, CI, hook,
`.gitignore`, a tidy tool that is an allowlist) was built reactively, each piece the answer to
one specific loss: a leaked key, a test run that collected nothing, a deleted dependency, a
purge that ate source code, a record that existed on one laptop only. Every branch is already
inside `main`; nothing would be lost by deleting any of them.

## 2. The goal it serves, and how it changed

The git history serves three jobs, and they appear in this order:

1. **Version a personal trading-research codebase** (07-30 to 08-14): notebooks, a FastAPI app,
   live scripts, backtests. Messages are one-liners with typos ("redudant"). No explanatory bodies.
2. **Be the audit trail of a research programme** (from 08-31): each result document names its
   falsifier before the computation, nulls are committed as readily as positives, and the commit
   message records the error found in the previous commit. `c748c2b` states the rule: a
   pre-registration's frozen design text stays verbatim; only its status line may move.
3. **Be the durable store of a live forward test** (from 09-17 automatically, 09-21 on the
   log): `47c6835` and `archive.py:1-30` say why: on 2026-09-16 the committed record stopped
   at 09-08 while six sessions existed only on the travelling laptop, and "committing by hand
   is not a plan". The record is the one input that cannot be bought back
   (`archive.py:14-18`).

## 3. How it works, end to end

### 3.1 The life of a change

1. Edit on Windows (`core.autocrlf=true`, read from git config) or in a cloud Claude Code session.
2. `git commit` runs `.githooks/pre-commit` **only if `core.hooksPath` is set**. On this machine it
   is the absolute path `C:\Users\chaitanyakharche\Documents\stock\.githooks` (git config, shared
   by the worktrees). It was never set until `b4c33b0` (08-31), so the hook "has never run" before
   then. A fresh clone does not have it **(inferred from `.githooks/pre-commit:9`: "Enable once per clone")**.
3. Push to GitHub (`origin https://github.com/ChaitanyaKharche/stock.git`). `.github/workflows/tests.yml`
   runs on every push to any branch, every PR, and manually (`tests.yml:17-20`).
4. Work reaches `main` by merge or PR. Two PRs exist (#1, #2, both 09-24, from
   `claude/laughing-shannon-yfynal`).
5. The lab laptop runs whatever is checked out in `C:\Users\chaitanyakharche\Documents\stock`
   at 05:30 via `run_lab.cmd` (see 3.5 and 7.3).

### 3.2 The daily data commit (`archive.py`)

At the end of each session `autostart` calls `archive_session()` (`archive.py:150`). It stages
only `live_lab_data/` by explicit pathspec (`archive.py:170` `git add -- <lab dir>`), makes no commit if
nothing changed, commits with the identity passed per call: `user.name=live-lab-archive`,
`user.email=live-lab-archive@localhost` (`archive.py:73-74, 187`), then pushes to the current
branch. It never raises, force-pushes, rebases or amends. One recovery was added `c17aaa0`
(09-24): if the push is rejected as non-fast-forward, fetch, merge, push once
(`archive.py:76`, flag `RECOVER_FROM_NON_FF`). Reason: four consecutive nights (09-21 to 09-24)
the push lost a race with a research session pushing to the same branch.

### 3.3 The project in phases (question 1)

Commit counts per day, from `git log`: 07-30:2, 08-02:2, 08-14:1, **08-31:8**, 09-03:1, 09-04:4,
**09-08:12, 09-09:11**, 09-10:3, **09-11:20**, 09-13:3, 09-14:3, **09-15:10**, 09-16:4, 09-17:8,
09-18:2, 09-19:3, 09-20:1, 09-21:5, 09-22:5, 09-23:6, **09-24:11**, 09-25:2, 09-28:2, 09-29:2,
09-30:2, 10-01:1, 10-02:4, 10-05:1, 10-06:3, 10-07:1, **10-08:11**, 10-09:1, 10-12:1.
There are no commits between 08-14 and 08-31 (17 days).

**Phase 0 - Before git (before 2026-07-30).** *Not recoverable from git as steps.* The root commit
`e17df27` "Checkpoint before dead-code cleanup sweep" adds 93 files at once: `app1.py`, eight
`stock3.*.ipynb` notebooks, `trade_analysis/` (live_deploy, momentum engine, crypto scripts,
TFT training), `trained_models/*.pth`, `local_data/*_external_data.json`, `requirements.txt`,
`Dockerfile`, `README.md`. The original app, the notebooks, the TFT models and the live scripts predate it. The first options backtests arrive
in `5466212`/`4154740` (Phase 2); the 5-minute lookahead bug lives in those `find_entries` backtests, but its
discovery is not a commit of its own. It is recorded only in prose:
`research/PROJECT_REPORT.md:90` ("`find_entries` recorded a 5-min bar's label, filling 5 min before
the signal existed ... 88-95.7% of the entire measured edge; two whole result sets void").
Earliest committed backtest code mentioning "lookahead": `4154740` (08-14). **Open: why history
starts on 07-30 (new repo? squash?) is not recorded.**

**Phase 1 - Cleanup and re-structure, 07-30 to 08-02** (`f384d81`, `e04d237`).
Built: subpackages `backtesting/ data_sources/ live_trading/ models/ signals/ trade_journal/ utils/ scripts/`,
`paths.py`, a reconstructed `models/tft_model.py`, `signals/gamma_exposure.py` (GEX: dealer gamma exposure
signal; "3 day confluence layer" per `f384d81` subject). Thrown away: `f384d81` deletes 10 modules
(`advanced_risk, gap_predictor, market_microstructure, prop_integration, real_data_sources,
statistical_arbitrage, test5min_breakout, test_data, train_tft, walkforward`); `e04d237` deletes
seven notebooks (`stock3.1` ... `stock3.3.4.1`, keeping `stock3.3.4.ipynb`), `codebase_context.json`,
HTML reports and five trading/download logs (-34,156 lines). Reason: only the subject lines
("Cleanup redudant code"). `README.md:224-228` adds that `train_tft.py` was removed as non-functional
because the model-definition file it needed was lost (the `/scratch` purge), and `README.md:232-241`
calls the notebooks "superseded/removed" (`:236`).

**Phase 2 - Options backtests on real data, 08-02 to 08-14** (`5466212`, `4154740`).
Built: `options_premium_backtest.py` (+ `_confluence`, `_scalp`), `retest_vs_gapandgo_backtest.py`,
`spy_qqq_0dte_quotes_backtest.py`, `spy_qqq_0dte_real_backtest.py`, `vilkov_0dte_conditional_backtest.py`
(the Vilkov SPXW option-quote panel), and the vendor clients `massive_client.py`, `thetadata_client.py`
(10 files, +2,504 lines in `4154740`). Thrown away: nothing. Reason for the move to real quotes:
not in the commit message **(reason not recorded)**; the later cost-model result (`05c5406`)
shows why real bid/ask mattered (t=+7.52 at mid fills became t=-5.38 net).

**Phase 3 - The research record and the live lab enter git, 08-31** (8 commits, first with long
bodies and the trailer `Co-Authored-By: Claude Opus 5`).
- `b4c33b0` turns on the credential hook and adds `.claude/agents/` (three research subagents).
- `4956008` hardens the live lab: supervisor with backoff restarts, dated recovery file, OS-level
  single-instance lock, runner output persisted, dead `degraded_bars` fixed.
- `743ba83` adds the **SHARES arm** (IMB = Intraday Momentum Boundary, SSRN 4824172) as a store
  separate from the options arm (`live_lab_data/shares/`), all 13 setups run so the Holm m=13
  multiplicity correction stays valid; `PLACES_ORDERS=False`.
- `9fcdce9` journal reverse-engineering (443 trades): his entry rule is recoverable (AUC 0.847) and
  loses money; records its own bias (anchoring forward returns on the last known bar gave +2.271 bp and
  "cleared Holm"; interpolating to his actual entry second gave +0.850 bp, Holm 0.9736).
- `31a0731` IMB stress test: survives, but 29 trades = 73.9% of P&L; 9.2%/yr on $10k.
- `b9dd78b` adds the standing research record (`research/*preregistration*.md`, `*results*.md`);
  `38ee3d3` first commit of the frozen forward-test record (two sessions, 42 ATM trades, -$2,213.39).
- `9e52f44` narrows the hook to ignore ALL-CAPS environment-variable names; adds the `bulk_download` package.
The live lab had been collecting since 2026-08-28 (`38ee3d3` body: "frozen 2026-08-28"), so its
code ran for three sessions before it was first committed.

**Phase 4 - First live incidents, 09-03 to 09-04** (`264068e`, `7d31cf2`, `b42b2fc`, `88606c3`, `6985258`).
The supervisor killed both arms at 15:55:02, the exact minute they flatten, so sessions 09-01 to 09-03
have no end-of-day exits and the loss was biased (it removed only the trades that had not stopped out,
i.e. the winners on a +95 bp day). Fix plus `reconstruct_eod.py`, which rebuilds exits and flags them
`eod_reconstructed`. Also a dashboard "signal reconciliation" (decisions x arms must equal trades).
Same day class: a shares flatten with no fallback mark left positions unrecorded; the early-close
detector mistook a dead network for a closed market (it blamed the vendor first; the real cause was the
laptop losing wifi then power); an exposure check that could never pass was replaced by a query to
Windows Firewall.

**Phase 5 - "Complete overhaul", the cluster, the Space, and the variance-premium (VRP) experiment, 09-08 to 09-11.**
- `c0c869d` 09-08 "Complete overhaul + live runner lab" (78 files, +27,922; **no body**): first
  appearance of `huggingface_space/`, `trade_analysis/hpc/` (Discovery cluster code), more
  `live_lab/` modules (`ledger.py`, `backfill_daily.py`), shares record for 09-01..03, and
  `research/vrp_preregistration.md`.
- Cluster scripts `26330e7`, `f81f143`, `9725244`: sbatch jobs must not depend on `conda activate`;
  `train_vrp` now refuses to silently run AdamW labelled "muon".
- VRP chain: `ce754a7` (audit passes, HAR-RV is the bar), `c1ea49a` move 1-minute bars to 5-minute (the 1m
  archive has an un-refillable hole, 2024-01..07 SPY, entitlement lapsed), `fb39273`, `80e1d4c` (zero-priced
  halt bars -> log(0) poisoned 170 sessions), `6037995`, **`05c5406` "The 0DTE variance premium is real and
  1.84x too small to trade"**, `cad54eb` correction of HAR bias, `5565a71` HAR forecast replaces the dead
  TFT in the app.
- 09-09 to 09-11: zero-shot foundation-model arm (`7d32fa4` to `f451625`), conditional-0DTE Arms A and B
  (`45eb21a`, `98b664d`, `d0de4c9`, `c2d74ab`, `29b8592`), phase-0/diurnal/ceiling checks
  (`485d48d`, `0a27a30`, `119ed54`), the project report (`20d4dea`), the two host-sleep incidents
  (`dbf0bba`, `a1eda80`), and the tidy tool (`d7f9bd3`).
- Thrown away: the idea of a GPU neural sweep for the variance arm (`05c5406`, `485d48d`: "The queued sweep
  has no route to a positive"); the previously reported HAR-RV 0.187365 (`80e1d4c`) and "IV beats HAR p=0.0222"
  (`cad54eb`).

**Phase 6 - Pre-registered sweep of one strategy form, 09-13 to 09-16** (`dd3e40b` ... `65afbb9`).
`momo_v2` ran after sitting pre-registered and unexecuted for nine days (`dd3e40b`); then the
2,822,400-cell MOMO_CHASE sweep was pre-registered (`5b083e7`), a Hansen SPA test was written by hand because
`arch`'s SPA ignores its own `studentize` flag (`5136f3e`), a Sharpe floor added (`6d02fe7`), pre-condition 3
amended *before any shard ran* (`c54d2e4`, `972bc7f`), sbatch bugs fixed on the real cluster (`881171c`,
`7c9fbff`, `3f6179e`), result `65afbb9`: **null, p=0.8227, form closed**.

**Phase 7 - The cloud-session era and the breakout programme, 09-17 to 09-23** (24 commits authored
"Claude", all with trailer `Claude-Session: https://claude.ai/code/session_012Tn9aFNNDiLnSRjMPSoj3E`, on branch
`claude/laughing-shannon-yfynal`).
Repo ops built here: `requirements-dev.txt`, `pytest.ini`, CI (`d662890`); `pandas_ta` removed (`74f3476`,
`e08e849`); `archive.py` (`47c6835`); feed waits through a hotspot switch (`e7bf6de`); coverage tool fixed
(`9f1691f`); nine pre-registration headers corrected (`c748c2b`); `CLAUDE.md` written (`c5a4f31`, 09-21).
Research: 8-line breakout pre-registered (`1e5cbfd`), profit cap added then voided for lookahead
(`13e948e`, `d639c7c`), six lines as actually specified (`61996a2`), six-line null (`3c111a9`), early-R1 studies
(`761b5a9`, `860d621`, `67eef3f`). Thrown away: the 8-line level set (it was never his set, `61996a2`); the
capped +4.98 bp result (void); "1-2 trades around 11:39" (`0f99213`).

**Phase 8 - Six lines go live, Space re-aimed, recovery tooling begins, 09-24 to 09-30.**
`9ca4aa0` programme review; `88bae1e` start both arms before the open; `4cc694f` the trader's own six-line
setups run live (config hash changed to `ccd00ac71f8243e9`, `e423c3a`); `25cd841` the HF Space stops opening on
a BUY/HOLD generator and opens on the findings; `1e7aeef` gap recovery; `494c314` freeze-and-catch-up after a
commute; `9274f06` replay of all 22 sessions ("0 of 22 sessions ran clean"); `5d1ceb7` signals gzipped per
session so the push stays under GitHub's 100 MB file limit.

**Phase 9 - Network-outage recovery wave and its own retractions, 10-02 to 10-08.**
`e4174e2` a stale Friday chain no longer aborts a session; `2d99165` restart catches up from
`managed_through`, not the last save; `13d80b3`, `53bffa3`, `9d2a911`, `1d295bc` stranded-position fills;
then a review-driven chain on 10-08: `7d0feb1`, `3d24460` (retracts 219 corrections), `31ddee2`, `8e1eda0`
(retracts 68 more), `6405392`, `4114f1b`, `1768e8f` (reinstates two), `f7b4429`; plus `2156955`/`4937f3e`
day charts.

**Phase 10 - Steady state, 10-09 to now.** Only two commits after the last code change (`4937f3e`, 10-08):
data commits `4d91790` (10-09) and `6f9c1ab` (10-12). `main`, `origin/main` and this worktree's branch
are all at `6f9c1ab`.

### 3.4 Commit conventions (question 3)

- **Subject lines are claims, not verbs.** "The audit came out WORSE, and that was the audit's fault",
  "A NameError in an argument list killed today's session", "Null: removing the close-back-inside
  stop is worth nothing". No Conventional-Commits prefixes. The first five commits (07-30 to 08-14) are
  lowercase one-liners with no body.
- **Bodies are incident reports.** Typical order: symptom with numbers; ROOT CAUSE; what it cost;
  FIX; what was verified; and (for live-lab commits) a line such as "Config hashes unchanged:
  1f7247d7839d9950, 53229d6f1f24df10" (e.g. `7d31cf2`, `b42b2fc`). Headings in capitals inside the body.
- **First-person correction of earlier work.** "THREE BUGS I INTRODUCED WHILE FIXING IT" (`264068e`),
  "A BIAS I INTRODUCED AND THEN FOUND" (`9fcdce9`), "my last commit blamed the wrong thing" (`ecd64e3`).
- **Corrections are new commits, not rewrites.** `97e742a`: "The commit is not rewritten; this one corrects
  it, because a message that describes a change the diff does not contain is worse than a second
  commit." Retractions in the live record are appended `retracted` lines, never edits (`3d24460`).
- **Trailers.** `Co-Authored-By: Claude Opus 5` on 87 commits (08-31 to 09-24), `Claude Opus 5.5
  (1M context)` on 9 (09-24 to 09-30), `Claude Opus 5.5` on 16 (10-02 to 10-08). The 24 "Claude"-authored
  commits (09-17 to 09-24) carry `Claude-Session: ...session_012Tn9aFNNDiLnSRjMPSoj3E` instead
  (a cloud session; this id may help recover that transcript). The 15 data commits carry neither.
- **Authors** (`git log --format=%an`): ChaitanyaKharche 112, Claude 24, live-lab-archive 15, claude (lowercase)
  3, Chaitanya Kharche 2. The lowercase `claude` and several `ChaitanyaKharche` commits are automatic
  "Merge branch 'claude/laughing-shannon-yfynal' of https://github.com/..." merges (`e459cb9`, `16819de`,
  `79e9345`, `dca445c`, `23f3460`, `f58185b`) **(inferred: pulls between the cloud session and the lab laptop
  when both pushed to one branch)**. The two "Chaitanya Kharche" commits are GitHub web merges of PR #1/#2.
- **Data commits.** Author `live-lab-archive <live-lab-archive@localhost>`; subject `Live lab record for
  YYYY-MM-DD`; body "Written by trade_analysis/live_lab/archive.py at the end of the session." plus a
  `git diff --shortstat`. Fifteen exist, one per session day from 2026-09-21 to 2026-10-12. The first
  (`b5f8ae4`, 23 files, +398,469 lines) swept up six sessions (09-14 to 09-21). A typical one (`6f9c1ab`) touches
  12 files: `live_lab_data/daily/<date>.json`, `events.jsonl`, `session_ledger.jsonl`, `trades.jsonl`,
  `signals.jsonl`, `signals_archive/<prev-day>.jsonl.gz`, and the same set under `live_lab_data/shares/`
  (+14,604 / -11,892 lines because `signals.jsonl` is rewritten to hold only today). **No data commit
  exists for 10-07**, and no `daily/*.json` exists for 09-08 to 09-11 or 10-05 to 10-07 (computed from
  `live_lab_data/daily/`; 09-07 is Labor Day).

### 3.5 The test suite (question 5)

- **Files.** 32 tracked `*_test.py`: 29 under `trade_analysis/` (25 pytest-style plus 4 script-style),
  3 under `huggingface_space/`.
- **Count.** `grep`-style count of `def test_` in `trade_analysis/**/*_test.py` at HEAD = **327**, spread over
  25 files (largest: `gap_recovery_test` 30, `breakout_levels_test` 28, `day_charts_test` 21, `after_break_test`
  20, `entry_histogram_test` 19, `feed_resilience_test` 19). `parametrize` appears in 2 files
  (`opening_window_test`: 2 uses, `six_lines_setup_test`: 3), so pytest's collected number is somewhat higher;
  I did not run pytest, so the exact figure is **not measured**. The same count method over history:
  12 at `d662890` (09-17), 78 at `6c4ce64` (09-18), **123 at `3c111a9` (09-21)**, 256 at `9274f06` (09-29),
  327 at HEAD. So **"~123 checks" in `CLAUDE.md` and `README.md:75` was true on 09-21 and is stale by
  about 200.** Four test files have no `def test_` and are script-style (`autostart_guard_test`,
  `preflight_gate_test`, `momo_sweep/stats_test`, plus `levels_test.py`, which is a research script, not a test);
  they are run by `python -m`, print PASS/FAIL, and exit non-zero. They are NOT counted in 327.
- **Where they live.** Next to the code: `trade_analysis/live_lab/*_test.py` (26 files), `trade_analysis/backtesting/entry_histogram_test.py`,
  `trade_analysis/indicators_pandas_test.py`, `trade_analysis/momo_sweep/stats_test.py`. There is no `tests/` directory.
- **How to run.** `pip install -r requirements-dev.txt` then `pytest` (`README.md:73-76`). Tests are offline and need no
  Theta Terminal or network.
- **`pytest.ini`** (read in full): `python_files = *_test.py` (line 5; the repo names files `<thing>_test.py` and
  default discovery found zero tests, exiting 5 which looks like a clean run); `python_functions = test_*` (line 6);
  `testpaths = trade_analysis` (line 20; `huggingface_space/` is excluded on purpose because its tests hit yfinance and
  "a network flake must never read as a code failure"); `addopts = -q --strict-markers` (line 22: quiet output; an
  unregistered marker is an error rather than a silently-skipped test).
- **CI** (`.github/workflows/tests.yml`): Ubuntu only (`:24`), Python 3.11 and 3.12 (`:30`, because `.python-version`
  pins 3.12.10 while `environment.yml` pins 3.11), `fail-fast: false`. Steps: `pip install -r requirements-dev.txt`,
  `pytest`, then 12 explicit `python -m` runs of the script-style/house tests (ledger, indicators_pandas,
  feed_resilience, archive, orb_veto, orb_veto_backtest, breakout_levels, six_lines, preflight_gate, bar_cache,
  autostart_guard, momo_sweep.stats), then `ledger --lab-dir live_lab_data` as a report that never fails the
  build (`:97`). No Windows runner, no secret scan, no lint. Platform checks SKIP rather than FAIL
  (`README.md:114-117`; the Windows firewall assertion and the wake-lock check).
  Newer pytest-style files (stranded, gap_recovery, catchup, day_charts ...) run under the plain `pytest` step only.
- **Pre-commit hook** (`.githooks/pre-commit`, `sh`): scans the *added lines* of staged files only
  (`:27-32`), blocks on Alpaca-shaped `PK`/`AK` assignments (`:19`), `td1_prod_` ThetaData keys (`:20`),
  `sk-`, `hf_`, Slack `xox*`, `AKIA` AWS (patterns block `:18-25`), and a generic net for a 16+ character
  quoted literal assigned to a `*_KEY`/`*_TOKEN`/`CLIENT_*` name (`:53-55`) that skips placeholders,
  `os.getenv` and ALL-CAPS values (`9e52f44`). It prints which pattern matched and how many lines, never the
  text, and exits 1. `--no-verify` bypasses it. **Gap (inferred from the regex at `:55`):** the generic net requires
  a quote immediately after `=`, so shell forms like `export KEY=value` or `echo 'export KEY=value'` (the form
  that leaked in `clean_trade_venv.sh`) would not match.
- **"Every test must be verified to FAIL against the broken version" - is it visible?** Yes, in prose, not in
  tooling. Docstrings say it: `backfill_test.py:5` ("verified to fail with the corresponding guard removed"),
  `early_r1_hold_test.py:13`, `opening_window_test.py:23` ("verified to fail against autostart.py / ledger.py as of
  e4dfaf6"), `options_runner_guards_test.py:21`. `archive_test.py:270-279` explains a mutation that exposed a decorative
  test and the test written to catch it. Commit messages record the verification: `d662890` ("Verified to FAIL against
  the pre-fix ledger.py at f451625: 7 of 11 fail there, 11 pass here"), `2d99165` ("the main one fails against the old
  saved_at-only"), `1d295bc` ("Test fails on the shipped version"), `6405392` ("4 new tests; 4 mutations, all killed"). There is no mutation-testing
  tool in the repo; the check is manual and attested in prose **(inferred: no automation found)**. `d662890` also found a test that
  could not fail (`bar_cache_test.test_fan_out` returned a bool, so pytest passed it regardless).

### 3.6 The environment tangle (question 6)

State it this way and you cannot get it wrong:

1. **The live lab runs on pyenv-win Python 3.12.10, not on any repo venv.** `trade_analysis/live_lab/run_lab.cmd:7`
   calls `C:\Users\chaitanyakharche\.pyenv\pyenv-win\versions\3.12.10\python.exe -m trade_analysis.live_lab.autostart --symbols QQQ SPY --start-terminal`
   after `cd /d "C:\Users\chaitanyakharche\Documents\stock"` (`:6`). `autostart.py:848` spawns its arms with `sys.executable`,
   so the arms inherit the same interpreter. It is registered in Windows Task Scheduler as `LiveLab-QQQ-SPY` (`run_lab.cmd:3`).
2. **What that interpreter holds (read from `.dist-info` directory names, nothing run):** numpy 2.3.5, pandas 2.3.3,
   scipy 1.18.1, httpx 0.28.1, pytz 2025.2, python_dotenv 1.2.1, torch 2.9.1+cu128. **pytest is not installed in it.**
3. **Other environments on the laptop (all gitignored via `venv*/`, `.gitignore:25`):**
   `Documents\stock\venv` (built on a python.org 3.12.10 at `AppData\Local\Programs\Python\Python312`; numpy 2.5.3, pandas 3.0.5,
   dotenv; no scipy/httpx dist-info seen) is hand-activated and irrelevant to the lab. `Documents\stock\venv_charts`
   (`include-system-site-packages = true`, home = the pyenv 3.12.10; adds matplotlib 3.11.2 and pytest 9.1.1) is
   the day-charts tool environment and, because it inherits pyenv's packages, the one place pytest can run against the
   lab's own numpy/pandas **(inferred: not confirmed this is how the owner runs tests)**.
4. **`.python-version` = `3.12.10`** (`.python-version:1`). pyenv-win shims sit first on PATH, so an unactivated
   `pip install` on this laptop targets the pyenv environment, i.e. the lab's. This is a wrong-target install, not an error
   (memory note `python-environment-split.md:20-22`).
5. **Which requirements file is real?** None of them describes the lab.
   - `requirements.txt` (111 lines) is a `pip freeze` containing `torch==2.8.0` (`:96`), `triton==3.4.0` (`:99`) and about 15 `nvidia-*-cu12`
     wheels (`:43-57`) - Linux-only, so it was frozen on a Linux environment, not the Windows lab **(inferred; the memory note says
     "frozen on the cluster")**. It is not installable on the lab and `pandas_ta` was removed from it on 09-17 (`:60-64`). **Dead as a spec.**
   - `conda_requirements.txt` (135 lines, header `platform: linux-64`) and `environment.yml` (`name: trade-venv`,
     `python=3.11`, `prefix: /home/kharche.c/.conda/envs/trade-venv`) are the Discovery cluster's `trade-venv`. **Cluster only.**
   - `trade_analysis/requirements.txt` is an older, different freeze (`accelerate==1.6.0`, `antropy`, ...). **Dead.**
   - `requirements-dev.txt` is the only live one, and it is for *tests*: numpy 1.26.4, pandas 2.3.2, scipy 1.16.1, httpx 0.28.1, pytz 2025.2,
     pytest 9.1.1 (`:38-43`). Its pins match `requirements.txt`, not the lab: **lab numpy is 2.3.5 and pandas 2.3.3**.
   - `huggingface_space/requirements.txt` is the Space's own file (pandas `>=2.2,<3` with a written reason).
   - Root `Dockerfile` is dead (it copies a nonexistent `app/`, `README.md:150-153`). `huggingface_space/Dockerfile` is the real one.
6. **Never on the lab machine:** `pip install -r requirements.txt` (multi-GB CUDA stack; would try to downgrade numpy to 1.26.4);
   `pip install -r requirements-dev.txt` into the pyenv interpreter (same numpy/pandas downgrade, changing the numerics the
   frozen forward test trades on); any numpy/pandas/scipy upgrade (the recorded variance numbers were produced under specific
   versions; `python-environment-split.md:31-33` notes they already drifted). Install test dependencies only in a throwaway
   venv or CI. `config.py` raises `ImportError` on import unless `FINNHUB_KEY` and `REDDIT_CLIENT_ID` are set
   (`trade_analysis/config.py:14-15`); it is imported by `data_sources/unified_data_provider.py:8`, so importing that
   module on the lab machine fails unless those variables exist.
7. **Cluster:** conda env `vrp` (python 3.11) built by `trade_analysis/hpc/setup_env.sh` on a compute node, never the login node;
   batch jobs call the env's python by absolute path (`f81f143`). Details belong to the HPC report.

## 4. Map

Repo machinery and history-adjacent files in my scope:

| path | what it is for |
|---|---|
| `README.md` | Root readme. Layout (stale, stops at the 08-02 refactor), test instructions (`:63-117`), pandas_ta note, dead Dockerfile note, TFT caveats, known-issues incl. the credential table (`:245-274`). |
| `CLAUDE.md` | Owner's working rules and settled-findings table (written `c5a4f31`). Not in my slice but its "~123 checks" is wrong now. |
| `pytest.ini` | Teaches pytest the `*_test.py` naming; scopes to `trade_analysis`; `-q --strict-markers`. |
| `requirements-dev.txt` | Offline test dependencies, ~150 MB, no CUDA. |
| `requirements.txt` | Dead pip freeze of a Linux/CUDA env. |
| `conda_requirements.txt`, `environment.yml` | Discovery `trade-venv` (linux-64, python 3.11). |
| `trade_analysis/requirements.txt` | Older dead freeze. |
| `.python-version` | `3.12.10` (pyenv). |
| `Dockerfile` (root) | Dead; refers to a nonexistent `app/`. |
| `.gitattributes` | Forces LF on `*.sh`, `*.sbatch`, `*.bash` (Windows edit, Linux run; a CRLF `#SBATCH` line is read by Slurm as a comment and the job would run with the wrong partition/wall clock); marks `*.npz *.pkl *.parquet *.gz` binary. |
| `.gitignore` | See section 6.3. |
| `.githooks/pre-commit` | Credential-literal blocker (3.5). |
| `.github/workflows/tests.yml` | CI (3.5). |
| `.claude/settings.local.json` | Tracked permission allowlist; contains leftover scratchpad paths from old sessions `bda6ddc5-...`. |
| `.claude/agents/{quantitative-researcher,statistical-auditor,strategy-contrarian}.md` | Three research subagent definitions, added `b4c33b0`. |
| `trade_analysis/scripts/tidy.py` | Clutter remover that is an ALLOWLIST with a DENY list (`DENY_PARTS`, `tidy.py:53`); dry run by default (`--apply` to delete). Written `d7f9bd3` after an "older than 7-10 days" request would have deleted 2.3 GB of irreplaceable data. |
| `trade_analysis/scripts/clean_trade_venv.sh` | Discovery cache cleanup for conda env `trade-venv`; refuses to run outside it. Working-tree file is CRLF (`file`), `.gitattributes` normalises on commit. Held literal credentials at lines 112-115 until `9e52f44`. |
| `trade_analysis/config.py` | Reads `TWELVE_KEY`, `FINNHUB_KEY`, `REDDIT_*` from env; `raise ImportError` if `FINNHUB_KEY` or `REDDIT_CLIENT_ID` missing (`:14-15`). Legacy of the rejected news work. |
| `trade_analysis/paths.py` | Anchored `LOGS_DIR/DATA_DIR/CACHE_DIR/TRAINED_MODELS_DIR/LOCAL_DATA_DIR` constants; creates `logs/ data/ .cache/` on import (`:15-16`). |
| `trade_analysis/live_lab/run_lab.cmd` | The scheduled-task launcher (3.6). |
| `trade_analysis/live_lab/archive.py` | Daily data commit (3.2). |
| `app1.py`, `stock3.3.4.ipynb`, `trained_models/` (19 files), `local_data/` (9 json) | Phase-0 survivors (FastAPI app, the one kept notebook, TFT checkpoints, collected alt-data). |

Branches (question 4): see section 4.1.

### 4.1 Branch picture (question 4)

`git branch -a -vv`, `git rev-list --count`, `git log main..<branch>` (all read-only):

| branch | tip | ahead of main | behind main | what it holds | verdict |
|---|---|---|---|---|---|
| `main` (also `origin/main`, `origin/HEAD`) | `6f9c1ab` 10-12 | 0 | 0 | everything; checked out in the main checkout (the lab runs from it) | live |
| `claude/live-runner-architecture-review-259bf0` (this worktree) | `6f9c1ab` | 0 | 0 | identical to main; nothing yet | empty, safe to drop |
| `claude/blissful-kalam-999916` | `f451625` 09-10 | 0 | 112 | old tip ("The audit passes, but it is weak"); same name as this worktree's folder but not checked out there | fully merged, stale |
| `fix/live-lab-host-suspend` and `origin/fix/live-lab-host-suspend` | `c61d70b` 09-16 | 0 | 72 | host-sleep guards; merged by `6c4ce64` | fully merged |
| `lab/shares-arm-and-journal-study` (local only) | `c0c869d` 09-08 | 0 | 137 | the shares-arm / journal-study line ("Complete overhaul + live runner lab") | fully merged |
| `origin/claude/laughing-shannon-yfynal` | `224968f` 09-24 | 0 | 41 | the cloud-session branch; merged by PR #1 (`93f1962`) and PR #2 (`e4dfaf6`) | fully merged |

**Answer: no branch has a commit that is not already in `main`.** Deleting any of them loses no commit.
Not visible from here and therefore *could* still be lost: uncommitted or untracked work inside the
main checkout (I may not run git there). `git worktree list` shows exactly two worktrees (main checkout and this one);
`git stash list` is empty; there are no tags. One structural warning: `archive.py` pushes `HEAD` to whatever branch the main checkout is on,
so that branch name matters to the nightly push (see 7.3).

## 5. Design decisions and the reason for each

| decision | reason (evidence) |
|---|---|
| Hook scans only staged added lines and never prints matches | So a clean working tree cannot hide a staged key and the hook cannot itself leak one (`.githooks/pre-commit:6` and the closing block, which says "Nothing was printed above by design"). |
| Hook was narrowed, not bypassed | "Not bypassed with --no-verify. The hook exists because a key was once pushed to a public remote; the fix belongs in the rule." (`9e52f44`). |
| Git history was not rewritten after the leak | "Once a key is published, rotation is the only real remediation" (`README.md:267-271`). |
| Test files named `*_test.py`, plus `pytest.ini` | Bare `pytest` collected zero and exited 5, indistinguishable from a clean run (`pytest.ini:1-4`; `d662890`). |
| `requirements-dev.txt` instead of installing `requirements.txt` | The only documented install was a multi-GB CUDA freeze, so the only host that could run a test was the host that must trade (`d662890`; `requirements-dev.txt:1-25`). |
| CI on 3.11 and 3.12 | Both are in use (`tests.yml:27-30`). |
| Platform-only checks SKIP | "A gate above the attainable range is broken, not strict" (`d662890`; `README.md:114-117`). |
| Real throwaway git repos in `archive_test` | "Mocking git would have tested my model of git" (`47c6835`). |
| `archive.py` stages an explicit pathspec only | An unattended 16:00 `git add -A` on a machine with `.env` is how a credential reaches a public remote (`archive.py:23-26`). |
| `archive.py` is on by default | "an opt-in safeguard is the safeguard that was already failing" (`47c6835`). |
| Gzip one signals file per session | Shares `signals.jsonl` was 37.7 MB and would pass GitHub's 100 MB file limit in about four weeks (`5d1ceb7`). |
| `tidy.py` is an allowlist with a DENY root list | "mtime is not a proxy for value" (`tidy.py:19`); the `/scratch` purge had already eaten the TFT model source. |
| LF line endings forced for shell and sbatch | `.gitattributes:1-7`: edited on Windows with autocrlf, run on Linux. |
| `.gitignore` splits `live_lab_data/` into versioned record vs regenerable caches | The frozen record "cannot be regenerated"; caches and ~27 MB of backtest dumps can (`.gitignore:85-99`, `b4c33b0`). |
| Model weights and `data/vrp/` ignored | "Git keeps blobs forever" (`.gitignore:110-119`). |
| Corrections go in as new commits or appended lines | A record that removes its own mistakes is not a record (`CLAUDE.md`; `3d24460`, `97e742a`). |
| pandas_ta dropped and replaced by `indicators_pandas.py` pinned to the lab's own indicators at 1e-12 | PyPI deleted the 0.3.x releases; the lab's definition wins because `replay.py` proves live and batch agree on it (`README.md:121-148`; `74f3476`, `e08e849`). |

## 6. Records, formats, invariants, frozen things

### 6.1 Never change (as stated in the repo)
- `live_lab_data/FREEZE.json`, `config/`, `daily/`, append-only `*.jsonl`: "the experiment itself ... cannot be regenerated"
  (`38ee3d3`). Config hashes (options `1f7247d7839d9950`, shares `53229d6f1f24df10` then `b53ca8a58aa11718`, six-line config
  `ccd00ac71f8243e9`) are pooled under acceptance rules; the commits state "Config hashes unchanged" each time a lab commit
  could have disturbed them.
- Live trade records are corrected by **appending** (`trade_corrections.jsonl`, `retracted` lines), never editing `trades.jsonl`
  (`1e7aeef`, `3d24460`).
- Pre-registration text stays verbatim after results arrive; only the status line moves (`c748c2b`).
- History: not rewritten (`README.md:267`).
- `.githooks/pre-commit`: do not weaken without a reason in the commit; do not `--no-verify`.

### 6.2 Operational invariants
- The lab machine runs the checked-out branch of the main checkout. Do not switch branches there during a session.
- Nothing may be deleted by age. `tidy.py` DENY roots: `data/`, `live_lab_data/`, `trained_models/`, `models/`, `research/`, `venv/`, `.git/`,
  `~/Documents/research_data` (licensed Vilkov panel), `~/Desktop/data` (raw option and stock archives).

### 6.3 Gitignored but important (question 7; names only, from `.gitignore`)
- **Secrets:** `.env`, `.venv`, `env.txt`, `.env.local`, `.env.development`, `.env.test`, `.env.production` (`:46-52`); `*.lock`, `*.lock.info` runtime locks.
- **Environments:** `venv*/`, `env*/`, `venv_py310/`, `bin/ include/ lib/ lib64/ pyvenv.cfg` (`:25-33`) (so `venv` and `venv_charts` are untracked).
- **Logs:** `*.log`, `logs/`, `lightning_logs/`, `slurm-*.out`, and root `/*.out`, `/*.err`.
- **Data (broad):** `*.csv *.pickle *.pkl *.h5 *.hdf5 *.npy *.npz` (so `research/round_trips.csv` is ignored), `*.onnx`, `data/vrp/`, `/data/`, `vrp_frame.tar.gz`.
- **Personal broker history:** `research/order_history_raw.json`, `research/order_history_raw.txt`, `order_history_raw.*` (`:81-83`).
- **Live-lab regenerables/transient:** `live_lab_data/bars_cache/ quote_cache/ optcache/ ext_cache/ namebars/ recovery_archive/ logs/`, `sharewf_trades*.json`, `walkforward_signals.json`, `levels_trades.json`, `imb_option_pnl*.json`, `positions_open.json` (options and shares). Un-ignored on purpose: `live_lab_backfill/*/logs/` (the replay logs are the record).
- **Charts:** `live_lab_charts/` (day-chart PNGs).
- **Space weights:** `huggingface_space/trained_models/` (rebuilt by `sync_shared.py --push`).
- **Unrelated:** `/pokemon/` (a 218 MB unrelated project that "must never enter this history"), `.playwright-mcp/`.
- **Outside the repo entirely (not in git, easily lost):** the 0DTE option quote archive `option_quote_1m_0dte/` (667 MB), minute bars `stock_ohlc_1m` (1.4 GB) under `~/Desktop/data/raw`, and the licensed Vilkov panel `data_opt.parquet` (252 MB) under `~/Documents/research_data` (`tidy.py:10-13`, `48ebea2`). The options archive cannot be re-downloaded (ThetaData entitlement is FREE and 403s).

## 7. Failure modes, incidents, traps

### 7.1 Every reversal - things built, then deleted, voided or corrected (question 2)

Deleted from the tree:

| when | what | why |
|---|---|---|
| `f384d81` 07-30 | 10 modules incl. `train_tft.py`, `walkforward.py`, `gap_predictor.py` | subject "Cleanup redundant code"; `train_tft` also non-functional (`README.md:224-226`) |
| `e04d237` 08-02 | 7 notebooks, `codebase_context.json`, 5 logs, 4 HTML reports | "superseded/removed" (`README.md:236`) |
| `5d1ceb7` 09-30 | `live_lab_backfill/shares/signals.jsonl.gz` | replaced by one gzip per session so pushes stay under 100 MB |
| `5565a71` 09-08 | the TFT role in the app | "Replaces the dead TFT's role" (`5565a71`); TFT measured degenerate |

Voided or corrected in place (the project's scars; all from commit messages):

| commit | what was believed | what happened |
|---|---|---|
| `cad54eb`, `485d48d`, `9686c6b` | "implied variance beats HAR, p=0.0222" | HAR was missing the `exp(s^2/2)` lognormal factor; corrected p=0.5832; the wrong number survived in an agent memory file and in a docstring (`har_forecast.py`) and the HPC README, corrected 09-24. |
| `80e1d4c` | HAR-RV 0.187365 | a zero-priced halt bar made log(0) = -inf; poisoned 170 sessions; 193 of 769 silently dropped; baseline recomputed. |
| `6e139a4` | "zero sessions dropped" | the doc contradicted its own split 346+250+151=747; the 22 "lost" sessions are a rolling-window warmup; also `np.math.erf` removed in numpy 2.0 broke the scipy fallback. |
| `cdafc2d`, `66be51b`, `5c4d6c8` | chronos "mean" forecast | chronos-bolt's "mean" IS its median; then a mean of logs exponentiated; then negative quantiles exploded to RMSE 6.7e+11. Three bias bugs in the file whose docstring warns about that bias. |
| `6bbd102`, `522f033`, `f451625` | lookahead audit for the zero-shot arm | audit design 1 and 2 were themselves defective (moved the origin; changed the horizon); third design (write the answer into the last context slot), then a "saturate" mode because the third was weak. |
| `264068e` | sessions 09-01 to 09-03 | supervisor killed arms at 15:55:02 so end-of-day exits were never written; biased sample; rebuilt as `eod_reconstructed`; 09-01 snapshot unrecoverable. |
| `b42b2fc` | "worst feed day, the vendor" | it was the laptop losing wifi and power. |
| `919e47a` | 09-09 session | a `NameError` in an argument list killed it; stderr was not teed so no trace existed. |
| `977f4ae` | preflight freshness check | passed future-dated quotes (a one-sided test); also 43 of 208 trades (21%) were stale-bar entries. |
| `88606c3` | persistence test T5 | circular by construction, caught after running, redone with prior-20-session vol. |
| `9fcdce9` | "+2.271 bp, cleared Holm" | an anchoring bias of the author's own; honest figure +0.850 bp, Holm 0.97. Also a pre-registered criterion that was unachievable with 200 permutations. |
| `6aa6c85` | gate calibration | the calibrator held its own copy of the gate, wrong since the 09-07 recalibration; real fire rate 2.5% not 8.8%. |
| `aa80e9b` | "/scratch purges after 45 days of no access" | actual policy wipes everything monthly (first Tuesday), no access-time condition (`setup_env.sh:16`). Fixed in `trade_analysis/hpc/` but **not in the root README** (`README.md:206-207`). |
| `5136f3e` | `arch.SPA(studentize=True)` | the flag does nothing (verified by 40/40 bit-identical p-values); the sweep implements Hansen's test directly. |
| `5b083e7`, `6d02fe7`, `8860e1f` | own test statistic, Sharpe floor, percentile label | a stray sqrt(n); the large-N form of the floor overstated the maximum; "94.1th pct" read as top when the cell was bottom 5.9%. |
| `c54d2e4`, `972bc7f` | pre-condition 3 of the sweep | amended before any shard ran; the gate in code was then updated to match the document. |
| `a782478` | "partial reduction is biased down" | wrong: dropping shards shrinks both the observed maximum and the null; it is a scope defect. |
| `9f1691f` | "usable 7/7 = 100%" | the coverage tool divided the ledger by itself; truth 7/13 (six sessions unrecorded). |
| `d662890` | "all tests are failing" | nothing was installed; pytest found nothing; a platform check scored as FAIL; one test could not fail. |
| `c748c2b` | nine pre-registrations marked NOT YET RUN | results had landed 8 days earlier; a session concluded the VRP experiment was unrun. |
| `13e948e`, `d639c7c`, `ea8f2bb` | 8-line breakout, then capped result | the first sweep silently omitted the profit cap; the capped result (+4.98 bp, p=0.0003) was VOID (cap could fill after the position had exited); two runs on identical data disagreed (MFE 34.26 vs 40.70 bp), which exposed it. |
| `61996a2` | the 8-line sweep | built on the wrong level set (three prior sessions merged, premarket erased); `breakout_options_results.md` is "a null about levels he does not draw". |
| `0f99213` | "he takes 1-2 trades around 11:39" | wrong count (2.90/day) and a median quoted as a cluster; `breakout_options_results.md` section 5 withdrawn. |
| `ed565df` | histogram bimodality rule | fixed 0.70 valley rule ignored counting noise; replaced by a 2-sigma rule. |
| `9ca4aa0` | daily summaries | field `net` (3 strike arms summed) vs `atm_net` (ATM only) differ ~3x across files; recorded as open, not patched. |
| `cf1fb53`, `e131b40` | shares session 09-15 | a watcher the assistant armed held the file an `os.replace` writes, killing the arm for 15 minutes; 37.5% of the day's net was contaminated. |
| `1e7aeef` | shares exits 09-25 | after-hours quotes used to flatten (-$1,478.61); now never take a fresh quote after 16:00. |
| `7d0feb1`, `3d24460`, `31ddee2`, `8e1eda0`, `4114f1b`, `1768e8f`, `f7b4429` | gap-recovery corrections | the recovery overwrote exits that live had observed. 219 then 68 corrections retracted (by appending `retracted` lines), then two reinstated because only the first of two sleeps was checked; `f7b4429` finally recovers the 32 shares exits of 09-11 under the revised gate. Net effect tracked in each message (e.g. combined -$16,256.24 to -$16,238.61). |
| `25cd841` | HF Space front page | opened on a BUY/HOLD signal generator, "the one thing a year of pre-registered work found no edge in"; now opens on the findings. |
| `5a02617` | the word "wings" in a preflight line | read as a multi-leg structure; the lab only buys single naked contracts. Display change only. |
| `74f3476`, `e08e849` | `pandas_ta` dependency | deleted from PyPI; replaced and pinned by test. |

### 7.2 Secrets in history (question 8)

What `README.md:245-274` says (read): a "Known outstanding issues" entry, "Credentials committed to PUBLIC git history -
rotation still outstanding", "Audited 2026-08-16", with a table whose "Rotated?" column reads **NO** for every row:
Alpaca key and secret (`trade_analysis/5min_alpaca.py:10-11` in `f384d81`, `e17df27`; row marked "do this first, it is a
brokerage credential"), Finnhub, Twelve Data, Reddit client id and secret (README.md and `clean_trade_venv.sh`), and an
`HF_TOKEN` (README.md in `f384d81`, `e17df27`). It adds `slurm-1612669.out`. It says history was deliberately not rewritten and that
every affected commit is an ancestor of `origin/main` on the public remote.

What I verified in git (values never printed; lengths from a character count):

| exposure | file : lines | commits where a literal is present | gone from tree |
|---|---|---|---|
| Alpaca API key (26 chars) and secret (44 chars), assigned as quoted literals | `trade_analysis/5min_alpaca.py:10` and `:11` | `e17df27` (07-30), `f384d81` (07-30) | `e04d237` (08-02) renamed it to `data_sources/download_5min_alpaca.py`, which reads `os.getenv` (`:12-13`) |
| Finnhub (40), Twelve Data (32), Reddit client id (22), Reddit client secret (30) as literals in an `echo 'export ...'` hint | `trade_analysis/clean_trade_venv.sh:112-115` | `e17df27`, `f384d81` | file moved to `trade_analysis/scripts/` in `e04d237` and **kept the literals at the same lines** until `9e52f44` (08-31) replaced them with `your-...` placeholders |
| the same four variable names with literal values in a captured Slurm job log | `trade_analysis/slurm-1612669.out:43-46` | `e17df27`, `f384d81` | deleted `e04d237`; `slurm-*.out` now ignored (`.gitignore:58-60`) |
| `HF_TOKEN` | **not found**. `README.md` lines 41 and 82 at `e17df27`/`f384d81` read `your-huggingface-token`. No `hf_[A-Za-z0-9]{20,}` shape exists in any non-binary file at `e17df27`, `f384d81`, `e04d237` or HEAD (names-only search). | - | - |

Other findings: `README.md` itself held **placeholders in every revision** (all 14 revisions checked); the table's claim that
README.md held the Finnhub/Twelve/Reddit/HF values is not supported by git; the values were in the shell script and the job
log. A PK/AK-shaped Alpaca literal appears only in `e17df27` (added) and `e04d237` (removed); no other commit in any branch.
No `td1_prod_`, `AKIA`, `sk-`, `xox*` shapes at those commits (names-only search). The Reddit user-agent in the script
includes a Reddit username (`e17df27`), not a secret. **The README says "working tree is clean" at the 2026-08-16 audit, but
`clean_trade_venv.sh:112-115` still held literals until `9e52f44` on 08-31**, the same commit that wrote the README section.
Whether the GitHub repo is currently public, and whether any key was since rotated, is not in git: **README says NO, and nothing in
the log says otherwise.** Separately `CLAUDE.md` lists an HF token exposed in `~/.bash_history` on the Discovery filesystem
(`7d32fa4` also says a token "already put one in ~/.bash_history in plaintext on a shared filesystem once"); that is a different
exposure from anything in git.

### 7.3 Traps for whoever touches the repo next
1. **The lab runs from the main checkout.** Whatever branch is checked out at `C:\Users\chaitanyakharche\Documents\stock` runs at 05:30, and `archive.py` pushes `HEAD` to that branch at 16:00 (`run_lab.cmd:6`; memory `lab-runs-from-main-checkout.md`). On 2026-09-24 the checkout was found on a local `main` 76 commits stale. Today `main` equals `origin/main` (`6f9c1ab`), so that incident is resolved, but nothing pins the branch.
2. **A non-fast-forward push** loses a race with any other pusher on that branch; one automatic merge-and-retry exists (`c17aaa0`).
3. **Hooks are per clone.** A new clone or a cloud session has no `core.hooksPath` unless set **(inferred)**. CI has no secret scan.
4. **CRLF.** `clean_trade_venv.sh` is CRLF on disk; a Linux copy would fail with `$'\r': command not found`.
5. **Dirty tree is normal.** The lab writes into `live_lab_data/` continuously, so any "is the tree clean?" gate must ignore it (`915a512`).
6. **Slurm logs.** A relative `--output` lands in the git clone and trips provenance gates (`ecd64e3`); all sbatch files now write under `/home/%u/vrp/logs/` which must exist first.
7. **`/scratch` on Discovery is wiped in full monthly.** Nothing important may live there.
8. **Do not run tidy by age.** Use `tidy.py` dry-run only.
9. **Nightly commit size.** Each data commit rewrites `signals.jsonl` (about +12k/-12k lines); a push of a very large file would break the nightly push (100 MB limit).
10. **`config.py` import trap** (section 3.6).
11. **A workaround for a wrong-target install:** use `python -m pip` with an explicit interpreter path, never bare `pip`, on the lab laptop.

## 8. Current state

- `main` = `origin/main` = this worktree's branch = `6f9c1ab` (10-12). Working tree: only untracked `context_recovery/`.
- No commits unmerged on any branch; no tags; no stash.
- Last code change `4937f3e` (10-08); since then only data commits. The 10-08 review wave (gap recovery, stranded pass, charts) is done.
- 621 tracked files; pack size about 86 MB (`git count-objects -v`, the 19 TFT checkpoints and `local_data/` are in history forever).
- Data record gaps in git: no `daily/*.json` for 09-08 to 09-11 and 10-05 to 10-07; no data commit for 10-07. Backfill facts live in `session_ledger.jsonl` and `research/live_lab_coverage_audit.md` (other slices).
- Open, per `CLAUDE.md`: HF token exposure revocation unconfirmed; per `README.md:247`: Alpaca/Finnhub/Twelve/Reddit rotation outstanding.
- Half-done: the README layout section is stale; the dead root `Dockerfile` is still there ("Delete it or point it somewhere real", `README.md:152`); `requirements-dev.txt` pins do not match the lab.

## 9. Contradictions (doc or memory versus code; code wins)

1. **Test count.** `CLAUDE.md` ("~123 checks") and `README.md:75` ("123 checks") versus **327 `def test_`** at HEAD (123 was the count at `3c111a9` on 09-21).
2. **pytest.ini versus the tree.** `pytest.ini:16-18` says backtesting files are "collected by nobody"; `trade_analysis/backtesting/entry_histogram_test.py` (19 tests) matches `*_test.py` under `testpaths`, so it is collected **(inferred; not run)**. `d662890` says `testpaths` was `trade_analysis/live_lab`; now it is `trade_analysis`.
3. **Which machine `requirements.txt` freezes.** `requirements-dev.txt:5`, `tests.yml:5`, `README.md:65-66` call it a freeze "of the trading machine"; its contents (Linux-only nvidia/triton wheels), `conda_requirements.txt` (`platform: linux-64`) and the memory note point to the Linux cluster env. The lab's pyenv has numpy 2.3.5, pandas 2.3.3, scipy 1.18.1, torch 2.9.1+cu128, none matching.
4. **`requirements-dev.txt:35-36`** claims its versions mean "the same code passed on the trading machine's numpy and pandas". Lab numpy is 2.3.5 (dist-info), dev pin is 1.26.4. It matches the cluster freeze only.
5. **`config.py` importers.** Memory `python-environment-split.md:35-38` says `config.py` has zero importers. `trade_analysis/data_sources/unified_data_provider.py:8` does `from .. import config`, and `collect_alt_data.py:8` imports that module. Also `config.py` reads `FINNHUB_KEY` (`:8`) while `README.md:177` and `collect_alt_data.py:61` name `FINNHUB_API_KEY`.
6. **Memory says scipy/sklearn absent in pyenv** (`python-environment-split.md:15`); scipy 1.18.1 `.dist-info` is now present there (installed after 09-12 or memory was wrong). torch 2.9.1+cu128 is present too.
7. **Root README says /scratch "purges after 45 days of no access"** (`README.md:206-207`); `aa80e9b` and `setup_env.sh:16` correct this to a full monthly wipe (`setup_env.sh:16`).
8. **README credential table** claims `README.md` held live values; git shows only placeholders (7.2). It claims a clean tree at 08-16 while the script held literals until 08-31. It lists an `HF_TOKEN` in `README.md`, which I could not find.
9. **README layout** (`README.md:14-59`) omits `live_lab/`, `hpc/`, `momo_sweep/`, `research/`, `huggingface_space/`, `live_lab_data/`, `live_lab_backfill/` - everything built after 08-02.
10. **`lab-runs-from-main-checkout.md`** said local `main` was `f451625`; today `main` is `6f9c1ab` and `f451625` is the tip of the stale local branch `claude/blissful-kalam-999916`.
11. **`d662890` message** says CI scope is `trade_analysis/live_lab`; `pytest.ini:20` says `trade_analysis`.

## 10. Questions only the owner can answer

1. Why does git history begin on 2026-07-30? Was there an earlier repository, a squash, or a history rewrite? Where is the record of the original app and of the lookahead discovery?
2. Have the Alpaca, Finnhub, Twelve Data and Reddit credentials in `e17df27`/`f384d81` (and the HF token) been rotated? Is the GitHub repo public today? (`README.md:252-258` says NO for all.)
3. Is `core.hooksPath` set on every machine and cloud session that commits? CI does not scan for secrets.
4. Is the main checkout clean right now, and which branch will it be on at 05:30? (I cannot run git there.)
5. Which of the six branches do you want deleted? Git shows each is fully contained in `main`, so only naming/tidiness is at stake.
6. Do you want `requirements-dev.txt` re-pinned to the lab's numpy/pandas/scipy, and a real lab requirements file written? Is torch 2.9.1 in pyenv intentional?
7. Did a session run on 2026-10-07 (no data commit, no daily file)? And 09-09 to 09-11 and 10-05/06: are those sessions represented only in the ledger?
8. Is `session_012Tn9aFNNDiLnSRjMPSoj3E` (24 commits, 09-17 to 09-24) a transcript you can still reach?
9. The root `Dockerfile`, `environment.yml`, `conda_requirements.txt`, `trade_analysis/requirements.txt`, `app1.py` and the 19 TFT checkpoints are dead or reference-only: keep or remove?
