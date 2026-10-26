# Architecture and business review, 2026-10-23

Asked for: a multi-agent chain-of-command architecture, a reliability audit, and a revenue
plan. Grounded in the repository at `da556a5`, the live record through 2026-10-22, and the
twenty reports in `context_recovery/`. Each claim is tagged:

- **[verified]** read from code, the record, or the git history today, with the path
- **[report NN]** stated by a context-recovery agent, not re-checked today
- **[assumption]** my judgement or something nobody has measured

## 0. The verdict in four sentences

The plateau is not an architecture problem and a multi-agent orchestration layer will not
move it. The research answered its question: fifteen rules, three months, no setup predicts
direction [verified: `PROJECT_REPORT.md`, `CLAUDE.md` settled table]. The thing that is
actually unreliable is a laptop that sleeps in a car: only 17 of 44 trading sessions ran
clean [verified below], and that is fixable this week for $0. There is nothing here to sell
as a trading signal, and the most realistic revenue from this repository in the next twelve
months is a salary, so the business plan below is sized accordingly.

## 1. What the record says today [verified]

Computed from `live_lab_data/trades.jsonl`, `live_lab_data/shares/trades.jsonl` and
`live_lab_data/session_ledger.jsonl` at `da556a5`.

| arm | closed trades | sessions | net | per trade |
|---|---|---|---|---|
| options, ATM strike, 1 contract | 680 | 33 | **+$356.06** | +$0.52 |
| shares, $10,000 per trade | 4,938 | 35 | **−$11,899.85** | −$2.41 |

- Options: +$0.52 per trade on a contract costing $150–300 is zero. The spread between best
  rule (`Crabel_Stretch` +$1,981 on 62 trades) and worst (`PDH_PDL_FailedBreak` −$1,390 on
  33) is what fifteen rules times noise looks like; `CLAUDE.md` already forbids reading it.
- Shares: the loss sits in the thirteen symbols added 2026-09-05 (AAPL −$3,572, XLI −$3,143;
  SPY and QQQ together −$301). 18 of 35 days positive. The last eight sessions swing
  −$1,720 to +$2,639 a day. On $10,000 a trade that is 0.1–0.3% a day: a book that is flat
  and loud.
- 256 of 6,976 rows (3.7%) are reconstructed from history after an outage, flagged, none
  edited [verified: `reconstructed` flags; `stranded_recovery.jsonl`].

**Session reliability, the number that matters:**

| ledger outcome | count | meaning |
|---|---|---|
| COLLECTED | 17 | ran open to close, record complete |
| PARTIAL | 21 | host suspended or network lost during hours; exits rebuilt |
| ABORTED | 5 | no feed / failed preflight |
| INTERRUPTED | 1 | |

**17 clean of 44 attempted = 39%.** The last two sessions each lost 2–3 hours to the laptop
sleeping (10-21: 10:54–14:03; 10-22: 12:32–14:59 ET). Entries are frozen during a gap, so a
PARTIAL day is not a false record, but it is a smaller sample than a clean day, and the
estimate of how many months the forward test needs assumes clean days.

## 2. Audit of the existing implementation

### 2.1 What is good and must be preserved [verified]

- `trade_analysis/live_lab/`: 21,082 lines, 1,166 offline tests passing, append-only JSONL
  record, decision written before price is fetched (`store.py`), session ledger, outage log,
  checkpoint replay, self-checking reconstruction with a ≥ agreement gate. This is better
  engineering than most paid retail backtesters ship. Do not rewrite it.
- Pre-registration discipline in `research/`: decision rule fixed before data is seen. It is
  the only reason the 2026-09-23 path-versus-endpoint error was caught (`CLAUDE.md`).
- Three adversarial agent definitions already exist (`.claude/agents/`): researcher,
  auditor, contrarian. That is the entire multi-agent system this project needs.
- CI runs the suite on every push and is green since 2026-10-22.

### 2.2 Bottlenecks and reliability gaps, ranked by cost

1. **The host sleeps.** [verified: 21 PARTIAL; the 10-21 and 10-22 logs] Software guards
   exist (`hold_system_awake`, suspend detector, catch-up). They work. They do not stop a lid
   closing. Nothing in the repo fixes this; it is a Windows power setting or a different host.
2. **Uncommitted lab code ran live for a day and blocked the daily push.** [verified:
   `9919dc2`, 10-22 log] Nothing enforces "frozen"; the hash covers parameters, not code
   [report 01]. Report 19's spec-first rule addresses this; it is not yet adopted.
3. **Daylight-saving kill of the running session.** Fixed on the laptop 2026-10-21
   [verified: `fix_schedule_dst.ps1`, `CLAUDE.md`]. First real test 2026-11-02.
4. **Half day 2026-11-27**: shares arm handles a 13:00 close, options arm does not
   [report 01, report 13]. Six weeks away. Needs a spec and a test.
5. **Checkpoint rule is undefined.** The plan says "200 trades"; nobody wrote down whether
   that is per setup or total [report 01 vs report 16]. The largest single setup has 141
   ATM trades (`MOMO_CHASE`). Without the unit, the forward test has no finish line.
6. **`trade_corrections.jsonl` is not applied by the dashboard** [report 01]. A reader
   who does not know this gets wrong totals.
7. **`TTM_Squeeze` exits at 10 min in options, 45 in shares** (`positions.py:123` missing
   ×5) [report 01]. Defect or intent, undecided; either way it must be recorded by amendment.
8. **Clock drift 2 s** [verified: 10-22 log]. Quotes are rejected past 5 s. `w32tm /resync`.

### 2.3 Unnecessary complexity

- `trade_analysis/live_trading/` (2,892 lines) [verified count]: orphaned Alpaca scripts,
  nothing imports them, no path to a real order [report 09]. Delete or move to `attic/`.
- `trained_models/`, the TFT checkpoints, `stock3.3.4.ipynb`, `app1.py`: coursework-era
  [report 03, README]. The README still describes the project as a "live-trading codebase"
  with a GEX signal and ML research. It is wrong about what the repo is. Rewrite the
  README's first screen to say what §0 says.
- `requirements.txt` (111 pins, CUDA stack, uninstallable) [verified]. Keep only as a
  record of the lab machine; it already says so.
- The HPC leg: every question it was built for is closed [report 03, report 19]. Park it.

### 2.4 Security [report 08, report 09; none re-checked today]

Not code, and ahead of everything in §4.

1. Alpaca, Finnhub, Twelve Data, Reddit keys in git history, rotation recorded as "NO".
   If the GitHub repo is public these are public now. Rotate; do not rewrite history.
2. Hugging Face token in `~/.bash_history` on the shared cluster filesystem. Revoke.
3. Secrets pasted into old chat logs on the laptop (HF, cluster password, Databento,
   ThetaData, Robinhood). Rotate each. Delete `~/.tokens/robinhood.pickle` and
   `Documents\research_data\thetaterminal\creds.txt` or move them out of plain text.
4. Theta Terminal reachable from the LAN on the Public profile. The log prints the fix
   every morning. You asked not to be nagged; it is listed once here.
5. `.githooks/pre-commit` scans for key shapes but is only active after
   `git config core.hooksPath .githooks` on each clone. Whether the laptop clone has it
   enabled is unknown [assumption: probably not, since it was never mentioned].

## 3. Should you build a multi-agent orchestration system? No.

Checked against what the repo does, not against what the frameworks promise.

| need | daily volume | what exists | what an orchestration framework adds |
|---|---|---|---|
| run the lab | 1 session/day | `autostart.py`, Task Scheduler, ledger | nothing; it is one process on one box |
| research experiment | ~1 per week at peak, 0 now | prereg.md → script → results.md → `CLAUDE.md` | a queue for a queue of one |
| audit a result | per experiment | `statistical-auditor` agent | same agent, more plumbing |
| context recovery | once | 16 agents, 20 reports, hit usage limits 3× | this is where it failed, see §3.2 |

Temporal, LangGraph, CrewAI, AutoGen, a message bus, a vector store: each is built for
many concurrent tasks with retries across services. This project has one task a day and a
few LLM sessions a week. Every one of those tools would be the largest component in the
repository and would protect a workload that does not exist. **Cost: weeks. Benefit: none
measurable.** [assumption, but a strong one]

### 3.1 What a chain of command does buy here

The multi-agent layer that exists is Claude sessions plus three agent definitions. Its
failures in the last six weeks are real and specific [verified unless noted]:

- an assistant edited lab code during a session and left it uncommitted (`9919dc2`);
- 16 agents fanned out in one session, hit the usage limit three times, and 7 of 20
  reports were written by a cheaper model and not re-checked (`00_START_HERE.md` §11);
- agents disagreed on a fact and the lead could not adjudicate (`00_START_HERE.md` Q7);
- eight stale or wrong citations sat in `CLAUDE.md` until 2026-10-21.

Every one is a handoff or scope problem, and every one is fixed by rules in git, not by
infrastructure. The design below is the minimum that enforces them.

### 3.2 Roles

One orchestrator. Workers do one atomic task each and return one artifact. Nobody else
writes to `main` or to `CLAUDE.md`'s settled table.

| role | who | one task per assignment | may write | may not |
|---|---|---|---|---|
| **Orchestrator** | the main Claude session, with you | plan, delegate, verify, synthesize, commit | anything, after gates below | run an experiment itself; edit `live_lab_data/` |
| **Owner** | you | approve specs and amendments; decide the questions in §2.2 | `CLAUDE.md`, frozen rules by dated amendment | nothing is off limits, but a rule change is an amendment, never an edit |
| `quantitative-researcher` (exists) | agent | write ONE pre-registration | `research/<name>_preregistration.md` | results, code under `live_lab/` |
| `experiment-runner` (new definition, general-purpose agent) | agent | run the pre-registered script exactly once | `research/<name>_results.md`, output data | change the decision rule; interpret beyond it |
| `statistical-auditor` (exists) | agent | PASS/FAIL/QUESTIONABLE per claim | `research/<name>_audit.md` | edit anything else (already has no Edit tool) |
| `strategy-contrarian` (exists) | agent | alternative hypotheses + killing tests | `research/<name>_redteam.md` | tune or rescue |
| `lab-operator` | **not an LLM**: `autostart.py` | one session | `live_lab_data/` | be edited without an approved spec (report 19) |
| `doc-writer` | agent | one report | `context_recovery/`, `research/*_review*.md` | code, `CLAUDE.md` |

Permission changes to make now [concrete]:

- Remove `Edit` from `quantitative-researcher` and `strategy-contrarian`. Both have
  `WebFetch`; an agent that can read the open web must not be able to edit code. They keep
  `Write` for new files in `research/`.
- Add `.claude/settings.json` deny rules so no agent or session can touch the record or
  rewrite history: `Edit(live_lab_data/**)`, `Write(live_lab_data/**)`,
  `Edit(live_lab_backfill/**)`, `Bash(git push --force*)`, `Bash(git push -f*)`,
  `Bash(git rebase*)`, `Bash(git commit --amend*)` (the lab's own `archive_test.py`
  already bans these for the archiver; this extends the ban to assistants).
- Escalation: any agent that needs something outside its row stops and returns
  `status: blocked` with the reason. The orchestrator decides. Nobody self-promotes.

### 3.3 Source of truth and versioning

Already exists; name it and stop improvising:

- **Trading record:** `live_lab_data/` on `main`, append-only, corrected by appended rows.
- **Rules:** the config hash plus dated amendments in the pre-registration. A rule change is
  a new amendment paragraph with a date, never a diff to an old one.
- **Findings:** `CLAUDE.md` settled table, each row citing a file that must exist.
- **Code:** `main`. The lab runs from `main`. Nothing uncommitted on the lab machine, ever.
- **Version:** the git SHA. Every artifact below records the SHA it was produced at.

### 3.4 Handoff contract

A YAML front-matter block at the top of every pre-registration, results, audit, red-team
and review file. Short enough to be filled by hand; validated by one test (§4, phase 1).

```yaml
---
task_id: R-2026-10-23-01          # one per assignment
trace_id: session_01TqhCAiwZi…    # the Claude session or the human-run date
parent: R-2026-10-23-00            # the task that requested this one, or none
role: experiment-runner            # exactly one of the roles in §3.2
repo_sha: da556a5                  # code and record this was produced from
inputs:                            # paths + sha, never summaries
  - research/foo_preregistration.md@da556a5
  - live_lab_data/trades.jsonl@da556a5
decision_rule: "copied verbatim from the pre-registration, not paraphrased"
outputs:
  - research/foo_results.md
evidence:                          # every number cites file:line or a reproducible command
  - "n=2438: six_lines_results.md:41"
assumptions: []                    # things taken as true without checking, one line each
open_questions: []                 # things only the owner can answer
errors: []                         # anything that failed, with the exact message
status: done | blocked | failed    # never "mostly done"
verified_by: null                  # filled by the auditor task_id, or stays null
budget: {tokens_or_minutes: …, cap: …}   # what was allowed, what was used
---
```

Rules the contract enforces:

- **No silent summarization.** Inputs are paths at a SHA. If a worker condenses, the
  condensed version is a new output file with its own front matter, and the original stays
  referenced.
- **Acknowledgement.** The orchestrator's synthesis lists every `task_id` it consumed. A
  task not listed was not used; a task listed must exist.
- **Idempotence.** Re-running a task with the same `inputs` SHAs must produce the same
  numbers. A different number is a defect (`CLAUDE.md`: "a number that changes when nothing
  changed is the cheapest bug detector there is").
- **Bounded retries, timeouts, loop limits.** One retry per task. A second failure returns
  `status: failed` to the orchestrator, who decides. No agent spawns agents. The context
  recovery job that hit limits three times is the counterexample.
- **Budget.** Fan-out is capped at 4 concurrent agents per session, by rule. The 16-agent
  recovery ran out of budget mid-task and produced a two-tier report set.
- **Conflict resolution.** Two tasks disagree on a fact → the orchestrator opens a third
  task with `role: statistical-auditor` scoped to that one fact. Code beats report beats chat.
- **Independent verification at the critical boundary.** Nothing enters the `CLAUDE.md`
  settled table without an audit file with `PASS` on the specific claim, and the audit's
  `task_id` goes in `verified_by`. This is the one gate that matters; it is where the 88%
  artifact would have been caught.

### 3.5 Prompt-injection and untrusted data

Untrusted inputs this project actually has: ThetaData responses (parsed as numbers, never
executed; fine), web pages fetched by the researcher and contrarian agents, the old chat
logs used in recovery, GitHub comments and CI logs, and anything in `live_lab_backfill/`
or downloaded archives.

- Fetched content is evidence, never instruction. Agents that fetch cannot edit code (§3.2).
- Workers never see secrets. The lab reads keys from config on the laptop; agents work in
  clones that have none. Keep it that way; never paste a key into a session again (§2.4).
- Downloaded or user-supplied files are read with `python -I` from a separate directory,
  never imported.

### 3.6 Storage, messaging, observability: use what exists

- Storage: JSONL on disk, in git. Already append-only, already versioned, already diffable.
  A database would add a service to run on a laptop that sleeps.
- Messaging: files plus commits. The orchestrator reads a worker's output file. Latency is
  irrelevant at this volume.
- Observability: `logs/<day>.log`, `session_ledger.jsonl`, `outages.jsonl` already exist.
  Add one thing: a `lab_health` report (phase 1) that turns them into the five numbers in
  §5 so a trend is visible without reading logs.

## 4. Phased plan

Each item: what, test, acceptance. Nothing touches trading rules. Anything under
`trade_analysis/live_lab/` goes through a spec file first (report 19).

### Phase 0, this week, $0, laptop PowerShell, no code

1. **Stop the host sleeping during 09:00–16:30 ET.** Power & sleep: sleep on battery
   `Never`; lid close action `Do nothing` (both power states). Acceptance: the next five
   sessions log no `HOST SUSPEND DETECTED`, ledger shows `COLLECTED`.
   [assumption: lid-close is the cause; the alternative is a mini PC or VPS, §6]
2. `w32tm /resync /force` as admin; set Windows Time to Automatic. Acceptance: no
   `[clock]` drift line in the next log.
3. Rotate every key in §2.4; revoke the HF token; delete the two plaintext credential
   files. Acceptance: you write the date into `CLAUDE.md` STILL OPEN and strike the item.
4. `git config core.hooksPath .githooks` on the laptop clone.
5. Decide the four owner questions: checkpoint unit (per setup or total), `TTM_Squeeze`
   timing, 11-27 half-day handling for options, keep or drop the thirteen share symbols at
   the next checkpoint. Each answer becomes one dated amendment paragraph.

### Phase 1, two weeks, code, every change spec-first

1. **`lab_health.py`** (new file, reads the ledger, outages and trades; no writes).
   Prints the §5 metrics for the trailing 20 sessions. Test: a fixture ledger with known
   outcomes produces known numbers. Acceptance: runs in CI as report-only, like `ledger`.
2. **`handoff_test.py`**: every `research/*_results.md` newer than 2026-10-23 has front
   matter (§3.4) that parses, cites a pre-registration that exists, and whose
   `repo_sha` is an ancestor of HEAD. Acceptance: fails on a results file with no prereg
   (verified to fail first).
3. **`settled_table_test.py`**: every file cited in `CLAUDE.md`'s settled tables exists.
   Would have caught `setup_sweep_results.md` on day one.
4. **Spec-gate test**: a change under `trade_analysis/live_lab/*.py` in a commit requires a
   `specs/*.md` file with `Status: approved` touched in the same commit or the one before.
   Enforced in CI, not pre-commit, so it cannot be skipped silently.
5. **Options arm early close** for 2026-11-27 (spec, tests first).
6. Agent permission changes from §3.2 (`.claude/agents/*.md` tools, `.claude/settings.json`).
7. README first screen rewritten to §0; `live_trading/` moved to `attic/`.

### Phase 2, by end of November

1. Re-measure session clean rate after phase 0. If still under 80%, decide the host
   question (§6) with the number in hand rather than by argument.
2. First checkpoint evaluation under the rule decided in phase 0, as a pre-registered
   results file with an audit, through the §3.4 contract. This is the first real use of the
   chain of command, on the thing it was built for.
3. Upload the rebuilt Hugging Face Space once, freeze it [report 19].

### Not planned

A new strategy, a new instrument, a new data purchase, an orchestration framework, a
database, a dashboard rewrite, more agents.

## 5. Metrics

| metric | definition | now | target |
|---|---|---|---|
| session clean rate | COLLECTED / sessions attempted, trailing 20 | 39% all-time [verified] | ≥ 90% |
| hours lost to suspend | sum of `HOST SUSPEND DETECTED` seconds / 3600 per session | 2.4 h (10-21), 2.5 h (10-22) | 0 |
| reconstructed row share | rows with `reconstructed` / all rows | 3.7% [verified] | < 1% of new rows |
| push success | sessions whose record is on `origin/main` by 16:30 ET | failed 10-22 | 100% |
| late starts | ledger `late_start: true` | 1 in last 3 | 0 |
| handoff integrity | research artifacts since 2026-10-23 passing `handoff_test` | n/a | 100% |
| citation integrity | `CLAUDE.md` citations that resolve | 100% after 10-21 fixes | 100%, in CI |
| audited claims | settled-table rows with `verified_by` | 0 of 9 (retroactive not required) | all new rows |
| cost per clean session | monthly data cost / clean sessions | unknown: your ThetaData invoice | track |
| agent cost per accepted artifact | tokens or minutes per `status: done` task that passed audit | unknown | track from first use |

Accuracy, latency and failure rate for agents are the last three rows; for the lab they are
the first five. Do not add a metric that nobody will read weekly.

## 6. Costs and risks

**Current cash costs** [assumption; check invoices]: ThetaData options tier (status unknown
since 2026-09-14, `CLAUDE.md`), one Claude subscription, laptop power. Cluster time is free.

**Host options for the lab:**

| option | cost | fixes | risk |
|---|---|---|---|
| laptop, power settings changed | $0 | lid-close sleep | car battery, Wi-Fi handoffs still drop; already guarded |
| always-on mini PC at home | ~$150–250 once | sleep, Wi-Fi, travel | one-time spend; Windows guards carry over unchanged |
| Linux VPS, US east | ~$6–12/month | everything above, plus distance to the exchange | `autostart.py` has Windows-only guards (wake lock, psutil terminal kill) that would need a spec'd Linux path; Theta Terminal runs on Java so it ports [assumption] |

Measure phase 0 first. If the clean rate goes above 80% on the laptop, the question is
closed for free.

**Risks in the plan:** the spec gate adds friction to every lab change (intended); the
handoff front matter is only worth it if the owner refuses artifacts without it; CI tests
on `CLAUDE.md` citations will break the moment someone renames a file (also intended).

## 7. Business assessment

Blunt, then evidence.

**What cannot be sold:** a signal, a strategy, "AI trading". Eight pre-registered nulls
and a flat forward test. Selling it would be selling nothing, and the record that proves
it is public in this repository.

**What is real and unusual here:** a forward-test harness for 0DTE options that writes
every decision to disk before it fetches a price, keeps an append-only record, survives
outages with flagged reconstruction, and has 1,166 tests. Plus a documented methodology
(pre-registration, auditor, contrarian) that caught an 88% artifact. That is a credible
engineering portfolio and a niche tool, not a company yet.

### 7.1 Three candidate theses, with the experiment that kills each

**A. Employment or contract work (highest expected value, lowest cost).** The last
recorded conversation before the account loss was about resume wording [report 16]. The
repo is the evidence: quant-adjacent Python, data engineering under failure, testing
discipline, honest reporting. Market: quant dev, data engineer, research engineer roles at
prop shops, fintech, data vendors (ThetaData itself hires). Willingness to pay: salaries in
that band are public. Acquisition: the repo link plus a one-page write-up of §0 and the
88%-artifact story. Gross margin: all of it.
Experiment: ten applications or outreach messages in two weeks with the repo as the
centerpiece; count first-round conversations. Kill criterion: zero conversations from ten
means the framing is wrong, not the work.

**B. A bring-your-own-data forward-test harness for retail 0DTE traders.** Customer
problem: people who trade 0DTE by rules have no honest way to paper-test them against real
option quotes; brokers' paper accounts fill at mid and do not log decisions.
Competitors: OptionOmega (backtesting, roughly $50–100/month [assumption]), Option Alpha,
broker simulators, QuantConnect, ThetaData's own API. None of them give an append-only
decision-before-price record; most give a backtest, which this project has shown is
exactly where lookahead hides.
Constraints [verified in principle, terms not re-read]: ThetaData's licence forbids
redistributing data, so each customer brings their own subscription. That caps the market
at existing ThetaData options subscribers, a few thousand people at most [assumption].
Operating cost: near zero if the customer runs it locally; hosting it for them means a
terminal per customer and support. Pricing that could work: a one-time licence or
$20–40/month; willingness to pay is unmeasured.
Experiment (two weeks, $0): a one-page landing page, "forward-test your 0DTE rules on
your own ThetaData feed with a tamper-proof record", a waitlist, posted once in the
ThetaData Discord and r/options. Kill criterion: fewer than 25 sign-ups or fewer than 3
people willing to do a 20-minute call. Do not write a line of product code before this.

**C. Content: the honest negative result.** "Three months, 2.8 million parameter
combinations, fifteen rules, no edge, here is how we made sure." Market: retail traders
and the people who sell them courses. Willingness to pay: low; this is an audience
builder for A or B, not revenue.
Experiment: publish `PROJECT_REPORT.md` cleaned up, once, in two places. Measure inbound
within two weeks.

### 7.2 What I would not do

- Build B before A's and B's experiments report. The repo already has one product nobody
  asked for (the Space).
- Spend on data, compute or hosting to chase an edge the record says is not there.
- Add agents or infrastructure to make the project look like a platform. Investors and
  hiring managers both read the record, and the record is the asset.

### 7.3 Revenue roadmap, honestly sized

| when | action | cost | measurable outcome |
|---|---|---|---|
| weeks 1–2 | phase 0; thesis A outreach; thesis B landing page; thesis C post | $0 | clean rate; conversations; sign-ups; inbound |
| weeks 3–4 | phase 1; read the three experiments; kill what died | $0 | handoff and citation tests in CI |
| November | first checkpoint evaluation through the contract; host decision | $0–250 | one audited finding; clean rate ≥ 90% |
| December | if B survived: 3 customer calls, decide build/no-build with their words | $0 | a written yes/no with evidence |

No line above promises profit. The forward test's own verdict so far is that there is no
edge to monetize; the plan monetizes the engineering and the honesty instead, and makes
the lab reliable enough that its eventual answer can be trusted.

## 8. Open questions for the owner

Carried from `context_recovery/00_START_HERE.md` §8, still unanswered:

1. ThetaData: renewed? which tier? history depth?
2. Checkpoint unit: per setup or total trades?
3. Which keys are rotated; is the GitHub repo public?
4. Keep or drop the thirteen share symbols at the checkpoint?
5. `TTM_Squeeze` 10 vs 45 min: defect or intent?
6. Premarket prints in the six-line draw?
7. Upload the rebuilt Space?

New from this review:

8. Is the laptop lid being closed during the session? (Phase 0 item 1 depends on it.)
9. Are you willing to spend $150–250 once on an always-on box if phase 0 does not get the
   clean rate above 80%?
10. Which of theses A, B, C do you want to run first? A costs nothing but time and has the
    highest expected value.
