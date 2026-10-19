# 19 — Next step for each of the three legs, under spec-driven development

Written 2026-10-16. My recommendation, built on reports 01–18. Where a point rests on one of
those reports, it says which.

## In one paragraph

The research question has been answered as far as this data can answer it: no setup
predicts direction. What is left to learn has to come from the forward test running cleanly
for many more months. So the next phase is not research and not new features. It is making
the live lab trustworthy and keeping it that way. That is exactly the work spec-driven
development is good at: an existing system with hard rules, where the enemy is a silent
failure rather than a missing feature.

- **Live lab:** a short list of reliability specs, in deadline order. The first fixes what
  zeroed 2026-10-15.
- **HPC cluster:** park it. Every question it was built for is closed. The one thread still
  open is a short straddle, which you do not trade.
- **Hugging Face Space:** finish it, upload it once, freeze it.

## What "spec-driven" should mean in this repo

**You already do half of it.** Every research study here starts as a pre-registration: the
question and the decision rule are written down before any data is looked at. That is a spec.
It is the reason the path-versus-endpoint mistake of 2026-09-23 was caught. Code changes
never got the same discipline. Changes to the lab were made straight away, often by an
assistant, and the record shows the cost:

- the README and the lab spec no longer match the code (report 02 §9, report 17 §7);
- "frozen" turned out to be unenforced (report 01);
- preflight and the runners disagree on what "fresh" means — the cause of 10-15 (report 18);
- the 15:55 kill that deleted end-of-day exits for three sessions (report 13).

**The proposal: one short spec file per change, approved before any code.** Keep it light;
this is a one-person repo.

```
specs/NNN-short-name.md

Status:      draft | approved | implemented | verified-live
Why:         the incident or evidence, with dates and file:line
Requirements: one sentence each, in the form
             "When <something happens>, the <part of the lab> shall <do this>."
Touches the frozen record?  config hash / trading rules / both arms / none
Acceptance tests: named tests, written FIRST, each shown to FAIL on today's code
Rollout:     lands after 16:05 ET or on a weekend; how the next session's log proves it worked
Out of scope: what this spec deliberately does not change
```

The "When …, the … shall …" form comes from EARS, a standard way to write requirements.
Each sentence is one testable behaviour, so it maps one-to-one onto a test.

**Three rules to add to `CLAUDE.md`:**

1. No change under `trade_analysis/live_lab/` without an approved spec.
2. A spec's tests are written first and must fail on the current code. The project already
   asks this of tests; the spec makes it unavoidable.
3. Anything that changes a trading rule is a dated amendment to the pre-registration, not a
   spec. That decision is yours, not an assistant's.

**Tooling: start repo-local, not with a framework.** GitHub Spec Kit (specify, plan, tasks,
implement) and AWS's Kiro (requirements, design, tasks) both work. For a solo repo they add
more ceremony than they return. Claude Code can follow one template plus a project command,
and `CLAUDE.md` already acts as the project-wide spec. Spec Kit works with Claude Code if you
later want more structure.

**The known weakness of spec-driven development is drift:** the spec says one thing, the code
quietly does another. This repo has already suffered exactly that. The fix is to make each
spec's key claims executable. Example: a test asserting that preflight and both runners
import the *same* staleness limit, so they cannot drift apart again.

## Leg 1 — Live lab: reliability specs, in deadline order

| # | spec | evidence | deadline |
|---|---|---|---|
| — | **You, no code: keep the laptop plugged in overnight, set it not to sleep while plugged in, and set the Windows Time service to Automatic** | 10-15 lost to a battery sleep; 8 sessions hit by mid-session sleeps; clock left 7 s fast (report 18) | now — it can recur tonight |
| 001 | **Trust the clock, and use one staleness limit** (drafted below) | 10-15: every quote rejected, while preflight said "All checks passed" | now |
| 002 | **The scheduled task must still cover the close in winter** | Task runs 05:30 local + 8 h. After 2026-11-01 that ends at 15:30 ET, 25 min before the flatten, with `StopAtDurationEnd = True` (report 02 §3.1) | before Fri 2026-10-30 |
| 003 | **Half-day close for the options arm** | 2026-11-27 closes 13:00 ET. The shares arm handles early closes; the options arm does not (report 01) | before 2026-11-25 |
| 004 | **Do not give up on the network after 20 minutes** | Six terminal restarts, then "leaving it for a human". Cost 2–3.5 h on 10-02, 10-05 and 10-06 (report 18) | soon |
| 005 | **A truthful session label** | 10-05 and 10-06 are `COLLECTED` after losing hours. The checkpoint counts depend on this label (reports 02, 18) | before the checkpoint evaluation |
| 006 | **Decide the two rule defects**: options exits firing on the entry bar; `TTM_Squeeze` 10 vs 45 minutes | report 01 | your decision first; then a dated amendment, or recorded as "known, kept as frozen" |

**After 005: run the checkpoint evaluation the pre-registration promises.** The old chats say
the first 200-trade checkpoint was crossed with no formal evaluation (report 16).

**Not on the list, on purpose:** no new setups, no new symbols, no parameter changes. The
lab's only value is an unbroken, honest record.

**The structural fix for sleeping is a machine that does not sleep.** That could be an old
desktop at home or a free cloud VM. Moving the lab is a big change with its own risks. Only
spec it if the sleeps continue after you keep the laptop plugged in.

## Leg 2 — HPC cluster: park it

- **Every question it was built for is closed.** The variance premium is real but too small to
  trade; the momentum sweep is null; the TFT is dead (reports 03, 04).
- **The binding limit is data, not compute.** A boosted-tree model on all 17 features did 1.77%
  worse than the 6-parameter HAR. That points at one asset and about 346 effective sessions,
  not at model size (report 15).
- **The one open thread is not your trade.** Conditional 0DTE Arm B "clears all five gates,
  reported FRAGILE" — but what it measures is *selling* a straddle on days with high expected
  premium (`research/conditional_0dte_armB_results.md:1-20`). You buy calls and puts, so I do
  not recommend spending the sealed 2024 data on it.

**One housekeeping spec, then stop:**

- revoke the Hugging Face token sitting in shell history on the shared filesystem;
- copy results off `/scratch` (wiped on the first Tuesday of every month);
- fix the two hazards in `submit_arm_a.sbatch` (report 03) so nothing breaks if it is ever
  re-run;
- correct the "45 days" wording in the README and the report.

**If you still want the machine-learning view of your days** (the bubble chart): it can only
be descriptive — "these are the kinds of days, and here is where you made and lost money" —
never a predictor. It belongs in the Space, not on the cluster.

## Leg 3 — Hugging Face Space: finish and freeze

One spec, five steps (report 05):

1. **New token.** Rotate the token and use a fine-grained one scoped to this Space only.
2. **Remove the leftovers that contradict the research:**
   - the "Allocation … % of capital" tile;
   - fixed profit targets on every suggested structure (capping winners lost money in the
     two mechanical tests);
   - suggested spreads (not your instrument);
   - the "SOTA … multi-LLM ensemble" self-description, when no language model loads on the
     deployed container.
3. **Trace every number.** Each number on the research tab comes from a file in `research/`
   through `tools/build_research_snapshot.py`. Add a test that fails if one does not.
4. **Upload and check.** Upload the research-first build (commit `25cd841`, never confirmed
   uploaded) and confirm the page runs.
5. **Stop.** No live data, no new features.

## Order of work

1. Rotate the exposed keys (reports 08, 11–15). Yours; no code.
2. Plug-in, sleep and time-service settings. Yours; no code. This alone would have saved
   10-15.
3. Specs 001 → 002 (by 10-30) → 003 (by 11-25) → 004 → 005 → checkpoint evaluation.
4. Space: finish and freeze.
5. Cluster: housekeeping, then park.

---

## Spec 001 (draft) — the lab must not trust a clock it has not checked

**Status:** draft, for your approval.

**Why.** On 2026-10-15 the laptop woke from hibernation with its clock about 7 seconds fast.
Windows time sync is set to Manual and did not correct the clock until 15:19 MST, after the
close. Both arms discard any quote older than 5 seconds (`runner.py:52`,
`shares_runner.py:73`), so every quote was discarded: 946 + 1,770 `stale_quote` rows. Result:
zero trades. Preflight passed with `[OK] … REAL-TIME (age 6.2s)` because its limit is
120 seconds (`preflight.py:39`). Full trace in report 18.

**Requirements.**

1. When preflight runs, the lab shall measure the laptop clock's offset from internet time
   and write it to the log and to the ledger's `OPENED` row.
2. When that offset is larger than 2 seconds either way, preflight shall FAIL with a
   `CLOCK SKEW` line giving the offset. (2 seconds is a proposal; the number is yours.)
3. Preflight's real-time check on the underlying shall use the runners' limit, imported from
   one place, so preflight cannot pass a quote the runners would reject.
4. When every symbol's quote has been rejected as stale for 60 seconds in a row during the
   session, the supervisor shall log one loud line naming clock skew as the likely cause,
   with the measured offset.
5. When more than a set share of a session's ticks had no usable quote, the ledger shall not
   record that session as `COLLECTED`. This overlaps with spec 005; settle it there if you
   prefer.

**Touches the frozen record?** No trading rule and not the config hash. The staleness limits
are deliberately outside the hash already (comment at `runner.py:69`). Both arms are affected
equally.

**Acceptance tests.** Write them first; each must fail on today's code.

| test | asserts | today |
|---|---|---|
| T1 | Preflight given quotes 6.2 seconds old reports FAIL | fails (it reports OK) |
| T2 | Preflight given a clock offset of +7 seconds reports `CLOCK SKEW` | fails (there is no such check) |
| T3 | Preflight, `runner.py` and `shares_runner.py` use the identical limit object | fails (120 against 5) |
| T4 | A simulated minute of all-stale ticks produces the loud supervisor line once | fails |

**Rollout.** Land on a weekend or after 16:05 ET, on `main`; the lab runs from the main
checkout. The next session's log must show a clock-offset line marked OK.

**Out of scope.**

- Correcting quote ages by the measured offset. That would hide a broken clock instead of
  reporting it.
- Changing the 5-second limit.
- Changing Windows settings. That is yours to do (Windows Time service to Automatic).

---

## Sources on spec-driven development tooling

All secondary sources, consulted 2026-10-16:

- [GitHub Spec Kit vs Kiro vs Claude Code SDD Workflows — glukhov.org](https://glukhov.org/ai-devtools/ai-coding-assistants/spec-kit-vs-kiro-vs-claude-code/)
- [Spec-Driven Development: How Kiro and AI Agents Build From Specs — morphllm.com](https://www.morphllm.com/spec-driven-development)
- [7 Spec-Driven Development Tools: Spec Kit, Kiro, OpenSpec, Tessl & More — ssojet.com](https://ssojet.com/blog/best-spec-driven-development-tools)
- [Comprehensive Guide to Spec-Driven Development: Kiro, GitHub Spec Kit, and BMAD-METHOD — Medium](https://medium.com/@visrow/comprehensive-guide-to-spec-driven-development-kiro-github-spec-kit-and-bmad-method-5d28ff61b9b1)
