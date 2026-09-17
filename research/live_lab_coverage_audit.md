# Audit — the live lab's committed record stops on 2026-09-08

**Run 2026-09-16 22:33 ET, from a clean clone of `origin/main` at `f451625`.**
Read-only with respect to every data file: `reconcile()` was deliberately NOT run, for the
reason in §4. The only code change is to `coverage()`, which was reporting the gap as
100% success.

This exists because the forward test's entire value is an unbroken record, and the tool
built to prove the record unbroken was asserting it rather than checking it.

---

## 1. What the coverage tool said, and what it should have said

`python -m trade_analysis.live_lab.ledger --lab-dir live_lab_data`

| | trading days | usable | reported coverage |
|---|---|---|---|
| before this commit | 7 | 7 | **100.0%** |
| after | **13** | 7 | **53.8%** |

Both numbers were computed from the same ledger on the same day. The first is wrong, and
it is wrong in the project's signature way: **no crash, no warning, a confident number
measuring nothing.**

`coverage()` built its day set from ledger records alone. A trading day with no record was
therefore absent from the numerator *and* the denominator — invisible rather than counted.
Six consecutive unrecorded sessions could not move a percentage that only ever divided the
ledger by itself.

`ledger.py`'s own module docstring says *"a denominator you cannot reconstruct is not a
denominator."* The fix reconstructs it: the denominator now walks the exchange calendar
from the lab's first evidence to the last closed session, and a day the calendar expects
but the ledger never mentions is reported **UNRECORDED**.

`UNRECORDED` is synthesised in memory and never written. It ranks below `MISSED` in `RANK`
because MISSED is a decision the lab reached and UNRECORDED is the absence of one.

## 2. The gap, stated precisely

Six completed sessions have **no record of any kind** in the committed repository — no
ledger line, no `daily/` file, no trade, no signal, no event, no outage:

    2026-09-09  2026-09-10  2026-09-11  2026-09-14  2026-09-15  2026-09-16

Session dates come from `trade_analysis/bulk_download/trading_days.py`, the same calendar
the lab uses. 09-12 and 09-13 were a weekend. 2026-09-17 is excluded: in exchange time it
had not begun when this ran.

Last committed evidence, per source:

| source | last entry |
|---|---|
| `session_ledger.jsonl` | 2026-09-08 `PARTIAL`, recorded 15:53:52 ET |
| `shares/trades.jsonl` | 2026-09-08, 65 trades |
| `shares/events.jsonl` | 2026-09-08T11:40:20 |
| `shares/daily/` | **2026-09-04** |
| `daily/` (options arm) | 2026-09-04 |
| any commit in the repo | 2026-09-10 |

Two smaller holes fall out of that table and are part of the same failure:

- **`shares/daily/2026-09-08.json` was never written**, though 65 trades were recorded
  that day. The session has trades and no rollup.
- **The 09-08 ledger line is superseded locally.** `research/PROJECT_REPORT.md:264` quotes
  a `PARTIAL … supervisor_rc: 0` record written at **16:00:03**; the last committed line
  for that date is 15:53:52 and carries no `supervisor_rc`. The local ledger is therefore
  already ahead of the committed one.

## 3. Ran-but-uncommitted, or never ran?

**For 2026-09-09, proven: it ran, and its record was never committed.**
`research/PROJECT_REPORT.md:227` reports that session as 130 trades at −$1,339.77. That
number cannot have come from anywhere but a local record, and no such record is in the
repo.

**For 09-10 through 09-16, undetermined from here, and this clone cannot settle it.** The
two hypotheses are not symmetric, so the discriminating evidence is worth naming:

`autostart.py:517` calls `ledger.reconcile()` at the top of **every** invocation. So on the
lab machine, any single run on any later day would have backfilled `MISSED` lines for all
the earlier gaps. The local ledger consequently already contains the answer, whichever it
is — nobody needs to reconstruct it, only to read and commit it.

What that implies for the committed record: the absence of even a `MISSED` line proves
only that nothing has been *pushed* since 09-10. It does not distinguish a lab that was
off from a lab that ran into an uncommitted directory.

**The cost if the sessions were genuinely lost.** The gate is 60 sessions
(`research/PROJECT_REPORT.md:229`). Six is **10% of the entire experiment**, and it is the
one input the project cannot buy back: the options arm is already dead with ThetaData
lapsed to `Options: FREE`, and CI width on the live lab scales as $\sigma/\sqrt{n_\text{sessions}}$
with sessions, not trades, because 15 correlated names collapse into one cluster. Against
a per-trade SE of \$8.57 on a mean of \$8.72 — $t \approx 1.0$, needing $4\times$ the
sample to reach $t = 2$ — every lost session is unrecoverable evidence.

## 4. Why `reconcile()` was not run here, deliberately

It would have appeared to fix this and would have corrupted the record.

`reconcile()` calls `_evidence_days()` before it dares call a day `MISSED`, and that
function's docstring records why:

> *The first version did not, and on a lab that had been collecting since 2026-08-28 it
> confidently backfilled six good sessions as "the lab did not run" — turning a working
> forward test into a phantom one. A record that contradicts the data on disk is worse
> than no record.*

`_evidence_days()` globs `daily/` and `shares/daily/` **on the host it runs on**. This is
an ephemeral container holding a fresh clone; its evidence set is empty by construction for
every day after 09-04. Running `--reconcile` here would have written six `MISSED` lines for
sessions that — as §3 proves for at least one of them — did run. That is the identical bug,
reintroduced from the identical blind spot, six days later.

**So the rule is now explicit, and `main()` prints it whenever an UNRECORDED day appears:
`--reconcile` may only be run on the lab machine.** Only that host can tell a session it
cannot see from a session that did not happen.

## 5. What to do, in order

1. **On the lab machine**, from the repo root:

       python -m trade_analysis.live_lab.ledger --lab-dir live_lab_data --reconcile

   This writes whatever is true — `COLLECTED` where a daily file exists, `MISSED` where the
   task never fired — and prints the honest coverage. It is idempotent.

2. **Commit `live_lab_data/` from that machine.** The local record is already ahead of the
   committed one (§2); nothing needs reconstructing, only pushing. The frozen artifacts —
   `FREEZE.json`, `shares/FREEZE.json`, `config/*.json`, `shares/config/*.json` — must
   arrive unmodified.

3. **Find out why nothing has been pushed since 09-10** and whether the scheduled task is
   still enabled. That is the actual defect; this audit only measures it.

4. **Do not restart any setup's count.** The pre-registration forbids it, and a `MISSED`
   session is a recorded fact about coverage, not a reason to reset a denominator.

## 6. What this does NOT claim

- Nothing about the live lab's results. This audit reads coverage only, and
  `ledger.py` is deliberately built to know nothing about P&L.
- Nothing about whether the six sessions were collected. §3 is explicit that this clone
  cannot tell, and that the lab machine can.
- Nothing about the 09-09 session's reported −$1,339.77 beyond the fact that it was
  measured somewhere and is not in the repository.
