"""An append-only record of what happened to every trading session, successful or not.

WHY THIS EXISTS
---------------
The forward test's value is an unbroken record, and until now a missing
`live_lab_data/daily/<date>.json` was ambiguous in five different ways:

  * the market was shut (holiday), so there was correctly no session;
  * the lab ran and found no trades;
  * the lab aborted -- feed down, preflight failed, entitlement lapsed;
  * the machine was off, or the scheduled task never fired;
  * the lab started and died mid-session.

Those mean completely different things when reading the result, and the file system
cannot tell them apart. 2026-09-08 proved it: the options arm was blocked by a lapsed
subscription and the shares arm started two hours late, and the only trace of either was
prose in a log nobody parses. A gap that is not explained is indistinguishable from a gap
that is, and a denominator you cannot reconstruct is not a denominator.

So: one durable, structured line per state change, written locally, needing no network,
no API entitlement and no feed. Plus `reconcile()`, which supplies the one record the
lab can never write for itself -- the day it never ran at all.

CONTINGENCIES, EXPLICITLY
-------------------------
  feed/API unavailable      the ledger is a local file; nothing here touches a network.
  machine off, task missed   nothing gets written, so reconcile() walks the exchange
                             calendar and emits MISSED for any session with no record.
  crash mid-session          a day left OPENED with no terminal state is marked
                             INTERRUPTED by the next reconcile, not silently dropped.
  the task retries all day   states are DEDUPED: an unchanged outcome does not append a
                             20th identical row, but the attempt counter still rises, so
                             "aborted once" and "aborted all day" stay distinguishable.
  disk full / read-only      every write is best-effort and swallowed. A ledger failure
                             must never be the reason a session does not run.
  corrupt or partial line    tolerated on read and skipped, so one bad append cannot
                             make the whole history unreadable.
  concurrent writers         a single O_APPEND write per record; the runner lock already
                             prevents two supervisors.
  clock/timezone             every timestamp is exchange time and carries its offset.

DELIBERATELY NOT HERE
---------------------
No P&L, no trade data, no judgement about whether a session was *good*. This answers
only "did the lab observe this session, and if not, why not". Mixing the two would make
the coverage record depend on the very results it is supposed to contextualise.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from .clock import now_et
from .store import DEFAULT_LAB_DIR

try:
    from ..bulk_download.trading_days import is_trading_day as _is_session
except Exception:                                            # pragma: no cover
    def _is_session(d):                                      # noqa: D103
        return d.weekday() < 5

LEDGER_NAME = "session_ledger.jsonl"
RTH_OPEN = dt.time(9, 30)
# A start later than this is a PARTIAL session rather than a clean one. Five minutes
# absorbs a slow supervisor without excusing a two-hour late start.
PARTIAL_AFTER = dt.time(9, 35)

# Terminal states, worst to best. `reconcile` and `summary` rank with this, so a day that
# aborted at 09:20 and collected at 11:34 reads as its best achieved state.
RANK = {"MISSED": 0, "INTERRUPTED": 1, "ABORTED": 2, "OPENED": 3,
        "PARTIAL": 4, "COLLECTED": 5, "NOT_A_SESSION": 6}


def ledger_path(lab_dir=None) -> Path:
    return Path(lab_dir or DEFAULT_LAB_DIR) / LEDGER_NAME


def _read(lab_dir=None) -> list[dict]:
    """Every record, oldest first. A malformed line is skipped, never fatal."""
    path = ledger_path(lab_dir)
    out = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(rec, dict) and rec.get("date"):
                    out.append(rec)
    except OSError:
        return []
    return out


def latest_for(day, lab_dir=None) -> dict | None:
    """The most recent record for `day`, or None."""
    key = day.isoformat() if hasattr(day, "isoformat") else str(day)
    found = None
    for rec in _read(lab_dir):
        if rec.get("date") == key:
            found = rec
    return found


def record(day, outcome: str, reason: str = "", lab_dir=None, dedupe: bool = False,
           **extra) -> bool:
    """Append a record. Returns True if a line was written.

    `dedupe=False` (the default, and what the live paths use) appends EVERY call, so
    `attempts` counts real invocations and a day that aborted twenty times is genuinely
    distinguishable from one that aborted once. The scheduled task retries every 30
    minutes, so a bad day costs about fourteen lines -- nothing, against being able to
    say how long it was broken.

    `dedupe=True` is for IDEMPOTENT backfill: reconcile() runs at the top of every
    invocation and must not append another NOT_A_SESSION for the same holiday each time.

    An earlier version deduped unconditionally and claimed the counter still rose. It did
    not -- a suppressed write suppressed the count with it -- so a day that failed all
    day recorded `attempts: 1`. The claim was in the docstring and the test caught it.

    Never raises. A ledger that can break the lab is worse than no ledger.
    """
    try:
        key = day.isoformat() if hasattr(day, "isoformat") else str(day)
        prev = latest_for(key, lab_dir)
        attempts = int((prev or {}).get("attempts", 0)) + 1
        if (dedupe and prev and prev.get("outcome") == outcome
                and prev.get("reason", "") == reason):
            return False
        now = now_et()
        rec = {
            "date": key,
            "outcome": outcome,
            "reason": reason,
            "attempts": attempts,
            "is_session": bool(_is_session(_as_date(key))),
            "recorded_at_et": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "utc_offset_hours": round(-(now.utcoffset().total_seconds() / 3600), 2)
            if now.utcoffset() else None,
        }
        rec.update({k: v for k, v in extra.items() if v is not None})
        path = ledger_path(lab_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        # One O_APPEND write, so a concurrent writer cannot interleave a half line.
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True
    except Exception as exc:                                 # noqa: BLE001
        print(f"[ledger] could not record {outcome} for {day}: {exc!r}")
        return False


def note_opened(day, lab_dir=None, **extra) -> bool:
    """The arms actually started. Marks PARTIAL when the start was late."""
    now = now_et()
    late = now.time() > PARTIAL_AFTER
    reason = (f"started {now:%H:%M} ET, after the {RTH_OPEN:%H:%M} open"
              if late else "")
    return record(day, "OPENED", reason, lab_dir,
                  started_et=now.strftime("%H:%M:%S"), late_start=late, **extra)


def note_finished(day, lab_dir=None, **extra) -> bool:
    """The session ran to its end. PARTIAL if it began after the grace window."""
    prev = latest_for(day, lab_dir) or {}
    late = bool(prev.get("late_start"))
    outcome = "PARTIAL" if late else "COLLECTED"
    reason = prev.get("reason", "") if late else ""
    return record(day, outcome, reason, lab_dir, **extra)


def _as_date(v) -> dt.date:
    if isinstance(v, dt.date):
        return v
    return dt.date.fromisoformat(str(v)[:10])


def _evidence_days(lab_dir=None) -> set[str]:
    """Days for which a completed daily file exists, from either arm.

    reconcile() MUST consult this before calling a day MISSED. The first version did not,
    and on a lab that had been collecting since 2026-08-28 it confidently backfilled six
    good sessions as "the lab did not run" -- turning a working forward test into a
    phantom one. A record that contradicts the data on disk is worse than no record.
    """
    out = set()
    root = Path(lab_dir or DEFAULT_LAB_DIR)
    for sub in ("daily", "shares/daily"):
        try:
            for p in (root / sub).glob("*.json"):
                out.add(p.stem)
        except OSError:
            continue
    return out


def _start_date(lab_dir=None) -> dt.date | None:
    """The first day the lab ever produced evidence, from any source."""
    candidates = []
    recs = _read(lab_dir)
    if recs:
        candidates.append(min(_as_date(r["date"]) for r in recs))
    for sub in ("daily", "shares/daily"):
        d = Path(lab_dir or DEFAULT_LAB_DIR) / sub
        try:
            days = [_as_date(p.stem) for p in d.glob("*.json")]
        except (OSError, ValueError):
            days = []
        if days:
            candidates.append(min(days))
    return min(candidates) if candidates else None


def reconcile(lab_dir=None, today=None) -> list[dict]:
    """Write the records the lab could not write for itself.

    Two cases only it can see, because both are defined by ABSENCE:

      * a trading day with no record at all -- the machine was off, the task never fired,
        or the process died before it could write anything. MISSED.
      * a day left in OPENED with no terminal state -- started and never finished.
        INTERRUPTED.

    Idempotent: running it twice adds nothing the second time.
    """
    written = []
    try:
        start = _start_date(lab_dir)
        if start is None:
            return []
        end = today or now_et().date()
        seen = {}
        for rec in _read(lab_dir):
            seen[rec["date"]] = rec
        evidence = _evidence_days(lab_dir)
        day = start
        while day <= end:
            key = day.isoformat()
            rec = seen.get(key)
            if not _is_session(day):
                if rec is None and record(day, "NOT_A_SESSION",
                                          "market holiday or weekend", lab_dir,
                                          dedupe=True):
                    written.append({"date": key, "outcome": "NOT_A_SESSION"})
            elif rec is None and key in evidence:
                # The ledger predates this session, but the daily file proves it ran.
                if record(day, "COLLECTED",
                          "backfilled: daily file exists, predates the ledger", lab_dir, dedupe=True):
                    written.append({"date": key, "outcome": "COLLECTED"})
            elif rec is None:
                # Today is still in progress until the close; do not call it missed yet.
                if day < end or now_et().time() >= dt.time(16, 0):
                    if record(day, "MISSED",
                              "no record written and no daily file; the lab did not run",
                              lab_dir, dedupe=True):
                        written.append({"date": key, "outcome": "MISSED"})
            elif rec.get("outcome") == "OPENED" and day < end:
                if key in evidence:
                    if record(day, "COLLECTED",
                              "end never recorded, but the daily file was written",
                              lab_dir, dedupe=True):
                        written.append({"date": key, "outcome": "COLLECTED"})
                elif record(day, "INTERRUPTED",
                            "started but never recorded an end", lab_dir, dedupe=True):
                    written.append({"date": key, "outcome": "INTERRUPTED"})
            day += dt.timedelta(days=1)
    except Exception as exc:                                 # noqa: BLE001
        print(f"[ledger] reconcile failed: {exc!r}")
    return written


def coverage(lab_dir=None) -> dict:
    """Best achieved state per day, plus the counts that make a denominator."""
    best = {}
    for rec in _read(lab_dir):
        d = rec["date"]
        if d not in best or RANK.get(rec.get("outcome"), 0) >= RANK.get(
                best[d].get("outcome"), 0):
            best[d] = rec
    sessions = {d: r for d, r in best.items() if r.get("is_session")}
    counts = {}
    for r in sessions.values():
        counts[r.get("outcome", "?")] = counts.get(r.get("outcome", "?"), 0) + 1
    usable = counts.get("COLLECTED", 0) + counts.get("PARTIAL", 0)
    return {
        "days": dict(sorted(best.items())),
        "trading_days": len(sessions),
        "counts": counts,
        "usable": usable,
        "coverage_pct": round(usable / len(sessions) * 100, 1) if sessions else 0.0,
    }


def main(argv=None) -> int:
    """Print the coverage table. `--reconcile` fills in the gaps first."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lab-dir", default=str(DEFAULT_LAB_DIR))
    ap.add_argument("--reconcile", action="store_true",
                    help="write MISSED / INTERRUPTED records before reporting")
    args = ap.parse_args(argv)

    if args.reconcile:
        for w in reconcile(args.lab_dir):
            print(f"  + {w['date']}  {w['outcome']}")

    cov = coverage(args.lab_dir)
    if not cov["days"]:
        print("No ledger yet. It is written as sessions run; use --reconcile to "
              "backfill what can be inferred.")
        return 0
    print(f"\n  {'date':<12}{'outcome':<15}{'att':>4}  reason")
    print("  " + "-" * 76)
    for d, r in cov["days"].items():
        if not r.get("is_session"):
            continue
        print(f"  {d:<12}{r.get('outcome', '?'):<15}{r.get('attempts', 1):>4}  "
              f"{(r.get('reason') or '')[:52]}")
    print("  " + "-" * 76)
    print(f"  {cov['trading_days']} trading days | " +
          " ".join(f"{k} {v}" for k, v in sorted(cov["counts"].items())))
    print(f"  usable {cov['usable']}/{cov['trading_days']} = {cov['coverage_pct']}%")
    print("\n  A gap you can explain is data. A gap you cannot is a hole in the "
          "denominator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
