"""Rebuild daily/<date>.json for sessions that ran but never wrote a summary.

    python -m trade_analysis.live_lab.backfill_daily --lab-dir live_lab_data/shares
    python -m trade_analysis.live_lab.backfill_daily --lab-dir live_lab_data/shares --apply

Three shares sessions (2026-09-01..03) traded normally and are fully present in
trades.jsonl, but wrote no daily summary: the supervisor terminated both arms at 15:55,
the exact minute they flatten, so the process died between its last fill and write_daily.
Signal reconciliation shows 0 lost trades on all three, so nothing is missing from the
durable record -- only the per-day rollup built FROM it.

That distinction is the whole reason this is safe. Every field below is recomputed from
trades.jsonl and signals.jsonl, which were fsynced as they happened. Nothing is inferred,
nothing is re-priced, and a day with no trades is left alone rather than given an empty
summary -- a session that genuinely never ran must keep showing up as a gap.

Summaries written here carry "backfilled": true and the reason, so they can never be
mistaken for a rollup written live by the runner.

**The reason must not be a guess.** Until 2026-09-11 this module hardcoded the 09-01..03
explanation ("supervisor terminated the arm at 15:55") and stamped it on every session it
touched. That was already false for 2026-09-10, where the host suspended at 15:41:57 ET --
so the tool would have written a confident wrong cause into the durable record, which is
the exact defect class this project exists to avoid. Now the cause is DERIVED from the
record (last trade, last outage, session-end event, any `host_suspend` outage), the
evidence is written alongside it, and an operator-supplied cause is labelled as such.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

from .store import DEFAULT_LAB_DIR, LabStore

REASON_UNKNOWN = ("the arm did not reach write_daily; rebuilt from trades.jsonl. "
                  "Cause NOT determinable from the durable record")
EOD_FLAT_HHMM = "15:55"


def diagnose(store: LabStore, day: str) -> tuple[str, dict]:
    """Work out WHY a session has no summary, from the record alone.

    Returns (reason, evidence). Evidence is written into the summary so a reader can
    re-derive the reason instead of trusting this function.
    """
    outages = [o for o in store.read("outages.jsonl")
               if str(o.get("ts", "")).startswith(day)]
    events = [e for e in store.read("events.jsonl")
              if str(e.get("ts", "")).startswith(day)]
    trades = [t for t in store.read("trades.jsonl")
              if str(t.get("entry_ts", "")).startswith(day)]

    suspends = [o for o in outages if o.get("kind") == "host_suspend"]
    last_trade = max((str(t.get("exit_ts") or "") for t in trades), default="")
    last_outage = max((str(o.get("ts") or "") for o in outages), default="")
    end_events = [e.get("kind") for e in events
                  if e.get("kind") in ("stop", "early_close")]
    last_activity = max(last_trade, last_outage)

    evidence = {
        "last_trade_exit_et": last_trade or None,
        "last_outage_et": last_outage or None,
        "session_end_event": end_events[0] if end_events else None,
        "host_suspend_outages": len(suspends),
        "n_outages": len(outages),
    }

    if suspends:
        first = suspends[0]
        reason = (f"host suspended: {first.get('detail', 'host_suspend outage recorded')}; "
                  f"the arm never reached write_daily")
    elif end_events:
        reason = (f"session recorded '{end_events[0]}' but no summary was written; "
                  f"rebuilt from trades.jsonl")
    elif last_activity and last_activity[11:16] < EOD_FLAT_HHMM:
        reason = (f"the arm stopped producing records at {last_activity[11:19]} ET, "
                  f"before the {EOD_FLAT_HHMM} flatten; cause not in the record")
    else:
        reason = REASON_UNKNOWN
    return reason, evidence


def sessions_missing_summary(store: LabStore) -> list[str]:
    traded: dict[str, list] = defaultdict(list)
    for t in store.read("trades.jsonl"):
        d = str(t.get("entry_ts", ""))[:10]
        if d and t.get("return_pct") is not None:
            traded[d].append(t)
    out = []
    for d in sorted(traded):
        if not (store.root / "daily" / f"{d}.json").exists():
            out.append(d)
    return out


def build(store: LabStore, day: str, arm: str,
          operator_reason: str | None = None) -> dict:
    derived, evidence = diagnose(store, day)
    trades = [t for t in store.read("trades.jsonl")
              if str(t.get("entry_ts", "")).startswith(day)
              and t.get("return_pct") is not None]
    syms = sorted({t.get("symbol") for t in trades if t.get("symbol")})
    hashes = sorted({t.get("config_hash") for t in trades if t.get("config_hash")})
    degraded: dict[str, int] = defaultdict(int)
    for o in store.read("outages.jsonl"):
        if str(o.get("ts", "")).startswith(day) and o.get("kind") == "degraded_bar":
            degraded[o.get("symbol") or "?"] += 1
    # still_open is NOT recoverable after the fact and must not be guessed: a position the
    # runner was holding when it died was either flattened by reconstruct_eod or is still
    # sitting in the archive. Record it as unknown rather than asserting zero.
    return {
        "date": day,
        "arm": arm,
        "config_hash": hashes[0] if len(hashes) == 1 else hashes,
        "symbols": syms,
        "trades_closed": len(trades),
        "net": round(sum(t.get("pnl_net", 0.0) for t in trades), 2),
        "still_open": None,
        "degraded_bars": dict(degraded),
        "feed": None,
        "backfilled": True,
        "backfill_reason": operator_reason or derived,
        "backfill_reason_source": "operator" if operator_reason else "derived",
        "backfill_derived_reason": derived,
        "backfill_evidence": evidence,
        "backfilled_at": dt.datetime.now().isoformat(timespec="seconds"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lab-dir", default=str(DEFAULT_LAB_DIR))
    ap.add_argument("--arm", default="SHARES")
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry run")
    ap.add_argument("--date", action="append", default=None,
                    help="restrict to this session (repeatable). Required when --reason "
                         "is given, so an operator cause can never be smeared across days.")
    ap.add_argument("--reason", default=None,
                    help="operator-supplied cause, recorded with source='operator'. Use "
                         "ONLY for a cause established outside the record (e.g. a Windows "
                         "Kernel-Power event). Needs exactly one --date.")
    args = ap.parse_args(argv)

    if args.reason and len(args.date or []) != 1:
        print("--reason requires exactly one --date", file=sys.stderr)
        return 2

    store = LabStore(args.lab_dir)
    missing = sessions_missing_summary(store)
    if args.date:
        wanted = set(args.date)
        skipped = wanted - set(missing)
        if skipped:
            print(f"note: {', '.join(sorted(skipped))} already has a summary or no "
                  f"trades; not touching it")
        missing = [d for d in missing if d in wanted]
    if not missing:
        print("every session with trades already has a daily summary")
        return 0

    print(f"{'APPLY' if args.apply else 'DRY RUN'} -- {len(missing)} session(s) "
          f"with trades but no summary\n")
    for day in missing:
        rec = build(store, day, args.arm, operator_reason=args.reason)
        print(f"  {day}  trades {rec['trades_closed']:>3}  net "
              f"${rec['net']:+9.2f}  symbols {','.join(rec['symbols'])}"
              f"  degraded {sum(rec['degraded_bars'].values())}")
        print(f"      reason [{rec['backfill_reason_source']}]: {rec['backfill_reason']}")
        print(f"      evidence: {rec['backfill_evidence']}")
        if args.apply:
            store.write_daily(dt.date.fromisoformat(day), rec)
    if not args.apply:
        print("\n  nothing written. re-run with --apply")
    else:
        print(f"\n  wrote {len(missing)} summaries, each marked backfilled:true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
