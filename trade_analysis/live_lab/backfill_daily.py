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
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

from .store import DEFAULT_LAB_DIR, LabStore

REASON = ("supervisor terminated the arm at 15:55, before write_daily; "
          "rebuilt from trades.jsonl (reconciliation: 0 lost)")


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


def build(store: LabStore, day: str, arm: str) -> dict:
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
        "backfill_reason": REASON,
        "backfilled_at": dt.datetime.now().isoformat(timespec="seconds"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lab-dir", default=str(DEFAULT_LAB_DIR))
    ap.add_argument("--arm", default="SHARES")
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry run")
    args = ap.parse_args(argv)

    store = LabStore(args.lab_dir)
    missing = sessions_missing_summary(store)
    if not missing:
        print("every session with trades already has a daily summary")
        return 0

    print(f"{'APPLY' if args.apply else 'DRY RUN'} -- {len(missing)} session(s) "
          f"with trades but no summary\n")
    for day in missing:
        rec = build(store, day, args.arm)
        print(f"  {day}  trades {rec['trades_closed']:>3}  net "
              f"${rec['net']:+9.2f}  symbols {','.join(rec['symbols'])}"
              f"  degraded {sum(rec['degraded_bars'].values())}")
        if args.apply:
            store.write_daily(dt.date.fromisoformat(day), rec)
    if not args.apply:
        print("\n  nothing written. re-run with --apply")
    else:
        print(f"\n  wrote {len(missing)} summaries, each marked backfilled:true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
