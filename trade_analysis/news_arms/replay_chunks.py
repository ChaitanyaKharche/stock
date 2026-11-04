"""Chunked replay runner for the news_arms study (preregistration 17.2, 14.11).

    python -m trade_analysis.news_arms.replay_chunks --arm shares --start 2026-08-28 --end 2026-10-30 --label accept
    python -m trade_analysis.news_arms.replay_chunks --arm options --start 2026-08-28 --end 2026-10-30 --label accept
    python -m trade_analysis.news_arms.replay_chunks --arm shares --year 2018

Each call replays one window through backfill.run into
live_lab_backfill/news_arms/<label>/<arm>/ (label defaults to the year), records the
window, hash, runtime and the git SHA in RUN.json, and for the options arm writes
sessions.json with has_0dte per session (14.11: sessions without a 0DTE are excluded from
every options table; a Mon/Wed/Fri without one, or any such session from 2022-11-01 on,
halts the run for inspection). Refuses to start during 09:00-16:30 ET on a weekday.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

from ..live_lab import backfill as B
from ..live_lab.clock import now_et
from ..live_lab.walkforward import sessions_between

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "live_lab_backfill" / "news_arms"
ALWAYS_0DTE_FROM = dt.date(2022, 11, 1)        # SPY 0DTE every weekday from here
MWF = {0, 2, 4}


def in_market_hours(now: dt.datetime) -> bool:
    return now.weekday() < 5 and dt.time(9, 0) <= now.time() <= dt.time(16, 30)


def zero_dte_map(symbol: str, days) -> dict:
    """Which sessions had a 0DTE expiry, via the feed (cached, non-empty only)."""
    from ..live_lab.feed import ThetaLiveFeed
    feed = ThetaLiveFeed()
    out = {}
    try:
        for d in days:
            key = f"strikes_{symbol}_{d}"
            n = B._cached(key, lambda d=d: len(feed._get_csv("/option/list/strikes", symbol=symbol,
                                                               expiration=d.isoformat())))
            out[d.isoformat()] = bool(n)
    finally:
        feed.close()
    return out


def halt_reasons(has_0dte: dict) -> list[str]:
    bad = []
    for d, ok in has_0dte.items():
        day = dt.date.fromisoformat(d)
        if ok:
            continue
        if day >= ALWAYS_0DTE_FROM or day.weekday() in MWF:
            bad.append(d)
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=["shares", "options"], required=True)
    ap.add_argument("--year", type=int)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--label")
    ap.add_argument("--force", action="store_true", help="run even inside market hours (never for the lab machine)")
    a = ap.parse_args(argv)
    if a.year:
        start, end = dt.date(a.year, 1, 1), dt.date(a.year, 12, 31)
    else:
        start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    label = a.label or str(start.year)
    out = OUT_ROOT / label / a.arm
    if in_market_hours(now_et()) and not a.force:
        print("REFUSING: inside 09:00-16:30 ET on a weekday; the lab needs the terminal. Rerun after close.")
        return 2
    if (out / "RUN.json").exists():
        print(f"REFUSING: {out} already holds a completed run; use a new --label or delete it by hand.")
        return 2
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    t0 = time.time()
    print(f"replay {a.arm} {start} -> {end} into {out} (sha {sha})", flush=True)
    summary = B.run(start, end, out=out, arm=a.arm)
    run_meta = {"arm": a.arm, "start": start.isoformat(), "end": end.isoformat(), "label": label,
                "repo_sha": sha, "minutes": round((time.time() - t0) / 60.0, 1),
                "sessions_replayed": len(summary), "started_at": dt.datetime.now().isoformat(timespec="seconds")}
    if a.arm == "options":
        days = sessions_between(start, end)
        has = zero_dte_map("SPY", days)
        (out / "sessions.json").write_text(json.dumps(has, indent=1), encoding="utf-8")
        bad = halt_reasons(has)
        run_meta["sessions_without_0dte"] = sum(1 for v in has.values() if not v)
        run_meta["halt"] = bad
        if bad:
            print(f"HALT (14.11): {len(bad)} session(s) without a 0DTE where one was expected: {bad[:10]}")
    (out / "RUN.json").write_text(json.dumps(run_meta, indent=1), encoding="utf-8")
    print(json.dumps(run_meta, indent=1))
    return 3 if run_meta.get("halt") else 0


if __name__ == "__main__":
    raise SystemExit(main())
