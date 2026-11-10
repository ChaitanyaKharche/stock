"""Chunked replay runner for the news_arms study (preregistration 17.2, 14.11).

A --year run is split into MONTHS, each replayed into <label>/<arm>/<YYYY-MM>/ with its own
RUN.json. A month that already has RUN.json is skipped, so the same command resumes after a
crash, a sleep or a dead feed (2026-11-05: the 2019 run died mid-way and lost two hours).
A failure writes FAILED.json beside the month and stops with exit 1. --start/--end runs
stay one chunk (the acceptance run used that).

    python -m trade_analysis.news_arms.replay_chunks --arm shares --start 2026-08-28 --end 2026-10-30 --label accept
    python -m trade_analysis.news_arms.replay_chunks --arm options --start 2026-08-28 --end 2026-10-30 --label accept
    python -m trade_analysis.news_arms.replay_chunks --arm shares --year 2018

Each call replays one window through backfill.run into
live_lab_backfill/news_arms/<label>/<arm>/ (label defaults to the year), records the
window, hash, runtime and the git SHA in RUN.json, and for the options arm writes
sessions.json with has_0dte per session (14.11: sessions without a 0DTE are excluded from
every options table; a Mon/Wed/Fri without one, or any such session from 2022-11-14 on,
halts the run for inspection). Refuses to start during 09:00-16:30 ET on a weekday.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import subprocess
import time
import traceback
from pathlib import Path

from ..live_lab import backfill as B
from ..live_lab.clock import now_et
from ..live_lab.walkforward import sessions_between

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "live_lab_backfill" / "news_arms"
ALWAYS_0DTE_FROM = dt.date(2022, 11, 14)       # SPY Tue/Thu expiries listed from here: the
                                               # 2022-11 replay found none on 11-01/03/08/10
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


def months(year: int) -> list[tuple[dt.date, dt.date]]:
    return [(dt.date(year, m, 1), dt.date(year, m, calendar.monthrange(year, m)[1]))
            for m in range(1, 13)]


def terminal_answering() -> bool:
    try:
        from ..live_lab.autostart import terminal_up
        return terminal_up()
    except Exception:                                     # noqa: BLE001
        return False


def replay_window(arm: str, start: dt.date, end: dt.date, out: Path, sha: str) -> dict:
    """One window into `out`. Writes RUN.json on success, FAILED.json on an exception
    (and re-raises so the caller stops)."""
    t0 = time.time()
    print(f"replay {arm} {start} -> {end} into {out} (sha {sha})", flush=True)
    try:
        summary = B.run(start, end, out=out, arm=arm)
    except Exception as exc:                              # noqa: BLE001
        out.mkdir(parents=True, exist_ok=True)
        (out / "FAILED.json").write_text(json.dumps({
            "arm": arm, "start": start.isoformat(), "end": end.isoformat(), "repo_sha": sha,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "traceback_tail": traceback.format_exc().strip().splitlines()[-6:],
            "at": dt.datetime.now().isoformat(timespec="seconds")}, indent=1), encoding="utf-8")
        raise
    meta = {"arm": arm, "start": start.isoformat(), "end": end.isoformat(), "repo_sha": sha,
            "minutes": round((time.time() - t0) / 60.0, 1), "sessions_replayed": len(summary),
            "started_at": dt.datetime.now().isoformat(timespec="seconds")}
    if arm == "options":
        days = [dt.date.fromisoformat(p.stem) for p in sorted((out / "daily").glob("*.json"))]
        has = zero_dte_map("SPY", days)
        (out / "sessions.json").write_text(json.dumps(has, indent=1), encoding="utf-8")
        bad = halt_reasons(has)
        meta["sessions_without_0dte"] = sum(1 for v in has.values() if not v)
        meta["halt"] = bad
        if bad:
            print(f"HALT (14.11): {len(bad)} session(s) without a 0DTE where one was expected: {bad[:10]}")
    (out / "FAILED.json").unlink(missing_ok=True)
    (out / "RUN.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return meta


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
    ap.add_argument("--workers", type=int, default=1,
                    help="months of a --year run replayed in parallel (the terminal allows 4 "
                         "concurrent requests; 4 is the sensible maximum)")
    ap.add_argument("--month", help="internal: replay one YYYY-MM into <label>/<arm>/<YYYY-MM>")
    a = ap.parse_args(argv)
    if in_market_hours(now_et()) and not a.force:
        print("REFUSING: inside 09:00-16:30 ET on a weekday; the lab needs the terminal. Rerun after close.")
        return 2
    if not terminal_answering() and not a.force:
        print("REFUSING: Theta Terminal is not answering on 127.0.0.1:25503. Start it, wait 30 s, rerun.")
        return 2
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    if a.month:
        y, m = (int(x) for x in a.month.split("-"))
        m_start, m_end = months(y)[m - 1]
        out = OUT_ROOT / (a.label or str(y)) / a.arm / a.month
        if (out / "RUN.json").exists():
            return 0
        try:
            meta = replay_window(a.arm, m_start, m_end, out, sha)
        except Exception:                                 # noqa: BLE001
            return 1
        return 3 if meta.get("halt") else 0
    if a.year:
        label = a.label or str(a.year)
        base = OUT_ROOT / label / a.arm
        halted = False
        if a.workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            import sys as _sys
            todo = [f"{a.year}-{ms.month:02d}" for ms, _ in months(a.year)
                    if not (base / f"{a.year}-{ms.month:02d}" / "RUN.json").exists()]
            print(f"{a.arm} {a.year}: {len(todo)} month(s) to run, {a.workers} at a time", flush=True)

            def one(month):
                cmd = [_sys.executable, "-m", "trade_analysis.news_arms.replay_chunks", "--arm", a.arm,
                       "--month", month, "--label", label] + (["--force"] if a.force else [])
                log = base / f"{month}.log"
                base.mkdir(parents=True, exist_ok=True)
                with open(log, "w", encoding="utf-8") as fh:
                    rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
                print(f"  {month}: rc={rc} (log {log.name})", flush=True)
                return month, rc
            import threading
            stop = threading.Event()

            def heartbeat():
                # the month logs are the only live view; echo their last line every 2 min
                while not stop.wait(120):
                    for month in todo:
                        log = base / f"{month}.log"
                        if log.exists() and not (base / month / "RUN.json").exists():
                            try:
                                last = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:]
                            except OSError:
                                last = []
                            if last:
                                print(f"    {month}: {last[0][:90]}", flush=True)
            hb = threading.Thread(target=heartbeat, daemon=True)
            hb.start()
            try:
                with ThreadPoolExecutor(max_workers=a.workers) as ex:
                    results = list(ex.map(one, todo))
            finally:
                stop.set()
            halted = any(rc == 3 for _, rc in results)
            failed = [m for m, rc in results if rc not in (0, 3)]
            if failed:
                print(f"STOPPED: {len(failed)} month(s) failed {failed}; see FAILED.json in each. Rerun to resume.")
            done = sorted(p.name for p in base.glob("*") if (p / "RUN.json").exists())
            print(f"{a.arm} {a.year}: {len(done)}/12 months complete")
            return 3 if halted else (0 if len(done) == 12 else 1)
        for m_start, m_end in months(a.year):
            out = base / f"{a.year}-{m_start.month:02d}"
            if (out / "RUN.json").exists():
                print(f"  {out.name}: done, skipped", flush=True)
                continue
            try:
                meta = replay_window(a.arm, m_start, m_end, out, sha)
            except Exception:                             # noqa: BLE001
                print(f"STOPPED at {out.name}: see {out / 'FAILED.json'}. Rerun the same command to resume.")
                return 1
            if meta.get("halt"):
                halted = True
                break
        done = sorted(p.name for p in base.glob("*") if (p / "RUN.json").exists())
        print(f"{a.arm} {a.year}: {len(done)}/12 months complete {done}")
        return 3 if halted else (0 if len(done) == 12 else 1)
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    label = a.label or str(start.year)
    out = OUT_ROOT / label / a.arm
    if (out / "RUN.json").exists():
        print(f"REFUSING: {out} already holds a completed run; use a new --label or delete it by hand.")
        return 2
    try:
        meta = replay_window(a.arm, start, end, out, sha)
    except Exception:                                     # noqa: BLE001
        print(f"STOPPED: see {out / 'FAILED.json'}. Rerun to regenerate this window.")
        return 1
    return 3 if meta.get("halt") else 0


if __name__ == "__main__":
    raise SystemExit(main())
