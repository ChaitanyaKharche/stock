"""The laptop probes the news_arms pre-registration requires before any replay (section 12,
14.1 probe (h)). One command, one JSON, nothing else touched:

    python -m trade_analysis.live_lab.backfill_probe

Writes research/news_arms/probes.json. Theta Terminal must be running. Takes the shares
timing run (20 sessions into live_lab_backfill/timing_test, a scratch folder) plus four
feed probes. The orchestrator copies the numbers into the pre-registration; it does not
edit them.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from .feed import ThetaLiveFeed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "news_arms" / "probes.json"
STAMP_SESSIONS = ["2021-03-10", "2022-09-14", "2023-06-14", "2024-06-12", "2025-06-11"]
STAMP_SYMBOLS = ["SPY", "QQQ", "XLF"]


def _rows(feed, path, **p):
    try:
        return feed._get_csv(path, **p)
    except Exception as exc:                                  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def stamp_probe(feed) -> dict:
    """Probe (h): is a 1-minute quote row stamped T the quote AT T or the LAST quote of the
    minute? Compare with 1-second rows for the same minute at 10:00 and 13:30."""
    out = {"checks": [], "verdict": None}
    at_T = last = other = 0
    for day in STAMP_SESSIONS:
        for sym in STAMP_SYMBOLS:
            for hhmm in ("10:00", "13:30"):
                h, m = hhmm.split(":")
                m1 = _rows(feed, "/stock/history/quote", symbol=sym, date=day, interval="1m",
                           start_time=f"{hhmm}:00", end_time=f"{h}:{int(m)+1:02d}:00")
                s1 = _rows(feed, "/stock/history/quote", symbol=sym, date=day, interval="1s",
                           start_time=f"{hhmm}:00", end_time=f"{h}:{int(m)+1:02d}:00")
                rec = {"day": day, "symbol": sym, "minute": hhmm}
                if isinstance(m1, dict) or isinstance(s1, dict) or not m1 or not s1:
                    rec["result"] = "no data"
                    out["checks"].append(rec)
                    continue
                r = m1[0]
                mb, ma = float(r["bid"]), float(r["ask"])
                fb, fa = float(s1[0]["bid"]), float(s1[0]["ask"])
                lb, la = float(s1[-1]["bid"]), float(s1[-1]["ask"])
                rec.update({"min_row_stamp": r.get("timestamp"), "min_bid_ask": [mb, ma],
                            "first_sec": [s1[0].get("timestamp"), fb, fa],
                            "last_sec": [s1[-1].get("timestamp"), lb, la], "n_sec_rows": len(s1)})
                if (mb, ma) == (fb, fa) and (mb, ma) != (lb, la):
                    rec["result"] = "AT_T"; at_T += 1
                elif (mb, ma) == (lb, la) and (mb, ma) != (fb, fa):
                    rec["result"] = "LAST_OF_MINUTE"; last += 1
                elif (mb, ma) == (fb, fa) == (lb, la):
                    rec["result"] = "unchanged minute"
                else:
                    rec["result"] = "neither"; other += 1
                out["checks"].append(rec)
    out["counts"] = {"AT_T": at_T, "LAST_OF_MINUTE": last, "neither": other}
    out["verdict"] = ("AT_T" if at_T and not last else "LAST_OF_MINUTE" if last and not at_T
                      else "MIXED_OR_UNDETERMINED")
    return out


def main() -> int:
    feed = ThetaLiveFeed()
    res = {"run_at": dt.datetime.now().isoformat(timespec="seconds"),
           "repo_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip()}
    try:
        print("probe 1/5: stock 1m NBBO depth, SPY 2016-06-14 ...", flush=True)
        r = _rows(feed, "/stock/history/quote", symbol="SPY", date="2016-06-14", interval="1m",
                  start_time="09:30:00", end_time="16:00:00")
        res["stock_nbbo_2016_rows"] = r if isinstance(r, dict) else len(r)
        print("probe 2/5: stock 1m NBBO depth, XLF 2018-06-13 ...", flush=True)
        r = _rows(feed, "/stock/history/quote", symbol="XLF", date="2018-06-13", interval="1m",
                  start_time="09:30:00", end_time="16:00:00")
        res["stock_nbbo_2018_XLF_rows"] = r if isinstance(r, dict) else len(r)
        print("probe 3/5: option chain 1m, SPY 0DTE 2020-06-12 ...", flush=True)
        r = _rows(feed, "/option/history/quote", symbol="SPY", expiration="2020-06-12",
                  date="2020-06-12", interval="1m", start_time="09:30:00", end_time="16:00:00")
        res["option_chain_2020_rows"] = r if isinstance(r, dict) else len(r)
        print("probe 4/5: minute-stamp content (h), 30 checks ...", flush=True)
        res["stamp_probe_h"] = stamp_probe(feed)
        print(f"   verdict: {res['stamp_probe_h']['verdict']} {res['stamp_probe_h']['counts']}")
        try:
            res["feed_stats"] = feed.stats()
        except Exception:                                     # noqa: BLE001
            pass
    finally:
        feed.close()
    print("probe 5/5: shares replay timing, 20 sessions (minutes) ...", flush=True)
    t0 = time.time()
    pr = subprocess.run([sys.executable, "-m", "trade_analysis.live_lab.backfill", "--arm", "shares",
                         "--start", "2024-06-03", "--end", "2024-06-28",
                         "--out", str(ROOT / "live_lab_backfill" / "timing_test")],
                        cwd=ROOT, capture_output=True, text=True)
    res["shares_replay_20_sessions_minutes"] = round((time.time() - t0) / 60.0, 1)
    res["shares_replay_rc"] = pr.returncode
    res["shares_replay_tail"] = (pr.stdout or "").strip().splitlines()[-5:]
    if pr.returncode != 0:
        res["shares_replay_stderr_tail"] = (pr.stderr or "").strip().splitlines()[-5:]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "stamp_probe_h"}, indent=1))
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
