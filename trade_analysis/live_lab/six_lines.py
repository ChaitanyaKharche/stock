"""The trader's actual 6 lines, and a plain tally of which ones break.

    python -m trade_analysis.live_lab.six_lines --symbol QQQ
    python -m trade_analysis.live_lab.six_lines --symbol SPY

THE LINES, exactly as specified on 2026-09-20
---------------------------------------------
Six at a time, never more, rolling forward one session at a time:

    R3 / R2 / R1   the three HIGHS, R1 nearest above
    S1 / S2 / S3   the three LOWS,  S1 nearest below

sourced from:

    yesterday's PREMARKET high and low      (04:00-09:29 ET on D-1)
    yesterday's MARKET-HOURS high and low   (09:30-16:00 ET on D-1)
    today's PREMARKET high and low          (04:00-09:29 ET on D)

Rolling into D+1: yesterday's four drop off, today's four become "yesterday's",
tomorrow's premarket two arrive. Always exactly six.

WHAT THIS IS NOT, AND WHY
-------------------------
No P&L, no options, no cap, no entries. This is a TALLY -- how often each line breaks,
how many break per session, when the first break happens. Characterise the thing before
pricing anything against it.

`breakout_levels.py` got this wrong in a way worth recording: it used THREE prior
sessions and took each one's high/low over the whole extended session 04:00-20:00. That
merges premarket into market hours, so yesterday's premarket high -- one of the six --
vanishes whenever it sits inside yesterday's RTH range, which is most days. Different
lookback and a different decomposition, so the null in
`research/breakout_options_results.md` is a null about levels the trader does not draw.

BOTH DEFINITIONS OF "BREAK" ARE RECORDED
----------------------------------------
Close-based and touch-based, side by side. For a hypothesis TEST choosing after seeing
results would be a forking path, which is why `orb_veto` froze close-based in advance.
For a tally there is no test to corrupt, and the gap between the two is itself the
interesting number -- on 2026-09-17 QQQ printed 718 on the session's largest volume bar
and closed back inside, which a touch rule calls a breakout and a close rule calls a
failed one.

Dependency-free: bar dicts in, records out.
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from .orb_veto import resample_10m

LAB = Path(__file__).resolve().parents[2] / "live_lab_data"
PRE_FROM, PRE_TO = dt.time(4, 0), dt.time(9, 30)
RTH_FROM, RTH_TO = dt.time(9, 30), dt.time(16, 0)
INTERVAL_MIN = 10


@dc.dataclass(frozen=True)
class Line:
    name: str        # R1..R3 / S1..S3
    price: float
    source: str      # "yday premarket high", "today premarket low", ...
    side: str        # "R" broken by going above; "S" by going below


def _hi_lo(bars: Sequence[dict], lo_t: dt.time, hi_t: dt.time):
    hi = lo = None
    for b in bars:
        t = b["ts"].time()
        if not (lo_t <= t < hi_t):
            continue
        hi = b["high"] if hi is None else max(hi, b["high"])
        lo = b["low"] if lo is None else min(lo, b["low"])
    return hi, lo


def build_six(yday_bars: Sequence[dict], today_pre: Sequence[dict]) -> list[Line]:
    """The six lines for today. `yday_bars` is yesterday's FULL extended session.

    Yesterday contributes FOUR levels, not two: its premarket extremes and its
    market-hours extremes are separate lines even when one contains the other.
    """
    ypre_h, ypre_l = _hi_lo(yday_bars, PRE_FROM, PRE_TO)
    yrth_h, yrth_l = _hi_lo(yday_bars, RTH_FROM, RTH_TO)
    tpre_h, tpre_l = _hi_lo(today_pre, PRE_FROM, PRE_TO)

    highs = [(ypre_h, "yday premarket high"), (yrth_h, "yday market high"),
             (tpre_h, "today premarket high")]
    lows = [(ypre_l, "yday premarket low"), (yrth_l, "yday market low"),
            (tpre_l, "today premarket low")]
    if any(p is None for p, _ in highs + lows):
        return []

    # R1 is the nearest resistance above, so the LOWEST of the three highs.
    # S1 is the nearest support below, so the HIGHEST of the three lows.
    highs.sort(key=lambda x: x[0])
    lows.sort(key=lambda x: -x[0])
    out = [Line(f"R{i}", p, s, "R") for i, (p, s) in enumerate(highs, 1)]
    out += [Line(f"S{i}", p, s, "S") for i, (p, s) in enumerate(lows, 1)]
    return out


def breaks_today(rth_1m: Sequence[dict], lines: Sequence[Line]) -> dict:
    """Which of the six break during market hours, on CLOSE and on TOUCH.

    A line breaks at most once -- the first time. Re-counting every bar above a level
    would turn 'how many lines broke' into 'how long the trend lasted'.

    PRE-BROKEN LINES, and why they are excluded rather than counted
    ---------------------------------------------------------------
    The three highs are not all above price and the three lows are not all below it.
    Yesterday's premarket high is often under today's open, at which point it is not a
    resistance -- it is a level price is already past. Counting it would mark a break at
    09:30 on a session where nothing happened, and on a quiet day inside the remaining
    levels the tally would read "1 of 6 broke" when the honest answer is zero.

    So each line is compared to the RTH OPEN. One already beyond it is flagged
    `pre_broken` and takes no part in the break counts. How many of the six start
    pre-broken is reported on its own, because it says how much of the setup the
    overnight gap has already consumed before the session begins.
    """
    ten = [b for b in resample_10m(rth_1m)
           if RTH_FROM <= b["ts"].time() < RTH_TO]
    rec = {l.name: {"price": round(l.price, 4), "source": l.source, "side": l.side,
                    "pre_broken": False, "close_break": None, "touch_break": None}
           for l in lines}
    if not ten:
        return rec
    open_px = ten[0]["open"]
    for l in lines:
        above = l.side == "R"
        if (above and open_px > l.price) or (not above and open_px < l.price):
            rec[l.name]["pre_broken"] = True

    for b in ten:
        for l in lines:
            r = rec[l.name]
            if r["pre_broken"]:
                continue
            above = l.side == "R"
            if r["touch_break"] is None and (
                    (above and b["high"] > l.price) or
                    (not above and b["low"] < l.price)):
                r["touch_break"] = b["ts"].strftime("%H:%M")
            if r["close_break"] is None and (
                    (above and b["close"] > l.price) or
                    (not above and b["close"] < l.price)):
                r["close_break"] = b["ts"].strftime("%H:%M")
    return rec


def run(symbol: str, start: dt.date, end: dt.date) -> list[dict]:
    from .feed import ThetaLiveFeed
    from .levels_test import get_ext
    from .walkforward import sessions_between

    feed = ThetaLiveFeed()
    rows, prev, tried, thin = [], None, 0, 0
    try:
        for day in sessions_between(start, end):
            tried += 1
            if tried % 200 == 0:
                print(f"  {tried} days tried, {len(rows)} sessions recorded, "
                      f"{thin} thin ...", flush=True)
            try:
                pre, rth = get_ext(feed, symbol, day)
            except Exception as exc:                          # noqa: BLE001
                print(f"  {day}: skipped ({exc!r})", flush=True)
                continue
            if len(rth) < 300:
                thin += 1
                continue
            if prev is not None:
                lines = build_six(prev, pre)
                if lines:
                    rec = breaks_today(rth, lines)
                    rhi, rlo = _hi_lo(rth, RTH_FROM, RTH_TO)
                    rows.append({
                        "day": day.isoformat(),
                        "rth_high": round(rhi, 4), "rth_low": round(rlo, 4),
                        "rth_range_bp": round((rhi / rlo - 1) * 10_000, 1),
                        "lines": rec,
                        "n_close_breaks": sum(1 for v in rec.values()
                                              if v["close_break"]),
                        "n_touch_breaks": sum(1 for v in rec.values()
                                              if v["touch_break"]),
                        "n_pre_broken": sum(1 for v in rec.values()
                                            if v["pre_broken"]),
                    })
            prev = pre + rth
    finally:
        feed.close()
    print(f"\n  {tried} days tried, {len(rows)} sessions recorded, {thin} thin")
    return rows


def report(symbol: str, rows: list[dict]) -> dict:
    if not rows:
        print("no sessions")
        return {}
    n = len(rows)
    dist_c = Counter(r["n_close_breaks"] for r in rows)
    dist_t = Counter(r["n_touch_breaks"] for r in rows)
    per_line = defaultdict(lambda: [0, 0, 0])       # [close, touch, pre_broken]
    first_at = []
    for r in rows:
        for name, v in r["lines"].items():
            if v["close_break"]:
                per_line[name][0] += 1
            if v["touch_break"]:
                per_line[name][1] += 1
            if v["pre_broken"]:
                per_line[name][2] += 1
        t = [v["close_break"] for v in r["lines"].values() if v["close_break"]]
        if t:
            first_at.append(min(t))

    print("\n" + "=" * 78)
    print(f"  {symbol}  SIX LINES -- BREAK TALLY, {n} sessions")
    print("=" * 78)
    print("\n  lines broken per session")
    print(f"  {'n':>3}  {'close':>8} {'':>7}  {'touch':>8}")
    for k in range(7):
        c, t = dist_c.get(k, 0), dist_t.get(k, 0)
        print(f"  {k:>3}  {c:>8} {c / n:>6.1%}  {t:>8} {t / n:>6.1%}")

    print("\n  per line, share of sessions broken")
    print(f"  {'line':<5}{'close':>9}{'touch':>9}{'pre-brk':>9}   most common source")
    src = defaultdict(Counter)
    for r in rows:
        for name, v in r["lines"].items():
            src[name][v["source"]] += 1
    for name in ("R1", "R2", "R3", "S1", "S2", "S3"):
        c, t, pb = per_line[name]
        top = src[name].most_common(1)[0] if src[name] else ("-", 0)
        print(f"  {name:<5}{c / n:>8.1%}{t / n:>9.1%}{pb / n:>9.1%}   "
              f"{top[0]} ({top[1] / n:.0%})")

    contained = dist_c.get(0, 0)
    print(f"\n  sessions where NO line broke on a close: {contained} ({contained / n:.1%})")
    if first_at:
        first_at.sort()
        print(f"  median time of the first close break: {first_at[len(first_at) // 2]} ET")
    gap = sum(r["n_touch_breaks"] - r["n_close_breaks"] for r in rows) / n
    pb = sum(r["n_pre_broken"] for r in rows) / n
    print(f"  of the six, already beyond the open at 09:30: {pb:.2f} on average")
    print(f"  touch minus close, per session: {gap:+.2f} lines "
          f"-- wicks that did not hold")
    rng = sorted(r["rth_range_bp"] for r in rows)
    print(f"  RTH range bp: median {rng[len(rng) // 2]:.0f}   "
          f"p10 {rng[len(rng) // 10]:.0f}   p90 {rng[len(rng) * 9 // 10]:.0f}")

    return {"symbol": symbol, "sessions": n,
            "dist_close": {str(k): v for k, v in sorted(dist_c.items())},
            "dist_touch": {str(k): v for k, v in sorted(dist_t.items())},
            "per_line": {k: {"close": v[0], "touch": v[1], "pre_broken": v[2]}
                         for k, v in sorted(per_line.items())},
            "no_break_sessions": contained}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--start", default="2016-01-04")
    ap.add_argument("--end", default="2026-09-19")
    args = ap.parse_args(argv)

    print(f"SIX LINES  {args.symbol}  {args.start} -> {args.end}")
    print("yday premarket H/L + yday market H/L + today premarket H/L = 6, rolling\n",
          flush=True)
    rows = run(args.symbol, dt.date.fromisoformat(args.start),
               dt.date.fromisoformat(args.end))
    if not rows:
        print("nothing recorded; is Theta Terminal running?")
        return 1
    summary = report(args.symbol, rows)
    out = LAB / f"six_lines_{args.symbol}.json"
    out.write_text(json.dumps({"summary": summary, "sessions": rows}, indent=1),
                   encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
