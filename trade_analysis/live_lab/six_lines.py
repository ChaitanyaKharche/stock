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


CAP_BP = 20.0        # ~ +50% on an ATM option at delta 0.5
COST_FLOOR_BP = 5.0  # what an ATM 0DTE needs for spread and theta (shares_runner.py)


def first_break_trade(rth_1m: Sequence[dict], lines: Sequence[Line],
                      rec: dict, cap_bp: float = CAP_BP) -> dict | None:
    """ONE trade per session: take the FIRST line to close-break, and hold it.

    One per day, not one per line. The 6-line tally shows 92% of sessions break
    something and the median first break is 09:40, so "take every break" is 2-3 trades a
    day against a journal median of 1-2. First-break-only is the version that matches
    what he actually does, and it is the only one of the two that is a strategy rather
    than a description of the weather.

    Entry on the NEXT bar's open -- a close is not knowable until the bar ends.
    Exits: +cap_bp (intrabar limit) | close back inside the broken line | 15:55.
    Precedence is cap-before-close-stop within a bar, because a limit fills intrabar.

    Returns None when nothing broke. Those sessions are the no-trade days and are
    counted separately; they are not zeros in the P&L series.
    """
    broken = [(v["close_break"], name) for name, v in rec.items() if v["close_break"]]
    if not broken:
        return None
    when, name = min(broken)
    line = next(l for l in lines if l.name == name)
    long = line.side == "R"

    ten = [b for b in resample_10m(rth_1m) if RTH_FROM <= b["ts"].time() < RTH_TO]
    sig_i = next((i for i, b in enumerate(ten)
                  if b["ts"].strftime("%H:%M") == when), None)
    if sig_i is None or sig_i + 1 >= len(ten):
        return None                       # no next bar to fill on
    entry = ten[sig_i + 1]["open"]
    if entry <= 0:
        return None
    tp = entry * (1 + cap_bp / 10_000.0) if long else entry * (1 - cap_bp / 10_000.0)

    mfe, cap_px, exit_px, why = 0.0, None, None, None
    for b in ten[sig_i + 1:]:
        fav = b["high"] if long else b["low"]
        mfe = max(mfe, (fav / entry - 1) * 10_000 * (1 if long else -1))
        if cap_px is None and ((long and b["high"] >= tp) or
                               (not long and b["low"] <= tp)):
            cap_px = tp
        back = b["close"] < line.price if long else b["close"] > line.price
        if b["ts"] > ten[sig_i + 1]["ts"] and back:
            exit_px, why = b["close"], "failed"
            break
    if exit_px is None:
        exit_px, why = ten[-1]["close"], "eod"

    def bp(px):
        return round((px / entry - 1) * 10_000 * (1 if long else -1), 2)

    return {"line": name, "source": line.source, "signal_at": when,
            "direction": "long" if long else "short", "entry": round(entry, 4),
            "exit_reason": why, "move_bp": bp(exit_px),
            "move_capped_bp": bp(cap_px) if cap_px is not None else bp(exit_px),
            "cap_hit": cap_px is not None, "mfe_bp": round(mfe, 2)}


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
                        "trade": first_break_trade(rth, lines, rec),
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

    # ---- one trade per session, on the first close break -------------------------
    tr = [r["trade"] for r in rows if r.get("trade")]
    if tr:
        import random
        by_day = {r["day"]: [r["trade"]["move_bp"]] for r in rows if r.get("trade")}
        by_cap = {r["day"]: [r["trade"]["move_capped_bp"]] for r in rows if r.get("trade")}

        def boot(d, reps=3000, seed=20260828):
            ks = list(d)
            if len(ks) < 8:
                return (None, None, None, None)
            rng = random.Random(seed)
            flat = [v for k in ks for v in d[k]]
            obs = sum(flat) / len(flat)
            ms = []
            for _ in range(reps):
                pool = []
                for _ in range(len(ks)):
                    pool.extend(d[ks[rng.randrange(len(ks))]])
                ms.append(sum(pool) / len(pool))
            ms.sort()
            neg = sum(1 for x in ms if x <= 0) / len(ms)
            pos = sum(1 for x in ms if x >= 0) / len(ms)
            return (obs, ms[int(.025 * len(ms))], ms[int(.975 * len(ms))],
                    max(2 * min(neg, pos), 1 / reps))

        uo, ul, uh, up = boot(by_day)
        co, cl, ch, cp = boot(by_cap)
        mv = sorted(x["move_bp"] for x in tr)
        print("\n  " + "-" * 74)
        print(f"  ONE TRADE PER SESSION, on the first close break -- {len(tr)} trades")
        print(f"  no-trade sessions: {n - len(tr)} ({(n - len(tr)) / n:.1%})")
        print(f"  uncapped   mean {uo:+.2f} bp   median {mv[len(mv) // 2]:+.2f}   "
              f"95% CI [{ul:+.2f}, {uh:+.2f}]   p={up:.4f}")
        print(f"  capped@{CAP_BP:.0f}  mean {co:+.2f} bp   "
              f"cap hit {sum(1 for x in tr if x['cap_hit']) / len(tr):.1%}   "
              f"95% CI [{cl:+.2f}, {ch:+.2f}]   p={cp:.4f}")
        print(f"  the {COST_FLOOR_BP:.0f}bp ATM 0DTE cost floor sits "
              f"{'BELOW' if max(uo, co) > COST_FLOOR_BP else 'ABOVE'} both means")
        longs = [x for x in tr if x["direction"] == "long"]
        print(f"  direction split: {len(longs) / len(tr):.0%} long   "
              f"long mean {sum(x['move_bp'] for x in longs) / max(len(longs), 1):+.2f} bp"
              f"   short mean "
              f"{sum(x['move_bp'] for x in tr if x['direction'] == 'short') / max(len(tr) - len(longs), 1):+.2f} bp")
        print("  " + "-" * 74)

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
