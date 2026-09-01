"""EXPLORATORY: the trader's own levels — premarket H/L and $25 round numbers.

    python -m trade_analysis.live_lab.levels_test --start 2018-01-02 --end 2026-08-27

NOT part of the frozen family. This is a hypothesis he proposed from reading his own chart,
and two of the three levels were genuinely never tested: the frozen PDH/PDL setups are
RTH-only by explicit design, and round numbers are absent from the panel entirely.

Same execution discipline as everything else: signal on a bar CLOSE, fill on the NEXT bar
at the real NBBO. That rule is what turned an apparent +16.46% option edge into +0.70%.

Every parameter is stated. [R] marks a value chosen here rather than taken from a source.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

from .feed import FeedOutage, ThetaLiveFeed
from .sharewf import NOTIONAL, boot, get_quotes
from .walkforward import sessions_between

XCACHE = Path(__file__).resolve().parents[2] / "live_lab_data" / "ext_cache"
BUFFER = 0.0005        # 0.05% of price, matching his VWAP_Reclaim convention
TARGET_R = 2.0         # [R]
CUTOFF = dt.time(15, 0)
EOD = dt.time(15, 55)
ROUND = 25.0


def get_ext(feed, symbol: str, day: dt.date):
    """Full extended-hours 1m bars 04:00-20:00. Returns (premarket, rth)."""
    XCACHE.mkdir(parents=True, exist_ok=True)
    p = XCACHE / f"{symbol}_{day.isoformat()}.pkl"
    if p.exists():
        try:
            return pickle.load(open(p, "rb"))
        except Exception:                                    # noqa: BLE001
            pass
    try:
        rows = feed._get_csv("/stock/history/ohlc", symbol=symbol,
                             start_date=day.isoformat(), end_date=day.isoformat(),
                             interval="1m", start_time="04:00:00", end_time="20:00:00")
    except FeedOutage:
        return ([], [])
    pre, rth = [], []
    for r in rows:
        try:
            ts = dt.datetime.fromisoformat(r["timestamp"].strip('"').replace("Z", ""))
            o, h, l, c = (float(r["open"]), float(r["high"]),
                          float(r["low"]), float(r["close"]))
            v = float(r["volume"])
        except (KeyError, ValueError):
            continue
        if min(o, h, l, c) <= 0:
            continue
        bar = {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}
        t = ts.time()
        if dt.time(4, 0) <= t < dt.time(9, 30):
            pre.append(bar)
        elif dt.time(9, 30) <= t <= dt.time(15, 59):
            rth.append(bar)
    pre.sort(key=lambda b: b["ts"]); rth.sort(key=lambda b: b["ts"])
    pickle.dump((pre, rth), open(p, "wb"))
    return (pre, rth)


def to5(bars, day):
    out, buck = [], defaultdict(list)
    for b in bars:
        m = (b["ts"].hour * 60 + b["ts"].minute) - (9 * 60 + 30)
        if m < 0:
            continue
        ct = dt.datetime.combine(day, dt.time(9, 30)) + dt.timedelta(minutes=(m // 5 + 1) * 5)
        buck[ct].append(b)
    for ct in sorted(buck):
        g = buck[ct]
        if len(g) == 5:
            out.append({"ts": ct, "open": g[0]["open"], "high": max(x["high"] for x in g),
                        "low": min(x["low"] for x in g), "close": g[-1]["close"],
                        "i_end": bars.index(g[-1])})
    return out


def fill(q, bars, i, side):
    if i >= len(bars):
        return None
    v = q.get(bars[i]["ts"].strftime("%H:%M"))
    return None if not v else (v[1] if side == "buy" else v[0])


def run_day(day, pre, rth, quotes, pdh, pdl):
    """All level setups for one session. Returns trade dicts."""
    if len(rth) < 300 or not quotes:
        return []
    out = []
    b5 = to5(rth, day)
    if len(b5) < 5:
        return []
    pmh = max(b["high"] for b in pre) if pre else None
    pml = min(b["low"] for b in pre) if pre else None
    cut = dt.datetime.combine(day, CUTOFF)
    eod = dt.datetime.combine(day, EOD)

    def trade(name, k_end, direction, stop, target):
        ei = k_end + 1
        entry = fill(quotes, rth, ei, "buy" if direction == "long" else "sell")
        if entry is None or entry <= 0:
            return
        j, reason = None, None
        for k in range(ei, len(rth)):
            b = rth[k]
            if direction == "long":
                if b["low"] <= stop:
                    j, reason = k, "stop"; break
                if b["high"] >= target:
                    j, reason = k, "target"; break
            else:
                if b["high"] >= stop:
                    j, reason = k, "stop"; break
                if b["low"] <= target:
                    j, reason = k, "target"; break
            if b["ts"] >= eod:
                j, reason = k, "eod"; break
        if j is None:
            j, reason = len(rth) - 2, "eod"
        px = fill(quotes, rth, j + 1, "sell" if direction == "long" else "buy") \
            or fill(quotes, rth, j, "sell" if direction == "long" else "buy")
        if px is None or px <= 0:
            return
        sgn = 1.0 if direction == "long" else -1.0
        ret = sgn * (px / entry - 1.0)
        out.append({"day": day.isoformat(), "setup": name, "direction": direction,
                    "entry": entry, "exit": px, "exit_reason": reason,
                    "ret": ret, "pnl": ret * NOTIONAL, "hold_min": (j + 1) - ei})

    done = set()
    for n, bar in enumerate(b5):
        if bar["ts"] >= cut:
            break
        c, buf = bar["close"], BUFFER * bar["close"]
        prev = b5[n - 1] if n else None

        # --- PM_Break: first close beyond the premarket range -------------
        if pmh and "PM_Break_L" not in done and c > pmh + buf:
            done.add("PM_Break_L")
            trade("PM_Break", bar["i_end"], "long", pmh, c + TARGET_R * (c - pmh))
        if pml and "PM_Break_S" not in done and c < pml - buf:
            done.add("PM_Break_S")
            trade("PM_Break", bar["i_end"], "short", pml, c - TARGET_R * (pml - c))

        # --- PM_Reject: poked outside the PM range, closed back inside -----
        if pmh and prev and "PM_Reject_S" not in done and prev["high"] > pmh and c < pmh - buf:
            done.add("PM_Reject_S")
            stop = prev["high"]
            if stop > c:
                trade("PM_Reject", bar["i_end"], "short", stop, c - TARGET_R * (stop - c))
        if pml and prev and "PM_Reject_L" not in done and prev["low"] < pml and c > pml + buf:
            done.add("PM_Reject_L")
            stop = prev["low"]
            if stop < c:
                trade("PM_Reject", bar["i_end"], "long", stop, c + TARGET_R * (c - stop))

        # --- Round25: first close through a $25 multiple -------------------
        if prev:
            lvl = round(c / ROUND) * ROUND
            key = f"R{lvl:.0f}"
            if key not in done and abs(c - lvl) < 3.0:
                if prev["close"] <= lvl < c - buf:
                    done.add(key)
                    trade("Round25", bar["i_end"], "long", lvl, c + TARGET_R * (c - lvl))
                elif prev["close"] >= lvl > c + buf:
                    done.add(key)
                    trade("Round25", bar["i_end"], "short", lvl, c - TARGET_R * (lvl - c))

        # --- PDH_PDL including EXTENDED-hours prior session ----------------
        if pdh and "PDX_L" not in done and c > pdh + buf:
            done.add("PDX_L")
            trade("PDH_PDL_ext", bar["i_end"], "long", pdh, c + TARGET_R * (c - pdh))
        if pdl and "PDX_S" not in done and c < pdl - buf:
            done.add("PDX_S")
            trade("PDH_PDL_ext", bar["i_end"], "short", pdl, c - TARGET_R * (pdl - c))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Exploratory test of premarket / round levels.")
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--end", default="2026-08-27")
    ap.add_argument("--symbol", default="QQQ")
    args = ap.parse_args(argv)

    feed = ThetaLiveFeed()
    days = sessions_between(dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end))
    print(f"LEVELS TEST  {args.start} -> {args.end}   next-bar NBBO fills, "
          f"${NOTIONAL:,.0f}/trade\n", flush=True)

    rows, prev, n = [], None, 0
    for day in days:
        pre, rth = get_ext(feed, args.symbol, day)
        if len(rth) < 300:
            continue
        q = get_quotes(feed, args.symbol, day)
        pdh = pdl = None
        if prev:
            ppre, prth = prev
            allb = ppre + prth
            pdh = max(b["high"] for b in allb)
            pdl = min(b["low"] for b in allb)
        if q and prev:
            rows.extend(run_day(day, pre, rth, q, pdh, pdl))
        prev = (pre, rth)
        n += 1
        if n % 100 == 0:
            print(f"  {n} sessions, {len(rows)} trades ...", flush=True)

    print(f"\ncomplete: {n} sessions, {len(rows)} trades")
    (XCACHE.parent / "levels_trades.json").write_text(json.dumps(rows), encoding="utf-8")

    by = defaultdict(list)
    for r in rows:
        by[r["setup"]].append(r)
    print("\n" + "=" * 100)
    print(f"HIS LEVELS -- QQQ shares, ${NOTIONAL:,.0f}/trade, real NBBO, next-bar fills")
    print("=" * 100)
    print(f"  {'setup':<18}{'n':>6}{'win%':>7}{'mean$':>9}{'total$':>12}{'mean%':>10}"
          f"{'95% CI on mean$':>26}{'p':>8}")
    for k, v in sorted(by.items(), key=lambda x: -sum(t["pnl"] for t in x[1])):
        pn = [t["pnl"] for t in v]
        lo, hi, p = boot(pn, [t["day"] for t in v])
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "      --      "
        print(f"  {k:<18}{len(v):>6}{100*sum(1 for x in pn if x>0)/len(pn):>6.1f}%"
              f"{sum(pn)/len(pn):>9.2f}{sum(pn):>12,.0f}"
              f"{100*sum(t['ret'] for t in v)/len(v):>9.4f}%{ci:>26}"
              f"{(f'{p:.4f}' if p is not None else '  --  '):>8}")
    print(f"\n  Holm across {len(by)} new level setups would apply before believing any of these.")
    print("  EXPLORATORY. Not in the frozen family; changes no live-lab rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
