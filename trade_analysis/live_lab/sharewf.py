"""Blind walk-forward: intraday QQQ SHARES. No options, no futures.

    python -m trade_analysis.live_lab.sharewf --start 2016-01-04 --end 2026-08-27

Simulates the live runner trading QQQ stock same-day with no knowledge of the future.

EXECUTION -- the part that killed the options version
    A 1-minute bar stamped T covers [T, T+60s). Its CLOSE triggers the signal and is not
    knowable until T+60s. So every fill happens on the NEXT bar, at the REAL NBBO:

        long  : buy at bar T+1 ASK   ->  sell at bar J+1 BID
        short : sell at bar T+1 BID  ->  buy  at bar J+1 ASK

    Stops and targets are detected intrabar on bar J's high/low, but the FILL is bar J+1's
    quote -- so gapping through a stop costs what it really costs. Historical NBBO comes
    from /v3/stock/history/quote, so the spread is measured, never assumed.

    Pricing options this way turned +16.46% (p=0.0003) into +0.70% (p=0.79). 95.7% of that
    "edge" was one minute of hindsight. The same discipline is applied here from the start.

WHY SHARES MIGHT WORK WHERE OPTIONS DID NOT
    QQQ's spread is ~1 cent on ~$450, about 0.002% round trip. An ATM 0DTE option costs
    ~1% of premium plus theta. Setups whose mean move is +0.02% to +0.05% are hopeless
    against options and comfortably above a share-trading cost. This is not a new signal --
    it is the same signals against a cost floor ~450x lower.

Bars and quotes are cached, so re-running any window costs no API calls.
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
from .session import aggregate_5m
from .setups import ALL_SETUPS
from .walkforward import _Sess, _warm_from, get_bars, holm, sessions_between

QCACHE = Path(__file__).resolve().parents[2] / "live_lab_data" / "quote_cache"
NOTIONAL = 10_000.0
EOD_FLAT = dt.time(15, 55)


def get_quotes(feed, symbol: str, day: dt.date) -> dict:
    """{HH:MM: (bid, ask)} from real historical NBBO."""
    QCACHE.mkdir(parents=True, exist_ok=True)
    p = QCACHE / f"{symbol}_{day.isoformat()}.pkl"
    if p.exists():
        try:
            return pickle.load(open(p, "rb"))
        except Exception:                                    # noqa: BLE001
            pass
    try:
        rows = feed._get_csv("/stock/history/quote", symbol=symbol,
                             start_date=day.isoformat(), end_date=day.isoformat(),
                             interval="1m")
    except FeedOutage:
        return {}
    out = {}
    for r in rows:
        try:
            b, a = float(r["bid"]), float(r["ask"])
            ts = r["timestamp"].strip('"')
        except (KeyError, ValueError):
            continue
        if b > 0 and a >= b:
            out[ts[11:16]] = (b, a)
    pickle.dump(out, open(p, "wb"))
    return out


def _fill(q, bars, idx, side):
    """Real NBBO at bar `idx`. side='buy' -> ask, 'sell' -> bid. None if unquoted."""
    if idx >= len(bars):
        return None
    k = bars[idx]["ts"].strftime("%H:%M")
    v = q.get(k)
    if not v:
        return None
    return v[1] if side == "buy" else v[0]


def replay_day(symbol, day, bars, quotes, warm) -> list[dict]:
    """SINGLE forward pass, exactly the order the live runner uses:
    build each bar's context once -> manage open positions -> then look for new entries.

    The earlier version rebuilt session state inside a nested exit loop for every open
    position, which was O(n^2) per trade on top of the O(n^2) indicator cost. Same results,
    far cheaper, and it matches production ordering instead of merely approximating it.
    """
    live = _Sess(symbol, day, warm, warm.get("prior_day"))
    out, seen5 = [], set()
    counts, dircnt = defaultdict(int), defaultdict(int)
    openpos: dict[str, dict] = {}
    cut = dt.datetime.combine(day, EOD_FLAT)

    def close_at(pos, k, reason):
        xside = "sell" if pos["dir"] == "long" else "buy"
        px = _fill(quotes, bars, k + 1, xside) or _fill(quotes, bars, k, xside)
        if px is None or px <= 0:
            return
        sgn = 1.0 if pos["dir"] == "long" else -1.0
        ret = sgn * (px / pos["entry"] - 1.0)
        out.append({
            "day": day.isoformat(), "setup": pos["setup"], "direction": pos["dir"],
            "entry_ts": bars[pos["ei"]]["ts"].isoformat(), "entry": pos["entry"],
            "exit": px, "exit_reason": reason, "hold_min": (k + 1) - pos["ei"],
            "ret": ret, "pnl": ret * NOTIONAL, "spread_bp": pos["spread_bp"],
            "shares": NOTIONAL / pos["entry"],
        })

    for i in range(len(bars)):
        live.load(bars[:i + 1])
        b = bars[i]
        ctx1 = live.context(b["ts"], "1m", None)
        ctx5 = None
        if live.bars_5m and live.bars_5m[-1]["ts"] not in seen5:
            seen5.add(live.bars_5m[-1]["ts"])
            ctx5 = live.context(live.bars_5m[-1]["ts"], "5m", None)

        # ---- 1. manage open positions FIRST (production ordering) ----
        for sid, pos in list(openpos.items()):
            if i <= pos["ei"]:
                continue
            setup = pos["obj"]
            sig = pos["sig"]
            reason = None
            if sig.direction == "long":
                if sig.stop is not None and b["low"] <= sig.stop:
                    reason = "stop"
                elif sig.target is not None and b["high"] >= sig.target:
                    reason = "target"
            else:
                if sig.stop is not None and b["high"] >= sig.stop:
                    reason = "stop"
                elif sig.target is not None and b["low"] <= sig.target:
                    reason = "target"
            if reason is None and sig.time_exit_min is not None and \
                    (i - pos["ei"]) >= sig.time_exit_min:
                reason = "time"
            if reason is None and sig.bar_exit is not None and \
                    (i - pos["ei"]) >= sig.bar_exit * (5 if setup.timeframe == "5m" else 1):
                reason = "bars"
            if reason is None and sig.trailing:
                c = ctx5 if setup.timeframe == "5m" else ctx1
                if c is not None:
                    try:
                        if setup.manage({"direction": sig.direction,
                                         "state": sig.state}, c):
                            reason = "trail"
                    except Exception:                        # noqa: BLE001
                        pass
            if reason is None and b["ts"] >= cut:
                reason = "eod"
            if reason:
                close_at(pos, i, reason)
                del openpos[sid]

        # ---- 2. then look for new entries ----
        for tf, ctx in (("1m", ctx1), ("5m", ctx5)):
            if ctx is None:
                continue
            for setup in ALL_SETUPS:
                if setup.timeframe != tf or counts[setup.id] >= setup.max_per_day:
                    continue
                if setup.id in openpos:
                    continue
                try:
                    sig = setup.evaluate(ctx)
                except Exception:                            # noqa: BLE001
                    continue
                if sig is None:
                    continue
                if setup.max_per_direction is not None and \
                        dircnt[(setup.id, sig.direction)] >= setup.max_per_direction:
                    continue
                ei = i + 1
                eside = "buy" if sig.direction == "long" else "sell"
                entry = _fill(quotes, bars, ei, eside)
                if entry is None or entry <= 0:
                    continue
                qq = quotes.get(bars[ei]["ts"].strftime("%H:%M"), (entry, entry))
                openpos[setup.id] = {
                    "setup": setup.id, "obj": setup, "sig": sig, "dir": sig.direction,
                    "ei": ei, "entry": entry,
                    "spread_bp": 10000.0 * (qq[1] - qq[0]) / entry,
                }
                counts[setup.id] += 1
                dircnt[(setup.id, sig.direction)] += 1

    for sid, pos in list(openpos.items()):
        close_at(pos, len(bars) - 2, "eod")
    return out


def boot(vals, days, reps=3000, seed=20260828):
    if len(vals) < 10:
        return (None, None, None)
    rng = random.Random(seed)
    by = defaultdict(list)
    for v, d in zip(vals, days):
        by[d].append(v)
    keys = list(by)
    if len(keys) < 8:
        return (None, None, None)
    ms = []
    for _ in range(reps):
        pool = []
        for _ in range(len(keys)):
            pool.extend(by[keys[rng.randrange(len(keys))]])
        ms.append(sum(pool) / len(pool))
    ms.sort()
    neg = sum(1 for x in ms if x <= 0) / len(ms)
    pos = sum(1 for x in ms if x >= 0) / len(ms)
    return (ms[int(.025 * len(ms))], ms[int(.975 * len(ms))],
            max(2 * min(neg, pos), 1 / reps))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Blind walk-forward, QQQ shares only.")
    ap.add_argument("--start", default="2016-01-04")
    ap.add_argument("--end", default="2026-08-27")
    ap.add_argument("--symbol", default="QQQ")
    args = ap.parse_args(argv)

    feed = ThetaLiveFeed()
    days = sessions_between(dt.date.fromisoformat(args.start),
                            dt.date.fromisoformat(args.end))
    print(f"QQQ SHARES blind walk-forward  {args.start} -> {args.end}  ({len(days)} weekdays)")
    print("fills on the NEXT bar at real NBBO; no future information anywhere\n", flush=True)

    rows, hist, n = [], [], 0
    for day in days:
        bars = get_bars(feed, args.symbol, day)
        if len(bars) < 300:
            continue
        if len(hist) >= 10:
            q = get_quotes(feed, args.symbol, day)
            if q:
                rows.extend(replay_day(args.symbol, day, bars,
                                       q, _warm_from(hist[-25:])))
        hist.append((day, bars))
        n += 1
        if n % 100 == 0:
            print(f"  {n} sessions, {len(rows)} trades ...", flush=True)

    print(f"\ncomplete: {n} sessions, {len(rows)} trades")
    out = QCACHE.parent / "sharewf_trades.json"
    out.write_text(json.dumps(rows), encoding="utf-8")
    report(rows)
    return 0


def report(rows):
    if not rows:
        print("no trades")
        return
    by = defaultdict(list)
    for r in rows:
        by[r["setup"]].append(r)
    print("\n" + "=" * 112)
    print(f"QQQ SHARES -- ${NOTIONAL:,.0f} notional per trade, real NBBO both sides, "
          f"next-bar fills")
    print("=" * 112)
    print(f"  {'setup':<27}{'n':>6}{'win%':>7}{'mean$':>9}{'total$':>12}"
          f"{'mean%':>9}{'spread bp':>10}{'p':>8}{'Holm':>8}")
    ps, cache = {}, {}
    for k, v in by.items():
        lo, hi, p = boot([x["pnl"] for x in v], [x["day"] for x in v])
        cache[k] = (v, lo, hi, p)
        if p is not None:
            ps[k] = p
    adj = holm(ps, len(ALL_SETUPS)) if ps else {}
    for k, (v, lo, hi, p) in sorted(cache.items(), key=lambda x: -sum(t["pnl"] for t in x[1][0])):
        pn = [x["pnl"] for x in v]
        a = adj.get(k)
        print(f"  {k:<27}{len(v):>6}{100*sum(1 for x in pn if x>0)/len(pn):>6.1f}%"
              f"{sum(pn)/len(pn):>9.2f}{sum(pn):>12,.0f}"
              f"{100*sum(x['ret'] for x in v)/len(v):>8.4f}%"
              f"{sum(x['spread_bp'] for x in v)/len(v):>10.2f}"
              f"{(f'{p:.4f}' if p is not None else '  --  '):>8}"
              f"{(f'{a:.4f}' if a is not None else '  --  '):>8}")
    clears = [k for k, a in adj.items() if a < 0.05 and sum(t["pnl"] for t in by[k]) > 0]
    print(f"\n  clearing Holm with POSITIVE P&L (m={len(ALL_SETUPS)}): {len(clears)}"
          + (f" -> {clears}" if clears else ""))
    tot = sum(r["pnl"] for r in rows)
    dd = len({r["day"] for r in rows})
    print(f"  all setups pooled: {len(rows)} trades over {dd} sessions, "
          f"${tot:,.0f} on ${NOTIONAL:,.0f}/trade")


if __name__ == "__main__":
    raise SystemExit(main())
