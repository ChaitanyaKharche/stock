"""Historical walk-forward over the FROZEN setups. Underlying only.

    python -m trade_analysis.live_lab.walkforward --start 2024-01-01 --end 2026-08-27

Strictly walk-forward: session D is replayed with indicators seeded ONLY from sessions
before D, exactly as the live runner does. Bars are admitted one at a time; a Context can
never hold a bar past the evaluation instant. Entries and exits are simulated in a single
forward pass per session, in the same order the runner uses.

WHAT THIS IS
    A large-sample test of whether the frozen setups predict the UNDERLYING. Direction is
    the binding question: day 1 of the live lab showed that when direction was right the
    ATM option converted 83% of the time, and the conversion curve crosses zero at roughly
    a 0.05% underlying move. So if a setup cannot call direction, no strike choice saves it.

WHAT THIS IS NOT
    * Not the forward test. Results here CANNOT change a frozen rule, reset a clock, or
      promote a setup. The freeze governs; this is context.
    * Not free of publication bias. The nine researched setups exist because somebody
      published them, which is a selection effect no amount of history removes.
    * Not option P&L. Spread, theta and the ~$6.85/trade NBBO-vs-cash gap are absent.
      Underlying edge is necessary, not sufficient.

Bars are cached to disk, so re-running over the same window costs no API calls.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path

from .feed import FeedOutage, ThetaLiveFeed
from .session import SessionState, aggregate_5m, build_warmup
from .setups import ALL_SETUPS, DEAD_SETUPS, SLOW_SETUPS

CACHE = Path(__file__).resolve().parents[2] / "live_lab_data" / "bars_cache"
EOD_FLAT = dt.time(15, 55)
WARMUP_SESSIONS = 25


class _Sess(SessionState):
    def load(self, bars):
        self._bars_1m = [dict(b) for b in bars]
        self._seen = {b["ts"] for b in self._bars_1m}
        self._rebuild_5m()


# --------------------------------------------------------------------------- bars


def get_bars(feed, symbol: str, day: dt.date) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{symbol}_{day.isoformat()}.pkl"
    if p.exists():
        try:
            return pickle.load(open(p, "rb"))
        except Exception:                                    # noqa: BLE001
            pass
    try:
        bars = feed.minute_bars(symbol, day)
    except FeedOutage:
        return []
    pickle.dump(bars, open(p, "wb"))
    return bars


def sessions_between(start: dt.date, end: dt.date) -> list[dt.date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


# --------------------------------------------------------------------------- exits


def simulate_exit(sig, bars, i0, setup, sess_for_ctx, day):
    """Walk forward from the signal bar and apply the setup's own exit rules.

    Triggers on the bar HIGH/LOW for stop/target, matching how the sources define levels
    and how positions.py behaves live. Returns (exit_index, reason, exit_price).
    """
    entry = bars[i0]["close"]
    cut = dt.datetime.combine(day, EOD_FLAT)
    for j in range(i0 + 1, len(bars)):
        b = bars[j]
        if sig.direction == "long":
            if sig.stop is not None and b["low"] <= sig.stop:
                return j, "stop", sig.stop
            if sig.target is not None and b["high"] >= sig.target:
                return j, "target", sig.target
        else:
            if sig.stop is not None and b["high"] >= sig.stop:
                return j, "stop", sig.stop
            if sig.target is not None and b["low"] <= sig.target:
                return j, "target", sig.target
        if sig.time_exit_min is not None and (j - i0) >= sig.time_exit_min:
            return j, "time", b["close"]
        if sig.bar_exit is not None and (j - i0) >= sig.bar_exit * (5 if setup.timeframe == "5m" else 1):
            return j, "bars", b["close"]
        if sig.trailing:
            sess_for_ctx.load(bars[:j + 1])
            ctx = sess_for_ctx.context(bars[j]["ts"], setup.timeframe, None)
            try:
                if setup.manage({"direction": sig.direction, "state": sig.state}, ctx):
                    return j, "trail", b["close"]
            except Exception:                                # noqa: BLE001
                pass
        if b["ts"] >= cut:
            return j, "eod", b["close"]
    return len(bars) - 1, "eod", bars[-1]["close"]


# --------------------------------------------------------------------------- replay


def replay_session(symbol, day, bars, warm) -> list[dict]:
    live = _Sess(symbol, day, warm, warm.get("prior_day"))
    exitsess = _Sess(symbol, day, warm, warm.get("prior_day"))
    out, seen5, counts, dircnt = [], set(), defaultdict(int), defaultdict(int)
    open_until: dict[str, int] = {}

    for i in range(len(bars)):
        live.load(bars[:i + 1])
        todo = [(bars[i]["ts"], "1m")]
        if live.bars_5m and live.bars_5m[-1]["ts"] not in seen5:
            seen5.add(live.bars_5m[-1]["ts"])
            todo.append((live.bars_5m[-1]["ts"], "5m"))

        for ts, tf in todo:
            ctx = live.context(ts, tf, None)
            for setup in ALL_SETUPS:
                if setup.timeframe != tf:
                    continue
                if counts[setup.id] >= setup.max_per_day:
                    continue
                if open_until.get(setup.id, -1) > i:
                    continue                                  # already_open, as live
                try:
                    sig = setup.evaluate(ctx)
                except Exception:                             # noqa: BLE001
                    continue
                if sig is None:
                    continue
                if setup.max_per_direction is not None and \
                        dircnt[(setup.id, sig.direction)] >= setup.max_per_direction:
                    continue
                j, reason, px = simulate_exit(sig, bars, i, setup, exitsess, day)
                entry = bars[i]["close"]
                sgn = 1.0 if sig.direction == "long" else -1.0
                r = sgn * (px / entry - 1.0)
                risk = abs(entry - sig.stop) / entry if sig.stop else None
                out.append({
                    "day": day.isoformat(), "symbol": symbol, "setup": setup.id,
                    "ts": ts.isoformat(), "direction": sig.direction,
                    "entry": entry, "exit": px, "exit_reason": reason,
                    "bars_held": j - i, "und_return": r,
                    "R": (r / risk) if risk else None,
                })
                counts[setup.id] += 1
                dircnt[(setup.id, sig.direction)] += 1
                open_until[setup.id] = j
    return out


# --------------------------------------------------------------------------- stats


def day_boot(vals, days, reps=3000, seed=20260828):
    if len(vals) < 8:
        return (None, None, None)
    rng = random.Random(seed)
    by = defaultdict(list)
    for v, d in zip(vals, days):
        by[d].append(v)
    keys = list(by)
    if len(keys) < 5:
        return (None, None, None)
    ms = []
    for _ in range(reps):
        pool = []
        for _ in range(len(keys)):
            pool.extend(by[keys[rng.randrange(len(keys))]])
        ms.append(sum(pool) / len(pool))
    ms.sort()
    m = sum(ms) / len(ms)
    neg = sum(1 for x in ms if x <= 0) / len(ms)
    pos = sum(1 for x in ms if x >= 0) / len(ms)
    return (ms[int(0.025 * len(ms))], ms[int(0.975 * len(ms))],
            max(2 * min(neg, pos), 1.0 / reps))


def holm(p: dict, m: int) -> dict:
    adj, run = {}, 0.0
    for i, k in enumerate(sorted(p, key=lambda x: p[x])):
        run = max(run, (m - i) * p[k])
        adj[k] = min(run, 1.0)
    return adj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Historical walk-forward, underlying only.")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-08-27")
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    feed = ThetaLiveFeed()
    days = sessions_between(start, end)
    print(f"walk-forward {start} -> {end}  ({len(days)} weekdays x {len(args.symbols)} symbols)")
    print("strictly walk-forward: session D seeded only from sessions before D\n")

    rows, done = [], 0
    for sym in args.symbols:
        hist: list[tuple[dt.date, list[dict]]] = []
        for day in days:
            bars = get_bars(feed, sym, day)
            if len(bars) < 300:
                continue
            if len(hist) >= 10:
                warm = _warm_from(hist[-WARMUP_SESSIONS:])
                rows.extend(replay_session(sym, day, bars, warm))
            hist.append((day, bars))
            done += 1
            if done % 50 == 0:
                print(f"  {done} symbol-days, {len(rows)} signals ...", flush=True)
    print(f"\ncomplete: {done} symbol-days, {len(rows)} signals")

    out = Path(args.out or (CACHE.parent / "walkforward_signals.json"))
    out.write_text(json.dumps(rows), encoding="utf-8")
    report(rows, done)
    return 0


def _warm_from(hist) -> dict:
    """Build warmup structures from already-fetched prior sessions. No API calls."""
    sessions = {d: b for d, b in hist}
    rvol, imb = defaultdict(list), defaultdict(list)
    for b in list(sessions.values())[-20:]:
        cum = 0.0
        for x in b:
            cum += x["volume"]
            rvol[x["ts"].strftime("%H:%M")].append(cum)
    for b in list(sessions.values())[-14:]:
        o = b[0]["open"]
        if o > 0:
            for x in b:
                imb[x["ts"].strftime("%H:%M")].append(abs(x["close"] / o - 1.0))
    noise = []
    for d in sorted(sessions)[-10:]:
        b = sessions[d]
        o = b[0]["open"]
        noise.append(min(o - min(x["low"] for x in b), max(x["high"] for x in b) - o))
    or60 = []
    for d in sorted(sessions)[-20:]:
        g = [x for x in sessions[d] if x["ts"].time() <= dt.time(10, 30)]
        if g:
            or60.append(max(x["high"] for x in g) - min(x["low"] for x in g))
    last = sorted(sessions)[-1]
    lb = sessions[last]
    pre1, pre5 = [], []
    for d in sorted(sessions)[-2:]:
        pre1.extend(sessions[d])
        pre5.extend(aggregate_5m(sessions[d], d))
    return {
        "rvol_cum_by_minute": {k: sum(v) / len(v) for k, v in rvol.items() if v},
        "imb_sigma_by_minute": {k: sum(v) / len(v) for k, v in imb.items() if v},
        "crabel_stretch": (sum(noise) / len(noise)) if len(noise) >= 5 else None,
        "or60_avg_20d": (sum(or60) / len(or60)) if or60 else None,
        "prior_day": {"date": last, "open": lb[0]["open"],
                      "high": max(x["high"] for x in lb), "low": min(x["low"] for x in lb),
                      "close": lb[-1]["close"]},
        "prefix_1m": pre1, "prefix_5m": pre5, "sessions_used": len(sessions),
    }


def report(rows, symbol_days):
    if not rows:
        print("no signals")
        return
    by = defaultdict(list)
    for r in rows:
        by[r["setup"]].append(r)
    print("\n" + "=" * 108)
    print("UNDERLYING WALK-FORWARD -- does the setup call DIRECTION?")
    print("=" * 108)
    print(f"  {'setup':<27}{'n':>6}{'win%':>7}{'mean move':>12}{'mean R':>9}"
          f"{'95% CI on move':>24}{'p':>8}{'Holm':>8}")
    ps, cache = {}, {}
    for k, v in sorted(by.items(), key=lambda x: -len(x[1])):
        u = [r["und_return"] for r in v]
        d = [r["day"] for r in v]
        lo, hi, p = day_boot(u, d)
        Rs = [r["R"] for r in v if r["R"] is not None]
        cache[k] = (v, u, lo, hi, p, Rs)
        if p is not None:
            ps[k] = p
    adj = holm(ps, len(ALL_SETUPS)) if ps else {}
    for k, (v, u, lo, hi, p, Rs) in sorted(cache.items(), key=lambda x: -len(x[1][0])):
        ci = f"[{100*lo:+.4f}%, {100*hi:+.4f}%]" if lo is not None else "     --     "
        tag = ("  [DEAD]" if k in DEAD_SETUPS else "  [SLOW]" if k in SLOW_SETUPS else "")
        a = adj.get(k)
        print(f"  {k:<27}{len(v):>6}{100*sum(1 for x in u if x>0)/len(u):>6.1f}%"
              f"{100*sum(u)/len(u):>11.4f}%"
              f"{(sum(Rs)/len(Rs)) if Rs else 0:>9.3f}{ci:>24}"
              f"{(f'{p:.4f}' if p is not None else '  --  '):>8}"
              f"{(f'{a:.4f}' if a is not None else '  --  '):>8}{tag}")
    clears = [k for k, a in adj.items() if a < 0.05]
    print(f"\n  clearing Holm (m={len(ALL_SETUPS)}): {len(clears)}"
          + (f" -> {clears}" if clears else ""))
    print("  A positive underlying edge is NECESSARY, not sufficient: spread, theta and the")
    print("  measured ~$6.85/trade NBBO-vs-cash gap are all absent from these numbers.")
    print("  Nothing here may change a frozen rule or reset a forward-test clock.")


if __name__ == "__main__":
    raise SystemExit(main())
