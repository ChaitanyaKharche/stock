"""Offline replay -- the pre-freeze verification gate.

Two jobs:

1. PATH-INDEPENDENCE + STRUCTURAL PROBE. For every bar T, evaluate each setup twice:
   once against the session built INCREMENTALLY bar by bar, and once against a session
   rebuilt in ONE SHOT from the full day truncated to T. A setup that carries hidden
   state across calls, or an indicator whose value depends on arrival order, differs
   between the two.

   Note what this does and does not prove. It cannot prove "no lookahead" by feeding a
   setup the future, because the architecture makes that impossible: a Context is built
   from SessionState's buffer, which ends at the evaluation bar, so future bars are not
   merely unused -- they are absent. The structural assertion here checks exactly that
   invariant (newest bar <= the evaluation instant) on every single evaluation, which is
   the guarantee that actually matters. The one-shot comparison catches the residual
   class of bugs the invariant cannot: order-dependence and leaked state.

   Truncation is timeframe-aware. 1m bars carry the vendor's OPEN stamp; 5m buckets are
   stamped at their CLOSE. Getting that wrong is what the probe flagged on its first run.

2. FREQUENCY CALIBRATION. Appendix B's time-to-verdict table rests on *estimated*
   signals/month. This measures them, so the timeline can be corrected before anyone
   waits nine months on a guess.

Replay uses historical bars only and opens no positions -- it never touches the options
subscription, so it is safe to run after 2026-09-05.

    python -m trade_analysis.live_lab.replay --days 20
    python -m trade_analysis.live_lab.replay --days 5 --symbols QQQ --verbose
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import defaultdict

from .clock import today_et
from .feed import FeedOutage, ThetaLiveFeed
from .session import SessionState, build_warmup
from .setups import ALL_SETUPS, DEAD_SETUPS, SLOW_SETUPS


def _sig_key(sig):
    """Comparable fingerprint of a Signal, rounded so float noise is not a violation."""
    if sig is None:
        return None
    def r(x):
        return None if x is None else round(float(x), 6)
    return (sig.setup_id, sig.direction, r(sig.stop), r(sig.target),
            sig.time_exit_min, sig.bar_exit, sig.trailing)


def _sessions_before(day: dt.date, n: int) -> list[dt.date]:
    out, d = [], day - dt.timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return sorted(out)


class _FrozenSession(SessionState):
    """A session pre-loaded with bars, bypassing the completeness clock (offline only)."""

    def load(self, bars):
        self._bars_1m = [dict(b) for b in bars]
        self._seen = {b["ts"] for b in self._bars_1m}
        self._rebuild_5m()


def replay_day(feed, symbol: str, day: dt.date, warmup: dict, verbose=False):
    """Returns (signals, lookahead_violations)."""
    bars = feed.minute_bars(symbol, day)
    if len(bars) < 200:
        return [], []

    prior = warmup.get("prior_day")
    live = _FrozenSession(symbol, day, warmup, prior)
    full = _FrozenSession(symbol, day, warmup, prior)
    full.load(bars)                                  # holds the WHOLE day, always

    signals, violations = [], []
    seen_5m: set = set()
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for i in range(len(bars)):
        live.load(bars[:i + 1])
        bar_ts = bars[i]["ts"]

        todo = [(bar_ts, "1m")]
        if live.bars_5m and live.bars_5m[-1]["ts"] not in seen_5m:
            seen_5m.add(live.bars_5m[-1]["ts"])
            todo.append((live.bars_5m[-1]["ts"], "5m"))

        for ts, tf in todo:
            ctx_live = live.context(ts, tf, None)

            # Probe: rebuild the same instant in ONE SHOT from the full day's bars,
            # instead of incrementally. Catches hidden state carried across calls and
            # any order-dependence in indicator computation.
            #
            # Truncation is timeframe-aware, because the two stamping conventions differ:
            #   1m bars carry the vendor's OPEN stamp   -> the bar stamped T is included
            #   5m buckets are stamped at their CLOSE   -> constituents are ts < T
            cutoff = [b for b in bars if (b["ts"] <= ts if tf == "1m" else b["ts"] < ts)]
            full_trunc = _FrozenSession(symbol, day, warmup, prior)
            full_trunc.load(cutoff)
            ctx_probe = full_trunc.context(ts, tf, None)

            # Structural assertion: a Context can never hold a bar past its own instant.
            if ctx_live.bars_1m:
                newest = ctx_live.bars_1m[-1]["ts"]
                limit = ts if tf == "1m" else ts - dt.timedelta(minutes=1)
                if newest > limit:
                    violations.append({"kind": "LOOKAHEAD", "setup": "<session>",
                                       "ts": ts.isoformat(),
                                       "live": f"newest_bar={newest}", "probe": f"limit={limit}"})

            for setup in ALL_SETUPS:
                if setup.timeframe != tf:
                    continue
                try:
                    a = setup.evaluate(ctx_live)
                except Exception as exc:                     # noqa: BLE001
                    violations.append({"kind": "exception", "setup": setup.id,
                                       "ts": ts.isoformat(), "detail": repr(exc)})
                    continue
                try:
                    b = setup.evaluate(ctx_probe)
                except Exception as exc:                     # noqa: BLE001
                    violations.append({"kind": "exception_probe", "setup": setup.id,
                                       "ts": ts.isoformat(), "detail": repr(exc)})
                    continue
                if _sig_key(a) != _sig_key(b):
                    violations.append({"kind": "LOOKAHEAD", "setup": setup.id,
                                       "ts": ts.isoformat(),
                                       "live": _sig_key(a), "probe": _sig_key(b)})
                if a is None:
                    continue
                key = (setup.id, symbol)
                if counts[key] >= setup.max_per_day:
                    continue
                counts[key] += 1
                signals.append({"setup": setup.id, "symbol": symbol, "day": day,
                                "ts": ts, "direction": a.direction,
                                "stop": a.stop, "target": a.target})
                if verbose:
                    print(f"    {ts:%H:%M} {setup.id:<26} {a.direction}")
    return signals, violations


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline replay gate for the live lab.")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    feed = ThetaLiveFeed()
    today = today_et()
    days = _sessions_before(today, args.days)

    all_sigs, all_viol = [], []
    per_symbol_days: dict[str, int] = defaultdict(int)

    for sym in args.symbols:
        for day in days:
            try:
                warm = build_warmup(feed, sym, day, _sessions_before(day, 25))
            except FeedOutage as exc:
                print(f"  ! warmup {sym} {day}: {exc}")
                continue
            if warm["sessions_used"] < 10:
                continue
            try:
                sigs, viol = replay_day(feed, sym, day, warm, args.verbose)
            except FeedOutage as exc:
                print(f"  ! replay {sym} {day}: {exc}")
                continue
            if sigs or viol:
                print(f"  {sym} {day}: {len(sigs):>3} signals"
                      + (f"   {len(viol)} VIOLATIONS" if viol else ""))
            per_symbol_days[sym] += 1
            all_sigs.extend(sigs)
            all_viol.extend(viol)

    print("\n" + "=" * 84)
    print("PATH-INDEPENDENCE + STRUCTURAL PROBE")
    print("=" * 84)
    hard = [v for v in all_viol if v["kind"] == "LOOKAHEAD"]
    errs = [v for v in all_viol if v["kind"] != "LOOKAHEAD"]
    if hard:
        print(f"  *** {len(hard)} LOOKAHEAD VIOLATIONS -- DO NOT FREEZE ***")
        for v in hard[:10]:
            print(f"    {v['setup']:<26} {v['ts']}  live={v['live']} probe={v['probe']}")
    else:
        print("  PASS -- every setup returned an identical signal whether or not the")
        print("  session also held the rest of the day.")
    if errs:
        print(f"\n  {len(errs)} setup exceptions:")
        seen = set()
        for v in errs:
            k = (v["setup"], v["detail"][:60])
            if k in seen:
                continue
            seen.add(k)
            print(f"    {v['setup']:<26} {v['detail'][:90]}")

    print("\n" + "=" * 84)
    print("MEASURED SIGNAL FREQUENCY  (replaces Appendix B estimates)")
    print("=" * 84)
    n_days = sum(per_symbol_days.values())
    by_setup: dict[str, int] = defaultdict(int)
    for s in all_sigs:
        by_setup[s["setup"]] += 1
    print(f"  symbol-days replayed: {n_days}\n")
    print(f"  {'setup':<28}{'signals':>9}{'per sym-day':>13}{'per month':>11}   note")
    N_REQUIRED = 377
    for setup in ALL_SETUPS:
        n = by_setup.get(setup.id, 0)
        per_day = n / n_days if n_days else 0.0
        per_month = per_day * 21
        both = per_month * len(args.symbols)
        note = ""
        if per_month <= 0:
            note = "NO SIGNALS in window"
        else:
            months = N_REQUIRED / both if both else 9e9
            note = f"~{months:.1f} mo to n={N_REQUIRED} on {len(args.symbols)} symbols"
            if setup.id in DEAD_SETUPS:
                note += "  [DEAD]"
            elif setup.id in SLOW_SETUPS:
                note += "  [SLOW]"
        print(f"  {setup.id:<28}{n:>9}{per_day:>13.2f}{per_month:>11.1f}   {note}")

    print(f"\n  total signals: {len(all_sigs)}")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
