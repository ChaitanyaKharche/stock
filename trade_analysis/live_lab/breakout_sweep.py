"""Arm A of the breakout pre-registration: does the signal produce enough underlying move?

    python -m trade_analysis.live_lab.breakout_sweep --symbol QQQ
    python -m trade_analysis.live_lab.breakout_sweep --symbol SPY
    python -m trade_analysis.live_lab.breakout_sweep --symbol QQQ --six-line   # S1

Frozen by `research/breakout_options_preregistration.md`.

WHY THIS ARM FIRST, AND WHY IT MAY BE THE ONLY ONE
--------------------------------------------------
The full strategy is: 8-line breakout -> ATM option -> cap at 25-50% -> switch to 1DTE
after 13:00. Three of those four cannot be measured with data this project can obtain:

    QQQ options, any expiry   there is no QQQ option archive. None.
    SPY 1DTE                  the archive is option_quote_1m_0dte/ -- 0DTE by name
    anything new              ThetaData lapsed to `Options: FREE` on 2026-09-08

What remains is the **necessary condition**, and it is the cheap one: if the underlying
does not move enough after the signal, no strike, expiry or profit cap rescues it. From
`shares_runner.py`, an ATM 0DTE needs roughly **+5 bp** of underlying move just to clear
spread and theta. And a **+50% gain on an ATM option** -- the pre-registered cap -- needs
roughly **20 bp**, because at delta ~0.5 a premium of P on spot S needs dS = 0.5*P/0.5 =
P, and P/S is about 20 bp for a 0DTE ATM on these names. Theta only makes that worse.

So this reports, per signal, the underlying move to the strategy's own exits and the
maximum favourable excursion, in basis points, against both thresholds. **If the median
signal does not clear 5 bp, claim B is dead on the underlying alone** and no option
backtest is needed -- which would be the cheapest null this project has ever bought.

WHAT THIS IS NOT
----------------
Not a P&L. There is no option price here and none is invented. Arm B/C needs the SPY
0DTE archive read through `hpc/build_vrp_dataset.py`, and is a separate run.

Needs the feed for bars; `levels_test.get_ext` caches extended hours in
live_lab_data/ext_cache, so the second run is offline.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from .breakout_levels import (COST_FLOOR_BP, build_levels, find_breakouts, move_bp)
from .orb_veto import resample_10m

LAB = Path(__file__).resolve().parents[2] / "live_lab_data"
# Underlying move needed for a +50% gain on an ATM option at delta ~0.5. See the module
# docstring; stated as a constant so the reported fraction is auditable, not folklore.
CAP50_BP = 20.0
SWITCH_AT = dt.time(13, 0)      # primary; after this the strategy wants 1DTE (unmeasurable)
EOD = dt.time(15, 55)


def evaluate_signal(sig, ten) -> dict | None:
    """Underlying outcome for one signal. No option, no P&L -- move only.

    Exits, per §5 of the pre-registration:
      * failed breakout -- a 10-min CLOSE back inside the level
      * 15:55 flatten
    Whichever comes first. MFE is measured over the same window.
    """
    fill_i = next((i for i, b in enumerate(ten) if b["ts"] == sig.fill_ts), None)
    if fill_i is None:
        return None
    entry = ten[fill_i]["open"]
    if entry <= 0:
        return None
    lv = sig.level

    exit_px, exit_reason, exit_ts = None, None, None
    mfe = 0.0
    for b in ten[fill_i:]:
        if b["ts"].time() >= EOD:
            exit_px, exit_reason, exit_ts = b["open"], "eod", b["ts"]
            break
        # MFE on the bar EXTREME in the trade's favour -- an option would have been
        # worth most there, and the cap is a limit order, so it fills on the extreme.
        fav = b["high"] if sig.direction == "long" else b["low"]
        mfe = max(mfe, move_bp(entry, fav, sig.direction))
        # Failed breakout: closed back through the level that triggered the entry.
        back_inside = (b["close"] < lv.price if sig.direction == "long"
                       else b["close"] > lv.price)
        if b["ts"] > sig.fill_ts and back_inside:
            exit_px, exit_reason, exit_ts = b["close"], "failed", b["ts"]
            break
    if exit_px is None:
        exit_px, exit_reason, exit_ts = ten[-1]["close"], "eod", ten[-1]["ts"]

    return {
        "day": sig.ts.date().isoformat(),
        "signal_ts": sig.ts.isoformat(),
        "fill_ts": sig.fill_ts.isoformat(),
        "direction": sig.direction,
        "level": lv.name,
        "entry": round(entry, 4),
        "exit": round(exit_px, 4),
        "exit_reason": exit_reason,
        "exit_ts": exit_ts.isoformat(),
        "move_bp": round(move_bp(entry, exit_px, sig.direction), 2),
        "mfe_bp": round(mfe, 2),
        "post_switch": sig.fill_ts.time() >= SWITCH_AT,
        "hold_min": int((exit_ts - sig.fill_ts).total_seconds() // 60),
    }


def run(symbol: str, start: dt.date, end: dt.date, six_line: bool = False) -> list[dict]:
    from .feed import ThetaLiveFeed
    from .levels_test import get_ext
    from .walkforward import sessions_between

    feed = ThetaLiveFeed()
    rows, history, n = [], [], 0
    try:
        for day in sessions_between(start, end):
            try:
                pre, rth = get_ext(feed, symbol, day)
            except Exception as exc:                          # noqa: BLE001
                print(f"  {day}: skipped ({exc!r})", flush=True)
                continue
            if len(rth) < 300:
                continue                                      # holiday or half day
            if len(history) >= 3:
                levels = build_levels(history[-3:], pre,
                                      include_premarket=not six_line)
                ten = [b for b in resample_10m(rth)
                       if b["ts"].time() >= dt.time(9, 30)]
                for sig in find_breakouts(rth, levels):
                    r = evaluate_signal(sig, ten)
                    if r:
                        rows.append(r)
            history.append(pre + rth)
            history = history[-4:]          # only the lookback is ever needed
            n += 1
            if n % 100 == 0:
                print(f"  {n} sessions, {len(rows)} signals ...", flush=True)
    finally:
        feed.close()
    print(f"\n  {n} sessions, {len(rows)} signals")
    return rows


def boot_mean(by_day: dict, reps=3000, seed=20260828):
    """Day-clustered bootstrap on the mean. Days are the unit; a trending day can throw
    several signals and they are not independent."""
    import random
    keys = list(by_day)
    if len(keys) < 8:
        return (None, None, None, None)
    rng = random.Random(seed)
    flat = [v for k in keys for v in by_day[k]]
    obs = sum(flat) / len(flat)
    ms = []
    for _ in range(reps):
        pool = []
        for _ in range(len(keys)):
            pool.extend(by_day[keys[rng.randrange(len(keys))]])
        ms.append(sum(pool) / len(pool))
    ms.sort()
    neg = sum(1 for x in ms if x <= 0) / len(ms)
    pos = sum(1 for x in ms if x >= 0) / len(ms)
    return (obs, ms[int(.025 * len(ms))], ms[int(.975 * len(ms))],
            max(2 * min(neg, pos), 1 / reps))


def report(symbol: str, rows: list[dict]) -> dict:
    if not rows:
        print("no signals")
        return {}
    pre = [r for r in rows if not r["post_switch"]]
    post = [r for r in rows if r["post_switch"]]

    by_day = defaultdict(list)
    for r in pre:
        by_day[r["day"]].append(r["move_bp"])
    obs, lo, hi, p = boot_mean(by_day)

    mv = [r["move_bp"] for r in pre]
    mfe = [r["mfe_bp"] for r in pre]
    clears5 = sum(1 for x in mv if x > COST_FLOOR_BP) / len(mv) if mv else 0.0
    mfe20 = sum(1 for x in mfe if x > CAP50_BP) / len(mfe) if mfe else 0.0

    print("\n" + "=" * 78)
    print(f"  {symbol}  ARM A -- UNDERLYING MOVE ONLY. NO OPTION PRICE, NO P&L.")
    print("=" * 78)
    print(f"  signals {len(rows)}   pre-switch {len(pre)}   "
          f"post-switch {len(post)} (would need 1DTE; UNMEASURABLE, excluded)")
    print(f"  sessions with a signal: {len(by_day)}")
    print()
    print(f"  move to exit, bp     mean {obs if obs is None else round(obs, 2)}   "
          f"median {round(st.median(mv), 2) if mv else '-'}   "
          f"95% CI [{None if lo is None else round(lo, 2)}, "
          f"{None if hi is None else round(hi, 2)}]   p={p}")
    print(f"  MFE, bp              median {round(st.median(mfe), 2) if mfe else '-'}   "
          f"p75 {round(st.quantiles(mfe, n=4)[2], 2) if len(mfe) > 3 else '-'}")
    print()
    print(f"  clears the {COST_FLOOR_BP:.0f}bp cost floor (move to exit): {clears5:.1%}")
    print(f"  MFE clears {CAP50_BP:.0f}bp (~ +50% on an ATM option):      {mfe20:.1%}")

    # The tail, always, before any verdict. Every edge in this project has lived in it.
    s = sorted(mv)
    k = max(1, len(s) // 100)
    print(f"\n  top 1% of moves ({k}) carry "
          f"{sum(s[-k:]) / sum(x for x in s if x > 0):.0%} of all favourable bp"
          if any(x > 0 for x in s) else "")
    print(f"  worst {k}: {[round(x, 1) for x in s[:k]]}   "
          f"best {k}: {[round(x, 1) for x in s[-k:]]}")

    verdict = []
    if lo is not None and lo > 0:
        verdict.append("  A: move is positive with CI excluding zero -- SURVIVES so far")
    elif lo is None:
        verdict.append("  A: too few sessions for a clustered bootstrap -- INCONCLUSIVE")
    else:
        verdict.append("  A: CI includes zero -- NULL on this symbol")
    if mv and st.median(mv) <= COST_FLOOR_BP:
        verdict.append(f"  B: median move {st.median(mv):.1f}bp does NOT clear the "
                       f"{COST_FLOOR_BP:.0f}bp floor -- claim B is dead on the "
                       f"underlying alone;\n     no option backtest is needed.")
    else:
        verdict.append("  B: median clears the floor; the SPY 0DTE arm is worth running.")
    print("\n" + "\n".join(verdict))
    print("\n  Both symbols must survive A (prereg §8). One symbol is not two observations.")

    return {"symbol": symbol, "n_signals": len(rows), "n_pre_switch": len(pre),
            "n_post_switch_excluded": len(post), "n_sessions_with_signal": len(by_day),
            "mean_move_bp": obs, "ci95": [lo, hi], "p": p,
            "median_move_bp": st.median(mv) if mv else None,
            "median_mfe_bp": st.median(mfe) if mfe else None,
            "frac_clearing_cost_floor": round(clears5, 4),
            "frac_mfe_clearing_cap50": round(mfe20, 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--start", default="2016-01-04")
    ap.add_argument("--end", default="2026-09-17")
    ap.add_argument("--six-line", action="store_true",
                    help="secondary S1: prior days only, no premarket levels")
    args = ap.parse_args(argv)

    print(f"BREAKOUT SWEEP  {args.symbol}  {args.start} -> {args.end}"
          f"{'  [S1 six-line]' if args.six_line else ''}")
    print("frozen by research/breakout_options_preregistration.md")
    print("ARM A ONLY: underlying move. QQQ options and all 1DTE are UNMEASURABLE.\n",
          flush=True)

    rows = run(args.symbol, dt.date.fromisoformat(args.start),
               dt.date.fromisoformat(args.end), six_line=args.six_line)
    if not rows:
        print("no signals; is Theta Terminal running?")
        return 1
    summary = report(args.symbol, rows)
    out = LAB / f"breakout_sweep_{args.symbol}{'_s1' if args.six_line else ''}.json"
    out.write_text(json.dumps({"summary": summary, "signals": rows}, indent=1),
                   encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
