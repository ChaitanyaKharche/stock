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


def evaluate_signal(sig, ten, take_profit_bp: float = CAP50_BP) -> dict | None:
    """Underlying outcome for one signal, BOTH capped and uncapped. No option price.

    Exits, per §5 of the pre-registration, in this precedence:
      1. **profit cap** -- an intrabar touch of `take_profit_bp` in the trade's favour
      2. failed breakout -- a 10-min CLOSE back inside the level
      3. 15:55 flatten

    THE FIRST VERSION OF THIS FUNCTION OMITTED EXIT 1 ENTIRELY, and that was not a
    detail -- it is the exit the strategy is built around. It measured "hold the breakout
    until it fails", which is close to the opposite of "cap profit at 25-50% of premium",
    and it reported a median of -13.79 bp on QQQ as though that were his strategy. It was
    a strategy nobody trades.

    The two numbers are compatible and they answer different questions:

      `move_bp`        terminal drift with no cap -- does the breakout CONTINUE?
      `move_capped_bp` path-dependent with the cap -- is the favourable excursion
                       REACHABLE before the adverse one?

    A capped trade does not need positive drift. It needs the target to be touched
    first, which is a statement about PATH, not about where price ends up. On QQQ the
    median MFE was +34.26 bp against a +20 bp target, so this distinction decides the
    result rather than refining it.

    Precedence note: the cap fires before the same-bar close-stop because a limit order
    fills intrabar and a close is only known at the bar's end. That ordering is
    favourable, so it is stated rather than buried -- if the bar touched the target at
    all, the limit filled.
    """
    fill_i = next((i for i, b in enumerate(ten) if b["ts"] == sig.fill_ts), None)
    if fill_i is None:
        return None
    entry = ten[fill_i]["open"]
    if entry <= 0:
        return None
    lv = sig.level
    long = sig.direction == "long"
    tp_px = entry * (1.0 + take_profit_bp / 10_000.0) if long \
        else entry * (1.0 - take_profit_bp / 10_000.0)

    exit_px = exit_reason = exit_ts = None
    cap_px = cap_ts = None
    mfe = 0.0
    for b in ten[fill_i:]:
        if b["ts"].time() >= EOD:
            if exit_px is None:
                exit_px, exit_reason, exit_ts = b["open"], "eod", b["ts"]
            break
        # MFE on the bar EXTREME in the trade's favour -- an option is worth most there,
        # and the cap is a limit order, so it fills on the extreme.
        fav = b["high"] if long else b["low"]
        mfe = max(mfe, move_bp(entry, fav, sig.direction))

        # 1. the cap. Recorded separately so the uncapped control survives.
        if cap_px is None and ((long and b["high"] >= tp_px)
                               or (not long and b["low"] <= tp_px)):
            cap_px, cap_ts = tp_px, b["ts"]

        # 2. failed breakout: closed back through the level that triggered the entry.
        back_inside = (b["close"] < lv.price if long else b["close"] > lv.price)
        if exit_px is None and b["ts"] > sig.fill_ts and back_inside:
            exit_px, exit_reason, exit_ts = b["close"], "failed", b["ts"]
            # BREAK UNCONDITIONALLY. The position is closed; there is nothing left for a
            # limit order to fill against and nothing left to excurse.
            #
            # This read `if cap_px is not None: break`, so when the breakout failed
            # BEFORE the cap was touched the loop carried on and could register a cap
            # fill on a later bar -- after the position had already been exited. Pure
            # lookahead, and it was not subtle in its effect: on identical QQQ data it
            # moved median MFE 34.26 -> 40.70 bp and the cap-hit rate 64.8% -> 74.5%,
            # which is where the capped mean of +4.98 bp came from.
            break
    if exit_px is None:
        exit_px, exit_reason, exit_ts = ten[-1]["close"], "eod", ten[-1]["ts"]

    capped_px = cap_px if cap_px is not None else exit_px
    capped_ts = cap_ts if cap_ts is not None else exit_ts
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
        "move_capped_bp": round(move_bp(entry, capped_px, sig.direction), 2),
        "cap_hit": cap_px is not None,
        "capped_exit_ts": capped_ts.isoformat(),
        "mfe_bp": round(mfe, 2),
        "post_switch": sig.fill_ts.time() >= SWITCH_AT,
        "hold_min": int((exit_ts - sig.fill_ts).total_seconds() // 60),
        "hold_capped_min": int((capped_ts - sig.fill_ts).total_seconds() // 60),
    }


def run(symbol: str, start: dt.date, end: dt.date, six_line: bool = False) -> list[dict]:
    from .feed import ThetaLiveFeed
    from .levels_test import get_ext
    from .walkforward import sessions_between

    feed = ThetaLiveFeed()
    rows, history, n, tried, thin = [], [], 0, 0, 0
    try:
        for day in sessions_between(start, end):
            # Progress on ATTEMPTS, not successes. The first version incremented only
            # after a usable session, and `len(rth) < 300` skipped silently -- so a
            # symbol whose every fetch came back thin printed NOTHING, for hours, with
            # no way to tell a slow run from a broken one. SPY did exactly that.
            # A counter that only counts successes cannot report failure; the ledger's
            # coverage denominator had the identical defect.
            tried += 1
            if tried % 100 == 0:
                print(f"  {tried} days tried, {n} usable, {thin} thin, "
                      f"{len(rows)} signals ...", flush=True)
            try:
                pre, rth = get_ext(feed, symbol, day)
            except Exception as exc:                          # noqa: BLE001
                print(f"  {day}: skipped ({exc!r})", flush=True)
                continue
            if len(rth) < 300:
                thin += 1
                if thin <= 3 or thin % 250 == 0:
                    print(f"  {day}: only {len(rth)} RTH bars "
                          f"({len(pre)} premarket) -- skipped as thin [{thin} so far]",
                          flush=True)
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
    finally:
        feed.close()
    print(f"\n  {tried} days tried, {n} usable, {thin} thin, {len(rows)} signals")
    if n == 0:
        print("  NOTHING USABLE. Every day came back with <300 RTH bars, which is an\n"
              "  entitlement or endpoint problem, not a slow run. Check that\n"
              "  /stock/history/ohlc serves this symbol with start_time=04:00:00.")
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

    # CLAIM C. The capped series is the strategy he actually trades; the uncapped one
    # above is the control. Reporting only the control is what the first version did,
    # and it announced a null on a strategy nobody trades.
    cap = [r["move_capped_bp"] for r in pre]
    by_day_cap = defaultdict(list)
    for r in pre:
        by_day_cap[r["day"]].append(r["move_capped_bp"])
    cobs, clo, chi, cp = boot_mean(by_day_cap)
    hit = sum(1 for r in pre if r["cap_hit"]) / len(pre) if pre else 0.0
    cwin = sum(1 for x in cap if x > 0) / len(cap) if cap else 0.0
    print("\n  " + "-" * 74)
    print(f"  CAPPED at +{CAP50_BP:.0f}bp -- the exit the strategy is built around")
    print(f"  cap touched first: {hit:.1%} of signals   win rate {cwin:.1%}   "
          f"median hold "
          f"{st.median([r['hold_capped_min'] for r in pre]) if pre else '-'} min")
    print(f"  move, bp             mean {cobs if cobs is None else round(cobs, 2)}   "
          f"median {round(st.median(cap), 2) if cap else '-'}   "
          f"95% CI [{None if clo is None else round(clo, 2)}, "
          f"{None if chi is None else round(chi, 2)}]   p={cp}")
    print(f"  uncapped control     mean {obs if obs is None else round(obs, 2)}   "
          f"median {round(st.median(mv), 2) if mv else '-'}")
    if cobs is not None and obs is not None:
        print(f"  the cap is worth     {cobs - obs:+.2f} bp per signal on the mean")
    print("  " + "-" * 74)

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
    if clo is not None and clo > 0:
        verdict.append("  C: CAPPED mean is positive with CI excluding zero -- the cap is "
                       "doing the work.\n     That is a statement about PATH, not drift, "
                       "so A can be null and C still survive.")
    elif clo is not None:
        verdict.append("  C: capped CI includes zero -- the cap does not rescue it either.")
    print("\n" + "\n".join(verdict))
    print("\n  Both symbols must survive A (prereg §8). One symbol is not two observations.")

    return {"symbol": symbol, "n_signals": len(rows), "n_pre_switch": len(pre),
            "n_post_switch_excluded": len(post), "n_sessions_with_signal": len(by_day),
            "mean_move_bp": obs, "ci95": [lo, hi], "p": p,
            "median_move_bp": st.median(mv) if mv else None,
            "median_mfe_bp": st.median(mfe) if mfe else None,
            "frac_clearing_cost_floor": round(clears5, 4),
            "frac_mfe_clearing_cap50": round(mfe20, 4),
            "cap_bp": CAP50_BP,
            "frac_cap_hit_first": round(hit, 4),
            "capped_win_rate": round(cwin, 4),
            "capped_mean_bp": cobs, "capped_ci95": [clo, chi], "capped_p": cp,
            "capped_median_bp": st.median(cap) if cap else None,
            "cap_value_bp": None if (cobs is None or obs is None) else round(cobs - obs, 3)}


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
