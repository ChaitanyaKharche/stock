"""Does removing the close-back-inside stop help, on early R1 breaks?

    python -m trade_analysis.live_lab.early_r1_hold --symbol QQQ

Executes `research/early_r1_hold_preregistration.md`, frozen 2026-09-23.
NEEDS THE LIVE FEED -- the stored six-line sweep has no session close price.

Three arms, all long, all uncapped, all exiting at the SESSION CLOSE unless stated:

    A  incumbent  early R1 break, exit on a close back inside R1 (else session close)
    B  proposal   the SAME sessions and entry, exit only at the session close
    C  benchmark  R1 live but NO early break, enter 09:50 open, exit session close

ARM C IS THE POINT. QQQ went from about $110 to about $740 over this sample, so any
buy-and-hold-to-close rule earns that drift. "B returns +X bp" is unreadable on its own;
only B minus C says whether the setup did anything.

EXIT PRICE: `six_lines.py`'s docstring says 15:55, but its code exits at the last
10-minute bar's close, which is the 16:00 session close. This module matches the CODE.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

from .orb_veto import resample_10m
from .six_lines import RTH_FROM, RTH_TO, build_six, breaks_today

LAB = Path(__file__).resolve().parents[2] / "live_lab_data"
EARLY = {"09:30", "09:40"}
CONTROL_BAR = "09:50"        # arm C entry; also the entry of a 09:40 break
COST_FLOOR_BP = 5.0          # what an ATM 0DTE needs for spread and theta
SPLIT = dt.date(2021, 1, 1)  # pre-registered split-half boundary
SEED = 20260923
REPS = 3000


def _bp(entry: float, exit_px: float) -> float:
    """Long move in basis points. 1 bp = 0.01%; on a $740 stock that is ~7.4 cents."""
    return (exit_px / entry - 1.0) * 10_000.0


def evaluate_session(rth_1m, rec) -> dict | None:
    """Arms A/B/C for one session. Returns None when the session is unusable.

    Entry is the open of the bar AFTER the break bar: a close is not knowable until its
    bar has ended, and entering on the break bar's own open would be lookahead.
    """
    ten = [b for b in resample_10m(rth_1m) if RTH_FROM <= b["ts"].time() < RTH_TO]
    if len(ten) < 3:
        return None
    close_px = ten[-1]["close"]
    r1 = rec.get("R1")
    if r1 is None or r1["pre_broken"]:
        return None
    line_px = r1["price"]
    broke_at = r1["close_break"]

    # ---- arms A and B: an early break -------------------------------------------
    if broke_at in EARLY:
        sig = next((i for i, b in enumerate(ten)
                    if b["ts"].strftime("%H:%M") == broke_at), None)
        if sig is None or sig + 1 >= len(ten):
            return None
        entry = ten[sig + 1]["open"]
        if entry <= 0:
            return None

        # A: first bar AFTER the entry bar that closes back below R1 ends it.
        # The entry bar itself cannot trigger it -- matches six_lines.first_break_trade.
        a_exit, a_why = None, "eod"
        for b in ten[sig + 1:]:
            if b["ts"] > ten[sig + 1]["ts"] and b["close"] < line_px:
                a_exit, a_why = b["close"], "failed"
                break
        if a_exit is None:
            a_exit = close_px

        return {"arm": "AB", "broke_at": broke_at,
                "entry_bar": ten[sig + 1]["ts"].strftime("%H:%M"),
                "entry": round(entry, 4),
                "a_bp": round(_bp(entry, a_exit), 3), "a_exit": a_why,
                "b_bp": round(_bp(entry, close_px), 3)}

    # ---- arm C: R1 was live but did not break early ------------------------------
    ci = next((i for i, b in enumerate(ten)
               if b["ts"].strftime("%H:%M") == CONTROL_BAR), None)
    if ci is None:
        return None
    entry = ten[ci]["open"]
    if entry <= 0:
        return None
    return {"arm": "C", "broke_at": broke_at, "entry_bar": CONTROL_BAR,
            "entry": round(entry, 4), "c_bp": round(_bp(entry, close_px), 3)}


# ------------------------------------------------------------------------ inference

def _boot_mean(xs, reps=REPS, seed=SEED):
    """Mean with a bootstrap 95% CI and a two-sided p against zero."""
    if len(xs) < 8:
        return (float("nan"),) * 4
    rng = random.Random(seed)
    n = len(xs)
    obs = sum(xs) / n
    ms = []
    for _ in range(reps):
        ms.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    ms.sort()
    neg = sum(1 for x in ms if x <= 0) / len(ms)
    pos = sum(1 for x in ms if x >= 0) / len(ms)
    return obs, ms[int(.025 * len(ms))], ms[int(.975 * len(ms))], \
        max(2 * min(neg, pos), 1.0 / reps)


def _boot_diff(xs, ys, reps=REPS, seed=SEED):
    """mean(xs) - mean(ys), two independent groups resampled separately."""
    if len(xs) < 8 or len(ys) < 8:
        return (float("nan"),) * 4
    rng = random.Random(seed)
    nx, ny = len(xs), len(ys)
    obs = sum(xs) / nx - sum(ys) / ny
    ds = []
    for _ in range(reps):
        a = sum(xs[rng.randrange(nx)] for _ in range(nx)) / nx
        b = sum(ys[rng.randrange(ny)] for _ in range(ny)) / ny
        ds.append(a - b)
    ds.sort()
    neg = sum(1 for x in ds if x <= 0) / len(ds)
    pos = sum(1 for x in ds if x >= 0) / len(ds)
    return obs, ds[int(.025 * len(ds))], ds[int(.975 * len(ds))], \
        max(2 * min(neg, pos), 1.0 / reps)


def _line(label, tup, n):
    o, lo, hi, p = tup
    if o != o:
        return f"  {label:<34} n={n:<5} too few"
    return (f"  {label:<34} n={n:<5} {o:+8.2f} bp   "
            f"95% CI [{lo:+7.2f}, {hi:+7.2f}]   p={p:.4f}")


# --------------------------------------------------------------------------- report

def report(symbol: str, rows: list[dict]) -> dict:
    ab = [r for r in rows if r["arm"] == "AB"]
    c = [r for r in rows if r["arm"] == "C"]
    if len(ab) < 8 or len(c) < 8:
        print("\nnot enough sessions in one of the arms")
        return {}

    A = [r["a_bp"] for r in ab]
    B = [r["b_bp"] for r in ab]
    C = [r["c_bp"] for r in c]

    print("\n" + "=" * 78)
    print(f"  {symbol}  EARLY R1 BREAK -- does dropping the stop help?")
    print(f"  {len(ab)} early-break sessions, {len(c)} benchmark sessions")
    print("=" * 78)
    print(_line("A  incumbent (stop on re-entry)", _boot_mean(A), len(A)))
    print(_line("B  proposal  (hold to close)", _boot_mean(B), len(B)))
    print(_line("C  benchmark (no early break)", _boot_mean(C), len(C)))

    print("\n  " + "-" * 74)
    d_ba = _boot_diff(B, A)
    d_bc = _boot_diff(B, C)
    print(_line("B - A   what the stop costs", d_ba, min(len(A), len(B))))
    print(_line("B - C   what the SETUP adds", d_bc, min(len(B), len(C))))
    print("  " + "-" * 74)

    # ---- matched entry bar: only the 09:40 breaks, which enter at 09:50 like C ----
    m = [r for r in ab if r["entry_bar"] == CONTROL_BAR]
    if len(m) >= 8:
        print(_line("B - C   matched entry bar only",
                    _boot_diff([r["b_bp"] for r in m], C), len(m)))

    # ---- the pre-registered decision rule ----------------------------------------
    o, lo, hi, p = d_bc
    excl = (lo > 0) or (hi < 0)
    beats = o > COST_FLOOR_BP
    print(f"\n  DECISION RULE (frozen before the run):")
    print(f"    (B - C) > {COST_FLOOR_BP:.0f} bp ?          {o:+.2f}  -> "
          f"{'PASS' if beats else 'FAIL'}")
    print(f"    95% CI excludes zero ?     [{lo:+.2f}, {hi:+.2f}]  -> "
          f"{'PASS' if excl else 'FAIL'}")

    # ---- split-half: the sign must agree -----------------------------------------
    print(f"\n  SPLIT-HALF (sign must agree across both):")
    halves = []
    for lab, keep in (("2016-01 -> 2020-12", lambda d: d < SPLIT),
                      ("2021-01 -> 2026-08", lambda d: d >= SPLIT)):
        bb = [r["b_bp"] for r in ab if keep(dt.date.fromisoformat(r["day"]))]
        cc = [r["c_bp"] for r in c if keep(dt.date.fromisoformat(r["day"]))]
        if len(bb) < 8 or len(cc) < 8:
            print(f"    {lab}  too few")
            halves.append(None)
            continue
        h = _boot_diff(bb, cc)
        halves.append(h[0])
        print(_line(f"  {lab}  B - C", h, min(len(bb), len(cc))))
    agree = (len([h for h in halves if h is not None]) == 2
             and halves[0] * halves[1] > 0)
    print(f"    signs agree ?              -> {'PASS' if agree else 'FAIL'}")

    verdict = beats and excl and agree
    print("\n  " + "=" * 74)
    print(f"  VERDICT: {'SURVIVES -- now run SPY before believing it' if verdict else 'FALSIFIED'}")
    print(f"  This number is IN-SAMPLE (pre-registration section 6). SPY is the only")
    print(f"  genuine out-of-sample test and has never been run.")
    print("  " + "=" * 74)

    return {"symbol": symbol, "n_early": len(ab), "n_bench": len(c),
            "A": _boot_mean(A)[0], "B": _boot_mean(B)[0], "C": _boot_mean(C)[0],
            "B_minus_A": d_ba[0], "B_minus_C": d_bc[0],
            "ci_excludes_zero": excl, "beats_cost_floor": beats,
            "split_half_agree": agree, "verdict": verdict}


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
                print(f"  {tried} days tried, {len(rows)} usable, {thin} thin ...",
                      flush=True)
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
                    got = evaluate_session(rth, rec)
                    if got:
                        got["day"] = day.isoformat()
                        rows.append(got)
            prev = pre + rth
    finally:
        feed.close()
    print(f"\n  {tried} days tried, {len(rows)} usable, {thin} thin")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--start", default="2016-01-04")
    ap.add_argument("--end", default="2026-09-19")
    a = ap.parse_args(argv)

    print(f"EARLY R1 HOLD  {a.symbol}  {a.start} -> {a.end}")
    print("executes research/early_r1_hold_preregistration.md\n", flush=True)
    rows = run(a.symbol, dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end))
    if not rows:
        print("nothing recorded; is Theta Terminal running?")
        return 1
    summary = report(a.symbol, rows)
    out = LAB / f"early_r1_hold_{a.symbol}.json"
    out.write_text(json.dumps({"summary": summary, "sessions": rows}, indent=1),
                   encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
