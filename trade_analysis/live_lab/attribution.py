"""Pipeline attribution — where does the money actually leak?

The lab records every stage of

    signal -> option selection -> hypothetical fill -> exit -> underlying twin -> P&L

so failure can be localised instead of guessed at. This module answers the question the
whole forward test exists for:

    "Did the setup correctly predict the underlying, and IF SO, did ATM/ATM+-1 convert
     that into money?"

Those are two independent failures with opposite remedies:

  * direction wrong        -> the setup has no edge. No instrument choice can rescue it.
  * direction right, money wrong -> the setup has an edge and the 0DTE structure eats it.
    That points at the instrument (strike, hold, expiry), not the signal.

No prior study in this project could tell those apart. Everything here is DESCRIPTIVE:
attribution localises a failure, it does not establish an effect. Effect claims come from
dashboard.py and only at the pre-registered promotion bar.

    python -m trade_analysis.live_lab.attribution
    python -m trade_analysis.live_lab.attribution --setup MOMO_CHASE
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from .store import DEFAULT_LAB_DIR, LabStore


def load(store: LabStore):
    return [t for t in store.read("trades.jsonl")
            if t.get("return_pct") is not None
            and t.get("underlying_return") is not None]


def funnel(trades: list[dict]) -> dict:
    """The four-stage pipeline funnel for one group of trades."""
    n = len(trades)
    if not n:
        return {}
    right = [t for t in trades if t["underlying_return"] > 0]
    wrong = [t for t in trades if t["underlying_return"] <= 0]
    conv = [t for t in right if t["return_pct"] > 0]      # direction right AND paid
    leak = [t for t in right if t["return_pct"] <= 0]     # direction right, money lost
    lucky = [t for t in wrong if t["return_pct"] > 0]     # direction wrong, paid anyway
    return {
        "n": n,
        "dir_right": len(right),
        "dir_right_pct": 100.0 * len(right) / n,
        "converted": len(conv),
        "conversion_pct": (100.0 * len(conv) / len(right)) if right else None,
        "leaked": len(leak),
        "lucky": len(lucky),
        "mean_und_right": (sum(t["underlying_return"] for t in right) / len(right)) if right else None,
        "mean_und_leak": (sum(t["underlying_return"] for t in leak) / len(leak)) if leak else None,
        "mean_opt_conv": (sum(t["return_pct"] for t in conv) / len(conv)) if conv else None,
        "mean_opt_leak": (sum(t["return_pct"] for t in leak) / len(leak)) if leak else None,
        "net": sum(t["pnl_net"] for t in trades if t.get("pnl_net") is not None),
    }


def conversion_curve(trades: list[dict], edges=None):
    """Mean option return by underlying-move bucket.

    The bucket where this crosses zero is the CONVERSION THRESHOLD: how far the underlying
    must travel, in its own direction, before an ATM 0DTE option pays for its own spread
    and theta. Measured, not modelled.
    """
    edges = edges or [-9, -0.30, -0.15, -0.05, 0.0, 0.05, 0.10, 0.15, 0.30, 9]
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        grp = [t for t in trades
               if lo <= 100.0 * t["underlying_return"] < hi]
        if not grp:
            out.append((lo, hi, 0, None, None))
            continue
        out.append((lo, hi, len(grp),
                    sum(t["return_pct"] for t in grp) / len(grp),
                    100.0 * sum(1 for t in grp if t["return_pct"] > 0) / len(grp)))
    return out


def _fmt(v, pct=True, dp=1):
    if v is None:
        return "  --  "
    return f"{100*v:+.{dp}f}%" if pct else f"{v:+.{dp}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pipeline attribution (descriptive only).")
    ap.add_argument("--lab-dir", default=str(DEFAULT_LAB_DIR))
    ap.add_argument("--arm", default="ATM", choices=["ATM", "ATM-1", "ATM+1"])
    ap.add_argument("--setup", default=None)
    args = ap.parse_args(argv)

    store = LabStore(args.lab_dir)
    allt = load(store)
    trades = [t for t in allt if t["arm"] == args.arm]
    if args.setup:
        trades = [t for t in trades if t["setup_id"] == args.setup]
    if not trades:
        print("no trades yet")
        return 0

    days = len({str(t["entry_ts"])[:10] for t in trades})
    print("\n" + "=" * 100)
    print(f"PIPELINE ATTRIBUTION  arm={args.arm}"
          + (f"  setup={args.setup}" if args.setup else "")
          + f"   n={len(trades)} trades over {days} session(s)")
    print("=" * 100)
    print("  DESCRIPTIVE ONLY. Localises where money leaks; establishes nothing.")

    f = funnel(trades)
    print(f"""
  STAGE 1  signal fired                              {f['n']:>5}
  STAGE 2  underlying moved the RIGHT way            {f['dir_right']:>5}   ({f['dir_right_pct']:.1f}%)
  STAGE 3  ... and the option PAID                   {f['converted']:>5}   ({f['conversion_pct'] if f['conversion_pct'] is None else round(f['conversion_pct'],1)}% of those)
           ... direction right but money LOST        {f['leaked']:>5}   <- structure leak
           direction wrong but paid anyway           {f['lucky']:>5}   <- noise""")

    print(f"""
  When direction was RIGHT and it PAID    : underlying {_fmt(f['mean_und_right'],dp=3)}  option {_fmt(f['mean_opt_conv'])}
  When direction was RIGHT and it LOST    : underlying {_fmt(f['mean_und_leak'],dp=3)}  option {_fmt(f['mean_opt_leak'])}""")

    print("\n  CONVERSION CURVE -- mean option return by underlying move")
    print("  (where this crosses zero is how far the underlying must travel before an")
    print("   ATM 0DTE option pays for its own spread and theta)")
    print(f"  {'underlying move':<22}{'n':>5}{'mean option':>14}{'win%':>8}")
    for lo, hi, n, m, w in conversion_curve(trades):
        lab = (f"{lo:+.2f}% .. {hi:+.2f}%" if abs(lo) < 9 and abs(hi) < 9
               else (f"<= {hi:+.2f}%" if abs(lo) >= 9 else f">= {lo:+.2f}%"))
        bar = ""
        if m is not None:
            bar = ("#" * min(int(abs(m) * 40), 30)) if m > 0 else ("." * min(int(abs(m) * 40), 30))
        print(f"  {lab:<22}{n:>5}{_fmt(m):>14}{('  --  ' if w is None else f'{w:.0f}%'):>8}  {bar}")

    if not args.setup:
        print("\n  BY SETUP")
        print(f"  {'setup':<28}{'n':>4}{'dir right':>11}{'converted':>11}{'leaked':>8}{'net$':>10}")
        by = defaultdict(list)
        for t in trades:
            by[t["setup_id"]].append(t)
        for k, v in sorted(by.items(), key=lambda x: -len(x[1])):
            g = funnel(v)
            cp = "  --  " if g["conversion_pct"] is None else f"{g['conversion_pct']:.0f}%"
            print(f"  {k:<28}{g['n']:>4}{g['dir_right_pct']:>10.0f}%{cp:>11}"
                  f"{g['leaked']:>8}{g['net']:>10.2f}")

        print("\n  ARM COMPARISON, CONDITIONAL ON DIRECTION BEING RIGHT")
        print("  (this is the strike question: given the setup called it, which strike pays?)")
        print(f"  {'arm':<8}{'n right':>9}{'converted':>11}{'mean option':>14}{'net$ (all)':>12}")
        for arm in ("ATM-1", "ATM", "ATM+1"):
            a = [t for t in allt if t["arm"] == arm]
            if args.setup:
                a = [t for t in a if t["setup_id"] == args.setup]
            if not a:
                continue
            g = funnel(a)
            cp = "  --  " if g["conversion_pct"] is None else f"{g['conversion_pct']:.0f}%"
            r = [t for t in a if t["underlying_return"] > 0]
            mo = (sum(t["return_pct"] for t in r) / len(r)) if r else None
            print(f"  {arm:<8}{g['dir_right']:>9}{cp:>11}{_fmt(mo):>14}{g['net']:>12.2f}")
        print("\n  ATM is the pre-registered primary arm. The wings are recorded for")
        print("  measurement and cannot be promoted, whatever this table shows.")

    print("\n" + "=" * 100)
    if f["dir_right_pct"] < 45:
        print("  READING: direction is failing. No strike choice fixes a signal that is wrong.")
    elif f["conversion_pct"] is not None and f["conversion_pct"] < 60:
        print("  READING: direction is working and the option structure is eating it.")
        print("  That points at the instrument -- strike, hold length, expiry -- not the signal.")
    else:
        print("  READING: both stages functioning at this sample size.")
    print("  None of this is evidence until the promotion bar in dashboard.py is met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
