"""What happens AFTER R1/S1 breaks early: momentum to R2/S2, or reversal?

    python -m trade_analysis.live_lab.after_break --symbol QQQ

Executes `research/after_break_preregistration.md`, frozen 2026-09-23. Reads the stored
sweep `live_lab_data/six_lines_<SYM>.json` -- no feed, no API calls, offline.

Three questions, from the trader:

  1. R1 (or S1) breaks in the first 15 minutes. Does price then run to R2/S2, or turn
     around straight away?
  2. If it gets to R2/S2, does it break that too, or stall?
  3. Do R1/S1 sit at round numbers -- multiples of 25?

The 10-minute grid means "first 15 minutes" is really the first TWO bars, 09:30 and
09:40, so 20 minutes. Recorded in the pre-registration rather than quietly rounded.

THE CONTROL IS THE POINT. Every rate is printed early / late / all. "70% go on to break
R2 after an early break" means nothing if 70% break R2 anyway. Only the early-vs-late
GAP is evidence that the timing of the break carries information.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "live_lab_data"
EARLY = {"09:30", "09:40"}          # first two 10-minute bars
ROUND_MULTIPLES = (1, 5, 10, 25)
ERA_SPLIT = 300.0                    # QQQ went 110 -> 740; test both halves separately


# --------------------------------------------------------------------------- stats

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion with a Wilson 95% interval. Behaves at k=0 and k=n, unlike normal."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def two_prop_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p for two proportions being equal (pooled normal approximation)."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def median(xs):
    s = sorted(xs)
    if not s:
        return float("nan")
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def round_dist(price: float, m: int) -> float:
    """Distance in DOLLARS from `price` to the nearest multiple of `m`. In [0, m/2]."""
    r = price % m
    return min(r, m - r)


def mins(hhmm: str) -> int:
    h, mm = hhmm.split(":")
    return int(h) * 60 + int(mm)


# ----------------------------------------------------------------------- extraction

def side_rows(sessions, side: str):
    """One row per session where the side's nearest line (R1/S1) was LIVE and broke.

    Conditioning on R1 being live makes R2 and R3 live automatically: the three highs are
    sorted ascending and `pre_broken` means the 09:30 open is already past the line, so
    open <= R1 implies open <= R2 <= R3. Mirror holds on the S side. Nothing is silently
    dropped by that choice.
    """
    n1, n2, n3 = (f"{side}1", f"{side}2", f"{side}3")
    up = side == "R"
    out = []
    for s in sessions:
        L = s["lines"]
        a, b, c = L[n1], L[n2], L[n3]
        if a["pre_broken"] or not a["close_break"]:
            continue
        p1, p2 = a["price"], b["price"]
        far = s["rth_high"] if up else s["rth_low"]
        opp = L["S1"]["price"] if up else L["R1"]["price"]
        reach2 = far >= p2 if up else far <= p2
        row = {
            "day": s["day"],
            "broke_at": a["close_break"],
            "early": a["close_break"] in EARLY,
            "p1": p1, "p2": p2,
            "dist_12_bp": abs(p2 / p1 - 1) * 10_000,
            "reach2": reach2,
            "break2": b["close_break"] is not None,
            "break3": c["close_break"] is not None,
            "gap_min": (mins(b["close_break"]) - mins(a["close_break"]))
                       if b["close_break"] else None,
            "full_reversal": (s["rth_low"] < opp) if up else (s["rth_high"] > opp),
            "r25_bp": round_dist(p1, 25) / p1 * 10_000,
        }
        t = s.get("trade")
        if t and t["line"] == n1:
            row["failed"] = t["exit_reason"] == "failed"
            row["mfe_bp"] = t["mfe_bp"]
            row["move_bp"] = t["move_bp"]
            row["giveback_bp"] = t["mfe_bp"] - t["move_bp"]
            row["mfe_reached_2"] = t["mfe_bp"] >= row["dist_12_bp"]
        out.append(row)
    return out


# --------------------------------------------------------------------------- report

def _rate_line(label, rows, key):
    k = sum(1 for r in rows if r[key])
    p, lo, hi = wilson(k, len(rows))
    return f"  {label:<18}{k:>5}/{len(rows):<5} {p:>6.1%}  [{lo:.1%}, {hi:.1%}]"


def report_side(side: str, rows):
    nm = f"{side}1"
    n2 = f"{side}2"
    early = [r for r in rows if r["early"]]
    late = [r for r in rows if not r["early"]]
    print("\n" + "=" * 78)
    print(f"  {nm} BROKE -- what happened next.  {len(rows)} sessions "
          f"({len(early)} early, {len(late)} late)")
    print("=" * 78)

    for key, title in (("reach2", f"price REACHED {n2}"),
                       ("break2", f"price CLOSED beyond {n2}"),
                       ("break3", f"price CLOSED beyond {side}3"),
                       ("full_reversal", "session traded through the opposite side "
                                         "[DEFECTIVE - see note]")):
        print(f"\n  {title}")
        print(_rate_line(f"early ({nm}<=09:40)", early, key))
        print(_rate_line("late  (>=09:50)", late, key))
        print(_rate_line("all", rows, key))
        pv = two_prop_p(sum(1 for r in early if r[key]), len(early),
                        sum(1 for r in late if r[key]), len(late))
        k1, k2 = sum(1 for r in early if r[key]), sum(1 for r in late if r[key])
        d = (k1 / max(len(early), 1) - k2 / max(len(late), 1)) * 100
        print(f"  {'early - late':<18}{d:>+5.1f} pp                 p={pv:.3f}"
              f"   {'<-- SIGNAL' if pv < 0.05 else 'no evidence'}")

    print(f"\n  NOTE on the [DEFECTIVE] row: it uses the WHOLE session's extreme, which")
    print(f"  includes the hours BEFORE {nm} broke. A late {nm} break happens on a day that")
    print(f"  already swung around, so that row measures 'the day was wide', not 'it")
    print(f"  reversed after the break'. The stored sweep has no post-break extreme, so")
    print(f"  this cannot be fixed here. Do not read the early-vs-late gap on that row.")

    # how far apart the lines are, and whether the move covered it
    print(f"\n  distance {nm} -> {n2}: median {median([r['dist_12_bp'] for r in rows]):.0f} bp"
          f"   (early {median([r['dist_12_bp'] for r in early]):.0f} bp)")
    gaps = [r["gap_min"] for r in early if r["gap_min"] is not None and r["gap_min"] > 0]
    if gaps:
        print(f"  when {n2} did break after an early {nm}: median "
              f"{median(gaps):.0f} min later")

    # ---- THE CONTROL THAT DECIDES IT -------------------------------------------
    # An early break has six hours left to reach R2; a 15:30 break has twenty minutes.
    # So "early breaks reach R2 more often" is what pure time-exposure produces on its
    # own, with no momentum anywhere. The only honest comparison gives both groups the
    # SAME window: did R2 break within 60 minutes of R1 breaking, among breaks with at
    # least 60 minutes of session left (so at or before 14:50)?
    win = 60
    elig = [r for r in rows if mins(r["broke_at"]) <= mins("14:50")]
    for r in elig:
        r["_w"] = r["gap_min"] is not None and 0 <= r["gap_min"] <= win
    e2 = [r for r in elig if r["early"]]
    l2 = [r for r in elig if not r["early"]]
    print(f"\n  TIME-MATCHED: {n2} broke within {win} min of {nm} "
          f"(breaks by 14:50 only)")
    print(_rate_line(f"early ({nm}<=09:40)", e2, "_w"))
    print(_rate_line("late  (09:50-14:50)", l2, "_w"))
    k1, k2 = sum(1 for r in e2 if r["_w"]), sum(1 for r in l2 if r["_w"])
    pv = two_prop_p(k1, len(e2), k2, len(l2))
    d = (k1 / max(len(e2), 1) - k2 / max(len(l2), 1)) * 100
    print(f"  {'early - late':<18}{d:>+5.1f} pp                 p={pv:.3f}"
          f"   {'<-- SURVIVES' if pv < 0.05 else 'dies once time is matched'}")

    tr = [r for r in early if "mfe_bp" in r]
    if tr:
        print(f"\n  of the {len(tr)} early breaks where {nm} was also the session's FIRST break:")
        print(_rate_line("closed back inside", tr, "failed"))
        print(_rate_line(f"best move reached {n2}", tr, "mfe_reached_2"))
        print(f"  {'median MFE':<18}{median([r['mfe_bp'] for r in tr]):>6.1f} bp"
              f"      (MFE = best unrealised move in your favour)")
        print(f"  {'median end move':<18}{median([r['move_bp'] for r in tr]):>6.1f} bp")
        print(f"  {'median giveback':<18}{median([r['giveback_bp'] for r in tr]):>6.1f} bp"
              f"      (how much of the best move was handed back)")


def report_round(side: str, rows, sessions):
    nm = f"{side}1"
    print("\n" + "-" * 78)
    print(f"  ROUND NUMBERS -- is {nm} near a multiple of M?  {len(rows)} sessions")
    print("-" * 78)
    print("  If levels don't care about round numbers, the distance to the nearest")
    print("  multiple of M is uniform on [0, M/2], so its mean is M/4. Below M/4 = clustering.")
    print(f"\n  {'M':>4} {'mean dist $':>12} {'expected M/4':>13} {'ratio':>7}  verdict")
    for m in ROUND_MULTIPLES:
        ds = [round_dist(r["p1"], m) for r in rows]
        obs, exp = sum(ds) / len(ds), m / 4.0
        ratio = obs / exp
        v = "clusters" if ratio < 0.90 else ("avoids" if ratio > 1.10 else "uniform, no effect")
        print(f"  {m:>4} {obs:>12.3f} {exp:>13.3f} {ratio:>7.2f}  {v}")

    print("\n  same test split by price era (QQQ ran $110 -> $740):")
    for lab, sub in (("under $300", [r for r in rows if r["p1"] < ERA_SPLIT]),
                     ("over  $300", [r for r in rows if r["p1"] >= ERA_SPLIT])):
        if not sub:
            continue
        bits = []
        for m in ROUND_MULTIPLES:
            ds = [round_dist(r["p1"], m) for r in sub]
            bits.append(f"M={m}: {(sum(ds) / len(ds)) / (m / 4.0):.2f}")
        print(f"    {lab}  n={len(sub):<5} " + "  ".join(bits))

    # the one pre-specified subgroup: split early breaks at the median distance to a x25
    early = [r for r in rows if r["early"]]
    if len(early) >= 20:
        cut = median([r["r25_bp"] for r in early])
        near = [r for r in early if r["r25_bp"] <= cut]
        far = [r for r in early if r["r25_bp"] > cut]
        print(f"\n  PRE-SPECIFIED SPLIT: early breaks, {nm} near vs far from a multiple")
        print(f"  of 25 (median cut = {cut:.0f} bp away)")
        for key, title in (("reach2", f"reached {side}2"), ("break2", f"broke {side}2")):
            k1 = sum(1 for r in near if r[key])
            k2 = sum(1 for r in far if r[key])
            pv = two_prop_p(k1, len(near), k2, len(far))
            print(f"    {title:<14} near {k1}/{len(near)} {k1 / len(near):>6.1%}   "
                  f"far {k2}/{len(far)} {k2 / len(far):>6.1%}   p={pv:.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    path = Path(a.json) if a.json else LAB / f"six_lines_{a.symbol}.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found.\nRun the six-line sweep first:\n"
            f"  python -m trade_analysis.live_lab.six_lines --symbol {a.symbol}")
    sessions = json.loads(path.read_text(encoding="utf-8"))["sessions"]

    print(f"AFTER THE BREAK  {a.symbol}  {sessions[0]['day']} -> {sessions[-1]['day']}"
          f"  ({len(sessions)} sessions)")
    print("executes research/after_break_preregistration.md")
    print('"first 15 minutes" = the first two 10-minute bars, 09:30 and 09:40')

    for side in ("R", "S"):
        rows = side_rows(sessions, side)
        if not rows:
            print(f"\nno {side}1 breaks recorded")
            continue
        report_side(side, rows)
        report_round(side, rows, sessions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
