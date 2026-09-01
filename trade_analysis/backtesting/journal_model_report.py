"""Q3b reporting, reading the cached model scores.

journal_model_wf.py writes every scored minute to model_wf.pkl before it reports, so the
expensive part (scoring 721,887 out-of-sample QQQ minutes with the 45-feature classifier)
does not need repeating. This reads that file and produces the pre-registered tables.
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from trade_analysis.backtesting.journal_zone_wf import boot_ci, holm  # noqa: E402

SCRATCH = (r"C:\Users\CHAITA~1\AppData\Local\Temp\claude"
           r"\C--Users-chaitanyakharche-Documents-stock"
           r"\3ad3e562-0324-4d02-bf8b-208ff71f108d\scratchpad")
HORIZONS = (5, 10, 15, 25)
COOLDOWN = 25


def report(tag, sel, n_universe):
    days = [r["day"] for r in sel]
    print("\n  %s   n=%d (%.2f%% of scored minutes, %d days)"
          % (tag, len(sel), 100 * len(sel) / max(n_universe, 1), len(set(days))))
    if len(sel) < 30:
        print("    too few to test")
        return
    ps, res = [], []
    for h in HORIZONS:
        v = np.array([r["fwd_%d" % h] for r in sel])
        lo, hi, p = boot_ci(v, days, nboot=4000)
        res.append((h, v.mean(), 100 * (v > 0).mean(), lo, hi, p))
        ps.append(p)
    hp = holm(ps)
    print("    %-8s %10s %8s %22s %9s %9s"
          % ("horizon", "mean bp", "win%", "95% CI (bp)", "p raw", "p Holm"))
    for (h, m, w, lo, hi, p), ph in zip(res, hp):
        print("    %-8s %+10.3f %7.1f%%   [%+8.3f, %+8.3f] %9.4f %9.4f"
              % ("%d min" % h, m, w, lo, hi, p, ph))


def main():
    D = pickle.load(open(os.path.join(SCRATCH, "model_wf.pkl"), "rb"))
    for tag, bag in (("OUT-OF-SAMPLE  2016-01-04 .. 2024-09-10", D["oos"]),
                     ("IN-PERIOD      2024-09-11 .. 2026-08-27", D["ins"])):
        print("\n" + "=" * 92)
        print("Q3b -- " + tag)
        print("=" * 92)
        if not bag:
            print("  no rows (his journal window is not in this bar cache)")
            continue
        ps = np.array([r["p"] for r in bag])
        print("  scored minutes %d over %d sessions"
              % (len(bag), len({r["day"] for r in bag})))
        print("  P(he would enter here): p50 %.3f  p90 %.3f  p99 %.3f  max %.3f"
              % (np.percentile(ps, 50), np.percentile(ps, 90),
                 np.percentile(ps, 99), ps.max()))
        report("ALL scored minutes (baseline: what a QQQ minute does)", bag, len(bag))
        for name, q in (("TOP DECILE  P>=p90", 90), ("TOP CENTILE P>=p99", 99)):
            thr = np.percentile(ps, q)
            cand = sorted((r for r in bag if r["p"] >= thr),
                          key=lambda r: (r["day"], r["k"]))
            sel, last = [], {}
            for r in cand:
                if r["k"] - last.get(r["day"], -10 ** 9) < COOLDOWN:
                    continue
                last[r["day"]] = r["k"]
                sel.append(r)
            report(name + "  (25-min cooldown, non-overlapping)", sel, len(bag))
            if sel:
                yb = defaultdict(list)
                for r in sel:
                    yb[r["day"][:4]].append(r["fwd_15"])
                print("    by year (15m bp): " + "  ".join(
                    "%s %+.2f" % (y, np.mean(v)) for y, v in sorted(yb.items())))
                lg = [r["fwd_15"] for r in sel if r["dir"] > 0]
                sh = [r["fwd_15"] for r in sel if r["dir"] < 0]
                print("    long n=%d mean %+.3f   short n=%d mean %+.3f"
                      % (len(lg), np.mean(lg) if lg else 0,
                         len(sh), np.mean(sh) if sh else 0))
    print("\n  CONVERSION FLOOR: an ATM 0DTE needs about +5 bp of underlying move to clear")
    print("  spread and theta. Compare every CI upper bound above against +5.000.")


if __name__ == "__main__":
    main()
