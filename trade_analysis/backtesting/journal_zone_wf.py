"""Q3 -- blind walk-forward on the ZONE the classifier recovered from his journal.

Q1 shows his entry minutes are separable from same-session placebos at OOF AUC 0.85, and the
separation lives almost entirely in momentum-alignment features. That means the rule is
recoverable. It says nothing about whether the rule pays.

This file answers that. It takes the interquartile zone HIS OWN 418 entries occupy, then
scans every QQQ minute 2016-2026 and every SPY minute 2022-2026 with no knowledge of his
trades whatsoever, and measures the forward underlying move.

THE EXECUTION RULE THAT MATTERS
    Bar k is stamped T_k and covers [T_k, T_k + 60). Its close is knowable only at T_{k+1}.
    A signal computed from bar k is therefore actionable no earlier than T_{k+1}, so entry
    is at bar k+1 OPEN and exit at bar k+1+h OPEN. Both legs lagged identically.
    Pricing at bar k instead moved IntradayMomentumBoundary from p=0.0003 to p=0.79.
    This is not optional.

Pre-registered as amendment A2 of research/journal_reverse_engineering_preregistration.md.
Falsifier, fixed before this ran: if Holm-adjusted p > 0.05 at every horizon, OR the 95% CI
upper bound is below +5 bp at every horizon, the recovered rule has no tradeable edge.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from trade_analysis.backtesting.journal_features import Panel, bars_5m  # noqa: E402

CACHE = os.path.join(ROOT, "live_lab_data", "bars_cache")
SCRATCH = (r"C:\Users\CHAITA~1\AppData\Local\Temp\claude"
           r"\C--Users-chaitanyakharche-Documents-stock"
           r"\3ad3e562-0324-4d02-bf8b-208ff71f108d\scratchpad")

# ---- THE ZONE: 25th percentile of his own 418 entries on the four features that
# ---- discriminate. Fixed before this file was run. Captures 55.5% of his entries and
# ---- fires on 13.6% of random same-session minutes (4.1x lift).
ZONE = {
    "px_vs_ema9_al": 0.71,      # ATR units above the 5-min 9-EMA, in his direction
    "di_diff_1m_al": 9.39,      # 1-minute (DI+ - DI-), aligned
    "trail15_sigma_al": 0.39,   # 15-min move in 5-min sigma units, aligned
    "ret_15m_al": 6.65,         # 15-min move in bp, aligned
}
HORIZONS = (5, 10, 15, 25)
COOLDOWN = 25                   # no overlapping trades; equals the longest horizon
PREFIX_SESSIONS = 2
BOOT = 10000
SEED = 20260829


def sessions(symbol):
    out = []
    for f in sorted(os.listdir(CACHE)):
        if not f.startswith(symbol + "_") or not f.endswith(".pkl"):
            continue
        day = f[len(symbol) + 1:-4]
        try:
            b = pickle.load(open(os.path.join(CACHE, f), "rb"))
        except Exception:
            continue
        if isinstance(b, list) and len(b) >= 300:
            out.append((day, b))
    return out


def scan_session(sess, prior):
    """Every zone signal in one session, with honest one-minute-lagged fills."""
    pre1 = [b for s in prior for b in s]
    pre5 = [b for s in prior for b in bars_5m(s)]
    p1 = Panel(pre1 + sess)
    b5 = bars_5m(sess)
    p5 = Panel(pre5 + b5)
    n1, n5 = len(pre1), len(pre5)

    # 1-min index -> index of the last CLOSE-STAMPED 5-min bucket at or before it.
    # bucket j closes at sess index 5j+5, so it is usable from 1-min index 5j+5 onward.
    out = []
    last_fire = -10 ** 9
    hi = len(sess) - max(HORIZONS) - 2
    for k in range(30, hi):
        j5 = k // 5 - 1
        if j5 < 3:
            continue
        J5 = n5 + j5
        K = n1 + k
        a5 = p5.atr[J5]
        if not a5 or a5 <= 0 or p5.ema9[J5] is None:
            continue
        if p1.dip[K] is None or p1.din[K] is None:
            continue
        px = sess[k]["close"]

        # direction is set by the trailing move; the zone is one-sided by construction
        raw15 = px / b5[j5 - 2]["close"] - 1.0 if j5 >= 2 else 0.0
        d = 1.0 if raw15 > 0 else -1.0
        ret15 = d * raw15 * 1e4
        if ret15 < ZONE["ret_15m_al"]:
            continue
        if d * (px - p5.ema9[J5]) / a5 < ZONE["px_vs_ema9_al"]:
            continue
        if d * (p1.dip[K] - p1.din[K]) < ZONE["di_diff_1m_al"]:
            continue
        r5 = [b5[i]["close"] / b5[i - 1]["close"] - 1.0 for i in range(1, j5 + 1)]
        sd5 = float(np.std(r5, ddof=1)) if len(r5) > 2 else 0.0
        if sd5 <= 0:
            continue
        if (d * raw15) / (sd5 * math.sqrt(3)) < ZONE["trail15_sigma_al"]:
            continue
        if k - last_fire < COOLDOWN:
            continue
        last_fire = k

        entry = sess[k + 1]["open"]          # one full minute after the signal bar stamp
        if entry <= 0:
            continue
        rec = {"k": k, "dir": d, "entry": entry, "tod": k}
        for h in HORIZONS:
            rec["fwd_%d" % h] = d * (sess[k + 1 + h]["open"] / entry - 1.0) * 1e4
        out.append(rec)
    return out


def boot_ci(vals, days, nboot=BOOT, seed=SEED):
    """Day-clustered bootstrap: resample DATES, not trades.

    Vectorised. Resampling whole day clusters and taking the pooled mean is exactly
    sum(selected day sums) / sum(selected day counts), so the per-day sums and counts are
    all that is needed. Identical estimator to the naive pooling loop, ~1000x faster, which
    matters once a selection runs to hundreds of thousands of rows.
    """
    vals = np.asarray(vals, float)
    _, inv = np.unique(np.asarray(days), return_inverse=True)
    D = inv.max() + 1
    S = np.bincount(inv, weights=vals, minlength=D)
    N = np.bincount(inv, minlength=D).astype(float)
    rng = np.random.default_rng(seed)
    ms = np.empty(nboot)
    step = max(1, 20_000_000 // max(D, 1))
    done = 0
    while done < nboot:
        b = min(step, nboot - done)
        idx = rng.integers(0, D, size=(b, D))
        ms[done:done + b] = S[idx].sum(1) / N[idx].sum(1)
        done += b
    ms.sort()
    lo, hi = ms[int(0.025 * nboot)], ms[int(0.975 * nboot)]
    p = max(2 * min((ms <= 0).mean(), (ms >= 0).mean()), 1.0 / nboot)
    return float(lo), float(hi), float(p)


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [0.0] * m
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pvals[i])
        out[i] = min(run, 1.0)
    return out


def main():
    allrows = []
    for sym in ("QQQ", "SPY"):
        S = sessions(sym)
        if not S:
            continue
        print("%s: %d sessions  %s -> %s" % (sym, len(S), S[0][0], S[-1][0]), flush=True)
        for i in range(PREFIX_SESSIONS, len(S)):
            day, sess = S[i]
            prior = [S[i - j][1] for j in range(PREFIX_SESSIONS, 0, -1)]
            try:
                sigs = scan_session(sess, prior)
            except Exception as ex:
                print("  skip %s %s: %s" % (sym, day, ex))
                continue
            for s in sigs:
                s.update({"symbol": sym, "day": day})
                allrows.append(s)
            if (i + 1) % 400 == 0:
                print("  %s %d/%d  signals=%d" % (sym, i + 1, len(S), len(allrows)),
                      flush=True)
    pickle.dump(allrows, open(os.path.join(SCRATCH, "zone_signals.pkl"), "wb"))

    days = [r["symbol"] + "_" + r["day"] for r in allrows]
    nd = len(set(days))
    print("\n" + "=" * 92)
    print("Q3 -- ZONE WALK-FORWARD, one-minute-lagged fills on both legs")
    print("=" * 92)
    print("  signals %d over %d symbol-days   (%.2f per active day)"
          % (len(allrows), nd, len(allrows) / max(nd, 1)))
    longs = sum(1 for r in allrows if r["dir"] > 0)
    print("  long %d (%.1f%%)   short %d (%.1f%%)"
          % (longs, 100 * longs / len(allrows), len(allrows) - longs,
             100 * (1 - longs / len(allrows))))

    ps, res = [], []
    for h in HORIZONS:
        v = np.array([r["fwd_%d" % h] for r in allrows])
        lo, hi, p = boot_ci(v, days)
        res.append((h, v.mean(), np.median(v), 100 * (v > 0).mean(), lo, hi, p))
        ps.append(p)
    hp = holm(ps)
    print("\n  %-8s %10s %9s %8s %20s %9s %9s"
          % ("horizon", "mean bp", "med bp", "win%", "95% CI (bp)", "p raw", "p Holm"))
    for (h, m, md, w, lo, hi, p), ph in zip(res, hp):
        print("  %-8s %+10.3f %+9.3f %7.1f%%   [%+7.3f, %+7.3f] %9.4f %9.4f"
              % ("%d min" % h, m, md, w, lo, hi, p, ph))

    print("\n  BY YEAR (15-minute horizon, bp)")
    yb = defaultdict(list)
    for r in allrows:
        yb[r["day"][:4]].append(r["fwd_15"])
    for y in sorted(yb):
        v = np.array(yb[y])
        print("    %s  n=%5d  mean %+7.3f  win %5.1f%%" % (y, len(v), v.mean(),
                                                           100 * (v > 0).mean()))

    print("\n  BY DIRECTION (15-minute horizon, bp)")
    for nm, dd in (("long", 1.0), ("short", -1.0)):
        v = np.array([r["fwd_15"] for r in allrows if r["dir"] == dd])
        if len(v):
            lo, hi, p = boot_ci(v, [d for d, r in zip(days, allrows) if r["dir"] == dd])
            print("    %-6s n=%5d  mean %+7.3f  [%+7.3f, %+7.3f]  p %.4f"
                  % (nm, len(v), v.mean(), lo, hi, p))

    print("\n  TAIL DEPENDENCE (15-minute horizon)")
    v = np.sort(np.array([r["fwd_15"] for r in allrows]))[::-1]
    tot = v.sum()
    n1 = max(1, len(v) // 100)
    if tot != 0:
        print("    top 1%% (%d trades) = %.1f%% of total" % (n1, 100 * v[:n1].sum() / tot))
        print("    mean without top 1%%: %+.3f bp" % v[n1:].mean())

    print("\n  CONVERSION FLOOR: an ATM 0DTE needs roughly +5 bp of underlying move to")
    print("  clear spread and theta. Compare the CI upper bounds above against +5.000.")


if __name__ == "__main__":
    main()
