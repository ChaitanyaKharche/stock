"""METHOD 4 -- Lo, Mamaysky & Wang (2000) kernel pattern detection, intraday.

"Foundations of Technical Analysis" formalises chart-pattern recognition by smoothing price
with a Nadaraya-Watson kernel regression, defining local extrema mathematically as sign
changes of the smoothed first difference, and then testing whether the forward-return
distribution conditional on a pattern differs from the unconditional one.

Adapted to 5-minute bars, within-session windows only (a window spanning the overnight gap
would manufacture extrema out of the gap itself).

Two uses, both pre-registered:
  (a) DESCRIPTIVE  which patterns were live at his 418 entry minutes vs the base rate
  (b) PREDICTIVE   conditional vs unconditional forward returns on all QQQ/SPY 5-min bars

Family size m = 10 (the ten canonical patterns), Holm. A pattern counts only if its
Kolmogorov-Smirnov test clears Holm at 0.05.

Honest timing: a pattern whose window ends at bar t is detectable only once bar t has
closed, so the forward return is measured from bar t+1 OPEN.
"""
from __future__ import annotations

import datetime as dt
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from trade_analysis.backtesting.journal_features import bars_5m  # noqa: E402

CACHE = os.path.join(ROOT, "live_lab_data", "bars_cache")
SCRATCH = (r"C:\Users\CHAITA~1\AppData\Local\Temp\claude"
           r"\C--Users-chaitanyakharche-Documents-stock"
           r"\3ad3e562-0324-4d02-bf8b-208ff71f108d\scratchpad")

WINDOW = 38          # LMW use 38 observations
HORIZONS = (3, 5)    # in 5-minute bars: 15 and 25 minutes
SEED = 20260829
PATTERNS = ("HS", "IHS", "BTOP", "BBOT", "TTOP", "TBOT", "RTOP", "RBOT", "DTOP", "DBOT")


# ------------------------------------------------------------------- kernel smoothing

def weight_matrix(h, n=WINDOW):
    """Nadaraya-Watson weights for evenly spaced x = 0..n-1. Constant across windows, so the
    whole smoothing step is one matrix multiply."""
    x = np.arange(n, dtype=float)
    d = (x[:, None] - x[None, :]) / h
    K = np.exp(-0.5 * d * d)
    return K / K.sum(1, keepdims=True)


def cv_bandwidth(windows, grid=np.arange(0.6, 8.01, 0.2)):
    """Leave-one-out cross-validated bandwidth, then LMW's 0.3 multiplier (they report that
    the CV optimum oversmooths for pattern detection)."""
    best, bh = np.inf, grid[0]
    for h in grid:
        W = weight_matrix(h)
        d = np.diag(W).copy()
        if np.any(d >= 0.999):
            continue
        sm = windows @ W.T
        loo = (sm - windows * d) / (1.0 - d)      # leave-one-out identity for linear smoothers
        err = float(np.mean((windows - loo) ** 2))
        if err < best:
            best, bh = err, h
    return bh, 0.3 * bh


# ------------------------------------------------------------------------- extrema

def extrema(m):
    """Indices and kinds of local extrema of a smoothed series. +1 max, -1 min."""
    d = np.sign(np.diff(m))
    idx, kind = [], []
    for i in range(1, len(d)):
        if d[i] != d[i - 1] and d[i] != 0 and d[i - 1] != 0:
            idx.append(i)
            kind.append(1 if d[i - 1] > 0 else -1)
    return np.asarray(idx, int), np.asarray(kind, int)


def classify(E, T, kinds):
    """LMW pattern definitions on the extrema of one window. Returns a set of names."""
    out = set()
    n = len(E)
    if n < 5:
        return out
    e1, e2, e3, e4, e5 = E[-5:]
    t1, t2, t3, t4, t5 = T[-5:]
    k1 = kinds[-5]

    def close(a, b, tol):
        avg = (a + b) / 2.0
        return avg != 0 and abs(a - b) <= tol * abs(avg)

    if k1 == 1:                                     # E1 is a maximum
        if e3 > e1 and e3 > e5 and close(e1, e5, 0.015) and close(e2, e4, 0.015):
            out.add("HS")
        if e1 < e3 < e5 and e2 > e4:
            out.add("BTOP")
        if e1 > e3 > e5 and e2 < e4:
            out.add("TTOP")
        tops, bots = [e1, e3, e5], [e2, e4]
        ta, ba = np.mean(tops), np.mean(bots)
        if (ta != 0 and ba != 0
                and all(abs(v - ta) <= 0.0075 * abs(ta) for v in tops)
                and all(abs(v - ba) <= 0.0075 * abs(ba) for v in bots)
                and min(tops) > max(bots)):
            out.add("RTOP")
    else:                                           # E1 is a minimum
        if e3 < e1 and e3 < e5 and close(e1, e5, 0.015) and close(e2, e4, 0.015):
            out.add("IHS")
        if e1 > e3 > e5 and e2 < e4:
            out.add("BBOT")
        if e1 < e3 < e5 and e2 > e4:
            out.add("TBOT")
        bots, tops = [e1, e3, e5], [e2, e4]
        ba, ta = np.mean(bots), np.mean(tops)
        if (ta != 0 and ba != 0
                and all(abs(v - ba) <= 0.0075 * abs(ba) for v in bots)
                and all(abs(v - ta) <= 0.0075 * abs(ta) for v in tops)
                and min(tops) > max(bots)):
            out.add("RBOT")

    # DTOP / DBOT use the whole window, not just the last five extrema
    mx = [(T[i], E[i]) for i in range(n) if kinds[i] == 1]
    mn = [(T[i], E[i]) for i in range(n) if kinds[i] == -1]
    if len(mx) >= 2:
        ta, ea = mx[0]
        for tb, eb in mx[1:]:
            if tb - ta > 11 and abs(ea - eb) <= 0.015 * abs((ea + eb) / 2.0):
                out.add("DTOP")
                break
    if len(mn) >= 2:
        ta, ea = mn[0]
        for tb, eb in mn[1:]:
            if tb - ta > 11 and abs(ea - eb) <= 0.015 * abs((ea + eb) / 2.0):
                out.add("DBOT")
                break
    return out


# ---------------------------------------------------------------------------- data

def load_5m(symbol):
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
            out.append((day, bars_5m(b)))
    return out


def holm(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    out = [0.0] * m
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        out[i] = min(run, 1.0)
    return out


def boot_p(vals, days, nboot=4000):
    by = defaultdict(list)
    for v, d in zip(vals, days):
        by[d].append(v)
    keys = list(by)
    rng = np.random.default_rng(SEED)
    ms = np.empty(nboot)
    for i in range(nboot):
        pool = []
        for j in rng.integers(0, len(keys), len(keys)):
            pool.extend(by[keys[j]])
        ms[i] = np.mean(pool)
    ms.sort()
    return (float(ms[int(0.025 * nboot)]), float(ms[int(0.975 * nboot)]),
            float(max(2 * min((ms <= 0).mean(), (ms >= 0).mean()), 1.0 / nboot)))


# ---------------------------------------------------------------------------- main

def main():
    data = []
    for sym in ("QQQ", "SPY"):
        s = load_5m(sym)
        if s:
            print("%s: %d sessions" % (sym, len(s)), flush=True)
            data.append((sym, s))

    # bandwidth from a sample of windows, log price so the tolerance percentages behave
    samp = []
    for sym, S in data:
        for day, b5 in S[::37]:
            c = np.log([x["close"] for x in b5])
            for i in range(0, len(c) - WINDOW, 9):
                samp.append(c[i:i + WINDOW])
    samp = np.asarray(samp)
    samp = samp - samp.mean(1, keepdims=True)
    hcv, h = cv_bandwidth(samp)
    print("bandwidth: h_CV = %.2f   LMW h = 0.3 x h_CV = %.3f   (%d sample windows)"
          % (hcv, h, len(samp)), flush=True)
    W = weight_matrix(h)

    hits = {p: {"ret": defaultdict(list), "day": defaultdict(list)} for p in PATTERNS}
    uncond = {hz: [] for hz in HORIZONS}
    uncond_day = {hz: [] for hz in HORIZONS}
    nwin = 0
    for sym, S in data:
        for si, (day, b5) in enumerate(S):
            c = np.asarray([x["close"] for x in b5], float)
            o = np.asarray([x["open"] for x in b5], float)
            lc = np.log(c)
            r = np.diff(lc)
            if len(c) < WINDOW + max(HORIZONS) + 2:
                continue
            for i in range(0, len(c) - WINDOW - max(HORIZONS) - 1):
                t = i + WINDOW - 1                      # last bar of the window
                w = lc[i:i + WINDOW]
                sd = float(np.std(r[max(0, t - 20):t + 1]))
                if sd <= 0:
                    continue
                base = o[t + 1]                          # honest: act after bar t closes
                if base <= 0:
                    continue
                fwd = {hz: (o[t + 1 + hz] / base - 1.0) / sd for hz in HORIZONS
                       if t + 1 + hz < len(o)}
                if len(fwd) < len(HORIZONS):
                    continue
                nwin += 1
                for hz in HORIZONS:
                    uncond[hz].append(fwd[hz])
                    uncond_day[hz].append(sym + day)
                m = W @ (w - w.mean())
                E_idx, E_kind = extrema(m)
                if len(E_idx) < 2:
                    continue
                pats = classify(m[E_idx], E_idx, E_kind)
                for p in pats:
                    for hz in HORIZONS:
                        hits[p]["ret"][hz].append(fwd[hz])
                        hits[p]["day"][hz].append(sym + day)
        print("  %s done, windows so far %d" % (sym, nwin), flush=True)

    print("\n" + "=" * 96)
    print("METHOD 4 -- LMW KERNEL PATTERNS, PREDICTIVE TEST")
    print("=" * 96)
    print("  windows scanned %d   unconditional mean (25m) %+.4f sigma"
          % (nwin, np.mean(uncond[5])))
    for hz in HORIZONS:
        print("\n  HORIZON %d bars (%d minutes)" % (hz, hz * 5))
        print("  %-6s %8s %8s %12s %22s %10s %10s"
              % ("patt", "n", "freq%", "mean sigma", "95% CI", "KS p", "KS Holm"))
        ksp, rows = [], []
        for p in PATTERNS:
            v = np.asarray(hits[p]["ret"][hz])
            if len(v) < 20:
                rows.append((p, len(v), np.nan, np.nan, np.nan, np.nan, 1.0))
                ksp.append(1.0)
                continue
            ks = stats.ks_2samp(v, np.asarray(uncond[hz]))
            lo, hi, _ = boot_p(v, hits[p]["day"][hz])
            rows.append((p, len(v), 100 * len(v) / nwin, float(v.mean()), lo, hi,
                         float(ks.pvalue)))
            ksp.append(float(ks.pvalue))
        hp = holm(ksp)
        for (p, n, f, mn, lo, hi, kp), ph in zip(rows, hp):
            if n < 20:
                print("  %-6s %8d %8s %12s %22s %10s %10s"
                      % (p, n, "-", "too few", "-", "-", "-"))
            else:
                print("  %-6s %8d %8.2f %+12.4f   [%+8.4f,%+8.4f] %10.4f %10.4f"
                      % (p, n, f, mn, lo, hi, kp, ph))
        surv = [p for (p, *_), ph in zip(rows, hp) if ph < 0.05]
        print("  patterns clearing Holm: %s" % (", ".join(surv) if surv else "NONE"))

    pickle.dump({"hits": {p: dict(hits[p]["ret"]) for p in PATTERNS},
                 "uncond": uncond, "nwin": nwin, "h": h},
                open(os.path.join(SCRATCH, "kernel_patterns.pkl"), "wb"))
    print("\nsaved -> kernel_patterns.pkl")


if __name__ == "__main__":
    main()
