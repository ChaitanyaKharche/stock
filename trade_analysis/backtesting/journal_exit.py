"""Q4 -- reverse-engineering the EXIT, on his actual contracts at real NBBO.

Every other study in this project reverse-engineers the entry. Test B says that is the wrong
end: a mechanical +25% target loses to his discretion by 8.97pp at Holm p = 0.027, the only
comparison here that has ever cleared a correction.

This reconstructs the minute-by-minute quote path of each contract he actually traded and
asks two questions, pre-registered as amendment A3:

  Q4a  ECONOMIC        does his exit beat 14 mechanical alternatives on the same trades?
  Q4b  MECHANISABLE    is the minute he exits predictable from observables?

Entry at the ASK, exit at the BID, $0.0404/contract/side, paired per trade, day-clustered
bootstrap resampling dates. Where a target or stop never triggers, the counterfactual falls
through to HIS exit time, so the comparison isolates the rule and not the terminal.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import pickle
import sys
import urllib.request
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

SCRATCH = (r"C:\Users\CHAITA~1\AppData\Local\Temp\claude"
           r"\C--Users-chaitanyakharche-Documents-stock"
           r"\3ad3e562-0324-4d02-bf8b-208ff71f108d\scratchpad")
QCACHE = os.path.join(SCRATCH, "optquotes")
TRADES = os.path.join(ROOT, "research", "round_trips.csv")
FEE = 0.0404
BOOT = 10000
SEED = 20260829
os.makedirs(QCACHE, exist_ok=True)

TARGETS = (0.25, 0.50, 0.75, 1.00)
STOPS = (0.25, 0.50)
TIMES = (5, 10, 15, 25, 45)
TRAILS = (0.20, 0.30, 0.50)


def quotes(sym, expiry, day, right, strike):
    """1-minute NBBO for one contract on one day, keyed HH:MM -> (bid, ask)."""
    key = "%s_%s_%s_%s_%s.pkl" % (sym, expiry, day, right, strike)
    p = os.path.join(QCACHE, key)
    if os.path.exists(p):
        try:
            return pickle.load(open(p, "rb"))
        except Exception:
            pass
    url = ("http://127.0.0.1:25503/v3/option/history/quote"
           "?symbol=%s&expiration=%s&date=%s&right=%s&strike=%s&interval=1m"
           % (sym, expiry, day, right, strike))
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            body = r.read().decode()
    except Exception:
        return None
    out = {}
    for row in csv.DictReader(body.splitlines()):
        try:
            ts = row["timestamp"].strip('"')
            b, a = float(row["bid"]), float(row["ask"])
        except (KeyError, ValueError):
            continue
        if b > 0 and a >= b:
            out[ts[11:16]] = (b, a)
    pickle.dump(out, open(p, "wb"))
    return out


def path_for(r):
    """Ordered (minute, bid, ask) from his entry minute to his exit minute inclusive."""
    ent = dt.datetime.fromisoformat(r["entry_ts"]).replace(tzinfo=None)
    ex = dt.datetime.fromisoformat(r["exit_ts"]).replace(tzinfo=None)
    if ent.date() != ex.date():
        return None, "multi_day"
    q = quotes(r["symbol"], r["expiry"], r["date"],
               "C" if r["right"] == "call" else "P", r["strike"])
    if not q:
        return None, "no_quotes"
    out = []
    t = ent.replace(second=0, microsecond=0)
    while t <= ex:
        k = t.strftime("%H:%M")
        if k in q:
            out.append((int((t - ent.replace(second=0, microsecond=0)).total_seconds() // 60),
                        q[k][0], q[k][1]))
        t += dt.timedelta(minutes=1)
    if len(out) < 2 or out[0][0] != 0:
        return None, "path_too_short"
    return out, None


def pnl(entry_ask, exit_bid, qty):
    return qty * 100.0 * (exit_bid - entry_ask) - 2 * FEE * qty


def counterfactuals(path, qty):
    """His exit and all 14 alternatives, in dollars, on one trade."""
    ea = path[0][2]
    if ea <= 0:
        return None
    his_bid = path[-1][1]
    out = {"his": pnl(ea, his_bid, qty)}
    peak = path[0][1]
    hit_t = {t: None for t in TARGETS}
    hit_s = {s: None for s in STOPS}
    hit_tr = {tr: None for tr in TRAILS}
    for m, b, a in path[1:]:
        for t in TARGETS:
            if hit_t[t] is None and b >= ea * (1 + t):
                hit_t[t] = b
        for s in STOPS:
            if hit_s[s] is None and b <= ea * (1 - s):
                hit_s[s] = b
        for tr in TRAILS:
            if hit_tr[tr] is None and peak > ea and b <= peak - tr * (peak - ea):
                hit_tr[tr] = b
        peak = max(peak, b)
    for t in TARGETS:
        out["target_%d" % int(100 * t)] = pnl(ea, hit_t[t] if hit_t[t] is not None
                                              else his_bid, qty)
    for s in STOPS:
        out["stop_%d" % int(100 * s)] = pnl(ea, hit_s[s] if hit_s[s] is not None
                                            else his_bid, qty)
    for tr in TRAILS:
        out["trail_%d" % int(100 * tr)] = pnl(ea, hit_tr[tr] if hit_tr[tr] is not None
                                              else his_bid, qty)
    for tm in TIMES:
        cand = [b for m, b, a in path if m <= tm]
        out["time_%d" % tm] = pnl(ea, cand[-1] if cand else his_bid, qty)
    mx = max(b for m, b, a in path)
    mn = min(b for m, b, a in path)
    out["_mfe"] = 100.0 * (mx - ea) * qty
    out["_mae"] = 100.0 * (mn - ea) * qty
    out["_capture"] = ((his_bid - ea) / (mx - ea)) if mx > ea else np.nan
    out["_entry_ask"] = ea
    out["_hold"] = path[-1][0]
    out["_peak_min"] = [m for m, b, a in path if b == mx][0]
    return out


def boot_paired(diff, days, nboot=BOOT, seed=SEED):
    by = defaultdict(list)
    for v, d in zip(diff, days):
        by[d].append(v)
    keys = list(by)
    rng = np.random.default_rng(seed)
    ms = np.empty(nboot)
    for i in range(nboot):
        pool = []
        for j in rng.integers(0, len(keys), len(keys)):
            pool.extend(by[keys[j]])
        ms[i] = np.mean(pool)
    ms.sort()
    return (float(ms[int(.025 * nboot)]), float(ms[int(.975 * nboot)]),
            float(max(2 * min((ms <= 0).mean(), (ms >= 0).mean()), 1.0 / nboot)))


def holm(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    out = [0.0] * m
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        out[i] = min(run, 1.0)
    return out


# ------------------------------------------------------------------------------ Q4b

def hazard(rows):
    """Is the minute he exits predictable? Per-minute panel, within-trade permutation null."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    X, y, trade, day = [], [], [], []
    for i, (r, path, _) in enumerate(rows):
        ea = path[0][2]
        peak, trough = path[0][1], path[0][1]
        n = len(path)
        for j, (m, b, a) in enumerate(path):
            if j == 0:
                continue
            peak = max(peak, b)
            trough = min(trough, b)
            X.append([m, (b / ea - 1.0), (peak / ea - 1.0), (trough / ea - 1.0),
                      (b - peak) / ea, (b - trough) / ea, (a - b) / max(a, 1e-9),
                      float(j) / n if n else 0.0, m / 60.0])
            y.append(1 if j == n - 1 else 0)
            trade.append(i)
            day.append(r["date"])
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    trade = np.asarray(trade)
    day = np.asarray(day)
    # 'frac_of_hold' leaks the answer by construction -- drop it, keep the rest
    X = X[:, [0, 1, 2, 3, 4, 5, 6, 8]]
    names = ["minutes_held", "unreal_ret", "peak_ret", "trough_ret",
             "drawdown_from_peak", "runup_from_trough", "spread_pct", "hours_held"]

    def run(yy):
        oof = np.full(len(yy), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(X, yy, day):
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                               max_iter=300, early_stopping=False,
                                               random_state=SEED)
            m.fit(X[tr], yy[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        return roc_auc_score(yy, oof)

    real = run(y)
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(200):
        yp = np.zeros_like(y)
        for t in np.unique(trade):
            idx = np.flatnonzero(trade == t)
            yp[idx[rng.integers(0, len(idx))]] = 1
        null.append(run(yp))
    null = np.asarray(null)
    p = (1 + (null >= real).sum()) / (1 + len(null))
    print("\n  Q4b HAZARD -- can the exit minute be predicted?")
    print("    panel rows %d over %d trades" % (len(y), len(np.unique(trade))))
    print("    OOF AUC %.4f   within-trade permutation null mean %.4f sd %.4f  p=%.4f"
          % (real, null.mean(), null.std(), p))
    print("    verdict: %s" % ("MECHANISABLE from these observables"
                               if (real > 0.55 and p < 0.05) else
                               "NOT recoverable from these observables"))
    return {"auc": float(real), "p": float(p), "null_mean": float(null.mean()),
            "names": names}


# ----------------------------------------------------------------------------- main

def main():
    R = list(csv.DictReader(open(TRADES, encoding="utf-8")))
    rows, skip = [], defaultdict(int)
    for i, r in enumerate(R, 1):
        path, why = path_for(r)
        if path is None:
            skip[why] += 1
            continue
        cf = counterfactuals(path, float(r["qty"]))
        if cf is None:
            skip["bad_entry_ask"] += 1
            continue
        rows.append((r, path, cf))
        if i % 100 == 0:
            print("  %d/%d  usable=%d  skips=%s" % (i, len(R), len(rows), dict(skip)),
                  flush=True)

    print("\n" + "=" * 96)
    print("Q4 -- THE EXIT, on his actual contracts at real NBBO")
    print("=" * 96)
    print("  journal round trips        %d" % len(R))
    for k, v in sorted(skip.items()):
        print("  dropped %-22s %d" % (k, v))
    print("  reconstructed              %d" % len(rows))
    days = [r["date"] for r, _, _ in rows]
    print("  date clusters              %d" % len(set(days)))

    his = np.array([cf["his"] for _, _, cf in rows])
    actual = np.array([float(r["net"]) for r, _, _ in rows])
    print("\n  SANITY: simulated-NBBO total ${:+,.2f} vs his broker-reported ${:+,.2f}"
          .format(his.sum(), actual.sum()))
    print("          per trade  $%+.2f vs $%+.2f   (difference $%+.2f is simulation"
          " optimism)" % (his.mean(), actual.mean(), his.mean() - actual.mean()))

    names = (["target_%d" % int(100 * t) for t in TARGETS]
             + ["stop_%d" % int(100 * s) for s in STOPS]
             + ["time_%d" % t for t in TIMES]
             + ["trail_%d" % int(100 * t) for t in TRAILS])
    print("\n  Q4a -- HIS EXIT vs 14 MECHANICAL ALTERNATIVES (paired, $ per trade)")
    print("  %-12s %11s %11s %22s %9s %9s" % ("rule", "$/trade", "vs his",
                                              "95% CI on diff", "p raw", "p Holm"))
    ps, res = [], []
    for nm in names:
        v = np.array([cf[nm] for _, _, cf in rows])
        d = v - his
        lo, hi, p = boot_paired(d, days)
        res.append((nm, v.mean(), d.mean(), lo, hi, p))
        ps.append(p)
    hp = holm(ps)
    for (nm, mv, md, lo, hi, p), ph in zip(res, hp):
        flag = "  <-- clears Holm" if ph < 0.05 else ""
        print("  %-12s %+11.2f %+11.2f   [%+8.2f, %+8.2f] %9.4f %9.4f%s"
              % (nm, mv, md, lo, hi, p, ph, flag))
    print("  %-12s %+11.2f %11s" % ("HIS EXIT", his.mean(), "--"))

    print("\n  EXIT QUALITY DIAGNOSTICS")
    cap = np.array([cf["_capture"] for _, _, cf in rows], float)
    cap = cap[~np.isnan(cap)]
    mfe = np.array([cf["_mfe"] for _, _, cf in rows])
    mae = np.array([cf["_mae"] for _, _, cf in rows])
    hold = np.array([cf["_hold"] for _, _, cf in rows], float)
    pk = np.array([cf["_peak_min"] for _, _, cf in rows], float)
    print("    capture of MFE: mean %.1f%%  median %.1f%%   (n=%d trades that ever showed"
          " a gain)" % (100 * cap.mean(), 100 * np.median(cap), len(cap)))
    print("    mean MFE $%+.2f   mean MAE $%+.2f   mean realised $%+.2f"
          % (mfe.mean(), mae.mean(), his.mean()))
    print("    he holds %.1f min on average; the peak bid arrives at minute %.1f"
          % (hold.mean(), pk.mean()))
    print("    trades exited AFTER the peak: %.1f%%" % (100 * np.mean(pk < hold)))
    win = his > 0
    print("    winners: hold %.1f min, capture %.1f%% of MFE"
          % (hold[win].mean(), 100 * np.nanmean([cf["_capture"] for (_, _, cf), w
                                                 in zip(rows, win) if w])))
    print("    losers : hold %.1f min" % hold[~win].mean())

    haz = hazard(rows)
    pickle.dump({"rows": [(r["date"], cf) for r, _, cf in rows], "q4b": haz},
                open(os.path.join(SCRATCH, "journal_exit_results.pkl"), "wb"))
    print("\nsaved -> journal_exit_results.pkl")


if __name__ == "__main__":
    main()
