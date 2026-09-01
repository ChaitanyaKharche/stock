"""Q3b -- score every QQQ minute with the FULL Q1 classifier, out-of-sample in time.

Q3 tested a 4-feature quartile zone and found it slightly negative. The obvious objection is
that the zone is broader than he is: it fires 9.06 times a session against his ~2.8 trades a
day, so it is a coarse shadow of his selection and his sub-selection might be the edge.

This removes the objection by using the whole Q1 model -- all 45 features, OOF AUC 0.85 at
identifying his minutes -- as the signal, and applying it to 2016-01-04 .. 2024-09-10, which
ends the day before his first journal trade. That window is out-of-sample in time by
construction. The in-period window is scored separately and labelled as such.

Same execution rule as everywhere else: a signal from bar k is actionable at bar k+1 OPEN,
and the exit is bar k+1+h OPEN. Both legs lagged identically.

Pre-registered as amendment A4. Falsifier: if the top-centile 95% CI upper bound is below
+5 bp at every horizon, then even a faithful reconstruction of his selection has no tradeable
edge, and the failure is in the strategy rather than in the reconstruction.
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from trade_analysis.backtesting.journal_features import Panel, bars_5m  # noqa: E402
from trade_analysis.backtesting.journal_ml import TARGETS, META, clf_models  # noqa: E402
from trade_analysis.backtesting.journal_zone_wf import boot_ci, holm, sessions  # noqa: E402

SCRATCH = (r"C:\Users\CHAITA~1\AppData\Local\Temp\claude"
           r"\C--Users-chaitanyakharche-Documents-stock"
           r"\3ad3e562-0324-4d02-bf8b-208ff71f108d\scratchpad")
TABLE = os.path.join(SCRATCH, "journal_features.pkl")
SPLIT = "2024-09-11"            # his first journal trade
HORIZONS = (5, 10, 15, 25)
COOLDOWN = 25
PREFIX = 2
STRIDE = 1


def score_session(sess, prior, feat_names):
    """Feature rows for every eligible minute of one session, plus forward outcomes."""
    pre1 = [b for s in prior for b in s]
    pre5 = [b for s in prior for b in bars_5m(s)]
    p1 = Panel(pre1 + sess)
    b5 = bars_5m(sess)
    p5 = Panel(pre5 + b5)
    n1, n5 = len(pre1), len(pre5)
    rows, meta = [], []
    hi = len(sess) - max(HORIZONS) - 2
    op = sess[0]["open"]
    cum_v = 0.0
    cum_pv = 0.0
    for k in range(30, hi, STRIDE):
        j5 = k // 5 - 1
        if j5 < 3:
            continue
        J5, K = n5 + j5, n1 + k
        a5, a1 = p5.atr[J5], p1.atr[K]
        if not a5 or a5 <= 0 or not a1:
            continue
        if p5.macd[J5] is None or p5.adx[J5] is None or p1.adx[K] is None:
            continue
        if p1.dip[K] is None or p5.ema9[J5] is None or p5.ema50[J5] is None:
            continue
        px = sess[k]["close"]
        raw15 = px / b5[j5 - 2]["close"] - 1.0
        d = 1.0 if raw15 > 0 else -1.0
        r5s = [b5[i]["close"] / b5[i - 1]["close"] - 1.0 for i in range(1, j5 + 1)]
        sd5 = float(np.std(r5s, ddof=1)) if len(r5s) > 2 else 0.0
        if sd5 <= 0:
            continue
        f = {}
        m5 = p5.macd[J5]
        f["macd_line_al"] = d * m5[0] / px * 1e4
        f["macd_sig_al"] = d * m5[1] / px * 1e4
        f["macd_hist_al"] = d * m5[2] / px * 1e4
        prev = p5.macd[J5 - 1]
        f["macd_hist_slope_al"] = (d * (m5[2] - prev[2]) / px * 1e4) if prev else 0.0
        f["adx5"] = p5.adx[J5]
        f["di_diff_al"] = d * (p5.dip[J5] - p5.din[J5])
        f["di_plus"] = p5.dip[J5]
        f["di_minus"] = p5.din[J5]
        f["adx1"] = p1.adx[K]
        f["di_diff_1m_al"] = d * (p1.dip[K] - p1.din[K])
        f["vol_rel_ema14"] = (sess[k]["volume"] / p1.vema14[K]) if p1.vema14[K] else 1.0
        f["vol_rel_ema20"] = (sess[k]["volume"] / p1.vema20[K]) if p1.vema20[K] else 1.0
        f["vol5_rel_ema14"] = (b5[j5]["volume"] / p5.vema14[J5]) if p5.vema14[J5] else 1.0
        for nm, s_ in (("ema9", p5.ema9), ("ema20", p5.ema20), ("ema50", p5.ema50)):
            f["px_vs_" + nm + "_al"] = (d * (px - s_[J5]) / a5) if s_[J5] else 0.0
        f["ema9_vs_ema20_al"] = (d * (p5.ema9[J5] - p5.ema20[J5]) / a5
                                 if p5.ema9[J5] and p5.ema20[J5] else 0.0)
        f["ema20_vs_ema50_al"] = (d * (p5.ema20[J5] - p5.ema50[J5]) / a5
                                  if p5.ema20[J5] and p5.ema50[J5] else 0.0)
        st = (1.0 if p5.ema9[J5] > p5.ema20[J5] > p5.ema50[J5] else
              -1.0 if p5.ema9[J5] < p5.ema20[J5] < p5.ema50[J5] else 0.0)
        f["ema_stack_al"] = d * st
        f["rsi5_al"] = d * (p5.rsi[J5] - 50.0) if p5.rsi[J5] is not None else 0.0
        hh = max(b["high"] for b in sess[:k + 1])
        ll = min(b["low"] for b in sess[:k + 1])
        rng = hh - ll
        f["day_range_pos_al"] = d * ((px - ll) / rng - 0.5) * 2 if rng > 0 else 0.0
        f["day_range_atr"] = rng / a5
        f["ret_from_open_al"] = d * (px / op - 1.0) * 1e4
        orh = max(b["high"] for b in sess[:15])
        orl = min(b["low"] for b in sess[:15])
        orr = orh - orl
        f["or15_pos_al"] = d * ((px - orl) / orr - 0.5) * 2 if orr > 0 else 0.0
        f["or15_cleared_al"] = d * (1.0 if px > orh else -1.0 if px < orl else 0.0)
        tv = sum(b["volume"] for b in sess[:k + 1])
        vw = (sum((b["high"] + b["low"] + b["close"]) / 3 * b["volume"]
                  for b in sess[:k + 1]) / tv) if tv > 0 else px
        f["vwap_dist_al"] = d * (px - vw) / a5
        f["trail15_sigma_al"] = d * raw15 / (sd5 * math.sqrt(3))
        for w in (1, 5, 15, 30):
            f["ret_%dm_al" % w] = (d * (px / sess[k - w]["close"] - 1.0) * 1e4
                                   if k - w >= 0 else 0.0)
        upn = sum(1 for b in sess[max(0, k - 14):k + 1] if b["close"] > b["open"])
        f["upbar_frac_al"] = d * (upn / min(15, k + 1) - 0.5) * 2
        f["realised_vol_bp"] = sd5 * 1e4
        f["dist_round25_atr"] = abs(px - round(px / 25.0) * 25.0) / a5
        f["dist_round5_atr"] = abs(px - round(px / 5.0) * 5.0) / a5
        ph = max(b["high"] for b in prior[-1])
        pl = min(b["low"] for b in prior[-1])
        pc = prior[-1][-1]["close"]
        f["dist_pdh_atr_al"] = d * (px - ph) / a5
        f["dist_pdl_atr_al"] = d * (px - pl) / a5
        f["gap_al"] = d * (op / pc - 1.0) * 1e4
        f["prior_day_ret_al"] = d * (pc / prior[-1][0]["open"] - 1.0) * 1e4
        f["prior_range_atr"] = (ph - pl) / a5
        mins = (sess[k]["ts"] - sess[0]["ts"]).total_seconds() / 60.0
        f["tod_min"] = mins
        f["tod_first30"] = 1.0 if mins < 30 else 0.0
        f["tod_last60"] = 1.0 if mins > 330 else 0.0
        f["weekday"] = float(sess[k]["ts"].weekday())
        f["atr_pct"] = a5 / px * 1e4
        entry = sess[k + 1]["open"]
        if entry <= 0:
            continue
        rows.append([f.get(n, 0.0) for n in feat_names])
        m = {"k": k, "dir": d}
        for h in HORIZONS:
            m["fwd_%d" % h] = d * (sess[k + 1 + h]["open"] / entry - 1.0) * 1e4
        meta.append(m)
    return rows, meta


def report(tag, sel, days, n_universe):
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
    df = pd.read_pickle(TABLE)
    allf = [c for c in df.columns
            if c not in TARGETS and c not in META and not c.startswith("fwd_")]
    plac = df[df.is_entry == 0]
    FEATS = [c for c in allf if plac[c].notna().mean() > 0.99]
    d = df.dropna(subset=FEATS)
    Xtr = d[FEATS].to_numpy(float)
    ytr = d.is_entry.to_numpy(int)
    mdl = clf_models()["randomforest"]
    mdl.fit(Xtr, ytr)
    print("Q1 model fitted on %d rows (%d his entries), %d features"
          % (len(ytr), int(ytr.sum()), len(FEATS)), flush=True)

    S = sessions("QQQ")
    print("QQQ sessions %d   %s -> %s" % (len(S), S[0][0], S[-1][0]), flush=True)
    oos, ins = [], []
    for i in range(PREFIX, len(S)):
        day, sess = S[i]
        prior = [S[i - j][1] for j in range(PREFIX, 0, -1)]
        try:
            rows, meta = score_session(sess, prior, FEATS)
        except Exception:
            continue
        if not rows:
            continue
        pr = mdl.predict_proba(np.asarray(rows, float))[:, 1]
        bag = oos if day < SPLIT else ins
        for p_, m in zip(pr, meta):
            m["p"] = float(p_)
            m["day"] = day
            bag.append(m)
        if (i + 1) % 300 == 0:
            print("  %d/%d  oos=%d in=%d" % (i + 1, len(S), len(oos), len(ins)),
                  flush=True)

    pickle.dump({"oos": oos, "ins": ins, "feats": FEATS},
                open(os.path.join(SCRATCH, "model_wf.pkl"), "wb"))

    for tag, bag in (("OUT-OF-SAMPLE  2016-01-04 .. 2024-09-10", oos),
                     ("IN-PERIOD      2024-09-11 .. 2026-08-27", ins)):
        print("\n" + "=" * 92)
        print("Q3b -- " + tag)
        print("=" * 92)
        if not bag:
            print("  no rows")
            continue
        ps = np.array([r["p"] for r in bag])
        print("  scored minutes %d over %d sessions   P(entry): p50 %.3f  p90 %.3f  "
              "p99 %.3f  max %.3f"
              % (len(bag), len({r["day"] for r in bag}), np.percentile(ps, 50),
                 np.percentile(ps, 90), np.percentile(ps, 99), ps.max()))
        allsel = [r for r in bag]
        report("ALL scored minutes", allsel, [r["day"] for r in allsel], len(bag))
        for name, q in (("TOP DECILE  P>=p90", 90), ("TOP CENTILE P>=p99", 99)):
            thr = np.percentile(ps, q)
            cand = sorted([r for r in bag if r["p"] >= thr], key=lambda r: (r["day"], r["k"]))
            sel, last = [], {}
            for r in cand:                       # enforce the same 25-minute cooldown
                if r["k"] - last.get(r["day"], -10 ** 9) < COOLDOWN:
                    continue
                last[r["day"]] = r["k"]
                sel.append(r)
            report(name + "  (cooldown applied)", sel, [r["day"] for r in sel], len(bag))
            if sel:
                yb = defaultdict(list)
                for r in sel:
                    yb[r["day"][:4]].append(r["fwd_15"])
                print("    by year (15m): " + "  ".join(
                    "%s %+.2f" % (y, np.mean(v)) for y, v in sorted(yb.items())))
    print("\n  CONVERSION FLOOR reminder: an ATM 0DTE needs about +5 bp of underlying move")
    print("  to clear spread and theta.")


if __name__ == "__main__":
    main()
