"""Is MOMO_CHASE simply MIS-SPECIFIED? Three variants, ten years, one declared falsifier.

PRE-REGISTERED BEFORE RUNNING (2026-09-04):

  The journal study (research/journal_reverse_engineering_results.md) measured what
  actually separates his entry minutes from same-session placebos. The frozen MOMO_CHASE
  does not use the top discriminator at all, and gates the DMI on the wrong timeframe:

      px_vs_ema9_al     AUC 0.809   NOT in MOMO_CHASE
      di_diff_1m_al     AUC 0.806   MOMO_CHASE uses the 5-min DMI (AUC 0.634)
      trail15_sigma_al  AUC 0.791   in MOMO_CHASE
      macd_hist_al      AUC 0.681   in MOMO_CHASE
      adx1              AUC 0.641   in MOMO_CHASE (5-min)

  Those AUCs came from a behaviour study with NO P&L in it, so using them to correct the
  specification is not fitting to outcomes. Whether the correction pays is the question.

VARIANTS (exactly three, fixed now, no others will be tried):
  V0  MOMO_CHASE as frozen                      -- sanity check, must reproduce ~+$0.05
  V1  V0 + px_vs_ema9 >= 0.71 ATR               -- add the missing top discriminator
  V2  V0 with the 1-minute DMI instead of 5-min -- fix the timeframe

FALSIFIER: if no variant beats +$1.00/trade with a day-clustered p < 0.05 after Holm m=3,
the mis-specification does not matter, MOMO_CHASE is closed, and no further variant is
tried. A variant that merely beats V0 without clearing that bar is noise.

WHAT IS ALREADY KNOWN AND CONSTRAINS THE EXPECTATION: Q3 of the journal study ran the
correctly-specified version of his rule as a ZONE -- px_vs_ema9>=0.71, di_diff_1m>=9.39,
trail15_sigma>=0.39, ret_15m>=6.65 -- over 34,534 signals and got -0.149 bp at 5 minutes,
significantly NEGATIVE. So the prior here is that fidelity to his behaviour makes things
worse, not better. This test differs only in using MOMO_CHASE's own 25-minute time exit
and share P&L rather than fixed-horizon underlying returns.

Execution: signal from a closed 5-minute bar, fill at the NEXT 1-minute bar's open, exit
25 minutes later at the open. Both legs lagged identically. $10,000 notional.
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
from trade_analysis.backtesting.journal_zone_wf import boot_ci, holm, sessions  # noqa: E402

NOTIONAL = 10_000.0
TIME_EXIT = 25
PX_VS_EMA9_MIN = 0.71      # his own 25th percentile, from the journal study
SIGMA_MULT = 0.60          # frozen MOMO_CHASE threshold, unchanged
PREFIX = 2


def scan(sess, prior, variant):
    pre1 = [b for s in prior for b in s]
    pre5 = [b for s in prior for b in bars_5m(s)]
    p1 = Panel(pre1 + sess)
    b5 = bars_5m(sess)
    p5 = Panel(pre5 + b5)
    n1, n5 = len(pre1), len(pre5)
    out, last = [], -10 ** 9
    hi = len(sess) - TIME_EXIT - 2
    for k in range(30, hi):
        ts = sess[k]["ts"].time()
        if not (dt.time(10, 30) <= ts <= dt.time(14, 30)):
            continue
        if k % 5 != 4:                       # 5-minute setup: evaluate on bucket closes
            continue
        j5 = k // 5 - 1
        if j5 < 3:
            continue
        J5, K = n5 + j5, n1 + k
        a5 = p5.atr[J5]
        if not a5 or a5 <= 0 or p5.ema9[J5] is None or p5.macd[J5] is None:
            continue
        if p5.adx[J5] is None or p5.dip[J5] is None or p1.dip[K] is None:
            continue
        px = sess[k]["close"]
        # trailing 15-minute z, 1-min sigma convention -- exactly as frozen
        c1 = [b["close"] for b in sess[:k + 1]]
        if len(c1) < 16:
            continue
        r1 = np.diff(c1) / np.asarray(c1[:-1])
        s1 = float(np.std(r1, ddof=1)) if len(r1) > 2 else 0.0
        if s1 <= 0:
            continue
        trail = px / c1[-16] - 1.0
        z = trail / (s1 * math.sqrt(15))
        if abs(z) < SIGMA_MULT:
            continue
        d = 1.0 if z > 0 else -1.0
        if p5.macd[J5][2] * d <= 0:
            continue
        if variant == "V2":
            if (p1.dip[K] - p1.din[K]) * d <= 0:       # 1-MINUTE DMI
                continue
        else:
            if (p5.dip[J5] - p5.din[J5]) * d <= 0:     # 5-minute DMI, as frozen
                continue
        if p5.adx[J5] <= 20:
            continue
        if variant == "V1":
            if d * (px - p5.ema9[J5]) / a5 < PX_VS_EMA9_MIN:
                continue
        if k - last < TIME_EXIT:
            continue
        last = k
        entry = sess[k + 1]["open"]
        ex = sess[min(k + 1 + TIME_EXIT, len(sess) - 1)]["open"]
        if entry <= 0:
            continue
        sh = NOTIONAL / entry
        out.append({"pnl": sh * (ex - entry) * d, "dir": d})
    return out


def main():
    variants = ["V0", "V1", "V2"]
    res = {v: ([], []) for v in variants}
    for sym in ("QQQ", "SPY"):
        S = sessions(sym)
        if not S:
            continue
        print(f"{sym}: {len(S)} sessions", flush=True)
        for i in range(PREFIX, len(S)):
            day, sess = S[i]
            prior = [S[i - j][1] for j in range(PREFIX, 0, -1)]
            for v in variants:
                try:
                    for r in scan(sess, prior, v):
                        res[v][0].append(r["pnl"])
                        res[v][1].append(sym + day)
                except Exception:                        # noqa: BLE001
                    pass
            if (i + 1) % 500 == 0:
                print(f"  {sym} {i+1}/{len(S)}  V0={len(res['V0'][0])}", flush=True)

    print("\n" + "=" * 88)
    print("MOMO_CHASE VARIANTS -- $10,000 notional, next-bar fills, 25-min time exit")
    print("=" * 88)
    labels = {"V0": "as frozen (sanity check)",
              "V1": "+ px_vs_ema9 >= 0.71 ATR",
              "V2": "1-min DMI instead of 5-min"}
    ps, rows = [], []
    for v in variants:
        pnl, days = res[v]
        if len(pnl) < 50:
            rows.append((v, len(pnl), np.nan, np.nan, np.nan, 1.0)); ps.append(1.0)
            continue
        lo, hi, p = boot_ci(np.array(pnl), days, nboot=10000)
        rows.append((v, len(pnl), float(np.mean(pnl)), lo, hi, p))
        ps.append(p)
    hp = holm(ps)
    print(f"  {'':4}{'variant':<32}{'n':>7}{'$/trade':>10}{'95% CI':>22}{'p':>9}{'Holm':>9}")
    for (v, n, m, lo, hi, p), a in zip(rows, hp):
        if np.isnan(m):
            print(f"  {v:<4}{labels[v]:<32}{n:>7}   too few")
            continue
        flag = "  <-- clears" if (a < 0.05 and m > 1.0) else ""
        print(f"  {v:<4}{labels[v]:<32}{n:>7}{m:>+10.2f}"
              f"   [{lo:>+7.2f},{hi:>+7.2f}]{p:>9.4f}{a:>9.4f}{flag}")
    print("\n  FALSIFIER: a variant must beat +$1.00/trade AND clear Holm at 0.05.")
    win = [v for (v, n, m, lo, hi, p), a in zip(rows, hp)
           if not np.isnan(m) and m > 1.0 and a < 0.05]
    print(f"  variants clearing it: {win if win else 'NONE -- MOMO_CHASE is closed'}")


if __name__ == "__main__":
    main()
