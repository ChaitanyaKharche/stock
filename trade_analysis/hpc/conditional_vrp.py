"""Arm B of research/conditional_0dte_preregistration.md -- condition the measured straddle.

    python -m trade_analysis.hpc.conditional_vrp --data data/vrp \
        --trades data/cost_model_trades.parquet

Every threshold here is fixed by the pre-registration and none may be tuned. If you find
yourself wanting to change the tercile, the horizon or the evaluation window, that is a new
pre-registration, not an edit to this file.

WHAT THIS IS
------------
`cost_model.py` already measured 37,517 origins against the raw option archive with real
`entry_bid` and `exit_ask`. The unconditional short straddle nets **-$0.0171/trade at
t=-5.38**. This adds a SELECTION RULE and re-reads the same P&L, so the comparison is exact
and nothing is re-priced. That is the whole design: a conditioning test on an
already-measured null, not a new backtest with new opportunities to invent an edge.

    expected_VRP(t) = iv_var_atm(t) - HAR_IV_forecast(rv_fwd_30 | t)

Positive means implied variance exceeds the forecast of realised variance, so the premium
is expected to be collectable at this origin. Sell only in the TOP TERCILE of that
quantity, with the cut taken from TRAIN origins and applied unchanged -- computing the cut
on the evaluation sample is the specific lookahead most likely to manufacture a result.

The gate is evaluated mechanically at the bottom. It is not a judgement call and this
module returns a non-zero exit code when it fails, so a failed arm cannot be read as a
passed one by someone skimming the output.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

from .har_baseline import TARGET, clean, load, split
from .harp_baseline import HAR_FEATURES, IV_COL, design, fit, predict

TERCILE = 2.0 / 3.0            # pre-registered: top tercile only
MIN_SESSIONS = 50              # pre-registered floor
MIN_TRADES = 1000              # pre-registered floor
T_GATE = 2.0                   # pre-registered
FRAGILE_TOP5_SHARE = 0.50      # pre-registered fragility call


def session_t(x: pd.Series, by: pd.Series) -> tuple[float, float, int]:
    """One mean per session, then a one-sample t across sessions.

    Intraday origins are ~0.9 autocorrelated; a per-trade t inflates n roughly 300x. Same
    treatment as har_baseline.dm_test and cost_model, on purpose.
    """
    d = pd.DataFrame({"s": by.to_numpy(), "x": x.to_numpy()}).groupby("s")["x"].mean()
    n = len(d)
    if n < 3:
        return float("nan"), float("nan"), n
    se = d.std(ddof=1) / math.sqrt(n)
    t = float(d.mean() / se) if se > 0 else float("nan")
    return float(d.mean()), t, n


def paired_session_t(sel: pd.DataFrame, allt: pd.DataFrame) -> tuple[float, float, int]:
    """Per session: mean net of SELECTED origins minus mean net of ALL origins.

    Sessions with no selected origin have no conditioned mean and are dropped -- that is a
    selection, so the surviving session count is reported rather than buried.
    """
    a = sel.groupby("date")["net"].mean()
    b = allt.groupby("date")["net"].mean()
    d = (a - b).dropna()
    n = len(d)
    if n < 3:
        return float("nan"), float("nan"), n
    se = d.std(ddof=1) / math.sqrt(n)
    return float(d.mean()), (float(d.mean() / se) if se > 0 else float("nan")), n


def concentration(sel: pd.DataFrame) -> dict:
    """Session-level concentration, measured against GROSS flow.

    The first version of this divided by the NET total and printed **185.3%**, which is the
    identical defect already found and fixed in dm_concentration.py earlier the same day:
    a share of a near-zero residual is not a share of anything. Writing it the broken way a
    second time, in the file auditing the only live positive in the programme, is the
    strongest possible argument for the ratio being reported against gross.
    """
    per = sel.groupby("date")["net"].sum().sort_values(ascending=False)
    total = float(per.sum())
    gross = float(per.abs().sum())
    k = max(1, int(round(len(per) * 0.05)))
    top5 = float(per.iloc[:k].sum())
    return {"total": total, "gross": gross, "k_top5": k, "top5_sum": top5,
            "top5_share_gross": float(top5 / gross) if gross else float("nan"),
            "net_over_gross": float(total / gross) if gross else float("nan"),
            "n_pos": int((per > 0).sum()), "n_neg": int((per < 0).sum()),
            "n_sessions": int(len(per))}


def leave_k_out_sessions(sel: pd.DataFrame, k_max: int = 12) -> list[dict]:
    """Drop the best sessions one at a time and watch the session-clustered t decay."""
    per = sel.groupby("date")["net"].mean().sort_values(ascending=False)
    out = []
    for k in range(0, min(k_max + 1, len(per) - 3)):
        kept = per.iloc[k:]
        se = kept.std(ddof=1) / math.sqrt(len(kept))
        out.append({"k": k, "n": int(len(kept)), "mean": float(kept.mean()),
                    "t": float(kept.mean() / se) if se > 0 else float("nan")})
    return out


def sign_test_sessions(sel: pd.DataFrame) -> dict:
    """Do MOST sessions make money, or do a few make all of it?"""
    per = sel.groupby("date")["net"].mean()
    wins, n = int((per > 0).sum()), int((per != 0).sum())
    try:
        from scipy import stats
        pv = float(stats.binomtest(wins, n, 0.5).pvalue)
    except Exception:                                              # noqa: BLE001
        z = (wins - n / 2) / math.sqrt(n / 4)
        pv = float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))
    return {"wins": wins, "n": n, "rate": wins / n if n else float("nan"), "p": pv}


def run(data_dir: str, trades_path: str) -> int:
    # ---- the forecast, fitted on TRAIN only -----------------------------------------
    df = clean(load(data_dir))
    parts = split(df)
    train, val = parts["train"], parts["validation"]
    har = [f for f in HAR_FEATURES if f in train.columns]
    build = lambda d: design(d, log_feats=har, iv=True)              # noqa: E731
    beta, rv = fit(train, build)

    for name, part in (("train", train), ("validation", val)):
        part = part.copy()
        part["fc"] = predict(part, build, beta, rv)
        part["expected_vrp"] = part[IV_COL].to_numpy(float) - part["fc"].to_numpy(float)
        if name == "train":
            cut = float(np.nanquantile(part["expected_vrp"], TERCILE))
            train_ev = part
        else:
            val_ev = part
    print(f"  HAR-IV fitted on {len(train)} train origins over "
          f"{train['date'].nunique()} sessions")
    print(f"  expected_VRP top-tercile cut from TRAIN ONLY: {cut:+.8f}")
    print(f"  train expected_VRP  mean {train_ev['expected_vrp'].mean():+.8f}  "
          f"share above cut {100 * (train_ev['expected_vrp'] > cut).mean():.1f}%")

    # ---- the already-measured P&L ----------------------------------------------------
    tr = pd.read_parquet(trades_path)
    tr["date"] = pd.to_datetime(tr["date"]).dt.date
    tr["t"] = pd.to_datetime(tr["t"])
    val_ev = val_ev.copy()
    val_ev["date"] = pd.to_datetime(val_ev["date"]).dt.date
    val_ev["t"] = pd.to_datetime(val_ev["t"])

    merged = tr.merge(val_ev[["date", "t", "expected_vrp", "fc", IV_COL]],
                      on=["date", "t"], how="inner")
    tr_2023 = tr[(tr["t"] >= "2023-01-01") & (tr["t"] < "2024-01-01")]
    rate = len(merged) / max(1, len(tr_2023))
    print(f"\n  cost-model trades total {len(tr)}, in 2023 {len(tr_2023)}")
    print(f"  joined to a forecast: {len(merged)}  ({100 * rate:.1f}% of 2023 origins)")
    if rate < 0.80:
        print("  *** JOIN RATE BELOW 80% -- the origin grids do not line up. STOP. ***")
        print("  A conditioning test on a biased subset of origins measures the bias.")
        return 2
    if merged.empty:
        print("  *** NOTHING JOINED ***")
        return 2

    # ---- the pre-registered rule -----------------------------------------------------
    merged["selected"] = merged["expected_vrp"] > cut
    sel = merged[merged["selected"]]
    rej = merged[~merged["selected"]]

    print("\n" + "=" * 78)
    print("ARM B -- top-tercile expected_VRP, 2023 validation, 30m unhedged short straddle")
    print("=" * 78)
    print(f"  origins      selected {len(sel)}  rejected {len(rej)}  "
          f"({100 * len(sel) / len(merged):.1f}% selected)")
    print(f"  sessions     selected {sel['date'].nunique()}  all {merged['date'].nunique()}")

    for label, d in (("ALL 2023 origins (unconditional)", merged),
                     ("SELECTED (conditioned)", sel),
                     ("REJECTED (the other two terciles)", rej)):
        if d.empty:
            continue
        gm, gt, _ = session_t(d["gross"], d["date"])
        nm, nt, ns = session_t(d["net"], d["date"])
        # PER-TRADE and SESSION-EQUAL-WEIGHTED are different numbers when sessions hold
        # unequal counts of selected origins, and they differ by 4.5x here. The gate names
        # "mean net P&L per trade", so that is what must be gated; the session mean is the
        # right basis for the t because the session is the independent unit. Reporting only
        # one of them, labelled as the other, overstated this result by 4.5x on the first
        # run.
        print(f"\n  {label}")
        print(f"    n {len(d):>6}  sessions {ns:>4}")
        print(f"    gross  per-trade {d['gross'].mean():+.5f}   "
              f"session-mean {gm:+.5f}  session-t {gt:+.2f}")
        print(f"    net    per-trade {d['net'].mean():+.5f}   "
              f"session-mean {nm:+.5f}  session-t {nt:+.2f}")
        print(f"    net    median {d['net'].median():+.5f}  "
              f"win {100 * (d['net'] > 0).mean():.1f}%   "
              f"abs spread cost {d['spread_cost'].mean():.5f}")

    # ---- gate 3: paired against unconditional on the same sessions -------------------
    pm, pt, pn = paired_session_t(sel, merged)
    print(f"\n  PAIRED (selected minus all, per session): mean {pm:+.6f}  "
          f"t {pt:+.2f}  over {pn} sessions")

    # ---- declared-in-advance diagnostics --------------------------------------------
    con = (concentration(sel) if not sel.empty
           else {"top5_share_gross": float("nan"), "net_over_gross": float("nan")})
    print(f"\n  MANDATORY CONCENTRATION AUDIT (pre-registration section 3)")
    print(f"    net {con.get('total', float('nan')):+.4f} over "
          f"{con.get('n_sessions')} sessions "
          f"({con.get('n_pos')} up / {con.get('n_neg')} down)")
    print(f"    gross flow {con.get('gross', float('nan')):.4f}   "
          f"net/gross {con.get('net_over_gross', float('nan')):+.3f}")
    print(f"    top 5% of sessions ({con.get('k_top5')}) hold "
          f"{100 * con['top5_share_gross']:.1f}% of GROSS flow")
    st = sign_test_sessions(sel) if not sel.empty else {"rate": float("nan"), "p": 1.0}
    print(f"    sessions profitable: {st['wins']}/{st['n']} = "
          f"{100 * st['rate']:.1f}%   sign-test p={st['p']:.4f}")
    print("    leave-k-out (drop the best sessions):")
    for r in leave_k_out_sessions(sel) if not sel.empty else []:
        if r["k"] <= 6 or r["k"] % 3 == 0:
            print(f"      k={r['k']:<3} n={r['n']:<4} mean {r['mean']:+.5f}  t {r['t']:+.2f}")
    sp_sel = sel["entry_spread_bp"].median() if not sel.empty else float("nan")
    sp_rej = rej["entry_spread_bp"].median() if not rej.empty else float("nan")
    print(f"  entry spread (median bp): selected {sp_sel:.1f}  rejected {sp_rej:.1f}   "
          f"<- a large gap would smuggle a cost advantage in as a premium")

    # ---- the gate, evaluated mechanically -------------------------------------------
    nm, nt, ns = session_t(sel["net"], sel["date"]) if not sel.empty else (np.nan,) * 3
    per_trade = float(sel["net"].mean()) if not sel.empty else float("nan")
    checks = [
        ("net mean PER TRADE > 0", bool(per_trade > 0), f"{per_trade:+.5f}"),
        (f"session-clustered t > {T_GATE}", bool(nt > T_GATE), f"{nt:+.2f}"),
        ("paired vs unconditional p<0.05 (|t|>1.96, favourable)",
         bool(pt > 1.96), f"t={pt:+.2f}"),
        (f">= {MIN_SESSIONS} sessions", bool(ns >= MIN_SESSIONS), f"{ns}"),
        (f">= {MIN_TRADES} trades", bool(len(sel) >= MIN_TRADES), f"{len(sel)}"),
    ]
    print("\n" + "=" * 78)
    print("GATE (fixed by research/conditional_0dte_preregistration.md)")
    print("=" * 78)
    for name, ok, val in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {val}")
    passed = all(ok for _, ok, _ in checks)
    fragile_reasons = []
    if np.isfinite(con["top5_share_gross"]) and con["top5_share_gross"] > FRAGILE_TOP5_SHARE:
        fragile_reasons.append(f"top 5% of sessions hold "
                               f"{100 * con['top5_share_gross']:.0f}% of gross flow")
    if np.isfinite(st["p"]) and st["p"] > 0.05:
        fragile_reasons.append(f"sessions-profitable sign test is null (p={st['p']:.3f}, "
                               f"{100 * st['rate']:.0f}%)")
    lko = leave_k_out_sessions(sel) if not sel.empty else []
    k_break = next((r["k"] for r in lko if r["k"] > 0 and r["t"] < T_GATE), None)
    if k_break is not None and k_break <= 5:
        fragile_reasons.append(f"dropping {k_break} session(s) takes t under {T_GATE}")
    # The first version flagged this on RELATIVE spread (bp of mid) and called the edge a
    # cost advantage. That was wrong and backwards: selected origins are high-volatility,
    # so the straddle mid is larger, the spread is a smaller FRACTION of it, and the
    # ABSOLUTE cost paid is HIGHER. Only absolute cost moves P&L, so only absolute cost
    # can smuggle in an advantage.
    if not sel.empty and not rej.empty:
        abs_sel = float(sel["spread_cost"].mean())
        abs_rej = float(rej["spread_cost"].mean())
        if abs_rej > 0 and abs_sel < abs_rej * 0.85:
            fragile_reasons.append(
                f"selected origins pay {100 * (1 - abs_sel / abs_rej):.0f}% LESS absolute "
                f"spread ({abs_sel:.5f} vs {abs_rej:.5f}) -- part of the edge is a cost "
                f"advantage, not a premium")
    # Short gamma's signature: a tail far larger than the typical win, in a sample that
    # may not contain a crisis. The unconditional version of this trade had its worst 1%
    # of trades account for 94% of net loss, so this is measured rather than assumed.
    if not sel.empty:
        med_win = sel.loc[sel["net"] > 0, "net"].median()
        p01 = float(np.percentile(sel["net"], 0.1))
        if np.isfinite(med_win) and med_win > 0 and abs(p01) > 10 * med_win:
            fragile_reasons.append(
                f"tail risk: worst 0.1% of trades is {abs(p01) / med_win:.0f}x the median "
                f"win ({p01:+.3f} vs {med_win:+.3f}) -- short gamma, and 2023 contains no "
                f"crisis")
    fragile = bool(fragile_reasons)

    print()
    if passed and not fragile:
        print("  ARM B PASSES. This is a live positive and warrants a forward test, not")
        print("  a claim -- the whole programme's history is of in-sample positives that")
        print("  did not survive honest timing or honest costs.")
    elif passed and fragile:
        print("  ARM B CLEARS ITS GATE BUT IS FRAGILE. Per section 3 of the")
        print("  pre-registration this is reported as fragile regardless of the t:")
        for r in fragile_reasons:
            print(f"    - {r}")
        print("\n  A fragile pass is not a positive. It exits non-zero on purpose, so")
        print("  nobody skimming the output reads it as a clean one.")
    else:
        print("  ARM B FAILS. Conditioning on our own forecast does not rescue the")
        print("  premium. Per section 4 of the pre-registration this is the Pollok")
        print("  warning made concrete: HAR-IV forecasts realised variance 22% better")
        print("  and that does not convert into P&L. QLIKE and money are loosely")
        print("  coupled, and we just measured the coupling on our own instrument.")
    return 0 if (passed and not fragile) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/vrp")
    ap.add_argument("--trades", default="data/cost_model_trades.parquet")
    args = ap.parse_args(argv)
    return run(args.data, args.trades)


if __name__ == "__main__":
    sys.exit(main())
