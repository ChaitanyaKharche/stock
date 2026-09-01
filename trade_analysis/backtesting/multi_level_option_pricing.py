"""Price the multi-level confluence signals on REAL NBBO chains.

WHY THIS REPLACES THE PROXY
---------------------------
multi_level_orb_backtest judges signals by whether the underlying's max
favourable excursion cleared 0.20%, on the reasoning that ~0.20% of travel is
what a 25% option gain needs. That proxy has a confound: it is a FIXED
percentage, so a higher-volatility symbol clears it more often for free. QQQ's
daily sigma is ~1.23% vs SPY's ~0.96%, which inflates QQQ's apparent hit rate
against SPY's. Measured in sigma units the two symbols were indistinguishable
(t=0.25), so the cross-symbol gap the proxy reports is partly volatility.

Real option pricing has no such confound: the strike that costs $1.25 already
adapts to volatility, which is the whole point. This module therefore buys the
observed ASK, sells the observed BID, and reports what the signal would have
paid -- the same machinery as spy_qqq_0dte_quotes_backtest, applied to the
multi-level signal set instead of the single-level one.

SCOPE: options history floors at 2020-01-01 (measured), so the 2016-2019
signals cannot be priced. That is a coverage limit, not a filter -- it is
reported explicitly rather than silently dropping rows.

ISOLATION: imports only, edits nothing, writes its own CSV.
"""
import argparse
import math

import numpy as np
import pandas as pd

from ..data_sources.thetadata_client import ThetaDataClient, OPTION_HISTORY_START
from .spy_qqq_0dte_quotes_backtest import _chain_for_entry, _pick_contract, simulate
from .spy_qqq_0dte_real_backtest import PROFIT_TARGET_GRID
from ..paths import LOGS_DIR

SIGNALS = LOGS_DIR / "multi_level_orb_signals.csv"
BREAKEVEN_WIN = 75.3   # from measured payoffs: avg win +31.5%, avg loss -95.7%

GATE_SETS = [
    ("levels only",          []),
    ("+ volume > 20 EMA",    ["gate_vol"]),
    ("+ MACD 9/17/9",        ["gate_vol", "gate_macd"]),
    ("+ DMI/ADX",            ["gate_vol", "gate_macd", "gate_dmi"]),
    ("+ 10m/15m  PRIMARY",   ["gate_vol", "gate_macd", "gate_dmi", "gate_multi_tf"]),
]


def _stat(v):
    v = pd.Series(v).dropna()
    if len(v) < 3:
        return None
    se = v.std() / math.sqrt(len(v))
    return {"n": len(v), "mean": v.mean(), "se": se, "t": v.mean() / se,
            "win": (v > 0).mean() * 100,
            "lo": v.mean() - 1.96 * se, "hi": v.mean() + 1.96 * se}


def price_signals(df, theta, symbol):
    """One chain fetch per signal; price at the honest decision time."""
    out = []
    skips = {"chain_empty": 0, "no_quoted_strike": 0, "fetch_error": 0}
    for i, r in enumerate(df.itertuples(), 1):
        e = {"date": r.date.date() if hasattr(r.date, "date") else r.date,
             "direction": r.direction,
             "entry_time": pd.Timestamp(r.entry_time),
             "entry_price": r.entry_price,
             # simulate() only echoes gap_regime into its output row (no logic
             # depends on it) and the multi-level signal set never computed it,
             # so a sentinel is honest here rather than inventing a regime.
             "gap_regime": "NA"}
        if e["entry_time"].tzinfo is None:
            e["entry_time"] = e["entry_time"].tz_localize("America/New_York")
        try:
            chain = _chain_for_entry(theta, symbol, e)
        except Exception:
            skips["fetch_error"] += 1
            continue
        if chain.empty:
            skips["chain_empty"] += 1
            continue
        pick = _pick_contract(chain, e["entry_time"])
        if pick is None:
            skips["no_quoted_strike"] += 1
            continue
        row = {"date": e["date"], "direction": r.direction,
               "gate_vol": r.gate_vol, "gate_macd": r.gate_macd,
               "gate_dmi": r.gate_dmi, "gate_multi_tf": r.gate_multi_tf,
               "entry_ask": pick["entry_ask"],
               "year": pd.Timestamp(e["date"]).year}
        for pt in PROFIT_TARGET_GRID + [None]:
            tag = f"t{int(pt*100)}" if pt else "hold"
            s = simulate(e, chain, pick, pt)
            row[tag] = s["ret_pct_net"]
            if pt == 0.25:
                row["exit_reason"] = s["exit_reason"]
        out.append(row)
        if i % 50 == 0:
            print(f"    {i}/{len(df)} priced {len(out)} | {theta.stats()}", flush=True)
    print(f"  priced {len(out)}, skipped {sum(skips.values())} {skips}", flush=True)
    return pd.DataFrame(out)


def report(priced, symbol, n_total, n_in_window):
    line = "=" * 96
    print(f"\n{line}")
    print(f"{symbol}: MULTI-LEVEL SIGNALS ON REAL NBBO  (buy ask / sell bid, net of fees)")
    print(line)
    print(f"  {n_total} signals total, {n_in_window} on/after {OPTION_HISTORY_START} "
          f"(options floor), {len(priced)} priced")
    print(f"  breakeven win rate ~{BREAKEVEN_WIN}% -- judge the win column against it\n")

    hdr = (f"  {'gate set':22s} {'n':>4s} {'t25 mean':>9s} {'win':>6s} {'t':>6s} "
           f"{'95% CI':>17s} {'t35':>8s} {'hold':>8s}")
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for label, gates in GATE_SETS:
        sub = priced
        for g in gates:
            sub = sub[sub[g]]
        s = _stat(sub["t25"])
        if not s:
            print(f"  {label:22s} {len(sub):>4d}   too few to report")
            continue
        s35, sh = _stat(sub["t35"]), _stat(sub["hold"])
        flag = "" if s["win"] >= BREAKEVEN_WIN else "  <below BE"
        print(f"  {label:22s} {s['n']:>4d} {s['mean']:>+8.2f}% {s['win']:>5.1f}% "
              f"{s['t']:>+6.2f} [{s['lo']:>+6.2f},{s['hi']:>+6.2f}] "
              f"{s35['mean']:>+7.2f}% {sh['mean']:>+7.2f}%{flag}")

    prim = priced
    for g in GATE_SETS[-1][1]:
        prim = prim[prim[g]]
    if len(prim) > 3:
        print(f"\n  PRIMARY endpoint, by year:")
        for y, g in prim.groupby("year"):
            st = _stat(g["t25"])
            if st:
                print(f"    {y}  n={st['n']:>3d}  {st['mean']:>+7.2f}%  win {st['win']:>5.1f}%")
        pos = sum(1 for _, g in prim.groupby("year") if g["t25"].mean() > 0)
        print(f"    -> positive in {pos}/{prim['year'].nunique()} years")
        if "exit_reason" in prim:
            print(f"    exit reasons: {prim['exit_reason'].value_counts().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["QQQ", "SPY"])
    args = ap.parse_args()

    sig = pd.read_csv(SIGNALS, parse_dates=["date", "entry_time"])
    theta = ThetaDataClient()
    frames = []
    for sym in args.symbols:
        s = sig[sig["symbol"] == sym]
        inwin = s[s["date"] >= OPTION_HISTORY_START]
        print(f"\n=== {sym}: {len(s)} signals, {len(inwin)} priceable ===", flush=True)
        if inwin.empty:
            continue
        try:
            p = price_signals(inwin, theta, sym)
        except Exception as exc:
            print(f"  {sym} FAILED: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        if not p.empty:
            p["symbol"] = sym
            report(p, sym, len(s), len(inwin))
            frames.append(p)

    if frames:
        allp = pd.concat(frames, ignore_index=True)
        out = LOGS_DIR / "multi_level_option_priced.csv"
        allp.to_csv(out, index=False)
        print(f"\nSaved -> {out}")
    print(f"theta API: {theta.stats()}")
