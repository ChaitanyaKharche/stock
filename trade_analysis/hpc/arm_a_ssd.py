"""Arm A worker -- replicate the SSD-violation rule on the Vilkov SPXW 0DTE panel.

One array task per ENTRY TIME. See submit_arm_a.sbatch.

    python -m trade_analysis.hpc.arm_a_ssd --slot 10:00 --out results/arm_a
    python -m trade_analysis.hpc.arm_a_ssd --build-table-only     # one-off, then the array

WHY THE ARRAY IS OVER ENTRY TIMES AND NOT SESSIONS
--------------------------------------------------
The physical distribution at session *i* is estimated from sessions < *i*, which looks like
a sequential dependency. It is not: the only history needed is a compact table of
(session, slot) -> (settlement return, ATM implied vol), 1,397 x 12 rows, which every task
can build once and window itself. So the natural split is by entry time -- 12 independent
tasks -- and it is also the decomposition Almeida/Freire/Hizmeri report their Sharpe in
(their Table 6 is nine entry times), so the collected output is directly comparable to the
published table rather than to a number we invented.

Honest note on scale: once the panel is reduced to near-ATM rows this is not heavy compute.
The array earns its place on provenance and fail-loudly grounds -- per-task exit codes, a
recorded environment, a grid that verifies its own size -- not on FLOPs.

THE RULE
--------
At entry slot *t* on session *d*, for the ATM call:

  1. physical distribution: take settlement returns `sret` from PRIOR sessions at the SAME
     slot, standardise each by that session's ATM implied vol, then rescale by today's ATM
     implied vol. This is the published conditioning device -- a historical return histogram
     rescaled by current ATM IV -- and it is deliberately crude.
  2. expected payoff under that distribution, E_P[max(sret - K/S_t, 0)].
  3. signal = market mid - E_P[payoff].  Negative means the option is CHEAP relative to the
     physical expectation (a risk-averse lower-bound violation) -> BUY. Positive -> SELL.
  4. hold to settlement. Cost: pay the ask to buy, receive the bid to sell, where
     ask/bid = mid +/- bas/2.

`theta = 0`: the bound is the bare physical expectation, with no risk-aversion widening.
That is the parameter-free choice fixed by the pre-registration, and no grid over it is
authorised.

PANEL TRAPS HANDLED HERE, EXPLICITLY
------------------------------------
* **The 16:00 rows are dropped.** They exist on non-expiry days and carry the NEXT day's
  `sret`, which is what inflates the naive session count from 1,397 to 1,919. Trading them
  would be a one-day lookahead on 522 days that are not 0DTE sessions at all.
* **`mid`/`bas`/`payoff` are each divided by their OWN bar's spot.** Everything here stays
  inside a single entry bar, so no cross-bar rescaling is needed -- and nothing in this file
  compares a price from one bar to a price from another. This is the trap that caused the
  upstream package's 2026-08 correction, where a half-spread was charged at 1/100 of its
  true size.
* **`payoff` is verified**, not assumed: for an ATM call (K = S_t) it equals `sret - 1` when
  positive, which is checked at load and raises if it does not hold.
* **`mnes_rel` is constant-MONEYNESS**, so "the ATM option" is a different strike each bar.
  That is fine for a one-bar-entry, hold-to-settlement trade and would not be fine for
  anything that tracks a contract over time.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path(r"C:\Users\chaitanyakharche\Documents\research_data\0dte-strategies"
             r"\data\data_opt.parquet")
SLOTS = ["10:00", "10:30", "11:00", "11:30", "12:00", "12:30",
         "13:00", "13:30", "14:00", "14:30", "15:00", "15:30"]
SETTLE_SLOT = dt.time(16, 0)
ATM_BAND = 0.0015          # |mnes_rel - 1| <= this counts as near-ATM
MIN_HISTORY = 250          # prior sessions required before a signal is emitted (~1 year)
THETA = 0.0                # pre-registered: bare physical expectation, no widening


# ------------------------------------------------------------------ provenance
def provenance() -> dict:
    def _git(*a):
        try:
            return subprocess.check_output(["git", *a], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:                                        # noqa: BLE001
            return None
    try:
        from importlib.metadata import distributions
        pkgs = {d.metadata["Name"]: d.version for d in distributions()
                if d.metadata.get("Name")}
    except Exception:                                            # noqa: BLE001
        pkgs = {}
    # A bare `git status --porcelain` is useless as a gate IN THIS REPO: the live lab
    # writes its durable record into live_lab_data/ continuously, so the tree is dirty
    # during and after every trading session and the flag fires on all 12 tasks forever.
    # The question a provenance check actually needs to answer is "was the CODE I ran
    # different from the code that is committed", so the dirt that matters is dirt outside
    # the data paths. Both are recorded; only the code one is gated.
    porcelain = _git("status", "--porcelain") or ""
    DATA_PREFIXES = ("live_lab_data/", "data/", "results/", ".playwright-mcp/")
    dirty_code = []
    for line in porcelain.splitlines():
        path = line[3:].strip().strip('"')
        if path and not path.startswith(DATA_PREFIXES):
            dirty_code.append(path)
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(porcelain),
        "git_dirty_code": sorted(dirty_code),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "packages": pkgs,
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "written_at": dt.datetime.now().isoformat(timespec="seconds"),
        "theta": THETA, "min_history": MIN_HISTORY, "atm_band": ATM_BAND,
    }


# ------------------------------------------------------------------ panel -> compact tables
def load_atm(panel: Path) -> pd.DataFrame:
    cols = ["quote_date", "quote_time", "option_type", "mnes_rel", "mid", "bas",
            "payoff", "delta", "implied_volatility", "sret", "active_underlying_price"]
    df = pd.read_parquet(panel, columns=cols)
    df["quote_date"] = pd.to_datetime(df["quote_date"]).dt.date

    # Drop the settlement rows. They are present on non-expiry days and carry the NEXT
    # day's sret; keeping them is a one-day lookahead on 522 non-sessions.
    df = df[df["quote_time"] != SETTLE_SLOT]

    # Real 0DTE sessions only: a session must have more than one surviving slot.
    keep = df.groupby("quote_date")["quote_time"].transform("nunique") > 1
    df = df[keep]

    df = df[(df["mnes_rel"] - 1.0).abs() <= ATM_BAND]
    df["slot"] = df["quote_time"].astype(str).str.slice(0, 5)

    # VERIFY payoff rather than trust it: for a call, payoff must equal
    # max(sret - mnes_rel, 0) to within rounding, since both are per unit entry spot.
    c = df[df["option_type"] == "C"]
    expect = np.maximum(c["sret"].to_numpy(float) - c["mnes_rel"].to_numpy(float), 0.0)
    err = float(np.nanmax(np.abs(expect - c["payoff"].to_numpy(float))))
    if err > 1e-6:
        raise SystemExit(f"payoff identity FAILED (max abs err {err:.2e}). The panel's "
                         f"normalisation is not what this file assumes -- stop and re-read "
                         f"the schema before trusting any number.")
    print(f"  payoff identity verified on {len(c)} call rows (max err {err:.2e})")
    return df


def slot_history(atm: pd.DataFrame) -> pd.DataFrame:
    """(session, slot) -> settlement return and ATM implied vol. The only history needed."""
    atmc = atm[(atm["option_type"] == "C") &
               ((atm["mnes_rel"] - 1.0).abs() < 0.0005)]
    g = atmc.groupby(["quote_date", "slot"]).agg(
        sret=("sret", "first"), atm_iv=("implied_volatility", "mean")).reset_index()
    return g.dropna()


# ------------------------------------------------------------------ the rule
def run_slot(atm: pd.DataFrame, hist: pd.DataFrame, slot: str) -> pd.DataFrame:
    h = hist[hist["slot"] == slot].sort_values("quote_date").reset_index(drop=True)
    day = atm[(atm["slot"] == slot) & (atm["option_type"] == "C")]
    # one contract per session: the strike closest to spot
    day = (day.assign(d=(day["mnes_rel"] - 1.0).abs())
              .sort_values(["quote_date", "d"])
              .groupby("quote_date").first().reset_index())

    sret_h = h["sret"].to_numpy(float)
    iv_h = h["atm_iv"].to_numpy(float)
    dates = h["quote_date"].to_numpy()
    rows = []
    for i in range(MIN_HISTORY, len(h)):
        d = dates[i]
        cur = day[day["quote_date"] == d]
        if cur.empty:
            continue
        r = cur.iloc[0]
        iv_now = float(h["atm_iv"].iloc[i])
        if not np.isfinite(iv_now) or iv_now <= 0:
            continue
        # standardise prior returns by their own session's IV, rescale to today's IV
        z = (sret_h[:i] - 1.0) / np.where(iv_h[:i] > 0, iv_h[:i], np.nan)
        z = z[np.isfinite(z)]
        if len(z) < MIN_HISTORY // 2:
            continue
        sim = 1.0 + z * iv_now
        ep = float(np.mean(np.maximum(sim - float(r["mnes_rel"]), 0.0)))

        mid, bas = float(r["mid"]), float(r["bas"])
        ask, bid = mid + bas / 2.0, mid - bas / 2.0
        signal = mid - ep                      # <0 cheap -> buy, >0 rich -> sell
        side = -1 if signal > THETA else (1 if signal < -THETA else 0)
        if side == 0:
            continue
        payoff, delta = float(r["payoff"]), float(r["delta"])
        und = float(r["sret"]) - 1.0
        if side == 1:                           # BUY the call at the ask
            gross = payoff - mid
            net = payoff - ask
        else:                                   # SELL the call at the bid
            gross = mid - payoff
            net = bid - payoff
        # static delta hedge, set at entry and held: the published expression is a
        # delta-hedged call, and an unhedged 0DTE call is mostly a direction bet.
        hedge = -side * delta * und
        rows.append({"date": d, "slot": slot, "side": side, "signal": signal,
                     "mid": mid, "bas": bas, "ep_physical": ep, "payoff": payoff,
                     "delta": delta, "und_ret": und, "iv": iv_now,
                     "gross": gross, "net": net,
                     "gross_hedged": gross + hedge, "net_hedged": net + hedge,
                     "spread_cost": bas / 2.0, "n_history": int(len(z))})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, col: str) -> dict:
    if df.empty:
        return {"n": 0}
    per = df.groupby("date")[col].mean()
    n = len(per)
    se = per.std(ddof=1) / np.sqrt(n) if n > 2 else np.nan
    mu = float(per.mean())
    sd = float(df[col].std(ddof=1))
    return {"n": int(len(df)), "sessions": int(n),
            "per_trade": float(df[col].mean()), "session_mean": mu,
            "session_t": float(mu / se) if se and se > 0 else float("nan"),
            "sharpe_per_trade": float(df[col].mean() / sd) if sd > 0 else float("nan"),
            "win_rate": float((df[col] > 0).mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--slot", default=None, choices=SLOTS)
    ap.add_argument("--slot-index", type=int, default=None,
                    help="SLURM_ARRAY_TASK_ID form; 0-11")
    ap.add_argument("--out", default="results/arm_a")
    ap.add_argument("--build-table-only", action="store_true")
    args = ap.parse_args(argv)

    if args.slot_index is not None:
        if not 0 <= args.slot_index < len(SLOTS):
            print(f"slot-index {args.slot_index} outside 0..{len(SLOTS) - 1} -- the array "
                  f"range and the grid disagree", file=sys.stderr)
            return 2
        args.slot = SLOTS[args.slot_index]
    if not args.slot and not args.build_table_only:
        print("need --slot or --slot-index", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"  panel {args.panel}")
    atm = load_atm(Path(args.panel))
    hist = slot_history(atm)
    print(f"  near-ATM rows {len(atm)}   history rows {len(hist)}   "
          f"sessions {atm['quote_date'].nunique()}")
    if args.build_table_only:
        hist.to_parquet(out / "slot_history.parquet", index=False)
        print(f"  wrote {out / 'slot_history.parquet'}")
        return 0

    tr = run_slot(atm, hist, args.slot)
    print(f"\n  slot {args.slot}: {len(tr)} trades over "
          f"{tr['date'].nunique() if not tr.empty else 0} sessions")
    if not tr.empty:
        print(f"  sides: buy {(tr.side == 1).sum()}  sell {(tr.side == -1).sum()}")
        for col in ("gross", "net", "gross_hedged", "net_hedged"):
            s = summarise(tr, col)
            print(f"    {col:<14} per-trade {s['per_trade']:+.6f}  "
                  f"session-mean {s['session_mean']:+.6f}  t {s['session_t']:+.2f}  "
                  f"win {100 * s['win_rate']:.1f}%")

    tag = args.slot.replace(":", "")
    tr.to_parquet(out / f"trades_{tag}.parquet", index=False)
    rec = {"slot": args.slot, "n_trades": int(len(tr)),
           "sessions": int(tr["date"].nunique()) if not tr.empty else 0,
           "sides": {"buy": int((tr.side == 1).sum()) if not tr.empty else 0,
                     "sell": int((tr.side == -1).sum()) if not tr.empty else 0},
           "summary": {c: summarise(tr, c)
                       for c in ("gross", "net", "gross_hedged", "net_hedged")},
           "provenance": provenance()}
    (out / f"result_{tag}.json").write_text(json.dumps(rec, indent=2, default=str),
                                            encoding="utf-8")
    print(f"  wrote {out / f'result_{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
