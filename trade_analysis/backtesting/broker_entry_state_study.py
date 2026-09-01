"""Duration-matched entry-timing percentile + one pre-entry state variable, on the
BROKER's real executions (research/round_trips.csv + research/order_history_raw.json).

SAFE TO DELETE. Imports only from `..data_sources.thetadata_client`; edits nothing.
Writes exclusively to its own scratchpad/output paths.

Pre-registration: research/entry_state_preregistration.md (written before this ran).

WHAT THIS IS NOT
----------------
Not a strategy backtest. There is no signal, no entry rule, no target. It measures the
trader's own realised executions against a duration-matched counterfactual in the same
contract on the same session.

TIMING CONVENTION (the bug this project lost a year to)
-------------------------------------------------------
ThetaData minute bars/quotes are LEFT-LABELLED: the row labelled 11:44 covers
[11:44, 11:45) and its close is not knowable until 11:45:00. Therefore:

  * pre-entry underlying state at execution second `t` uses the last bar whose label
    is <= t - 1 minute, i.e. the last bar that has CLOSED at t.
  * the option-quote enumeration indexes minutes by label but only ever compares
    label-to-label, so the labelling convention cancels and no lookahead enters.

DATA TRAPS HANDLED
------------------
  * ThetaData zero-fills minutes with no trades -> require bid>0 and ask>0.
  * 09:30 quote row is universally bid=0/ask=0; 16:00 is a settlement stub. Window
    restricted to 09:31..15:59.
  * Market holidays return a full zero-filled grid and are not `.empty`.
  * client_bid/ask_at_submission is SIGN-FLIPPED on 289 sell orders placed from the
    position-detail screen (client_ask < 0). Repaired by negate-and-swap.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trade_analysis.data_sources.thetadata_client import ThetaDataClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(
    "C:/Users/CHAITA~1/AppData/Local/Temp/claude/"
    "C--Users-chaitanyakharche-Documents-stock/"
    "8e8c7ab5-fde0-46f3-85ac-5b93b9447703/scratchpad"
)
SCRATCH.mkdir(parents=True, exist_ok=True)
ET = "America/New_York"


# --------------------------------------------------------------------------- load
def load_round_trips() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "research" / "round_trips.csv")
    for c in ("entry_ts", "exit_ts"):
        df[c] = pd.to_datetime(df[c], format="ISO8601", utc=True).dt.tz_convert(ET)
    df["edate"] = df["entry_ts"].dt.date.astype(str)
    df["xdate"] = df["exit_ts"].dt.date.astype(str)
    return df


# ------------------------------------------------------------------- option quotes
def fetch_option_quotes(trips: pd.DataFrame, tag: str) -> dict:
    """One request per exact contract-session. Persists after every request."""
    out_path = SCRATCH / f"optq_{tag}.pkl"
    store = pickle.loads(out_path.read_bytes()) if out_path.exists() else {}
    cli = ThetaDataClient()
    keys = (
        trips[["symbol", "expiry", "edate", "right", "strike"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    keys = list(keys)
    for i, k in enumerate(keys, 1):
        if k in store:
            continue
        sym, exp, date, right, strike = k
        try:
            q = cli.quotes(sym, exp, date, right=right, strike=strike, interval="1m")
        except Exception as exc:  # noqa: BLE001
            store[k] = ("ERR", str(exc)[:160])
            out_path.write_bytes(pickle.dumps(store))
            continue
        if q is None or len(q) == 0:
            store[k] = ("EMPTY", 0)
        else:
            q = q[["timestamp", "bid", "ask"]].copy()
            q["timestamp"] = pd.to_datetime(q["timestamp"])
            store[k] = ("OK", q)
        if i % 10 == 0 or i == len(keys):
            out_path.write_bytes(pickle.dumps(store))
            print(f"  optq {i}/{len(keys)}  live={cli.live_calls} hit={cli.cache_hits}",
                  flush=True)
    out_path.write_bytes(pickle.dumps(store))
    return store


def valid_minutes(q: pd.DataFrame) -> pd.DataFrame:
    """Filter to enumerable minutes. Never forward-fills."""
    t = q["timestamp"]
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    mins = t.dt.hour * 60 + t.dt.minute
    ok = (
        (mins >= 9 * 60 + 31)
        & (mins <= 15 * 60 + 59)
        & (q["bid"] > 0)
        & (q["ask"] > 0)
        & (q["ask"] >= q["bid"])
    )
    v = q.loc[ok, ["bid", "ask"]].copy()
    v["min_idx"] = mins[ok].to_numpy()
    v["mid"] = (v["bid"] + v["ask"]) / 2.0
    return v.drop_duplicates("min_idx").set_index("min_idx").sort_index()


# ------------------------------------------------------------------- stock bars
def fetch_stock_bars(trips: pd.DataFrame, tag: str) -> dict:
    """One request per (symbol, month). Persists after every request."""
    out_path = SCRATCH / f"stk_{tag}.pkl"
    store = pickle.loads(out_path.read_bytes()) if out_path.exists() else {}
    cli = ThetaDataClient()
    t = trips.copy()
    t["ym"] = t["entry_ts"].dt.strftime("%Y-%m")
    # one extra calendar month of lead-in so sigma_20 has ~20 sessions
    want = set()
    for sym, ym in t[["symbol", "ym"]].drop_duplicates().itertuples(index=False):
        want.add((sym, ym))
        prev = (pd.Timestamp(ym + "-01") - pd.offsets.MonthBegin(1)).strftime("%Y-%m")
        want.add((sym, prev))
    want = sorted(want)
    for i, (sym, ym) in enumerate(want, 1):
        if (sym, ym) in store:
            continue
        start = pd.Timestamp(ym + "-01")
        end = start + pd.offsets.MonthEnd(1)
        try:
            b = cli.stock_minute_bars(sym, start.strftime("%Y-%m-%d"),
                                      end.strftime("%Y-%m-%d"))
            store[(sym, ym)] = b[["Close"]] if len(b) else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            store[(sym, ym)] = ("ERR", str(exc)[:160])
        if i % 10 == 0 or i == len(want):
            out_path.write_bytes(pickle.dumps(store))
            print(f"  stk {i}/{len(want)}  live={cli.live_calls} hit={cli.cache_hits}",
                  flush=True)
    out_path.write_bytes(pickle.dumps(store))
    return store


def assemble_stock(store: dict) -> dict:
    """symbol -> DataFrame(Close) indexed by tz-aware ET minute, positive prices only."""
    by = {}
    for (sym, _ym), v in store.items():
        if isinstance(v, tuple) or v is None or len(v) == 0:
            continue
        by.setdefault(sym, []).append(v)
    return {
        s: pd.concat(fr).sort_index().pipe(lambda d: d[d["Close"] > 0])
                                    .pipe(lambda d: d[~d.index.duplicated()])
        for s, fr in by.items()
    }


# ------------------------------------------------------------------- statistics
def cluster_bootstrap(values, clusters, reps=10000, seed=7, stat=np.mean):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uniq}
    draws = np.empty(reps)
    for r in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by[c] for c in pick])
        draws[r] = stat(values[sel])
    point = stat(values)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0.5).mean(), (draws >= 0.5).mean()) if stat is np.mean else np.nan
    return point, lo, hi, p, draws


def design_effect(values, clusters):
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    n = len(values)
    mu = values.mean()
    num = sum(
        (values[clusters == c].sum() - (clusters == c).sum() * mu) ** 2
        for c in np.unique(clusters)
    )
    se_cr = np.sqrt(num) / n
    se_iid = values.std(ddof=1) / np.sqrt(n)
    return (se_cr / se_iid) ** 2, se_cr, se_iid
