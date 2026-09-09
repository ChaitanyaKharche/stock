"""Turn the raw 0DTE quote and minute-bar archives into one training frame.

    python -m trade_analysis.hpc.build_vrp_dataset --out data/vrp --symbol SPY

Implements section 3 of research/vrp_preregistration.md. Every feature is computed from
bars and quotes that CLOSED AT OR BEFORE the forecast origin; the bar containing the
origin is excluded, not merely trimmed. That single rule is the difference between this
and the 95.7%-of-edge lookahead already in this project's history, so it is enforced by
construction here rather than by care later: `_features_at` never receives a frame that
extends past `t`.

Output is one parquet per session, plus a manifest. Per-session files because the
walk-forward evaluates by date and a monolithic frame invites accidental shuffling across
the temporal split.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(r"C:\Users\chaitanyakharche\Desktop\data\raw")
MINUTES_PER_YEAR = 252 * 390
ORIGIN_START, ORIGIN_END = dt.time(10, 0), dt.time(15, 0)
HORIZONS = (30, 60)


# --------------------------------------------------------------------------- loading
_MONTH_CACHE: dict[tuple[str, str], pd.DataFrame | None] = {}


def load_month(symbol: str, year: int, month: int) -> pd.DataFrame | None:
    """One month of 1-minute bars.

    The stock archive is stored PER MONTH (SPY_2023-06.csv.gz), unlike the option archive
    which is per session. Reading it per day would decompress the same file ~22 times, so
    months are cached; over 769 sessions that is the difference between minutes and hours.
    """
    key = (symbol, f"{year:04d}-{month:02d}")
    if key in _MONTH_CACHE:
        return _MONTH_CACHE[key]
    p = RAW / "stock_ohlc_1m" / symbol / str(year) / f"{symbol}_{year:04d}-{month:02d}.csv.gz"
    df = None
    if p.exists():
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        if "timestamp" in df.columns and "close" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()
        else:
            df = None
    # Keep only the most recent few months; the walk is chronological so older months are
    # never revisited, and 769 sessions of full months would otherwise sit in memory.
    if len(_MONTH_CACHE) > 3:
        _MONTH_CACHE.clear()
    _MONTH_CACHE[key] = df
    return df


def load_bars(symbol: str, day: dt.date) -> pd.DataFrame | None:
    """Underlying 1-minute OHLCV for one session, RTH only, ascending."""
    month = load_month(symbol, day.year, day.month)
    if month is None or month.empty:
        return None
    df = month[month.index.date == day]
    if df.empty:
        return None
    return df.between_time("09:30", "15:59")


def load_option_quotes(symbol: str, day: dt.date) -> pd.DataFrame | None:
    """0DTE quotes for one session.

    Rows with zero size on BOTH sides are dropped: the archives contain placeholder rows
    at 09:30 with bid=ask=0 and size 0, which are not tradeable prices and would otherwise
    become an implied volatility of zero.
    """
    p = RAW / "option_quote_1m_0dte" / symbol / str(day.year) / f"{symbol}_{day}.csv.gz"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    if "timestamp" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[(df["bid_size"] > 0) | (df["ask_size"] > 0)]
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])]
    if df.empty:
        return None
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["rel_spread_bp"] = (df["ask"] - df["bid"]) / df["mid"] * 1e4
    return df.sort_values("timestamp")


# ------------------------------------------------------------------- variance pieces
def _rv(logret: np.ndarray) -> float:
    """Annualised realised variance from 1-minute log returns."""
    if logret.size == 0:
        return np.nan
    return float(np.sum(logret ** 2) * (MINUTES_PER_YEAR / logret.size))


def _bipower(logret: np.ndarray) -> float:
    """Jump-robust variation. Separates a genuine vol regime from one 9:31 print."""
    if logret.size < 2:
        return np.nan
    mu = np.sqrt(2.0 / np.pi)
    bv = np.sum(np.abs(logret[1:]) * np.abs(logret[:-1])) / (mu ** 2)
    return float(bv * (MINUTES_PER_YEAR / max(logret.size - 1, 1)))


def _quarticity(logret: np.ndarray) -> float:
    if logret.size == 0:
        return np.nan
    n = logret.size
    return float(n / 3.0 * np.sum(logret ** 4) * (MINUTES_PER_YEAR / n) ** 2)


def _bs_iv_straddle(straddle_mid: float, spot: float, minutes_left: float) -> float:
    """Implied vol from an ATM straddle, via the Brenner-Subrahmanyam approximation.

    ATM straddle ~= 0.7979 * S * sigma * sqrt(T), which inverts in closed form. Exact for
    the ATM forward and accurate to well under a vol point near the money -- and unlike a
    Newton solve it cannot fail to converge on a wide 0DTE quote, which matters when this
    runs unattended over 769 sessions.
    """
    if not (straddle_mid > 0 and spot > 0 and minutes_left > 0):
        return np.nan
    T = minutes_left / MINUTES_PER_YEAR
    return float(straddle_mid / (0.7978845608 * spot * np.sqrt(T)))


def option_state_at(quotes: pd.DataFrame, t: pd.Timestamp, spot: float,
                    minutes_left: float) -> dict:
    """ATM implied variance, skew, smile and spread from quotes at or before `t`."""
    q = quotes[quotes["timestamp"] <= t]
    if q.empty:
        return {}
    q = q[q["timestamp"] >= t - pd.Timedelta(minutes=5)]     # no stale forward-fill
    if q.empty:
        return {}
    last = q.groupby(["strike", "right"], as_index=False).last()
    calls = last[last["right"].str.upper().str.startswith("C")].set_index("strike")
    puts = last[last["right"].str.upper().str.startswith("P")].set_index("strike")
    common = calls.index.intersection(puts.index)
    if len(common) < 3:
        return {}
    strikes = np.asarray(sorted(common), dtype=float)
    atm_k = float(strikes[np.argmin(np.abs(strikes - spot))])
    straddle = float(calls.loc[atm_k, "mid"] + puts.loc[atm_k, "mid"])
    iv = _bs_iv_straddle(straddle, spot, minutes_left)
    if not np.isfinite(iv) or iv <= 0:
        return {}

    out = {
        "iv_atm": iv,
        "iv_var_atm": iv ** 2,
        "straddle_spread_bp": float(calls.loc[atm_k, "rel_spread_bp"]
                                    + puts.loc[atm_k, "rel_spread_bp"]) / 2.0,
        "n_strikes": int(len(strikes)),
    }
    # Wing proxies. True 25-delta needs a full surface solve; on 0DTE the 1%-away strike
    # is a stable stand-in and does not pretend to a precision the quotes cannot support.
    lo = strikes[np.argmin(np.abs(strikes - spot * 0.99))]
    hi = strikes[np.argmin(np.abs(strikes - spot * 1.01))]
    if lo != hi:
        put_wing = _bs_iv_straddle(float(puts.loc[lo, "mid"]) * 2, spot, minutes_left)
        call_wing = _bs_iv_straddle(float(calls.loc[hi, "mid"]) * 2, spot, minutes_left)
        if np.isfinite(put_wing) and np.isfinite(call_wing):
            out["risk_reversal"] = call_wing - put_wing
            out["butterfly"] = (call_wing + put_wing) / 2.0 - iv
    return out


# ------------------------------------------------------------------------- assembly
def _features_at(hist: pd.DataFrame, t: pd.Timestamp, prior: dict) -> dict:
    """Features from `hist`, which the caller guarantees ends at or before `t`."""
    r = np.log(hist["close"]).diff().dropna().to_numpy()
    open_t = t.normalize() + pd.Timedelta(hours=9, minutes=30)
    return {
        "rv_from_open": _rv(r),
        "rv_30m": _rv(r[-30:]),
        "rv_5m": _rv(r[-5:]),
        "bipower_30m": _bipower(r[-30:]),
        "quarticity_30m": _quarticity(r[-30:]),
        "rv_prev_day": prior.get("rv_prev_day", np.nan),
        "rv_prev_5": prior.get("rv_prev_5", np.nan),
        "rv_prev_22": prior.get("rv_prev_22", np.nan),
        "vix_prev_close": prior.get("vix_prev_close", np.nan),
        "minutes_since_open": (t - open_t).total_seconds() / 60.0,
        "minutes_to_close": (t.normalize() + pd.Timedelta(hours=16) - t
                             ).total_seconds() / 60.0,
        "dow": t.dayofweek,
    }


def build_session(symbol: str, day: dt.date, prior: dict) -> pd.DataFrame | None:
    bars = load_bars(symbol, day)
    if bars is None or len(bars) < 120:
        return None
    quotes = load_option_quotes(symbol, day)
    logret_all = np.log(bars["close"]).diff().dropna()

    rows = []
    for t in bars.index:
        if not (ORIGIN_START <= t.time() <= ORIGIN_END):
            continue
        # THE no-lookahead line. `< t` excludes the bar stamped t, which on this archive
        # covers [t, t+60s) and therefore contains information from after the origin.
        hist = bars.loc[bars.index < t]
        if len(hist) < 31:
            continue

        row = {"date": day.isoformat(), "t": t}
        row.update(_features_at(hist, t, prior))

        spot = float(hist["close"].iloc[-1])
        minutes_left = (t.normalize() + pd.Timedelta(hours=16) - t).total_seconds() / 60.0
        row["spot"] = spot
        if quotes is not None:
            row.update(option_state_at(quotes, t, spot, minutes_left))

        # Targets. Forward returns start at the bar AFTER t, so a forecast made at t is
        # never scored against a bar it could have seen.
        fwd = logret_all[logret_all.index > t]
        for h in HORIZONS:
            seg = fwd.iloc[:h].to_numpy()
            row[f"rv_fwd_{h}"] = _rv(seg) if seg.size == h else np.nan
            if row.get("iv_var_atm") and np.isfinite(row[f"rv_fwd_{h}"]):
                row[f"vrp_{h}"] = row["iv_var_atm"] - row[f"rv_fwd_{h}"]
        rows.append(row)

    if not rows:
        return None
    out = pd.DataFrame(rows)
    return out.dropna(subset=[f"rv_fwd_{HORIZONS[0]}"])


def sessions_for(symbol: str) -> list[dt.date]:
    base = RAW / "option_quote_1m_0dte" / symbol
    days = []
    for p in base.rglob(f"{symbol}_*.csv.gz"):
        try:
            days.append(dt.date.fromisoformat(p.stem.replace(f"{symbol}_", "")
                                              .replace(".csv", "")))
        except ValueError:
            continue
    return sorted(days)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--out", default="data/vrp")
    ap.add_argument("--limit", type=int, default=0, help="first N sessions (smoke test)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    days = sessions_for(args.symbol)
    if args.limit:
        days = days[:args.limit]
    if not days:
        print(f"no 0DTE sessions found for {args.symbol} under {RAW}")
        return 1
    print(f"{len(days)} sessions {days[0]} .. {days[-1]}")

    prior_rv: list[float] = []
    manifest, kept = [], 0
    for i, day in enumerate(days, 1):
        prior = {
            "rv_prev_day": prior_rv[-1] if prior_rv else np.nan,
            "rv_prev_5": float(np.mean(prior_rv[-5:])) if len(prior_rv) >= 5 else np.nan,
            "rv_prev_22": float(np.mean(prior_rv[-22:])) if len(prior_rv) >= 22 else np.nan,
        }
        frame = build_session(args.symbol, day, prior)
        if frame is None or frame.empty:
            manifest.append({"date": day.isoformat(), "rows": 0, "reason": "no usable data"})
            continue
        frame.to_parquet(out / f"{args.symbol}_{day}.parquet", index=False)
        kept += 1
        manifest.append({"date": day.isoformat(), "rows": int(len(frame)),
                         "has_iv": bool(frame.get("iv_atm", pd.Series(dtype=float))
                                        .notna().any())})
        bars = load_bars(args.symbol, day)
        if bars is not None:
            prior_rv.append(_rv(np.log(bars["close"]).diff().dropna().to_numpy()))
        if i % 25 == 0:
            print(f"  {i}/{len(days)}  kept {kept}")

    (out / "manifest.json").write_text(
        json.dumps({"symbol": args.symbol, "sessions": manifest}, indent=1),
        encoding="utf-8")
    with_iv = sum(1 for m in manifest if m.get("has_iv"))
    print(f"\nwrote {kept} session files to {out}")
    print(f"  {with_iv} have usable option quotes (needed for the VRP target)")
    print("  sessions with 0 rows are listed in manifest.json with a reason -- a silent "
          "drop is how a sample becomes unrepresentative without anyone noticing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
