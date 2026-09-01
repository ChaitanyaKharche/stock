"""The ACTUAL manual strategy: 3-level confluence breakout with indicator gates.

WHY THIS EXISTS
---------------
Every prior backtest in this repo tested a stripped-down rule: break yesterday's
REGULAR-HOURS high/low, one level, no confirmation. That is one of six levels and
none of the confirmation the trader actually uses. Its honest-timing result
(QQQ -8.83%, win 68.6% against a 75.3% breakeven) says that simplified rule is
dead. It says nothing about the real one, which is what this file tests.

THE REAL RULE, as specified by the trader
-----------------------------------------
  levels (6, in 3 pairs)  yesterday PREMARKET high/low   (04:00-09:29 prior)
                          yesterday REGULAR  high/low    (09:30-15:59 prior)
                          today     PREMARKET high/low   (04:00-09:29 today)
  trigger                 ALL THREE pairs cleared in the same direction, so the
                          effective line is max(3 highs) for UP, min(3 lows) DOWN
  volume gate             bar volume > its own 20-period EMA
  MACD                    9 / 17 / 9, source (O+H+L+C)/4, EMA oscillator and EMA
                          signal -- the trader's charted settings, NOT 12/26/9
  DMI / ADX               Wilder 14, +DI > -DI for UP, ADX above a floor
  timeframes              5m execution, 10m and 15m agreement
  plus discretion         "hard gates + judgement calls too"

WHAT THIS CAN AND CANNOT MEASURE
--------------------------------
Only the MECHANICAL subset. The discretionary overlay -- "price action", the
1-hour visual check -- is not encoded and cannot be. So this is a FLOOR if the
trader's discretion adds value and a CEILING if discretion is where the losses
come from, and nothing here can tell you which. Quote it with that caveat.

TWO DATA TRAPS, BOTH MEASURED, BOTH HANDLED
-------------------------------------------
1. ZERO-FILLED BARS. ThetaData returns a full grid and zero-fills minutes with no
   trades. Measured inside 09:30-15:59 over 2016-2026: 43,597 such minutes for
   QQQ, 141,707 for SPY. `min()` is the dangerous aggregate -- a single empty
   minute sets the session low to 0.00.
2. MARKET HOLIDAYS. A closed day still returns a full grid of zero-filled bars,
   so it is not `.empty` and survives the usual guard. `find_entries` takes
   dates[i-1] as "yesterday", so every post-holiday session inherited high=0 and
   `price > 0` fired trivially. Measured: 105 fabricated QQQ entries and 98 SPY,
   100% of them UP, ~99% at the 09:30 bar -- 6.8% and 7.0% of all signals.

TIMING
------
Decision time is the bar's CLOSE, never its label. That convention elsewhere in
this repo was worth +13 to +16 points per trade of pure lookahead. `entry_time`
here is always the moment the information actually existed.

ISOLATION: imports only, edits nothing, writes its own CSV.
"""
import argparse
import math
from datetime import time as dt_time

import numpy as np
import pandas as pd

from ..data_sources.thetadata_client import ThetaDataClient
from ..paths import LOGS_DIR

ET = "America/New_York"
PM_START, PM_END = "04:00:00", "09:29:00"
RTH_START, RTH_END = "09:30:00", "15:59:00"

ENTRY_WINDOW_START = dt_time(9, 30)
ENTRY_WINDOW_END = dt_time(9, 45)   # decision must land at or before this

EXEC_TF = 5
CONFIRM_TFS = [10, 15]
VOL_EMA = 20
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 9, 17, 9
ADX_PERIOD = 14
ADX_MIN = 20.0                       # ASSUMPTION - trader did not specify a floor
LOOKBACK_MINUTES = 3000              # ~8 sessions of RTH minutes: enough to warm
                                     # MACD(17) and ADX(14) on a 15m timeframe
MIN_SESSION_VOLUME = 1_000_000       # below this it is a holiday, not a session


# --------------------------------------------------------------------- data
def fetch_window(theta, symbol, start, end, start_time, end_time):
    """Minute bars for a clock window, chunked on CALENDAR MONTH boundaries.

    Month anchoring rather than start+27d is deliberate: the cache key hashes the
    request params, so anchoring chunks to the caller's `start` meant changing the
    start date invalidated the whole cache and re-fetched a decade. Calendar
    anchoring makes every interior chunk reusable across any query window.
    """
    frames = []
    cur = pd.Timestamp(start).normalize().replace(day=1)
    last = pd.Timestamp(end).normalize()
    while cur <= last:
        stop = min((cur + pd.offsets.MonthEnd(1)).normalize(), last)
        payload = theta.get("/stock/history/ohlc", symbol=symbol,
                            start_date=cur.strftime("%Y%m%d"),
                            end_date=stop.strftime("%Y%m%d"),
                            interval="1m", venue="utp_cta",
                            start_time=start_time, end_time=end_time)
        rows = []
        for b in (payload.get("response") or []):
            rows.extend(b.get("data", []) if isinstance(b, dict) and "data" in b else [b])
        if rows:
            frames.append(pd.DataFrame(rows))
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = (pd.to_datetime(df["timestamp"])
                       .dt.tz_localize(ET, nonexistent="shift_forward", ambiguous=True))
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    # TRAP 1 -- filter before anything reads a high or a low
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    return df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                              "close": "Close", "volume": "Volume"})


def real_sessions(rth):
    """Dates with genuine trading volume (TRAP 2)."""
    v = rth.groupby(rth.index.date)["Volume"].sum()
    return set(v[v >= MIN_SESSION_VOLUME].index)


# --------------------------------------------------------------- indicators
def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def macd_ohlc4(df):
    src = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    line = _ema(src, MACD_FAST) - _ema(src, MACD_SLOW)
    return line, _ema(line, MACD_SIGNAL)


def dmi_adx(df, n=ADX_PERIOD):
    h, l, c = df["High"], df["Low"], df["Close"]
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pdi, mdi, dx.ewm(alpha=1 / n, adjust=False).mean()


def resample_tf(rth, minutes):
    """Bars for one timeframe. Index is the bar LABEL; decision time is
    label + `minutes`. Never conflate the two."""
    out = rth.resample(f"{minutes}min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return out.dropna(subset=["Open"])


def with_indicators(bars):
    b = bars.copy()
    b["vol_ema"] = _ema(b["Volume"], VOL_EMA)
    b["macd"], b["macd_sig"] = macd_ohlc4(b)
    b["pdi"], b["mdi"], b["adx"] = dmi_adx(b)
    return b


def _aligned(row, up):
    """Indicator gates for one bar, one timeframe."""
    return {
        "vol": bool(row["Volume"] > row["vol_ema"]),
        "macd": bool(row["macd"] > row["macd_sig"]) if up else bool(row["macd"] < row["macd_sig"]),
        "dmi": (bool(row["pdi"] > row["mdi"]) if up else bool(row["mdi"] > row["pdi"]))
               and bool(row["adx"] >= ADX_MIN),
    }


# --------------------------------------------------------------------- core
def find_signals(symbol, theta, start, end):
    """One row per session that produced a signal, with every gate recorded
    separately so their marginal contribution can be measured rather than
    assumed."""
    pm = fetch_window(theta, symbol, start, end, PM_START, PM_END)
    rth = fetch_window(theta, symbol, start, end, RTH_START, RTH_END)
    if pm.empty or rth.empty:
        return pd.DataFrame()

    good = real_sessions(rth)
    rth = rth[[d in good for d in rth.index.date]]
    days = sorted(good)

    pm_by = {d: g for d, g in pm.groupby(pm.index.date)}
    rth_by = {d: g for d, g in rth.groupby(rth.index.date)}

    # Indicators are computed ONCE on the CONTINUOUS multi-day series, not per
    # session. This matters enormously and was wrong in the first draft: with
    # ewm(adjust=False) the first bar of a session has ema == its own value, so
    # `volume > vol_ema` can never be true at 09:30, and MACD (17 bars to warm
    # up) and ADX (~28) are pure noise three bars into a day. A charting
    # platform carries all of these across the session boundary, which is what
    # the trader actually sees. Overnight gaps produce empty resample buckets
    # that dropna() removes, so no synthetic bars are created.
    exec_all = with_indicators(resample_tf(rth, EXEC_TF))
    conf_all = {tf: with_indicators(resample_tf(rth, tf)) for tf in CONFIRM_TFS}

    rows = []
    for i in range(1, len(days)):
        d, prev = days[i], days[i - 1]
        t_rth, t_pm = rth_by.get(d), pm_by.get(d)
        y_rth, y_pm = rth_by.get(prev), pm_by.get(prev)
        if t_rth is None or t_pm is None or y_rth is None or y_pm is None:
            continue
        if t_rth.empty or t_pm.empty or y_rth.empty or y_pm.empty:
            continue

        highs = {"y_pm": float(y_pm["High"].max()), "y_rth": float(y_rth["High"].max()),
                 "t_pm": float(t_pm["High"].max())}
        lows = {"y_pm": float(y_pm["Low"].min()), "y_rth": float(y_rth["Low"].min()),
                "t_pm": float(t_pm["Low"].min())}
        if min(highs.values()) <= 0 or min(lows.values()) <= 0:
            continue
        up_line, dn_line = max(highs.values()), min(lows.values())

        exec_bars = exec_all[exec_all.index.date == d]
        conf = {tf: cb[cb.index.date == d] for tf, cb in conf_all.items()}

        for label, bar in exec_bars.iterrows():
            decision = label + pd.Timedelta(minutes=EXEC_TF)   # HONEST TIMING
            if decision.time() > ENTRY_WINDOW_END:
                break
            if label.time() < ENTRY_WINDOW_START:
                continue

            close = float(bar["Close"])
            if close > up_line:
                up = True
            elif close < dn_line:
                up = False
            else:
                continue

            g = _aligned(bar, up)
            # Confirmation timeframes read the PARTIAL, in-progress candle as of
            # the decision moment -- which is exactly what is on screen. At 09:35
            # the 10m candle spans 09:30-09:40 and is half formed; requiring a
            # CLOSED 10m bar instead would mean no confirmation can ever exist
            # inside a 15-minute entry window (it collapsed 21 signals to 1).
            # The partial bar contains only data up to `decision`, so this is not
            # lookahead. Lookback is bounded so MACD(17)/ADX(14) are warm without
            # recomputing a decade per signal.
            multi = True
            hist = rth[rth.index <= decision].tail(LOOKBACK_MINUTES)
            for tf in CONFIRM_TFS:
                cb = with_indicators(resample_tf(hist, tf))
                if cb.empty or len(cb) < MACD_SLOW + 2:
                    multi = False
                    break
                cg = _aligned(cb.iloc[-1], up)
                if not (cg["macd"] and cg["dmi"]):
                    multi = False
                    break

            rows.append({
                "date": d, "symbol": symbol,
                "direction": "UP" if up else "DOWN",
                "bar_label": label, "entry_time": decision,
                "entry_price": close,
                "up_line": up_line, "dn_line": dn_line,
                "y_pm_h": highs["y_pm"], "y_rth_h": highs["y_rth"], "t_pm_h": highs["t_pm"],
                "y_pm_l": lows["y_pm"], "y_rth_l": lows["y_rth"], "t_pm_l": lows["t_pm"],
                "gate_vol": g["vol"], "gate_macd": g["macd"], "gate_dmi": g["dmi"],
                "gate_multi_tf": multi,
                "all_gates": g["vol"] and g["macd"] and g["dmi"] and multi,
                "adx": float(bar["adx"]) if pd.notna(bar["adx"]) else np.nan,
            })
            break   # one trade per session

    return pd.DataFrame(rows)


def outcomes(sig, theta, symbol, start, end):
    """Underlying forward return and MFE from the honest decision time.

    Measured on the underlying deliberately: option payoffs have ~53pt SD, the
    underlying has far better SNR, and a signal with no forward content on the
    stock cannot be rescued by any option overlay.
    """
    rth = fetch_window(theta, symbol, start, end, RTH_START, RTH_END)
    by = {d: g for d, g in rth.groupby(rth.index.date)}
    out = []
    for r in sig.itertuples():
        day = by.get(r.date)
        if day is None or day.empty:
            continue
        after = day[day.index >= r.entry_time]
        if len(after) < 2:
            continue
        spot = float(after["Open"].iloc[0])
        up = r.direction == "UP"
        sign = 1.0 if up else -1.0
        out.append({
            "fwd": sign * (float(after["Close"].iloc[-1]) / spot - 1) * 100,
            "mfe": ((float(after["High"].max()) / spot - 1) * 100 if up
                    else (1 - float(after["Low"].min()) / spot) * 100),
            "mae": ((float(after["Low"].min()) / spot - 1) * 100 if up
                    else (1 - float(after["High"].max()) / spot) * 100),
        })
    return pd.concat([sig.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def _st(v):
    v = pd.Series(v).dropna()
    if len(v) < 3:
        return None
    se = v.std() / math.sqrt(len(v))
    return {"n": len(v), "mean": v.mean(), "t": v.mean() / se}


def report(df):
    line = "=" * 92
    print(f"\n{line}\nMULTI-LEVEL CONFLUENCE BREAKOUT - honest timing, mechanical gates only\n{line}")
    print("A 25% option target needs roughly 0.20% of favourable underlying travel,")
    print("and the option breakeven win rate is ~75.3%. Judge MFE>0.20% against that.\n")

    for sym, g in df.groupby("symbol"):
        print(f"\n{sym}: {len(g)} raw 3-level breakout signals")
        print(f"{'gate set':34s} {'n':>5s} {'fwd ret':>9s} {'t':>6s} {'MFE>0.20%':>10s}")
        print("-" * 70)
        combos = [
            ("levels only (no gates)", g),
            ("+ volume > 20 EMA", g[g.gate_vol]),
            ("+ MACD 9/17/9", g[g.gate_vol & g.gate_macd]),
            ("+ DMI/ADX", g[g.gate_vol & g.gate_macd & g.gate_dmi]),
            ("+ 10m/15m agreement  << PRIMARY", g[g.all_gates]),
        ]
        for name, sub in combos:
            s = _st(sub["fwd"])
            if not s:
                print(f"{name:34s} {len(sub):5d}   (too few)")
                continue
            hit = (sub["mfe"] > 0.20).mean() * 100
            print(f"{name:34s} {s['n']:5d} {s['mean']:+8.4f}% {s['t']:+6.2f} {hit:9.1f}%")

        prim = g[g.all_gates]
        if len(prim) > 3:
            print(f"  primary: UP {int((prim.direction=='UP').sum())} / "
                  f"DOWN {int((prim.direction=='DOWN').sum())}, "
                  f"entries/yr ~{len(prim)/max(g.date.map(lambda x: x.year).nunique(),1):.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["QQQ", "SPY"])
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-08-14")
    a = ap.parse_args()

    theta = ThetaDataClient()
    frames = []
    for s in a.symbols:
        # Per-symbol isolation: a vendor hiccup on the second symbol used to
        # discard the first symbol's completed work as well. Long runs over a
        # flaky gateway must degrade, not vanish.
        try:
            sig = find_signals(s, theta, a.start, a.end)
            print(f"{s}: {len(sig)} signals", flush=True)
            if not sig.empty:
                frames.append(outcomes(sig, theta, s, a.start, a.end))
        except Exception as exc:
            print(f"{s}: FAILED - {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        report(df)
        out = LOGS_DIR / "multi_level_orb_signals.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved -> {out}\ntheta API: {theta.stats()}")
