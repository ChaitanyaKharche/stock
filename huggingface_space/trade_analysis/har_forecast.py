"""A HAR volatility forecast, fitted per symbol on the bars the app already fetches.

This replaces nothing and votes on nothing. It exists because the app previously carried a
398,854-parameter TFT that had collapsed to its prior -- it returned the same answer for
every symbol -- and a six-parameter linear model that demonstrably responds to its input is
a better thing to show than a large one that does not.

WHAT IT FORECASTS
-----------------
Next session's variance, as an annualised volatility, which is directly comparable to the
implied vol already displayed from the option chain. The difference between the two is the
**variance risk premium**: the amount the option market charges above what the underlying
has historically gone on to do.

Measured on 618 sessions of SPY 0DTE data (research/vrp_cost_model_results.md), that
premium is real and strongly significant -- and 1.84x too small to trade after the bid-ask
spread. So this number is shown as information, never as a signal.

WHY GARMAN-KLASS AND NOT CLOSE-TO-CLOSE
---------------------------------------
A squared close-to-close return is one observation per day and is a very noisy variance
estimate. Garman-Klass uses the full daily OHLC and is roughly 7x more efficient, which
matters because the whole model has only four parameters and ~500 daily observations to fit
them on. Using the noisier proxy would mostly fit noise.

The intraday alternative -- realised variance from the 15-minute bars -- is a better
estimator still, but the app only fetches 60 days of them, which is not enough history for
a 22-day lag plus a fit. Mixing the two estimators across the sample would be worse than
either: two different measurement definitions in one regression.

WHY HAR
-------
The daily / weekly / monthly lag structure (Corsi) is the standard baseline in the
realised-volatility literature and it is hard to beat. In this project's own measurement it
beats naive persistence at t=+7.82 over 250 sessions -- and is itself beaten by the option
market's implied variance at p=0.0222. Both facts are reported to the user, because a demo
that hides the second one is selling something.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
TRADING_DAYS = 252
MIN_OBS = 60            # below this a 4-parameter fit is not worth reporting
LOG2 = np.log(2.0)


def garman_klass(df: pd.DataFrame) -> pd.Series:
    """Annualised daily variance: Garman-Klass intraday PLUS the overnight jump.

    The overnight term is not a refinement, it is a correctness fix. Garman-Klass measures
    only the intraday range and is blind to the close-to-open gap, so on its own it
    systematically UNDERSTATES full-session variance. Option implied volatility prices
    close-to-close INCLUDING overnight, so comparing bare GK against implied would inflate
    the apparent variance risk premium by exactly the part of the risk the estimator cannot
    see -- and overnight gaps are a large share of daily variance for single names.

        var = GK(open, high, low, close)  +  ln(open_t / close_{t-1})^2

    The gap term uses the PREVIOUS close, so row t still depends only on data available by
    the end of day t.
    """
    cols = {c.lower(): c for c in df.columns}
    need = ("open", "high", "low", "close")
    if not all(k in cols for k in need):
        return pd.Series(dtype=float)
    o, h, l, c = (df[cols[k]].astype(float) for k in need)
    ok = (o > 0) & (h > 0) & (l > 0) & (c > 0) & (h >= l)
    intraday = 0.5 * np.log(h / l) ** 2 - (2.0 * LOG2 - 1.0) * np.log(c / o) ** 2
    overnight = np.log(o / c.shift(1)) ** 2
    var = intraday + overnight.fillna(0.0)
    var = var.where(ok & (var > 0))
    return (var * TRADING_DAYS).dropna()


def _design(v: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HAR design matrix in log space, plus the row for forecasting the NEXT day.

    Lags are built with `shift(1)` so row t uses only days strictly before t. The final
    row -- built from the last observed day inclusive -- is the out-of-sample forecast
    point, and is deliberately excluded from the fit.
    """
    d = np.log(np.maximum(v, EPS))
    daily = d.shift(1)
    weekly = d.rolling(5).mean().shift(1)
    monthly = d.rolling(22).mean().shift(1)
    frame = pd.concat({"y": d, "d": daily, "w": weekly, "m": monthly}, axis=1).dropna()
    X = np.column_stack([np.ones(len(frame)), frame["d"], frame["w"], frame["m"]])
    y = frame["y"].to_numpy()
    # The forecast row uses the most recent day INCLUSIVE, which is exactly what shift(1)
    # would supply to a row one step past the end of the sample.
    x_next = np.array([1.0, d.iloc[-1], d.iloc[-5:].mean(), d.iloc[-22:].mean()])
    return X, y, x_next


def forecast(daily_bars: pd.DataFrame, implied_vol_pct: float | None = None) -> dict:
    """Fit HAR on this symbol's own daily bars and forecast next session's volatility.

    Fitted per symbol rather than shipping SPY coefficients: the HAR lag structure is
    stable across assets but its coefficients are not, and applying SPY's to a single
    name would be a claim nobody checked.

    Never raises. A broken forecast must not take the app's signal down with it.
    """
    out: dict = {"available": False}
    try:
        if daily_bars is None or daily_bars.empty:
            out["reason"] = "no daily bars"
            return out
        v = garman_klass(daily_bars)
        if len(v) < MIN_OBS:
            out["reason"] = f"only {len(v)} usable daily bars, need {MIN_OBS}"
            return out

        X, y, x_next = _design(v)
        if len(y) < 30:
            out["reason"] = f"only {len(y)} fittable rows after lags"
            return out
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        # LOGNORMAL BIAS CORRECTION. exp(x @ beta) is the conditional MEDIAN of a
        # log-space fit, not the mean: E[exp(Z)] = exp(E[Z] + s2/2). Without the exp(s2/2)
        # factor the forecast comes out systematically LOW -- here about 40% below an
        # independent close-to-close estimate, which is how the omission was noticed.
        #
        # The same bug shipped in trade_analysis/hpc/har_baseline.py and was enough to
        # turn "implied variance beats HAR, p=0.023" into a result that vanishes once
        # corrected (p=0.583). It is not a rounding detail.
        resid_var = float(np.var(y - X @ beta, ddof=X.shape[1]))
        var_hat = float(np.exp(x_next @ beta) * np.exp(resid_var / 2.0))

        # In-sample fit quality, reported so the number can be discounted. Compared
        # against the naive "tomorrow looks like today" forecast, which is the thing a
        # model has to beat before it is worth any attention at all.
        fitted = np.exp(X @ beta) * np.exp(resid_var / 2.0)
        actual = np.exp(y)
        naive = np.exp(X[:, 1])
        ql = lambda a, f: float(np.mean(a / np.maximum(f, EPS)
                                        - np.log(a / np.maximum(f, EPS)) - 1.0))
        out.update({
            "available": True,
            "forecast_vol_pct": round(np.sqrt(max(var_hat, 0.0)) * 100, 2),
            "n_days_fitted": int(len(y)),
            "lognormal_correction": round(float(np.exp(resid_var / 2.0)), 4),
            "qlike_har": round(ql(actual, fitted), 4),
            "qlike_naive": round(ql(actual, naive), 4),
            "beats_naive": bool(ql(actual, fitted) < ql(actual, naive)),
            "last_realised_vol_pct": round(float(np.sqrt(v.iloc[-1]) * 100), 2),
        })
        if implied_vol_pct:
            prem = float(implied_vol_pct) - out["forecast_vol_pct"]
            out["implied_vol_pct"] = round(float(implied_vol_pct), 2)
            out["premium_vol_pts"] = round(prem, 2)
            out["premium_note"] = (
                "Options priced ABOVE the forecast -- the usual case, and the variance "
                "risk premium." if prem > 0 else
                "Options priced BELOW the forecast, which is unusual.")
    except Exception as exc:                                 # noqa: BLE001
        out["reason"] = f"{type(exc).__name__}: {exc}"
    return out
