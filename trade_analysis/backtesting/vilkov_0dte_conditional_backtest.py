"""
Buy-only 0DTE backtest on REAL option quotes, conditioned on a price-action
breakout - the one thing Vilkov's replication package does not test.

WHY THIS FILE EXISTS
--------------------
Every previous options backtest in this package priced the option itself with
Black-Scholes off a realized-vol proxy (options_premium_backtest.py and its
two variants). That has two known defects: a static IV that never expands
when the move actually happens, and a guessed bid/ask. It also ran on ~30
trades, which per Lo (2002) gives a +/-0.36 confidence interval on per-trade
Sharpe - i.e. no information at all.

This file replaces the synthetic pricing with the real SPXW 0DTE option panel
from Grigory Vilkov's replication package for "0DTE Trading Rules: Tail Risk,
Implementation, and Tactical Timing" (SSRN 4641356), github.com/vilkovgr/
0dte-strategies. Real mids, real bid/ask spreads, real IV, real Greeks,
1,919 sessions instead of 30.

WHAT IS GENUINELY NEW HERE (vs. what Vilkov already published)
-------------------------------------------------------------
His conditional analysis (docs/agent-context/method.md) conditions on the
10:00 ET *implied state* - IV, implied skew, surface slopes - plus lagged
realized moments and lagged strategy PNL. There is no price-action signal in
his feature set. And his strategy panel covers exactly 7 structures:
strangle, iron condor, risk reversal, bull call spread, call/put ratio
spread, bear put spread. Every one needs a short leg or is a spread, so NONE
of them are tradeable under the buy-only constraint.

So: naked single-leg long, conditioned on a prior-day-high/low breakout, is
untested on this data. That is what this file measures.

ISOLATION CONTRACT
------------------
Imports only from options_premium_backtest.py (daily fetch, ATR, constants)
and paths.py. Edits nothing. Writes its own CSV. Delete this one file and
nothing else in the package changes.

HONEST LIMITS - read before trusting any number this produces
-------------------------------------------------------------
1. ENTRY TIME IS 10:00 ET, NOT 9:45. The panel's earliest bar of the day is
   10:00 (quote_time has exactly 13 values: 10:00-15:30 in 30-min steps plus
   a 16:00 row). The live strategy confirms a retest between 9:30 and 9:45.
   That entry is physically unrepresentable here. 10:00 is a strictly worse
   entry than 9:45 on a trending day, so this is a conservative proxy, not an
   equivalent one.

2. 30-MINUTE EXIT GRID. A +25%/+50% premium target is checked at 30-minute
   marks, so an intrabar touch that reverses before the next mark is missed.
   Same class of approximation flagged in options_premium_backtest_scalp.py,
   coarser here.

3. THE PANEL IS CONSTANT-MONEYNESS, NOT CONSTANT-STRIKE. mnes_rel is
   strike/spot *at that bar* - verified: the implied strike for mnes_rel=1.000
   drifts every bar with spot. So you cannot follow one contract by holding
   mnes fixed. reprice_fixed_strike() reconstructs the real contract by
   recomputing K/S_t2 and interpolating the 0.001-step grid. Where K/S_t2
   leaves [0.98, 1.02] the quote does not exist and the value is floored at
   intrinsic - which UNDERSTATES a deep-ITM winner. Censoring frequency is
   measured and reported, not hidden.

4. UNDERLYING IS SPX, NOT SPY/QQQ. SPX and SPY track closely (SPY ~ SPX/10)
   but they are not identical instruments: SPXW is European and cash-settled,
   SPY options are American and physically settled, and SPY has an early-
   exercise/dividend wrinkle SPX does not. QQQ is not covered at all, so
   nothing here validates QQQ.

5. SAMPLE ENDS 2024-05-01. The README and paper both say the sample runs to
   January 2026; the shipped data_opt.parquet stops 2024-05-01 and holds
   1,369,301 rows, not the ~3.5M the README claims. Verified by direct
   inspection. Do not cite the README's figures.

6. Prior-day high/low come from ^GSPC daily bars (the index itself, so
   auto_adjust is a no-op). Using SPY here would be a bug: dividend
   back-adjustment shifts historical highs/lows and corrupts breakout levels.

DATA LICENSE
------------
Vilkov's code is MIT; the data panels are "provided for academic replication
only; redistribution of raw exchange data is not permitted." The parquets
live OUTSIDE this repo (default C:/Users/<you>/Documents/research_data) so
they cannot be committed by accident. Do not copy them into this project.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .options_premium_backtest import (
    fetch_daily, compute_atr, ATR_PERIOD, GAP_ATR_MULT_THRESHOLD,
)
from ..paths import LOGS_DIR

# --- where the cloned replication package lives (override with an env var) ---
VILKOV_DIR = Path(os.environ.get(
    "VILKOV_0DTE_DIR",
    Path.home() / "Documents" / "research_data" / "0dte-strategies",
))
PANEL_PATH = VILKOV_DIR / "data" / "data_opt.parquet"

SPX_TICKER = "^GSPC"          # the panel's own underlying, so the join is exact
ENTRY_TIME = pd.Timestamp("10:00:00").time()
SETTLE_ROW_TIME = pd.Timestamp("16:00:00").time()   # excluded, see _load_panel

MNES_LO, MNES_HI = 0.98, 1.02   # the panel's moneyness band

# The live rule buys ~$1.00-1.50 of SPY premium. Expressed relative to spot so
# it means the same thing across a sample where SPY went from ~$220 to ~$630:
# $1.25 / $630 = 0.00198 of spot. Scale-invariant translation of "a cheap OTM
# option", which is what the rule actually is.
TARGET_REL_PREMIUM = 0.00198
FEE_BP = 0.5                   # matches Vilkov's implementation-cost tier

PROFIT_TARGET_GRID = [0.25, 0.35, 0.50]   # the user's stated 25-50% band
STOP_PCT = -0.50               # ASSUMPTION - the user never stated a stop


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def _load_panel():
    """Option panel with the 16:00 row dropped and settlement recovered.

    The 16:00 row is excluded deliberately. For every bar from 10:00 to 15:30,
    sret * active_underlying_price recovers ONE consistent settlement level
    (verified to the cent on sampled dates). The 16:00 row implies a different
    level, so it is not a bar of the same expiry's life - it is an end-of-day
    snapshot. Including it would silently corrupt the settlement identity.
    """
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Vilkov 0DTE panel not found at {PANEL_PATH}.\n"
            f"Clone it with:  git clone https://github.com/vilkovgr/0dte-strategies.git\n"
            f"then set VILKOV_0DTE_DIR to the checkout if it isn't at the default path."
        )
    df = pd.read_parquet(PANEL_PATH)
    df = df[df["quote_time"] != SETTLE_ROW_TIME].copy()
    # settlement level, recoverable because payoff = max(sret - mnes_rel, 0)
    df["settle"] = df["sret"] * df["active_underlying_price"]
    return df


def _daily_levels():
    """^GSPC daily OHLC -> prior-session high/low and ATR for the breakout test."""
    daily = fetch_daily(SPX_TICKER, period="10y")
    daily = daily[["Open", "High", "Low", "Close"]].dropna()
    daily["atr"] = compute_atr(daily, ATR_PERIOD)
    daily["prior_high"] = daily["High"].shift(1)
    daily["prior_low"] = daily["Low"].shift(1)
    daily["prior_close"] = daily["Close"].shift(1)
    daily["prior_atr"] = daily["atr"].shift(1)      # shift: no same-day lookahead
    daily.index = pd.to_datetime(daily.index).tz_localize(None).normalize()
    return daily


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def validate(panel=None, daily=None):
    """Independent checks that the panel means what the docs claim. Run this
    before trusting any backtest built on top of it."""
    panel = _load_panel() if panel is None else panel
    daily = _daily_levels() if daily is None else daily

    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)
    print(f"rows                : {len(panel):,}")
    print(f"sessions            : {panel['quote_date'].nunique():,}")
    print(f"date range          : {panel['quote_date'].min().date()} -> "
          f"{panel['quote_date'].max().date()}")
    print(f"intraday bar times  : {len(panel['quote_time'].unique())} "
          f"({min(panel['quote_time'])} -> {max(panel['quote_time'])})")

    # 1. is the recovered settlement internally consistent within a date?
    spread = panel.groupby("quote_date")["settle"].agg(lambda s: s.max() - s.min())
    print(f"\nsettle self-consistency within a session (max-min, SPX points):")
    print(f"  median {spread.median():.6f} | 99th pct {spread.quantile(0.99):.6f} "
          f"| max {spread.max():.6f}")

    # 2. does that settlement equal the SAME day's actual SPX close?
    #    This is the decisive same-day-expiry (0DTE) test - it does not rely on
    #    recalling Cboe's SPXW listing schedule.
    settle = panel.groupby("quote_date")["settle"].median().rename("settle")
    joined = pd.concat([settle, daily["Close"]], axis=1, join="inner").dropna()
    err = (joined["settle"] - joined["Close"]).abs()
    rel = err / joined["Close"]
    print(f"\nrecovered settle vs same-day ^GSPC close  (n={len(joined):,}):")
    print(f"  median abs err {err.median():.4f} pts ({rel.median()*1e4:.3f} bp)")
    print(f"  95th pct       {err.quantile(0.95):.4f} pts")
    print(f"  share within 0.1%: {(rel < 0.001).mean()*100:.2f}%")
    if (rel < 0.001).mean() > 0.95:
        print("  -> CONFIRMED same-day expiry: settlement is that session's close.")
    else:
        print("  -> WARNING: settlement does NOT match same-day close. Investigate "
              "before using; the panel may not be 0DTE throughout.")

    # 3. weekday coverage by year - pre-2022 SPXW did not expire every weekday
    print("\nsessions by year and weekday (0=Mon):")
    wd = panel.groupby([panel["quote_date"].dt.year,
                        panel["quote_date"].dt.weekday])["quote_date"].nunique()
    print(wd.unstack(fill_value=0).to_string())
    print("\n  NOTE: if pre-2022 years show all five weekdays, that is worth "
          "reconciling against\n  Cboe's SPXW listing history (Tue/Thu 0DTE only "
          "arrived in 2022). The settle\n  test above is the authority on whether "
          "these are genuinely same-day.")
    return panel, daily


# ----------------------------------------------------------------------------
# entry detection
# ----------------------------------------------------------------------------
def find_breakout_entries(panel, daily):
    """One row per session: the 10:00 ET state plus whether the prior session's
    high/low had been broken by then.

    This is an ORB-family signal, not the live rule. The live rule needs a
    9:30-9:45 retest; the panel starts at 10:00. Documented in the module
    docstring as limit (1).
    """
    at_entry = panel[panel["quote_time"] == ENTRY_TIME]
    spot = at_entry.groupby("quote_date")["active_underlying_price"].first()
    settle = panel.groupby("quote_date")["settle"].median()

    rows = []
    for date, s10 in spot.items():
        if date not in daily.index:
            continue
        d = daily.loc[date]
        if pd.isna(d["prior_high"]) or pd.isna(d["prior_atr"]) or d["prior_atr"] <= 0:
            continue

        if s10 > d["prior_high"]:
            direction, level = "UP", d["prior_high"]
        elif s10 < d["prior_low"]:
            direction, level = "DOWN", d["prior_low"]
        else:
            direction, level = None, np.nan

        # Rosa (2022): the intraday-momentum effect survives out of sample only
        # with a THRESHOLD on signal strength, so record displacement magnitude.
        displacement_atr = (abs(s10 - level) / d["prior_atr"]
                            if direction else np.nan)
        gap = abs(d["Open"] - d["prior_close"])

        rows.append({
            "date": date,
            "direction": direction,
            "spot_10am": float(s10),
            "settle": float(settle.get(date, np.nan)),
            "breakout_level": float(level) if direction else np.nan,
            "displacement_atr": float(displacement_atr) if direction else np.nan,
            "prior_atr": float(d["prior_atr"]),
            "gap_atr": float(gap / d["prior_atr"]),
            "gap_regime": "ELEVATED" if gap > GAP_ATR_MULT_THRESHOLD * d["prior_atr"] else "NORMAL",
            # overnight-to-10:00 return: Gao et al's r1 analogue (from prior close)
            "r1": float(s10 / d["prior_close"] - 1),
        })

    entries = pd.DataFrame(rows).set_index("date").sort_index()

    # Gao/Han/Li/Zhou (2018): predictability is concentrated in the high-
    # volatility tercile (R2 3.3% vs 0.6%, and r1 insignificant when quiet).
    # Terciles are assigned on a 60-session TRAILING window - a full-sample
    # tercile would leak future information into a historical label.
    rv = np.log(daily["Close"]).diff().rolling(20).std() * np.sqrt(252)
    rv = rv.shift(1).reindex(entries.index)          # shift: prior-session info only
    q33 = rv.rolling(60, min_periods=30).quantile(0.33)
    q67 = rv.rolling(60, min_periods=30).quantile(0.67)
    entries["realized_vol"] = rv
    entries["vol_regime"] = np.where(rv <= q33, "LOW",
                             np.where(rv >= q67, "HIGH", "MID"))
    entries.loc[rv.isna() | q33.isna(), "vol_regime"] = "UNKNOWN"
    return entries


# ----------------------------------------------------------------------------
# option selection and repricing
# ----------------------------------------------------------------------------
def build_slices(panel):
    """{(date, option_type): {quote_time: (mnes_rel[], mid[], bas[], spot)}}

    Built once and reused across every variant. Filtering the 1.26M-row panel
    per date inside the trade loop instead would rescan it hundreds of times
    per variant, which dominates the runtime for no reason.
    """
    cache = {}
    cols = ["quote_time", "mnes_rel", "mid", "bas", "active_underlying_price"]
    for (date, otype), g in panel.groupby(["quote_date", "option_type"], sort=False):
        per_time = {}
        for t, gt in g[cols].groupby("quote_time", sort=False):
            gt = gt.sort_values("mnes_rel")
            per_time[t] = (gt["mnes_rel"].to_numpy(), gt["mid"].to_numpy(),
                           gt["bas"].to_numpy(),
                           float(gt["active_underlying_price"].iloc[0]))
        cache[(date, otype)] = per_time
    return cache


def select_contract(mnes, mid, option_type, target_rel_premium=TARGET_REL_PREMIUM):
    """Pick the OTM strike whose spot-relative mid is closest to target.

    Restricted to the OTM side so a coincidentally similar ITM price can't be
    chosen. Returns (index, achieved_premium, hit_target) - hit_target is False
    when the band cannot reach the target (happens on high-vol days, where even
    the 2%-OTM wing is still richer than the target), which is recorded rather
    than dropped.
    """
    otm = mnes >= 1.0 if option_type == "C" else mnes <= 1.0
    if not otm.any():
        return None, np.nan, False
    idx = np.where(otm)[0]
    j = idx[np.argmin(np.abs(mid[idx] - target_rel_premium))]
    achieved = float(mid[j])
    # "hit" = within 25% of the requested premium
    return int(j), achieved, bool(abs(achieved - target_rel_premium) <= 0.25 * target_rel_premium)


def reprice_fixed_strike(strike, slices, t, option_type, ref_spot):
    """Value the SAME contract (fixed strike K) at a later bar.

    The panel is constant-moneyness, so K's moneyness at t is K/S_t and must be
    looked up by interpolating the 0.001-step grid.

    UNITS MATTER HERE. Every `mid` in the panel is divided by the spot of ITS
    OWN bar, so a mid at 14:00 and a mid at 10:00 are in different units and
    cannot be divided by each other to get a return. Everything returned here
    is rescaled by spot_t/ref_spot into the entry bar's units so the caller can
    compare directly against the entry fill. Skipping this leaks the underlying
    move into the premium return - understating calls and flattering puts.

    Returns (mid, bas, censored); censored=True means K/S_t left [0.98, 1.02],
    where no quote exists and the value is floored at intrinsic - which
    understates a deep-ITM winner.
    """
    mnes, mid, bas, spot = slices[t]
    m = strike / spot
    scale = spot / ref_spot
    if MNES_LO <= m <= MNES_HI:
        return (float(np.interp(m, mnes, mid)) * scale,
                float(np.interp(m, mnes, bas)) * scale, False)
    # outside the quoted band: intrinsic only, in this bar's units then rescaled
    intrinsic = max(1.0 - m, 0.0) if option_type == "C" else max(m - 1.0, 0.0)
    edge_bas = float(bas[-1] if m > MNES_HI else bas[0])
    return intrinsic * scale, edge_bas * scale, True


# ----------------------------------------------------------------------------
# trade simulation
# ----------------------------------------------------------------------------
def simulate(entries, slices_cache, profit_target=None, stop_pct=STOP_PCT):
    """One trade per breakout session.

    Two P&L columns are produced and they answer different questions:

      ret_hold_pct  - buy at 10:00, hold to cash settlement. Uses the panel's
                      own settlement payoff. NO exit-path reconstruction, NO
                      profit-target assumption, NO stop assumption. This is the
                      assumption-free headline number.

      ret_target_pct- buy at 10:00, sell at the first 30-min bar where the
                      real mid crosses +profit_target or -stop. Requires the
                      fixed-strike reconstruction and both exit assumptions,
                      so it is the softer of the two.

    Both are net of the real half-spread on entry and exit plus FEE_BP.
    """
    trades = []
    for date, e in entries.dropna(subset=["direction"]).iterrows():
        option_type = "C" if e["direction"] == "UP" else "P"
        slices = slices_cache.get((date, option_type))
        if not slices or ENTRY_TIME not in slices:
            continue

        mnes, mid, bas, spot = slices[ENTRY_TIME]
        j, entry_mid, hit_target_premium = select_contract(mnes, mid, option_type)
        if j is None or not np.isfinite(entry_mid) or entry_mid <= 0:
            continue

        moneyness = float(mnes[j])
        strike = moneyness * spot
        fee = FEE_BP / 1e4
        entry_fill = entry_mid + bas[j] / 2 + fee          # pay the ask, plus fee

        # --- (a) hold to settlement: exact, no reconstruction ---
        settle_ratio = e["settle"] / spot
        payoff = (max(settle_ratio - moneyness, 0.0) if option_type == "C"
                  else max(moneyness - settle_ratio, 0.0))
        ret_hold = (payoff - entry_fill) / entry_fill * 100

        # --- (b) profit-target exit: needs fixed-strike reconstruction ---
        ret_target, exit_time, exit_reason, censored_any = np.nan, None, None, False
        if profit_target is not None:
            later = sorted(t for t in slices if t > ENTRY_TIME)
            for t in later:
                m2, b2, censored = reprice_fixed_strike(strike, slices, t,
                                                        option_type, spot)
                censored_any |= censored
                exit_fill = max(m2 - b2 / 2 - fee, 0.0)
                change = exit_fill / entry_fill - 1
                if change >= profit_target:
                    ret_target, exit_time, exit_reason = change * 100, t, "TARGET"
                    break
                if stop_pct is not None and change <= stop_pct:
                    ret_target, exit_time, exit_reason = change * 100, t, "STOP"
                    break
            if exit_reason is None:
                ret_target, exit_time, exit_reason = ret_hold, "settle", "SETTLE"

        trades.append({
            "date": date, "direction": e["direction"], "option_type": option_type,
            "vol_regime": e["vol_regime"], "gap_regime": e["gap_regime"],
            "displacement_atr": e["displacement_atr"], "r1": e["r1"],
            "moneyness": round(moneyness, 4), "strike": round(strike, 2),
            "entry_rel_premium": round(entry_mid, 6),
            "premium_target_hit": hit_target_premium,
            "ret_hold_pct": round(ret_hold, 2),
            "ret_target_pct": round(ret_target, 2) if np.isfinite(ret_target) else np.nan,
            "exit_time": str(exit_time) if exit_time else None,
            "exit_reason": exit_reason,
            "censored": censored_any,
        })
    return pd.DataFrame(trades)


# ----------------------------------------------------------------------------
# statistics that respect sample size
# ----------------------------------------------------------------------------
def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def describe(returns, label):
    """Mean CI, Wilson win-rate CI, and Sharpe with Lo (2002) standard error.
    Reported because a point estimate on a small subsample is not a finding."""
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 2:
        return {"segment": label, "n": n}
    mean, sd = r.mean(), r.std(ddof=1)
    se = sd / np.sqrt(n)
    wins = int((r > 0).sum())
    sr = mean / sd if sd > 0 else np.nan
    sr_se = np.sqrt((1 + sr**2 / 2) / n) if np.isfinite(sr) else np.nan
    lo, hi = _wilson(wins, n)
    return {
        "segment": label, "n": n,
        "win_rate": round(wins / n * 100, 1),
        "win_ci_lo": round(lo, 1), "win_ci_hi": round(hi, 1),
        "mean_ret_pct": round(mean, 2),
        "mean_ci_lo": round(mean - 1.96 * se, 2),
        "mean_ci_hi": round(mean + 1.96 * se, 2),
        "median_ret_pct": round(r.median(), 2),
        "sharpe_per_trade": round(sr, 3) if np.isfinite(sr) else np.nan,
        "sharpe_se": round(sr_se, 3) if np.isfinite(sr_se) else np.nan,
        "significant": bool(abs(mean) > 1.96 * se),
    }


if __name__ == "__main__":
    import sys

    panel, daily = validate()
    entries = find_breakout_entries(panel, daily)

    n_break = entries["direction"].notna().sum()
    print("\n" + "=" * 78)
    print(f"ENTRIES: {len(entries):,} sessions | {n_break:,} with a 10:00 breakout "
          f"({n_break/len(entries)*100:.1f}%)")
    print(entries["direction"].value_counts(dropna=False).to_string())

    if "--validate" in sys.argv:
        sys.exit(0)

    summary, all_trades = [], []
    print("\nbuilding per-session option slices...")
    slices_cache = build_slices(panel)

    # --- headline: assumption-free hold-to-settlement ---
    held = simulate(entries, slices_cache, profit_target=None)
    held["variant"] = "hold_to_settle"
    all_trades.append(held)
    print("\n" + "=" * 78)
    print("A) BUY AT 10:00 ON BREAKOUT, HOLD TO SETTLEMENT (no exit assumptions)")
    print("=" * 78)
    print(f"censored (K left the 0.98-1.02 band): "
          f"{held['censored'].mean()*100:.1f}% | requested premium reachable: "
          f"{held['premium_target_hit'].mean()*100:.1f}%")

    summary.append(describe(held["ret_hold_pct"], "hold | all breakouts"))
    for reg in ["HIGH", "MID", "LOW"]:
        seg = held[held["vol_regime"] == reg]
        summary.append(describe(seg["ret_hold_pct"], f"hold | vol={reg}"))
    for thr in [0.0, 0.25, 0.5, 1.0]:
        seg = held[held["displacement_atr"] >= thr]
        summary.append(describe(seg["ret_hold_pct"], f"hold | displacement>={thr}atr"))
    for d in ["UP", "DOWN"]:
        seg = held[held["direction"] == d]
        summary.append(describe(seg["ret_hold_pct"], f"hold | {d}"))
    # calls-only cross-cuts: the UP side is the only one not significantly bad,
    # so this is where a conditional filter has anything to work with
    ups = held[held["direction"] == "UP"]
    for reg in ["HIGH", "MID", "LOW"]:
        summary.append(describe(ups[ups["vol_regime"] == reg]["ret_hold_pct"],
                                f"hold | UP & vol={reg}"))
    for thr in [0.25, 0.5, 1.0]:
        summary.append(describe(ups[ups["displacement_atr"] >= thr]["ret_hold_pct"],
                                f"hold | UP & displacement>={thr}atr"))

    # --- secondary: the profit-target scalp, with and WITHOUT a stop ---
    # The user's stated rule is "exit at 25-50% profit" and says nothing about a
    # stop, so the no-stop column is the faithful one; the -50% stop column is
    # this file's assumption and is labelled as such.
    print("\n" + "=" * 78)
    print("B) PROFIT-TARGET EXIT ON THE 30-MIN GRID (real mids)")
    print("=" * 78)
    for pt in PROFIT_TARGET_GRID:
        for stop, tag in [(STOP_PCT, f"stop{int(STOP_PCT*100)}"), (None, "nostop")]:
            tdf = simulate(entries, slices_cache, profit_target=pt, stop_pct=stop)
            tdf["variant"] = f"target_{int(pt*100)}pct_{tag}"
            all_trades.append(tdf)
            lbl = f"target {int(pt*100)}% {tag}"
            summary.append(describe(tdf["ret_target_pct"], f"{lbl} | all"))
            summary.append(describe(tdf[tdf["direction"] == "UP"]["ret_target_pct"],
                                    f"{lbl} | UP"))
            summary.append(describe(tdf[tdf["direction"] == "DOWN"]["ret_target_pct"],
                                    f"{lbl} | DOWN"))
            for reg in ["HIGH", "LOW"]:
                summary.append(describe(tdf[tdf["vol_regime"] == reg]["ret_target_pct"],
                                        f"{lbl} | vol={reg}"))
            print(f"  {lbl}: exit mix {tdf['exit_reason'].value_counts().to_dict()}")

    sdf = pd.DataFrame(summary)
    print("\n" + "=" * 78)
    print("RESULTS  (95% CIs; 'significant' = mean CI excludes zero)")
    print("=" * 78)
    print(sdf.to_string(index=False))

    trades_out = LOGS_DIR / "vilkov_0dte_conditional_trades.csv"
    summary_out = LOGS_DIR / "vilkov_0dte_conditional_summary.csv"
    pd.concat(all_trades, ignore_index=True).to_csv(trades_out, index=False)
    sdf.to_csv(summary_out, index=False)
    print(f"\nSaved trades  -> {trades_out}")
    print(f"Saved summary -> {summary_out}")
    print("\nREMINDER: entry is 10:00 ET, not the live rule's 9:30-9:45 retest; "
          "underlying is\nSPX, not SPY/QQQ. See the module docstring's limits "
          "before acting on any of this.")
