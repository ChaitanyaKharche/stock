"""Can the app emit anything other than HOLD? And does either dropdown do anything?

    venv\Scripts\python.exe huggingface_space\signal_reachability_test.py

Audited 2026-09-07: all 12 timeframe x strategy combinations returned ONE distinct answer,
always HOLD, at a confidence pinned near 25. Four separate defects, each of which alone was
enough to cause it:

  1. min_confidence was 70/65/60/55 against a quantity whose measured MAXIMUM was 47.
     The gate fired on 0 of 80 observations. Not strict -- unreachable.
  2. sentiment abstains almost always, but was scored as 0 confidence rather than excluded,
     permanently spending 30% of the confidence budget on a constant zero.
  3. momentum_score is a MAGNITUDE (every term abs()-wrapped or non-negative) and
     _convert_signal_format hardcoded `return 'CALLS'`. weighted_score could not go
     negative, so PUTS was unreachable for any input, and a collapsing stock read as a
     bullish setup.
  4. the selected timeframe never reached the momentum engine, which hardcoded its
     timeframe weights -- so every timeframe analysed identical data.

These assert all four stay fixed. Synthetic bars are used deliberately: a live-market test
would pass or fail on that morning's tape.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _k in ("FINNHUB_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
    os.environ.setdefault(_k, "unused")

import numpy as np
import pandas as pd

from trade_analysis.enhanced_api import _generate_master_signal
from trade_analysis.indicators import enrich_with_indicators, identify_current_setup
from trade_analysis.momentum_trading_engine import IntegratedMomentumEngine

SENT = {"composite_score": 0.05, "confidence": "LOW"}
ALT = {"vix_level": 14.5, "implied_vol_pct": 60.0}
LLM = {"signal": "HOLD", "confidence": 40}
TFS = ["15m", "1h", "4h", "1d"]


def trend_bars(n: int = 200, drift: float = 0.005, seed: int = 0) -> pd.DataFrame:
    """A clean, decisive trend. drift < 0 falls."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.0004, n))
    high = close * 1.002
    low = close * 0.998
    vol = np.linspace(1e6, 2e6, n)
    vol[-1] *= 3.0                                  # closing volume spike
    idx = pd.date_range("2026-01-02 09:30", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"Open": close, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


def osc_bars(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A mean-reverting tape: price cycles, so RSI sits mid-range and there is no
    extension for the reversal mode to fade."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8 * np.pi, n)
    close = 100.0 + 2.0 * np.sin(t) + rng.normal(0, 0.15, n)
    vol = np.linspace(1e6, 2e6, n)
    vol[-1] *= 3.0
    idx = pd.date_range("2026-01-02 09:30", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"Open": close, "High": close * 1.002, "Low": close * 0.998,
                         "Close": close, "Volume": vol}, index=idx)


def bar_set(drift: float) -> dict:
    return {tf: trend_bars(drift=drift, seed=i)
            for i, tf in enumerate(["15m", "hourly", "4h", "daily"])}


def run(bars, timeframe="15m", strategy="momentum", gap=None):
    eng = IntegratedMomentumEngine()
    mom = eng.generate_enhanced_signal(bars, SENT, ALT, timeframe)
    tech = {tf: identify_current_setup(enrich_with_indicators(d.copy(), tf), tf)
            for tf, d in bars.items()}
    sig = _generate_master_signal(mom, LLM, SENT, None, timeframe, strategy, gap, tech)
    return mom, sig


failures = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  -- ' + detail) if detail else ''}")
    if not cond:
        failures.append(label)


print("1. the confidence gate is reachable at all")
up = bar_set(+0.005)
mom, sig = run(up)
check("a decisive uptrend clears the gate and is not HOLD", sig["signal"] != "HOLD",
      f"signal={sig['signal']} conf={sig['confidence']} score={sig['weighted_score']:+.3f}")
check("...and it is bullish", sig["signal"] == "CALLS", f"got {sig['signal']}")

print("\n2. PUTS is reachable -- the direction bug")
dn = bar_set(-0.005)
mom_d, sig_d = run(dn)
check("a decisive DOWNtrend is not reported as a CALLS setup",
      sig_d["signal"] != "CALLS", f"got {sig_d['signal']}")
check("...it is PUTS", sig_d["signal"] == "PUTS",
      f"signal={sig_d['signal']} score={sig_d['weighted_score']:+.3f}")
check("weighted_score carries a negative sign", sig_d["weighted_score"] < 0,
      f"{sig_d['weighted_score']:+.3f}")
check("engine direction is BEARISH",
      mom_d["momentum_analysis"]["master_signal"].get("direction") == "BEARISH",
      str(mom_d["momentum_analysis"]["master_signal"].get("direction")))

print("\n3. the timeframe selector reaches the analysis")
mixed = {"15m": trend_bars(drift=+0.006, seed=1),
         "hourly": trend_bars(drift=+0.004, seed=2),
         "4h": trend_bars(drift=-0.001, seed=3),
         "daily": trend_bars(drift=-0.005, seed=4)}
scores = {}
for tf in TFS:
    _, s = run(mixed, timeframe=tf)
    scores[tf] = round(s["weighted_score"], 6)
print(f"      weighted_score by timeframe: {scores}")
check("timeframes do not all produce one identical score",
      len(set(scores.values())) > 1, f"{len(set(scores.values()))} distinct")
check("a fast setting differs from the slow one", scores["15m"] != scores["1d"])

print("\n4. the gap strategy responds to a real gap")
_, no_gap = run(up, strategy="gap", gap=0.05)
_, big_gap = run(up, strategy="gap", gap=2.4)
check("a 0.05% move is reported as below the floor",
      "below the 0.5% floor" in no_gap["reasoning"], no_gap["reasoning"])
check("a 2.4% gap qualifies and says so",
      "qualifies" in big_gap["reasoning"], big_gap["reasoning"])
check("the measured gap is returned to the caller",
      big_gap.get("overnight_gap_pct") == 2.4, str(big_gap.get("overnight_gap_pct")))

print()
print("5. a directionless tape stays HOLD (the gates must remain demanding)")
# One case is not a test: direction is a noisy statistic and any single draw can clear a
# threshold by luck, so this fires an ENSEMBLE and asserts the rate. Note the DISTINCT
# seed per bar set -- an earlier version of this test reused one seed for all four, which
# made the timeframes literally identical, forced agreement to 1.0, and guaranteed a fire.
noise_fires, noise_n = 0, 0
for k in range(25):
    flat = {tf: trend_bars(drift=0.0, seed=500 + k * 4 + j)
            for j, tf in enumerate(["15m", "hourly", "4h", "daily"])}
    for tf in TFS:
        noise_n += 1
        if run(flat, timeframe=tf)[1]["signal"] != "HOLD":
            noise_fires += 1
rate = noise_fires / noise_n * 100
check("fires on under 10% of zero-drift tapes", rate < 10.0,
      f"{noise_fires}/{noise_n} = {rate:.1f}%")

print()
print("6. reversal is a real counter-hypothesis, not a relabelled momentum")
# A sustained uptrend drives RSI to an extreme. momentum should follow it; reversal
# should fade it. If the two agree on every input, the dropdown is decorative again.
_, rev_up = run(up, strategy="reversal")
_, mom_up = run(up, strategy="momentum")
_rsi = identify_current_setup(
    enrich_with_indicators(up["15m"].copy(), "15m"), "15m").get("rsi")
_ms, _rs = mom_up["signal"], rev_up["signal"]
_msc, _rsc = mom_up["weighted_score"], rev_up["weighted_score"]
check("an overbought uptrend: momentum follows, reversal fades",
      _ms == "CALLS" and _rs == "PUTS",
      "RSI " + str(_rsi) + " -> momentum=" + _ms + " reversal=" + _rs)
check("reversal inverts the score sign", _rsc < 0 < _msc,
      "%+.3f vs %+.3f" % (_msc, _rsc))

# With nothing extended, reversal must SAY it has nothing to fade rather than silently
# degrading into momentum -- that is exactly how "gap" hid for so long.
# An OSCILLATING tape, not a slow trend: any consistent drift, however small, still
# pins RSI at an extreme over a 14-period window, which is the opposite of what this
# case needs to exercise.
mid = {tf: osc_bars(seed=40 + j)
       for j, tf in enumerate(["15m", "hourly", "4h", "daily"])}
_, rev_mid = run(mid, strategy="reversal")
check("no extension -> reversal says so and stands down",
      "no extension to fade" in rev_mid["reasoning"] and rev_mid["signal"] == "HOLD",
      rev_mid["reasoning"][:78])

print()
print("7. the removed scalp option degrades safely")
_, sc = run(up, strategy="scalp")
check("a caller passing the removed scalp falls through to momentum",
      (sc["signal"], sc["confidence"]) == (_ms, mom_up["confidence"]),
      sc["signal"] + "/" + str(sc["confidence"]))

print()
print("8. the session banner knows what the calendar knows")
import datetime as _dt
from zoneinfo import ZoneInfo as _Z
from trade_analysis.market_session import market_status, banner as _bn
_ET = _Z("America/New_York")
# 2026-09-07 is Labor Day. A weekday-only check calls it a session; this must not.
_labor = market_status(_dt.datetime(2026, 9, 7, 22, 22, tzinfo=_ET))
check("Labor Day is not treated as a session",
      _labor["state"] == "holiday" and _labor["is_stale"] is True,
      _labor["state"] + " ref=" + str(_labor["reference_session"]))
check("it points at the previous session, not today",
      _labor["reference_session"] == "2026-09-04", str(_labor["reference_session"]))
_open = market_status(_dt.datetime(2026, 9, 4, 11, 0, tzinfo=_ET))
check("a live session reads as open and not stale",
      _open["is_open"] and _open["is_stale"] is False, _open["state"])
# the Friday after Thanksgiving closes at 13:00 ET
_half = market_status(_dt.datetime(2026, 11, 27, 14, 0, tzinfo=_ET))
check("an early close is respected", _half["state"] == "closed",
      _bn(_half)[:60])
_wknd = market_status(_dt.datetime(2026, 9, 5, 12, 0, tzinfo=_ET))
check("weekend resolves to Friday", _wknd["reference_session"] == "2026-09-04",
      str(_wknd["reference_session"]))

print()
print("9. 4h bars are folded inside sessions, never across the overnight gap")
from trade_analysis.data import fold_to_4h
_h = trend_bars(n=140, drift=0.001, seed=7)
_h.index = pd.date_range("2026-09-01 09:30", periods=140, freq="1h", tz="UTC")
_f = fold_to_4h(_h)
_spans = [(_f.index[k].date() == _f.index[k].date()) for k in range(len(_f))]
_days_in = {d: len(g) for d, g in _h.groupby(_h.index.date)}
_days_out = {d: len(g) for d, g in _f.groupby(_f.index.date)}
check("every 4h bar belongs to exactly one calendar day",
      set(_days_out) == set(_days_in), str(sorted(_days_out)[:3]))
check("each day folds to ceil(hours/4) buckets",
      all(_days_out[d] == -(-_days_in[d] // 4) for d in _days_in),
      str({k: (_days_in[k], _days_out[k]) for k in list(_days_in)[:3]}))

print("\n" + "=" * 66)
if failures:
    print(f"FAILED ({len(failures)}): " + "; ".join(failures))
    sys.exit(1)
print("All reachability checks passed.")
