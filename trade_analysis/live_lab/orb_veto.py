"""Close-based range-expansion veto: is today tradeable at all?

Frozen by `research/orb_veto_preregistration.md`. Read that first; this file implements
it and adds nothing.

    from .orb_veto import classify_day
    v = classify_day(premarket_bars, rth_bars)
    v.decide      # True  -> suppress NEW entries after T. Decidable at T.
    v.post        # True  -> the whole session never expanded. ATTRIBUTION ONLY.

WHAT THIS IS
------------
The trader read a QQQ 10-minute chart on 2026-09-17 and said the day was untradeable:
premarket carried all the momentum, RTH never broke out of the range, so there was nothing
to take in either arm. That is not a direction call. It is a statement about whether the
day's available move clears the cost floor -- the same economics as the VRP cost-model
null, applied as a filter instead of a strategy.

TWO THINGS IT RETURNS, AND ONLY ONE MAY GATE A TRADE
----------------------------------------------------
`decide` is evaluated from bars closing in (OR_end, T]. It is what a runner could act on.
`post` is evaluated over the whole session, so it is knowable only after the close.

They are separate fields rather than a parameter because the spoken rule -- "it never
broke out during trading hours" -- IS the post version, and using it to gate an 11:00
entry reads the future. This project lost two result sets to a one-minute lookahead that
supplied 88-95.7% of a measured edge. Making the honest one the default and the dishonest
one impossible to mistake for it is cheaper than remembering.

CLOSE-BASED, NOT TOUCH-BASED
----------------------------
Expansion requires a 10-minute bar to CLOSE outside the band. On 2026-09-17 QQQ spiked to
~718 on the session's largest volume bar and closed back inside ~715.0-716.7. A touch rule
calls that expansion; a close rule calls it a failed breakout, which is what it was. A rule
a single tick can flip is not a rule.

DEPENDENCY-FREE on purpose, like `indicators.py`: no pandas, no feed, no network. It takes
bar dicts and returns a verdict, so the same code can run in a backtest, in the live
runner, and in a test with hand-written bars.
"""
from __future__ import annotations

import dataclasses as dc
import datetime as dt
from typing import Iterable, Sequence

# --- frozen parameters, all from the pre-registration -------------------------------
# Each is either taken from an existing file in this repo or is the timeframe he reads.
# None was chosen by trying alternatives; see preregistration §3 on why that matters.
INTERVAL_MIN = 10                      # the timeframe he reads
RTH_OPEN = dt.time(9, 30)
OR_END = dt.time(9, 40)                # first 10-min bar covers 09:30-09:39 inclusive
DECIDE_AT = dt.time(11, 0)             # CrabelStretch's window already ends here
BUFFER = 0.0005                        # levels_test.BUFFER, his VWAP_Reclaim convention
PREMARKET_FROM = dt.time(4, 0)
PREMARKET_TO = RTH_OPEN


@dc.dataclass(frozen=True)
class Verdict:
    """The classification, plus every intermediate needed to audit it.

    The band and the bar that broke it are returned rather than just the booleans,
    because a veto that cannot be checked against a chart by eye is a veto nobody will
    trust -- and the whole hypothesis came from someone reading a chart.
    """
    decide: bool                       # TRADEABLE GATE: no expansion by DECIDE_AT
    post: bool                         # ATTRIBUTION ONLY: no expansion all session
    band_low: float | None
    band_high: float | None
    premarket_high: float | None
    premarket_low: float | None
    or_high: float | None
    or_low: float | None
    first_break_ts: dt.datetime | None   # first bar to CLOSE outside, any time
    first_break_dir: str | None          # "up" | "down"
    n_bars_10m: int
    usable: bool                       # False -> not classifiable; see `reason`
    reason: str = ""

    def gate(self) -> bool:
        """The only field a runner may act on. True = suppress new entries after T.

        An unusable day never vetoes. Refusing to trade because the data was too thin to
        classify would silently turn a data gap into a strategy decision, and the
        resulting P&L would be attributed to the veto.
        """
        return bool(self.usable and self.decide)


def resample_10m(bars: Sequence[dict], day: dt.date | None = None) -> list[dict]:
    """1-minute bars -> 10-minute bars aligned to 09:30.

    Aligned to the session open, NOT to the wall clock's tens. 09:30-09:39 must be one
    bucket; bucketing by `minute // 10` would split it at 09:30-09:29 boundaries and make
    the opening range a 9-minute bar on some days and 10 on others.
    """
    out: list[dict] = []
    cur: dict | None = None
    anchor = None
    for b in sorted(bars, key=lambda x: x["ts"]):
        ts = b["ts"]
        if anchor is None:
            anchor = ts.replace(hour=RTH_OPEN.hour, minute=RTH_OPEN.minute,
                                second=0, microsecond=0)
        offset = int((ts - anchor).total_seconds() // 60)
        bucket = anchor + dt.timedelta(minutes=(offset // INTERVAL_MIN) * INTERVAL_MIN)
        if cur is None or cur["ts"] != bucket:
            if cur is not None:
                out.append(cur)
            cur = {"ts": bucket, "open": b["open"], "high": b["high"],
                   "low": b["low"], "close": b["close"], "volume": b.get("volume", 0.0)}
        else:
            cur["high"] = max(cur["high"], b["high"])
            cur["low"] = min(cur["low"], b["low"])
            cur["close"] = b["close"]
            cur["volume"] = cur.get("volume", 0.0) + b.get("volume", 0.0)
    if cur is not None:
        out.append(cur)
    return out


def _hi_lo(bars: Iterable[dict]) -> tuple[float | None, float | None]:
    hi = lo = None
    for b in bars:
        hi = b["high"] if hi is None else max(hi, b["high"])
        lo = b["low"] if lo is None else min(lo, b["low"])
    return hi, lo


def classify_day(premarket_1m: Sequence[dict], rth_1m: Sequence[dict],
                 decide_at: dt.time = DECIDE_AT,
                 buffer: float = BUFFER,
                 interval_min: int = INTERVAL_MIN) -> Verdict:
    """Classify one session. `premarket_1m` may be empty; see `usable`.

    Both inputs are 1-minute bars as the feed returns them --
    `{"ts": datetime, "open","high","low","close","volume"}` -- because that is what
    `levels_test.get_ext` already produces and caches.
    """
    blank = dict(decide=False, post=False, band_low=None, band_high=None,
                 premarket_high=None, premarket_low=None, or_high=None, or_low=None,
                 first_break_ts=None, first_break_dir=None, n_bars_10m=0)

    if not rth_1m:
        return Verdict(**blank, usable=False, reason="no RTH bars")

    pm = [b for b in premarket_1m
          if PREMARKET_FROM <= b["ts"].time() < PREMARKET_TO]
    pmh, pml = _hi_lo(pm)

    ten = [b for b in resample_10m(rth_1m, None)
           if b["ts"].time() >= RTH_OPEN]
    if not ten:
        return Verdict(**blank, usable=False, reason="no RTH 10m bars")

    opening = ten[0]
    orh, orl = opening["high"], opening["low"]

    # The union. Expansion must set a genuinely new extreme for the day, not merely
    # exceed whichever of premarket / opening range happened to be narrower. The
    # conservative direction: it fires the veto LESS often, so the veto has a harder
    # time looking good by accident.
    band_high = orh if pmh is None else max(pmh, orh)
    band_low = orl if pml is None else min(pml, orl)

    if band_high <= 0 or band_low <= 0 or band_high < band_low:
        return Verdict(**{**blank, "premarket_high": pmh, "premarket_low": pml,
                          "or_high": orh, "or_low": orl},
                       usable=False, reason="degenerate band")

    up_level = band_high * (1.0 + buffer)
    dn_level = band_low * (1.0 - buffer)

    # Bars strictly AFTER the opening range. The opening-range bar cannot break a band it
    # helped define, and including it would make every wide-open day read as expansion.
    later = [b for b in ten[1:]]
    first_ts = first_dir = None
    broke_by_decide = False
    for b in later:
        broke = ("up" if b["close"] > up_level
                 else "down" if b["close"] < dn_level else None)
        if broke and first_ts is None:
            first_ts, first_dir = b["ts"], broke
        # A bar is inside the decision window when its CLOSE is known by `decide_at`.
        # The bar stamped 10:50 covers 10:50-10:59 and is not complete until 11:00, so
        # its close IS available at 11:00 and it counts. The bar stamped 11:00 is not.
        bar_closes_at = (b["ts"] + dt.timedelta(minutes=interval_min)).time()
        if broke and bar_closes_at <= decide_at:
            broke_by_decide = True

    if len(pm) == 0:
        reason = "no premarket bars; band is the opening range alone"
    else:
        reason = ""

    return Verdict(
        decide=not broke_by_decide,
        post=first_ts is None,
        band_low=band_low, band_high=band_high,
        premarket_high=pmh, premarket_low=pml, or_high=orh, or_low=orl,
        first_break_ts=first_ts, first_break_dir=first_dir,
        n_bars_10m=len(ten), usable=True, reason=reason,
    )


def touch_based_for_comparison(premarket_1m: Sequence[dict],
                               rth_1m: Sequence[dict],
                               decide_at: dt.time = DECIDE_AT,
                               buffer: float = BUFFER) -> bool:
    """The rule NOT chosen, kept so the choice can be shown to have mattered.

    Returns the touch-based `decide`. 2026-09-17 QQQ is the case that separates them: a
    spike to ~718 that closed back inside. If this ever agrees with `classify_day` on
    every day of the sample, the close/touch distinction was cosmetic and the
    pre-registration's §2 should say so.
    """
    if not rth_1m:
        return False
    pm = [b for b in premarket_1m if PREMARKET_FROM <= b["ts"].time() < PREMARKET_TO]
    pmh, pml = _hi_lo(pm)
    ten = [b for b in resample_10m(rth_1m, None) if b["ts"].time() >= RTH_OPEN]
    if not ten:
        return False
    orh, orl = ten[0]["high"], ten[0]["low"]
    bh = orh if pmh is None else max(pmh, orh)
    bl = orl if pml is None else min(pml, orl)
    for b in ten[1:]:
        if (b["ts"] + dt.timedelta(minutes=INTERVAL_MIN)).time() > decide_at:
            continue
        if b["high"] > bh * (1.0 + buffer) or b["low"] < bl * (1.0 - buffer):
            return False
    return True
