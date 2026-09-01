"""Session state and the immutable evaluation context handed to setups.

This module owns the no-lookahead guarantee at the data level:

  * `SessionState.accept_bars()` admits a 1-minute bar only once it is definitively
    complete -- a bar stamped T covers [T, T+60s) and is accepted no earlier than
    T + 60s + settle. The forming bar is never admitted, so setup code cannot see it.
  * 5-minute bars are aggregated from admitted 1-minute bars only, and are stamped at
    their CLOSE (09:30-09:34 -> 09:35), matching the spec.
  * `Context` exposes tuples, not lists, and holds no reference to a mutable buffer.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Sequence

from . import indicators as ind

RTH_OPEN = dt.time(9, 30)
RTH_LAST_BAR = dt.time(15, 59)
BAR_SECONDS = 60

# How many completed prior sessions seed the indicator series. TTM_Squeeze is the
# hungriest consumer (42 x 5m bars = 210 minutes), so one session already suffices;
# two is kept as headroom for a holiday-shortened day.
PREFIX_SESSIONS = 2


def aggregate_5m(bars_1m, day: dt.date) -> list[dict]:
    """Aggregate one session's 1m bars into 5m buckets stamped at their CLOSE.

    A bucket is emitted only when all five of its minutes are present, so a data gap
    yields no bar rather than a silently short one. Callers must pass a SINGLE session:
    aggregating across days would straddle the overnight break.
    """
    buckets: dict[dt.datetime, list[dict]] = {}
    for b in bars_1m:
        mins = (b["ts"].hour * 60 + b["ts"].minute) - (9 * 60 + 30)
        if mins < 0:
            continue
        close_ts = (dt.datetime.combine(day, RTH_OPEN)
                    + dt.timedelta(minutes=(mins // 5 + 1) * 5))
        buckets.setdefault(close_ts, []).append(b)
    out = []
    for close_ts in sorted(buckets):
        grp = buckets[close_ts]
        if len(grp) != 5:
            continue
        out.append({"ts": close_ts, "open": grp[0]["open"],
                    "high": max(x["high"] for x in grp),
                    "low": min(x["low"] for x in grp),
                    "close": grp[-1]["close"],
                    "volume": sum(x["volume"] for x in grp)})
    return out


# --------------------------------------------------------------------------- context


@dataclass(frozen=True)
class Context:
    """Everything a setup may look at. Immutable, ends at `bar_ts`, never reaches past it."""

    symbol: str
    day: dt.date
    bar_ts: dt.datetime               # CLOSE timestamp of the bar being evaluated
    tf: str                           # "1m" or "5m" -- which timeframe triggered this call
    bars_1m: tuple                    # TODAY's closed 1m bars -- session structure only
    bars_5m: tuple                    # TODAY's closed 5m bars
    ind_1m: tuple                     # prior sessions + today -- INDICATOR seeding
    ind_5m: tuple
    vwap: float | None
    vwap_sigma: float | None
    prior_day: dict | None            # {"high","low","close","open"} RTH-only
    session_open: float | None
    opening_range: dict               # {"5": {...}, "15": {...}} once formed
    warmup: dict                      # per-symbol baselines (rvol, imb sigma, stretch, ...)
    quote: dict | None                # live underlying NBBO at decision time

    # ---- convenience accessors used by setups -------------------------------

    @property
    def last(self) -> dict:
        return self.bars_1m[-1] if self.tf == "1m" else self.bars_5m[-1]

    @property
    def price(self) -> float:
        return self.last["close"]

    def closes(self, tf: str = "1m") -> list[float]:
        """TODAY's closes. For indicators use ind_closes() instead."""
        src = self.bars_1m if tf == "1m" else self.bars_5m
        return [b["close"] for b in src]

    def ohlcv(self, tf: str = "1m"):
        """TODAY's OHLCV. For indicators use ind_ohlcv() instead."""
        src = self.bars_1m if tf == "1m" else self.bars_5m
        return ([b["high"] for b in src], [b["low"] for b in src],
                [b["close"] for b in src], [b["volume"] for b in src])

    # ---- indicator series: prior sessions + today ---------------------------
    #
    # A moving average on a real chart does not reset at 09:30, and the trader is
    # looking at a real chart. Seeding only from today made EMA20-on-5m unusable
    # until 11:35 and MACD(9,17,9)+ADX(14) unusable until 12:25 -- past the windows
    # several setups trade in, and past his own median entry of 11:39.
    #
    # ATR/ADX therefore see the overnight gap as one large true range, exactly as
    # they do on any continuous chart. That is the standard convention, not a bug,
    # but it is a real modelling choice and is recorded as one.

    def ind_closes(self, tf: str = "1m") -> list[float]:
        src = self.ind_1m if tf == "1m" else self.ind_5m
        return [b["close"] for b in src]

    def ind_ohlcv(self, tf: str = "1m"):
        src = self.ind_1m if tf == "1m" else self.ind_5m
        return ([b["high"] for b in src], [b["low"] for b in src],
                [b["close"] for b in src], [b["volume"] for b in src])

    def ind_len(self, tf: str = "1m") -> int:
        return len(self.ind_1m if tf == "1m" else self.ind_5m)

    def today_offset(self, tf: str = "1m") -> int:
        """Index in the ind_ series where TODAY begins -- lets a setup align the two."""
        src = self.ind_1m if tf == "1m" else self.ind_5m
        today = self.bars_1m if tf == "1m" else self.bars_5m
        return len(src) - len(today)

    def minutes_since_open(self) -> int:
        o = dt.datetime.combine(self.day, RTH_OPEN)
        return int((self.bar_ts - o).total_seconds() // 60)


@dataclass
class Signal:
    setup_id: str
    direction: str                    # "long" | "short"
    stop: float | None                # underlying price level
    target: float | None              # underlying price level
    time_exit_min: int | None = None  # hard time exit, minutes
    bar_exit: int | None = None       # hard exit after N bars of the setup's timeframe
    trailing: str | None = None       # name of a trailing rule the setup implements
    note: str = ""
    state: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- session


class SessionState:
    """Per-symbol, per-day bar buffer with completeness enforcement."""

    def __init__(self, symbol: str, day: dt.date, warmup: dict,
                 prior_day: dict | None, settle_ms: int = 1500):
        self.symbol = symbol
        self.day = day
        self.warmup = warmup
        self.prior_day = prior_day
        self.settle = dt.timedelta(milliseconds=settle_ms)
        self._bars_1m: list[dict] = []
        self._bars_5m: list[dict] = []
        self._seen: set[dt.datetime] = set()
        self.degraded_bars: list[dt.datetime] = []
        # Prior-session bars, used ONLY to seed indicators. Never used for session
        # structure (VWAP, opening range, gap, session high/low) -- those stay anchored
        # to today, which is what keeps the VWAP-anchoring guarantee intact.
        self.prefix_1m: list[dict] = list(warmup.get("prefix_1m") or [])
        self.prefix_5m: list[dict] = list(warmup.get("prefix_5m") or [])

    # ------------------------------------------------------------------ admission

    def bar_is_complete(self, bar_ts: dt.datetime, now: dt.datetime) -> bool:
        """A bar stamped T is complete at T + 60s (+ settle). Never before."""
        return now >= bar_ts + dt.timedelta(seconds=BAR_SECONDS) + self.settle

    def accept_bars(self, candidates: Sequence[dict], now: dt.datetime) -> list[dict]:
        """Admit newly-completed 1m bars. Returns the bars actually admitted.

        Duplicates are suppressed by timestamp. Out-of-order arrivals are ignored --
        the buffer is append-only and strictly ascending.
        """
        admitted = []
        for b in sorted(candidates, key=lambda x: x["ts"]):
            ts = b["ts"]
            if ts in self._seen:
                continue
            if ts.time() < RTH_OPEN or ts.time() > RTH_LAST_BAR:
                continue
            if not self.bar_is_complete(ts, now):
                continue                       # the forming bar stops here, always
            if self._bars_1m and ts <= self._bars_1m[-1]["ts"]:
                continue                       # late arrival behind the buffer head
            if self._bars_1m:
                gap = int((ts - self._bars_1m[-1]["ts"]).total_seconds() // 60) - 1
                if gap > 0:
                    self.degraded_bars.append(ts)
            self._seen.add(ts)
            self._bars_1m.append(dict(b))
            admitted.append(self._bars_1m[-1])
        if admitted:
            self._rebuild_5m()
        return admitted

    def _rebuild_5m(self) -> None:
        self._bars_5m = aggregate_5m(self._bars_1m, self.day)

    # ------------------------------------------------------------------ derived

    @property
    def bars_1m(self) -> list[dict]:
        return self._bars_1m

    @property
    def bars_5m(self) -> list[dict]:
        return self._bars_5m

    def session_open(self) -> float | None:
        return self._bars_1m[0]["open"] if self._bars_1m else None

    def opening_range(self) -> dict:
        """09:30-09:35 and 09:30-09:45 ranges, present only once fully formed."""
        out: dict[str, dict] = {}
        for label, minutes in (("5", 5), ("15", 15)):
            end = dt.datetime.combine(self.day, RTH_OPEN) + dt.timedelta(minutes=minutes)
            grp = [b for b in self._bars_1m if b["ts"] < end]
            if len(grp) == minutes:
                out[label] = {
                    "high": max(b["high"] for b in grp),
                    "low": min(b["low"] for b in grp),
                    "end": end,
                }
                out[label]["mid"] = (out[label]["high"] + out[label]["low"]) / 2.0
        return out

    def context(self, bar_ts: dt.datetime, tf: str, quote: dict | None) -> Context:
        vw = ind.session_vwap(self._bars_1m)          # anchored 09:30 TODAY, always
        return Context(
            symbol=self.symbol,
            day=self.day,
            bar_ts=bar_ts,
            tf=tf,
            bars_1m=tuple(self._bars_1m),
            bars_5m=tuple(self._bars_5m),
            ind_1m=tuple(self.prefix_1m) + tuple(self._bars_1m),
            ind_5m=tuple(self.prefix_5m) + tuple(self._bars_5m),
            vwap=vw,
            vwap_sigma=ind.vwap_sigma(self._bars_1m, vw) if vw else None,
            prior_day=self.prior_day,
            session_open=self.session_open(),
            opening_range=self.opening_range(),
            warmup=self.warmup,
            quote=quote,
        )


# --------------------------------------------------------------------------- warmup


def build_warmup(feed, symbol: str, day: dt.date, prior_days: list[dt.date]) -> dict:
    """Baselines that need prior sessions. Computed once at startup, then cached.

    Nothing here touches `day` itself -- every input is a completed prior session.
    """
    sessions: dict[dt.date, list[dict]] = {}
    for d in prior_days:
        bars = feed.minute_bars(symbol, d)
        if len(bars) >= 300:
            sessions[d] = bars

    # --- time-of-day cumulative volume baseline (last 20 sessions) ---
    rvol_base: dict[str, list[float]] = {}
    for bars in list(sessions.values())[-20:]:
        cum = 0.0
        for b in bars:
            cum += b["volume"]
            rvol_base.setdefault(b["ts"].strftime("%H:%M"), []).append(cum)
    rvol_cum = {k: (sum(v) / len(v)) for k, v in rvol_base.items() if v}

    # --- IntradayMomentumBoundary sigma: |Close_d(m)/Open_d(0930) - 1|, last 14 ---
    imb: dict[str, list[float]] = {}
    for bars in list(sessions.values())[-14:]:
        o = bars[0]["open"]
        if o <= 0:
            continue
        for b in bars:
            imb.setdefault(b["ts"].strftime("%H:%M"), []).append(abs(b["close"] / o - 1.0))
    imb_sigma = {k: (sum(v) / len(v)) for k, v in imb.items() if v}

    # --- Crabel stretch: SMA(min(Open-Low, High-Open), 10) over prior RTH days ---
    noise = []
    for d in sorted(sessions)[-10:]:
        bars = sessions[d]
        o = bars[0]["open"]
        hi = max(b["high"] for b in bars)
        lo = min(b["low"] for b in bars)
        noise.append(min(o - lo, hi - o))
    stretch = (sum(noise) / len(noise)) if len(noise) >= 5 else None

    # --- 09:30-10:30 range, 20-day average (VWAP_2sigma_Fade regime filter) ---
    or60 = []
    for d in sorted(sessions)[-20:]:
        grp = [b for b in sessions[d] if b["ts"].time() <= dt.time(10, 30)]
        if grp:
            or60.append(max(b["high"] for b in grp) - min(b["low"] for b in grp))
    or60_avg = (sum(or60) / len(or60)) if or60 else None

    prior = None
    if sessions:
        d = sorted(sessions)[-1]
        bars = sessions[d]
        prior = {
            "date": d,
            "open": bars[0]["open"],
            "high": max(b["high"] for b in bars),
            "low": min(b["low"] for b in bars),
            "close": bars[-1]["close"],
        }

    # --- indicator seeding prefix: the last PREFIX_SESSIONS completed sessions ---
    # Aggregated PER SESSION so a 5m bucket never straddles two days.
    prefix_1m: list[dict] = []
    prefix_5m: list[dict] = []
    for d in sorted(sessions)[-PREFIX_SESSIONS:]:
        prefix_1m.extend(sessions[d])
        prefix_5m.extend(aggregate_5m(sessions[d], d))

    return {
        "rvol_cum_by_minute": rvol_cum,
        "imb_sigma_by_minute": imb_sigma,
        "crabel_stretch": stretch,
        "or60_avg_20d": or60_avg,
        "prior_day": prior,
        "prefix_1m": prefix_1m,
        "prefix_5m": prefix_5m,
        "sessions_used": len(sessions),
    }
