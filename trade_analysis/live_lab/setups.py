"""The 13 frozen setups.

Every setup is a pure function of its Context. It cannot see another setup's state,
signals, or P&L, and it cannot see any bar past `ctx.bar_ts`.

Parameter provenance is recorded in `params` on every setup and hashed into the config,
so a later reader can tell a sourced number from an invented one. `"[R]"` in a param name
suffix marks a value chosen by us rather than by a published source -- those are the
candidate false positives if a setup ever "works".

Stop/target semantics: triggers are detected on the bar's HIGH/LOW (matching how the
sources define levels), but the option is always filled at the next OBSERVED quote, never
at the theoretical level. positions.py records trigger_level and fill separately so the
gap between them is measurable rather than assumed away.
"""
from __future__ import annotations

import datetime as dt

from . import indicators as ind
from .session import Context, Signal

ET = dt.timezone.utc  # placeholder; all lab timestamps are naive ET


def _t(h: int, m: int) -> dt.time:
    return dt.time(h, m)


class Setup:
    id: str = "base"
    version: str = "1"
    timeframe: str = "1m"
    max_per_day: int = 99
    max_per_direction: int | None = None
    params: dict = {}

    def evaluate(self, ctx: Context) -> Signal | None:      # pragma: no cover - interface
        raise NotImplementedError

    def manage(self, pos: dict, ctx: Context) -> str | None:
        """Optional trailing/custom exit. Return an exit reason string, or None."""
        return None

    def describe(self) -> dict:
        return {"id": self.id, "version": self.version, "timeframe": self.timeframe,
                "max_per_day": self.max_per_day,
                "max_per_direction": self.max_per_direction, "params": self.params}


# ===========================================================================
# The trader's three fixed setups -- taken verbatim, not redefined
# ===========================================================================


class ORB5min(Setup):
    id, version, timeframe = "ORB_5min", "1", "1m"
    max_per_day = 1
    params = {"range_min": 5, "min_width_pct": 0.15, "vol_mult": 1.5,
              "vol_lookback": 10, "target_R": 1.5, "cutoff": "11:00",
              "source": "edgeful (user-supplied, verbatim)"}

    def evaluate(self, ctx):
        rng = ctx.opening_range.get("5")
        if not rng or ctx.bar_ts <= rng["end"] or ctx.bar_ts.time() >= _t(11, 0):
            return None
        width = rng["high"] - rng["low"]
        if width < 0.0015 * ctx.price:                      # >= 0.15% of price
            return None
        vols = [b["volume"] for b in ctx.ind_1m]        # seeded: usable from 09:36
        rv = ind.rvol_bar(vols, 10)
        if rv is None or rv < 1.5:
            return None
        c = ctx.price
        if c > rng["high"]:
            return Signal(self.id, "long", stop=rng["low"],
                          target=c + 1.5 * (c - rng["low"]),
                          state={"or_high": rng["high"], "or_low": rng["low"], "rvol": rv})
        if c < rng["low"]:
            return Signal(self.id, "short", stop=rng["high"],
                          target=c - 1.5 * (rng["high"] - c),
                          state={"or_high": rng["high"], "or_low": rng["low"], "rvol": rv})
        return None


class ORB15min(Setup):
    id, version, timeframe = "ORB_15min", "1", "1m"
    max_per_day = 1
    params = {"range_min": 15, "target_R": 2.0, "cutoff": "10:30",
              "stop": "mid-range", "vwap_filter": True,
              "source": "journalplus (user-supplied, verbatim)"}

    def evaluate(self, ctx):
        rng = ctx.opening_range.get("15")
        if not rng or ctx.bar_ts <= rng["end"] or ctx.bar_ts.time() >= _t(10, 30):
            return None
        if ctx.vwap is None:
            return None
        c, mid = ctx.price, rng["mid"]
        if c > rng["high"] and c > ctx.vwap:
            return Signal(self.id, "long", stop=mid, target=c + 2.0 * (c - mid),
                          state={"or_high": rng["high"], "or_low": rng["low"], "vwap": ctx.vwap})
        if c < rng["low"] and c < ctx.vwap:
            return Signal(self.id, "short", stop=mid, target=c - 2.0 * (mid - c),
                          state={"or_high": rng["high"], "or_low": rng["low"], "vwap": ctx.vwap})
        return None


class VWAPReclaim(Setup):
    id, version, timeframe = "VWAP_Reclaim", "1", "1m"
    max_per_day = 2
    params = {"buffer_pct": 0.05, "ema_slope_len": 20, "swing_lookback": 5,
              "target_R": 1.5, "source": "user-supplied, verbatim"}

    def evaluate(self, ctx):
        if ctx.vwap is None or len(ctx.bars_1m) < 3 or ctx.ind_len("1m") < 25:
            return None
        closes = ctx.closes("1m")                        # today: the cross itself
        e = ind.ema_series(ctx.ind_closes("1m"), 20)     # seeded: the slope filter
        if e[-1] is None or e[-4] is None:
            return None
        slope = e[-1] - e[-4]
        buf = 0.0005 * ctx.price                            # 0.05% of price
        prev, c = closes[-2], closes[-1]
        prior = ctx.bars_1m[-6:-1]
        if c > ctx.vwap + buf and prev <= ctx.vwap and slope > 0:
            stop = min(b["low"] for b in prior)
            if stop >= c:
                return None
            return Signal(self.id, "long", stop=stop, target=c + 1.5 * (c - stop),
                          state={"vwap": ctx.vwap, "ema20_slope": slope})
        if c < ctx.vwap - buf and prev >= ctx.vwap and slope < 0:
            stop = max(b["high"] for b in prior)
            if stop <= c:
                return None
            return Signal(self.id, "short", stop=stop, target=c - 1.5 * (stop - c),
                          state={"vwap": ctx.vwap, "ema20_slope": slope})
        return None


# ===========================================================================
# The trader's reconstructed discretionary behaviour -- a HYPOTHESIS
# ===========================================================================


class MomoChase(Setup):
    """Reconstructed from measured behaviour only. Expected to fail.

    Every threshold traces to a number measured from his own tape, not chosen to make
    his history look profitable. Notably there is NO volume condition: his median entry
    sits at 0.986x the 20-EMA of volume and only 48.3% clear it, so requiring volume
    would encode a belief he holds rather than a behaviour he exhibits.
    """

    id, version, timeframe = "MOMO_CHASE", "1", "5m"
    max_per_day = 3
    params = {
        "window": "10:30-14:30", "trail_min": 15, "sigma_mult": 0.60,
        "macd": [9, 17, 9], "adx_min": 20, "dmi_len": 14,
        "volume_condition": None, "time_exit_min": 25,
        "provenance": {
            "sigma_mult": "0.60 = where his density ratio vs a same-session placebo crosses 1.0x (0.4-0.6 bucket 0.71x -> 0.6-0.8 bucket 1.14x). Captures 75.9% of his trades, fires on 38.6% of random midday minutes, lift 1.96x, ~1.97 trades/session vs his observed ~2.6. NOTE: the earlier +0.2029 figure was in DAILY-sigma units; this setup normalises by 15-min sigma, which is ~5.1x smaller. Fit to his BEHAVIOUR, never to his outcomes -- no P&L entered this calibration.",
            "window": "his median entry 11:39 ET",
            "macd_align": "69.4% of his entries", "dmi_align": "78.5%", "adx_gt_20": "69.6%",
            "no_volume": "his median entry is 0.986x vol-EMA20; only 48.3% clear it",
            "time_exit_min": "his stated typical hold",
        },
    }

    def evaluate(self, ctx):
        t = ctx.bar_ts.time()
        if not (_t(10, 30) <= t <= _t(14, 30)):
            return None
        if len(ctx.bars_1m) < 20 or ctx.ind_len("5m") < 35:
            return None

        closes1 = ctx.closes("1m")
        sig1 = ind.realised_sigma(closes1)                  # session 1-min return SD
        if not sig1:
            return None
        sigma15 = sig1 * (15 ** 0.5)
        if len(closes1) < 16:
            return None
        trail = closes1[-1] / closes1[-16] - 1.0            # trailing 15-minute return
        z = trail / sigma15
        if abs(z) < 0.60:
            return None
        direction = "long" if z > 0 else "short"
        sign = 1.0 if direction == "long" else -1.0

        closes5 = ctx.ind_closes("5m")                  # seeded across sessions
        h5, l5, c5, _ = ctx.ind_ohlcv("5m")
        m = ind.macd_hist(closes5, 9, 17, 9)
        adx, pdi, mdi = ind.adx_dmi(h5, l5, c5, 14)
        if m is None or adx is None or pdi is None or mdi is None:
            return None
        if m * sign <= 0:                                   # MACD histogram aligned
            return None
        if (pdi - mdi) * sign <= 0:                         # DMI aligned
            return None
        if adx <= 20:
            return None

        return Signal(self.id, direction, stop=None, target=None, time_exit_min=25,
                      state={"trail_15m_sigma": z, "macd_hist": m, "adx": adx,
                             "plus_di": pdi, "minus_di": mdi, "sigma_1m": sig1})


# ===========================================================================
# Researched setups (Appendix A)
# ===========================================================================


class IntradayMomentumBoundary(Setup):
    """Zarattini/Aziz/Barbon, SSRN 4824172. The only item needing no invented parameter."""

    id, version, timeframe = "IntradayMomentumBoundary", "1", "1m"
    max_per_day = 4
    params = {"lookback_days": 14, "band_mult": 1.0, "checkpoints": "HH:00 and HH:30",
              "window": "10:00-15:30", "exit": "trailing vs max(VWAP, boundary)",
              "source": "SSRN 4824172 / Concretum"}

    def evaluate(self, ctx):
        t = ctx.bar_ts.time()
        if t.minute not in (0, 30) or not (_t(10, 0) <= t <= _t(15, 30)):
            return None
        sig = ctx.warmup.get("imb_sigma_by_minute", {}).get(t.strftime("%H:%M"))
        prior = ctx.prior_day
        if not sig or not prior or ctx.session_open is None or ctx.vwap is None:
            return None
        hi_anchor = max(ctx.session_open, prior["close"])
        lo_anchor = min(ctx.session_open, prior["close"])
        ub = hi_anchor * (1.0 + 1.0 * sig)
        lb = lo_anchor * (1.0 - 1.0 * sig)
        c = ctx.price
        if c > ub and c > ctx.vwap:
            return Signal(self.id, "long", stop=None, target=None, trailing="imb",
                          state={"ub": ub, "lb": lb, "sigma": sig, "vwap": ctx.vwap})
        if c < lb and c < ctx.vwap:
            return Signal(self.id, "short", stop=None, target=None, trailing="imb",
                          state={"ub": ub, "lb": lb, "sigma": sig, "vwap": ctx.vwap})
        return None

    def manage(self, pos, ctx):
        if ctx.vwap is None:
            return None
        c = ctx.price
        if pos["direction"] == "long":
            if c < max(ctx.vwap, pos["state"]["ub"]):
                return "trail_imb"
        elif c > min(ctx.vwap, pos["state"]["lb"]):
            return "trail_imb"
        return None


class TTMSqueeze(Setup):
    id, version, timeframe = "TTM_Squeeze", "1", "5m"
    max_per_day = 2
    params = {"bb_len": 20, "bb_mult": 2.0, "kc_len": 20, "kc_mult": 1.5,
              "mom_len": 20, "stop_atr_mult[R]": 1.5, "target_R[R]": 2.0,
              "bar_exit[R]": 9, "cutoff[R]": "15:00", "source": "StockCharts"}

    def _squeeze_on(self, closes, highs, lows, upto):
        c = closes[:upto]
        if len(c) < 21:
            return None
        m = ind.sma(c, 20)
        sd = ind.stdev(c, 20)
        e = ind.ema_series(c, 20)[-1]
        a = ind.atr(highs[:upto], lows[:upto], c, 20)
        if None in (m, sd, e, a):
            return None
        return (m + 2.0 * sd) < (e + 1.5 * a) and (m - 2.0 * sd) > (e - 1.5 * a)

    def evaluate(self, ctx):
        if ctx.bar_ts.time() >= _t(15, 0) or ctx.ind_len("5m") < 42:
            return None
        h, l, c, _ = ctx.ind_ohlcv("5m")
        now_on = self._squeeze_on(c, h, l, len(c))
        prev_on = self._squeeze_on(c, h, l, len(c) - 1)
        if now_on is None or prev_on is None or not (prev_on and not now_on):
            return None                                     # must FIRE, not merely be off

        deltas = []
        for i in range(20, len(c) + 1):
            w_h, w_l, w_c = h[:i], l[:i], c[:i]
            mid = (max(w_h[-20:]) + min(w_l[-20:])) / 2.0
            s = ind.sma(w_c, 20)
            if s is None:
                continue
            deltas.append(w_c[-1] - (mid + s) / 2.0)
        if len(deltas) < 21:
            return None
        hist = ind.linreg_endpoint(deltas, 20)
        hist_prev = ind.linreg_endpoint(deltas[:-1], 20)
        if hist is None or hist_prev is None:
            return None
        a = ind.atr(h, l, c, 20)
        if a is None:
            return None
        px = c[-1]
        if hist > 0 and hist > hist_prev:
            stop = px - 1.5 * a
            return Signal(self.id, "long", stop=stop, target=px + 2.0 * (px - stop),
                          bar_exit=9, state={"hist": hist, "atr20": a})
        if hist < 0 and hist < hist_prev:
            stop = px + 1.5 * a
            return Signal(self.id, "short", stop=stop, target=px - 2.0 * (stop - px),
                          bar_exit=9, state={"hist": hist, "atr20": a})
        return None


class PDHPDLBreakout(Setup):
    id, version, timeframe = "PDH_PDL_Breakout", "1", "5m"
    max_per_day = 2
    max_per_direction = 1
    params = {"atr_len": 14, "buffer_atr": 0.2, "body_ratio[R]": 0.5,
              "target_R": 2.0, "cutoff[R]": "15:00",
              "source": "NetPicks; buffer = midpoint of stated 0.1-0.3 range"}

    def evaluate(self, ctx):
        if ctx.bar_ts.time() >= _t(15, 0) or not ctx.prior_day or not ctx.bars_5m                 or ctx.ind_len("5m") < 16:
            return None
        h, l, c, _ = ctx.ind_ohlcv("5m")
        a = ind.atr(h, l, c, 14)
        if a is None:
            return None
        b = ctx.bars_5m[-1]
        rng = b["high"] - b["low"]
        if rng <= 0 or abs(b["close"] - b["open"]) < 0.5 * rng:
            return None
        pdh, pdl = ctx.prior_day["high"], ctx.prior_day["low"]
        if b["close"] >= pdh + 0.2 * a:
            return Signal(self.id, "long", stop=b["low"],
                          target=b["close"] + 2.0 * (b["close"] - b["low"]),
                          state={"pdh": pdh, "pdl": pdl, "atr14": a})
        if b["close"] <= pdl - 0.2 * a:
            return Signal(self.id, "short", stop=b["high"],
                          target=b["close"] - 2.0 * (b["high"] - b["close"]),
                          state={"pdh": pdh, "pdl": pdl, "atr14": a})
        return None


class PDHPDLFailedBreak(Setup):
    """Nearly every number here is [R] -- the sources are qualitative. Flagged in the spec."""

    id, version, timeframe = "PDH_PDL_FailedBreak", "1", "5m"
    max_per_day = 2
    max_per_direction = 1
    params = {"penetration_atr[R]": 0.1, "timeliness_bars[R]": 12, "target_R": 2.0,
              "window[R]": "09:45-15:00", "source": "TradeMomentum (mostly researcher-converted)"}

    def evaluate(self, ctx):
        t = ctx.bar_ts.time()
        if not (_t(9, 45) <= t < _t(15, 0)) or not ctx.prior_day or not ctx.bars_5m                 or ctx.ind_len("5m") < 16:
            return None
        h, l, c, _ = ctx.ind_ohlcv("5m")
        a = ind.atr(h, l, c, 14)
        if a is None:
            return None
        pdh, pdl = ctx.prior_day["high"], ctx.prior_day["low"]
        recent = ctx.bars_5m[-13:-1]
        b = ctx.bars_5m[-1]
        if any(x["high"] >= pdh + 0.1 * a for x in recent) and b["close"] < pdh:
            stop = max(x["high"] for x in ctx.bars_5m)      # running session high only
            if stop <= b["close"]:
                return None
            return Signal(self.id, "short", stop=stop,
                          target=b["close"] - 2.0 * (stop - b["close"]),
                          state={"pdh": pdh, "atr14": a})
        if any(x["low"] <= pdl - 0.1 * a for x in recent) and b["close"] > pdl:
            stop = min(x["low"] for x in ctx.bars_5m)
            if stop >= b["close"]:
                return None
            return Signal(self.id, "long", stop=stop,
                          target=b["close"] + 2.0 * (b["close"] - stop),
                          state={"pdl": pdl, "atr14": a})
        return None


class VWAP2SigmaFade(Setup):
    id, version, timeframe = "VWAP_2sigma_Fade", "1", "5m"
    max_per_day = 3
    params = {"band_mult": 2.0, "atr_len": 14, "wick_ratio[R]": 0.5, "vol_mult[R]": 1.5,
              "adx_skip_above": 25, "or60_skip_mult": 2.0,
              "windows": "10:00-11:30, 13:30-14:30", "target": "session VWAP",
              "source": "CrossTrade",
              "note": "source's Wednesday/Thursday-only rule deliberately NOT applied; "
                      "day-of-week recorded as a covariate instead"}

    def evaluate(self, ctx):
        t = ctx.bar_ts.time()
        if not (_t(10, 0) <= t <= _t(11, 30) or _t(13, 30) <= t <= _t(14, 30)):
            return None
        if ctx.vwap is None or ctx.vwap_sigma is None or not ctx.bars_5m                 or ctx.ind_len("5m") < 22:
            return None
        h, l, c, v = ctx.ind_ohlcv("5m")
        adx, _, _ = ind.adx_dmi(h, l, c, 14)
        if adx is not None and adx > 25:
            return None
        or60 = ctx.warmup.get("or60_avg_20d")
        if or60:
            grp = [b for b in ctx.bars_1m if b["ts"].time() <= _t(10, 30)]
            if grp and (max(b["high"] for b in grp) - min(b["low"] for b in grp)) > 2.0 * or60:
                return None
        a = ind.atr(h, l, c, 14)
        rv = ind.rvol_bar(v, 20)
        if a is None or rv is None or rv < 1.5:
            return None
        b = ctx.bars_5m[-1]
        rng = b["high"] - b["low"]
        if rng <= 0:
            return None
        upper = ctx.vwap + 2.0 * ctx.vwap_sigma
        lower = ctx.vwap - 2.0 * ctx.vwap_sigma
        body_hi = max(b["open"], b["close"])
        body_lo = min(b["open"], b["close"])
        if (b["high"] >= upper and b["close"] < upper and b["close"] < b["open"]
                and (b["high"] - body_hi) >= 0.5 * rng):
            return Signal(self.id, "short", stop=b["high"] + a, target=ctx.vwap,
                          state={"upper": upper, "vwap": ctx.vwap, "adx": adx, "rvol": rv,
                                 "dow": ctx.bar_ts.strftime("%a")})
        if (b["low"] <= lower and b["close"] > lower and b["close"] > b["open"]
                and (body_lo - b["low"]) >= 0.5 * rng):
            return Signal(self.id, "long", stop=b["low"] - a, target=ctx.vwap,
                          state={"lower": lower, "vwap": ctx.vwap, "adx": adx, "rvol": rv,
                                 "dow": ctx.bar_ts.strftime("%a")})
        return None


class EMA920Pullback(Setup):
    id, version, timeframe = "EMA_9_20_Pullback", "1", "5m"
    max_per_day = 2
    max_per_direction = 1
    params = {"ema_fast": 9, "ema_slow": 20, "slope_bars[R]": 3, "vol_decline[R]": 5,
              "target_R": 3.0, "window": "09:45-11:00", "stop": "wider of pullback low / EMA20",
              "source": "Bulls on Wall Street"}

    def evaluate(self, ctx):
        t = ctx.bar_ts.time()
        # EMA9/EMA20 come from the SEEDED series -- on a real chart a moving average does
        # not reset at 09:30. Seeding only from today made EMA20-on-5m unavailable until
        # 11:35, i.e. after this setup's window closes, so it could never fire at all.
        if not (_t(9, 45) <= t <= _t(11, 0)) or not ctx.bars_5m or ctx.ind_len("5m") < 25:
            return None
        e9 = ind.ema_series(ctx.ind_closes("5m"), 9)
        e20 = ind.ema_series(ctx.ind_closes("5m"), 20)
        if e9[-1] is None or e20[-1] is None or e20[-4] is None:
            return None

        bars = ctx.bars_5m                                   # TODAY only: "first pullback
        off = ctx.today_offset("5m")                         # of the session"
        b = bars[-1]
        vols = [x["volume"] for x in ctx.ind_5m]
        if len(vols) < 6 or vols[-1] >= (sum(vols[-6:-1]) / 5.0):
            return None

        def scan(up: bool) -> tuple[bool, bool]:
            """(price extended away from EMA9 earlier today, an earlier touch already used it)"""
            extended = touched = False
            for i, bar in enumerate(bars):
                v9 = e9[off + i]
                if v9 is None:
                    continue
                away = bar["low"] > v9 if up else bar["high"] < v9
                back = bar["low"] <= v9 if up else bar["high"] >= v9
                if away:
                    extended = True
                elif extended and back and i < len(bars) - 1:
                    touched = True
            return extended, touched

        ext_up, touched_up = scan(True)
        if (e9[-1] > e20[-1] and e20[-1] > e20[-4] and ext_up and not touched_up
                and b["low"] <= e9[-1] and b["close"] >= e20[-1]
                and b["close"] > b["open"] and b["close"] > e9[-1]):
            stop = min(b["low"], e20[-1])
            if stop < b["close"]:
                return Signal(self.id, "long", stop=stop,
                              target=b["close"] + 3.0 * (b["close"] - stop),
                              state={"ema9": e9[-1], "ema20": e20[-1]})

        ext_dn, touched_dn = scan(False)
        if (e9[-1] < e20[-1] and e20[-1] < e20[-4] and ext_dn and not touched_dn
                and b["high"] >= e9[-1] and b["close"] <= e20[-1]
                and b["close"] < b["open"] and b["close"] < e9[-1]):
            stop = max(b["high"], e20[-1])
            if stop > b["close"]:
                return Signal(self.id, "short", stop=stop,
                              target=b["close"] - 3.0 * (stop - b["close"]),
                              state={"ema9": e9[-1], "ema20": e20[-1]})
        return None


class ThreeBarPlay(Setup):
    id, version, timeframe = "ThreeBarPlay", "1", "5m"
    max_per_day = 3
    params = {"ignite_close_pct": 0.8, "ignite_vol_mult": 1.5, "ignite_range_atr[R]": 1.5,
              "retrace_cap": 0.5, "max_pullback_bars": 2, "target_R": 2.0,
              "window[R]": "09:35-15:00", "source": "TradingSim / HowToTrade",
              "deviation": "source triggers intrabar on a stop order; converted to bar CLOSE"}

    def evaluate(self, ctx):
        t = ctx.bar_ts.time()
        if not (_t(9, 35) <= t < _t(15, 0)) or len(ctx.bars_5m) < 4                 or ctx.ind_len("5m") < 22:
            return None
        h, l, c, v = ctx.ind_ohlcv("5m")
        a = ind.atr(h, l, c, 14)
        if a is None:
            return None
        # Pattern is scanned on the SEEDED series so the 20-bar volume baseline exists
        # from the open, but every pattern bar must belong to TODAY -- an igniting bar
        # from yesterday's close would be a different trade entirely.
        bars = ctx.ind_5m
        off = ctx.today_offset("5m")
        trig = bars[-1]
        for npb in (1, 2):
            i = len(bars) - 2 - npb
            if i < max(off, 20):
                continue
            ig = bars[i]
            pbs = list(bars[i + 1:-1])
            if len(pbs) != npb:
                continue
            rng = ig["high"] - ig["low"]
            if rng <= 0 or rng < 1.5 * a:
                continue
            vol_prior = [x["volume"] for x in bars[i - 20:i]]
            if not vol_prior or ig["volume"] < 1.5 * (sum(vol_prior) / len(vol_prior)):
                continue
            inside = all(p["high"] < ig["high"] and p["low"] > ig["low"] for p in pbs)
            if not inside:
                continue
            # long side: igniting bar closes in the top 20% of its range
            if ig["close"] >= ig["low"] + 0.8 * rng:
                if all(p["low"] >= ig["high"] - 0.5 * rng for p in pbs):
                    hi = max(p["high"] for p in pbs)
                    if trig["close"] > hi:
                        stop = pbs[-1]["low"]
                        if stop < trig["close"]:
                            return Signal(self.id, "long", stop=stop,
                                          target=trig["close"] + 2.0 * (trig["close"] - stop),
                                          state={"pullback_bars": npb, "atr14": a})
            # short side
            if ig["close"] <= ig["low"] + 0.2 * rng:
                if all(p["high"] <= ig["low"] + 0.5 * rng for p in pbs):
                    lo = min(p["low"] for p in pbs)
                    if trig["close"] < lo:
                        stop = pbs[-1]["high"]
                        if stop > trig["close"]:
                            return Signal(self.id, "short", stop=stop,
                                          target=trig["close"] - 2.0 * (stop - trig["close"]),
                                          state={"pullback_bars": npb, "atr14": a})
        return None


class CrabelStretch(Setup):
    id, version, timeframe = "Crabel_Stretch", "1", "1m"
    max_per_day = 1
    params = {"lookback_days": 10, "stretch_mult": 1.0, "window[R]": "09:31-11:00",
              "exit": "opposite level or 15:55", "source": "Crabel TASC V.6:9",
              "deviation": "resting stop order converted to bar CLOSE",
              "overlap_warning": "strongly correlated with ORB_5min"}

    def evaluate(self, ctx):
        t = ctx.bar_ts.time()
        if not (_t(9, 31) <= t <= _t(11, 0)):
            return None
        s = ctx.warmup.get("crabel_stretch")
        if not s or ctx.session_open is None:
            return None
        buy = ctx.session_open + 1.0 * s
        sell = ctx.session_open - 1.0 * s
        c = ctx.price
        if c >= buy:
            return Signal(self.id, "long", stop=sell, target=None,
                          state={"buy": buy, "sell": sell, "stretch": s})
        if c <= sell:
            return Signal(self.id, "short", stop=buy, target=None,
                          state={"buy": buy, "sell": sell, "stretch": s})
        return None


class GapFade(Setup):
    id, version, timeframe = "Gap_Fade", "1", "5m"
    max_per_day = 1
    params = {"min_gap_pct": 0.5, "buffer_pct[R]": 0.05, "window[R]": "09:45-12:00",
              "target": "prior RTH close", "source": "SharePlanner / QuantifiedStrategies",
              "evidence": "6005 QQQ sessions: 0.5-0.99% gaps fill 77% (down) / 72% (up); "
                          "avg open-to-close after >=1% gap up = -0.5%"}

    def evaluate(self, ctx):
        t = ctx.bar_ts.time()
        if not (_t(9, 45) <= t <= _t(12, 0)) or not ctx.prior_day or ctx.session_open is None:
            return None
        pc = ctx.prior_day["close"]
        if pc <= 0:
            return None
        gap = ctx.session_open / pc - 1.0
        if abs(gap) < 0.005:
            return None
        o, c = ctx.session_open, ctx.price
        if gap > 0 and c < o * (1 - 0.0005):
            stop = max(b["high"] for b in ctx.bars_1m)
            if stop <= c:
                return None
            return Signal(self.id, "short", stop=stop, target=pc,
                          state={"gap_pct": 100 * gap, "prev_close": pc, "open": o})
        if gap < 0 and c > o * (1 + 0.0005):
            stop = min(b["low"] for b in ctx.bars_1m)
            if stop >= c:
                return None
            return Signal(self.id, "long", stop=stop, target=pc,
                          state={"gap_pct": 100 * gap, "prev_close": pc, "open": o})
        return None


# --------------------------------------------------------------------------- registry

ALL_SETUPS: list[Setup] = [
    ORB5min(), ORB15min(), VWAPReclaim(), MomoChase(),
    IntradayMomentumBoundary(), TTMSqueeze(), PDHPDLBreakout(), PDHPDLFailedBreak(),
    VWAP2SigmaFade(), EMA920Pullback(), ThreeBarPlay(), CrabelStretch(), GapFade(),
]

# Setups whose MEASURED signal rate makes a verdict unreachable in a reasonable horizon.
# Kept and recorded -- deleting them would be the selective reporting the spec forbids --
# but labelled so an early reading is never mistaken for evidence.
#
# These sets are set from the replay calibration (measured signals per symbol-day), NEVER
# from performance. Reclassifying on P&L would be exactly the selection this lab exists to
# avoid. Reclassifying on frequency is a feasibility fact, knowable before any trade exists.
# Measured 2026-08-27 over 40 symbol-days (QQQ+SPY, 20 sessions), post indicator-seeding fix.
# Months to n=377 on two symbols: Gap_Fade 32.6, EMA_9_20_Pullback 35.9.
SLOW_SETUPS = {"Gap_Fade", "EMA_9_20_Pullback"}

# Measured to fire so rarely they cannot accumulate evidence at all, over 40 symbol-days:
#
#   VWAP_2sigma_Fade   0 signals.  Funnel over 8 QQQ sessions: 256 in-window bars
#                      -> 156 pass ADX<=25 -> 12 pass RVOL>=1.5 -> 1 touches the 2-sigma
#                      band -> 0 pass the wick filter. The binding constraint is the RVOL
#                      1.5x threshold, which was OUR [R] choice copied from ORB_5min,
#                      compounding with the source's 2.0-sigma band and our 0.5 wick filter.
#   ThreeBarPlay       1 signal in 40 symbol-days -> ~359 months to a verdict. The binding
#                      constraint is the 1.5x ATR igniting-range filter, also OUR [R].
#
# Both could be made to fire by loosening a parameter we invented. Neither will be. That is
# tuning, and tuning is how this project manufactured its earlier false positives.
#
# They are NOT removed from the family. m stays 13. Shrinking the family because a member
# cannot fire would lower the bar for every surviving setup -- the same move refused earlier
# when an API outage killed a pre-registered test and the family was held at m=5. Keeping
# them costs ~2% on the required n; the precedent would cost far more.
DEAD_SETUPS = {"VWAP_2sigma_Fade", "ThreeBarPlay"}

by_id = {s.id: s for s in ALL_SETUPS}
