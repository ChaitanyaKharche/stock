"""Evaluate many MOMO_CHASE parameterizations against one session, cheaply and faithfully.

THE ONE DESIGN DECISION THAT MATTERS
------------------------------------
Sessions are the OUTER loop and parameter cells are the INNER loop.

The obvious way round -- for each cell, scan the history -- rebuilds every indicator for
every cell. `Panel` is a Python loop over ~1,100 bars (session + 2 prefix sessions) and it
dominates the cost, so a 200k-cell grid would rebuild it 200k times.

Inverted, `Panel` is built ONCE per session and all cells read it. That is a speedup of
roughly the grid size, and it buys something worth more than speed: **the indicators are
computed by the proven code, not by a vectorized re-implementation of it.** A hand-rolled
numpy ADX that differs from `journal_features.Panel._adx` in the Wilder seeding would shift
every cell by a little, the V0 cell would fail to reproduce `momo_v2.py`, and the failure
would look like a bug in the sweep rather than a bug in the indicators. Reusing `Panel`
removes that entire class of error by construction.

WHAT IS SWEPT AND WHAT IS NOT
-----------------------------
Swept: the DECISION parameters -- lookback, threshold, holding time, which gates are on,
time window, and whether the rule chases momentum or fades it.

NOT swept: `macd=(9,17,9)` and `dmi_len=14`. These are not free parameters. They come from
the journal study's measurement of the trader's own behaviour (69.4% of his entries had
MACD aligned, 78.5% DMI), so they were fit to BEHAVIOUR and never to outcomes. Sweeping
them would convert a behavioural constant into a fitted one and quietly inflate the search
space that the multiple-testing correction has to pay for. They also happen to be the only
axes that would force `Panel` to be rebuilt per cell.

`direction` IS swept, and deliberately. The whole family assumes momentum CONTINUES. The
opposite reading -- that a stretched 15-minute move reverts -- is the same signal with the
sign flipped, costs one bit to test, and is the single most likely way this setup is
mis-specified. Testing only `chase` would be assuming the answer.

PER-SESSION AGGREGATION, AND WHY NOT PER-TRADE
----------------------------------------------
`run_cell` returns per-session SUMS, not individual trades. Individual trades across a full
grid do not fit anywhere: 200k cells x ~2 trades x 3,815 sessions is order 1e9 rows. Session
sums are also the correct granularity for the inference -- intraday origins inside a session
are ~0.9 autocorrelated, so the effective sample size is SESSIONS, and every test downstream
(session-clustered bootstrap, SPA) consumes a session-level series anyway.

COSTS ARE NOT OPTIONAL AND ARE NOT APPLIED HERE
-----------------------------------------------
This returns GROSS P&L and a trade count per session, so a cost model can be applied later
without re-running the grid. That separation is deliberate but it is also a trap, so it is
stated loudly: this project has already measured a setup at gross +$0.0205 and net -$0.0171.
**A gross number from this engine is not a result.** `n_trades` is returned alongside
precisely so the cost charge is always computable.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import NamedTuple

import numpy as np

from trade_analysis.backtesting.journal_features import Panel, bars_5m

NOTIONAL = 10_000.0
PREFIX = 2                      # prior sessions used to seed indicators, as in momo_v2


class Cell(NamedTuple):
    """One parameterization. Hashable, so it doubles as a dict key and a stable label."""
    trail_min: int              # momentum lookback in minutes
    sigma_mult: float           # |z| entry threshold, in trailing-sigma units
    time_exit: int              # minutes held
    adx_min: float              # 0.0 disables the ADX gate
    dmi_tf: str                 # "off" | "1m" | "5m"
    macd_gate: bool
    win_start: int              # minutes from midnight, exchange-local, inclusive
    win_end: int
    direction: str              # "chase" (momentum) | "fade" (reversion)
    ema9_min: float             # NaN disables; else required px-vs-EMA9 distance in ATRs
    max_per_day: int            # 0 = unlimited (cooldown still applies)
    gate_ref: str               # "trade" | "momentum" -- which sign the gates confirm
    sig_norm: str               # "session" | "tod" -- what the threshold is measured in
    vol_regime: str             # "off" | "high" | "low" -- prior-day volatility tercile


class Prep(NamedTuple):
    """Everything derivable from a session WITHOUT knowing the cell."""
    n_bars: int
    tod: np.ndarray             # minutes from midnight per 1-min bar
    open_: np.ndarray
    close: np.ndarray
    sigma1: np.ndarray          # running SD of 1-min returns, bars 0..k, ddof=1
    p1: Panel
    p5: Panel
    n1: int                     # offset of this session inside the prefixed 1-min panel
    n5: int
    tod_sig: dict               # {trail_min: array} 14-session mean |L-min return| per bar
    vol_rank: float             # prior-day volatility's rank in the trailing window, [0,1]


def prepare(sess: list, prior: list, tod_sig: dict | None = None,
            vol_rank: float = float("nan")) -> Prep:
    """Build both panels and the parameter-free running sigma, once.

    `sigma1[k]` is the standard deviation of every 1-minute return from the session open
    through bar k. It does NOT depend on any swept parameter, so it is computed here rather
    than inside the cell loop -- which matters because it is the only O(k) quantity in the
    inner loop and would otherwise dominate.

    It is computed with `np.std(..., ddof=1)` on the actual slice rather than from running
    sums of squares. The cumulative-sums identity is algebraically equal and numerically
    is NOT: these returns are ~1e-4 and the sum-of-squares form cancels catastrophically.
    Matching `momo_v2.py` bit-for-bit is the point, so the slower, stabler form wins.
    """
    pre1 = [b for s in prior for b in s]
    pre5 = [b for s in prior for b in bars_5m(s)]
    p1 = Panel(pre1 + sess)
    p5 = Panel(pre5 + bars_5m(sess))

    close = np.array([b["close"] for b in sess], dtype=np.float64)
    open_ = np.array([b["open"] for b in sess], dtype=np.float64)
    tod = np.array([b["ts"].hour * 60 + b["ts"].minute for b in sess], dtype=np.int32)

    n = len(sess)
    sigma1 = np.zeros(n, dtype=np.float64)
    if n > 3:
        rets = np.diff(close) / close[:-1]
        for k in range(3, n):
            sigma1[k] = np.std(rets[:k], ddof=1)
    return Prep(n, tod, open_, close, sigma1, p1, p5, len(pre1), len(pre5),
                tod_sig or {}, vol_rank)


def run_cell(pp: Prep, c: Cell) -> tuple[float, int, float]:
    """Gross P&L, trade count, and TOTAL SHARES TRADED for one cell on one session.

    Mirrors `momo_v2.scan` exactly for the V0 parameters; every difference is a swept axis.

    Shares are accumulated because the cost charge is per SHARE, not per dollar of notional.
    QQQ and SPY are penny-wide, so a round trip costs about $0.01 x shares regardless of
    price -- which means the same $10,000 position costs $1.00 to turn over when QQQ is $100
    and $0.14 when it is $700. Over 2016-2026 QQQ spans roughly that whole range, so
    charging a flat basis-point rate would understate cost in the early sample and overstate
    it in the late one, in a way that correlates with time and therefore with regime.
    Returning the exact share count lets the cost be applied afterwards without re-running
    the grid, and lets a BREAKEVEN spread be solved for per cell.
    """
    hi = pp.n_bars - c.time_exit - 2
    if hi <= 30:
        return 0.0, 0, 0.0
    # WHOLE-SESSION GATE, evaluated once. `vol_rank` is where the PRIOR session's range sits
    # in the trailing window -- prior only, never including today, because a regime gate
    # computed from today's range would be the exact lookahead that voided this project's
    # first decade test. A session with no history yet is skipped rather than admitted.
    if c.vol_regime != "off":
        r = pp.vol_rank
        if r != r:                                       # NaN: not enough prior sessions
            return 0.0, 0, 0.0
        if c.vol_regime == "high" and r < 2.0 / 3.0:
            return 0.0, 0, 0.0
        if c.vol_regime == "low" and r >= 1.0 / 3.0:
            return 0.0, 0, 0.0

    sq = math.sqrt(c.trail_min)
    lag = c.trail_min + 1                 # momo_v2's `c1[-16]` for a 15-minute lookback
    pnl, n, last, today, sh_tot = 0.0, 0, -10 ** 9, 0, 0.0

    for k in range(30, hi):
        if c.max_per_day and today >= c.max_per_day:
            break
        if not (c.win_start <= pp.tod[k] <= c.win_end):
            continue
        if k % 5 != 4:                    # 5-minute setup: only bucket closes are decisions
            continue
        j5 = k // 5 - 1
        if j5 < 3 or k < lag:
            continue
        J5, K = pp.n5 + j5, pp.n1 + k

        a5 = pp.p5.atr[J5]
        if not a5 or a5 <= 0:
            continue
        if pp.p5.ema9[J5] is None or pp.p5.macd[J5] is None or pp.p5.adx[J5] is None:
            continue
        if pp.p5.dip[J5] is None or pp.p1.dip[K] is None:
            continue

        px = pp.close[k]
        raw = px / pp.close[k - c.trail_min] - 1.0
        if c.sig_norm == "tod":
            # TIME-OF-DAY CONDITIONAL. The 14-session trailing mean of |L-minute return
            # ending at THIS minute|, so 10:35 is judged against past 10:35s rather than
            # against the whole session. Intraday volatility has a pronounced U-shape, so a
            # session-wide sigma makes a fixed threshold far easier to clear near the open
            # than at midday -- the threshold silently becomes a time-of-day filter. This is
            # Zarattini's boundary construction generalised from the from-open move to the
            # trailing L-minute move the momo family actually uses.
            ts = pp.tod_sig.get(c.trail_min)
            if ts is None or ts[k] <= 0:
                continue
            z = raw / ts[k]
        else:
            s1 = pp.sigma1[k]
            if s1 <= 0:
                continue
            z = raw / (s1 * sq)
        if abs(z) < c.sigma_mult:
            continue

        # The one-bit hypothesis flip. "chase" trades with the move, "fade" against it.
        m = 1.0 if z > 0 else -1.0                          # the raw momentum sign
        d = m if c.direction == "chase" else -m             # the direction actually traded

        # WHICH SIGN MUST THE GATES CONFIRM? For "chase" the two are identical and this
        # axis is inert. For "fade" they are opposite hypotheses and both are real:
        #   gate_ref="trade"     MACD/DMI/EMA9 agree with the SHORT -> a divergence fade
        #                        ("price rose but momentum is already rolling over")
        #   gate_ref="momentum"  MACD/DMI/EMA9 agree with the RALLY, and we short it anyway
        #                        -> an exhaustion fade ("strong confirmed move, bet it ends")
        # Only the first was reachable before this axis existed, which would have let the
        # sweep report "reversion does not work" while never having tested the version of
        # reversion that most practitioners actually mean.
        g = d if c.gate_ref == "trade" else m

        if c.macd_gate and pp.p5.macd[J5][2] * g <= 0:
            continue
        if c.dmi_tf == "5m":
            if (pp.p5.dip[J5] - pp.p5.din[J5]) * g <= 0:
                continue
        elif c.dmi_tf == "1m":
            if (pp.p1.dip[K] - pp.p1.din[K]) * g <= 0:
                continue
        if c.adx_min > 0 and pp.p5.adx[J5] <= c.adx_min:
            continue
        if c.ema9_min == c.ema9_min:                        # not NaN -> gate is on
            if g * (px - pp.p5.ema9[J5]) / a5 < c.ema9_min:
                continue
        if k - last < c.time_exit:                          # no overlapping positions
            continue

        last = k
        entry = pp.open_[k + 1]                             # fill at the NEXT bar's open
        if entry <= 0:
            continue
        ex = pp.open_[min(k + 1 + c.time_exit, pp.n_bars - 1)]
        sh = NOTIONAL / entry
        pnl += sh * (ex - entry) * d
        sh_tot += sh
        n += 1
        today += 1
    return pnl, n, sh_tot


def v0_cell() -> Cell:
    """The frozen MOMO_CHASE, as `momo_v2.py` V0 computes it.

    This cell exists to be checked, not to be swept. If the sweep's V0 does not reproduce
    `momo_v2.py`'s V0 to the cent, nothing else the sweep says is worth reading.
    Note `max_per_day=0`: momo_v2's scan caps overlap with the 25-minute cooldown only and
    does NOT apply the frozen setup's 3-per-day limit, so reproducing it means not applying
    it either.
    """
    return Cell(trail_min=15, sigma_mult=0.60, time_exit=25, adx_min=20.0,
                dmi_tf="5m", macd_gate=True,
                win_start=10 * 60 + 30, win_end=14 * 60 + 30,
                direction="chase", ema9_min=float("nan"), max_per_day=0,
                gate_ref="trade", sig_norm="session", vol_regime="off")


def _ts_ok(sess) -> bool:
    return bool(sess) and isinstance(sess[0].get("ts"), dt.datetime)
