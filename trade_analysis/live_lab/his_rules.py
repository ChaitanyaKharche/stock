"""The trader's rules as he described them on 2026-10-26, run against history.

    python -m trade_analysis.live_lab.his_rules --symbol QQQ                      # underlying bp
    python -m trade_analysis.live_lab.his_rules --symbol SPY --options --end 2023-12-29

Pre-registration: research/his_rules_preregistration.md. Every number below is frozen
there; nothing in this file is tuned after a run. New file, imports proven code
(six_lines.build_six, orb_veto.resample_10m, positions.FEE_PER_CONTRACT_SIDE,
breakout_sweep.boot_mean), edits none of it.

THE RULES, mechanically
-----------------------
Lines      the six lines (six_lines.build_six). From 12:30 two TEMPORARY lines join: the
           09:30-12:30 high and low, each only if it sits inside the six (below R3 /
           above S3). They die with the session.
Bars       10-minute RTH bars from 1-minute data.
Signal     a bar CLOSES beyond a line, its range >= RANGE_MULT x the median range of the
           previous LOOKBACK_BARS ten-minute bars (rolling across sessions), and its
           volume >= VOL_MULT x the median volume of the same clock-time bar over the
           previous LOOKBACK_SESSIONS sessions. Both baselines must be full.
Morning    R1 or S1 only. Entry at the NEXT bar's open, whose open time is 09:40-10:45.
Afternoon  any of the eight lines not close-broken before 12:30. Entry open 12:40-13:00.
           Only if the morning position (if any) is already closed. Max two trades a day.
Pullback   arm B, secondary: R1 close-broke without qualifying, price reached R2 within
           PULLBACK_MIN, then a bar dips to within PULLBACK_BP of R1 and closes above it.
Exits      ladder on the OPTION price: 2 contracts at +50%, 2 at +75%, 1 at +100%.
           Loser: a bar closing back inside the line sells STOP_FIRST contracts at that
           close; a second such close sells the rest. 15:55 sells whatever is left.
           Exit arm 1 (secondary): all 5 at +50%.
Options    SPY 0DTE from the archive: buy at the ask at the entry minute, sell at the bid
           (ladder legs at the limit once the bid reaches it). ATM; +-1 strike as
           sensitivity. 2024 is the held-out block and this file REFUSES to read it.
Underlying the same ladder translated to bp of underlying move at delta ~0.5:
           +100% ~ UNDERLYING_BP_PER_100PCT, so +50% ~ 20 bp, as six_lines_results used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
from collections import defaultdict, deque
from pathlib import Path
from typing import Sequence

from .breakout_sweep import boot_mean
from .orb_veto import resample_10m
from .positions import FEE_PER_CONTRACT_SIDE
from .six_lines import RTH_FROM, RTH_TO, Line, _hi_lo, build_six

# ------------------------------------------------------------------ frozen parameters
RANGE_MULT = 1.5
VOL_MULT = 1.5
LOOKBACK_BARS = 20
LOOKBACK_SESSIONS = 20
MORNING = (dt.time(9, 40), dt.time(10, 45))      # entry bar OPEN time, inclusive
AFTERNOON = (dt.time(12, 40), dt.time(13, 0))
TEMP_FROM, TEMP_TO = dt.time(9, 30), dt.time(12, 30)
PULLBACK_MIN = 30
PULLBACK_BP = 5.0
CONTRACTS = 5
LADDER = ((2, 0.50), (2, 0.75), (1, 1.00))       # (contracts, gain on option price)
LADDER_FLAT = ((5, 0.50),)                        # exit arm 1
STOP_FIRST = 3                                    # contracts sold at the first close back inside
EOD = dt.time(15, 55)
UNDERLYING_BP_PER_100PCT = 40.0                   # +100% on an ATM 0DTE ~ 40 bp of spot
HELD_OUT_FROM = dt.date(2024, 1, 1)               # SPY 2024 block: touched exactly once, not here
BAR_MIN = 10


# ------------------------------------------------------------------ baselines

class Baselines:
    """Rolling range baseline (previous LOOKBACK_BARS bars, across sessions) and
    per-clock-time volume baseline (previous LOOKBACK_SESSIONS sessions)."""

    def __init__(self):
        self.ranges: deque = deque(maxlen=LOOKBACK_BARS)
        self.vol: dict = defaultdict(lambda: deque(maxlen=LOOKBACK_SESSIONS))

    def rate(self, bar: dict) -> tuple[float | None, float | None]:
        """-> (range_ratio, vol_ratio); None where the baseline is not yet full.
        Pushes the bar's range AFTER rating it, so a bar never rates against itself."""
        rr = vr = None
        if len(self.ranges) == LOOKBACK_BARS:
            med = st.median(self.ranges)
            rr = (bar["high"] - bar["low"]) / med if med > 0 else None
        hist = self.vol[bar["ts"].time()]
        if len(hist) == LOOKBACK_SESSIONS:
            med = st.median(hist)
            vr = bar.get("volume", 0.0) / med if med > 0 else None
        self.ranges.append(bar["high"] - bar["low"])
        return rr, vr

    def end_session(self, ten: Sequence[dict]) -> None:
        for b in ten:
            self.vol[b["ts"].time()].append(b.get("volume", 0.0))


def rate_bars(ten: Sequence[dict], base: Baselines) -> list[dict]:
    out = []
    for b in ten:
        rr, vr = base.rate(b)
        out.append({"range_ratio": rr, "vol_ratio": vr,
                    "qualifies": rr is not None and vr is not None
                    and rr >= RANGE_MULT and vr >= VOL_MULT})
    return out


# ------------------------------------------------------------------ lines and breaks

def temp_lines(rth_1m: Sequence[dict], six: Sequence[Line]) -> list[Line]:
    """The 09:30-12:30 high and low, each only if it sits INSIDE the six."""
    hi, lo = _hi_lo(rth_1m, TEMP_FROM, TEMP_TO)
    if hi is None or not six:
        return []
    r_top = max(l.price for l in six if l.side == "R")
    s_bot = min(l.price for l in six if l.side == "S")
    out = []
    if hi < r_top:
        out.append(Line("TR", hi, "today 09:30-12:30 high", "R"))
    if lo > s_bot:
        out.append(Line("TS", lo, "today 09:30-12:30 low", "S"))
    return out


def closes_beyond(bar: dict, line: Line) -> bool:
    return bar["close"] > line.price if line.side == "R" else bar["close"] < line.price


def back_inside(bar: dict, line: Line) -> bool:
    return bar["close"] < line.price if line.side == "R" else bar["close"] > line.price


def is_break(ten: Sequence[dict], i: int, line: Line) -> bool:
    """Bar i CROSSES the line: closes beyond it, and the reference before it (the previous
    bar's close, or this bar's own open for the first bar) was not beyond. A session that
    opens already past R1 has not broken R1; six_lines calls that line pre-broken."""
    ref = ten[i - 1]["close"] if i > 0 else ten[i]["open"]
    was_beyond = ref > line.price if line.side == "R" else ref < line.price
    return closes_beyond(ten[i], line) and not was_beyond


def broken_before(ten: Sequence[dict], lines: Sequence[Line], until: dt.time) -> set:
    return {l.name for l in lines
            if any(closes_beyond(b, l) for b in ten if b["ts"].time() < until)}


def _in_window(t: dt.time, window) -> bool:
    return window[0] <= t <= window[1]


def first_qualifying(ten: Sequence[dict], rated: Sequence[dict], candidates: Sequence[Line],
                     window, arm: str, label: str) -> dict | None:
    """The first bar that close-breaks one of `candidates` AND qualifies, whose NEXT bar
    opens inside `window`. One signal; the next bar is the entry bar."""
    for i, bar in enumerate(ten[:-1]):
        entry_t = ten[i + 1]["ts"].time()
        if entry_t > window[1]:
            break
        if not _in_window(entry_t, window):
            continue
        for line in candidates:
            if is_break(ten, i, line) and rated[i]["qualifies"]:
                return {"signal_i": i, "entry_i": i + 1, "line": line.name,
                        "line_price": line.price, "side": line.side,
                        "direction": "long" if line.side == "R" else "short",
                        "window": label, "arm": arm,
                        "range_ratio": round(rated[i]["range_ratio"], 3),
                        "vol_ratio": round(rated[i]["vol_ratio"], 3)}
    return None


def morning_signal(ten, rated, six) -> dict | None:
    r1s1 = [l for l in six if l.name in ("R1", "S1")]
    return first_qualifying(ten, rated, r1s1, MORNING, "A", "morning")


def afternoon_signal(ten, rated, six, temps) -> dict | None:
    gone = broken_before(ten, six, TEMP_TO)
    live = [l for l in list(six) + list(temps) if l.name not in gone]
    return first_qualifying(ten, rated, live, AFTERNOON, "A", "afternoon")


def pullback_signal(ten, six) -> dict | None:
    """Arm B: R1 (S1) close-broke, price reached R2 (S2) within PULLBACK_MIN, then a bar
    comes back to within PULLBACK_BP of R1 and closes beyond it. Entry next bar's open,
    morning window. Size and volume filters do NOT apply here; that is the point of
    the arm."""
    by = {l.name: l for l in six}
    for one, two in (("R1", "R2"), ("S1", "S2")):
        l1, l2 = by.get(one), by.get(two)
        if l1 is None or l2 is None:
            continue
        j = next((i for i in range(len(ten)) if is_break(ten, i, l1)), None)
        if j is None:
            continue
        horizon = ten[j]["ts"] + dt.timedelta(minutes=PULLBACK_MIN)
        reached = any((b["high"] >= l2.price if l1.side == "R" else b["low"] <= l2.price)
                      for b in ten[j:] if b["ts"] <= horizon)
        if not reached:
            continue
        tol = l1.price * PULLBACK_BP / 10_000.0
        for k in range(j + 1, len(ten) - 1):
            b = ten[k]
            entry_t = ten[k + 1]["ts"].time()
            if entry_t > MORNING[1]:
                break
            touched = (b["low"] <= l1.price + tol) if l1.side == "R" else (b["high"] >= l1.price - tol)
            if touched and closes_beyond(b, l1) and _in_window(entry_t, MORNING):
                return {"signal_i": k, "entry_i": k + 1, "line": one, "line_price": l1.price,
                        "side": l1.side, "direction": "long" if l1.side == "R" else "short",
                        "window": "morning", "arm": "B", "range_ratio": None, "vol_ratio": None}
    return None


# ------------------------------------------------------------------ exits, underlying

def simulate_underlying(ten: Sequence[dict], entry_i: int, direction: str, line_price: float,
                        side: str, ladder=LADDER, contracts: int = CONTRACTS) -> dict:
    """The ladder in bp of underlying, the partial stop on a close back inside, EOD.
    Precedence inside a bar: ladder (intrabar limit) before the close stop."""
    long = direction == "long"
    sgn = 1 if long else -1
    entry = ten[entry_i]["open"]
    line = Line("x", line_price, "", side)

    def bp(px):
        return (px / entry - 1) * 10_000.0 * sgn

    remaining, legs, li, stops, mfe = contracts, [], 0, 0, 0.0
    for b in ten[entry_i:]:
        fav = b["high"] if long else b["low"]
        mfe = max(mfe, bp(fav))
        while li < len(ladder) and remaining > 0:
            n, gain = ladder[li]
            lvl = entry * (1 + sgn * gain * UNDERLYING_BP_PER_100PCT / 10_000.0)
            if (long and b["high"] >= lvl) or (not long and b["low"] <= lvl):
                k = min(n, remaining)
                legs.append({"n": k, "bp": round(bp(lvl), 2), "why": f"target{li + 1}",
                             "ts": b["ts"].isoformat()})
                remaining -= k
                li += 1
            else:
                break
        if remaining == 0:
            break
        if back_inside(b, line):
            stops += 1
            k = min(STOP_FIRST, remaining) if stops == 1 else remaining
            legs.append({"n": k, "bp": round(bp(b["close"]), 2),
                         "why": "stop" if stops == 1 else "stop_rest", "ts": b["ts"].isoformat()})
            remaining -= k
            if remaining == 0:
                break
    if remaining > 0:
        legs.append({"n": remaining, "bp": round(bp(ten[-1]["close"]), 2), "why": "eod",
                     "ts": ten[-1]["ts"].isoformat()})
    return {"entry": round(entry, 4), "legs": legs, "mfe_bp": round(mfe, 2),
            "weighted_bp": round(sum(l["n"] * l["bp"] for l in legs) / contracts, 3),
            "exit_ts": legs[-1]["ts"]}


# ------------------------------------------------------------------ exits, options

class OptionBook:
    """Minute NBBO for one session's 0DTE chain. Rows: ts, strike, right ('C'/'P'), bid, ask."""

    def __init__(self, rows):
        self.q: dict = defaultdict(list)
        for r in rows:
            right = str(r["right"]).upper()[:1]
            self.q[(float(r["strike"]), right)].append((r["ts"], float(r["bid"]), float(r["ask"])))
        for k in self.q:
            self.q[k].sort(key=lambda x: x[0])

    @classmethod
    def from_frame(cls, df):
        rows = ({"ts": t.to_pydatetime(), "strike": k, "right": r, "bid": b, "ask": a}
                for t, k, r, b, a in zip(df["timestamp"], df["strike"], df["right"],
                                         df["bid"], df["ask"]))
        return cls(rows)

    def strikes(self, right: str) -> list[float]:
        return sorted(k for (k, r) in self.q if r == right)

    def quote_at(self, strike: float, right: str, ts: dt.datetime, max_age_min: int = 2):
        path = self.q.get((strike, right), [])
        best = None
        for t, b, a in path:
            if t <= ts:
                best = (t, b, a)
            else:
                break
        if best is None or (ts - best[0]) > dt.timedelta(minutes=max_age_min):
            return None
        return best[1], best[2]

    def path(self, strike: float, right: str, t0: dt.datetime, t1: dt.datetime):
        return [(t, b, a) for (t, b, a) in self.q.get((strike, right), []) if t0 <= t <= t1]


def pick_strike(book: OptionBook, right: str, spot: float, offset: int = 0) -> float | None:
    """ATM = nearest strike to spot. offset +1 = one strike further OUT of the money
    (higher for calls, lower for puts); -1 = one strike further in."""
    ks = book.strikes(right)
    if not ks:
        return None
    i = min(range(len(ks)), key=lambda j: abs(ks[j] - spot))
    i += offset if right == "C" else -offset
    return ks[i] if 0 <= i < len(ks) else None


def simulate_option(book: OptionBook, ten: Sequence[dict], entry_i: int, direction: str,
                    line_price: float, side: str, ladder=LADDER, contracts: int = CONTRACTS,
                    strike_offset: int = 0) -> dict | None:
    """Buy at the ask at the entry minute; ladder legs sell at the limit once the bid
    reaches it; the close-back-inside stop and 15:55 sell at the bid. None = unpriced."""
    long = direction == "long"
    right = "C" if long else "P"
    t0 = ten[entry_i]["ts"]
    spot = ten[entry_i]["open"]
    strike = pick_strike(book, right, spot, strike_offset)
    if strike is None:
        return None
    q0 = book.quote_at(strike, right, t0)
    if q0 is None or q0[1] <= 0:
        return None
    entry_px = q0[1]
    line = Line("x", line_price, "", side)
    eod = dt.datetime.combine(t0.date(), EOD)
    # bar-close stop times: a bar's verdict is known at its end
    stop_at = [(b["ts"] + dt.timedelta(minutes=BAR_MIN), back_inside(b, line))
               for b in ten[entry_i:] if b["ts"] + dt.timedelta(minutes=BAR_MIN) <= eod]
    remaining, legs, li, stops, si = contracts, [], 0, 0, 0
    last_bid = None
    for t, bid, ask in book.path(strike, right, t0, eod):
        last_bid = bid
        while li < len(ladder) and remaining > 0:
            n, gain = ladder[li]
            lvl = entry_px * (1 + gain)
            if bid >= lvl:
                k = min(n, remaining)
                legs.append({"n": k, "px": round(lvl, 4), "why": f"target{li + 1}", "ts": t.isoformat()})
                remaining -= k
                li += 1
            else:
                break
        if remaining == 0:
            break
        while si < len(stop_at) and stop_at[si][0] <= t:
            _, back = stop_at[si]
            si += 1
            if back and remaining > 0:
                stops += 1
                k = min(STOP_FIRST, remaining) if stops == 1 else remaining
                legs.append({"n": k, "px": round(bid, 4), "why": "stop" if stops == 1 else "stop_rest",
                             "ts": t.isoformat()})
                remaining -= k
        if remaining == 0:
            break
    if remaining > 0:
        if last_bid is None:
            return None
        legs.append({"n": remaining, "px": round(last_bid, 4), "why": "eod", "ts": eod.isoformat()})
    gross = sum(l["n"] * (l["px"] - entry_px) for l in legs) * 100.0
    fees = 2.0 * FEE_PER_CONTRACT_SIDE * contracts
    return {"strike": strike, "right": right, "entry_px": round(entry_px, 4),
            "entry_spread_pct": round((q0[1] - q0[0]) / entry_px * 100, 2),
            "legs": legs, "pnl_gross": round(gross, 2), "pnl_net": round(gross - fees, 2),
            "exit_ts": legs[-1]["ts"]}


# ------------------------------------------------------------------ one session

def session_trades(ten: Sequence[dict], rth_1m: Sequence[dict], six: Sequence[Line],
                   base: Baselines, book: OptionBook | None = None) -> list[dict]:
    """All trades for one session, every arm, underlying exits always, option exits when
    a book is given. Updates `base` for the next session."""
    rated = rate_bars(ten, base)
    temps = temp_lines(rth_1m, six)
    out = []
    morning = morning_signal(ten, rated, six)
    pull = pullback_signal(ten, six) if morning is None else None
    busy_until = None
    for sig in (morning, pull):
        if sig is None:
            continue
        out.append(_trade(sig, ten, book))
        if sig["arm"] == "A":
            busy_until = dt.datetime.fromisoformat(out[-1]["underlying"]["exit_ts"])
    aft = afternoon_signal(ten, rated, six, temps)
    if aft is not None and (busy_until is None or ten[aft["entry_i"]]["ts"] > busy_until):
        out.append(_trade(aft, ten, book))
    base.end_session(ten)
    return out


def _trade(sig: dict, ten, book) -> dict:
    t = dict(sig)
    t["entry_ts"] = ten[sig["entry_i"]]["ts"].isoformat()
    t["underlying"] = simulate_underlying(ten, sig["entry_i"], sig["direction"],
                                          sig["line_price"], sig["side"])
    t["underlying_flat"] = simulate_underlying(ten, sig["entry_i"], sig["direction"],
                                               sig["line_price"], sig["side"], LADDER_FLAT)
    if book is not None:
        for name, off, lad in (("opt_atm", 0, LADDER), ("opt_atm_flat", 0, LADDER_FLAT),
                               ("opt_otm1", 1, LADDER), ("opt_itm1", -1, LADDER)):
            t[name] = simulate_option(book, ten, sig["entry_i"], sig["direction"],
                                      sig["line_price"], sig["side"], lad, strike_offset=off)
    return t


# ------------------------------------------------------------------ the run

def run(symbol: str, start: dt.date, end: dt.date, options: bool) -> list[dict]:
    from .feed import ThetaLiveFeed
    from .levels_test import get_ext
    from .walkforward import sessions_between

    if options and end >= HELD_OUT_FROM:
        raise SystemExit(f"refusing: --options with --end {end} reaches the held-out 2024 "
                         f"block (pre-registered to be touched exactly once, not by this study)")
    if options:
        from ..hpc.build_vrp_dataset import load_option_quotes
    feed, base = ThetaLiveFeed(), Baselines()
    rows, prev, tried, thin, no_book = [], None, 0, 0, 0
    try:
        for day in sessions_between(start, end):
            tried += 1
            if tried % 100 == 0:
                print(f"  {tried} days tried, {len(rows)} trades ...", flush=True)
            try:
                pre, rth = get_ext(feed, symbol, day)
            except Exception as exc:                              # noqa: BLE001
                print(f"  {day}: skipped ({exc!r})", flush=True)
                continue
            if len(rth) < 300:
                thin += 1
                prev = None
                continue
            ten = [b for b in resample_10m(rth) if RTH_FROM <= b["ts"].time() < RTH_TO]
            if prev is not None:
                six = build_six(prev, pre)
                if six:
                    book = None
                    if options:
                        df = load_option_quotes(symbol, day)
                        if df is None:
                            no_book += 1
                        else:
                            book = OptionBook.from_frame(df)
                    for t in session_trades(ten, rth, six, base, book):
                        t["day"] = day.isoformat()
                        rows.append(t)
                else:
                    base.end_session(ten)
            else:
                base.end_session(ten)
            prev = pre + rth
    finally:
        feed.close()
    print(f"\n  {tried} days tried, {thin} thin, {no_book} sessions without an option "
          f"book, {len(rows)} trades")
    return rows


# ------------------------------------------------------------------ report

def _series(rows, key, sub=None, pick=None):
    by = defaultdict(list)
    for r in rows:
        if pick and not pick(r):
            continue
        v = r.get(key)
        if v is None:
            continue
        if sub:
            v = v.get(sub)
        if v is None:
            continue
        by[r["day"]].append(v)
    return by


def _stat(by):
    n = sum(len(v) for v in by.values())
    obs, lo, hi, p = boot_mean(by)
    return {"n": n, "days": len(by), "mean": None if obs is None else round(obs, 3),
            "ci95": None if lo is None else [round(lo, 3), round(hi, 3)],
            "p": None if p is None else round(p, 4)}


def report(rows: list[dict], options: bool) -> dict:
    A = lambda r: r["arm"] == "A"          # noqa: E731
    rep = {"n_trades": len(rows),
           "by_arm_window": dict(sorted(defaultdict(int, {
               f"{r['arm']}/{r['window']}": sum(1 for x in rows if x["arm"] == r["arm"]
                                                and x["window"] == r["window"])
               for r in rows}).items()))}
    rep["underlying_bp"] = {
        "armA_ladder": _stat(_series(rows, "underlying", "weighted_bp", A)),
        "armA_flat50": _stat(_series(rows, "underlying_flat", "weighted_bp", A)),
        "armB_ladder": _stat(_series(rows, "underlying", "weighted_bp", lambda r: r["arm"] == "B")),
        "armA_morning": _stat(_series(rows, "underlying", "weighted_bp",
                                      lambda r: A(r) and r["window"] == "morning")),
        "armA_afternoon": _stat(_series(rows, "underlying", "weighted_bp",
                                        lambda r: A(r) and r["window"] == "afternoon")),
    }
    if options:
        rep["option_dollars_per_trade"] = {
            "PRIMARY_armA_atm_ladder": _stat(_series(rows, "opt_atm", "pnl_net", A)),
            "armA_atm_flat50": _stat(_series(rows, "opt_atm_flat", "pnl_net", A)),
            "armA_otm1_ladder": _stat(_series(rows, "opt_otm1", "pnl_net", A)),
            "armA_itm1_ladder": _stat(_series(rows, "opt_itm1", "pnl_net", A)),
            "armB_atm_ladder": _stat(_series(rows, "opt_atm", "pnl_net", lambda r: r["arm"] == "B")),
        }
        priced = [r for r in rows if A(r) and r.get("opt_atm")]
        rep["unpriced_armA"] = sum(1 for r in rows if A(r) and r.get("opt_atm") is None)
        if priced:
            rep["median_entry_spread_pct"] = round(st.median(r["opt_atm"]["entry_spread_pct"]
                                                             for r in priced), 2)
        prim = rep["option_dollars_per_trade"]["PRIMARY_armA_atm_ladder"]
        if prim["n"] < 150:
            rep["verdict"] = f"UNDERPOWERED: {prim['n']} trades < 150, no verdict"
        elif prim["ci95"] and prim["ci95"][0] > 0 and prim["mean"] > 0:
            rep["verdict"] = "PASS"
        else:
            rep["verdict"] = "FAIL"
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--start", default="2016-01-04")
    ap.add_argument("--end", default="2026-09-19")
    ap.add_argument("--options", action="store_true", help="price exits on the SPY 0DTE archive")
    ap.add_argument("--out", default=None, help="JSON path; default research/his_rules_<symbol>.json")
    a = ap.parse_args(argv)
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    rows = run(a.symbol, start, end, a.options)
    rep = report(rows, a.options)
    out = Path(a.out) if a.out else (Path(__file__).resolve().parents[2] / "research"
                                     / f"his_rules_{a.symbol}{'_options' if a.options else ''}.json")
    out.write_text(json.dumps({"pre_registration": "research/his_rules_preregistration.md",
                               "symbol": a.symbol, "start": a.start, "end": a.end,
                               "options": a.options, "report": rep, "trades": rows},
                              indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
