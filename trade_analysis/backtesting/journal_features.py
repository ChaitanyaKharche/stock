"""Feature table for reverse-engineering the discretionary trade journal.

STEP 0 of the standard trade-journal reverse-engineering workflow. Nothing here fits a
model or looks at an outcome -- it only builds the table every later method needs.

Two label questions are served by one table, and they must not be confused:

  SELECTION  is this minute one he entered on?   his entries  vs  same-session placebos
  OUTCOME    given he entered, what happened?    forward underlying move / his option P&L

The selection question is the actual "reverse-engineer the rule" question and has ~8x the
sample. The outcome question is contaminated by his discretionary EXIT, which is the one
thing in this project that ever cleared a correction -- so the outcome target is carried in
BOTH forms: forward underlying move over fixed horizons (exit-free) and his realised option
return (the money, but confounded).

TIMING RULE, enforced everywhere in this file:
  A 1-minute bar stamped T covers [T, T+60s); its close is not knowable until T+60s.
  The anchor bar for an entry at second E is the last bar with T + 60 <= E.
  5-minute buckets are CLOSE-stamped; the anchor is the last bucket closing at or before E.
  Indicators are seeded from PREFIX_SESSIONS prior sessions -- a real chart does not reset
  an EMA at 09:30.

Isolated experimental code: imports from trade_analysis/live_lab/ but modifies nothing.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

SCRATCH = (r"C:\Users\CHAITA~1\AppData\Local\Temp\claude"
           r"\C--Users-chaitanyakharche-Documents-stock"
           r"\3ad3e562-0324-4d02-bf8b-208ff71f108d\scratchpad")
BARS_LONG = (r"C:\Users\CHAITA~1\AppData\Local\Temp\claude"
             r"\C--Users-chaitanyakharche-Documents-stock"
             r"\8e8c7ab5-fde0-46f3-85ac-5b93b9447703\scratchpad\bars_long.pkl")
NAMECACHE = os.path.join(SCRATCH, "namebars")
TRADES = os.path.join(ROOT, "research", "round_trips.csv")
OUT = os.path.join(SCRATCH, "journal_features.pkl")

PREFIX_SESSIONS = 2
PLACEBOS_PER_TRADE = 8
HORIZONS = (5, 10, 15, 25)
SEED = 20260829

os.makedirs(NAMECACHE, exist_ok=True)


# --------------------------------------------------------------------------- bar access

def _theta_day(sym, day):
    """One RTH session of 1-minute bars from the local ThetaData gateway, cached."""
    p = os.path.join(NAMECACHE, sym + "_" + day + ".pkl")
    if os.path.exists(p):
        try:
            return pickle.load(open(p, "rb"))
        except Exception:
            pass
    import urllib.request
    url = ("http://127.0.0.1:25503/v3/stock/history/ohlc"
           "?symbol=" + sym + "&start_date=" + day + "&end_date=" + day + "&interval=1m")
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            body = r.read().decode()
    except Exception:
        return None
    out = []
    for r in csv.DictReader(body.splitlines()):
        try:
            ts = dt.datetime.fromisoformat(r["timestamp"])
            o, h, lo, c = (float(r[k]) for k in ("open", "high", "low", "close"))
            v = float(r["volume"])
        except (KeyError, ValueError):
            continue
        if c <= 0 or v < 0:
            continue
        out.append({"ts": ts, "open": o, "high": h, "low": lo, "close": c, "volume": v})
    out.sort(key=lambda x: x["ts"])
    pickle.dump(out, open(p, "wb"))
    return out


class BarStore:
    """Per-(symbol, date) 1-minute RTH sessions: pickle for QQQ/SPY, ThetaData otherwise."""

    def __init__(self):
        self.long = pickle.load(open(BARS_LONG, "rb"))
        self._sess = {}
        self._byday = {}
        for sym, df in self.long.items():
            idx = df.index.tz_localize(None) if getattr(df.index, "tz", None) else df.index
            days = np.asarray([d.isoformat() for d in idx.date])
            self._byday[sym] = days

    def session(self, sym, day):
        key = (sym, day)
        if key in self._sess:
            return self._sess[key]
        out = None
        if sym in self.long:
            df = self.long[sym]
            m = self._byday[sym] == day
            if m.sum() >= 30:
                s = df[m]
                si = (s.index.tz_localize(None) if getattr(s.index, "tz", None)
                      else s.index).to_pydatetime()
                out = [{"ts": t, "open": float(o), "high": float(h), "low": float(lo),
                        "close": float(c), "volume": float(v)}
                       for t, o, h, lo, c, v in zip(si, s.Open, s.High, s.Low,
                                                    s.Close, s.Volume)]
        else:
            out = _theta_day(sym, day)
            if out is not None and len(out) < 30:
                out = None
        self._sess[key] = out
        return out

    def prior_sessions(self, sym, day, n):
        """The n most recent sessions strictly before `day` that actually carry data."""
        out = []
        probe = dt.date.fromisoformat(day) - dt.timedelta(days=1)
        tries = 0
        while len(out) < n and tries < 12:
            tries += 1
            if probe.weekday() < 5:
                s = self.session(sym, probe.isoformat())
                if s:
                    out.append(s)
            probe -= dt.timedelta(days=1)
        out.reverse()
        return out


def bars_5m(bars):
    """CLOSE-stamped 5-minute buckets; a bucket is emitted only if all five minutes exist."""
    out, buf = [], []
    for b in bars:
        buf.append(b)
        if len(buf) == 5:
            out.append({"ts": buf[-1]["ts"] + dt.timedelta(minutes=1),
                        "open": buf[0]["open"],
                        "high": max(x["high"] for x in buf),
                        "low": min(x["low"] for x in buf),
                        "close": buf[-1]["close"],
                        "volume": sum(x["volume"] for x in buf)})
            buf = []
    return out


# ------------------------------------------------------------------------ indicator panel

def _ema(v, p):
    out = [None] * len(v)
    if len(v) < p:
        return out
    k = 2.0 / (p + 1)
    e = sum(v[:p]) / p
    out[p - 1] = e
    for i in range(p, len(v)):
        e = v[i] * k + e * (1 - k)
        out[i] = e
    return out


class Panel:
    """Full-length indicator series. Value at index k uses only bars 0..k."""

    def __init__(self, seq):
        self.seq = seq
        self.c = [b["close"] for b in seq]
        self.h = [b["high"] for b in seq]
        self.l = [b["low"] for b in seq]
        self.v = [b["volume"] for b in seq]
        n = len(seq)
        self.ema9 = _ema(self.c, 9)
        self.ema20 = _ema(self.c, 20)
        self.ema50 = _ema(self.c, 50)
        self.vema14 = _ema(self.v, 14)
        self.vema20 = _ema(self.v, 20)
        self.macd = [None] * n
        self.dip = [None] * n
        self.din = [None] * n
        self.adx = [None] * n
        self.atr = [None] * n
        self.rsi = [None] * n
        self._macd()
        self._adx()
        self._rsi()

    def _macd(self):
        c = self.c
        f, s = _ema(c, 9), _ema(c, 17)
        line = [(a - b) if (a is not None and b is not None) else None
                for a, b in zip(f, s)]
        vals = [x for x in line if x is not None]
        if not vals:
            return
        off = len(line) - len(vals)
        sig = _ema(vals, 9)
        for i in range(len(c)):
            j = i - off
            if 0 <= j < len(sig) and line[i] is not None and sig[j] is not None:
                self.macd[i] = (line[i], sig[j], line[i] - sig[j])

    def _adx(self, p=14):
        h, l, c = self.h, self.l, self.c
        n = len(c)
        if n <= p + 2:
            return
        tr = [0.0] * n
        pdm = [0.0] * n
        ndm = [0.0] * n
        for i in range(1, n):
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
            pdm[i] = up if (up > dn and up > 0) else 0.0
            ndm[i] = dn if (dn > up and dn > 0) else 0.0
        a = sum(tr[1:p + 1]) / p
        pp = sum(pdm[1:p + 1]) / p
        nn = sum(ndm[1:p + 1]) / p
        dx = [None] * n
        for i in range(p, n):
            if i > p:
                a = (a * (p - 1) + tr[i]) / p
                pp = (pp * (p - 1) + pdm[i]) / p
                nn = (nn * (p - 1) + ndm[i]) / p
            self.atr[i] = a
            if a > 0:
                self.dip[i] = 100 * pp / a
                self.din[i] = 100 * nn / a
                tot = self.dip[i] + self.din[i]
                dx[i] = (100 * abs(self.dip[i] - self.din[i]) / tot) if tot > 0 else 0.0
        first = next((i for i in range(n) if dx[i] is not None), None)
        if first is None or first + p >= n:
            return
        run = sum(dx[first:first + p]) / p
        self.adx[first + p - 1] = run
        for i in range(first + p, n):
            run = (run * (p - 1) + dx[i]) / p
            self.adx[i] = run

    def _rsi(self, p=14):
        c = self.c
        n = len(c)
        if n <= p + 1:
            return
        g = [max(c[i] - c[i - 1], 0.0) for i in range(1, n)]
        ls = [max(c[i - 1] - c[i], 0.0) for i in range(1, n)]
        ag = sum(g[:p]) / p
        al = sum(ls[:p]) / p
        self.rsi[p] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
        for i in range(p + 1, n):
            ag = (ag * (p - 1) + g[i - 1]) / p
            al = (al * (p - 1) + ls[i - 1]) / p
            self.rsi[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


# ---------------------------------------------------------------------------- features

def _anchor_1m(sess, when):
    """Last 1-minute bar whose CLOSE is knowable at `when`: ts + 60s <= when."""
    lim = when - dt.timedelta(seconds=60)
    k = -1
    for i, b in enumerate(sess):
        if b["ts"] <= lim:
            k = i
        else:
            break
    return k


def _anchor_5m(b5, when):
    k = -1
    for i, b in enumerate(b5):
        if b["ts"] <= when:
            k = i
        else:
            break
    return k


def features_at(sym, day, when, direction, ctx):
    """Feature dict at `when`, using only information available at `when`.

    `direction` is +1 for a call, -1 for a put; features named *_al are multiplied by it,
    so a positive value always means "aligned with the position he took".
    """
    sess, p1, b5, p5, pref_n, prior = ctx
    k = _anchor_1m(sess, when)
    if k < 5:
        return None
    K = pref_n + k                      # index into the prefix+session panel
    j5 = _anchor_5m(b5, when)
    if j5 < 2:
        return None
    J5 = p5.pref + j5

    px = sess[k]["close"]
    if px <= 0:
        return None
    d = float(direction)
    f = {}

    a1 = p1.atr[K]
    a5 = p5.atr[J5]
    if not a1 or not a5 or a5 <= 0:
        return None

    # ---- his declared stack, on the 5-minute chart he actually watches
    m5 = p5.macd[J5]
    if m5 is None:
        return None
    f["macd_line_al"] = d * m5[0] / px * 1e4
    f["macd_sig_al"] = d * m5[1] / px * 1e4
    f["macd_hist_al"] = d * m5[2] / px * 1e4
    prev = p5.macd[J5 - 1]
    f["macd_hist_slope_al"] = (d * (m5[2] - prev[2]) / px * 1e4) if prev else 0.0
    if p5.adx[J5] is None or p5.dip[J5] is None:
        return None
    f["adx5"] = p5.adx[J5]
    f["di_diff_al"] = d * (p5.dip[J5] - p5.din[J5])
    f["di_plus"] = p5.dip[J5]
    f["di_minus"] = p5.din[J5]
    if p1.adx[K] is None:
        return None
    f["adx1"] = p1.adx[K]
    f["di_diff_1m_al"] = d * (p1.dip[K] - p1.din[K])

    # ---- volume
    f["vol_rel_ema14"] = (sess[k]["volume"] / p1.vema14[K]) if p1.vema14[K] else 1.0
    f["vol_rel_ema20"] = (sess[k]["volume"] / p1.vema20[K]) if p1.vema20[K] else 1.0
    f["vol5_rel_ema14"] = (b5[j5]["volume"] / p5.vema14[J5]) if p5.vema14[J5] else 1.0

    # ---- price vs moving averages, in ATR units so symbols are comparable
    for nm, s in (("ema9", p5.ema9), ("ema20", p5.ema20), ("ema50", p5.ema50)):
        f["px_vs_" + nm + "_al"] = (d * (px - s[J5]) / a5) if s[J5] else 0.0
    f["ema9_vs_ema20_al"] = (d * (p5.ema9[J5] - p5.ema20[J5]) / a5
                             if p5.ema9[J5] and p5.ema20[J5] else 0.0)
    f["ema20_vs_ema50_al"] = (d * (p5.ema20[J5] - p5.ema50[J5]) / a5
                              if p5.ema20[J5] and p5.ema50[J5] else 0.0)
    stack = 0.0
    if p5.ema9[J5] and p5.ema20[J5] and p5.ema50[J5]:
        stack = (1.0 if p5.ema9[J5] > p5.ema20[J5] > p5.ema50[J5] else
                 -1.0 if p5.ema9[J5] < p5.ema20[J5] < p5.ema50[J5] else 0.0)
    f["ema_stack_al"] = d * stack
    f["rsi5_al"] = d * (p5.rsi[J5] - 50.0) if p5.rsi[J5] is not None else 0.0

    # ---- session structure
    op = sess[0]["open"]
    hi = max(b["high"] for b in sess[:k + 1])
    lo = min(b["low"] for b in sess[:k + 1])
    rng = hi - lo
    f["day_range_pos_al"] = d * ((px - lo) / rng - 0.5) * 2 if rng > 0 else 0.0
    f["day_range_atr"] = rng / a5
    f["ret_from_open_al"] = d * (px / op - 1.0) * 1e4
    or_end = min(15, k + 1)
    orh = max(b["high"] for b in sess[:or_end])
    orl = min(b["low"] for b in sess[:or_end])
    orr = orh - orl
    f["or15_pos_al"] = d * ((px - orl) / orr - 0.5) * 2 if orr > 0 else 0.0
    f["or15_cleared_al"] = d * (1.0 if px > orh else -1.0 if px < orl else 0.0)

    # ---- VWAP
    tv = sum(b["volume"] for b in sess[:k + 1])
    vw = (sum((b["high"] + b["low"] + b["close"]) / 3 * b["volume"]
              for b in sess[:k + 1]) / tv) if tv > 0 else px
    f["vwap_dist_al"] = d * (px - vw) / a5

    # ---- momentum / chase, 5-minute sigma convention
    r5 = [b5[i]["close"] / b5[i - 1]["close"] - 1.0 for i in range(1, j5 + 1)]
    sd5 = float(np.std(r5, ddof=1)) if len(r5) > 2 else 0.0
    f["trail15_sigma_al"] = (d * (px / b5[j5 - 2]["close"] - 1.0) / (sd5 * math.sqrt(3))
                             if sd5 > 0 and j5 >= 2 else 0.0)
    for w in (1, 5, 15, 30):
        if k - w >= 0:
            f["ret_%dm_al" % w] = d * (px / sess[k - w]["close"] - 1.0) * 1e4
        else:
            f["ret_%dm_al" % w] = 0.0
    up = sum(1 for b in sess[max(0, k - 14):k + 1] if b["close"] > b["open"])
    f["upbar_frac_al"] = d * (up / min(15, k + 1) - 0.5) * 2
    f["realised_vol_bp"] = sd5 * 1e4

    # ---- levels he says he uses
    f["dist_round25_atr"] = abs(px - round(px / 25.0) * 25.0) / a5
    f["dist_round5_atr"] = abs(px - round(px / 5.0) * 5.0) / a5
    if prior:
        ph = max(b["high"] for b in prior[-1])
        pl = min(b["low"] for b in prior[-1])
        pc = prior[-1][-1]["close"]
        f["dist_pdh_atr_al"] = d * (px - ph) / a5
        f["dist_pdl_atr_al"] = d * (px - pl) / a5
        f["gap_al"] = d * (op / pc - 1.0) * 1e4
        f["prior_day_ret_al"] = d * (pc / prior[-1][0]["open"] - 1.0) * 1e4
        f["prior_range_atr"] = (ph - pl) / a5
    else:
        for kk in ("dist_pdh_atr_al", "dist_pdl_atr_al", "gap_al",
                   "prior_day_ret_al", "prior_range_atr"):
            f[kk] = 0.0

    # ---- clock
    mins = (when - sess[0]["ts"]).total_seconds() / 60.0
    f["tod_min"] = mins
    f["tod_first30"] = 1.0 if mins < 30 else 0.0
    f["tod_last60"] = 1.0 if mins > 330 else 0.0
    f["weekday"] = float(when.weekday())
    f["atr_pct"] = a5 / px * 1e4
    return f


def forward(sess, when, k, direction, hz):
    """Forward underlying move from the anchor close, aligned. Uses the future by design --
    this is a TARGET, not a feature."""
    out = {}
    base = sess[k]["close"]
    for h in hz:
        i = k + h
        out["fwd_%dm_bp" % h] = (float(direction) * (sess[i]["close"] / base - 1.0) * 1e4
                                 if i < len(sess) else np.nan)
    return out


# --------------------------------------------------------------------------------- build

def main():
    rng = np.random.default_rng(SEED)
    store = BarStore()
    trades = list(csv.DictReader(open(TRADES, encoding="utf-8")))
    rows, funnel = [], {}

    def drop(why):
        funnel[why] = funnel.get(why, 0) + 1

    ctx_cache = {}

    def context(sym, day):
        if (sym, day) in ctx_cache:
            return ctx_cache[(sym, day)]
        sess = store.session(sym, day)
        got = None
        if sess:
            prior = store.prior_sessions(sym, day, PREFIX_SESSIONS)
            pre1 = [b for s in prior for b in s]
            p1 = Panel(pre1 + sess)
            b5 = bars_5m(sess)
            pre5 = [b for s in prior for b in bars_5m(s)]
            p5 = Panel(pre5 + b5)
            p1.pref, p5.pref = len(pre1), len(pre5)
            got = (sess, p1, b5, p5, len(pre1), prior)
        ctx_cache[(sym, day)] = got
        return got

    # per-day trade ordinal, and the previous trade's realised outcome (known at entry)
    trades.sort(key=lambda r: r["entry_ts"])
    seen, prev_pnl = {}, {}
    for r in trades:
        key = r["date"]
        seen[key] = seen.get(key, 0) + 1
        r["_ord"] = seen[key]
        r["_prev_net"] = prev_pnl.get(key, 0.0)
        prev_pnl[key] = float(r["net"])

    for i, r in enumerate(trades, 1):
        sym, day = r["symbol"], r["date"]
        ctx = context(sym, day)
        if ctx is None:
            drop("no_session_bars")
            continue
        sess = ctx[0]
        try:
            ent = dt.datetime.fromisoformat(r["entry_ts"]).replace(tzinfo=None)
        except ValueError:
            drop("bad_ts")
            continue
        direction = 1 if r["right"] == "call" else -1
        f = features_at(sym, day, ent, direction, ctx)
        if f is None:
            drop("too_early_or_no_indicator")
            continue
        k = _anchor_1m(sess, ent)
        row = dict(f)
        row.update(forward(sess, ent, k, direction, HORIZONS))
        # underlying move over HIS actual holding period
        try:
            ex = dt.datetime.fromisoformat(r["exit_ts"]).replace(tzinfo=None)
            hold = int(round((ex - ent).total_seconds() / 60.0))
        except ValueError:
            hold = 0
        row["hold_min"] = float(r["hold_min"])
        row["fwd_hishold_bp"] = (direction * (sess[k + hold]["close"] / sess[k]["close"] - 1)
                                 * 1e4) if 0 < hold < len(sess) - k else np.nan
        row["opt_ret"] = float(r["pct"])
        row["opt_net"] = float(r["net"])
        row["win"] = 1 if float(r["net"]) > 0 else 0
        row["qty"] = float(r["qty"])
        row["dte"] = float(r["dte"])
        try:
            row["opt_spread_pct"] = ((float(r["entry_ask"]) - float(r["entry_bid"]))
                                     / float(r["entry_ask"]))
        except (ValueError, ZeroDivisionError):
            row["opt_spread_pct"] = np.nan
        row["trade_ordinal"] = float(r["_ord"])
        row["prev_trade_net"] = r["_prev_net"]
        row["prev_trade_loss"] = 1.0 if r["_prev_net"] < 0 else 0.0
        row.update({"symbol": sym, "date": day, "is_entry": 1, "direction": direction,
                    "ts": ent, "is_etf": 1 if sym in ("QQQ", "SPY") else 0})
        rows.append(row)

        # ---- same-session placebos: minutes he could have entered and did not
        lo_i, hi_i = 5, len(sess) - max(HORIZONS) - 1
        if hi_i - lo_i > PLACEBOS_PER_TRADE * 2:
            for kk in rng.choice(range(lo_i, hi_i), size=PLACEBOS_PER_TRADE,
                                 replace=False):
                w = sess[int(kk)]["ts"] + dt.timedelta(seconds=90)
                pf = features_at(sym, day, w, direction, ctx)
                if pf is None:
                    continue
                pk = _anchor_1m(sess, w)
                prow = dict(pf)
                prow.update(forward(sess, w, pk, direction, HORIZONS))
                prow.update({"symbol": sym, "date": day, "is_entry": 0,
                             "direction": direction, "ts": w,
                             "is_etf": 1 if sym in ("QQQ", "SPY") else 0})
                rows.append(prow)
        if i % 100 == 0:
            print("  %d/%d  rows=%d" % (i, len(trades), len(rows)), flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_pickle(OUT)
    ent = df[df.is_entry == 1]
    print("\nFUNNEL")
    print("  trades in journal          %d" % len(trades))
    for kk, v in sorted(funnel.items()):
        print("  dropped %-24s %d" % (kk, v))
    print("  entries with features      %d" % len(ent))
    print("  placebo rows               %d" % int((df.is_entry == 0).sum()))
    print("  activity dates             %d" % ent.date.nunique())
    print("  symbols                    %d" % ent.symbol.nunique())
    print("  ETF entries                %d" % int(ent.is_etf.sum()))
    feats = [c for c in df.columns if c not in
             ("symbol", "date", "ts", "is_entry", "win", "opt_ret", "opt_net",
              "direction", "hold_min", "fwd_hishold_bp", "is_etf")
             and not c.startswith("fwd_")]
    print("  features                   %d" % len(feats))
    print("  written -> %s" % OUT)


if __name__ == "__main__":
    main()
