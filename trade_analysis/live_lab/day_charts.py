"""DAY CHARTS -- one picture per lab session per timeframe, with the six lines.

    venv_charts\\Scripts\\python -m trade_analysis.live_lab.day_charts --offline          # every lab day
    venv_charts\\Scripts\\python -m trade_analysis.live_lab.day_charts --offline --day 2026-10-08 --symbols QQQ
    ... --fetch-only     cache the bars and draw nothing (needs Theta Terminal)

Needs matplotlib, which the lab's interpreter deliberately does not have. Run it from
venv_charts (pyenv 3.12.10 with system site-packages, plus matplotlib); the lab never
imports this module. Without --offline it talks to the same Theta Terminal the runners use,
so it refuses to during the session (09:00-16:05 ET on weekdays) unless --force.

WHAT IS DRAWN
-------------
  * 07:00-16:00 ET candles at 2, 5, 10 and 15 minutes, built from the vendor's 1-minute
    bars (feed.extended_bars, the endpoint the runner's six lines come from). 07:00 ET is
    04:00 MST in summer and 05:00 MST once ET leaves daylight time; ticks show both. The
    premarket is shaded.
  * The six lines, from 09:30: exactly as the runner logged them at 09:31 (events.jsonl
    `levels`, from 2026-09-25) or, before the runner drew them, rebuilt with
    six_lines.build_six from the same bars. Where both exist they are compared and any
    disagreement is printed on the chart -- a number that changes when nothing changed is
    a bug until shown otherwise. Each label carries the minute that set the level, so a
    line sitting on one stray premarket print is visible as such.
  * EMA 9 / EMA 20, session VWAP, volume with a 20-bar average, MACD 12/26/9 on close and
    RSI 14 (Wilder). MACD 12/26/9 is what the trader's own Robinhood chart shows: on
    2026-10-08 its 12:10 ET 10-minute bar read signal +0.02, which 12/26/9 reproduces
    (+0.01) and the 9/17/9 of indicators.py does not (-0.02 to -0.04). EMA, MACD and RSI
    are warmed on the previous session so the first visible bar is not cold; VWAP is built
    from 1-minute bars so it is the same number on every timeframe.

Charts go to live_lab_charts/ (gitignored): <day>/<SYM>_<NN>m.png and index.html.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import os
import pickle
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "live_lab_data"
OUT = ROOT / "live_lab_charts"
TIMEFRAMES = (2, 5, 10, 15)
ALLOWED_TF = (1, 2, 3, 5, 6, 10, 15, 30)   # divide 30, so 09:30 and every hour are boundaries
SYMBOLS = ("QQQ", "SPY")
SHOW_FROM, RTH_OPEN, RTH_CLOSE = dt.time(7, 0), dt.time(9, 30), dt.time(16, 0)
FETCH_FROM = dt.time(4, 0)
SETTLED_AT = dt.time(16, 5)          # today's bars are not cached before the session is over
BUSY_FROM = dt.time(9, 0)            # network runs refused 09:00-16:05 ET: the runners own the terminal
MIN_RTH_BARS = 300                   # levels_live.MIN_RTH_BARS: yesterday must be a real day
RAN = {"OPENED", "COLLECTED", "PARTIAL", "INTERRUPTED"}
ET, MST = ZoneInfo("America/New_York"), ZoneInfo("America/Phoenix")   # Phoenix: MST, no DST
SHORT_SOURCE = {"today premarket high": "today PM high", "today premarket low": "today PM low",
                "yday premarket high": "yday PM high", "yday premarket low": "yday PM low",
                "yday market high": "yday high", "yday market low": "yday low"}

STYLE = {"bg": "#141414", "pre": "#232323", "grid": "#262626", "text": "#cfcfcf",
         "dim": "#8a8a8a", "up": "#00c805", "down": "#ff5000", "ema9": "#ff9f1a",
         "ema20": "#e8e8e8", "vwap": "#ff6a3d", "volma": "#3d7be0", "R": "#ff4fa3",
         "S": "#4fc3ff", "rsi": "#9fd3ff", "band": "#173a5e", "warn": "#ff5c5c"}


# --------------------------------------------------------------------------- data

def lab_days(lab_dir: Path = LAB) -> list[dt.date]:
    """Sessions the live runner actually ran (session_ledger.jsonl)."""
    days = set()
    p = Path(lab_dir) / "session_ledger.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines() if p.exists() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("is_session") and r.get("outcome") in RAN:
            days.add(dt.date.fromisoformat(r["date"]))
    return sorted(days)


def _now() -> dt.datetime:
    from .clock import now_et
    return now_et()


def _settled(day: dt.date, now: dt.datetime | None = None) -> bool:
    now = now or _now()
    return day < now.date() or (day == now.date() and now.time() >= SETTLED_AT)


def load_bars(sym: str, day: dt.date, cache: Path, feed=None, now=None) -> list[dict] | None:
    """1-minute bars 04:00-16:00 ET. None = not cached and no feed to fetch with.

    Never cached: today's bars before SETTLED_AT (a mid-session copy would be served
    forever), and an EMPTY answer (a holiday is cheap to ask again; a vendor hiccup cached
    as a holiday would silently move "yesterday" a day back). The write is atomic."""
    path = cache / f"{sym}_{day.isoformat()}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    if feed is None:
        return None
    bars = feed.extended_bars(sym, day, FETCH_FROM, RTH_CLOSE)
    if bars and _settled(day, now):
        cache.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(bars))
        os.replace(tmp, path)
    return bars


def _rth(bars):
    return [b for b in bars if RTH_OPEN <= b["ts"].time() < RTH_CLOSE]


def prior_session(sym: str, day: dt.date, cache: Path, feed=None, now=None) -> tuple[dt.date, list] | None:
    """The most recent earlier weekday with a full market-hours session -- the rule of
    levels_live.LevelsLoader.at_warmup (>= MIN_RTH_BARS). None if a day it needs is
    neither cached nor fetchable."""
    d = day
    for _ in range(10):
        d -= dt.timedelta(days=1)
        if d.weekday() >= 5:
            continue
        bars = load_bars(sym, d, cache, feed, now)
        if bars is None:
            return None
        if len(_rth(bars)) >= MIN_RTH_BARS:
            return d, bars
    return None


@functools.lru_cache(maxsize=4)
def _logged_levels(lab_dir: str) -> dict:
    """(symbol, day) -> the FIRST `levels` event's lines, options arm before shares. A
    restart can log the same day again; the first is what the setups traded on."""
    out: dict = {}
    for p in (Path(lab_dir) / "events.jsonl", Path(lab_dir) / "shares" / "events.jsonl"):
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if '"levels"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                k = (e.get("symbol"), str(e.get("ts", ""))[:10])
                if e.get("kind") == "levels" and e.get("lines") and k not in out:
                    out[k] = e["lines"]
    return out


def recorded_lines(lab_dir: Path, sym: str, day: dt.date) -> list[dict] | None:
    """The six lines the runner itself logged at 09:31 that day (either arm), if any."""
    return _logged_levels(str(lab_dir)).get((sym, day.isoformat()))


def rebuilt_lines(yday_bars, bars) -> list[dict]:
    from .levels_live import lines_as_dicts
    from .six_lines import build_six
    pre = [b for b in bars if FETCH_FROM <= b["ts"].time() < RTH_OPEN]
    return list(lines_as_dicts(build_six(yday_bars, pre)))


def compare_lines(a: list[dict], b: list[dict], tol: float = 0.005) -> list[str]:
    """Human-readable differences between two six-line sets ([] when they agree)."""
    da = {l["name"]: l for l in a}
    db = {l["name"]: l for l in b}
    out = []
    for name in sorted(set(da) | set(db)):
        x, y = da.get(name), db.get(name)
        if x is None or y is None:
            out.append(f"{name} missing on one side")
        elif abs(float(x["price"]) - float(y["price"])) > tol or x["source"] != y["source"]:
            out.append(f"{name} {float(x['price']):.2f} ({x['source']}) vs "
                       f"{float(y['price']):.2f} ({y['source']})")
    return out


def set_at(line: dict, yday_bars, bars, tol: float = 0.005) -> str:
    """HH:MM of the first bar that made this level, or "" if no bar matches -- which is
    itself informative: the vendor revised that bar after the runner read it."""
    src = line["source"]
    pool = bars if src.startswith("today") else yday_bars
    lo_t, hi_t = (FETCH_FROM, RTH_OPEN) if "premarket" in src else (RTH_OPEN, RTH_CLOSE)
    key = "high" if src.endswith("high") else "low"
    p = float(line["price"])
    for b in pool:
        if lo_t <= b["ts"].time() < hi_t and abs(b[key] - p) <= tol:
            return f"{b['ts']:%H:%M}"
    return ""


# --------------------------------------------------------------------------- bars and indicators

def resample(bars: list[dict], minutes: int) -> list[dict]:
    """Clock-aligned N-minute OHLCV. `last_ts` keeps each bucket's final 1-minute stamp."""
    if minutes not in ALLOWED_TF:
        raise ValueError(f"timeframe must be one of {ALLOWED_TF}, got {minutes}")
    out: list[dict] = []
    for b in sorted(bars, key=lambda x: x["ts"]):
        t = b["ts"]
        m = (t.hour * 60 + t.minute) // minutes * minutes
        start = t.replace(hour=m // 60, minute=m % 60, second=0, microsecond=0)
        if out and out[-1]["ts"] == start:
            o = out[-1]
            o["high"], o["low"] = max(o["high"], b["high"]), min(o["low"], b["low"])
            o["close"], o["last_ts"] = b["close"], t
            o["volume"] += b["volume"]
        else:
            out.append({"ts": start, "last_ts": t, "open": b["open"], "high": b["high"],
                        "low": b["low"], "close": b["close"], "volume": b["volume"]})
    return out


def ema(xs, n):
    """EMA seeded with the first value. Warmed on the previous session, so the seed is
    long gone by 07:00."""
    out, k, e = [], 2.0 / (n + 1), None
    for x in xs:
        e = x if e is None else e + k * (x - e)
        out.append(e)
    return out


def rsi(xs, n=14):
    """Wilder's RSI: first value at index n from the simple mean of the first n changes,
    then Wilder smoothing. A series that never moves reads 50, not 100."""
    out, gain, loss = [None] * len(xs), 0.0, 0.0
    for i in range(1, len(xs)):
        d = xs[i] - xs[i - 1]
        g, l = max(d, 0.0), max(-d, 0.0)
        if i <= n:
            gain += g / n
            loss += l / n
            if i < n:
                continue
        else:
            gain = (gain * (n - 1) + g) / n
            loss = (loss * (n - 1) + l) / n
        out[i] = (50.0 if gain == loss == 0 else 100.0 if loss == 0
                  else 100.0 - 100.0 / (1.0 + gain / loss))
    return out


def vwap_1m(bars: list[dict]) -> dict:
    """{1-minute ts: session VWAP through that minute}, from the 09:30 bar on."""
    out, pv, vol = {}, 0.0, 0.0
    for b in sorted(bars, key=lambda x: x["ts"]):
        if not (RTH_OPEN <= b["ts"].time() < RTH_CLOSE):
            continue
        pv += (b["high"] + b["low"] + b["close"]) / 3.0 * b["volume"]
        vol += b["volume"]
        if vol:
            out[b["ts"]] = pv / vol
    return out


def vwap_series(bars_1m: list[dict], buckets: list[dict]) -> list[float | None]:
    """VWAP at each bucket's last minute, from the 1-minute bars: the same session number
    on every timeframe (built from N-minute bars it was up to 84 cents different)."""
    m = vwap_1m(bars_1m)
    return [m.get(b["last_ts"]) for b in buckets]


# --------------------------------------------------------------------------- drawing

def _fmt_vol(v):
    return f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.2f}K"


def _mst(t: dt.datetime) -> dt.datetime:
    return t.replace(tzinfo=ET).astimezone(MST)


def _spread(prices: list[float], gap: float) -> list[float]:
    """Label heights: each at its line, pushed apart to at least `gap` so none overlap."""
    order = sorted(range(len(prices)), key=lambda i: -prices[i])
    pos = [0.0] * len(prices)
    last = None
    for i in order:
        y = prices[i] if last is None else min(prices[i], last - gap)
        pos[i], last = y, y
    return pos


def build_figure(sym: str, day: dt.date, minutes: int, bars: list[dict], lines: list[dict] | None,
                 lines_note: str, prev_close: float | None, warn: str = "", warm=(),
                 line_times: dict | None = None):
    """The figure, not yet saved -> (fig, {"price", "volume", "macd", "rsi": axes, "view": bars})."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["text.parse_math"] = False      # "$747.57 ... $10.14" is not maths
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle

    s = STYLE
    today = resample(bars, minutes)
    allb = resample(list(warm), minutes) + today       # indicators warm on yesterday
    closes = [b["close"] for b in allb]
    e9, e20 = ema(closes, 9), ema(closes, 20)
    macd = [a - b for a, b in zip(ema(closes, 12), ema(closes, 26))]
    sig = ema(macd, 9)
    hist = [a - b for a, b in zip(macd, sig)]
    r14 = rsi(closes, 14)
    vols_all = [b["volume"] for b in allb]
    vma = [None if i < 19 else sum(vols_all[i - 19:i + 1]) / 20 for i in range(len(vols_all))]

    off = len(allb) - len(today)
    keep = [off + i for i, b in enumerate(today) if SHOW_FROM <= b["ts"].time() < RTH_CLOSE]
    if not keep:
        raise ValueError(f"{sym} {day}: no bars between 07:00 and 16:00")
    i0, i1 = keep[0], keep[-1] + 1
    view = allb[i0:i1]
    e9, e20, vma, macd, sig, hist, r14 = (xs[i0:i1] for xs in (e9, e20, vma, macd, sig, hist, r14))
    vw = vwap_series(bars, view)
    vols = [b["volume"] for b in view]
    x = list(range(len(view)))
    rth = [b for b in view if b["ts"].time() >= RTH_OPEN]
    day_close = rth[-1]["close"] if rth else view[-1]["close"]
    up_day = prev_close is None or day_close >= prev_close
    nan = float("nan")

    fig = plt.figure(figsize=(16, 9.6), dpi=110, facecolor=s["bg"])
    gs = fig.add_gridspec(3, 1, height_ratios=[5.6, 1.35, 1.35], hspace=0.04,
                          left=0.055, right=0.855, top=0.89, bottom=0.075)
    ax, axm, axr = (fig.add_subplot(gs[i]) for i in range(3))
    axv = ax.twinx()                       # volume: drawn UNDER the price panel
    # Twinned axes: matplotlib shows the BOTTOM twin's background whatever we set, so the
    # volume axes must carry the dark background (it came out white otherwise).
    axv.set_facecolor(s["bg"])
    for a in (ax, axm, axr):
        a.set_facecolor(s["bg"])
        a.tick_params(colors=s["dim"], labelsize=8.5, length=0)
        for sp in a.spines.values():
            sp.set_visible(False)
        a.grid(color=s["grid"], linewidth=0.6)
        a.set_xlim(-1, len(view))
    axv.set_xlim(-1, len(view))

    # Premarket shade and a faint day tint. On the price panel they go on the VOLUME axes,
    # which sit beneath it: on the price axes the opaque shade hid every premarket volume bar.
    first_rth = next((i for i, b in enumerate(view) if b["ts"].time() >= RTH_OPEN), len(view))
    for a in (axv, axm, axr):
        a.axvspan(-1, first_rth - 0.5, color=s["pre"], zorder=0)
        a.axvspan(first_rth - 0.5, len(view), color=s["up"] if up_day else s["down"],
                  alpha=0.035, zorder=0)

    # candles
    w = 0.66
    for i, b in enumerate(view):
        c = s["up"] if b["close"] >= b["open"] else s["down"]
        ax.vlines(i, b["low"], b["high"], color=c, linewidth=0.9, zorder=3)
        lo, hi = sorted((b["open"], b["close"]))
        ax.add_patch(Rectangle((i - w / 2, lo), w, max(hi - lo, 1e-6), facecolor=c,
                               edgecolor=c, linewidth=0.5, zorder=4))
    ax.plot(x, e9, color=s["ema9"], linewidth=1.15, zorder=5, label="EMA 9")
    ax.plot(x, e20, color=s["ema20"], linewidth=1.0, alpha=0.85, zorder=5, label="EMA 20")
    ax.plot(x, [v if v is not None else nan for v in vw], color=s["vwap"],
            linewidth=1.2, linestyle=(0, (1, 2.2)), zorder=5, label="VWAP")

    # Price range: the candles, plus any line within 60% of their range. A line further out
    # (a stray print) is pinned to the edge rather than squashing the day into a sliver.
    lo_p = min(b["low"] for b in view)
    hi_p = max(b["high"] for b in view)
    span = hi_p - lo_p or 1.0
    near = [float(l["price"]) for l in (lines or [])
            if lo_p - 0.6 * span <= float(l["price"]) <= hi_p + 0.6 * span]
    lo_p, hi_p = min([lo_p] + near), max([hi_p] + near)
    pad = (hi_p - lo_p) * 0.05 or 0.5
    y0, y1 = lo_p - pad * 5.0, hi_p + pad       # room underneath for the volume bars
    ax.set_ylim(y0, y1)
    if lines:
        prices = [float(l["price"]) for l in lines]
        shown = [min(max(p, y0 + pad), y1 - pad * 0.3) for p in prices]
        for l, p, sp, ly in zip(lines, prices, shown, _spread(shown, (y1 - y0) * 0.032)):
            col = s[l["side"]]
            off_chart = "" if p == sp else (" ▲" if p > sp else " ▼")      # beyond the axis
            if not off_chart:
                ax.hlines(p, first_rth - 0.5, len(view) - 0.5, color=col, linewidth=1.05,
                          linestyle=(0, (6, 3)), zorder=6)
            when = (line_times or {}).get(l["name"], "")
            ax.annotate(f"{l['name']} {p:.2f} · {SHORT_SOURCE.get(l['source'], l['source'])}"
                        + (f" {when}" if when else "") + off_chart,
                        xy=(len(view) - 0.5, sp), xytext=(len(view) * 1.012, ly),
                        textcoords="data", va="center", ha="left", fontsize=8,
                        color="#101010", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=col, linewidth=0.8,
                                        shrinkA=0, shrinkB=0),
                        bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none"))

    # Volume, capped at the 98th percentile so one closing-auction bar does not flatten the
    # rest of the day into single pixels.
    cap = sorted(vols)[int(0.98 * (len(vols) - 1))] if vols else 0
    cap = cap or max(vols, default=0) or 1
    axv.set_ylim(0, cap * 4.6)
    axv.bar(x, [min(v, cap * 1.15) for v in vols], width=w, zorder=1,
            color=[s["up"] if b["close"] >= b["open"] else s["down"] for b in view], alpha=0.45)
    axv.plot(x, [v if v is not None else nan for v in vma], color=s["volma"], linewidth=1.1, zorder=2)
    axv.set_yticks([])
    for sp in axv.spines.values():
        sp.set_visible(False)
    ax.set_zorder(axv.get_zorder() + 1)
    ax.patch.set_visible(False)

    caption = dict(facecolor=s["bg"], edgecolor="none", alpha=0.85, pad=2)
    # MACD
    hcol = []
    for i, h in enumerate(hist):
        rising = i > 0 and h >= hist[i - 1]
        hcol.append(to_rgba(s["up"] if h >= 0 else s["down"], 0.9 if (h >= 0) == rising else 0.45))
    axm.bar(x, hist, width=w, color=hcol, zorder=2)
    axm.plot(x, macd, color=s["ema20"], linewidth=1.0, zorder=3)
    axm.plot(x, sig, color=s["ema9"], linewidth=1.0, zorder=3)
    axm.axhline(0, color=s["grid"], linewidth=0.8)
    axm.text(0.004, 0.86, f"MACD 12 26 9   {macd[-1]:+.2f}   {sig[-1]:+.2f}   {hist[-1]:+.2f}",
             transform=axm.transAxes, color=s["text"], fontsize=8.5, zorder=10, bbox=caption)

    # RSI
    rr = [v if v is not None else nan for v in r14]
    axr.axhspan(30, 70, color=s["band"], alpha=0.55, zorder=1)
    for lvl in (30, 70):
        axr.axhline(lvl, color="#3f6f9e", linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
    axr.plot(x, rr, color=s["rsi"], linewidth=1.0, zorder=3)
    axr.fill_between(x, rr, 30, where=[v < 30 for v in rr], color=s["down"], alpha=0.45,
                     interpolate=True, zorder=2)
    axr.fill_between(x, rr, 70, where=[v > 70 for v in rr], color=s["up"], alpha=0.45,
                     interpolate=True, zorder=2)
    axr.set_ylim(5, 95)
    last_rsi = next((v for v in reversed(r14) if v is not None), None)
    axr.text(0.004, 0.84, f"RSI 14   {last_rsi:.1f}" if last_rsi is not None else "RSI 14",
             transform=axr.transAxes, color=s["text"], fontsize=8.5, zorder=10, bbox=caption)

    # time axis: every hour and 09:30, ET over MST
    ticks = [i for i, b in enumerate(view)
             if (b["ts"].minute == 0 and b["ts"].hour >= 7) or b["ts"].time() == RTH_OPEN]
    for a in (ax, axm, axv):
        a.set_xticks(ticks)
        a.set_xticklabels([])
    axr.set_xticks(ticks)
    axr.set_xticklabels([f"{view[i]['ts']:%H:%M}\n{_mst(view[i]['ts']):%H:%M}" for i in ticks],
                        color=s["dim"])
    fig.text(0.004, 0.034, "ET\nMST", color=s["dim"], fontsize=8.5, va="center")

    # header
    rth_v = sum(b["volume"] for b in rth)
    o, h, l_, c = (rth[0]["open"], max(b["high"] for b in rth), min(b["low"] for b in rth),
                   rth[-1]["close"]) if rth else (nan,) * 4
    fig.text(0.055, 0.952, f"{sym}", color="white", fontsize=17, weight="bold")
    fig.text(0.102, 0.955, f"{minutes} min   ·   {day:%a %d %b %Y}", color=s["text"], fontsize=12)
    if prev_close:
        ch = c - prev_close
        arrow, col = ("▲", s["up"]) if ch >= 0 else ("▼", s["down"])
        fig.text(0.32, 0.955, f"${c:,.2f}  {arrow} ${abs(ch):.2f} ({abs(ch) / prev_close:.2%})",
                 color=col, fontsize=12, weight="bold")
    fig.text(0.055, 0.915, f"O {o:.2f}   H {h:.2f}   L {l_:.2f}   C {c:.2f}   V {_fmt_vol(rth_v)}"
             f"      market hours", color=s["text"], fontsize=10)
    fig.text(0.855, 0.955, lines_note, color=s["dim"], fontsize=9, ha="right")
    if warn:
        fig.text(0.855, 0.917, warn, color=s["warn"], fontsize=8.5, ha="right")
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=s["text"], ncol=3)
    return fig, {"price": ax, "volume": axv, "macd": axm, "rsi": axr, "view": view}


def render(sym, day, minutes, bars, lines, lines_note, prev_close, path: Path, warn: str = "",
           warm=(), line_times=None) -> Path:
    import matplotlib.pyplot as plt
    fig, _ = build_figure(sym, day, minutes, bars, lines, lines_note, prev_close, warn, warm,
                          line_times)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=STYLE["bg"])
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- one day

def chart_day(day: dt.date, symbols=SYMBOLS, timeframes=TIMEFRAMES, out: Path = OUT,
              lab_dir: Path = LAB, feed=None) -> list[dict]:
    """Draw every (symbol, timeframe) for one day. -> one report row per symbol."""
    cache = out / "_bars"
    rows = []
    for sym in symbols:
        rep = {"day": day.isoformat(), "symbol": sym, "charts": []}
        bars = load_bars(sym, day, cache, feed)
        prior = prior_session(sym, day, cache, feed)
        if bars is None or prior is None:
            rep["error"] = "bars not cached and no feed (run once without --offline)"
            rows.append(rep)
            continue
        if not _rth(bars):
            rep["error"] = "no market-hours bars (holiday?)"
            rows.append(rep)
            continue
        yday, ybars = prior
        logged = recorded_lines(lab_dir, sym, day)
        rebuilt = rebuilt_lines(ybars, bars)
        diff = compare_lines(logged, rebuilt) if logged else []
        lines = logged or rebuilt or None
        note = (("six lines: the runner's, logged 09:31" if logged else
                 "six lines: rebuilt (runner logged none before 09-25)"
                 if rebuilt else "six lines: unavailable") + f"  ·  yday {yday:%a %d %b}")
        warn = ""
        if diff:
            warn = (f"vendor revised since 09:31: {diff[0]}"
                    + (f"  (+{len(diff) - 1} more)" if len(diff) > 1 else ""))
        times = {l["name"]: set_at(l, ybars, bars) for l in (lines or [])}
        prev_close = _rth(ybars)[-1]["close"]
        rep.update({"yday": yday.isoformat(), "lines": lines, "lines_from":
                    "runner" if logged else "rebuilt", "mismatch": diff, "set_at": times,
                    "premarket_bars": sum(1 for b in bars if SHOW_FROM <= b["ts"].time() < RTH_OPEN),
                    "rth_bars": len(_rth(bars)), "first_bar": bars[0]["ts"].isoformat()})
        for m in timeframes:
            p = render(sym, day, m, bars, lines, note, prev_close,
                       out / day.isoformat() / f"{sym}_{m:02d}m.png", warn, warm=ybars,
                       line_times=times)
            rep["charts"].append(str(p.relative_to(out)))
        rows.append(rep)
    return rows


def write_index(out: Path = OUT) -> Path:
    days = sorted((p for p in out.iterdir() if p.is_dir() and p.name[:2] == "20"), reverse=True)
    cards = []
    for d in days:
        imgs = sorted(d.glob("*.png"))
        links = "".join(f'<a href="{d.name}/{i.name}"><img src="{d.name}/{i.name}" loading="lazy">'
                        f'<span>{i.stem.replace("_", " ")}</span></a>' for i in imgs)
        summ = d / "run_summary.md"
        extra = f' <a class="s" href="{d.name}/run_summary.md">run summary</a>' if summ.exists() else ""
        cards.append(f"<section><h2>{d.name}{extra}</h2><div class=g>{links}</div></section>")
    html = ("<!doctype html><meta charset=utf-8><title>Live lab day charts</title>"
            "<meta name=viewport content='width=device-width,initial-scale=1'><style>"
            "body{background:#141414;color:#cfcfcf;font:14px system-ui;margin:16px}"
            "h2{font-weight:600;margin:22px 0 8px}.g{display:grid;gap:10px;"
            "grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}"
            "a{color:#cfcfcf;text-decoration:none}img{width:100%;border-radius:6px;display:block}"
            "span{font-size:12px;color:#8a8a8a}.s{font-size:13px;color:#4fc3ff;margin-left:10px}"
            "</style><h1>Live lab day charts</h1>" + "".join(cards))
    p = out / "index.html"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, p)
    return p


def main(argv=None, now: dt.datetime | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--day", action="append", help="YYYY-MM-DD; repeatable (default: every lab day)")
    ap.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    ap.add_argument("--tf", nargs="+", type=int, default=list(TIMEFRAMES), choices=ALLOWED_TF)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--force", action="store_true", help="allow a network run during the session")
    a = ap.parse_args(argv)
    out = Path(a.out)
    now = now or _now()
    if not a.offline and not a.force and now.weekday() < 5 and BUSY_FROM <= now.time() < SETTLED_AT:
        print("refusing to use the data terminal during the session (09:00-16:05 ET); "
              "use --offline, or --force", file=sys.stderr)
        return 2
    days = [dt.date.fromisoformat(d) for d in a.day] if a.day else lab_days()
    feed = None
    if not a.offline:
        from .feed import ThetaLiveFeed
        feed = ThetaLiveFeed()
    from .feed import FeedOutage
    rc = 0
    try:
        for day in days:
            try:
                if a.fetch_only:
                    for sym in a.symbols:
                        ok = load_bars(sym, day, out / "_bars", feed) is not None and \
                            prior_session(sym, day, out / "_bars", feed) is not None
                        rc |= 0 if ok else 1
                        print(f"{day} {sym}: {'cached' if ok else 'MISSING'}", flush=True)
                    continue
                for rep in chart_day(day, a.symbols, a.tf, out, feed=feed):
                    if rep.get("error"):
                        rc = 1
                        print(f"{rep['day']} {rep['symbol']}: ERROR {rep['error']}", flush=True)
                        continue
                    print(f"{rep['day']} {rep['symbol']}: {len(rep['charts'])} chart(s), lines "
                          f"{rep['lines_from']}, premarket bars {rep['premarket_bars']}, "
                          f"rth bars {rep['rth_bars']}"
                          + (f", MISMATCH {rep['mismatch']}" if rep["mismatch"] else ""), flush=True)
            except FeedOutage as exc:           # one day's outage must not cost the rest
                rc = 1
                print(f"{day}: FEED OUTAGE, skipped: {exc!r}"[:200], flush=True)
    finally:
        if feed is not None:
            feed.close()
    if not a.fetch_only and out.exists():
        print(f"index: {write_index(out)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
