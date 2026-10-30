"""task_id: R-2026-10-30-04 | parent: R-2026-10-30-01 | role: feature-script-agent | repo_sha: a367245 | status: done | verified_by: null

Per-session day features for the news-impact study (research/news_impact/preregistration.md §4).

    python -m trade_analysis.news_impact.day_features --symbol QQQ           # laptop, needs Theta
    python -m trade_analysis.news_impact.day_features --symbol SPY
    python -m trade_analysis.news_impact.day_features --lab-days             # anywhere, from git

One row per session, columns exactly as the pre-registration table plus three band columns
(`rel_range_band`, `rel_gap_band`, `impact`) that the same section defines. Every threshold
is a frozen constant at the top of this file; nothing here is tuned after a run.

Imports `six_lines.build_six`, `six_lines.breaks_today`, `levels_test.get_ext`,
`walkforward.sessions_between`; edits none of them.

THE LOOKAHEAD RULE, spelled out
-------------------------------
`med20_range_bp` and `med20_absgap_bp` are medians over the PREVIOUS `LOOKBACK` sessions,
today excluded. `features_for_session` takes the history as an argument and never appends
to it; the caller (`run`, via `push_history`) appends today's values only AFTER the row is
built. The six lines come from yesterday's full extended session and today's premarket
only (`build_six`), so `ranged_a` sees nothing of today's market hours except through
`breaks_today`, which is what it measures.

THIN DAYS (fewer than MIN_RTH_BARS RTH bars: half days, data holes)
--------------------------------------------------------------------
A thin day still gets a row (`thin = True`) with whatever can be computed, so the file has
one line per weekday the feed answered. It is excluded from every statistic downstream.
Two further consequences, both deliberate:

  * a thin day does NOT enter the trailing-20 history. A 13:00 close has roughly half a
    normal day's range; letting it into the median would bias `rel_range` upward for the
    next twenty sessions.
  * the reference "previous session" for the NEXT day -- both for `build_six` (yesterday's
    four lines) and for `prev_close` -- stays the last NON-thin session. The gap on the day
    after a half day therefore spans the half day too (about three sessions a year); the
    alternative, trusting a thin day's last bar as its close, would silently use an 11:00
    print as "the close" on a day with a data hole. The uniform rule is preferred over the
    one that is right on half days and wrong on holes.

WHEN THE HISTORY IS SHORT (fewer than LOOKBACK prior sessions)
--------------------------------------------------------------
`med20_*`, `rel_*`, the bands, `impact`, `gap_reversal`, `ranged_b` and `day_class` are
None. The row is still written. `day_class` is None whenever either `ranged_a` or
`ranged_b` is unknown: a classification with a missing input is not written, even when
the trend test alone would have decided it. This is the orchestrator's instruction
(R-2026-10-30-04), stricter than the §4 table, which is silent on the case.

OUTPUT
------
CSV (fixed column order `COLUMNS`; booleans as True/False, None as an empty cell) and a
sibling JSON holding the same rows, the frozen constants, the diagnostics dict and the
repo git SHA, so a number in the CSV can always be traced to the code that produced it.

Amendment A6 (§9): `prev_session_date` names the non-thin session whose bars built today's
lines and `prev_close`, so a skipped half day is visible in the row; the diagnostics carry
`thin_by_year`.

`--lab-days` is the single place this module reads `live_lab_data/` other than
`ext_cache`: it builds `research/news_impact/lab_days.csv` from the three lab record files
(read-only) and needs no feed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics as st
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from ..live_lab.six_lines import PRE_FROM, PRE_TO, RTH_FROM, RTH_TO, _hi_lo, breaks_today, build_six

# ------------------------------------------------------------------ frozen (preregistration §4)
RANGED_B_REL = 0.6                     # ranged_b: rel_range < 0.6 (strict)
TREND_MIN = 0.6                        # day_class trend: |co_bp| / range_bp >= 0.6
BANDS = {"rel_range": (0.8, 1.5),      # low < 0.8, medium 0.8-1.5, high > 1.5
         "rel_gap": (1.0, 2.5)}        # low < 1.0, medium 1.0-2.5, high > 2.5
GAP_REVERSAL_MIN_REL = 1.0             # gap_reversal needs rel_gap >= 1.0
MIN_RTH_BARS = 300                     # fewer -> thin
LOOKBACK = 20                          # trailing sessions for the medians
FIRST60_TO = dt.time(10, 30)           # first60 window is 09:30-10:29 inclusive
BAND_ORDER = ("low", "medium", "high")

REPO = Path(__file__).resolve().parents[2]
PREREG = "research/news_impact/preregistration.md"

COLUMNS = ["date", "pre_high", "pre_low", "open", "high", "low", "close", "prev_close",
           "prev_session_date", "gap_bp", "range_bp", "co_bp", "first60_hi_bp", "first60_lo_bp",
           "med20_range_bp", "med20_absgap_bp", "rel_range", "rel_gap",
           "n_line_breaks", "ranged_a", "ranged_b", "day_class", "gap_reversal", "thin",
           "rel_range_band", "rel_gap_band", "impact"]

LAB_COLUMNS = ["date", "options_atm_net", "shares_net", "options_green", "shares_green",
               "both_green", "ledger_outcome"]


# ------------------------------------------------------------------ small pure helpers

def _bp(num: float | None, den: float | None) -> float | None:
    """(num / den - 1) x 10,000, rounded to 4 dp so float noise never decides a band."""
    if num is None or den is None or den == 0:
        return None
    return round((num / den - 1.0) * 10_000.0, 4)


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return round(num / den, 6)


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def band(x: float | None, edges: tuple[float, float]) -> str | None:
    """low below the first edge (strict), high above the second (strict), medium between
    and ON both edges -- the §4 table writes the middle band as '0.8 - 1.5' inclusive."""
    if x is None:
        return None
    lo, hi = edges
    if x < lo:
        return "low"
    if x > hi:
        return "high"
    return "medium"


def impact_of(b1: str | None, b2: str | None) -> str | None:
    """The higher of the two bands; None if either is unknown."""
    if b1 is None or b2 is None:
        return None
    return BAND_ORDER[max(BAND_ORDER.index(b1), BAND_ORDER.index(b2))]


def _last_rth_bar(bars: Sequence[dict]) -> dict | None:
    last = None
    for b in bars:
        if RTH_FROM <= b["ts"].time() < RTH_TO and (last is None or b["ts"] > last["ts"]):
            last = b
    return last


def rth_close(bars: Sequence[dict]) -> float | None:
    """Close of the last bar whose time is inside market hours. For yesterday's extended
    session this is the session close; premarket bars are ignored."""
    last = _last_rth_bar(bars)
    return None if last is None else last["close"]


def session_date(bars: Sequence[dict] | None) -> dt.date | None:
    """The date of a session's bars (its last RTH bar, else its first bar)."""
    if not bars:
        return None
    last = _last_rth_bar(bars)
    return (last or bars[0])["ts"].date()


def med20(history: Sequence[tuple[float, float]]) -> tuple[float | None, float | None]:
    """Medians of range_bp and |gap_bp| over the LAST `LOOKBACK` history entries. None
    unless the history holds at least LOOKBACK entries. The caller guarantees today is
    not in `history`."""
    if len(history) < LOOKBACK:
        return None, None
    tail = list(history)[-LOOKBACK:]
    return (round(st.median(r for r, _ in tail), 4),
            round(st.median(abs(g) for _, g in tail), 4))


# ------------------------------------------------------------------ the feature row

def features_for_session(prev_ext_bars: Sequence[dict] | None, pre_bars: Sequence[dict],
                         rth_bars: Sequence[dict], history: Sequence[tuple[float, float]],
                         date: dt.date | None = None) -> dict:
    """One §4 row. Pure: bars and history in, dict out; `history` is never modified.

    prev_ext_bars  the last NON-thin session's full extended bars (premarket + RTH), or
                   None when there is none. Supplies yesterday's four lines and prev_close.
    pre_bars       today's 04:00-09:29 1-minute bars.
    rth_bars       today's 09:30-15:59 1-minute bars.
    history        (range_bp, gap_bp) of previous non-thin sessions, oldest first, TODAY
                   EXCLUDED. Only the last LOOKBACK entries are used.
    date           the session date; derived from the bars when omitted.
    """
    if date is None:
        src = rth_bars[0] if rth_bars else (pre_bars[0] if pre_bars else None)
        date = None if src is None else src["ts"].date()

    thin = len(rth_bars) < MIN_RTH_BARS
    pre_high, pre_low = _hi_lo(pre_bars, PRE_FROM, PRE_TO)
    rth = sorted((b for b in rth_bars if RTH_FROM <= b["ts"].time() < RTH_TO),
                 key=lambda b: b["ts"])
    open_ = rth[0]["open"] if rth else None
    high, low = _hi_lo(rth, RTH_FROM, RTH_TO)
    close = rth[-1]["close"] if rth else None
    prev_close = rth_close(prev_ext_bars) if prev_ext_bars else None
    prev_date = session_date(prev_ext_bars)

    gap_bp = _bp(open_, prev_close)
    range_bp = _bp(high, low)
    co_bp = _bp(close, open_)
    f60_hi, f60_lo = _hi_lo(rth, RTH_FROM, FIRST60_TO)
    first60_hi_bp, first60_lo_bp = _bp(f60_hi, open_), _bp(f60_lo, open_)

    med_range, med_absgap = med20(history)
    rel_range = _ratio(range_bp, med_range)
    rel_gap = _ratio(None if gap_bp is None else abs(gap_bp), med_absgap)

    # six lines: yesterday's full session + today's premarket; [] when a source is missing
    n_line_breaks = ranged_a = None
    lines = build_six(prev_ext_bars, pre_bars) if prev_ext_bars else []
    if lines and rth:                          # no RTH bars -> no tally, not "zero breaks"
        rec = breaks_today(rth, lines)
        n_line_breaks = sum(1 for v in rec.values() if v["close_break"])
        ranged_a = n_line_breaks == 0

    ranged_b = None if rel_range is None else rel_range < RANGED_B_REL

    day_class = None
    if ranged_a is not None and ranged_b is not None and co_bp is not None and range_bp:
        if abs(co_bp) / range_bp >= TREND_MIN:
            day_class = "trend"
        elif ranged_a or ranged_b:
            day_class = "ranged"
        else:
            day_class = "chop"

    gap_reversal = None
    if rel_gap is not None and co_bp is not None and gap_bp is not None:
        gap_reversal = rel_gap >= GAP_REVERSAL_MIN_REL and _sign(co_bp) != _sign(gap_bp)

    rr_band, rg_band = band(rel_range, BANDS["rel_range"]), band(rel_gap, BANDS["rel_gap"])

    return {"date": None if date is None else date.isoformat(),
            "pre_high": pre_high, "pre_low": pre_low,
            "open": open_, "high": high, "low": low, "close": close, "prev_close": prev_close,
            "prev_session_date": None if prev_date is None else prev_date.isoformat(),
            "gap_bp": gap_bp, "range_bp": range_bp, "co_bp": co_bp,
            "first60_hi_bp": first60_hi_bp, "first60_lo_bp": first60_lo_bp,
            "med20_range_bp": med_range, "med20_absgap_bp": med_absgap,
            "rel_range": rel_range, "rel_gap": rel_gap,
            "n_line_breaks": n_line_breaks, "ranged_a": ranged_a, "ranged_b": ranged_b,
            "day_class": day_class, "gap_reversal": gap_reversal, "thin": thin,
            "rel_range_band": rr_band, "rel_gap_band": rg_band,
            "impact": impact_of(rr_band, rg_band)}


def push_history(history: list, row: dict) -> bool:
    """Append today's (range_bp, gap_bp) AFTER the row is built. Returns whether it did.
    Thin rows never enter; neither does a row whose gap is unknown (no previous close), so
    the two medians always rest on the same sessions."""
    if row["thin"] or row["range_bp"] is None or row["gap_bp"] is None:
        return False
    history.append((row["range_bp"], row["gap_bp"]))
    return True


# ------------------------------------------------------------------ the run (laptop)

def build_rows(days, quiet: bool = False) -> tuple[list[dict], dict]:
    """The whole walk, feed-free: `days` yields (date, premarket_bars, rth_bars) in session
    order. Keeps the trailing history and the last NON-thin session as `prev`; a thin day
    gets its row and changes neither. Returns (rows, diagnostics)."""
    rows, history, prev = [], [], None
    diag = defaultdict(int)
    thin_by_year: dict[str, int] = defaultdict(int)
    for day, pre, rth in days:
        diag["days_tried"] += 1
        if not quiet and diag["days_tried"] % 200 == 0:
            print(f"  {diag['days_tried']} days tried, {len(rows)} rows, "
                  f"{diag['thin']} thin, {diag['sessions_without_lines']} without lines ...",
                  flush=True)
        if not pre and not rth:
            diag["no_bars"] += 1                                  # holiday or outage: no row
            continue
        row = features_for_session(prev, pre, rth, history, date=day)
        rows.append(row)
        diag["rows_written"] += 1
        if row["thin"]:
            diag["thin"] += 1
            thin_by_year[str(day.year)] += 1
            continue                                              # prev and history unchanged
        if prev is None:
            diag["sessions_without_prev"] += 1
        elif row["n_line_breaks"] is None:
            diag["sessions_without_lines"] += 1
        if row["med20_range_bp"] is None:
            diag["rows_history_short"] += 1
        push_history(history, row)
        prev = list(pre) + list(rth)
    out = dict(diag)
    out["thin_by_year"] = dict(sorted(thin_by_year.items()))
    return rows, out


def run(symbol: str, start: dt.date, end: dt.date) -> tuple[list[dict], dict]:
    from ..live_lab.feed import ThetaLiveFeed
    from ..live_lab.levels_test import get_ext
    from ..live_lab.walkforward import sessions_between

    feed = ThetaLiveFeed()
    skipped = 0

    def days():
        nonlocal skipped
        for day in sessions_between(start, end):
            try:
                pre, rth = get_ext(feed, symbol, day)
            except Exception as exc:                              # noqa: BLE001
                skipped += 1
                print(f"  {day}: skipped ({exc!r})", flush=True)
                continue
            yield day, pre, rth

    try:
        rows, diag = build_rows(days())
    finally:
        feed.close()
    diag["feed_skipped"] = skipped
    print(f"\n  {diag.get('days_tried', 0) + skipped} days tried, {diag.get('thin', 0)} thin, "
          f"{diag.get('sessions_without_lines', 0)} sessions without lines, "
          f"{diag.get('rows_written', 0)} rows written")
    return rows, diag


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                             # noqa: BLE001
        return None


def write_outputs(rows: list[dict], diag: dict, out: Path, meta: dict) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})
    side = out.with_suffix(".json")
    side.write_text(json.dumps({"pre_registration": PREREG, **meta, "git_sha": git_sha(),
                                "constants": {"RANGED_B_REL": RANGED_B_REL, "TREND_MIN": TREND_MIN,
                                              "BANDS": BANDS, "MIN_RTH_BARS": MIN_RTH_BARS,
                                              "LOOKBACK": LOOKBACK},
                                "diagnostics": diag, "columns": COLUMNS, "rows": rows},
                               indent=1), encoding="utf-8")
    return side


# ------------------------------------------------------------------ lab join (from git)

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def lab_days(repo_root: Path | str, out: Path | None = None) -> list[dict]:
    """research/news_impact/lab_days.csv from the lab record as it stands in git.

    options_atm_net   sum of pnl_net over live_lab_data/trades.jsonl rows with arm == "ATM"
                      and pnl_net not None, keyed by entry_ts[:10]
    shares_net        sum of pnl_net over live_lab_data/shares/trades.jsonl, same key
    *_green           net > 0; None when that arm has no trade that day
    both_green        True only when both arms are green; None when either is unknown
    ledger_outcome    the LAST session_ledger.jsonl outcome for the date that is not OPENED

    Dates are the union of trade dates and ledger dates whose final outcome is not
    NOT_A_SESSION. Read-only on live_lab_data.
    """
    root = Path(repo_root)
    lab = root / "live_lab_data"
    opt, sh = defaultdict(float), defaultdict(float)
    for r in _read_jsonl(lab / "trades.jsonl"):
        if r.get("arm") == "ATM" and r.get("pnl_net") is not None and r.get("entry_ts"):
            opt[r["entry_ts"][:10]] += float(r["pnl_net"])
    for r in _read_jsonl(lab / "shares" / "trades.jsonl"):
        if r.get("pnl_net") is not None and r.get("entry_ts"):
            sh[r["entry_ts"][:10]] += float(r["pnl_net"])
    ledger: dict[str, str] = {}
    for r in _read_jsonl(lab / "session_ledger.jsonl"):       # file order = record order
        if r.get("date") and r.get("outcome") and r["outcome"] != "OPENED":
            ledger[r["date"]] = r["outcome"]

    dates = set(opt) | set(sh) | {d for d, o in ledger.items() if o != "NOT_A_SESSION"}
    rows = []
    for d in sorted(dates):
        o = round(opt[d], 4) if d in opt else None
        s = round(sh[d], 4) if d in sh else None
        og = None if o is None else o > 0
        sg = None if s is None else s > 0
        both = None if og is None or sg is None else (og and sg)
        rows.append({"date": d, "options_atm_net": o, "shares_net": s,
                     "options_green": og, "shares_green": sg, "both_green": both,
                     "ledger_outcome": ledger.get(d)})
    if out is None:
        out = root / "research" / "news_impact" / "lab_days.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LAB_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in LAB_COLUMNS})
    return rows


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--start", default="2016-01-04")
    ap.add_argument("--end", default="2026-10-29")
    ap.add_argument("--out", default=None,
                    help="CSV path; default research/news_impact/day_features_{symbol}.csv")
    ap.add_argument("--lab-days", action="store_true",
                    help="write research/news_impact/lab_days.csv from the lab record and exit")
    a = ap.parse_args(argv)

    if a.lab_days:
        rows = lab_days(REPO)
        print(f"lab_days: {len(rows)} dates -> {REPO / 'research/news_impact/lab_days.csv'}")
        return 0

    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    print(f"DAY FEATURES  {a.symbol}  {start} -> {end}   ({PREREG} §4)", flush=True)
    rows, diag = run(a.symbol, start, end)
    if not rows:
        print("nothing recorded; is Theta Terminal running?")
        return 1
    out = Path(a.out) if a.out else REPO / "research" / "news_impact" / f"day_features_{a.symbol}.csv"
    side = write_outputs(rows, diag, out, {"symbol": a.symbol, "start": a.start, "end": a.end,
                                            "run_at": dt.datetime.now().isoformat(timespec="seconds")})
    print("  diagnostics:", json.dumps(diag, indent=2))
    print(f"\nwritten: {out}\n         {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
