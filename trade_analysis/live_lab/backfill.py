"""BACKFILL -- replay the CURRENT options config over past sessions, through the real runner.

    python -m trade_analysis.live_lab.backfill --start 2026-08-28 --end 2026-09-24
    python -m trade_analysis.live_lab.backfill --report

THIS IS NOT THE FORWARD TEST, AND NOTHING HERE MAY BE COUNTED AS IF IT WERE
---------------------------------------------------------------------------
The live lab's central guarantee is that a decision is on disk before any price that could
influence it exists. A replay run after the fact cannot have that property, however
faithfully it reuses the code: the prices already existed when this ran. So the output
lives in `live_lab_backfill/`, OUTSIDE `live_lab_data/`, is never read by the dashboard,
the checkpoint diagnostic or any promotion bar, and every record carries
`"provenance": "BACKFILL"`.

What it is for: the 2026-09-25 amendment added the trader's six-line setups. This answers
"what would the new config have done on the sessions the lab already ran", so the
prospective record can be read against it without either one overwriting the other.

HOW IT RUNS
-----------
The real `runner.LiveLab`, unmodified. Only two things are swapped:

  * the feed -- `HistoryFeed` serves ThetaData HISTORY through the live feed's interface:
    RTH 1m bars, the 1m NBBO, the full 0DTE chain at 1-minute resolution (one bulk call per
    symbol-day), and extended-hours bars for the six lines. It never serves a bar that has
    not closed, a quote stamped after the simulated clock, or any date after the day being
    replayed.
  * the clock -- `runner.now_et` and `store.now_et` read a simulated clock, which steps
    once a minute at HH:MM:02, just after each bar can be admitted (T + 60s + 1.5s settle).

Everything else -- warmup, the six-line loader, every setup, the stale-bar, stale-quote and
option-quote guards, arm selection, fills at the ask, exits at the bid, the 15:55 flatten
-- is the production path. Warmup runs at 09:07, as it now does live.

KNOWN DIFFERENCES FROM A LIVE SESSION, stated rather than hidden
  * Quotes are 1-minute snapshots. Live polls every 5s, so a fill here is the NBBO at the
    minute boundary, not a few seconds later.
  * The feed never drops. Live sessions had outages, late starts and a blind opening
    window (incident_2026-09-24). This replay has none of them.
  * Open interest is not replayed; trades carry an empty OI.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import pickle
from collections import defaultdict
from pathlib import Path

from . import runner as R
from . import store as S
from .feed import FeedOutage, _parse_ts
from .levels_live import LevelsLoader
from .walkforward import sessions_between

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "live_lab_backfill" / "options"
CACHE = ROOT / ".cache" / "backfill"
STEP = dt.timedelta(minutes=1)
TICK_OFFSET = dt.timedelta(seconds=2)
WARMUP_AT = dt.time(9, 7)
MIN_RTH_BARS = 300


class SimClock:
    def __init__(self):
        self.now = dt.datetime(2000, 1, 1)

    def __call__(self) -> dt.datetime:
        return self.now


def _cached(key: str, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{key}.pkl"
    if p.exists():
        try:
            with open(p, "rb") as fh:
                return pickle.load(fh)
        except Exception:                                    # noqa: BLE001
            pass
    val = fn()
    with open(p, "wb") as fh:
        pickle.dump(val, fh)
    return val


class HistoryFeed:
    """The live feed's interface, served from history as of a simulated clock."""

    def __init__(self, real, clock: SimClock, day: dt.date):
        self.real, self.clock, self.day = real, clock, day
        self.calls = 0
        self._rth: dict = {}
        self._ext: dict = {}
        self._nbbo: dict = {}
        self._chain: dict = {}
        self._strikes: dict = {}

    # ---------------------------------------------------------------- guards
    def _no_future(self, day: dt.date) -> None:
        if day > self.day:
            raise AssertionError(f"backfill asked for {day} while replaying {self.day}")

    def _closed(self, bars):
        now = self.clock.now
        return [b for b in bars if b["ts"] + STEP <= now]

    # ---------------------------------------------------------------- bars
    def minute_bars(self, symbol, day, now=None):
        self._no_future(day)
        key = (symbol, day)
        if key not in self._rth:
            self._rth[key] = _cached(f"rth_{symbol}_{day}",
                                     lambda: self.real._fetch_minute_bars(symbol, day))
        bars = self._rth[key]
        # The replayed day is ALWAYS cut at the simulated clock, whatever `now` says.
        return self._closed(bars) if day == self.day else list(bars)

    def extended_bars(self, symbol, day, start=dt.time(4, 0), end=dt.time(16, 0)):
        self._no_future(day)
        key = (symbol, day)
        if key not in self._ext:
            self._ext[key] = _cached(
                f"ext_{symbol}_{day}",
                lambda: self.real.extended_bars(symbol, day, dt.time(4, 0), dt.time(20, 0)))
        bars = [b for b in self._ext[key] if start <= b["ts"].time() < end]
        return self._closed(bars) if day == self.day else bars

    # ---------------------------------------------------------------- quotes
    def _nbbo_rows(self, symbol):
        if symbol not in self._nbbo:
            def fetch():
                rows = self.real._get_csv("/stock/history/quote", symbol=symbol,
                                          date=self.day.isoformat(), interval="1m",
                                          start_time="09:30:00", end_time="16:00:00")
                out = []
                for r in rows:
                    try:
                        out.append((_parse_ts(r["timestamp"]), float(r["bid"]),
                                    float(r["ask"]), float(r.get("bid_size") or 0),
                                    float(r.get("ask_size") or 0)))
                    except (KeyError, ValueError):
                        continue
                out.sort()
                return out
            self._nbbo[symbol] = _cached(f"nbbo_{symbol}_{self.day}", fetch)
        return self._nbbo[symbol]

    def stock_quote(self, symbol):
        rows = self._nbbo_rows(symbol)
        i = bisect.bisect_right([r[0] for r in rows], self.clock.now) - 1
        if i < 0:
            return None
        ts, bid, ask, bs, az = rows[i]
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        return {"ts": ts, "recv_ts": self.clock.now, "bid": bid, "ask": ask,
                "mid": (bid + ask) / 2.0, "bid_size": bs, "ask_size": az}

    # ---------------------------------------------------------------- options
    def zero_dte(self, symbol, day):
        self._no_future(day)
        if symbol not in self._strikes:
            self._strikes[symbol] = _cached(
                f"strikes_{symbol}_{day}",
                lambda: len(self.real._get_csv("/option/list/strikes", symbol=symbol,
                                               expiration=day.isoformat())))
        return day if self._strikes[symbol] else None

    def _chain_table(self, symbol, expiration):
        key = (symbol, expiration)
        if key not in self._chain:
            def fetch():
                rows = self.real._get_csv("/option/history/quote", symbol=symbol,
                                          expiration=expiration.isoformat(),
                                          date=self.day.isoformat(), interval="1m",
                                          start_time="09:30:00", end_time="16:00:00")
                tab = defaultdict(list)
                for r in rows:
                    try:
                        k = (float(r["strike"]),
                             "call" if r["right"].strip('"').upper().startswith("C") else "put")
                        tab[k].append((_parse_ts(r["timestamp"]), float(r["bid"]),
                                       float(r["ask"]), float(r.get("bid_size") or 0),
                                       float(r.get("ask_size") or 0)))
                    except (KeyError, ValueError):
                        continue
                return {k: sorted(v) for k, v in tab.items()}
            tab = _cached(f"chain_{symbol}_{expiration}_{self.day}", fetch)
            self._chain[key] = {k: ([x[0] for x in v], v) for k, v in tab.items()}
        return self._chain[key]

    def chain_quotes(self, symbol, expiration):
        """Every contract's latest quote at or before the simulated clock, filtered exactly
        as ThetaLiveFeed.chain_quotes filters a snapshot."""
        self._no_future(expiration)
        now, out = self.clock.now, []
        for (strike, right), (stamps, rows) in self._chain_table(symbol, expiration).items():
            i = bisect.bisect_right(stamps, now) - 1
            if i < 0:
                continue
            ts, bid, ask, bs, az = rows[i]
            if ask <= 0 or ask < bid or bid < 0:
                continue
            out.append({"ts": ts, "strike": strike, "right": right, "bid": bid, "ask": ask,
                        "mid": (bid + ask) / 2.0, "spread": ask - bid,
                        "bid_size": bs, "ask_size": az})
        return out

    def open_interest(self, symbol, expiration):
        return {}

    # ---------------------------------------------------------------- plumbing
    def wait_for_upstream(self, *a, **k):
        return True

    def stats(self):
        return {"source": "HISTORY_BACKFILL", "calls": self.calls, "retries": 0,
                "bar_cache_hits": 0, "bar_cache_misses": 0, "outage_episodes": 0,
                "upstream_down_now": False}

    def close(self):
        pass


def _tag_backfill(store):
    """Every record this run writes says what it is."""
    for name in ("write_trade", "write_decision", "write_fill", "write_skip", "event",
                 "outage"):
        orig = getattr(store, name)

        def wrapped(*a, _orig=orig, _name=name, **k):
            if _name == "write_trade":
                a = ({**a[0], "provenance": "BACKFILL"},) + a[1:]
            elif _name == "write_decision":          # keyword-only, no **fields
                k["state"] = {**(k.get("state") or {}), "provenance": "BACKFILL"}
            else:
                k.setdefault("provenance", "BACKFILL")
            return _orig(*a, **k)
        setattr(store, name, wrapped)


def run(start: dt.date, end: dt.date, symbols=("QQQ", "SPY"), out=OUT) -> list[dict]:
    clock = SimClock()
    # store._now imports now_et INSIDE the function, so patching a module attribute would
    # miss it and stamp every record with tonight's wall clock. Patch _now itself.
    R.now_et, S._now = clock, clock
    out.mkdir(parents=True, exist_ok=True)
    for f in ("signals.jsonl", "trades.jsonl", "events.jsonl", "outages.jsonl"):
        (out / f).unlink(missing_ok=True)      # a backfill is regenerated whole, never appended
    clock.now = dt.datetime.combine(start, WARMUP_AT)
    lab = R.LiveLab(list(symbols), lab_dir=out)
    real = lab.feed
    _tag_backfill(lab.store)
    summary = []
    try:
        for day in sessions_between(start, end):
            feed = HistoryFeed(real, clock, day)
            clock.now = dt.datetime.combine(day, dt.time(16, 1))   # only to COUNT the day's bars
            if len(feed.minute_bars(symbols[0], day)) < MIN_RTH_BARS:
                print(f"  {day}: not a session, skipped", flush=True)
                continue
            lab.feed = feed
            lab.sessions, lab.open_pos = {}, []
            lab.trade_counts, lab.dir_counts = {}, {}
            lab._seen_5m, lab._degraded_seen, lab._chain_cache = {}, {}, {}
            lab.levels = LevelsLoader(lab.store, tag="backfill")
            clock.now = dt.datetime.combine(day, WARMUP_AT)
            if not lab.warmup(day):
                print(f"  {day}: warmup failed, skipped", flush=True)
                continue
            t = dt.datetime.combine(day, R.RTH_OPEN) + TICK_OFFSET
            close_at = dt.datetime.combine(day, R.RTH_CLOSE)
            while t < close_at:
                clock.now = t
                lab._chain_cache.clear()          # its 2s TTL is wall-clock; a replay minute is ~ms
                try:
                    lab._tick(t, day)
                except FeedOutage as exc:
                    lab.store.outage("tick", repr(exc))
                t += STEP
            clock.now = close_at
            lab._flatten_all(close_at, reason="shutdown")
            lab.write_daily(day)
            d = json.loads((out / "daily" / f"{day}.json").read_text(encoding="utf-8"))
            summary.append(d)
            print(f"  {day}: {d.get('trades_closed')} trades, ATM net {d.get('atm_net')}",
                  flush=True)
    finally:
        real.close()
    (out / "BACKFILL.json").write_text(json.dumps({
        "provenance": "BACKFILL -- NOT PROSPECTIVE. Never counted by the dashboard, the "
                      "checkpoint diagnostic or any promotion bar.",
        "config_hash": lab.config_hash, "start": start.isoformat(), "end": end.isoformat(),
        "symbols": list(symbols), "generated_by": "trade_analysis/live_lab/backfill.py",
        "differences_from_live": ["1-minute quote snapshots, live polls every 5s",
                                  "no feed outages, late starts or blind windows",
                                  "open interest not replayed"],
    }, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------- report

def _atm(path: Path, accepted=None):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("arm") != "ATM":
            continue
        if accepted is not None and r.get("config_hash") not in accepted:
            continue
        out.append(r)
    return out


def report(start: dt.date, end: dt.date, out=OUT) -> dict:
    lab = ROOT / "live_lab_data"
    accepted = set(json.loads((lab / "FREEZE.json").read_text(encoding="utf-8"))
                   ["accepted_config_hashes"])
    live = [r for r in _atm(lab / "trades.jsonl", accepted)
            if start.isoformat() <= r["entry_ts"][:10] <= end.isoformat()]
    back = _atm(out / "trades.jsonl")
    new = {"Six_Lines", "Six_Lines_NoCap"}
    days = sorted({r["entry_ts"][:10] for r in live} | {r["entry_ts"][:10] for r in back})
    rows = []
    for d in days:
        lv = [r["pnl_net"] for r in live if r["entry_ts"][:10] == d]
        b13 = [r["pnl_net"] for r in back if r["entry_ts"][:10] == d and r["setup_id"] not in new]
        b6 = [r["pnl_net"] for r in back if r["entry_ts"][:10] == d and r["setup_id"] == "Six_Lines"]
        b6n = [r["pnl_net"] for r in back
               if r["entry_ts"][:10] == d and r["setup_id"] == "Six_Lines_NoCap"]
        rows.append({"day": d, "live_n": len(lv), "live_net": round(sum(lv), 2),
                     "back13_n": len(b13), "back13_net": round(sum(b13), 2),
                     "six_n": len(b6), "six_net": round(sum(b6), 2),
                     "sixnc_n": len(b6n), "sixnc_net": round(sum(b6n), 2),
                     "newcfg_net": round(sum(b13) + sum(b6) + sum(b6n), 2)})
    by_setup = defaultdict(lambda: {"live_n": 0, "live_net": 0.0, "back_n": 0, "back_net": 0.0})
    for r in live:
        s = by_setup[r["setup_id"]]; s["live_n"] += 1; s["live_net"] += r["pnl_net"]
    for r in back:
        s = by_setup[r["setup_id"]]; s["back_n"] += 1; s["back_net"] += r["pnl_net"]
    return {"days": rows, "by_setup": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv
                                           for kk, vv in v.items()}
                                       for k, v in sorted(by_setup.items())},
            "six_trades": [{k: r.get(k) for k in ("entry_ts", "symbol", "setup_id", "direction",
                                                  "strike", "entry_ask", "exit_bid",
                                                  "exit_reason", "pnl_net", "underlying_return")}
                           for r in back if r["setup_id"] in new]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2026-08-28")
    ap.add_argument("--end", default="2026-09-24")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    s, e = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    if not a.report:
        run(s, e)
    rep = report(s, e)
    (OUT / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["days"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
