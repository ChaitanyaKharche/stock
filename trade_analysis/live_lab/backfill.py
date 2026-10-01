"""BACKFILL -- replay the CURRENT config of either arm over past sessions, through the real runner.

    python -m trade_analysis.live_lab.backfill --arm both --start 2026-08-28 --end 2026-09-29
    python -m trade_analysis.live_lab.backfill --arm both --report     # compare only
    python -m trade_analysis.live_lab.replay_audit                     # why live differs

THIS IS NOT THE FORWARD TEST, AND NOTHING HERE MAY BE COUNTED AS IF IT WERE
---------------------------------------------------------------------------
The live lab's central guarantee is that a decision is on disk before any price that could
influence it exists. A replay run after the fact cannot have that property, however
faithfully it reuses the code: the prices already existed when this ran. So the output
lives in `live_lab_backfill/`, OUTSIDE `live_lab_data/`, is never read by the dashboard,
the checkpoint diagnostic or any promotion bar, and every record carries
`"provenance": "BACKFILL"`.

What it is for:
  * the 2026-09-25 amendment added the trader's six-line setups. This answers "what would
    the new config have done on the sessions the lab already ran", so the prospective
    record can be read against it without either one overwriting the other.
  * a complete session log for every day, as a runner with a perfect connection would have
    written it: `<out>/logs/<day>.log`, `daily/`, `signals_archive/`, `trades.jsonl`. The live
    record lost minutes to late starts, sleeps and feed outages
    (research/replay_22_sessions.md); this is the same day with none of them.

HOW IT RUNS
-----------
The real `runner.LiveLab` (options) or `shares_runner.SharesLab` (shares), unmodified.
Only two things are swapped:

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
  * Bars are the vendor's FINAL bars. Live saw first prints, which the vendor later
    corrects: on identical decisions the runner's VWAP differed by a median 1.6 bp. So a
    replay matches live in totals, not trade for trade -- on clean hours about 3 in 4 live
    trades have a replay twin (research/replay_22_sessions.md).
"""
from __future__ import annotations

import argparse
import bisect
import contextlib
import datetime as dt
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

from . import runner as R
from . import store as S
from .feed import FeedOutage, _parse_ts
from .catchup import CatchUp
from .levels_live import LevelsLoader
from .walkforward import sessions_between

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "live_lab_backfill" / "options"
OUT_SHARES = ROOT / "live_lab_backfill" / "shares"
CACHE = ROOT / ".cache" / "backfill"
STEP = dt.timedelta(minutes=1)
TICK_OFFSET = dt.timedelta(seconds=2)
WARMUP_AT = dt.time(9, 7)
MIN_RTH_BARS = 300
SIX_LIVE_FROM = "2026-09-25"     # the amendment that put the six lines into the live config


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

    def __init__(self, real, clock: SimClock, day: dt.date, disk_cache: bool = True):
        self.real, self.clock, self.day = real, clock, day
        # A live runner catching up mid-session must NOT disk-cache: today's history is
        # still growing, and a cached partial day would later be served to the backfill
        # and gap-recovery tools as if it were the whole session.
        self.disk_cache = disk_cache
        self.calls = 0
        self._rth: dict = {}
        self._ext: dict = {}
        self._nbbo: dict = {}
        self._chain: dict = {}
        self._strikes: dict = {}

    def _c(self, key, fn):
        return _cached(key, fn) if self.disk_cache else fn()

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
            self._rth[key] = self._c(f"rth_{symbol}_{day}",
                                     lambda: self.real._fetch_minute_bars(symbol, day))
        bars = self._rth[key]
        # The replayed day is ALWAYS cut at the simulated clock, whatever `now` says.
        return self._closed(bars) if day == self.day else list(bars)

    def extended_bars(self, symbol, day, start=dt.time(4, 0), end=dt.time(16, 0)):
        self._no_future(day)
        key = (symbol, day)
        if key not in self._ext:
            self._ext[key] = self._c(
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
            self._nbbo[symbol] = self._c(f"nbbo_{symbol}_{self.day}", fetch)
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
            self._strikes[symbol] = self._c(
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
            tab = self._c(f"chain_{symbol}_{expiration}_{self.day}", fetch)
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

    # ---------------------------------------------------------------- shares-arm batch API
    def minute_bars_many(self, symbols, day, now=None):
        self.calls += 1          # shares_runner reads `calls` to decide the link is up
        return {s: self.minute_bars(s, day, now=now) for s in symbols}, {}

    def stock_quote_many(self, symbols):
        out = {}
        for s in symbols:
            q = self.stock_quote(s)
            if q is not None:
                out[s] = q
        return out, {}

    # ---------------------------------------------------------------- plumbing
    def wait_for_upstream(self, *a, **k):
        return True

    # Pass-throughs to the real feed's raw history calls, so a HistoryFeed can itself be
    # the "live" feed that catchup.py wraps -- which is how the catch-up is tested end to
    # end on a past session without a live market.
    def _fetch_minute_bars(self, symbol, day):
        self._no_future(day)
        return self.real._fetch_minute_bars(symbol, day)

    def _get_csv(self, path, **params):
        return self.real._get_csv(path, **params)

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


def _reset_options(lab) -> None:
    lab.sessions, lab.open_pos = {}, []
    lab.trade_counts, lab.dir_counts = {}, {}
    lab._seen_5m, lab._degraded_seen, lab._chain_cache = {}, {}, {}
    lab._decided = set()
    lab.levels = LevelsLoader(lab.store, tag="replay")
    lab.catchup = CatchUp(lab, tag="replay")


def _reset_shares(lab) -> None:
    lab.sessions, lab.open_pos = {}, []
    lab.counts, lab.dircnt = {}, {}
    lab._seen5, lab._degraded_seen = {}, {}
    lab._last_bar_at, lab._last_quote, lab._last_feed_ok = {}, {}, None
    lab._decided = set()
    lab.levels = LevelsLoader(lab.store, tag="replay")
    lab.catchup = CatchUp(lab, tag="replay")


def run(start: dt.date, end: dt.date, symbols=None, out=None, arm: str = "options") -> list[dict]:
    """Replay every session in [start, end] through the real runner of `arm`.

    Each session starts clean -- warmup at 09:07, the six lines at 09:31, every minute from
    09:30 to 16:00 ticked once, flatten, daily summary -- exactly the day a runner that
    never lost its feed would have had. The runner's own console output for each day is
    written to <out>/logs/<day>.log, so the replay leaves a log per session as live does.
    """
    if arm == "options":
        module, reset, flat = R, _reset_options, "shutdown"
        symbols = tuple(symbols or ("QQQ", "SPY"))
        out = out or OUT
        make = lambda: R.LiveLab(list(symbols), lab_dir=out)          # noqa: E731
    else:
        from . import shares_runner as SR
        module, reset, flat = SR, _reset_shares, "eod"
        symbols = tuple(symbols or SR.BROAD_UNIVERSE)
        out = out or OUT_SHARES
        make = lambda: SR.SharesLab(list(symbols), lab_dir=out)       # noqa: E731
    clock = SimClock()
    # store._now imports now_et INSIDE the function, so patching a module attribute would
    # miss it and stamp every record with tonight's wall clock. Patch _now itself.
    module.now_et, S._now = clock, clock
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    for f in ("signals.jsonl", "signals.jsonl.gz", "trades.jsonl", "events.jsonl",
              "outages.jsonl"):
        (out / f).unlink(missing_ok=True)      # a backfill is regenerated whole, never appended
    shutil.rmtree(out / S.SIGNALS_ARCHIVE, ignore_errors=True)
    clock.now = dt.datetime.combine(start, WARMUP_AT)
    lab = make()
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
            reset(lab)
            log_path = out / "logs" / f"{day}.log"
            with open(log_path, "w", encoding="utf-8") as fh, contextlib.redirect_stdout(fh):
                print(f"[replay {arm}] {day} -- config {lab.config_hash} -- history-backed, "
                      f"no outages; NOT the prospective record", flush=True)
                clock.now = dt.datetime.combine(day, WARMUP_AT)
                if not lab.warmup(day):
                    print(f"  {day}: warmup failed, skipped", flush=True)
                    continue
                t = dt.datetime.combine(day, module.RTH_OPEN) + TICK_OFFSET
                close_at = dt.datetime.combine(day, module.RTH_CLOSE)
                while t < close_at:
                    clock.now = t
                    if hasattr(lab, "_chain_cache"):
                        lab._chain_cache.clear()  # its 2s TTL is wall-clock; a replay minute is ~ms
                    try:
                        lab._tick(t, day)
                    except FeedOutage as exc:
                        lab.store.outage("tick", repr(exc))
                    t += STEP
                clock.now = close_at
                lab._flatten_all(close_at, reason=flat)
                lab.write_daily(day)
            d = json.loads((out / "daily" / f"{day}.json").read_text(encoding="utf-8"))
            summary.append(d)
            net = d.get("atm_net") if arm == "options" else d.get("net")
            print(f"  {arm} {day}: {d.get('trades_closed')} trades, net {net}", flush=True)
    finally:
        real.close()
    # Same layout as the live lab: one gzipped signals file per session (store.roll_signals).
    lab.store.roll_signals(end + dt.timedelta(days=1))
    (out / "BACKFILL.json").write_text(json.dumps({
        "provenance": "BACKFILL -- NOT PROSPECTIVE. Never counted by the dashboard, the "
                      "checkpoint diagnostic or any promotion bar.",
        "arm": arm, "config_hash": lab.config_hash, "start": start.isoformat(),
        "end": end.isoformat(), "symbols": list(symbols),
        "generated_by": "trade_analysis/live_lab/backfill.py",
        "differences_from_live": ["1-minute quote snapshots, live polls every 5s",
                                  "vendor's final (revised) bars, not first-print bars",
                                  "no feed outages, late starts, sleeps or blind windows",
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
    live_c = _corrected(lab, live, "position_id")
    days = sorted({r["entry_ts"][:10] for r in live} | {r["entry_ts"][:10] for r in back})
    rows = []
    for d in days:
        on = lambda xs: [r for r in xs if r["entry_ts"][:10] == d]           # noqa: E731
        # like with like: what the LIVE config ran that day (the six lines from 09-25)
        scope = [r for r in on(back) if d >= SIX_LIVE_FROM or r["setup_id"] not in new]
        recon = reconcile(on(live), scope)
        lv = [r["pnl_net"] for r in live if r["entry_ts"][:10] == d]
        b13 = [r["pnl_net"] for r in back if r["entry_ts"][:10] == d and r["setup_id"] not in new]
        b6 = [r["pnl_net"] for r in back if r["entry_ts"][:10] == d and r["setup_id"] == "Six_Lines"]
        b6n = [r["pnl_net"] for r in back
               if r["entry_ts"][:10] == d and r["setup_id"] == "Six_Lines_NoCap"]
        rows.append({"day": d, "live_n": len(lv), "live_net": round(sum(lv), 2),
                     "back13_n": len(b13), "back13_net": round(sum(b13), 2),
                     "six_n": len(b6), "six_net": round(sum(b6), 2),
                     "sixnc_n": len(b6n), "sixnc_net": round(sum(b6n), 2),
                     "newcfg_net": round(sum(b13) + sum(b6) + sum(b6n), 2),
                     "live_corrected_net": round(sum(r["pnl_net"] for r in on(live_c)), 2),
                     "replay_scope_n": len(scope),
                     "replay_scope_net": round(sum(r["pnl_net"] for r in scope), 2),
                     "recon": recon})
    by_setup = defaultdict(lambda: {"live_n": 0, "live_net": 0.0, "back_n": 0, "back_net": 0.0})
    for r in live:
        s = by_setup[r["setup_id"]]; s["live_n"] += 1; s["live_net"] += r["pnl_net"]
    for r in back:
        s = by_setup[r["setup_id"]]; s["back_n"] += 1; s["back_net"] += r["pnl_net"]
    return {"days": rows, "recon_total": _recon_total(rows),
            "by_setup": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv
                             for kk, vv in v.items()}
                         for k, v in sorted(by_setup.items())},
            "six_trades": [{k: r.get(k) for k in ("entry_ts", "symbol", "setup_id", "direction",
                                                  "strike", "entry_ask", "exit_bid",
                                                  "exit_reason", "pnl_net", "underlying_return")}
                           for r in back if r["setup_id"] in new]}


def _signal_key(r: dict) -> tuple:
    return (r["setup_id"], r["symbol"], r["direction"], r.get("entry_bar_ts"))


def reconcile(live: list[dict], replay: list[dict], shift_min: int = 3,
              detail: bool = False) -> dict:
    """Pair each live trade with the replay trade that took the SAME signal (setup, symbol,
    direction, entry bar), and sort every trade into exactly one bin:

      same         same signal, same exit rule, same exit minute
      moved        same signal, the exit differs (a bad exit, or vendor bar revisions)
      shifted      same setup/symbol/direction, entry bar <= shift_min minutes apart: the
                   same trade seen on a first-print bar live and a revised bar in history
      live_only    live took it and a clean session would not have (stale entry, restart)
      replay_only  a clean session took it and live never did (late start, sleep, block)

    live_net - replay_net == (same+moved+shifted: live - replay) + live_only - replay_only.
    With detail=True the unmatched trades themselves are returned too, for attribution.
    """
    pool = defaultdict(list)
    for r in replay:
        pool[_signal_key(r)].append(r)
    bins = {k: {"n": 0, "live": 0.0, "replay": 0.0}
            for k in ("same", "moved", "shifted", "live_only", "replay_only")}

    def put(k, lv=None, rp=None):
        b = bins[k]
        b["n"] += 1
        b["live"] += lv["pnl_net"] if lv else 0.0
        b["replay"] += rp["pnl_net"] if rp else 0.0

    unmatched, pairs = [], []
    for lv in live:
        cands = pool.get(_signal_key(lv))
        if not cands:
            unmatched.append(lv)
            continue
        rp = cands.pop(0)
        same = (lv.get("exit_reason") == rp.get("exit_reason")
                and (lv.get("exit_ts") or "")[:16] == (rp.get("exit_ts") or "")[:16])
        put("same" if same else "moved", lv, rp)
        pairs.append((lv, rp))
    rest = [r for rs in pool.values() for r in rs]
    live_only = []
    for lv in unmatched:
        k, t = _signal_key(lv)[:3], _parse_ts(lv["entry_bar_ts"])
        near = [r for r in rest if _signal_key(r)[:3] == k
                and abs((_parse_ts(r["entry_bar_ts"]) - t).total_seconds()) <= shift_min * 60]
        if near:
            rp = min(near, key=lambda r: abs((_parse_ts(r["entry_bar_ts"]) - t).total_seconds()))
            rest.remove(rp)
            put("shifted", lv, rp)
            pairs.append((lv, rp))
        else:
            live_only.append(lv)
            put("live_only", lv)
    for rp in rest:
        put("replay_only", rp=rp)
    out = {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()}
           for k, v in bins.items()}
    if detail:
        out["_live_only"], out["_replay_only"], out["_pairs"] = live_only, rest, pairs
    return out


def _recon_total(rows: list[dict]) -> dict:
    tot = {k: {"n": 0, "live": 0.0, "replay": 0.0}
           for k in ("same", "moved", "shifted", "live_only", "replay_only")}
    for row in rows:
        for k, b in row["recon"].items():
            for kk in b:
                tot[k][kk] += b[kk]
    return {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in tot.items()}


def _corrected(lab: Path, trades: list[dict], key: str) -> list[dict]:
    """Live trades with every gap-recovery correction applied (best available view)."""
    from .gap_recovery import apply_corrections
    p = lab / "trade_corrections.jsonl"
    if not p.exists():
        return trades
    cor = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return apply_corrections(trades, cor, key,
                             include=("recovered", "recovered_approximate", "estimated"))


def _live_scope(day: str) -> tuple[set, set]:
    """(symbols, setups-excluded) that the LIVE shares arm actually ran on `day`, so a replay
    of the current config can be compared like with like."""
    fz = json.loads((ROOT / "live_lab_data" / "shares" / "FREEZE.json").read_text(encoding="utf-8"))
    syms: set = set()
    for ch in sorted(fz.get("universe_changes", []), key=lambda c: c["date"]):
        if ch["date"] <= day:
            syms = set(ch["symbols"])
    excluded = set() if day >= SIX_LIVE_FROM else {"Six_Lines", "Six_Lines_NoCap"}
    return syms, excluded


def report_shares(start: dt.date, end: dt.date, out=OUT_SHARES) -> dict:
    lab = ROOT / "live_lab_data" / "shares"
    accepted = set(json.loads((lab / "FREEZE.json").read_text(encoding="utf-8"))
                   ["accepted_config_hashes"])
    rd = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()  # noqa: E731
                    if l.strip()] if p.exists() else []
    live = [r for r in rd(lab / "trades.jsonl") if r.get("config_hash") in accepted
            and start.isoformat() <= r["entry_ts"][:10] <= end.isoformat()]
    back = rd(out / "trades.jsonl")
    live_c = _corrected(lab, live, "signal_id")
    idx = {"QQQ", "SPY"}
    days = sorted({r["entry_ts"][:10] for r in live} | {r["entry_ts"][:10] for r in back})
    rows = []
    for d in days:
        syms, excl = _live_scope(d)
        lv = [r for r in live if r["entry_ts"][:10] == d]
        bk = [r for r in back if r["entry_ts"][:10] == d]
        sc = [r for r in bk if r["symbol"] in syms and r["setup_id"] not in excl]
        net = lambda xs: round(sum(r["pnl_net"] for r in xs), 2)                       # noqa: E731
        rows.append({"day": d, "live_n": len(lv), "live_net": net(lv),
                     "replay_scope_n": len(sc), "replay_scope_net": net(sc),
                     "replay_n": len(bk), "replay_net": net(bk),
                     "live_idx_net": net([r for r in lv if r["symbol"] in idx]),
                     "replay_idx_net": net([r for r in bk if r["symbol"] in idx]),
                     "replay_other_net": net([r for r in bk if r["symbol"] not in idx]),
                     "live_corrected_net": net([r for r in live_c if r["entry_ts"][:10] == d]),
                     "recon": reconcile(lv, sc)})
    by_setup = defaultdict(lambda: {"live_n": 0, "live_net": 0.0, "back_n": 0, "back_net": 0.0})
    for r in live:
        x = by_setup[r["setup_id"]]; x["live_n"] += 1; x["live_net"] += r["pnl_net"]
    for r in back:
        x = by_setup[r["setup_id"]]; x["back_n"] += 1; x["back_net"] += r["pnl_net"]
    return {"days": rows, "recon_total": _recon_total(rows),
            "by_setup": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv
                             for kk, vv in v.items()}
                         for k, v in sorted(by_setup.items())}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2026-08-28")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--arm", choices=["options", "shares", "both"], default="options")
    ap.add_argument("--report", action="store_true", help="report only, no replay")
    a = ap.parse_args(argv)
    s, e = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    for arm in (["options", "shares"] if a.arm == "both" else [a.arm]):
        if not a.report:
            run(s, e, arm=arm)
        rep = report(s, e) if arm == "options" else report_shares(s, e)
        out = OUT if arm == "options" else OUT_SHARES
        (out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"[{arm}] report written to {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
