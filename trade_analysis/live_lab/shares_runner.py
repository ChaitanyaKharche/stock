"""Live paper-trading laboratory, SHARES arm. No options, no orders, no real money.

WHY THIS EXISTS
    The options lab forward-tests 13 setups on ATM/ATM+-1 0DTE contracts. Every one of them
    was measured at or below zero there, and the historical work says that is structural: an
    ATM 0DTE needs roughly +5 bp of underlying move to clear spread and theta, and none of
    these signals produce that.

    The one exception runs the other way. IntradayMomentumBoundary is the only item in this
    project that has ever cleared a multiplicity correction on a large sample AND survived
    honest execution timing -- but only on SHARES: +$3.34/trade over 2,905 trades on QQQ
    (Holm 0.0043), and again the only 1 of 13 to clear on SPY (Holm 0.0087). On options it
    dies. QQQ's spread is ~0.5 bp round trip against an option cost floor ~450x higher.

    So the strategy with the best evidence in the entire project was not being forward
    tested at all. This closes that gap.

RELATIONSHIP TO THE FROZEN OPTIONS LAB
    Completely separate. Its own store (live_lab_data/shares/), its own config hash, its own
    freeze and checkpoints. It cannot add a trade to, or remove one from, any options setup's
    count. The options freeze `1f7247d7839d9950` is untouched -- this file imports from the
    proven modules and modifies none of them.

ALL THIRTEEN SETUPS RUN, NOT JUST IMB
    Deliberate. IMB won a 13-way sweep; forward-testing only the winner would be selection
    on the outcome and its p-value would mean nothing. Running the same family forward keeps
    Holm m=13 valid and lets IMB's result be read against its own peers.

EXECUTION -- mirrors sharewf.replay_day exactly, because otherwise the forward record is
not comparable to the 10.6-year backtest it exists to extend:
    * one open position per setup at a time
    * manage open positions BEFORE looking for new entries
    * exit precedence: stop -> target -> time -> bars -> trailing -> eod
    * stop/target detected on a CLOSED bar's high/low, filled at the live NBBO
    * long  buys the ASK and sells the BID; short sells the BID and buys the ASK
    * $10,000 notional per trade, max_per_day and max_per_direction caps honoured
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import datetime as dt
import hashlib
import json
import signal as os_signal
import sys
import time
from pathlib import Path

from .clock import now_et, today_et
from .feed import FeedOutage, ThetaLiveFeed
from .lock import SingleInstance
from .session import SessionState, build_warmup
from .setups import ALL_SETUPS, DEAD_SETUPS, SLOW_SETUPS
from .store import DEFAULT_LAB_DIR, LabStore

PLACES_ORDERS = False               # this file has no broker path and never will
SPEC_VERSION = "shares_lab_preregistration.md @ 2026-08-31"
RTH_OPEN, RTH_CLOSE = dt.time(9, 30), dt.time(16, 0)
EOD_FLAT = dt.time(15, 55)
NOTIONAL = 10_000.0
STALE_QUOTE_SEC = 5.0
EXIT_ALREADY_RUNNING = 4
# Early-close detection. On a half day the tape stops at 13:00 but EOD_FLAT is 15:55, so a
# hardcoded flatten would hold positions through three hours of dead air and then stamp the
# exits 15:55 at 13:00 prices. If no new bar has arrived for this long and we are past
# EARLY_CLOSE_AFTER, the session has ended and we flatten against the last real quote.
NO_BAR_MINUTES = 12
EARLY_CLOSE_AFTER = dt.time(12, 30)
FEED_HEALTHY_SEC = 120   # the feed must have answered this recently for a
                         # bar drought to mean 'market closed' rather than
                         # 'this machine cannot reach the data'


@dc.dataclass
class SharePos:
    setup_id: str
    symbol: str
    direction: str
    entry_ts: str
    entry_bar_ts: str
    entry_px: float                 # ask if long, bid if short
    shares: float
    spread_bp: float
    stop: float | None
    target: float | None
    time_exit_min: int | None
    bar_exit: int | None
    trailing: str | None
    state: dict
    timeframe: str
    entry_quote_ts: str = ""
    signal_id: str = ""

    def to_row(self) -> dict:
        return dc.asdict(self)


def build_config(symbols):
    hashed = {
        "arm": "SHARES",
        "symbols": sorted(symbols),
        "notional_per_trade": NOTIONAL,
        "fill_rule": "signal from closed bar -> fill at live NBBO; long buys ask/sells bid",
        "family_size": len(ALL_SETUPS),
        "slow_setups": sorted(SLOW_SETUPS),
        "dead_setups": sorted(DEAD_SETUPS),
        "setups": [s.describe() for s in ALL_SETUPS],
        "eod_flat": EOD_FLAT.isoformat(),
    }
    blob = json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()
    return {**hashed, "config_hash": hashlib.sha256(blob).hexdigest()[:16],
            "spec_version": SPEC_VERSION}


class SharesLab:
    def __init__(self, symbols, lab_dir, poll_open_sec=5, poll_idle_sec=15):
        self.symbols = symbols
        self.store = LabStore(lab_dir)
        self.config = build_config(symbols)
        self.config_hash = self.store.freeze_config(self.config)
        self.poll_open = poll_open_sec
        self.poll_idle = poll_idle_sec
        self.feed = ThetaLiveFeed(on_outage=self._on_outage)
        self.sessions: dict[str, SessionState] = {}
        self.open_pos: list[SharePos] = []
        self.counts: dict[tuple[str, str], int] = {}
        self.dircnt: dict[tuple[str, str, str], int] = {}
        self._seen5: dict[str, set] = {}
        self._degraded_seen: dict[str, int] = {}
        self._last_bar_at: dict[str, dt.datetime] = {}
        self._last_quote: dict[str, dict] = {}   # last GOOD NBBO per symbol
        self._last_feed_ok: dt.datetime | None = None  # last successful fetch
        self._stop = False
        self._setups = {s.id: s for s in ALL_SETUPS}
        os_signal.signal(os_signal.SIGINT, self._sigint)

    def _sigint(self, *_):
        print("\n[shares] shutting down; checkpointing open positions", flush=True)
        self._stop = True

    def _on_outage(self, path, detail):
        self.store.outage("feed", detail, path=path)

    # -------------------------------------------------------------------- startup

    def warmup(self, day) -> bool:
        for sym in self.symbols:
            prior, d = [], day - dt.timedelta(days=1)
            while len(prior) < 25:
                if d.weekday() < 5:
                    prior.append(d)
                d -= dt.timedelta(days=1)
            try:
                w = build_warmup(self.feed, sym, day, sorted(prior))
            except FeedOutage as exc:
                self.store.outage("warmup", repr(exc), symbol=sym)
                print(f"[shares] ABORT: feed unreachable warming up {sym}", flush=True)
                return False
            self.sessions[sym] = SessionState(sym, day, w, w.get("prior_day"))
            self.store.event("warmup", symbol=sym, sessions_used=w["sessions_used"],
                             has_stretch=w["crabel_stretch"] is not None)
            print(f"[shares] warmup {sym}: {w['sessions_used']} prior sessions, "
                  f"stretch={w['crabel_stretch']}", flush=True)
        return True

    def recover(self, day) -> None:
        # Per-day caps come from the DECISION record, not from recovered positions, and are
        # restored BEFORE the early return. Counting only what is still OPEN forgets every
        # signal that already closed, so a restart resets max_per_day to zero and the setup
        # fires again -- exactly what happened on 2026-09-02: a 15:00 restart re-entered
        # ORB_5min and ORB_15min (both cap 1) that had already traded that morning.
        if day is not None:
            self.counts, self.dircnt = self.store.decision_counts(day)
            if self.counts:
                print(f"[shares] restored per-day caps: "
                      f"{sum(self.counts.values())} signals already taken today", flush=True)
        raw, note = self.store.load_open_positions(expect_date=day)
        if note:
            print(f"[shares] recovery REJECTED: {note}", flush=True)
        if not raw:
            return
        fields = set(SharePos.__dataclass_fields__)
        self.open_pos = [SharePos(**{k: v for k, v in r.items() if k in fields})
                         for r in raw]
        self.store.event("recovered", n_positions=len(self.open_pos),
                         restored_counts=sum(self.counts.values()))
        print(f"[shares] recovered {len(self.open_pos)} open positions", flush=True)

    # ----------------------------------------------------------------- main loop

    def run(self, day=None) -> None:
        day = day or today_et()
        self.store.event("start", config_hash=self.config_hash, symbols=self.symbols,
                         spec=SPEC_VERSION, notional=NOTIONAL)
        print(f"[shares] config {self.config_hash} | {len(ALL_SETUPS)} setups | "
              f"${NOTIONAL:,.0f} notional | symbols {self.symbols}", flush=True)
        if not self.warmup(day):
            self.store.event("aborted", reason="warmup_feed_unreachable")
            self.feed.close()
            return
        self.recover(day)

        while not self._stop:
            now = now_et()
            if now.date() != day or now.time() >= RTH_CLOSE:
                break
            if now.time() < RTH_OPEN:
                time.sleep(min(30, self.poll_idle))
                continue
            try:
                self._tick(now, day)
            except FeedOutage as exc:
                self.store.outage("tick", repr(exc))
            except Exception as exc:                          # noqa: BLE001
                self.store.outage("unhandled", repr(exc))
                print(f"[shares] ERROR {exc!r}", flush=True)
            if self._early_close(now):
                print("[shares] tape has stopped well before 15:55 -- treating as an "
                      "early close and flattening now", flush=True)
                self.store.event("early_close", at=now.isoformat())
                break
            self.store.save_open_positions([p.to_row() for p in self.open_pos],
                                           session_date=day)
            time.sleep(self.poll_open if self.open_pos else self.poll_idle)

        self._flatten_all(now_et(), "eod")
        self.store.save_open_positions([p.to_row() for p in self.open_pos],
                                       session_date=day)
        self.write_daily(day)
        self.feed.close()
        print("[shares] stopped", flush=True)

    def _early_close(self, now) -> bool:
        """True when the TAPE has stopped -- not when the NETWORK has.

        The first version only asked "have bars stopped arriving?", which cannot tell a
        half-day close from a dead wifi link or a dead Theta Terminal. On 2026-09-04 the
        machine lost connectivity near the close and this fired at 15:58; the damage was
        small only because the session was ending anyway. Had the link dropped at 13:00 it
        would have flattened every position and abandoned three hours of the session.

        The discriminator is whether the FEED is answering. A closed market means healthy
        requests that return no NEW bars. A dead link means the requests themselves fail,
        and the correct response to that is to wait, not to liquidate.
        """
        if now.time() < EARLY_CLOSE_AFTER or not self._last_bar_at:
            return False
        if (now - max(self._last_bar_at.values())).total_seconds() <= NO_BAR_MINUTES * 60:
            return False        # bars are still arriving; the session is running
        if self._last_feed_ok is None:
            return False
        # the feed must have answered RECENTLY; otherwise this is connectivity, not a close
        return (now - self._last_feed_ok).total_seconds() <= FEED_HEALTHY_SEC

    def _tick(self, now, day) -> None:
        for sym in self.symbols:
            sess = self.sessions[sym]
            bars = self.feed.minute_bars(sym, day)   # raises FeedOutage if unreachable
            self._last_feed_ok = now                 # the feed ANSWERED, whatever it said
            admitted = sess.accept_bars(bars, now)
            n_deg = len(sess.degraded_bars)
            if n_deg > self._degraded_seen.get(sym, 0):
                for ts in sess.degraded_bars[self._degraded_seen.get(sym, 0):]:
                    self.store.outage("degraded_bar", f"1m gap before {ts:%H:%M}",
                                      symbol=sym)
                self._degraded_seen[sym] = n_deg
            if not admitted:
                continue
            self._last_bar_at[sym] = now

            quote = self.feed.stock_quote(sym)
            if quote and (now - quote["ts"]).total_seconds() > STALE_QUOTE_SEC:
                self.store.outage("stale_quote", f"{sym} age="
                                  f"{(now - quote['ts']).total_seconds():.1f}s")
                quote = None
            if quote is None:
                continue          # never fill against a stale or missing NBBO
            self._last_quote[sym] = quote

            seen5 = self._seen5.setdefault(sym, set())
            for bar in admitted:
                ctx1 = sess.context(bar["ts"], "1m", None)
                ctx5 = None
                if sess.bars_5m and sess.bars_5m[-1]["ts"] not in seen5:
                    seen5.add(sess.bars_5m[-1]["ts"])
                    ctx5 = sess.context(sess.bars_5m[-1]["ts"], "5m", None)
                # production ordering: manage first, then enter
                self._manage(sym, sess, bar, ctx1, ctx5, quote, now)
                self._enter(sym, sess, bar, ctx1, ctx5, quote, now)

    # ------------------------------------------------------------------ managing

    def _bars_since(self, sess, entry_bar_ts) -> int:
        t = dt.datetime.fromisoformat(entry_bar_ts)
        return sum(1 for b in sess.bars_1m if b["ts"] > t)

    def _manage(self, sym, sess, bar, ctx1, ctx5, quote, now) -> None:
        for p in [x for x in self.open_pos if x.symbol == sym]:
            if bar["ts"] <= dt.datetime.fromisoformat(p.entry_bar_ts):
                continue
            reason = None
            if p.direction == "long":
                if p.stop is not None and bar["low"] <= p.stop:
                    reason = "stop"
                elif p.target is not None and bar["high"] >= p.target:
                    reason = "target"
            else:
                if p.stop is not None and bar["high"] >= p.stop:
                    reason = "stop"
                elif p.target is not None and bar["low"] <= p.target:
                    reason = "target"
            held = self._bars_since(sess, p.entry_bar_ts)
            if reason is None and p.time_exit_min is not None and held >= p.time_exit_min:
                reason = "time"
            if reason is None and p.bar_exit is not None and \
                    held >= p.bar_exit * (5 if p.timeframe == "5m" else 1):
                reason = "bars"
            if reason is None and p.trailing:
                c = ctx5 if p.timeframe == "5m" else ctx1
                setup = self._setups.get(p.setup_id)
                if c is not None and setup is not None:
                    try:
                        if setup.manage({"direction": p.direction, "state": p.state}, c):
                            reason = "trail"
                    except Exception:                        # noqa: BLE001
                        pass
            if reason is None and bar["ts"].time() >= EOD_FLAT:
                reason = "eod"
            if reason:
                self._close(p, quote, now, reason)

    def _close(self, p, quote, now, reason) -> None:
        px = quote["bid"] if p.direction == "long" else quote["ask"]
        pnl = (p.shares * (px - p.entry_px) if p.direction == "long"
               else p.shares * (p.entry_px - px))
        self.store.write_trade({
            "arm": "SHARES", "config_hash": self.config_hash,
            "setup_id": p.setup_id, "symbol": p.symbol, "direction": p.direction,
            "entry_ts": p.entry_ts, "exit_ts": now.isoformat(),
            "entry_bar_ts": p.entry_bar_ts, "entry_px": p.entry_px, "exit_px": px,
            "shares": p.shares, "notional": NOTIONAL, "exit_reason": reason,
            "pnl_net": round(pnl, 4), "return_pct": (pnl / NOTIONAL),
            "entry_spread_bp": p.spread_bp, "signal_id": p.signal_id,
            "hold_minutes": round((now - dt.datetime.fromisoformat(p.entry_ts))
                                  .total_seconds() / 60.0, 2),
        })
        self.open_pos.remove(p)
        print(f"[shares] {now:%H:%M} CLOSE {p.setup_id:<26}{p.symbol} {p.direction:<5}"
              f" {reason:<7} net={pnl:+.2f}", flush=True)

    # ------------------------------------------------------------------ entering

    def _enter(self, sym, sess, bar, ctx1, ctx5, quote, now) -> None:
        openids = {p.setup_id for p in self.open_pos if p.symbol == sym}
        for tf, ctx in (("1m", ctx1), ("5m", ctx5)):
            if ctx is None:
                continue
            for setup in ALL_SETUPS:
                if setup.timeframe != tf or setup.id in openids:
                    continue
                if self.counts.get((setup.id, sym), 0) >= setup.max_per_day:
                    self.store.write_skip(setup_id=setup.id, config_hash=self.config_hash,
                                          symbol=sym, bar_ts=bar["ts"].isoformat(),
                                          reason="max_per_day")
                    continue
                try:
                    sig = setup.evaluate(ctx)
                except Exception:                            # noqa: BLE001
                    continue
                if sig is None:
                    continue
                if setup.max_per_direction is not None and \
                        self.dircnt.get((setup.id, sym, sig.direction), 0) >= \
                        setup.max_per_direction:
                    self.store.write_skip(setup_id=setup.id, config_hash=self.config_hash,
                                          symbol=sym, bar_ts=bar["ts"].isoformat(),
                                          reason="max_per_direction",
                                          direction=sig.direction)
                    continue
                # THE ORDERING GUARANTEE. The setup was handed a Context built only from
                # closed bars -- `context(..., quote=None)` -- so no price this runner holds
                # can have influenced the signal. The DECISION row is fsynced here anyway,
                # before the fill price is read, so the chain is auditable the same way the
                # options arm's is.
                sid = self.store.write_decision(
                    setup_id=setup.id, config_hash=self.config_hash, symbol=sym,
                    direction=sig.direction, bar_ts=bar["ts"].isoformat(),
                    state=sig.state)
                px = quote["ask"] if sig.direction == "long" else quote["bid"]
                if px <= 0:
                    self.store.write_fill(sid, status="SKIPPED", reason="bad_quote")
                    continue
                p = SharePos(
                    setup_id=setup.id, symbol=sym, direction=sig.direction,
                    entry_ts=now.isoformat(), entry_bar_ts=bar["ts"].isoformat(),
                    entry_px=px, shares=NOTIONAL / px,
                    spread_bp=10000.0 * (quote["ask"] - quote["bid"]) / px,
                    stop=sig.stop, target=sig.target, time_exit_min=sig.time_exit_min,
                    bar_exit=sig.bar_exit, trailing=sig.trailing, state=sig.state,
                    timeframe=setup.timeframe, entry_quote_ts=quote["ts"].isoformat(),
                    signal_id=sid)
                self.store.write_fill(sid, status="FILLED", symbol=sym, price=px,
                                      shares=p.shares, spread_bp=p.spread_bp)
                self.open_pos.append(p)
                openids.add(setup.id)
                self.counts[(setup.id, sym)] = self.counts.get((setup.id, sym), 0) + 1
                self.dircnt[(setup.id, sym, sig.direction)] = \
                    self.dircnt.get((setup.id, sym, sig.direction), 0) + 1
                print(f"[shares] {now:%H:%M} {setup.id:<26}{sym} {sig.direction:<5} "
                      f"{px:.2f} x{p.shares:.1f}sh spread={p.spread_bp:.2f}bp"
                      f"{'' if sid is None else ''}", flush=True)

    def _flatten_all(self, now, reason="eod") -> None:
        """Close everything. A position must NEVER be left unrecorded.

        The options arm already falls back to its last known mark here. This one did not,
        and on 2026-09-04 the feed was down at the close (60 outages, 57 failed ticks), so
        three positions were abandoned and silently vanished from the record -- the exact
        systematic hole the EOD reconstruction had just repaired.

        A stale mark is a compromise; an unrecorded position is a hole. Take the compromise
        and flag it, so the row is identifiable rather than absent.
        """
        for p in list(self.open_pos):
            quote = None
            try:
                quote = self.feed.stock_quote(p.symbol)
            except FeedOutage:
                quote = None
            r = reason
            if quote is None:
                quote = self._last_quote.get(p.symbol)
                if quote is not None:
                    r = reason + "_stale_mark"
                    self.store.outage("flatten_stale_mark",
                                      f"{p.symbol} {p.setup_id} closed on the last known "
                                      f"NBBO from {quote.get('ts')}")
            if quote is None:
                self.store.outage("flatten_no_quote",
                                  f"{p.symbol} {p.setup_id} LEFT OPEN -- no NBBO and no "
                                  f"prior mark this session")
                continue
            self._close(p, quote, now, r)

    def write_daily(self, day) -> None:
        trades = [t for t in self.store.read("trades.jsonl")
                  if str(t.get("entry_ts", "")).startswith(day.isoformat())]
        self.store.write_daily(day, {
            "date": day.isoformat(), "arm": "SHARES",
            "config_hash": self.config_hash, "symbols": self.symbols,
            "notional_per_trade": NOTIONAL,
            "trades_closed": len(trades),
            "net": round(sum(t.get("pnl_net", 0.0) for t in trades), 2),
            "still_open": len(self.open_pos),
            "degraded_bars": {k: len(v.degraded_bars) for k, v in self.sessions.items()
                              if v.degraded_bars},
            "feed": self.feed.stats(),
        })
        print(f"[shares] daily: {len(trades)} trades, "
              f"net ${sum(t.get('pnl_net', 0.0) for t in trades):+,.2f}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live SHARES paper lab (no orders).")
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    ap.add_argument("--lab-dir", default=str(Path(DEFAULT_LAB_DIR) / "shares"))
    ap.add_argument("--print-config", action="store_true")
    args = ap.parse_args(argv)

    if args.print_config:
        print(json.dumps(build_config(args.symbols), indent=2))
        return 0
    guard = SingleInstance("shares_runner", args.lab_dir)
    if not guard.acquire():
        print(f"[shares] another shares runner holds the lock ({guard.holder()})",
              flush=True)
        return EXIT_ALREADY_RUNNING
    try:
        SharesLab(args.symbols, args.lab_dir).run()
    finally:
        guard.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
