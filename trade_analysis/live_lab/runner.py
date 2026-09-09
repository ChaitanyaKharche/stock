"""The live lab runner.

Loop, once per poll:

    1. pull today's 1m bars; admit only bars that are definitively CLOSED
    2. for each newly-admitted bar, per timeframe, evaluate every setup independently
    3. on a signal:  write DECISION -> fsync  ->  THEN fetch the option chain
    4. mark open positions; close any whose stop/target/clock/trailing fired
    5. checkpoint open positions atomically

Step 3 is the reason this file exists in this order. The decision is durable on disk
before any price that could have influenced it is requested.

Run:  python -m trade_analysis.live_lab.runner --symbols QQQ SPY
      python -m trade_analysis.live_lab.runner --replay 2026-08-27   (offline verification)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import signal as os_signal
import subprocess
import sys
import time
from pathlib import Path

from . import options as opt
from .clock import now_et, today_et
from .feed import FeedOutage, ThetaLiveFeed
from .lock import SingleInstance
from .positions import Position, open_positions_from_signal, position_from_dict
from .session import SessionState, build_warmup
from .setups import ALL_SETUPS, DEAD_SETUPS, SLOW_SETUPS
from .store import DEFAULT_LAB_DIR, LabStore

SPEC_VERSION = "live_lab_specification.md @ 2026-08-27"
RTH_OPEN, RTH_CLOSE = dt.time(9, 30), dt.time(16, 0)
EOD_FLAT = dt.time(15, 55)
# A quote may be neither too OLD nor dated in the FUTURE. The staleness test used to be
# one-sided (`age > STALE_QUOTE_SEC`), so a quote stamped ahead of the local clock had a
# NEGATIVE age and sailed straight through. Live that can only mean clock skew, and this
# lab is unusually exposed to it: the machine runs MST with no DST and every reading is
# converted to ET through a fixed offset in clock.py, so a wrong offset presents exactly
# as future-dated quotes. The failure mode is silent and total -- every fill priced off a
# quote from another time. Cheap to detect, so detect it.
FUTURE_QUOTE_SEC = 5.0
STALE_QUOTE_SEC = 5.0
EXIT_ALREADY_RUNNING = 4    # distinct from a crash: the supervisor must NOT retry this


# --------------------------------------------------------------------------- config


def build_config(symbols: list[str], settle_ms: int) -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      cwd=Path(__file__).resolve().parents[2],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                        # noqa: BLE001
        sha = "unknown"
    # HASHED: the things that define what a trade MEANS. Changing any of these forks a
    # setup into a new identity with a fresh history, which is the whole point.
    hashed = {
        "symbols": sorted(symbols),
        "settle_ms": settle_ms,
        "fee_per_contract_side": 0.0404,
        "primary_arm": "ATM",
        "arms": ["ATM", "ATM-1", "ATM+1"],
        "family_size": len(ALL_SETUPS),
        "slow_setups": sorted(SLOW_SETUPS),
        "dead_setups": sorted(DEAD_SETUPS),
        "setups": [s.describe() for s in ALL_SETUPS],
    }
    # NOT hashed: repo state. git_sha was originally inside the hash, which meant any
    # commit -- a README typo, a logging tweak -- forked every setup's history and reset
    # the trade clock. That is precisely the "reset when it looks bad" failure mode this
    # experiment forbids, so it is recorded as provenance and excluded from the identity.
    blob = json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()
    return {**hashed,
            "config_hash": hashlib.sha256(blob).hexdigest()[:16],
            "spec_version": SPEC_VERSION,
            "git_sha": sha}


# --------------------------------------------------------------------------- runner


class LiveLab:
    def __init__(self, symbols, lab_dir=DEFAULT_LAB_DIR, settle_ms=1500,
                 poll_open_sec=5, poll_idle_sec=15, contracts=1.0):
        self.symbols = symbols
        self.store = LabStore(lab_dir)
        self.config = build_config(symbols, settle_ms)
        self.config_hash = self.store.freeze_config(self.config)
        self.settle_ms = settle_ms
        self.poll_open = poll_open_sec
        self.poll_idle = poll_idle_sec
        self.contracts = contracts
        self.feed = ThetaLiveFeed(on_outage=self._on_outage)
        self.sessions: dict[str, SessionState] = {}
        self.open_pos: list[Position] = []
        self.trade_counts: dict[tuple[str, str], int] = {}
        self.dir_counts: dict[tuple[str, str, str], int] = {}
        self._stop = False
        self._chain_cache: dict[tuple[str, str], tuple[float, list]] = {}
        self._seen_5m: dict[str, set] = {}
        self._degraded_seen: dict[str, int] = {}
        os_signal.signal(os_signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, *_):
        print("\n[lab] shutting down; checkpointing open positions", flush=True)
        self._stop = True

    def _on_outage(self, path, detail):
        self.store.outage("feed", detail, path=path)

    # ------------------------------------------------------------------ startup

    def warmup(self, day: dt.date) -> bool:
        """Returns False if the feed is unreachable, so run() can exit cleanly.

        Warmup was previously outside the loop's try/except, so a terminal that was down
        at start produced a raw traceback instead of a diagnosable failure. Under the
        scheduler that is the difference between a legible log line and a mystery.
        """
        for sym in self.symbols:
            prior = self._prior_sessions(sym, day, 25)
            try:
                w = build_warmup(self.feed, sym, day, prior)
            except FeedOutage as exc:
                self.store.outage("warmup", repr(exc), symbol=sym)
                print(f"[lab] ABORT: cannot reach the Theta Terminal on 127.0.0.1:25503 "
                      f"while warming up {sym}.\n"
                      f"       Start it, or run via autostart.py --start-terminal.",
                      flush=True)
                return False
            self.sessions[sym] = SessionState(sym, day, w, w.get("prior_day"),
                                              settle_ms=self.settle_ms)
            self.store.event("warmup", symbol=sym, sessions_used=w["sessions_used"],
                             has_stretch=w["crabel_stretch"] is not None,
                             prior_day=w.get("prior_day"))
            print(f"[lab] warmup {sym}: {w['sessions_used']} prior sessions, "
                  f"stretch={w['crabel_stretch']}", flush=True)
        return True

    @staticmethod
    def _prior_sessions(sym, day, n):
        out, d = [], day - dt.timedelta(days=1)
        while len(out) < n:
            if d.weekday() < 5:
                out.append(d)
            d -= dt.timedelta(days=1)
        return sorted(out)

    def recover(self, day: dt.date | None = None) -> None:
        # Per-day caps are rebuilt from the DECISION record, NOT from recovered positions,
        # and BEFORE the early return. Two bugs lived in the old version:
        #   * counting one per recovered position triples the count in this arm, which
        #     opens three (ATM, ATM+-1) per signal;
        #   * restarting with nothing open returned early, so every cap reset to zero and
        #     setups that had already hit max_per_day could fire again.
        if day is not None:
            self.trade_counts, self.dir_counts = self.store.decision_counts(day)
            if self.trade_counts:
                print(f"[lab] restored per-day caps: "
                      f"{sum(self.trade_counts.values())} signals already taken today",
                      flush=True)
        raw, note = self.store.load_open_positions(expect_date=day)
        if note:
            print(f"[lab] recovery REJECTED: {note}", flush=True)
        if not raw:
            return
        self.open_pos = [position_from_dict(p) for p in raw]
        self.store.event("recovered", n_positions=len(self.open_pos),
                         restored_counts=sum(self.trade_counts.values()))
        print(f"[lab] recovered {len(self.open_pos)} open positions", flush=True)

    # ------------------------------------------------------------------ main loop

    def run(self, day: dt.date | None = None) -> None:
        day = day or today_et()
        self.store.event("start", config_hash=self.config_hash, symbols=self.symbols,
                         spec=SPEC_VERSION)
        print(f"[lab] config {self.config_hash} | {len(ALL_SETUPS)} setups | "
              f"symbols {self.symbols}", flush=True)
        if not self.warmup(day):
            self.store.event("aborted", reason="warmup_feed_unreachable")
            self.feed.close()
            return
        self.recover(day)

        while not self._stop:
            now = now_et()
            if now.date() != day:
                break
            if now.time() >= RTH_CLOSE:
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
                print(f"[lab] ERROR {exc!r}", flush=True)
            self.store.save_open_positions([p.to_trade() for p in self.open_pos],
                                       session_date=day)
            time.sleep(self.poll_open if self.open_pos else self.poll_idle)

        self._flatten_all(now_et(), reason="shutdown")
        self.store.save_open_positions([p.to_trade() for p in self.open_pos],
                                       session_date=day)
        self.write_daily(day)
        self.feed.close()
        print("[lab] stopped", flush=True)

    def _tick(self, now: dt.datetime, day: dt.date) -> None:
        for sym in self.symbols:
            sess = self.sessions[sym]
            admitted = sess.accept_bars(self.feed.minute_bars(sym, day, now=now), now)
            # SessionState records a discontinuity in the admitted 1m sequence, but
            # nothing used to read it -- a permanently missing bar would sit in memory
            # and be discarded at shutdown, invisible to any later audit even though
            # store.py advertises outages.jsonl as recording exactly this. Report each
            # one once, when it appears.
            n_deg = len(sess.degraded_bars)
            if n_deg > self._degraded_seen.get(sym, 0):
                for ts in sess.degraded_bars[self._degraded_seen.get(sym, 0):]:
                    self.store.outage("degraded_bar", f"gap in admitted 1m sequence "
                                      f"before {ts:%H:%M}", symbol=sym)
                    print(f"[lab] DEGRADED {sym}: 1m gap before {ts:%H:%M}", flush=True)
                self._degraded_seen[sym] = n_deg
            quote = self.feed.stock_quote(sym)
            if quote is not None:
                age = (now - quote["ts"]).total_seconds()
                if age > STALE_QUOTE_SEC:
                    self.store.outage("stale_quote", f"{sym} age={age:.1f}s")
                    quote = None
                elif age < -FUTURE_QUOTE_SEC:
                    self.store.outage("future_quote", f"{sym} quote is {-age:.1f}s AHEAD "
                                      f"of the local clock -- suspect timezone/clock skew",
                                      symbol=sym)
                    quote = None

            self._manage_open(sym, sess, quote, now)
            if not admitted:
                continue

            # 1m setups: one evaluation per newly-closed 1m bar, in order
            for bar in admitted:
                self._evaluate(sym, sess, bar["ts"], "1m", quote, now, day)

            # 5m setups: one evaluation per newly-completed 5m bucket, in order.
            # A bucket only exists once all five of its minutes are present, so a data
            # gap yields no 5m bar rather than a silently short one.
            seen = self._seen_5m.setdefault(sym, set())
            for b5 in sess.bars_5m:
                if b5["ts"] in seen:
                    continue
                seen.add(b5["ts"])
                self._evaluate(sym, sess, b5["ts"], "5m", quote, now, day)

    # ------------------------------------------------------------------ evaluation

    def _evaluate(self, sym, sess, bar_ts, tf, quote, now, day) -> None:
        ctx = sess.context(bar_ts, tf, quote)
        for setup in ALL_SETUPS:
            if setup.timeframe != tf:
                continue
            try:
                sig = setup.evaluate(ctx)
            except Exception as exc:                          # noqa: BLE001
                self.store.outage("setup_error", f"{setup.id}: {exc!r}", symbol=sym)
                continue
            if sig is None:
                continue

            key = (setup.id, sym)
            if self.trade_counts.get(key, 0) >= setup.max_per_day:
                self.store.write_skip(setup_id=setup.id, config_hash=self.config_hash,
                                      symbol=sym, bar_ts=bar_ts, reason="max_per_day")
                continue
            if setup.max_per_direction is not None:
                dk = (setup.id, sym, sig.direction)
                if self.dir_counts.get(dk, 0) >= setup.max_per_direction:
                    self.store.write_skip(setup_id=setup.id, config_hash=self.config_hash,
                                          symbol=sym, bar_ts=bar_ts,
                                          reason="max_per_direction")
                    continue
            if any(p.setup_id == setup.id and p.symbol == sym for p in self.open_pos):
                self.store.write_skip(setup_id=setup.id, config_hash=self.config_hash,
                                      symbol=sym, bar_ts=bar_ts, reason="already_open")
                continue
            if now.time() >= EOD_FLAT:
                self.store.write_skip(setup_id=setup.id, config_hash=self.config_hash,
                                      symbol=sym, bar_ts=bar_ts, reason="past_eod_flat")
                continue

            self._open(setup, sig, sym, ctx, quote, now, bar_ts, day)

    def _open(self, setup, sig, sym, ctx, quote, now, bar_ts, day) -> None:
        # ---- THE ORDERING GUARANTEE -------------------------------------
        state = dict(sig.state)
        state.update({"price_at_bar": ctx.price, "vwap": ctx.vwap,
                      "bars_1m": len(ctx.bars_1m), "bars_5m": len(ctx.bars_5m),
                      "dow": bar_ts.strftime("%a"), "tf": ctx.tf,
                      "slow_setup": setup.id in SLOW_SETUPS,
                      "dead_setup": setup.id in DEAD_SETUPS})
        signal_id = self.store.write_decision(
            setup_id=setup.id, config_hash=self.config_hash, symbol=sym,
            direction=sig.direction, bar_ts=bar_ts, state=state)
        # ---- only now may a price be requested ---------------------------

        if quote is None:
            self.store.write_fill(signal_id, status="SKIPPED", skip_reason="no_underlying_quote")
            return
        expiration = self.feed.zero_dte(sym, day)
        if expiration is None:
            self.store.write_fill(signal_id, status="SKIPPED", skip_reason="no_0dte")
            return
        try:
            chain = self._chain(sym, expiration)
            arms = opt.select_arms(chain, quote["mid"],
                                   "call" if sig.direction == "long" else "put")
        except opt.UnusableQuote as exc:
            self.store.write_fill(signal_id, status="SKIPPED",
                                  skip_reason=f"unusable_quote: {exc}")
            return
        except FeedOutage as exc:
            self.store.write_fill(signal_id, status="SKIPPED", skip_reason=f"feed: {exc!r}")
            return

        for k in ("ATM", "ATM-1", "ATM+1"):
            if arms.get(k):
                arms[k] = opt.enrich(arms[k], quote["mid"], expiration, now)
        try:
            oi = self.feed.open_interest(sym, expiration)
        except FeedOutage:
            oi = {}

        pos = open_positions_from_signal(
            signal_id=signal_id, config_hash=self.config_hash, setup=setup, sig=sig,
            symbol=sym, arms=arms, expiration=expiration, spot_mid=quote["mid"],
            now=now, bar_ts=bar_ts, oi=oi, contracts=self.contracts)
        self.open_pos.extend(pos)
        self.trade_counts[(setup.id, sym)] = self.trade_counts.get((setup.id, sym), 0) + 1
        if setup.max_per_direction is not None:
            dk = (setup.id, sym, sig.direction)
            self.dir_counts[dk] = self.dir_counts.get(dk, 0) + 1

        self.store.write_fill(signal_id, status="OPEN", n_arms=len(pos),
                              atm_strike=arms["atm_strike"],
                              strike_distance_pct=arms["strike_distance_pct"],
                              underlying_mid=quote["mid"], expiration=expiration,
                              arms={k: {kk: arms[k][kk] for kk in
                                        ("strike", "bid", "ask", "spread_pct_of_mid",
                                         "iv_derived", "delta_derived")}
                                    for k in ("ATM", "ATM-1", "ATM+1") if arms.get(k)})
        print(f"[lab] {bar_ts:%H:%M} {setup.id:<26} {sym} {sig.direction:<5} "
              f"K={arms['atm_strike']} ask={arms['ATM']['ask']:.2f}", flush=True)

    def _chain(self, sym, expiration):
        """Cache the chain for one poll interval; a signal always forces a fresh pull."""
        key = (sym, expiration.isoformat())
        now = time.time()
        hit = self._chain_cache.get(key)
        if hit and now - hit[0] < 2.0:
            return hit[1]
        chain = self.feed.chain_quotes(sym, expiration)
        self._chain_cache[key] = (now, chain)
        return chain

    # ------------------------------------------------------------------ management

    def _manage_open(self, sym, sess, quote, now) -> None:
        mine = [p for p in self.open_pos if p.symbol == sym]
        if not mine or quote is None:
            return
        bars = sess.bars_1m
        if not bars:
            return
        last_bar = bars[-1]

        chains: dict[str, list] = {}
        for p in mine:
            try:
                chain = chains.setdefault(
                    p.expiration,
                    self._chain(sym, dt.date.fromisoformat(p.expiration)))
            except FeedOutage:
                continue
            q = next((c for c in chain
                      if abs(c["strike"] - p.strike) < 1e-9 and c["right"] == p.right), None)
            bid = q["bid"] if q else None
            p.mark(bid, quote["mid"], now)
            p.bars_held = sum(1 for b in bars
                              if b["ts"] > dt.datetime.fromisoformat(p.entry_bar_ts))

            reason, level = None, None
            hit = p.check_underlying_exit(last_bar, now)
            if hit:
                reason, level = hit
            if reason is None:
                reason = p.check_clock_exit(now)
            if reason is None and p.trailing:
                setup = next((s for s in ALL_SETUPS if s.id == p.setup_id), None)
                if setup is not None:
                    ctx = sess.context(last_bar["ts"], setup.timeframe, quote)
                    try:
                        reason = setup.manage({"direction": p.direction, "state": p.state}, ctx)
                    except Exception as exc:                  # noqa: BLE001
                        self.store.outage("manage_error", f"{p.setup_id}: {exc!r}")
            if reason is None:
                continue
            if bid is None:
                self.store.outage("exit_no_quote",
                                  f"{p.setup_id} {p.arm} wanted {reason}, no bid")
                continue
            trade = p.close(bid=bid, underlying=quote["mid"], ts=now,
                            reason=reason, trigger_level=level)
            self.store.write_trade(trade)
            self.open_pos.remove(p)
            print(f"[lab] {now:%H:%M} CLOSE {p.setup_id:<26} {p.arm:<5} {reason:<8} "
                  f"net={trade['pnl_net']:+.2f}", flush=True)

    def _flatten_all(self, now, reason="eod") -> None:
        for p in list(self.open_pos):
            quote = None
            try:
                quote = self.feed.stock_quote(p.symbol)
                chain = self._chain(p.symbol, dt.date.fromisoformat(p.expiration))
                q = next((c for c in chain if abs(c["strike"] - p.strike) < 1e-9
                          and c["right"] == p.right), None)
                bid = q["bid"] if q else p.last_bid
            except FeedOutage:
                bid = p.last_bid
            und = quote["mid"] if quote else (p.last_underlying or p.entry_underlying)
            trade = p.close(bid=bid, underlying=und, ts=now, reason=reason)
            self.store.write_trade(trade)
            self.open_pos.remove(p)

    # ------------------------------------------------------------------ summary

    def write_daily(self, day: dt.date) -> None:
        trades = [t for t in self.store.read("trades.jsonl")
                  if str(t.get("entry_ts", "")).startswith(day.isoformat())]
        sigs = [s for s in self.store.read("signals.jsonl")
                if str(s.get("ts", "")).startswith(day.isoformat())]
        summary = {
            "date": day.isoformat(),
            "config_hash": self.config_hash,
            "symbols": self.symbols,
            "signals_decided": sum(1 for s in sigs if s.get("phase") == "DECISION"),
            "signals_skipped": sum(1 for s in sigs if s.get("phase") == "SKIP"),
            "fills_skipped": sum(1 for s in sigs
                                 if s.get("phase") == "FILL" and s.get("status") == "SKIPPED"),
            "trades_closed": len(trades),
            "degraded_bars": {k: len(v.degraded_bars)
                              for k, v in self.sessions.items()
                              if v.degraded_bars},
            "atm_net": round(sum(t["pnl_net"] for t in trades
                                 if t["arm"] == "ATM" and t.get("pnl_net") is not None), 2),
            "feed": self.feed.stats(),
        }
        self.store.write_daily(day, summary)
        print(f"[lab] daily: {json.dumps(summary)}", flush=True)


# --------------------------------------------------------------------------- cli


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live paper-trading laboratory (no orders).")
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    ap.add_argument("--lab-dir", default=str(DEFAULT_LAB_DIR))
    ap.add_argument("--settle-ms", type=int, default=1500)
    ap.add_argument("--contracts", type=float, default=1.0)
    ap.add_argument("--print-config", action="store_true",
                    help="print the frozen config hash and exit")
    args = ap.parse_args(argv)

    if args.print_config:
        lab = LiveLab(args.symbols, lab_dir=args.lab_dir, settle_ms=args.settle_ms,
                      contracts=args.contracts)
        print(json.dumps(lab.config, indent=2))
        return 0

    # Two runners appending to the same jsonl files would interleave sequence numbers and
    # double-count every signal. Refuse rather than corrupt, and use a distinct exit code
    # so the supervisor in autostart.py knows this is not a crash to retry.
    guard = SingleInstance("runner", args.lab_dir)
    if not guard.acquire():
        print(f"[lab] another runner already holds the lock ({guard.holder()}); exiting",
              flush=True)
        return EXIT_ALREADY_RUNNING
    try:
        lab = LiveLab(args.symbols, lab_dir=args.lab_dir, settle_ms=args.settle_ms,
                      contracts=args.contracts)
        lab.run()
    finally:
        guard.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
