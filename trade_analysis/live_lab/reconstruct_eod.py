"""Reconstruct the end-of-day exits the supervisor destroyed on 2026-09-02 and 2026-09-03.

WHAT HAPPENED
    autostart.supervise() terminated both arms at 15:55:02 ET -- the exact minute they were
    due to flatten. Every position still open at the close was therefore never written to
    trades.jsonl, and no daily summary was produced.

    The loss is NOT random. It removes exactly one exit type: `eod`. Trades that had already
    hit a stop, target, time or trailing exit were recorded normally; the ones still alive at
    the close vanished. Three consecutive sessions show zero `eod` exits where the first two
    sessions of the freeze showed them normally. A hole that deletes one exit category biases
    the sample far more than a larger random one would.

WHAT THIS DOES
    Applies exactly the rule `_flatten_all` would have applied, with the same data source:
    close every surviving position at the 15:55 ET quote, `eod` reason, real historical NBBO,
    same fee. Nothing is invented and no rule is changed.

    Every reconstructed trade is written with:
        exit_reason      "eod_reconstructed"
        reconstructed    true
    so it can never be mistaken for a live fill, and can be excluded from any analysis with
    a one-line filter.

WHAT IT CANNOT DO
    2026-09-01's snapshot was overwritten by the following session before the archive
    existed. Those positions are gone permanently and that session keeps its hole.

    python -m trade_analysis.live_lab.reconstruct_eod --dry-run
    python -m trade_analysis.live_lab.reconstruct_eod --commit
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from .feed import FeedOutage, ThetaLiveFeed
from .store import DEFAULT_LAB_DIR, LabStore

FEE = 0.0404
EOD = "15:55"
# SharePos carries no config_hash; the live path stamps it from the runner. Reconstructed
# rows need it too or the dashboard's accepted-hash filter silently drops them.
SHARES_HASH = "53229d6f1f24df10"
SOURCES = [
    # (label, lab dir, snapshot path, notional-per-trade or None for options)
    ("options 2026-09-02", DEFAULT_LAB_DIR,
     DEFAULT_LAB_DIR / "recovery_archive"
     / "positions_open_2026-09-02_20260903093433.json", None),
    ("shares  2026-09-02", DEFAULT_LAB_DIR / "shares",
     DEFAULT_LAB_DIR / "shares" / "recovery_archive"
     / "positions_open_2026-09-02_20260903093448.json", 10_000.0),
    ("options 2026-09-03", DEFAULT_LAB_DIR, DEFAULT_LAB_DIR / "positions_open.json", None),
    ("shares  2026-09-03", DEFAULT_LAB_DIR / "shares",
     DEFAULT_LAB_DIR / "shares" / "positions_open.json", 10_000.0),
]


def und_at(feed, sym, day, hhmm):
    """Underlying mid at one minute, for the reconstructed underlying twin."""
    q = stock_quote_at(feed, sym, day, hhmm)
    return (q[0] + q[1]) / 2.0 if q else None


def _hold(entry_ts, day):
    try:
        e = dt.datetime.fromisoformat(str(entry_ts))
        x = dt.datetime.fromisoformat(f"{day}T{EOD}:00")
        return round((x - e).total_seconds() / 60.0, 2)
    except (ValueError, TypeError):
        return None


def option_bid_at(feed, sym, expiry, right, strike, day, hhmm):
    """Historical 1-minute NBBO for one contract at one minute. None if unavailable."""
    try:
        rows = feed._get_csv("/option/history/quote", symbol=sym, expiration=expiry,
                             date=day, right=right, strike=strike, interval="1m")
    except FeedOutage:
        return None
    for r in rows:
        try:
            if r["timestamp"].strip('"')[11:16] == hhmm:
                b = float(r["bid"])
                return b if b > 0 else None
        except (KeyError, ValueError):
            continue
    return None


def stock_quote_at(feed, sym, day, hhmm):
    try:
        rows = feed._get_csv("/stock/history/quote", symbol=sym,
                             start_date=day, end_date=day, interval="1m")
    except FeedOutage:
        return None
    for r in rows:
        try:
            if r["timestamp"].strip('"')[11:16] == hhmm:
                b, a = float(r["bid"]), float(r["ask"])
                return (b, a) if b > 0 and a >= b else None
        except (KeyError, ValueError):
            continue
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild the lost EOD exits.")
    ap.add_argument("--commit", action="store_true",
                    help="write to trades.jsonl (default is a dry run)")
    args = ap.parse_args(argv)

    feed = ThetaLiveFeed()
    grand = 0.0
    written = 0
    for label, labdir, snap, notional in SOURCES:
        if not Path(snap).exists():
            print(f"{label}: snapshot missing -- skipped")
            continue
        blob = json.loads(Path(snap).read_text(encoding="utf-8"))
        day = blob.get("session_date") or label.split()[-1]
        pos = blob["positions"]
        store = LabStore(labdir)
        # Dedup on (setup, symbol, entry_ts) ONLY. An earlier version included `arm`, which
        # silently failed for the shares arm: a shares snapshot position has no `arm` field
        # (None) while its trade record carries "SHARES", so the keys never matched and a
        # position that HAD already closed was written a second time. Caught by the
        # signal-reconciliation check, which showed 23 trades against 22 decisions.
        already = {(t.get("setup_id"), t.get("symbol"), str(t.get("entry_ts")))
                   for t in store.read("trades.jsonl")}
        print(f"\n{label}: {len(pos)} open positions in the snapshot, session {day}")
        tot = 0.0
        for p in pos:
            key = (p.get("setup_id"), p.get("symbol"), str(p.get("entry_ts")))
            if key in already:
                print(f"   SKIP already recorded: {p['setup_id']}/{p['symbol']}")
                continue
            if notional is None:
                bid = option_bid_at(feed, p["symbol"], p["expiration"],
                                    p["right"][0].upper() if p.get("right") else "C",
                                    p["strike"], day, EOD)
                if bid is None:
                    print(f"   NO QUOTE {p['setup_id']}/{p['symbol']} "
                          f"{p['strike']}{p.get('right')} -- left unrecorded")
                    continue
                c = float(p.get("contracts") or 1.0)
                pnl = 100.0 * (bid - float(p["entry_ask"])) * c - 2 * FEE * c
                # Start from the POSITION and override only the exit fields. Enumerating
                # fields by hand dropped 29 of them the first time -- including
                # underlying_return, which made every reconstructed trade silently vanish
                # from attribution.py rather than fail loudly.
                trade = dict(p)
                trade.update({
                    "exit_ts": f"{day}T{EOD}:00", "exit_bid": bid,
                    "pnl_gross": round(100.0 * (bid - float(p["entry_ask"])) * c, 2),
                    "pnl_net": round(pnl, 2),
                    "return_pct": (bid / float(p["entry_ask"]) - 1.0),
                    "exit_reason": "eod_reconstructed", "reconstructed": True})
                eu = p.get("entry_underlying")
                xu = und_at(feed, p["symbol"], day, EOD)
                if eu and xu:
                    sgn = 1.0 if (p.get("right") or "call").lower().startswith("c") else -1.0
                    trade["exit_underlying"] = xu
                    trade["underlying_return"] = sgn * (xu / eu - 1.0)
                trade["hold_minutes"] = _hold(p.get("entry_ts"), day)
            else:
                q = stock_quote_at(feed, p["symbol"], day, EOD)
                if q is None:
                    print(f"   NO QUOTE {p['setup_id']}/{p['symbol']} -- left unrecorded")
                    continue
                bid, ask = q
                px = bid if p["direction"] == "long" else ask
                sh = float(p["shares"])
                pnl = (sh * (px - p["entry_px"]) if p["direction"] == "long"
                       else sh * (p["entry_px"] - px))
                trade = dict(p)
                trade.update({
                    "arm": "SHARES", "config_hash": SHARES_HASH,
                    "exit_ts": f"{day}T{EOD}:00", "exit_px": px,
                    "notional": notional, "pnl_net": round(pnl, 4),
                    "return_pct": pnl / notional, "entry_spread_bp": p.get("spread_bp"),
                    "hold_minutes": _hold(p.get("entry_ts"), day),
                    "exit_reason": "eod_reconstructed", "reconstructed": True})
            tot += trade["pnl_net"]
            print(f"   {p['setup_id']:<26}{p['symbol']:<5}{str(p.get('arm') or 'SHARES'):<6}"
                  f"{trade['pnl_net']:+9.2f}")
            if args.commit:
                store.write_trade(trade)
                written += 1
        print(f"   subtotal {tot:+,.2f}")
        grand += tot

    feed.close()
    print(f"\ngrand total of reconstructed P&L: ${grand:+,.2f}")
    if args.commit:
        print(f"WROTE {written} trades, all flagged exit_reason=eod_reconstructed")
    else:
        print("DRY RUN -- nothing written. Re-run with --commit to apply.")
    print("2026-09-01 is not recoverable: its snapshot was overwritten before the "
          "archive existed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
