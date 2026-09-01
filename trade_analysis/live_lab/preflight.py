"""Preflight — run this BEFORE trusting a live session.

Answers the questions that silently ruin a run:

  1. Is the terminal reachable, and is it exposed to the LAN?
  2. Which subscription tiers are actually active?
  3. Is the options feed REAL-TIME or DELAYED? (the one that decides whether the lab
     is measuring what it thinks it is)
  4. Is a 0DTE expiration listed today, and is the ATM contract tradeable?
  5. Do warmup bars exist for every symbol?

Quote freshness can only be judged with the market OPEN. Run during RTH.

    python -m trade_analysis.live_lab.preflight
    python -m trade_analysis.live_lab.preflight --symbols QQQ SPY
"""
from __future__ import annotations

import argparse
import datetime as dt
import socket
import subprocess
import sys

from . import options as opt
from .clock import is_rth, now_et, offset_hours
from .feed import FeedOutage, ThetaLiveFeed

RTH_OPEN, RTH_CLOSE = dt.time(9, 30), dt.time(16, 0)
DELAYED_THRESHOLD_SEC = 120.0     # anything staler than this during RTH is not real-time

OK, WARN, FAIL = "  [OK]  ", "  [WARN]", "  [FAIL]"


def _lan_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def check_exposure() -> list[str]:
    """The terminal has been observed to ignore host=127.0.0.1 and bind 0.0.0.0."""
    out = []
    ip = _lan_ip()
    if not ip:
        return [f"{WARN} could not determine LAN IP; skipping exposure check"]
    try:
        import httpx
        r = httpx.get(f"http://{ip}:25503/v3/stock/snapshot/quote",
                      params={"symbol": "QQQ"}, timeout=5.0)
        if r.status_code == 200:
            out.append(f"{FAIL} paid feed is REACHABLE from {ip}:25503 -- the terminal is")
            out.append("         binding 0.0.0.0 regardless of host= in config.toml.")
            out.append("         Fix with an inbound firewall block on TCP 25503")
            out.append("         (loopback bypasses Windows Firewall, so the lab keeps working).")
        else:
            out.append(f"{OK} not reachable from {ip} (HTTP {r.status_code})")
    except Exception:                                        # noqa: BLE001
        out.append(f"{OK} not reachable from {ip} (connection refused)")
    return out


def check_freshness(feed, symbol, now) -> list[str]:
    """Real-time vs delayed. Only meaningful during RTH."""
    out = []
    in_rth = is_rth(now)
    q = feed.stock_quote(symbol)
    if not q:
        return [f"{FAIL} {symbol}: no underlying quote at all"]
    age = (now - q["ts"]).total_seconds()
    if not in_rth:
        out.append(f"{WARN} {symbol} underlying: last quote {q['ts']:%Y-%m-%d %H:%M:%S} "
                   f"(market closed -- freshness UNVERIFIABLE, rerun during RTH)")
        return out
    if age <= DELAYED_THRESHOLD_SEC:
        out.append(f"{OK} {symbol} underlying REAL-TIME (age {age:.1f}s)")
    else:
        out.append(f"{FAIL} {symbol} underlying is DELAYED by ~{age/60:.1f} min. Every entry")
        out.append("         and exit would be priced off stale quotes. Do NOT trust results.")
    return out


def check_options(feed, symbol, day, now) -> list[str]:
    out = []
    try:
        exp = feed.zero_dte(symbol, day)
    except FeedOutage as exc:
        return [f"{FAIL} {symbol}: expirations unavailable ({exc})"]
    if exp is None:
        return [f"{WARN} {symbol}: no 0DTE listed for {day} "
                "(expected on non-expiry weekdays for some symbols)"]
    try:
        chain = feed.chain_quotes(symbol, exp)
    except FeedOutage as exc:
        return [f"{FAIL} {symbol}: chain snapshot failed -- options entitlement? ({exc})"]
    if not chain:
        return [f"{FAIL} {symbol}: chain returned 0 usable rows"]
    out.append(f"{OK} {symbol} 0DTE {exp}: {len(chain)} usable contracts")

    in_rth = is_rth(now)
    ages = [(now - c["ts"]).total_seconds() for c in chain]
    med_age = sorted(ages)[len(ages) // 2]
    if in_rth:
        if med_age <= DELAYED_THRESHOLD_SEC:
            out.append(f"{OK} {symbol} options REAL-TIME (median quote age {med_age:.1f}s)")
        else:
            out.append(f"{FAIL} {symbol} options DELAYED by ~{med_age/60:.1f} min.")
            out.append("         The options half of the lab cannot be trusted on this tier.")
    else:
        out.append(f"{WARN} {symbol} options: median quote age {med_age/60:.1f} min "
                   "(market closed -- UNVERIFIABLE, rerun during RTH)")

    q = feed.stock_quote(symbol)
    if q:
        try:
            arms = opt.select_arms(chain, q["mid"], "call")
            atm = arms["ATM"]
            out.append(f"{OK} {symbol} ATM K={atm['strike']} bid={atm['bid']:.2f} "
                       f"ask={atm['ask']:.2f} spread={100*(atm['ask']-atm['bid'])/atm['mid']:.1f}% "
                       f"of mid | dist {arms['strike_distance_pct']:.3f}% from spot")
            wings = [k for k in ("ATM-1", "ATM+1") if arms.get(k)]
            out.append(f"{OK} {symbol} wings available: {wings or 'NONE'}")
        except opt.UnusableQuote as exc:
            out.append(f"{WARN} {symbol}: ATM not tradeable right now ({exc})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Preflight checks for the live lab.")
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    ap.add_argument("--ignore-exposure", action="store_true",
                    help="report LAN exposure but do not count it as a failure")
    args = ap.parse_args(argv)

    now = now_et()
    day = now.date()
    in_rth = is_rth(now)

    print(f"\nPREFLIGHT  {now:%Y-%m-%d %H:%M:%S}  "
          f"({'MARKET OPEN' if in_rth else 'market closed'})  local clock offset {offset_hours():+.1f}h")
    print("=" * 78)

    feed = ThetaLiveFeed()
    fails = 0

    print("\n-- exposure --")
    for line in check_exposure():
        print(line)
        fails += line.startswith(FAIL)

    print("\n-- feed --")
    for sym in args.symbols:
        for line in check_freshness(feed, sym, now):
            print(line)
            fails += line.startswith(FAIL)

    print("\n-- options --")
    for sym in args.symbols:
        for line in check_options(feed, sym, day, now):
            print(line)
            fails += line.startswith(FAIL)

    print("\n-- warmup bars --")
    prior, d = [], day - dt.timedelta(days=1)
    while len(prior) < 3:
        if d.weekday() < 5:
            prior.append(d)
        d -= dt.timedelta(days=1)
    for sym in args.symbols:
        try:
            counts = [len(feed.minute_bars(sym, p)) for p in prior]
        except FeedOutage as exc:
            print(f"{FAIL} {sym}: historical bars unavailable ({exc})")
            fails += 1
            continue
        if min(counts) < 300:
            print(f"{FAIL} {sym}: thin prior sessions {counts} -- indicator seeding will be short")
            fails += 1
        else:
            print(f"{OK} {sym}: prior sessions {counts} bars")

    print("\n" + "=" * 78)
    if fails:
        print(f"  {fails} FAILURE(S) -- resolve before trusting a live session.")
    elif not in_rth:
        print("  No hard failures, but quote freshness is UNVERIFIED while the market is")
        print("  closed. Rerun during RTH before the first real session.")
    else:
        print("  All checks passed. Safe to run the lab.")
    print(f"  feed stats: {feed.stats()}")
    feed.close()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
