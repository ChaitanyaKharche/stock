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

import os

import argparse
import datetime as dt
import socket
import subprocess
import sys

from . import options as opt
from .clock import is_rth, now_et, offset_hours
from .feed import FeedOutage, ThetaLiveFeed

# The exchange calendar, for the warmup lookback. A weekday test is not a session test.
try:
    from ..bulk_download.trading_days import is_trading_day as _is_session
except Exception:                                        # pragma: no cover
    def _is_session(d):                                  # noqa: D103
        return d.weekday() < 5

RTH_OPEN, RTH_CLOSE = dt.time(9, 30), dt.time(16, 0)
DELAYED_THRESHOLD_SEC = 120.0     # anything staler than this during RTH is not real-time

OK, WARN, FAIL = "  [OK]  ", "  [WARN]", "  [FAIL]"
# Stable token for callers to key on. autostart used to grep the PROSE
# ("REACHABLE from"), which silently stopped matching the moment the
# exposure check was rewritten to ask Windows instead of the socket --
# turning a non-blocking warning into an abort that would have killed
# every future session.
EXPOSURE_TAG = "EXPOSURE:"
# Stable token marking a failure that belongs to the OPTIONS arm ALONE. The shares arm
# never touches an option endpoint, so an options entitlement lapse must not abort it.
# Keyed on a token rather than the prose for the same reason EXPOSURE_TAG is -- the last
# time a caller grepped wording, a rewrite silently turned a warning into an abort.
OPTIONS_TAG = "OPTIONS:"


def _lan_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def _firewall_exposure() -> list[str]:
    """Ask WINDOWS whether the terminal is actually reachable, not the socket.

    The old check did httpx.get("http://<own-LAN-IP>:25503") from this machine. That
    never traverses the inbound firewall -- Windows treats a connection to your own LAN
    address as local -- so it proved the socket was bound to 0.0.0.0 and nothing about
    whether anyone else can reach it. It would have printed FAIL on a correctly
    firewalled box, which is the worst kind of check: one that is always red and
    therefore always ignored.

    What actually determines exposure is three facts, all of which Windows will tell us:
      1. is the firewall on for the ACTIVE profile,
      2. is there an inbound ALLOW rule matching the terminal binary on that profile,
      3. is the active network Public (untrusted) or Private.
    """
    import json as _json
    import subprocess as _sp
    ps = r"""
$prof = (Get-NetConnectionProfile | Select-Object -First 1)
$cat  = if ($prof) { $prof.NetworkCategory.ToString() } else { 'Unknown' }
$name = if ($prof) { $prof.Name } else { 'unknown' }
$fw   = Get-NetFirewallProfile | Where-Object { $_.Name -eq $cat }
$on   = if ($fw) { [bool]$fw.Enabled } else { $true }
$allow = @(Get-NetFirewallRule -Direction Inbound -Action Allow -Enabled True |
  Where-Object { $_.Profile -match $cat -or $_.Profile -match 'Any' } |
  Where-Object { ($_ | Get-NetFirewallApplicationFilter).Program -like '*thetaterminal*' })
[PSCustomObject]@{ net=$name; cat=$cat; fwOn=$on; nAllow=$allow.Count } | ConvertTo-Json -Compress
"""
    # -EncodedCommand, not -Command: a multi-line script passed as a single -Command
    # argument gets mangled by the Windows command-line parser and silently returns
    # nothing, which this check then reported as "exposure unknown".
    import base64 as _b64
    try:
        enc = _b64.b64encode(ps.encode("utf-16-le")).decode()
        r = _sp.run(["powershell", "-NoProfile", "-NonInteractive",
                     "-EncodedCommand", enc],
                    capture_output=True, text=True, timeout=60)
        line = next((x for x in reversed((r.stdout or "").splitlines())
                     if x.strip().startswith("{")), None)
        if not line:
            return [f"{WARN} firewall query returned nothing; exposure unknown"]
        d = _json.loads(line.strip())
    except Exception as exc:                                 # noqa: BLE001
        return [f"{WARN} could not query Windows Firewall ({type(exc).__name__}); "
                f"exposure unknown"]

    net, cat, on, n = d.get("net"), d.get("cat"), d.get("fwOn"), int(d.get("nAllow", 0))
    out = [f"  network '{net}' is {cat}; firewall {'ON' if on else 'OFF'} for that profile"]
    if not on:
        out.append(f"{FAIL} {EXPOSURE_TAG} firewall is OFF on the active "
                   f"profile -- the feed is exposed")
    elif n > 0:
        sev = FAIL if cat.lower() == "public" else WARN
        out.append(f"{sev} {EXPOSURE_TAG} {n} inbound ALLOW rule(s) match the Theta Terminal binary on "
                   f"the {cat} profile,")
        out.append(f"         so 25503 IS reachable from this network. On a PUBLIC or "
                   f"guest network that")
        out.append(f"         exposes a paid market-data feed to every other host on it.")
        out.append(f"         Remove with (run as Administrator):")
        out.append(f"           Get-NetFirewallRule -Direction Inbound -Action Allow | "
                   f"Where-Object {{")
        out.append(f"             ($_ | Get-NetFirewallApplicationFilter).Program -like "
                   f"'*thetaterminal*' }} |")
        out.append(f"             Disable-NetFirewallRule")
    else:
        out.append(f"{OK} no inbound allow rule for the terminal on the {cat} profile; "
                   f"default-deny holds")
    return out


def check_exposure() -> list[str]:
    """The terminal ignores host=127.0.0.1 and binds 0.0.0.0. Whether that MATTERS is a
    firewall question, not a socket question -- see _firewall_exposure."""
    if os.name != "nt":
        return [f"{WARN} exposure check is Windows-only; skipped"]
    return _firewall_exposure()


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
        return [f"{FAIL} {OPTIONS_TAG} {symbol}: expirations unavailable ({exc})"]
    if exp is None:
        return [f"{WARN} {symbol}: no 0DTE listed for {day} "
                "(expected on non-expiry weekdays for some symbols)"]
    try:
        chain = feed.chain_quotes(symbol, exp)
    except FeedOutage as exc:
        return [f"{FAIL} {OPTIONS_TAG} {symbol}: chain snapshot failed -- entitlement? ({exc})"]
    if not chain:
        return [f"{FAIL} {OPTIONS_TAG} {symbol}: chain returned 0 usable rows"]
    out.append(f"{OK} {symbol} 0DTE {exp}: {len(chain)} usable contracts")

    in_rth = is_rth(now)
    ages = [(now - c["ts"]).total_seconds() for c in chain]
    med_age = sorted(ages)[len(ages) // 2]
    if in_rth:
        if med_age <= DELAYED_THRESHOLD_SEC:
            out.append(f"{OK} {symbol} options REAL-TIME (median quote age {med_age:.1f}s)")
        else:
            out.append(f"{FAIL} {OPTIONS_TAG} {symbol} options DELAYED by ~{med_age/60:.1f} min.")
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
        # --ignore-exposure was declared from the start and then never read, so
        # the flag did nothing and exposure always counted as a hard failure.
        if args.ignore_exposure and line.startswith(FAIL):
            line = WARN + line[len(FAIL):]
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
    # The three prior SESSIONS, not the three prior weekdays.
    #
    # This was `if d.weekday() < 5`, which silently includes market holidays. On
    # 2026-09-08 -- the first session after Labor Day -- it picked 09-07 (a Monday, and
    # a full closure), read 0 bars for it, and failed EVERY symbol with 'thin prior
    # sessions [0, 390, 390]'. Preflight then aborted the entire session. That is not a
    # once-off: it would take out the first trading day after every holiday, roughly ten
    # sessions a year, in a forward test whose whole value is an unbroken record.
    #
    # trading_days.py already knows the calendar and every downloader uses it. There was
    # never a reason for this loop to guess.
    prior, d = [], day - dt.timedelta(days=1)
    guard = 0
    while len(prior) < 3 and guard < 40:
        if _is_session(d):
            prior.append(d)
        d -= dt.timedelta(days=1)
        guard += 1
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
