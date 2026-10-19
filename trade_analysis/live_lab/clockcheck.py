"""Is this machine's clock right? (specs/001-clock-trust.md)

Both arms throw away any stock quote whose age -- THIS machine's clock minus the
exchange's timestamp -- is more than QUOTE_REJECT_SEC, or that far in the future
(runner.py and shares_runner.py, STALE_QUOTE_SEC / FUTURE_QUOTE_SEC). So the arms are
only as right as the laptop's clock.

On 2026-10-15 the laptop woke from a ten-hour hibernate with its clock ~7 s fast (Windows
set it from the hardware clock; the Windows Time service, on Manual, did not correct it
until 15:19 MST, after the close). Every quote measured 6-8 s old, both arms rejected all
of them (946 + 1,770 `stale_quote` rows), and the session took zero trades -- while
preflight printed "All checks passed", because its own freshness limit is 120 s.

This module MEASURES the offset with SNTP (RFC 4330): one UDP packet to a public time
server, no admin rights, nothing installed. It deliberately does not:

  * set the clock -- that needs admin rights the lab does not run with;
  * correct quote ages by the offset -- bar admission runs on the same clock, so on a
    fast clock a "corrected" quote would let the arms act on a bar before it has closed.
    The fix for a wrong clock is to fix the clock, and the job here is to say so at once
    instead of letting a session go silently to zero.

Every result is advisory. Nothing here can fail a preflight or stop an arm: NTP is often
blocked on work networks, and a gate that cannot pass in its environment is broken, not
strict.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import socket
import struct
import time

CLOCK_TAG = "CLOCK:"
QUOTE_REJECT_SEC = 5.0          # == runner.STALE_QUOTE_SEC == shares_runner.STALE_QUOTE_SEC (pinned by a test)
CLOCK_WARN_SEC = 2.0
NTP_HOSTS = ("time.windows.com", "pool.ntp.org")
NTP_EPOCH = 2208988800          # seconds from 1900-01-01 (NTP) to 1970-01-01 (Unix)
FIX_HINT = ("Fix: in an ADMIN PowerShell run  w32tm /resync /force  "
            "(and set the Windows Time service to Automatic).")


def parse_sntp(packet: bytes, t_send: float, t_recv: float) -> float | None:
    """Seconds this machine's clock is AHEAD of the server (negative = behind).

    t_send / t_recv are this machine's clock (Unix seconds) when the request left and the
    reply arrived. Standard NTP offset, sign flipped so that a FAST local clock is positive.
    """
    if len(packet) < 48:
        return None
    if packet[0] & 0x07 != 4 or packet[1] == 0:        # must be a server reply, stratum > 0
        return None

    def stamp(at: int) -> float | None:
        sec, frac = struct.unpack("!II", packet[at:at + 8])
        return None if sec == 0 else sec - NTP_EPOCH + frac / 2 ** 32

    server_recv, server_send = stamp(32), stamp(40)
    if server_recv is None or server_send is None:
        return None
    server_minus_local = ((server_recv - t_send) + (server_send - t_recv)) / 2.0
    return -server_minus_local


def sntp_ahead(host: str, timeout: float = 2.0) -> float | None:
    """One SNTP exchange with `host`. None on any failure; never raises."""
    try:
        with cf.ThreadPoolExecutor(max_workers=1) as ex:        # getaddrinfo has no timeout
            family, _, _, _, addr = ex.submit(
                socket.getaddrinfo, host, 123, 0, socket.SOCK_DGRAM).result(timeout=timeout)[0]
        with socket.socket(family, socket.SOCK_DGRAM) as s:
            s.settimeout(timeout)
            t_send = time.time()
            s.sendto(b"\x23" + b"\x00" * 47, addr)              # LI 0, version 4, mode 3 (client)
            data, _ = s.recvfrom(512)
            t_recv = time.time()
        return parse_sntp(data, t_send, t_recv)
    except Exception:                                          # noqa: BLE001
        return None


def measure(hosts=NTP_HOSTS, timeout: float = 2.0, query=None) -> tuple[float | None, str | None]:
    """(seconds ahead, host) from the first server that answers; (None, None) if none do."""
    query = query or sntp_ahead
    for host in hosts:
        try:
            ahead = query(host, timeout)
        except Exception:                                      # noqa: BLE001
            ahead = None
        if ahead is not None:
            return ahead, host
    return None, None


def describe(ahead: float | None, host: str | None,
             reject_sec: float = QUOTE_REJECT_SEC) -> tuple[str, str]:
    """('OK' | 'WARN', one line of plain English). Never 'FAIL' -- see the module docstring."""
    if ahead is None:
        return "WARN", (f"{CLOCK_TAG} could not reach an internet time server (UDP 123 is often "
                        f"blocked on work networks) -- clock offset UNVERIFIED. The arms reject "
                        f"quotes more than {reject_sec:.0f}s old or ahead, judged by this clock.")
    side = "AHEAD of" if ahead > 0 else "BEHIND"
    mag = abs(ahead)
    if mag < CLOCK_WARN_SEC:
        return "OK", f"{CLOCK_TAG} this machine is {ahead:+.1f}s from internet time ({host})"
    if mag < reject_sec:
        return "WARN", (f"{CLOCK_TAG} this machine's clock is drifting: {mag:.1f}s {side} "
                        f"internet time ({host}). The arms reject quotes past {reject_sec:.0f}s, "
                        f"so every quote is being judged {mag:.1f}s wrong. {FIX_HINT}")
    return "WARN", (f"{CLOCK_TAG} this machine's clock is {mag:.1f}s {side} internet time "
                    f"({host}) -- past the arms' {reject_sec:.0f}s quote limit, so they will "
                    f"reject EVERY quote and take no trades until it is corrected (this is "
                    f"what zeroed 2026-10-15). {FIX_HINT}")


class ClockWatch:
    """In-session check, called from the supervisor loop. Logs a skew when it starts, at
    most once per `repeat_sec` while it lasts, and once when it ends. Never raises."""

    def __init__(self, day: dt.date, log, measure=None, enabled: bool = True,
                 interval_sec: float = 300, repeat_sec: float = 900):
        self.day, self.log, self.enabled = day, log, enabled
        self._measure = measure
        self.interval_sec, self.repeat_sec = interval_sec, repeat_sec
        self._last_check: float | None = None
        self._last_warn: float | None = None
        self._skewed = False
        self._unverified_logged = False

    def check(self, now: dt.datetime) -> str:
        if not self.enabled or now.date() != self.day or \
                not (dt.time(9, 25) <= now.time() < dt.time(16, 0)):
            return "idle"
        t = time.monotonic()
        if self._last_check is not None and t - self._last_check < self.interval_sec:
            return "waiting"
        self._last_check = t
        try:
            ahead, host = (self._measure or measure)()
            if ahead is None:
                if not self._unverified_logged:
                    self.log(f"[clock] {describe(None, None)[1]}")
                    self._unverified_logged = True
                return "unverified"
            if abs(ahead) >= CLOCK_WARN_SEC:
                if not self._skewed or self._last_warn is None or \
                        t - self._last_warn >= self.repeat_sec:
                    self.log(f"[clock] {describe(ahead, host)[1]}")
                    self._last_warn = t
                self._skewed = True
                return "skewed"
            if self._skewed:
                self.log(f"[clock] clock back within {CLOCK_WARN_SEC:.0f}s of internet time "
                         f"({ahead:+.1f}s, {host})")
                self._skewed = False
                return "recovered"
            return "ok"
        except Exception as exc:                               # noqa: BLE001
            try:
                self.log(f"[clock] check failed and was ignored: {exc!r}")
            except Exception:                                  # noqa: BLE001
                pass
            return "error"
