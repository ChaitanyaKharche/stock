"""Tests for clockcheck -- the lab must not trust a clock it has not checked (specs/001).

What can lie here:
  1. THE ARITHMETIC. A sign error reports a fast clock as slow; the operator then moves
     the clock the wrong way and doubles the skew.
  2. A SILENT PASS. On 2026-10-15 this machine's clock was ~7 s fast after a hibernate.
     Both arms rejected every quote as stale (946 + 1,770 rows, zero trades) while
     preflight printed "All checks passed". The clock line must name the problem.
  3. A GATE THAT CANNOT PASS. NTP (UDP 123) is often blocked on work networks. "Could not
     measure" must be a WARN, never a FAIL -- a gate that cannot pass in its environment is
     broken, not strict.
  4. DRIFT BETWEEN COPIES. The 5 s limit lives in both runners; the check that explains it
     must quote the same number, or it will explain the wrong thing.
"""
from __future__ import annotations

import datetime as dt
import struct

import pytest

from trade_analysis.live_lab import clockcheck as C
from trade_analysis.live_lab import preflight as P

NTP_EPOCH = 2208988800


def _ts(x: float) -> bytes:
    x += NTP_EPOCH
    sec = int(x)
    return struct.pack("!II", sec, int((x - sec) * 2 ** 32))


def _packet(server_recv: float, server_send: float) -> bytes:
    # 48 bytes: header(4) + root delay/dispersion/ref id(12) + reference(8) + originate(8)
    # + receive(8) + transmit(8)
    return b"\x24\x01\x00\x00" + b"\x00" * 28 + _ts(server_recv) + _ts(server_send)


def test_a_fast_clock_is_reported_as_ahead():
    # true time 993.00 at our send; our clock says 1000.00, i.e. 7 s fast
    pkt = _packet(server_recv=993.05, server_send=993.06)
    ahead = C.parse_sntp(pkt, t_send=1000.00, t_recv=1000.11)
    assert ahead == pytest.approx(7.0, abs=1e-3)


def test_a_slow_clock_is_reported_as_behind():
    pkt = _packet(server_recv=1007.05, server_send=1007.06)
    assert C.parse_sntp(pkt, t_send=1000.00, t_recv=1000.11) == pytest.approx(-7.0, abs=1e-3)


def test_a_short_or_garbage_packet_is_not_a_measurement():
    assert C.parse_sntp(b"\x00" * 10, 0.0, 0.0) is None
    assert C.parse_sntp(b"\x24\x01" + b"\x00" * 46, 1000.0, 1000.1) is None   # zero timestamps


def test_measure_falls_through_to_the_next_server():
    seen = []

    def query(host, timeout):
        seen.append(host)
        return None if host == "a" else 0.4

    assert C.measure(("a", "b"), query=query) == (0.4, "b")
    assert seen == ["a", "b"]


def test_measure_reports_none_when_every_server_is_unreachable():
    assert C.measure(("a", "b"), query=lambda h, t: None) == (None, None)


def test_measure_never_raises():
    def boom(host, timeout):
        raise OSError("blocked")
    assert C.measure(("a",), query=boom) == (None, None)


@pytest.mark.parametrize("ahead,level,needle", [
    (0.4, "OK", "+0.4s"),
    (-0.4, "OK", "-0.4s"),
    (3.0, "WARN", "drifting"),
    (7.0, "WARN", "EVERY quote"),
    (-7.0, "WARN", "EVERY quote"),
    (None, "WARN", "UNVERIFIED"),
])
def test_describe(ahead, level, needle):
    got_level, text = C.describe(ahead, "time.windows.com")
    assert got_level == level
    assert needle in text
    assert C.CLOCK_TAG in text


def test_the_limit_quoted_is_the_one_both_runners_enforce():
    from trade_analysis.live_lab import runner, shares_runner
    assert runner.STALE_QUOTE_SEC == shares_runner.STALE_QUOTE_SEC == C.QUOTE_REJECT_SEC


@pytest.mark.parametrize("ahead,prefix", [(7.0, P.WARN), (None, P.WARN), (0.3, P.OK)])
def test_preflight_clock_line_is_never_a_fail(monkeypatch, ahead, prefix):
    monkeypatch.setattr(C, "measure", lambda *a, **k: (ahead, None if ahead is None else "h"))
    lines = P.check_clock()
    assert lines and lines[0].startswith(prefix)
    assert not any(line.startswith(P.FAIL) for line in lines)


def test_preflight_names_the_2026_10_15_failure(monkeypatch):
    monkeypatch.setattr(C, "measure", lambda *a, **k: (6.9, "time.windows.com"))
    text = "\n".join(P.check_clock())
    assert "EVERY quote" in text and "6.9s" in text


# --- the in-session watch ---------------------------------------------------------------

DAY = dt.date(2026, 10, 15)
RTH = dt.datetime.combine(DAY, dt.time(14, 0))


def _watch(values, lines):
    it = iter(values)
    w = C.ClockWatch(DAY, log=lines.append, measure=lambda: (next(it), "h"))
    w.interval_sec = 0
    return w


def test_watch_logs_a_skew_once_and_its_recovery_once():
    lines = []
    w = _watch([0.2, 6.9, 7.0, 7.1, 0.3, 0.2], lines)
    for _ in range(6):
        w.check(RTH)
    assert sum("EVERY quote" in s for s in lines) == 1
    assert sum("back within" in s for s in lines) == 1


def test_watch_repeats_a_lasting_skew_only_after_its_quiet_period():
    lines = []
    w = _watch([7.0] * 10, lines)
    w.repeat_sec = 0
    for _ in range(3):
        w.check(RTH)
    assert sum("EVERY quote" in s for s in lines) == 3


def test_watch_is_idle_outside_the_session():
    lines = []
    w = _watch([9.0], lines)
    assert w.check(RTH.replace(hour=8)) == "idle"
    assert w.check(RTH.replace(day=16)) == "idle"
    assert lines == []


def test_watch_never_raises():
    def boom():
        raise RuntimeError("x")
    w = C.ClockWatch(DAY, log=lambda s: None, measure=boom)
    w.interval_sec = 0
    assert w.check(RTH) == "error"
