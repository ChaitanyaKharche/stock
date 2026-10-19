"""Tests for autostart.TerminalWatchdog -- restart a wedged terminal, never an offline one.

What can lie here:
  1. RESTARTING WHILE OFFLINE. In a car between networks, a restart cannot help and adds a
     cold start to the recovery. It must never fire when the internet is unreachable.
  2. NEVER RESTARTING A WEDGED TERMINAL. Answering, internet up, history failing: that is
     the post-network-change state, and it must be restarted.
  3. A RESTART LOOP. Past TERMINAL_MAX_RESTARTS a human is needed, not more restarts --
     except one slow retry every SLOW_RETRY_SEC, so a network that comes back later in the
     session is not wasted (specs/004).
  4. GIVING UP FOREVER. On 2026-10-05 and 10-06 the terminal stopped answering while the
     network was down; the not-answering branch restarted it without asking whether the
     internet was up, all six restarts failed offline inside ~20 minutes, and the watchdog
     then logged "leaving it for a human" every minute until the close (41 and 93 lines). On
     a commute day (home WiFi -> hotspot -> work WiFi) that ends collection at the first long
     gap, even after the laptop reaches the next network.
"""
from __future__ import annotations

import datetime as dt

import pytest

from trade_analysis.live_lab import autostart as A

DAY = dt.date(2026, 9, 28)
T = dt.datetime.combine(DAY, dt.time(10, 30))


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(A, "TERMINAL_CHECK_SEC", 0)
    monkeypatch.setattr(A, "log", lambda msg: None)


def _probes(answering=True, serving=True, internet=True):
    restarts = []
    return restarts, {"answering": lambda: answering, "serving": lambda: serving,
                      "internet": lambda: internet,
                      "restart": lambda why: restarts.append(why) or "restarted"}


def test_offline_never_restarts():
    w = A.TerminalWatchdog(True, DAY)
    restarts, p = _probes(serving=False, internet=False)
    assert all(w.check(T, p) == "offline" for _ in range(20))
    assert restarts == []


def test_wedged_after_a_network_change_is_restarted_on_the_third_check():
    w = A.TerminalWatchdog(True, DAY)
    restarts, p = _probes(serving=False, internet=True)
    assert [w.check(T, p) for _ in range(3)] == ["stuck", "stuck", "restarted"]
    assert len(restarts) == 1


def test_a_healthy_check_resets_the_count():
    w = A.TerminalWatchdog(True, DAY)
    r1, stuck = _probes(serving=False)
    _, ok = _probes()
    w.check(T, stuck); w.check(T, stuck); w.check(T, ok); w.check(T, stuck); w.check(T, stuck)
    assert r1 == [], "two separate blips added up to a restart"


def test_not_answering_is_restarted_on_the_second_check():
    w = A.TerminalWatchdog(True, DAY)
    restarts, p = _probes(answering=False)
    assert [w.check(T, p), w.check(T, p)] == ["down", "restarted"]


def test_the_restart_cap_holds(monkeypatch):
    w = A.TerminalWatchdog(True, DAY)
    monkeypatch.setattr(A, "_evict_stale_terminal", lambda: True)
    monkeypatch.setattr(A, "start_terminal", lambda: False)
    p = {"answering": lambda: False, "serving": lambda: False, "internet": lambda: True,
         "restart": w._restart}
    out = [w.check(T, p) for _ in range(2 * (A.TERMINAL_MAX_RESTARTS + 2))]
    assert out.count("restart_failed") == A.TERMINAL_MAX_RESTARTS
    assert out[-1] in ("gave_up", "down")


def test_idle_outside_rth_and_when_disabled():
    restarts, p = _probes(answering=False)
    assert A.TerminalWatchdog(True, DAY).check(T.replace(hour=8), p) == "idle"
    assert A.TerminalWatchdog(False, DAY).check(T, p) == "idle"
    assert restarts == []


# --- specs/004: survive a commute day ------------------------------------------------

@pytest.fixture
def mech(monkeypatch):
    """The real _restart, with the terminal mechanics faked. `state["start_ok"]` decides
    whether a restart brings the terminal back."""
    calls, state = [], {"start_ok": False}
    monkeypatch.setattr(A, "_graceful_shutdown", lambda *a, **k: True)
    monkeypatch.setattr(A, "_evict_stale_terminal", lambda: False)
    monkeypatch.setattr(A, "_mdds_status", lambda *a, **k: "TEST")

    def fake_start(*a, **k):
        calls.append(1)
        return state["start_ok"]

    monkeypatch.setattr(A, "start_terminal", fake_start)
    return calls, state


def _real(w, answering=False, serving=False, internet=True):
    return {"answering": lambda: answering, "serving": lambda: serving,
            "internet": lambda: internet, "restart": w._restart}


def _exhaust(w):
    for _ in range(2 * (A.TERMINAL_MAX_RESTARTS + 2)):
        w.check(T, _real(w))


def test_not_answering_while_offline_is_never_restarted():
    w = A.TerminalWatchdog(True, DAY)
    restarts, p = _probes(answering=False, serving=False, internet=False)
    assert all(w.check(T, p) == "offline" for _ in range(20))
    assert restarts == []


def test_not_answering_is_restarted_once_the_network_is_back():
    w = A.TerminalWatchdog(True, DAY)
    restarts, offline = _probes(answering=False, serving=False, internet=False)
    for _ in range(10):
        w.check(T, offline)
    assert restarts == []
    _, online = _probes(answering=False, serving=False, internet=True)
    online["restart"] = lambda why: restarts.append(why) or "restarted"
    assert [w.check(T, online), w.check(T, online)] == ["down", "restarted"]
    assert len(restarts) == 1


def test_the_budget_comes_back_after_the_terminal_has_been_healthy(mech):
    calls, _ = mech
    w = A.TerminalWatchdog(True, DAY)
    _exhaust(w)
    assert len(calls) == A.TERMINAL_MAX_RESTARTS
    for _ in range(30):
        assert w.check(T, _real(w, answering=True, serving=True)) == "ok"
    out = [w.check(T, _real(w)) for _ in range(2)]
    assert out == ["down", "restart_failed"], "a healthy stretch did not restore the budget"
    assert len(calls) == A.TERMINAL_MAX_RESTARTS + 1


def test_past_the_cap_a_slow_retry_still_happens(mech, monkeypatch):
    calls, _ = mech
    w = A.TerminalWatchdog(True, DAY)
    _exhaust(w)
    monkeypatch.setattr(A, "SLOW_RETRY_SEC", 0, raising=False)
    out = [w.check(T, _real(w)) for _ in range(2)]
    assert out[-1] == "restart_failed", "gave up for the rest of the session"
    assert len(calls) == A.TERMINAL_MAX_RESTARTS + 1


def test_no_slow_retry_before_its_interval(mech):
    calls, _ = mech
    w = A.TerminalWatchdog(True, DAY)
    _exhaust(w)
    for _ in range(20):
        w.check(T, _real(w))
    assert len(calls) == A.TERMINAL_MAX_RESTARTS


def test_leaving_it_for_a_human_is_logged_once_not_every_minute(mech, monkeypatch):
    lines = []
    monkeypatch.setattr(A, "log", lines.append)
    w = A.TerminalWatchdog(True, DAY)
    _exhaust(w)
    for _ in range(40):
        w.check(T, _real(w))
    assert sum("leaving it for a human" in s for s in lines) == 1
