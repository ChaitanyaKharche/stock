"""Tests for autostart.TerminalWatchdog -- restart a wedged terminal, never an offline one.

What can lie here:
  1. RESTARTING WHILE OFFLINE. In a car between networks, a restart cannot help and adds a
     cold start to the recovery. It must never fire when the internet is unreachable.
  2. NEVER RESTARTING A WEDGED TERMINAL. Answering, internet up, history failing: that is
     the post-network-change state, and it must be restarted.
  3. A RESTART LOOP. Past TERMINAL_MAX_RESTARTS a human is needed, not more restarts.
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
