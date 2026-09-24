"""Tests for the opening-window fix (autostart.py, ledger.py). 2026-09-24.

What happened: autostart held both runners until a post-open freshness preflight passed,
09:33-09:35, on the stated belief that "no setup can signal before 09:36". Crabel_Stretch
trades from 09:31. So on every session from 2026-08-31 both arms started after the open:
the shares arm processed nothing until ~09:41 and then fired Crabel_Stretch on the first
bar it could see (11 names at 09:41 on 2026-09-23), and the options arm took the 09:31
signal five minutes late. Meanwhile the ledger called every one of those days PARTIAL by
a few seconds, a label that was on so often it could warn about nothing.

What can lie here, in order of danger:

  1. THE RUNNERS STILL WAITING FOR THE GATE. The fix is worthless if anything still
     sequences the start after the post-open preflight. Driven through the real main().
  2. THE GATE SILENTLY GONE. Starting early must not mean the delayed-feed check never
     runs: it must still fire at FRESHNESS_AT, once, and stop the arms it names.
  3. AN ABORT OVERWRITTEN. The ledger ranks a day by its best state, so a COLLECTED
     written after the gate's ABORTED would erase the abort from the record.
  4. THE VERDICT CHANGING MEANING. Extracting the rules into _gate_verdict must not change
     their precedence -- any DELAYED stops every arm, as it always did.
  5. THE LABEL. A start after the open is PARTIAL; a pre-open start is not.

Each was verified to fail against autostart.py / ledger.py as of e4dfaf6.
"""
from __future__ import annotations

import datetime as dt
import sys
import types

import pytest

from trade_analysis.live_lab import autostart as A
from trade_analysis.live_lab import ledger
from trade_analysis.live_lab.preflight import FAIL, OK, OPTIONS_TAG

DAY = dt.date(2026, 9, 25)          # a Friday
CLEAN = f"{OK} QQQ underlying REAL-TIME\n{OK} SPY underlying REAL-TIME"
DELAYED = f"{FAIL} QQQ underlying is DELAYED by ~15.0 min. Every entry"


def _at(h, m, s=0):
    return dt.datetime.combine(DAY, dt.time(h, m, s))


# ------------------------------------------------------------ 1-3. through main()

@pytest.fixture
def drive(tmp_path, monkeypatch):
    """Run the real main() with the network, the OS and supervise() stubbed out."""
    clock = {"t": _at(9, 5)}
    calls = {"preflight": [], "supervise": None, "fire_gate": False}
    replies = [CLEAN]

    def fake_preflight(symbols):
        # A real preflight took 2m16s on 2026-09-23 (09:33:00 -> 09:35:16). A stub that
        # takes no time would let the old sequencing pass these tests.
        calls["preflight"].append(list(symbols))
        clock["t"] += dt.timedelta(seconds=135)
        return replies[min(len(calls["preflight"]) - 1, len(replies) - 1)]

    def fake_wait_for_open(target):
        clock["t"] = max(clock["t"], dt.datetime.combine(DAY, target))
        return True

    def fake_supervise(args, day, specs=None, child_cls=None, post_open=None):
        calls["supervise"] = {"post_open": post_open,
                              "preflights_before": len(calls["preflight"]),
                              "at": clock["t"]}
        if post_open and calls["fire_gate"]:
            clock["t"] = max(clock["t"], dt.datetime.combine(DAY, post_open[0]))
            post_open[1]()
        return 0

    monkeypatch.setattr(sys, "stdout", sys.stdout)     # main() installs a Tee; undo it
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    for mod in (A, ledger):
        monkeypatch.setattr(mod, "now_et", lambda: clock["t"])
    monkeypatch.setattr(A, "wait_for_open", fake_wait_for_open)
    monkeypatch.setattr(A, "terminal_up", lambda *a, **k: True)
    monkeypatch.setattr(A, "_wait_for_history", lambda *a, **k: True)
    monkeypatch.setattr(A, "ThetaLiveFeed", lambda *a, **k: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(A, "is_holiday", lambda *a, **k: False)
    monkeypatch.setattr(A, "hold_system_awake", lambda: "stubbed")
    monkeypatch.setattr(A, "release_system_awake", lambda: None)
    monkeypatch.setattr(A, "_preflight", fake_preflight)
    monkeypatch.setattr(A, "supervise", fake_supervise)

    def run(*extra):
        rc = A.main(["--lab-dir", str(tmp_path), "--no-archive", *extra])
        final = ledger.latest_for(DAY, str(tmp_path)) or {}
        return rc, final
    return types.SimpleNamespace(run=run, calls=calls, replies=replies, clock=clock)


def test_runners_start_after_the_structural_check_not_after_the_open(drive):
    rc, final = drive.run()
    sup = drive.calls["supervise"]
    assert sup is not None, "supervise was never reached"
    assert sup["preflights_before"] == 1, (
        f"{sup['preflights_before']} preflights ran before the runners started -- the "
        f"post-open freshness check is still sequenced ahead of the start")
    assert sup["at"].time() < dt.time(9, 30), f"runners started at {sup['at']:%H:%M:%S}"


def test_the_freshness_gate_is_still_scheduled_for_after_the_open(drive):
    drive.run()
    post_open = drive.calls["supervise"]["post_open"]
    assert post_open is not None, "starting early dropped the delayed-feed gate"
    assert post_open[0] == A.FRESHNESS_AT


def test_a_delayed_feed_at_the_open_ends_the_day_aborted(drive):
    drive.replies[:] = [CLEAN, DELAYED]          # pre-open: unverifiable; 09:33: delayed
    drive.calls["fire_gate"] = True
    rc, final = drive.run()
    assert drive.calls["preflight"][-1] == ["QQQ", "SPY"], "gate should check the feed, not 15 names"
    assert final.get("outcome") == "ABORTED", f"ledger ended {final.get('outcome')}: abort erased"
    assert rc == 3


def test_a_clean_pre_open_start_is_collected_not_partial(drive):
    rc, final = drive.run()
    assert final.get("outcome") == "COLLECTED", final


# ------------------------------------------------------------ 2. the hook in supervise()

class _Fake:
    made: list = []
    clock: dict = {}

    def __init__(self, name, cmd, root):
        self.name, self.done, self.rc, self.restarts = name, False, 0, 0
        self.stopped_at = None
        _Fake.made.append(self)

    def start(self):
        pass

    def poll(self):
        return None if self.stopped_at is None else 0

    def terminate(self):
        if self.stopped_at is None:
            self.stopped_at = _Fake.clock["t"]


@pytest.fixture
def sup(monkeypatch, tmp_path):
    """supervise() with a clock that ticks 30s per read and never really sleeps."""
    clock = {"t": _at(9, 7)}

    def tick():
        clock["t"] += dt.timedelta(seconds=30)
        return clock["t"]

    monkeypatch.setattr(A, "now_et", tick)
    monkeypatch.setattr(A, "_sleep_watched", lambda *a, **k: None)
    monkeypatch.setattr(A, "HARD_STOP_GRACE_SEC", 0)
    monkeypatch.setattr(A, "terminal_up", lambda *a, **k: True)
    _Fake.made, _Fake.clock = [], clock
    args = types.SimpleNamespace(lab_dir=str(tmp_path), start_terminal=False)
    specs = [("options", ["x"]), ("shares", ["y"])]

    def run(verdict):
        fired = []

        def gate():
            fired.append(clock["t"])
            return verdict
        A.supervise(args, DAY, specs=specs, child_cls=_Fake,
                    post_open=(A.FRESHNESS_AT, gate))
        return fired, {c.name: c for c in _Fake.made}
    return run


def test_gate_fires_once_after_its_time(sup):
    fired, kids = sup(None)
    assert len(fired) == 1, f"gate fired {len(fired)} times"
    assert fired[0].time() >= A.FRESHNESS_AT


def test_all_verdict_stops_every_arm_at_the_gate(sup):
    fired, kids = sup(("all", "feed is DELAYED"))
    for name in ("options", "shares"):
        assert kids[name].stopped_at is not None and kids[name].stopped_at <= fired[0] + \
            dt.timedelta(minutes=1), f"{name} not stopped by the gate"


def test_options_verdict_leaves_shares_running(sup):
    fired, kids = sup(("options", "1 OPTIONS-only failure(s)"))
    assert kids["options"].stopped_at <= fired[0] + dt.timedelta(minutes=1)
    assert kids["shares"].stopped_at.time() >= A.HARD_STOP, \
        "the shares arm was stopped by an options-only verdict"


# ------------------------------------------------------------ 4. verdict precedence

@pytest.mark.parametrize("report, options_only, want", [
    (CLEAN, False, None),
    (DELAYED, False, "all"),
    (f"{FAIL} {OPTIONS_TAG} QQQ options DELAYED by ~15.0 min.", False, "all"),
    (f"{FAIL} {OPTIONS_TAG} QQQ: chain snapshot failed -- entitlement?", False, "options"),
    (f"{FAIL} {OPTIONS_TAG} QQQ: chain snapshot failed -- entitlement?", True, "all"),
    (f"{FAIL} XLF: no warmup bars", False, "all"),
])
def test_gate_verdict_keeps_the_old_precedence(report, options_only, want):
    got = A._gate_verdict(report, options_only=options_only)
    assert (got[0] if got else None) == want, (report, got)


def test_the_start_leaves_room_for_the_shares_warmup():
    """Shares warmup took ~6 min on 2026-09-23 (09:35:18 -> first bar ~09:41). Starting at
    09:20 plus a ~2 min preflight would leave two minutes of margin; demand fifteen."""
    open_ = dt.datetime.combine(DAY, dt.time(9, 30))
    start = dt.datetime.combine(DAY, A.START_AT)
    assert open_ - start >= dt.timedelta(minutes=2, seconds=15) + dt.timedelta(minutes=6) \
        + dt.timedelta(minutes=15), f"START_AT {A.START_AT} leaves too little warmup margin"


# ------------------------------------------------------------ 5. the label

@pytest.mark.parametrize("hhmm, late", [((9, 7), False), ((9, 29), False),
                                        ((9, 31), True), ((9, 34), True)])
def test_partial_means_started_after_the_open(tmp_path, monkeypatch, hhmm, late):
    monkeypatch.setattr(ledger, "now_et", lambda: _at(*hhmm))
    ledger.note_opened(DAY, str(tmp_path))
    rec = ledger.latest_for(DAY, str(tmp_path))
    assert rec["late_start"] is late, (hhmm, rec)
