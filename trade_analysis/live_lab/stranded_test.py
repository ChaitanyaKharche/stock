"""Positions a session never closed get their exits, once, flagged, and never from the live book.

    python -m trade_analysis.live_lab.stranded_test

On 2026-10-05 and 10-06 the hotspot died before the close and stayed down. The runners froze,
the 16:10 backstop killed them, and 134 open positions got no exit row. gap_recovery could
not see them: it only re-prices rows that exist. These tests pin stranded.py and the
autostart hook that runs it. The replay itself needs the vendor's history and is checked
against the live record instead (the self-check); here it is faked.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from . import autostart as A
from . import gap_recovery as G
from . import stranded as ST

DAY = dt.date(2026, 10, 5)
AFTER = dt.datetime(2026, 10, 6, 8, 40)          # next morning
MANAGED = dt.datetime(2026, 10, 5, 14, 8, 24)


@contextlib.contextmanager
def _patched(obj, **attrs):
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def _pos(sid, symbol="XLE", entry="2026-10-05T14:05:02"):
    return {"setup_id": "TTM_Squeeze", "symbol": symbol, "direction": "long",
            "entry_ts": entry, "entry_bar_ts": entry[:16] + ":00", "entry_px": 63.69,
            "shares": 157.0, "spread_bp": 1.57, "stop": 63.4, "target": 64.5,
            "time_exit_min": None, "bar_exit": None, "trailing": None, "state": {},
            "timeframe": "5m", "entry_quote_ts": "", "signal_id": sid}


def _row(sid, reason="stop", exit_ts="2026-10-05T10:19:02", pnl=-10.0, entry="2026-10-05T10:00:02"):
    return {"seq": 1, "arm": "SHARES", "config_hash": "53229d6f1f24df10", "setup_id": "ORB_5min",
            "symbol": "XLF", "direction": "long", "entry_ts": entry, "entry_bar_ts": entry[:16] + ":00",
            "entry_px": 53.7, "shares": 186.2, "exit_ts": exit_ts, "exit_px": 53.6,
            "exit_reason": reason, "pnl_net": pnl, "signal_id": sid}


def _lab(tmp, positions, trades=(), managed=MANAGED, where="archive", saved="2026-10-05T16:13:32"):
    arm = Path(tmp) / "shares"
    (arm / "recovery_archive").mkdir(parents=True, exist_ok=True)
    blob = {"saved_at": saved, "managed_through": managed.isoformat() if managed else None,
            "session_date": DAY.isoformat(), "positions": list(positions)}
    path = (arm / "recovery_archive" / f"positions_open_{DAY}_{saved[11:19].replace(':', '')}.json"
            if where == "archive" else arm / "positions_open.json")
    path.write_text(json.dumps(blob), encoding="utf-8")
    with open(arm / "trades.jsonl", "a", encoding="utf-8") as fh:
        for t in trades:
            fh.write(json.dumps(t) + "\n")
    return arm


def _fake_replay(reason_for_recorded=None, calls=None):
    """Recorded trades replay by their live rule (or `reason_for_recorded`); every stranded
    position exits at the 15:55 flatten for +5."""
    def replay(day, trades, arm_dir, activate=None, extra=None):
        if calls is not None:
            calls.append([(t, p.signal_id) for t, p in extra or []])
        out = {t["signal_id"]: {**t, "exit_reason": reason_for_recorded or t["exit_reason"]}
               for t in trades}
        for _, p in extra or []:
            out[p.signal_id] = {"seq": 99, "arm": "SHARES", "config_hash": "53229d6f1f24df10",
                                "setup_id": p.setup_id, "symbol": p.symbol,
                                "direction": p.direction, "entry_ts": p.entry_ts,
                                "exit_ts": "2026-10-05T15:56:02", "exit_reason": "eod",
                                "pnl_net": 5.0, "signal_id": p.signal_id, "rebuild": None}
        return out, []
    return replay


def _no_options(tmp):
    (Path(tmp) / "trades.jsonl").write_text("", encoding="utf-8")


# --------------------------------------------------------------------------- finding them

def test_a_position_with_no_row_is_found_and_one_that_closed_is_not():
    with tempfile.TemporaryDirectory() as tmp:
        _lab(tmp, [_pos("a"), _pos("b", "XLK"), _pos("", "XLP", "2026-10-05T10:00:02")],
             trades=[_row("b"), {**_row("x"), "symbol": "XLP", "setup_id": "TTM_Squeeze"}])
        items = ST.pending(tmp, before=AFTER.date(), now=AFTER)
        assert [(i["arm"], i["session_date"], i["status"]) for i in items] == \
            [("shares", "2026-10-05", "pending")]
        # "b" closed by signal_id; the keyless XLP one closed by (setup, symbol, entry).
        assert [p["signal_id"] for p in items[0]["missing"]] == ["a"]


def test_todays_book_is_never_touched_while_the_session_can_still_run():
    with tempfile.TemporaryDirectory() as tmp:
        _lab(tmp, [_pos("a")], where="live")
        during = dt.datetime.combine(DAY, dt.time(15, 0))
        assert ST.pending(tmp, before=DAY + dt.timedelta(days=1), now=during) == []
        settled = dt.datetime.combine(DAY, dt.time(16, 10))
        assert len(ST.pending(tmp, before=DAY + dt.timedelta(days=1), now=settled)) == 1
        assert ST.pending(tmp, before=DAY, now=AFTER) == []          # `before` is exclusive


def test_the_latest_copy_of_a_session_wins():
    with tempfile.TemporaryDirectory() as tmp:
        _lab(tmp, [_pos("a")], saved="2026-10-05T14:00:00")
        _lab(tmp, [_pos("a"), _pos("b", "XLK")], saved="2026-10-05T16:13:32")
        items = ST.pending(tmp, before=AFTER.date(), now=AFTER)
        assert len(items) == 1 and len(items[0]["missing"]) == 2


def test_an_old_checkpoint_is_reported_and_never_resolved():
    with tempfile.TemporaryDirectory() as tmp:
        _lab(tmp, [_pos("a")], managed=None)
        (item,) = ST.pending(tmp, before=AFTER.date(), now=AFTER)
        assert item["status"] == "no_managed_through"
        calls = []
        with _patched(G, replay_shares=_fake_replay(calls=calls)):
            rep = ST.resolve(item, tmp, write=True)
        assert calls == [] and rep["written"] == 0 and "managed_through" in rep["refused"]


# --------------------------------------------------------------------------- resolving them

def test_exits_are_appended_flagged_from_the_next_minute_and_written_once():
    with tempfile.TemporaryDirectory() as tmp:
        _no_options(tmp)
        arm = _lab(tmp, [_pos("a"), _pos("b", "XLK")], trades=[_row("r1"), _row("r2", "target")])
        calls = []
        with _patched(G, replay_shares=_fake_replay(calls=calls)):
            (item,) = ST.pending(tmp, before=AFTER.date(), now=AFTER)
            rep = ST.resolve(item, tmp, write=True)
            # Managed from the minute after the last managed one, never from entry.
            assert calls == [[(dt.datetime(2026, 10, 5, 14, 9), "a"),
                              (dt.datetime(2026, 10, 5, 14, 9), "b")]]
            assert rep["written"] == 2 and rep["net"] == 10.0 and "refused" not in rep
            rows = G._jsonl(arm / "trades.jsonl")
            new = [r for r in rows if r.get("reconstructed")]
            assert [r["signal_id"] for r in new] == ["a", "b"]
            for r in new:
                assert r["exit_mode"] == ST.EXIT_MODE and r["exit_reason"] == "eod"
                assert r["reconstruction"]["managed_through"] == MANAGED.isoformat()
                assert r["seq"] != 99 and "rebuild" not in r
            assert [r["signal_id"] for r in rows if not r.get("reconstructed")] == ["r1", "r2"]
            # Running again finds nothing to do.
            assert ST.pending(tmp, before=AFTER.date(), now=AFTER) == []


def test_a_failed_self_check_writes_nothing_and_is_not_retried():
    with tempfile.TemporaryDirectory() as tmp:
        _no_options(tmp)
        arm = _lab(tmp, [_pos("a")], trades=[_row("r1"), _row("r2", "target")])
        with _patched(G, replay_shares=_fake_replay(reason_for_recorded="time")):
            (item,) = ST.pending(tmp, before=AFTER.date(), now=AFTER)
            rep = ST.resolve(item, tmp, write=True)
        assert rep["written"] == 0 and "0%" in rep["refused"]
        assert not any(r.get("reconstructed") for r in G._jsonl(arm / "trades.jsonl"))
        (again,) = ST.pending(tmp, before=AFTER.date(), now=AFTER)
        assert again["status"] == "refused"


def test_a_dry_run_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        _no_options(tmp)
        arm = _lab(tmp, [_pos("a")], trades=[_row("r1")])
        with _patched(G, replay_shares=_fake_replay()):
            (item,) = ST.pending(tmp, before=AFTER.date(), now=AFTER)
            rep = ST.resolve(item, tmp, write=False)
        assert rep["net"] == 5.0 and rep["written"] == 0
        assert len(G._jsonl(arm / "trades.jsonl")) == 1 and not (arm / ST.RUNS).exists()


# --------------------------------------------------------------------------- the autostart hook

class _Calls:
    def __init__(self, up=False, start_ok=True):
        self.log, self.up, self.start_ok = [], up, start_ok

    def fakes(self):
        def note(name, ret=None):
            def f(*a, **k):
                self.log.append(name)
                return ret
            return f

        def run(cmd, **k):
            self.log.append("run")
            return subprocess.CompletedProcess(cmd, 0, "[stranded] shares 2026-10-05: ok\n", "")
        return {"terminal_up": note("terminal_up", self.up),
                "start_terminal": note("start_terminal", self.start_ok),
                "_evict_stale_terminal": note("evict", False),
                "_graceful_shutdown": note("shutdown", True),
                "_kill_terminals": self.kill}, run

    def kill(self, since=None):
        # "kill" only when scoped to what the hook launched; an unscoped kill could take
        # down the terminal a runner is using.
        self.log.append("kill" if since is not None else "kill EVERY terminal")
        return 0


def _hook(calls, items, deadline=None, may_start=True):
    fakes, run = calls.fakes()
    with _patched(A, **fakes), _patched(subprocess, run=run), \
            _patched(ST, pending=lambda lab_dir, before: items):
        A.resolve_stranded(DAY, "unused", before=AFTER.date(), deadline=deadline,
                           may_start_terminal=may_start)


def test_hook_does_nothing_when_nothing_is_stranded():
    c = _Calls()
    _hook(c, [])
    assert c.log == []


def test_hook_shuts_down_a_terminal_it_started():
    c = _Calls(up=False)
    _hook(c, [{"arm": "shares", "session_date": "2026-10-05", "status": "pending", "missing": [1]}])
    assert c.log == ["terminal_up", "evict", "start_terminal", "run", "shutdown", "kill"]


def test_hook_kills_a_terminal_that_never_answered():
    # 2026-10-07 16:30: launched with the network down, it missed the 90s window and was left
    # running; it came up hours later beside the next terminal (HTTP 478, invalid session).
    c = _Calls(up=False, start_ok=False)
    _hook(c, [{"arm": "shares", "session_date": "2026-10-07", "status": "pending", "missing": [1]}])
    assert c.log == ["terminal_up", "evict", "start_terminal", "shutdown", "kill"]


def test_hook_leaves_a_terminal_that_was_already_up():
    c = _Calls(up=True)
    _hook(c, [{"arm": "shares", "session_date": "2026-10-05", "status": "pending", "missing": [1]}])
    assert c.log == ["terminal_up", "run"]


def test_hook_does_not_start_a_replay_it_cannot_finish_before_the_session():
    c = _Calls(up=True)
    soon = A.now_et() + dt.timedelta(seconds=60)
    _hook(c, [{"arm": "shares", "session_date": "2026-10-05", "status": "pending", "missing": [1]}],
          deadline=soon)
    assert c.log == []


def test_hook_only_reports_what_it_will_not_resolve():
    c = _Calls(up=False)
    _hook(c, [{"arm": "shares", "session_date": "2026-09-08", "status": "no_managed_through",
               "missing": [1, 2]}])
    assert c.log == []


def test_main_settles_stranded_positions_before_the_wait_and_in_the_evening():
    """The call sites: the 08:30 invocation resolves earlier sessions BEFORE waiting for
    START_AT, with a deadline ahead of it; an invocation past the cutoff includes today."""
    morning = dt.datetime(2026, 10, 7, 8, 30, 5)
    calls = []

    def hook(day, lab_dir, before, deadline=None, may_start_terminal=False):
        calls.append(("stranded", before, deadline, may_start_terminal))

    def wait(target):
        calls.append(("wait",))
        return False                          # as if past the cutoff: the evening branch

    out, err = sys.stdout, sys.stderr
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
            _patched(A, now_et=lambda: morning, resolve_stranded=hook, wait_for_open=wait,
                     hold_system_awake=lambda: "fake"):
        try:
            assert A.main(["--lab-dir", tmp, "--start-terminal"]) == 0
        finally:
            tee, sys.stdout, sys.stderr = sys.stdout, out, err
            getattr(tee, "fh", None) and tee.fh.close()
    day = morning.date()
    assert calls == [
        ("stranded", day, dt.datetime.combine(day, A.START_AT) - A.STRANDED_MARGIN, True),
        ("wait",),
        ("stranded", day + dt.timedelta(days=1), None, True)]


def test_hook_leaves_everything_alone_while_a_runner_holds_its_lock():
    # Review of 1d295bc: a runner orphaned by a killed supervisor keeps trading on the
    # terminal. One missed 4s probe must not start, replay or kill anything.
    from .lock import SingleInstance
    c = _Calls(up=False)
    fakes, run = c.fakes()
    item = {"arm": "shares", "session_date": "2026-10-05", "status": "pending", "missing": [1]}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        guard = SingleInstance("runner", tmp)
        assert guard.acquire()
        try:
            with _patched(A, **fakes), _patched(subprocess, run=run),                     _patched(ST, pending=lambda lab_dir, before: [item]):
                A.resolve_stranded(DAY, tmp, before=AFTER.date(), may_start_terminal=True)
        finally:
            guard.release()
    assert c.log == []


def test_the_lock_probe_sees_a_holder_and_does_not_relabel_it():
    from .lock import SingleInstance, held
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        assert held("runner", tmp) is False                  # no lock file at all
        guard = SingleInstance("runner", tmp)
        assert guard.acquire()
        before = guard.info.read_text(encoding="utf-8")
        try:
            assert held("runner", tmp) is True
            assert guard.info.read_text(encoding="utf-8") == before
        finally:
            guard.release()
        assert held("runner", tmp) is False


def test_kill_terminals_spares_a_terminal_started_before_the_launch():
    import psutil
    jar = A.TERMINAL_CMD[-1]
    killed = []

    class P:
        def __init__(self, pid, born):
            self.pid, self.born = pid, born
            self.info = {"cmdline": ["java", "-jar", jar]}

        def create_time(self):
            return self.born

        def children(self, recursive=False):
            return []

        def kill(self):
            killed.append(self.pid)

    procs = [P(1, 100.0), P(2, 200.0)]
    with _patched(psutil, process_iter=lambda attrs=None: iter(procs)):
        assert A._kill_terminals(since=150.0) == 1
        assert killed == [2]
        assert A._kill_terminals() == 2                      # no `since`: every terminal


def test_hook_never_raises():
    def boom(lab_dir, before):
        raise RuntimeError("disk on fire")
    with _patched(ST, pending=boom):
        A.resolve_stranded(DAY, "unused", before=AFTER.date())


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
