"""Tests for the signals archive (store.roll_signals / store.read_signals).

signals.jsonl is the forward test's record of every decision and every skip. Rolling it
into per-session files is only acceptable if NOTHING about the record changes. What could
go wrong, in order of danger:

  1. A ROW LOST OR ALTERED. Every row must come back from read_signals, in order,
     identical to what was written.
  2. A ROW COUNTED TWICE. A crash between writing an archive and replacing the active
     file, or rolling twice, must not duplicate a session's rows.
  3. A SEQUENCE NUMBER REUSED. After a roll the active file is nearly empty; a restart
     must still resume the counter above everything already spent.
  4. TODAY'S DECISIONS FORGOTTEN. decision_keys / decision_counts must see the same
     rows before and after a roll -- the decided-once guard depends on it.
"""
from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

from trade_analysis.live_lab import store as S
from trade_analysis.live_lab.store import LabStore, read_signals

TODAY = dt.date(2026, 10, 1)


def _clock(monkeypatch, when):
    monkeypatch.setattr(S, "_now", lambda: when)


def _fill(store, monkeypatch):
    """Two finished sessions and part of today, written through the real writers."""
    for day in (dt.date(2026, 9, 29), dt.date(2026, 9, 30), TODAY):
        for minute in (31, 32):
            t = dt.datetime.combine(day, dt.time(9, minute, 2))
            _clock(monkeypatch, t)
            sid = store.write_decision(setup_id="ORB_5min", config_hash="h", symbol="SPY",
                                       direction="long", bar_ts=t.replace(second=0),
                                       state={"x": minute})
            store.write_fill(sid, status="FILLED")
            store.write_skip(setup_id="ORB_5min", config_hash="h", symbol="QQQ",
                             bar_ts=t.replace(second=0), reason="max_per_day")


def test_rolling_loses_nothing_and_changes_nothing(tmp_path, monkeypatch):
    st = LabStore(tmp_path)
    _fill(st, monkeypatch)
    before = st.read("signals.jsonl")
    keys, counts = st.decision_keys(TODAY), st.decision_counts(TODAY)

    moved = st.roll_signals(TODAY)

    assert moved == {"2026-09-29": 6, "2026-09-30": 6}
    assert sorted(p.name for p in (tmp_path / "signals_archive").iterdir()) == \
        ["2026-09-29.jsonl.gz", "2026-09-30.jsonl.gz"]
    assert st.read("signals.jsonl") == before, "a row was lost, altered or reordered"
    active = [json.loads(l) for l in (tmp_path / "signals.jsonl").read_text().splitlines()]
    assert {r["ts"][:10] for r in active} == {"2026-10-01"}, "an old row stayed active"
    assert st.decision_keys(TODAY) == keys and st.decision_counts(TODAY) == counts
    assert len(read_signals(tmp_path, "2026-09-30")) == 6, "an archived day must stay readable"


def test_rolling_twice_or_after_a_crash_never_duplicates(tmp_path, monkeypatch):
    st = LabStore(tmp_path)
    _fill(st, monkeypatch)
    before = st.read("signals.jsonl")
    snapshot = (tmp_path / "signals.jsonl").read_text()
    st.roll_signals(TODAY)
    # A crash after the archives were written but before the active file was replaced:
    # the old rows are in BOTH places when the next runner starts.
    (tmp_path / "signals.jsonl").write_text(snapshot)
    st.roll_signals(TODAY)
    st.roll_signals(TODAY)
    assert st.read("signals.jsonl") == before


def test_a_restart_after_a_roll_resumes_the_sequence_above_the_archive(tmp_path, monkeypatch):
    st = LabStore(tmp_path)
    _fill(st, monkeypatch)
    hi = max(r["seq"] for r in st.read("signals.jsonl"))
    st.roll_signals(dt.date(2026, 10, 2))          # EVERYTHING archived, active empty
    assert (tmp_path / "signals.jsonl").read_text() == ""
    assert LabStore(tmp_path)._seq > hi            # the roll's own event is above them all
    # The archive alone must carry the counter: events.jsonl is not guaranteed to hold a
    # later seq (a session can die before writing any event after its last signal).
    (tmp_path / "events.jsonl").unlink()
    assert LabStore(tmp_path)._seq == hi, "a restart would reuse sequence numbers"


def test_a_torn_final_line_goes_with_its_session(tmp_path, monkeypatch):
    st = LabStore(tmp_path)
    _fill(st, monkeypatch)
    with open(tmp_path / "signals.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"seq":999,"phase":"SKI')               # killed mid-write
    st.roll_signals(dt.date(2026, 10, 2))
    assert (tmp_path / "signals.jsonl").read_text() == "", "the torn line was left behind"


def test_both_runners_roll_before_they_write_anything(tmp_path, monkeypatch):
    from trade_analysis.live_lab import runner as R
    from trade_analysis.live_lab import shares_runner as SR
    for make in (lambda d: R.LiveLab(["SPY"], lab_dir=d),
                 lambda d: SR.SharesLab(["SPY"], lab_dir=d)):
        d = tmp_path / str(id(make))
        lab = make(d)
        _fill(lab.store, monkeypatch)
        _clock(monkeypatch, dt.datetime.combine(TODAY, dt.time(9, 6)))
        lab.warmup = lambda day: False            # stop right after start-up
        lab.feed = SimpleNamespace(close=lambda: None)
        lab.run(TODAY)
        days = {json.loads(l)["ts"][:10] for l in (d / "signals.jsonl").read_text().splitlines()}
        assert days == {"2026-10-01"}, f"{type(lab).__name__} did not roll at start"
