"""Tests for gap recovery and the after-close flatten (2026-09-25).

What can lie here, in order of danger:

  1. AN AFTER-HOURS QUOTE BECOMING A FILL. The host slept 15:39 -> 16:19 and the shares
     arm, waking, sold two XLP longs at an after-hours bid 6.2% below entry. After 16:00
     neither arm may take a fresh quote; the last in-session mark is used and flagged.
  2. A RECOVERY THAT EDITS THE RECORD. Corrections are APPENDED; trades.jsonl is never
     touched; a second run adds nothing.
  3. A RECOVERY THAT INVENTS. Only positions that were open when a gap began are resolved;
     only EXACT rebuilds are applied by default; a refused run writes no correction.
  4. THE GAP ITSELF MISREAD. Back-to-back suspends are one gap; other days are ignored.
  5. A POSITION THAT CANNOT BE REBUILT. From 2026-09-25 a shares FILL carries the exit
     parameters, so a rebuild never depends on recomputing them from revised bars.
"""
from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest

from trade_analysis.live_lab import gap_recovery as G

DAY = dt.date(2026, 9, 25)


def _t(h, m, s=0):
    return dt.datetime.combine(DAY, dt.time(h, m, s))


def _write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


SUSPENDS = [  # the three records the 2026-09-25 session actually wrote
    {"ts": "2026-09-25T15:42:57", "kind": "host_suspend",
     "detail": "host suspended or clock stepped: 197s unaccounted between 15:39:33 and 15:42:56 ET"},
    {"ts": "2026-09-25T15:57:30", "kind": "host_suspend",
     "detail": "host suspended or clock stepped: 867s unaccounted between 15:42:57 and 15:57:29 ET"},
    {"ts": "2026-09-25T16:19:16", "kind": "host_suspend",
     "detail": "host suspended or clock stepped: 1289s unaccounted between 15:57:35 and 16:19:09 ET"},
    {"ts": "2026-09-24T12:00:00", "kind": "host_suspend",
     "detail": "host suspended or clock stepped: 60s unaccounted between 11:59:00 and 12:00:00 ET"},
]


# ------------------------------------------------------------ 4. the gap

def test_back_to_back_suspends_are_one_gap_and_other_days_are_ignored(tmp_path):
    _write(tmp_path / "outages.jsonl", SUSPENDS)
    assert G.find_gaps(tmp_path, DAY) == [(_t(15, 39, 33), _t(16, 19, 9))]


def test_a_trade_open_at_the_gap_is_affected_and_one_closed_before_is_not():
    gaps = [(_t(15, 39, 33), _t(16, 19, 9))]
    trades = [
        {"entry_ts": "2026-09-25T09:52:00", "exit_ts": "2026-09-25T16:19:11", "exit_reason": "eod"},
        {"entry_ts": "2026-09-25T09:52:00", "exit_ts": "2026-09-25T10:06:00", "exit_reason": "stop"},
        {"entry_ts": "2026-09-25T15:45:00", "exit_ts": "2026-09-25T15:50:00", "exit_reason": "stop"},
    ]
    assert set(G.affected(trades, gaps)) == {0}


def test_a_close_after_16_is_affected_even_without_a_recorded_gap():
    trades = [{"entry_ts": "2026-09-25T10:00:00", "exit_ts": "2026-09-25T16:05:00",
               "exit_reason": "eod"}]
    assert set(G.affected(trades, [])) == {0}


# ------------------------------------------------------------ 3. applying

CORR = [
    {"kind": "gap_recovery", "status": "recovered", "signal_id": "a",
     "recovered": {"exit_px": 81.9, "exit_reason": "eod", "pnl_net": 20.0},
     "original": {"exit_px": 76.66, "pnl_net": -620.34}},
    {"kind": "gap_recovery", "status": "recovered_approximate", "signal_id": "b",
     "recovered": {"exit_px": 1.0, "exit_reason": "eod", "pnl_net": 5.0},
     "original": {"pnl_net": -1.0}},
    {"kind": "gap_recovery_run", "day": "2026-09-25"},
]


def test_only_exact_recoveries_are_applied_by_default_and_originals_survive():
    trades = [{"signal_id": "a", "pnl_net": -620.34}, {"signal_id": "b", "pnl_net": -1.0},
              {"signal_id": "c", "pnl_net": 3.0}]
    out = G.apply_corrections(trades, CORR, "signal_id")
    assert out[0]["pnl_net"] == 20.0 and out[0]["corrected"] == "gap_recovery"
    assert out[0]["original_exit"]["pnl_net"] == -620.34
    assert out[1]["pnl_net"] == -1.0, "an APPROXIMATE rebuild was applied by default"
    assert trades[0]["pnl_net"] == -620.34, "apply_corrections mutated the raw rows"
    best = G.apply_corrections(trades, CORR, "signal_id",
                               include=("recovered", "recovered_approximate"))
    assert best[1]["pnl_net"] == 5.0


# ------------------------------------------------------------ 2. append-only

def _lab(tmp_path, monkeypatch, agreement_rule=True):
    _write(tmp_path / "outages.jsonl", SUSPENDS)
    trades = [
        {"signal_id": "open", "setup_id": "Crabel_Stretch", "symbol": "XLP",
         "entry_ts": "2026-09-25T09:52:00", "exit_ts": "2026-09-25T16:19:11",
         "exit_px": 76.66, "exit_reason": "eod", "pnl_net": -620.34},
        {"signal_id": "done", "setup_id": "ORB_5min", "symbol": "XLF",
         "entry_ts": "2026-09-25T09:48:00", "exit_ts": "2026-09-25T10:05:03",
         "exit_px": 54.4, "exit_reason": "target", "pnl_net": 36.65},
    ]
    _write(tmp_path / "trades.jsonl", trades)
    fake = {"open": {"signal_id": "open", "exit_ts": "2026-09-25T15:56:02", "exit_px": 82.08,
                     "exit_reason": "eod", "pnl_net": 42.82, "rebuild": "exact (recorded at entry)"},
            "done": {"signal_id": "done", "exit_ts": "2026-09-25T10:05:02", "exit_px": 54.4,
                     "exit_reason": "target" if agreement_rule else "stop", "pnl_net": 36.0,
                     "rebuild": "exact (recorded at entry)"}}
    monkeypatch.setattr(G, "replay_shares", lambda day, trades, arm_dir, activate=None: (fake, []))
    return trades


def test_recovery_appends_once_and_never_touches_trades(tmp_path, monkeypatch):
    _lab(tmp_path, monkeypatch)
    before = (tmp_path / "trades.jsonl").read_text()
    rep = G.recover(DAY, "shares", write=True, lab_dir=tmp_path)
    assert rep["written"] == 1 and rep["recovered_net"] == pytest.approx(42.82)
    again = G.recover(DAY, "shares", write=True, lab_dir=tmp_path)
    assert again["written"] == 0, "a second run duplicated the correction"
    assert (tmp_path / "trades.jsonl").read_text() == before, "trades.jsonl was edited"
    rows = [json.loads(l) for l in (tmp_path / "trade_corrections.jsonl").read_text().splitlines()]
    fixes = [r for r in rows if r["kind"] == "gap_recovery"]
    assert len(fixes) == 1 and fixes[0]["original"]["pnl_net"] == -620.34


def test_a_replay_that_gets_the_rules_wrong_is_refused(tmp_path, monkeypatch):
    _lab(tmp_path, monkeypatch, agreement_rule=False)
    rep = G.recover(DAY, "shares", write=True, lab_dir=tmp_path)
    assert rep.get("refused") and rep["written"] == 0
    rows = [json.loads(l) for l in (tmp_path / "trade_corrections.jsonl").read_text().splitlines()]
    assert not [r for r in rows if r["kind"] == "gap_recovery"], "refused run wrote a correction"


def test_report_mode_writes_nothing(tmp_path, monkeypatch):
    _lab(tmp_path, monkeypatch)
    G.recover(DAY, "shares", write=False, lab_dir=tmp_path)
    assert not (tmp_path / "trade_corrections.jsonl").exists()


# ------------------------------------------------------------ 1. after-close flatten

class _NoFeed:
    """Any fresh quote after the close is a bug; calling this is the failure."""
    calls = 0

    def stock_quote(self, sym):
        raise AssertionError("fetched a fresh quote after the close")

    def close(self):
        pass


def test_shares_after_close_flatten_uses_the_last_session_mark(tmp_path):
    from trade_analysis.live_lab import shares_runner as SR
    lab = SR.SharesLab(["XLP"], lab_dir=tmp_path)
    lab.feed.close()
    lab.feed = _NoFeed()
    lab._last_quote["XLP"] = {"ts": _t(15, 39, 4), "bid": 81.90, "ask": 81.92, "mid": 81.91}
    lab.open_pos = [SR.SharePos(
        setup_id="Crabel_Stretch", symbol="XLP", direction="long",
        entry_ts="2026-09-25T09:52:00", entry_bar_ts="2026-09-25T09:51:00", entry_px=81.73,
        shares=122.4, spread_bp=1.2, stop=None, target=None, time_exit_min=None,
        bar_exit=None, trailing=None, state={}, timeframe="1m", signal_id="x")]
    lab._flatten_all(_t(16, 19, 11), "eod")
    row = json.loads((tmp_path / "trades.jsonl").read_text().splitlines()[-1])
    assert row["exit_px"] == 81.90 and row["exit_reason"] == "eod_after_close_mark"


def test_options_after_close_flatten_uses_the_last_mark(tmp_path):
    from trade_analysis.live_lab import runner as R
    from trade_analysis.live_lab.positions import Position
    lab = R.LiveLab(["QQQ"], lab_dir=tmp_path)
    lab.feed.close()
    lab.feed = _NoFeed()
    lab.open_pos = [Position(
        position_id="p", signal_id="s", config_hash="h", setup_id="Crabel_Stretch",
        setup_version="1", symbol="QQQ", arm="ATM", direction="long", right="call",
        strike=745.0, expiration="2026-09-25", contracts=1.0,
        entry_ts="2026-09-25T10:00:00", entry_bar_ts="2026-09-25T09:59:00", entry_ask=1.5,
        entry_bid=1.49, entry_spread_pct=0.01, entry_underlying=745.0, iv_derived=None,
        delta_derived=None, open_interest=None, stop=None, target=None, time_exit_min=None,
        bar_exit=None, trailing=None, state={}, last_bid=1.20, last_underlying=744.0)]
    lab._flatten_all(_t(16, 19, 11), reason="shutdown")
    row = json.loads((tmp_path / "trades.jsonl").read_text().splitlines()[-1])
    assert row["exit_bid"] == 1.20 and row["exit_reason"] == "shutdown_after_close_mark"


# ------------------------------------------------------------ 5. exit params at entry

def test_a_shares_fill_records_the_exit_parameters(tmp_path, monkeypatch):
    from trade_analysis.live_lab import shares_runner as SR
    from trade_analysis.live_lab.session import Signal
    lab = SR.SharesLab(["XLP"], lab_dir=tmp_path)
    lab.feed.close()

    class _Setup:
        id, timeframe, max_per_day, max_per_direction = "STUB", "1m", 9, None

        def evaluate(self, ctx):
            return Signal("STUB", "long", stop=80.0, target=83.0, time_exit_min=25,
                          trailing="imb", state={"k": 1})
    monkeypatch.setattr(SR, "ALL_SETUPS", [_Setup()])
    bar = {"ts": _t(10, 0), "open": 81, "high": 81, "low": 81, "close": 81, "volume": 1}
    lab._enter("XLP", None, bar, SimpleNamespace(), None,
               {"ts": _t(10, 1), "bid": 81.0, "ask": 81.02, "mid": 81.01}, _t(10, 1, 2))
    fills = [json.loads(l) for l in (tmp_path / "signals.jsonl").read_text().splitlines()
             if '"FILL"' in l]
    f = fills[-1]
    assert (f["stop"], f["target"], f["time_exit_min"], f["trailing"], f["timeframe"]) == \
        (80.0, 83.0, 25, "imb", "1m")


# ------------------------------------------------------------ 6. only stolen exits (2026-10-08)
#
# The host slept 15:13-15:31, the catch-up replayed 15:12-15:51 at the real minutes and the
# 15:55 flatten ran live -- and this pass still re-priced all 40 positions open across the
# suspend, swapping live fills for history estimates. At 15:55 a 0DTE contract's history
# quote can sit $73 from the live fill of the same minute (2026-09-29, -$417 on the day).

OCT8 = dt.date(2026, 10, 8)


def _o8(h, m, s=0):
    return dt.datetime.combine(OCT8, dt.time(h, m, s))


def _oct8(tmp_path, monkeypatch, caught_up=True, replay_exit=("eod", "2026-10-08T15:56:02", 9.0),
          live_exit=("eod", "2026-10-08T15:56:01"), rebuild="exact (recorded at entry)"):
    _write(tmp_path / "outages.jsonl", [
        {"ts": "2026-10-08T15:31:52", "kind": "host_suspend",
         "detail": "host suspended or clock stepped: 1101s unaccounted between 15:13:26 and 15:31:51 ET"}])
    _write(tmp_path / "events.jsonl", [
        {"ts": "2026-10-08T15:52:13", "kind": "caught_up", "since": "2026-10-08T15:13:24.86",
         "until": "2026-10-08T15:52:01.65"}] if caught_up else [])
    trades = [
        {"signal_id": "open", "setup_id": "Crabel_Stretch", "symbol": "XLY",
         "entry_ts": "2026-10-08T10:02:02", "exit_ts": live_exit[1],
         "exit_px": 110.0, "exit_reason": live_exit[0], "pnl_net": 50.41},
        {"signal_id": "done", "setup_id": "ORB_5min", "symbol": "XLF",
         "entry_ts": "2026-10-08T09:48:00", "exit_ts": "2026-10-08T10:05:03",
         "exit_px": 54.4, "exit_reason": "target", "pnl_net": 36.65}]
    _write(tmp_path / "trades.jsonl", trades)
    reason, ts, pnl = replay_exit
    fake = {"open": {"signal_id": "open", "exit_ts": ts, "exit_px": 110.1, "exit_reason": reason,
                     "pnl_net": pnl, "rebuild": rebuild},
            "done": {"signal_id": "done", "exit_ts": "2026-10-08T10:05:02", "exit_px": 54.4,
                     "exit_reason": "target", "pnl_net": 36.0, "rebuild": "exact (recorded at entry)"}}
    monkeypatch.setattr(G, "replay_shares", lambda day, trades, arm_dir, activate=None: (fake, []))


def _fixes(tmp_path):
    p = tmp_path / "trade_corrections.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    return [r for r in rows if r["kind"] == "gap_recovery"]


def test_a_suspend_the_catch_up_replayed_steals_nothing(tmp_path, monkeypatch):
    _oct8(tmp_path, monkeypatch, caught_up=True, replay_exit=("stop", "2026-10-08T15:20:02", -30.0))
    rep = G.recover(OCT8, "shares", write=True, lab_dir=tmp_path)
    assert rep["gaps"] == [] and rep["affected"] == 0 and rep["written"] == 0
    assert _fixes(tmp_path) == []


def test_the_same_suspend_without_a_catch_up_is_still_recovered(tmp_path, monkeypatch):
    _oct8(tmp_path, monkeypatch, caught_up=False, replay_exit=("stop", "2026-10-08T15:20:02", -30.0))
    rep = G.recover(OCT8, "shares", write=True, lab_dir=tmp_path)
    assert rep["written"] == 1 and _fixes(tmp_path)[0]["recovered"]["exit_reason"] == "stop"


def test_a_replay_that_agrees_with_live_is_not_written(tmp_path, monkeypatch):
    _oct8(tmp_path, monkeypatch, caught_up=False)          # eod 15:56:02 vs live eod 15:56:01
    rep = G.recover(OCT8, "shares", write=True, lab_dir=tmp_path)
    assert rep["affected"] == 1 and rep["written"] == 0
    assert rep["by_status"]["live_confirmed"] == 1 and _fixes(tmp_path) == []


def test_only_the_minutes_no_catch_up_replayed_count():
    # 2026-10-02: the restart caught up from 13:05, so 12:01:30-13:05 was never managed.
    assert G.uncovered((_o8(12, 1, 30), _o8(13, 5, 25)), [(_o8(13, 5), _o8(15, 7))]) == \
        [(_o8(12, 1, 30), _o8(13, 5))]
    # 2026-09-29: a catch-up that starts 3s after the suspend leaves no unmanaged minute.
    assert G.uncovered((_o8(11, 48, 10), _o8(11, 52, 19)), [(_o8(11, 48, 13), _o8(11, 52, 23))]) == []
    assert G.uncovered((_o8(11, 0), _o8(12, 0)), []) == [(_o8(11, 0), _o8(12, 0))]


def test_a_retraction_restores_the_live_row():
    trades = [{"signal_id": "a", "pnl_net": -20.08}]
    fix = {"kind": "gap_recovery", "status": "recovered", "signal_id": "a",
           "recovered": {"pnl_net": 13.92}, "original": {"pnl_net": -20.08}}
    assert G.apply_corrections(trades, [fix], "signal_id")[0]["pnl_net"] == 13.92
    undo = {"kind": "gap_recovery", "status": "retracted", "signal_id": "a"}
    out = G.apply_corrections(trades, [fix, undo], "signal_id")
    assert out[0] == trades[0] and "corrected" not in out[0]


def test_overcorrections_keep_the_stolen_and_retract_the_rest(tmp_path):
    day = "2026-10-08"
    _write(tmp_path / "events.jsonl", [{"kind": "caught_up", "since": f"{day}T11:04:37",
                                        "until": f"{day}T11:15:43"}])
    _write(tmp_path / "trades.jsonl", [
        {"signal_id": k, "entry_ts": f"{day}T09:40:00"} for k in ("cov", "stolen", "same", "late")])

    def c(k, gap, live, rec):
        return {"kind": "gap_recovery", "status": "recovered", "day": day, "signal_id": k,
                "setup_id": "S", "symbol": "X", "gap": [f"{day}T{gap[0]}", f"{day}T{gap[1]}"],
                "original": {"exit_reason": live[0], "exit_ts": f"{day}T{live[1]}", "pnl_net": live[2]},
                "recovered": {"exit_reason": rec[0], "exit_ts": f"{day}T{rec[1]}", "pnl_net": rec[2]},
                "written_at": "2026-10-08T16:01:16"}
    _write(tmp_path / "trade_corrections.jsonl", [
        # 2026-09-28's shape: covered by the catch-up, and the replay even disagrees with live
        c("cov", ("11:04:46", "11:15:15"), ("trail", "12:30:04", -13.13), ("eod", "15:56:02", 33.58)),
        c("stolen", ("13:00:00", "14:00:00"), ("eod", "15:55:01", -50.0), ("stop", "13:20:02", -20.0)),
        c("same", ("13:00:00", "14:00:00"), ("eod", "15:55:01", 5.0), ("eod", "15:55:02", 6.0)),
        c("late", ("16:19:11", "16:19:11"), ("eod", "16:19:11", -620.34), ("eod", "15:56:02", 42.82))])
    bad = G.retract(tmp_path, OCT8, "signal_id", write=True)
    assert sorted(x["signal_id"] for x, _ in bad) == ["cov", "same"]
    view = {r["signal_id"]: r["pnl_net"] for r in G.apply_corrections(
        [{"signal_id": k, "pnl_net": p} for k, p in
         (("cov", -13.13), ("stolen", -50.0), ("same", 5.0), ("late", -620.34))],
        G._jsonl(tmp_path / "trade_corrections.jsonl"), "signal_id")}
    assert view == {"cov": -13.13, "stolen": -20.0, "same": 5.0, "late": 42.82}
    assert G.retract(tmp_path, OCT8, "signal_id", write=True) == [], "a second retract re-wrote"


# ------------------------------------------------------------ 7. the review's holes (2026-10-08)

def test_an_exit_live_made_after_the_gap_is_not_overridden(tmp_path, monkeypatch):
    # 09-24's shape: asleep 15:13-15:31, then a network freeze made the live flatten 2 min
    # late. The replay's 15:55 history quote must not replace the 15:57 fill live observed.
    _oct8(tmp_path, monkeypatch, caught_up=False, live_exit=("eod", "2026-10-08T15:57:04"),
          replay_exit=("eod", "2026-10-08T15:55:02", 80.0))
    rep = G.recover(OCT8, "shares", write=True, lab_dir=tmp_path)
    assert rep["affected"] == 1 and rep["written"] == 0 and _fixes(tmp_path) == []


def test_an_estimate_never_overrides_a_live_exit_after_the_gap(tmp_path, monkeypatch):
    # 09-18's shape: live saw the stop at 15:40, after waking at 15:31; the estimate only
    # assumes a 15:55 flatten because the exit parameters were unknown.
    _oct8(tmp_path, monkeypatch, caught_up=False, live_exit=("stop", "2026-10-08T15:40:02"),
          replay_exit=("eod_estimate", "2026-10-08T15:56:02", 20.0),
          rebuild="estimate (parameters unknown; 15:55 flatten assumed)")
    rep = G.recover(OCT8, "shares", write=True, lab_dir=tmp_path)
    assert rep["written"] == 0 and _fixes(tmp_path) == []


def test_a_different_rule_from_a_real_rebuild_after_the_gap_stands(tmp_path, monkeypatch):
    # A trail raised by a run-up inside the gap fires at 15:35, after the wake; live, blind
    # to the run-up, held to the 15:56 flatten. That exit was stolen.
    _oct8(tmp_path, monkeypatch, caught_up=False, replay_exit=("trail", "2026-10-08T15:35:02", 70.0))
    rep = G.recover(OCT8, "shares", write=True, lab_dir=tmp_path)
    assert rep["written"] == 1 and _fixes(tmp_path)[0]["recovered"]["exit_reason"] == "trail"


def test_a_position_due_after_the_close_still_joins_before_the_flatten():
    # Activated at the live exit time, 16:42 (2026-08-28's late "shutdown"), it used to never
    # enter the book -- no flatten, no row, always "not replayable".
    seen = []

    class Lab:
        def __init__(self):
            self.open_pos = []

        def _tick(self, t, day):
            seen.append((t.time(), list(self.open_pos)))

        def _flatten_all(self, now, reason):
            seen.append(("flatten", list(self.open_pos)))

    lab = Lab()
    G._drive(lab, SimpleNamespace(now=None), OCT8, lab._tick, "eod", [(_o8(16, 42), "late")])
    at_1555 = [ps for t, ps in seen if t == dt.time(15, 55, 2)]
    assert at_1555 == [["late"]], "the position missed the 15:55 tick"


def test_the_same_rule_inside_the_gap_is_still_stolen(tmp_path, monkeypatch):
    # The time exit was due at 15:20, while the host slept; live could only take it at 15:40.
    _oct8(tmp_path, monkeypatch, caught_up=False, live_exit=("time", "2026-10-08T15:40:02"),
          replay_exit=("time", "2026-10-08T15:20:02", -5.0))
    rep = G.recover(OCT8, "shares", write=True, lab_dir=tmp_path)
    assert rep["written"] == 1
