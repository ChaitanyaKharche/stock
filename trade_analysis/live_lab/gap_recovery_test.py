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
