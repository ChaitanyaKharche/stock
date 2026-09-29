"""Tests for replay_audit -- the causes it assigns must come from what live recorded.

The failure to guard against: a cause that is assigned by default. If every unmatched trade
fell through to "no live signal", the audit would call outage damage vendor noise.
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

from trade_analysis.live_lab import replay_audit as A

DAY = "2026-09-24"


def _t(setup, bar, entry=None, symbol="SPY"):
    return {"setup_id": setup, "symbol": symbol, "direction": "long",
            "entry_bar_ts": f"{DAY}T{bar}:00", "entry_ts": f"{DAY}T{entry or bar}:02",
            "pnl_net": 1.0}


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "LAB", tmp_path)
    (tmp_path / "session_ledger.jsonl").write_text(json.dumps(
        {"date": DAY, "outcome": "OPENED", "started_et": "09:35:02"}) + "\n")
    rows = [
        {"kind": "host_suspend", "ts": f"{DAY}T15:14:10", "from_et": f"{DAY}T14:55:45",
         "to_et": f"{DAY}T15:14:09"},
        {"kind": "stale_bar", "ts": f"{DAY}T10:40:28", "symbol": "QQQ",
         "detail": "QQQ bar 10:31 is 9.5m old -- entries suppressed"},
        {"kind": "stale_quote", "ts": f"{DAY}T12:01:03", "detail": "SPY age=6.9s"},
    ]
    (tmp_path / "outages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "signals.jsonl").write_text(json.dumps(
        {"phase": "SKIP", "setup_id": "ORB_5min", "symbol": "SPY",
         "bar_ts": f"{DAY}T13:00:00", "skip_reason": "max_per_day"}) + "\n")
    return tmp_path


def _ctx(lab, n_live=5):
    return {"live_trades_that_day": n_live, "spans": A.blind_spans(lab, DAY),
            "stale": A.stale_refused(lab, DAY), "guards": A.guard_hits(lab, DAY),
            "live_skips": A.skips(lab / "signals.jsonl", DAY), "replay_skips": {},
            "live_keys": {}}


def test_each_replay_only_trade_gets_the_cause_live_recorded(lab):
    ctx = _ctx(lab)
    assert A.why_replay_only(_t("Crabel_Stretch", "09:31"), ctx) == "late start"
    assert A.why_replay_only(_t("VWAP_Reclaim", "15:02"), ctx) == "host asleep"
    assert A.why_replay_only(_t("VWAP_Reclaim", "10:31", symbol="QQQ"), ctx) == "stale bar refused"
    assert A.why_replay_only(_t("TTM_Squeeze", "12:00"), ctx) == "quote too old"
    assert A.why_replay_only(_t("ORB_5min", "13:00"), ctx) == "live skipped: max_per_day"
    assert A.why_replay_only(_t("ORB_5min", "13:30"), ctx) == "no live signal"
    assert A.why_replay_only(_t("ORB_5min", "13:30"), _ctx(lab, n_live=0)) == "arm blocked"


def test_the_open_is_only_blind_on_a_late_start(lab):
    (lab / "session_ledger.jsonl").write_text(json.dumps(
        {"date": DAY, "outcome": "OPENED", "started_et": "09:06:47"}) + "\n")
    assert A.why_replay_only(_t("Crabel_Stretch", "09:31"), _ctx(lab)) == "no live signal"


def test_a_live_trade_filled_minutes_after_its_bar_is_a_late_entry(lab):
    ctx = _ctx(lab)
    assert A.why_live_only(_t("Crabel_Stretch", "09:31", entry="09:36"), ctx) == "late entry"
    assert A.why_live_only(_t("Crabel_Stretch", "10:31", entry="10:32"), ctx) == "no replay signal"


def test_blind_minutes_are_rth_only_and_overlaps_count_once():
    t = lambda h, m: dt.datetime(2026, 9, 17, h, m)                       # noqa: E731
    spans = [(t(15, 48), t(17, 8), "host asleep"),        # 12 RTH minutes, not 80
             (t(15, 50), t(15, 58), "frozen"),            # inside the sleep: adds nothing
             (t(9, 30), t(9, 41), "late start")]
    assert A.blind_minutes(spans, "2026-09-17") == 23.0
