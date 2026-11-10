"""17.2 / 14.11: market-hours refusal and the no-0DTE halt rule."""
from __future__ import annotations

import datetime as dt

from . import replay_chunks as R


def test_refuses_inside_market_hours_on_a_weekday():
    assert R.in_market_hours(dt.datetime(2026, 11, 4, 10, 0))          # Wednesday 10:00
    assert not R.in_market_hours(dt.datetime(2026, 11, 4, 17, 0))
    assert not R.in_market_hours(dt.datetime(2026, 11, 7, 10, 0))      # Saturday


def test_halt_only_where_a_zero_dte_was_expected():
    has = {"2021-03-09": False,   # Tuesday 2021: no 0DTE expected -> fine
           "2021-03-10": False,   # Wednesday 2021: expected -> halt
           "2022-11-03": False,   # Thursday before 2022-11-14: no Tue/Thu expiry yet -> fine
           "2022-11-10": False,   # Thursday 2022-11-10: the last one without -> fine
           "2022-11-15": False,   # Tuesday after 2022-11-14: expected -> halt
           "2023-06-14": True}
    assert R.halt_reasons(has) == ["2021-03-10", "2022-11-15"]


def test_a_year_is_twelve_calendar_months():
    ms = R.months(2019)
    assert len(ms) == 12 and ms[0] == (dt.date(2019, 1, 1), dt.date(2019, 1, 31))
    assert ms[1][1] == dt.date(2019, 2, 28) and ms[11][1] == dt.date(2019, 12, 31)


def test_a_failed_window_writes_failed_json_and_reraises(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("TerminalUnreachable: dead")
    monkeypatch.setattr(R.B, "run", boom)
    out = tmp_path / "2019-03"
    import pytest
    with pytest.raises(RuntimeError):
        R.replay_window("shares", dt.date(2019, 3, 1), dt.date(2019, 3, 31), out, "abc")
    f = (out / "FAILED.json")
    assert f.exists() and "TerminalUnreachable" in f.read_text(encoding="utf-8")
    assert not (out / "RUN.json").exists()


def test_a_good_window_writes_run_json_and_clears_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(R.B, "run", lambda *a, **k: [{"date": "2019-03-01"}])
    out = tmp_path / "2019-03"
    out.mkdir()
    (out / "FAILED.json").write_text("{}", encoding="utf-8")
    meta = R.replay_window("shares", dt.date(2019, 3, 1), dt.date(2019, 3, 31), out, "abc")
    assert meta["sessions_replayed"] == 1 and (out / "RUN.json").exists()
    assert not (out / "FAILED.json").exists()
