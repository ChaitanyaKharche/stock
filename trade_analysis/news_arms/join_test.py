"""The join: no duplicate sessions, events on `session`, overlap, m from both minimums."""
from __future__ import annotations

import datetime as dt
import json

import pytest

from . import join as J


def _chunk(tmp_path, name, days, trades_per_day=2, setup="ORB_5min", hash_="7c6414fc549315f3"):
    d = tmp_path / name
    (d / "daily").mkdir(parents=True)
    (d / "RUN.json").write_text(json.dumps({"start": days[0], "end": days[-1], "repo_sha": "x",
                                           "sessions_replayed": len(days)}), encoding="utf-8")
    lines = []
    for day in days:
        (d / "daily" / f"{day}.json").write_text(json.dumps({"date": day, "config_hash": hash_, "net": 1.0}), encoding="utf-8")
        for k in range(trades_per_day):
            lines.append(json.dumps({"setup_id": setup, "symbol": "QQQ", "entry_ts": f"{day}T10:0{k}:00",
                                     "pnl_net": 1.0 + k, "arm": "SHARES"}))
    (d / "trades.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def test_a_session_in_two_chunks_raises(tmp_path):
    a = _chunk(tmp_path, "a", ["2020-01-02", "2020-01-03"])
    b = _chunk(tmp_path, "b", ["2020-01-03", "2020-01-06"])
    with pytest.raises(RuntimeError):
        J.load_replay("shares", [2020], base_dirs=[a, b])


def test_a_failed_chunk_refuses_to_join(tmp_path):
    a = _chunk(tmp_path, "a", ["2020-01-02"])
    (a / "FAILED.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError):
        J.load_replay("shares", [2020], base_dirs=[a])


def test_join_rows_carry_categories_regime_and_counts(tmp_path):
    a = _chunk(tmp_path, "a", ["2020-01-02", "2020-01-03"])
    sessions, trades, meta = J.load_replay("shares", [2020], base_dirs=[a])
    events = {"2020-01-03": {"types": {"NFP", "war_military"}, "minutes": {}}}
    rows = J.build_join("shares", sessions, trades, events, {"2020-01-02": "low", "2020-01-03": "high"},
                        (dt.date(2020, 1, 1), dt.date(2020, 12, 31)))
    assert [r["date"] for r in rows] == ["2020-01-02", "2020-01-03"]
    assert rows[1]["categories"] == ["NFP", "war_military"] and rows[1]["scheduled_today"]
    assert rows[0]["categories"] == [] and not rows[0]["any_event"]
    assert rows[1]["n_trades"] == 2 and rows[1]["regime"] == "high"
    assert meta["hashes"] == ["7c6414fc549315f3"]


def _options_chunk(tmp_path, has_0dte: dict, trades=()):
    d = tmp_path / "o"
    (d / "daily").mkdir(parents=True)
    days = sorted(has_0dte)
    (d / "RUN.json").write_text(json.dumps({"start": days[0], "end": days[-1]}), encoding="utf-8")
    for day in days:
        (d / "daily" / f"{day}.json").write_text("{}", encoding="utf-8")
    (d / "sessions.json").write_text(json.dumps(has_0dte), encoding="utf-8")
    (d / "trades.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trades), encoding="utf-8")
    return d


def test_options_rows_count_atm_only(tmp_path):
    d = _options_chunk(tmp_path, {"2021-01-04": True},
                       [{"setup_id": "x", "entry_ts": "2021-01-04T10:00:00", "pnl_net": 1.0, "arm": arm}
                        for arm in ("ATM", "ATM-1", "ATM+1")])
    sessions, trades, _ = J.load_replay("options", [2021], base_dirs=[d])
    rows = J.build_join("options", sessions, trades, {}, {}, (dt.date(2021, 1, 1), dt.date(2021, 12, 31)))
    assert rows[0]["n_trades"] == 1


def test_options_sessions_without_a_0dte_never_become_rows(tmp_path):
    # 14.11: Tue 2021-01-05 had no SPY 0DTE; it must not enter as a $0 day
    d = _options_chunk(tmp_path, {"2021-01-04": True, "2021-01-05": False, "2021-01-06": True},
                       [{"setup_id": "x", "entry_ts": "2021-01-05T10:00:00", "pnl_net": 9.0, "arm": "ATM"}])
    sessions, trades, meta = J.load_replay("options", [2021], base_dirs=[d])
    assert sorted(sessions) == ["2021-01-04", "2021-01-06"]
    assert trades == []                                             # a trade on a dropped day goes too
    assert meta["excluded_no_0dte"] == {"n": 1, "by_year": {"2021": 1}, "by_weekday": {"1": 1}}
    rows = J.build_join("options", sessions, trades, {}, {}, (dt.date(2021, 1, 1), dt.date(2021, 12, 31)))
    assert [r["date"] for r in rows] == ["2021-01-04", "2021-01-06"]


def test_options_chunk_without_sessions_json_refuses(tmp_path):
    d = _options_chunk(tmp_path, {"2021-01-04": True})
    (d / "sessions.json").unlink()
    with pytest.raises(RuntimeError):
        J.load_replay("options", [2021], base_dirs=[d])
    s, _, meta = J.load_replay("shares", [2021], base_dirs=[d])      # shares never needs it
    assert list(s) == ["2021-01-04"] and meta["excluded_no_0dte"]["n"] == 0


def test_window_excludes_sessions_outside_it(tmp_path):
    a = _chunk(tmp_path, "a", ["2022-12-30", "2023-01-03"])
    sessions, trades, _ = J.load_replay("shares", [2022, 2023], base_dirs=[a])
    rows = J.build_join("shares", sessions, trades, {}, {}, J.DISCOVERY["shares"])
    assert [r["date"] for r in rows] == ["2022-12-30"]


def _row(day, cats, n_trades=3, setup="ORB_5min"):
    return {"date": day, "categories": cats, "n_categories": len(cats),
            "scheduled_today": bool(set(cats) & set(J.SCHEDULED)), "any_event": bool(cats),
            "n_trades": n_trades, "setup_n": {setup: n_trades}, "year": day[:4], "regime": "mid"}


def test_overlap_matrix_is_symmetric_with_diagonal_counts():
    rows = [_row("2020-01-02", ["CPI", "FOMC"]), _row("2020-01-03", ["CPI"]), _row("2020-01-06", [])]
    m = J.overlap_matrix(rows)
    assert m["CPI"]["CPI"] == 2 and m["FOMC"]["FOMC"] == 1
    assert m["CPI"]["FOMC"] == m["FOMC"]["CPI"] == 1 and m["NFP"]["CPI"] == 0


def test_m_requires_both_minimums_and_excludes_financial_stress():
    rows = [_row(f"2020-01-{d:02d}", ["CPI"], n_trades=2) for d in range(1, 31)]          # 30 sessions, 60 trades
    rows += [_row(f"2020-02-{d:02d}", ["NFP"], n_trades=1) for d in range(1, 30)]          # 29 sessions
    rows += [_row(f"2020-03-{d:02d}", ["PCE"], n_trades=1) for d in range(1, 31)]          # 30 sessions, 30 trades
    rows += [_row(f"2020-04-{d:02d}", ["financial_stress"], n_trades=5) for d in range(1, 31)]
    c = J.counts(rows)
    assert c["per_category"]["CPI"]["tested"] is True
    assert c["per_category"]["NFP"]["tested"] is False          # 29 < 30 sessions
    assert c["per_category"]["PCE"]["tested"] is False          # 30 < 50 trades
    assert c["per_category"]["financial_stress"]["tested"] is False and c["per_category"]["financial_stress"]["never_tested"]
    rows2 = rows + [_row(f"2020-05-{d:02d}", ["trade_tariff"], n_trades=2) for d in range(1, 31)]
    c2 = J.counts(rows2)
    assert c2["per_category"]["trade_tariff"]["tested"] is False          # A1: geopolitical never tested
    assert c2["per_category"]["trade_tariff"]["descriptive_placebo"] is True
    assert c["m_H2_scheduled_types"] == 1 and c["m_H3_scheduled_types"] == 1
    assert c["m_Q7_setup_x_category"] == 1                      # ORB_5min x CPI only
    assert c["n_scheduled_sessions"] == 89 and c["n_no_event_any_type"] == 0


def test_regime_cuts_use_discovery_sessions_only(tmp_path):
    p = tmp_path / "f.csv"
    lines = ["date,med20_range_bp,thin"]
    for i, v in enumerate(range(10, 100, 10)):                   # 9 discovery rows: 10..90
        lines.append(f"2019-01-{i+2:02d},{v},False")
    lines.append("2024-01-02,1000,False")                        # holdout row must not move the cuts
    lines.append("2019-02-01,5000,True")                         # thin row ignored
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    lo, hi = J.regime_cuts(p, dt.date(2018, 1, 1), dt.date(2022, 12, 31))
    assert (lo, hi) == (40.0, 70.0)
    reg = J.regimes(p, (lo, hi))
    assert reg["2019-01-02"] == "low" and reg["2019-01-10"] == "high" and reg["2024-01-02"] == "high"
