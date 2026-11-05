"""14.2: per-arm acceptance, qualifying sessions, suspend windows, scope, the 15 minimum."""
from __future__ import annotations

import json

from . import acceptance as A


def _w(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _trade(day, setup, sym, direction, bar, pnl, arm=None, h=None, exit_reason="target"):
    t = {"setup_id": setup, "symbol": sym, "direction": direction, "entry_ts": f"{day}T{bar}:07",
         "entry_bar_ts": f"{day}T{bar}:00", "exit_ts": f"{day}T10:30:00", "exit_reason": exit_reason,
         "pnl_net": pnl, "config_hash": h}
    if arm:
        t["arm"] = arm
    return t


def _lab(tmp_path, days, outcome="COLLECTED", started="09:07:02", late=False):
    lab = tmp_path / "lab"
    ledger = []
    for d in days:
        ledger.append({"date": d, "outcome": "OPENED", "started_et": started, "late_start": late,
                       "arms": ["options", "shares"]})
        ledger.append({"date": d, "outcome": outcome})
    _w(lab / "session_ledger.jsonl", ledger)
    _w(lab / "outages.jsonl", [])
    return lab


DAYS = [f"2026-09-{d:02d}" for d in range(1, 31) if d % 7 not in (5, 6)][:16]


def test_qualifying_sessions_apply_the_four_rules(tmp_path):
    lab = _lab(tmp_path, DAYS[:3])
    # day 2 is PARTIAL, day 3's trade is under an old hash
    ledger = A._rows(lab / "session_ledger.jsonl")
    ledger[3] = {"date": DAYS[1], "outcome": "PARTIAL"}
    _w(lab / "session_ledger.jsonl", ledger)
    live = [_trade(DAYS[0], "ORB_5min", "QQQ", "long", "09:40", 5.0, h=A.LIVE_HASH["shares"]),
            _trade(DAYS[2], "ORB_5min", "QQQ", "long", "09:40", 5.0, h="deadbeef00000000")]
    ok, why = A.qualifying_sessions(lab, "shares", live)
    assert ok == [DAYS[0]]
    assert why[DAYS[1]] == "PARTIAL" and why[DAYS[2]] == "not under the live hash"


def test_a_session_outside_the_replay_window_is_excluded(tmp_path):
    lab = _lab(tmp_path, DAYS[:2])
    ok, why = A.qualifying_sessions(lab, "shares", [], replayed={DAYS[0]})
    assert ok == [DAYS[0]] and why[DAYS[1]] == "not in the replay window"


def test_a_late_start_disqualifies(tmp_path):
    lab = _lab(tmp_path, DAYS[:1], started="09:39:10")
    ok, why = A.qualifying_sessions(lab, "options", [])
    assert ok == [] and why[DAYS[0]] == "late or unknown start"


def test_fewer_than_fifteen_sessions_is_not_established():
    res = A.compare_arm("shares", [], [], DAYS[:14], [])
    assert res["verdict"] == "not established" and "14" in res["why"]


def _paired(n_days, live_pnl=10.0, replay_pnl=10.0, extra_replay=0):
    live, rep = [], []
    for i, d in enumerate(DAYS[:n_days]):
        p = live_pnl + i
        live.append(_trade(d, "ORB_5min", "QQQ", "long", "09:40", p, h=A.LIVE_HASH["shares"]))
        rep.append(_trade(d, "ORB_5min", "QQQ", "long", "09:40", replay_pnl + i))
        for k in range(extra_replay):
            rep.append(_trade(d, "Gap_Fade", "SPY", "short", f"10:{k:02d}", 1.0))
    return live, rep


def test_identical_days_pass_every_bar():
    live, rep = _paired(16)
    res = A.compare_arm("shares", live, rep, DAYS[:16], [])
    assert res["verdict"] == "PASS"
    assert res["checks"]["spearman"]["value"] == 1.0 and res["checks"]["median_gap_$"]["value"] == 0


def test_a_three_dollar_gap_passes_and_three_and_a_half_fails():
    live, rep = _paired(16, 10.0, 13.0)
    assert A.compare_arm("shares", live, rep, DAYS[:16], [])["checks"]["median_gap_$"]["pass"]
    live, rep = _paired(16, 10.0, 13.5)
    res = A.compare_arm("shares", live, rep, DAYS[:16], [])
    assert not res["checks"]["median_gap_$"]["pass"] and res["verdict"] == "FAIL"


def test_too_many_replay_trades_fail_the_count_ratio():
    live, rep = _paired(16, extra_replay=1)         # replay has 2x the trades
    res = A.compare_arm("shares", live, rep, DAYS[:16], [])
    assert res["checks"]["count_ratio"]["value"] == 2.0 and not res["checks"]["count_ratio"]["pass"]


def test_replay_trades_inside_a_suspend_window_are_dropped():
    live, rep = _paired(16, extra_replay=1)         # the extra at 10:00 falls inside the window
    win = [(f"{d}T09:55:00", f"{d}T10:05:00") for d in DAYS[:16]]
    res = A.compare_arm("shares", live, rep, DAYS[:16], win)
    assert res["checks"]["count_ratio"]["value"] == 1.0


def test_shares_scope_drops_symbols_live_did_not_trade():
    live, rep = _paired(16, extra_replay=1)         # extras are SPY Gap_Fade
    res = A.compare_arm("shares", live, rep, DAYS[:16], [], scope=lambda d: ({"QQQ"}, set()))
    assert res["checks"]["count_ratio"]["value"] == 1.0


def test_options_compare_atm_only_and_check_exit_mix():
    live, rep = [], []
    for i, d in enumerate(DAYS[:16]):
        for arm in ("ATM", "ATM-1"):
            live.append(_trade(d, "ORB_5min", "QQQ", "long", "09:40", 10.0 + i, arm=arm,
                               h=A.LIVE_HASH["options"], exit_reason="target"))
            rep.append(_trade(d, "ORB_5min", "QQQ", "long", "09:40", 10.0 + i, arm=arm,
                              exit_reason="stop" if arm == "ATM" else "target"))
    res = A.compare_arm("options", live, rep, DAYS[:16], [])
    assert res["n_live_trades"] == 16 and res["n_replay_trades"] == 16      # ATM only
    assert res["checks"]["exit_mix_worst_pp"]["value"] == 100.0 and res["verdict"] == "FAIL"


def test_one_arm_failing_is_reported_as_failing(tmp_path):
    lab = _lab(tmp_path, DAYS[:16])
    live_s, rep_s = _paired(16)
    _w(lab / "shares" / "trades.jsonl", live_s)
    (lab / "shares" / "FREEZE.json").write_text(json.dumps({"universe_changes": []}), encoding="utf-8")
    _w(lab / "trades.jsonl", [])
    root = tmp_path / "replay"
    _w(root / "shares" / "trades.jsonl", rep_s)
    _w(root / "options" / "trades.jsonl", [])
    for arm in ("shares", "options"):
        for d in DAYS[:16]:
            (root / arm / "daily").mkdir(parents=True, exist_ok=True)
            (root / arm / "daily" / f"{d}.json").write_text("{}", encoding="utf-8")
    import trade_analysis.live_lab.backfill as B
    orig = B._live_scope
    B._live_scope = lambda d: (set(), set())
    try:
        res = A.run(root, lab)
    finally:
        B._live_scope = orig
    assert res["arms"]["shares"]["verdict"] == "PASS"
    assert res["arms"]["options"]["verdict"] == "not established"    # no options trades/sessions
