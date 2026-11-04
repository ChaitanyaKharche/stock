"""Live-vs-replay acceptance, per arm, independently. research/news_arms/preregistration.md 14.2.

    python -m trade_analysis.news_arms.acceptance --replay live_lab_backfill/news_arms/accept

Qualifying sessions: ledger COLLECTED, clean start (started before 09:31, not late_start),
every live trade of that arm that day under the live config hash. Replay trades inside a
live host_suspend window are dropped; replay trades outside the live shares scope (universe,
Six_Lines before 2026-09-25) are dropped. Each arm is judged on its own; one arm's pass
never rescues the other.

Metrics and bars (14.2):
  shares   Spearman(session net) >= 0.90, median |live - replay| on twinned trades <= $3.00,
           replay/live trade count in [0.9, 1.3]
  options  Spearman >= 0.85, median gap on twinned ATM trades <= $8.00, count ratio in
           [0.8, 1.5], exit-reason mix differing by <= 15 pp on any reason
  both     at least 15 qualifying sessions, else "not established"
Writes research/news_arms/acceptance.json. Never edits the live record.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from ..live_lab import backfill as B

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "live_lab_data"
LIVE_HASH = {"options": "ccd00ac71f8243e9", "shares": "7c6414fc549315f3"}
MIN_SESSIONS = 15
BARS = {
    "shares": {"spearman": 0.90, "gap": 3.00, "ratio": (0.9, 1.3), "exit_mix_pp": None},
    "options": {"spearman": 0.85, "gap": 8.00, "ratio": (0.8, 1.5), "exit_mix_pp": 15.0},
}
CLEAN_START = "09:31:00"


def _rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def qualifying_sessions(lab: Path, arm: str, live: list[dict]) -> tuple[list[str], dict]:
    """Dates that meet 14.2, with the reason each other session was excluded."""
    ledger = _rows(lab / "session_ledger.jsonl")
    opened = {r["date"]: r for r in ledger if r.get("outcome") == "OPENED"}
    final: dict = {}
    for r in ledger:
        if r.get("outcome") not in ("OPENED", "NOT_A_SESSION"):
            final[r["date"]] = r
    by_day = defaultdict(list)
    for t in live:
        by_day[t["entry_ts"][:10]].append(t)
    ok, why = [], {}
    for day, fin in sorted(final.items()):
        op = opened.get(day)
        if fin["outcome"] != "COLLECTED":
            why[day] = fin["outcome"]
        elif op is None or op.get("late_start") or (op.get("started_et") or "99") >= CLEAN_START:
            why[day] = "late or unknown start"
        elif arm not in (op.get("arms") or []):
            why[day] = "arm not run"
        elif any(t.get("config_hash") != LIVE_HASH[arm] for t in by_day.get(day, [])):
            why[day] = "not under the live hash"
        else:
            ok.append(day)
    return ok, why


def suspend_windows(lab: Path) -> list[tuple[str, str]]:
    return [(r["from_et"], r["to_et"]) for r in _rows(lab / "outages.jsonl")
            if r.get("kind") == "host_suspend" and r.get("from_et") and r.get("to_et")]


def _inside(ts: str, windows) -> bool:
    return any(a <= ts <= b for a, b in windows)


def compare_arm(arm: str, live: list[dict], replay: list[dict], days: list[str],
                windows, scope=None) -> dict:
    """The 14.2 metrics for one arm over `days`. `scope(day) -> (symbols, excluded_setups)`
    trims replay trades to what live could have traded (shares only)."""
    bars = BARS[arm]
    nets_l, nets_r, gaps, exits_l, exits_r = [], [], [], Counter(), Counter()
    n_live = n_rep = 0
    per_day = []
    for day in days:
        lv = [t for t in live if t["entry_ts"][:10] == day]
        rp = [t for t in replay if t["entry_ts"][:10] == day
              and not _inside(t["entry_ts"], windows)]
        if scope is not None:
            syms, excl = scope(day)
            rp = [t for t in rp if (not syms or t["symbol"] in syms) and t["setup_id"] not in excl]
        if arm == "options":
            lv = [t for t in lv if t.get("arm") == "ATM"]
            rp = [t for t in rp if t.get("arm") == "ATM"]
        rec = B.reconcile(lv, rp, detail=True)
        for a, b in rec["_pairs"]:
            gaps.append(abs((a.get("pnl_net") or 0.0) - (b.get("pnl_net") or 0.0)))
        nets_l.append(sum(t.get("pnl_net") or 0.0 for t in lv))
        nets_r.append(sum(t.get("pnl_net") or 0.0 for t in rp))
        exits_l.update(t.get("exit_reason") for t in lv)
        exits_r.update(t.get("exit_reason") for t in rp)
        n_live += len(lv)
        n_rep += len(rp)
        per_day.append({"day": day, "live_net": round(nets_l[-1], 2), "replay_net": round(nets_r[-1], 2),
                        "live_n": len(lv), "replay_n": len(rp), "twinned": len(rec["_pairs"]),
                        "live_only": rec["live_only"]["n"], "replay_only": rec["replay_only"]["n"]})
    res = {"arm": arm, "n_sessions": len(days), "n_live_trades": n_live, "n_replay_trades": n_rep,
           "per_day": per_day, "bars": bars}
    if len(days) < MIN_SESSIONS:
        res["verdict"] = "not established"
        res["why"] = f"{len(days)} qualifying sessions < {MIN_SESSIONS}"
        return res
    if n_live == 0 or not gaps:
        res["verdict"] = "not established"
        res["why"] = "no live trades or no twinned trades on the qualifying sessions"
        return res
    from scipy.stats import spearmanr
    rho = float(spearmanr(nets_l, nets_r).correlation) if len(set(nets_l)) > 1 else float("nan")
    gap = float(st.median(gaps)) if gaps else float("nan")
    ratio = n_rep / n_live if n_live else float("nan")
    tot_l, tot_r = sum(exits_l.values()) or 1, sum(exits_r.values()) or 1
    mix = {k: round(100.0 * (exits_r[k] / tot_r - exits_l[k] / tot_l), 1)
           for k in set(exits_l) | set(exits_r)}
    worst_pp = max((abs(v) for v in mix.values()), default=0.0)
    checks = {"spearman": (round(rho, 3), rho >= bars["spearman"]),
              "median_gap_$": (round(gap, 2), gap <= bars["gap"]),
              "count_ratio": (round(ratio, 3), bars["ratio"][0] <= ratio <= bars["ratio"][1])}
    if bars["exit_mix_pp"] is not None:
        checks["exit_mix_worst_pp"] = (round(worst_pp, 1), worst_pp <= bars["exit_mix_pp"])
    res.update({"checks": {k: {"value": v, "pass": bool(p)} for k, (v, p) in checks.items()},
                "exit_mix_pp": mix, "twinned_trades": len(gaps),
                "verdict": "PASS" if all(p for _, p in checks.values()) else "FAIL"})
    return res


def run(replay_root: Path, lab: Path = LAB) -> dict:
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "arms": {}}
    windows = suspend_windows(lab)
    for arm, sub, key in (("options", "", "position_id"), ("shares", "shares", "signal_id")):
        live = B._corrected(lab / sub if sub else lab, _rows((lab / sub if sub else lab) / "trades.jsonl"), key)
        live = [t for t in live if t.get("pnl_net") is not None]
        replay = [t for t in _rows(replay_root / arm / "trades.jsonl") if t.get("pnl_net") is not None]
        days, why = qualifying_sessions(lab, arm, live)
        scope = B._live_scope if arm == "shares" else None
        res = compare_arm(arm, live, replay, days, windows, scope)
        res["excluded_sessions"] = why
        res["replay_dir"] = str(replay_root / arm)
        bf = replay_root / arm / "BACKFILL.json"
        res["replay_meta"] = json.loads(bf.read_text(encoding="utf-8")) if bf.exists() else None
        out["arms"][arm] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replay", default=str(ROOT / "live_lab_backfill" / "news_arms" / "accept"))
    ap.add_argument("--lab-dir", default=str(LAB))
    ap.add_argument("--out", default=str(ROOT / "research" / "news_arms" / "acceptance.json"))
    a = ap.parse_args(argv)
    res = run(Path(a.replay), Path(a.lab_dir))
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for arm, r in res["arms"].items():
        print(f"{arm:<8} sessions {r['n_sessions']:>3}  live {r['n_live_trades']:>4} trades  "
              f"replay {r['n_replay_trades']:>4}  -> {r['verdict']}")
        for k, v in (r.get("checks") or {}).items():
            print(f"    {k:<20} {v['value']:>8}  {'ok' if v['pass'] else 'FAIL'}")
        if r.get("why"):
            print(f"    {r['why']}")
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
