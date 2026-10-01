"""REPLAY AUDIT -- why does the live record differ from a clean replay of the same day?

    python -m trade_analysis.live_lab.replay_audit            # both arms, every replayed day

`backfill.py` replays each session through the real runner with a feed that never drops.
`backfill.reconcile` pairs live trades with replay trades on the same signal. This module
explains every trade that did NOT pair, using only what the live lab itself recorded:

  replay_only (a clean session took it, live never did), first cause that applies:
    arm blocked       the live arm took no trade at all that day
    late start        the signal bar closed before the live runner could see bars
                      (runner start + LATE_OPEN_BLIND_MIN, incident_2026-09-24)
    host asleep       inside a `host_suspend` window (outages.jsonl)
    frozen            inside a `caught_up` window: entries are frozen by design
    feed down         inside a run of failed ticks / failed bar fetches
    stale bar refused the shares stale-bar guard refused that symbol's bar live
    quote too old     live's underlying quote failed the stale/future-quote guard in the
                      minute the signal was decided, so nothing could be entered
    degraded bar      a gap in live's admitted 1m bars, so that 5m bucket never formed
    live skipped: X   live saw the signal and skipped it (X = its skip_reason); usually a
                      daily cap already used by a trade the replay does not have
    no live signal    live was watching and its bars never produced it: first-print bars
                      differ from the vendor's revised bars, or state diverged earlier

  live_only (live took it, a clean session would not have):
    late entry        entered more than LATE_ENTRY_MIN after its signal bar (a batch of
                      missed bars admitted at once, pre-2026-09-28 catch-up)
    decided twice     the same signal traded more than once live (pre decided-once fix)
    replay skipped: X the replay saw it and skipped it (X = its skip_reason)
    no replay signal  history's revised bars never produced it

  matched, entered late: the same signal on both sides, but live filled it more than
    LATE_ENTRY_MIN after its bar (the blind-open batch), at a later price and often a
    different strike. Reported separately: the trade exists on both sides, its P&L does not
    mean the same thing.

It never edits the live record. Output: live_lab_backfill/<arm>/audit.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import backfill as B
from .feed import _parse_ts
from .store import read_signals

LAB = B.ROOT / "live_lab_data"
LATE_OPEN_BLIND_MIN = 6      # a runner started at T processed its first bar at ~T+6 min
LATE_ENTRY_MIN = 3
FEED_JOIN_SEC = 180          # failed ticks this close together are one outage
RTH_OPEN = dt.time(9, 30)


def _rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def live_start(day: str) -> dt.datetime | None:
    """When the live runners started that day: the ledger, else the session log."""
    for r in _rows(LAB / "session_ledger.jsonl"):
        if r.get("date") == day and r.get("started_et"):
            return dt.datetime.combine(dt.date.fromisoformat(day),
                                       dt.time.fromisoformat(r["started_et"]))
    log = LAB / "logs" / f"{day}.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "starting runner" in line:
                m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET", line)
                if m:
                    return dt.datetime.fromisoformat(m.group(1))
    return None


def blind_spans(lab: Path, day: str) -> list[tuple[dt.datetime, dt.datetime, str]]:
    """Every span on `day` when this live arm could not act on a new bar."""
    d = dt.date.fromisoformat(day)
    spans = []
    st = live_start(day)
    if st is not None and st.time() > dt.time(9, 25):
        spans.append((dt.datetime.combine(d, RTH_OPEN),
                      st + dt.timedelta(minutes=LATE_OPEN_BLIND_MIN), "late start"))
    for r in _rows(lab / "outages.jsonl"):
        if r.get("kind") == "host_suspend" and r.get("from_et", "")[:10] == day:
            spans.append((_parse_ts(r["from_et"]), _parse_ts(r["to_et"]), "host asleep"))
    for r in _rows(lab / "events.jsonl"):
        if r.get("kind") == "caught_up" and r.get("since", "")[:10] == day:
            spans.append((_parse_ts(r["since"]), _parse_ts(r["until"]), "frozen"))
    fails = sorted(_parse_ts(r["ts"]) for r in _rows(lab / "outages.jsonl")
                   if r.get("kind") in ("tick", "bars_failed") and r.get("ts", "")[:10] == day)
    run = []
    for t in fails + [None]:
        if run and (t is None or (t - run[-1]).total_seconds() > FEED_JOIN_SEC):
            spans.append((run[0] - dt.timedelta(minutes=1), run[-1], "feed down"))
            run = []
        if t is not None:
            run.append(t)
    return spans


def stale_refused(lab: Path, day: str) -> set[tuple[str, str]]:
    """(symbol, 'HH:MM') bars the live stale-bar guard refused to trade on."""
    out = set()
    for r in _rows(lab / "outages.jsonl"):
        if r.get("kind") == "stale_bar" and r.get("ts", "")[:10] == day:
            m = re.search(r"bar (\d{2}:\d{2})", r.get("detail", ""))
            if m:
                out.add((r.get("symbol"), m.group(1)))
    return out


def blind_minutes(spans, day: str) -> float:
    """Minutes of RTH covered by at least one span (overlaps counted once)."""
    d = dt.date.fromisoformat(day)
    lo, hi = dt.datetime.combine(d, RTH_OPEN), dt.datetime.combine(d, dt.time(16, 0))
    cut = sorted((max(a, lo), min(b, hi)) for a, b, _ in spans if b > lo and a < hi)
    total, end = 0.0, lo
    for a, b in cut:
        a = max(a, end)
        if b > a:
            total += (b - a).total_seconds()
            end = b
    return round(total / 60, 1)


def guard_hits(lab: Path, day: str) -> dict[str, list[tuple[str, dt.datetime]]]:
    """symbol -> [(kind, ts)] for quote-guard and degraded-bar records on `day`."""
    out = defaultdict(list)
    for r in _rows(lab / "outages.jsonl"):
        k = r.get("kind")
        if k in ("stale_quote", "future_quote", "degraded_bar") and r.get("ts", "")[:10] == day:
            sym = r.get("symbol") or (r.get("detail") or " ").split()[0]
            out[sym].append((k, _parse_ts(r["ts"])))
    return out


def skips(root: Path, day: str) -> dict[tuple, str]:
    """(setup, symbol, bar_ts) -> skip_reason, for SKIP rows on `day`."""
    out = {}
    for r in read_signals(root, day):
        if r.get("phase") == "SKIP" and (r.get("bar_ts") or "")[:10] == day:
            out.setdefault((r["setup_id"], r["symbol"], r["bar_ts"]), r.get("skip_reason"))
    return out


def why_replay_only(t: dict, ctx: dict) -> str:
    if ctx["live_trades_that_day"] == 0:
        return "arm blocked"
    decided = _parse_ts(t["entry_bar_ts"]) + dt.timedelta(minutes=1)
    for a, b, why in ctx["spans"]:
        if a <= decided <= b + dt.timedelta(minutes=1):
            return why
    if (t["symbol"], t["entry_bar_ts"][11:16]) in ctx["stale"]:
        return "stale bar refused"
    for kind, ts in ctx["guards"].get(t["symbol"], ()):
        if kind != "degraded_bar" and decided - dt.timedelta(seconds=5) <= ts                 <= decided + dt.timedelta(seconds=65):
            return "quote too old"
        if kind == "degraded_bar" and decided - dt.timedelta(minutes=10) <= ts                 <= decided + dt.timedelta(minutes=1):
            return "degraded bar"
    r = ctx["live_skips"].get((t["setup_id"], t["symbol"], t["entry_bar_ts"]))
    return f"live skipped: {r}" if r else "no live signal"


def entry_lag_min(t: dict) -> float:
    """Minutes between the signal bar closing and the fill."""
    return (_parse_ts(t["entry_ts"]) - _parse_ts(t["entry_bar_ts"])).total_seconds() / 60 - 1


def why_live_only(t: dict, ctx: dict) -> str:
    if entry_lag_min(t) > LATE_ENTRY_MIN:
        return "late entry"
    if ctx["live_keys"].get(B._signal_key(t), 0) > 1:
        return "decided twice"
    r = ctx["replay_skips"].get((t["setup_id"], t["symbol"], t["entry_bar_ts"]))
    return f"replay skipped: {r}" if r else "no replay signal"


def audit(arm: str) -> dict:
    lab = LAB if arm == "options" else LAB / "shares"
    out = B.OUT if arm == "options" else B.OUT_SHARES
    accepted = set(json.loads((lab / "FREEZE.json").read_text(encoding="utf-8"))
                   ["accepted_config_hashes"])
    live = [r for r in _rows(lab / "trades.jsonl") if r.get("config_hash") in accepted]
    back = _rows(out / "trades.jsonl")
    if arm == "options":
        live = [r for r in live if r.get("arm") == "ATM"]
        back = [r for r in back if r.get("arm") == "ATM"]
    days = sorted({r["entry_ts"][:10] for r in back})
    per_day, causes = [], defaultdict(lambda: {"n": 0, "net": 0.0})
    for d in days:
        lv = [r for r in live if r["entry_ts"][:10] == d]
        bk = [r for r in back if r["entry_ts"][:10] == d]
        if arm == "options":
            sc = [r for r in bk if d >= B.SIX_LIVE_FROM
                  or r["setup_id"] not in ("Six_Lines", "Six_Lines_NoCap")]
        else:
            syms, excl = B._live_scope(d)
            if not syms:
                continue          # the shares arm did not exist yet: nothing to audit
            sc = [r for r in bk if r["symbol"] in syms and r["setup_id"] not in excl]
        rec = B.reconcile(lv, sc, detail=True)
        ctx = {"live_trades_that_day": len(lv), "spans": blind_spans(lab, d),
               "stale": stale_refused(lab, d), "guards": guard_hits(lab, d),
               "live_skips": skips(lab, d), "replay_skips": skips(out, d),
               "live_keys": Counter(B._signal_key(r) for r in lv)}
        day_causes = Counter()
        for lvt, rpt in rec.pop("_pairs"):
            if entry_lag_min(lvt) > LATE_ENTRY_MIN:
                c = "matched / entered late live"
                causes[c]["n"] += 1; causes[c]["net"] += lvt["pnl_net"] - rpt["pnl_net"]
                day_causes[c] += 1
        for t in rec.pop("_replay_only"):
            c = "replay_only / " + why_replay_only(t, ctx)
            causes[c]["n"] += 1; causes[c]["net"] += t["pnl_net"]; day_causes[c] += 1
        for t in rec.pop("_live_only"):
            c = "live_only / " + why_live_only(t, ctx)
            causes[c]["n"] += 1; causes[c]["net"] += t["pnl_net"]; day_causes[c] += 1
        per_day.append({"day": d, "live_n": len(lv),
                        "live_net": round(sum(r["pnl_net"] for r in lv), 2),
                        "replay_n": len(sc), "replay_net": round(sum(r["pnl_net"] for r in sc), 2),
                        "bins": rec, "causes": dict(day_causes),
                        "blind_minutes": blind_minutes(ctx["spans"], d)})
    return {"arm": arm, "days": per_day,
            "causes": {k: {"n": v["n"], "net": round(v["net"], 2)}
                       for k, v in sorted(causes.items(), key=lambda kv: -kv[1]["n"])},
            "totals": B._recon_total([{"recon": x["bins"]} for x in per_day])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["options", "shares", "both"], default="both")
    a = ap.parse_args(argv)
    for arm in (["options", "shares"] if a.arm == "both" else [a.arm]):
        rep = audit(arm)
        out = B.OUT if arm == "options" else B.OUT_SHARES
        (out / "audit.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\n[{arm}] {len(rep['days'])} sessions audited -> {out / 'audit.json'}")
        for k, v in rep["totals"].items():
            print(f"  {k:12} n={v['n']:5}  live {v['live']:+10.2f}  replay {v['replay']:+10.2f}")
        for k, v in rep["causes"].items():
            print(f"  {k:48} n={v['n']:5}  net {v['net']:+10.2f}")
        print("  (for 'matched / entered late live', net is live minus replay on those pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
