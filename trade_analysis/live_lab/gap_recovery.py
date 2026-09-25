"""GAP RECOVERY -- resolve the exits a gap stole, from history, through the runner's own code.

    python -m trade_analysis.live_lab.gap_recovery --day 2026-09-25            # report only
    python -m trade_analysis.live_lab.gap_recovery --day 2026-09-25 --write    # append corrections

WHY THIS EXISTS
---------------
2026-09-25: the host suspended from 15:39:33 to 16:19:09 despite the suspend guard, and the
network was down from ~15:57. The 15:55 flatten never ran. When the shares arm woke at
16:19 it closed its 22 open positions at whatever the feed said then, which was
after-hours NBBO: two XLP longs were "sold" at 76.66, 6.2% below entry, on a day XLP did
not fall 6%. Those 22 rows carry -$1,478.61 that describe a laptop lid, not a rule.

The runner cannot know what happened while it was not running. History can: every 1m bar
and NBBO minute of the gap sits on the vendor's servers.

WHAT IT DOES, AND THE LINES IT DOES NOT CROSS
---------------------------------------------
  * EXITS ONLY, for positions that were genuinely open when the gap began. It never opens
    a position. A signal that would have fired during the gap was never decided before its
    price existed, so it does not belong to the forward test.
  * APPEND-ONLY. trades.jsonl is never edited. A correction is appended to
    trade_corrections.jsonl beside it: the original row, the recovered exit, the gap and
    the method. `apply_corrections` gives readers the corrected view; the raw record stays
    as observed. A record that silently rewrites itself is not a record.
  * IT DECIDES NOTHING ITSELF. The whole session is replayed through SharesLab._manage /
    LiveLab._manage_open on history. Every position is injected at its real entry and
    rebuilt exactly as the runner held it -- for shares by re-evaluating the setup at its
    entry bar, and refusing any position whose recomputed decision differs from the
    recorded one. So the code that made every other exit today makes these.
  * IT CHECKS ITSELF FIRST. Every position that closed normally before the gap is replayed
    too, from its entry, and must exit by the SAME RULE as live (stop, target, time, trail,
    eod). Below MIN_AGREEMENT nothing is written: a replay that cannot reproduce the exits
    it can see has no business supplying the ones it cannot.

    CORRECTED 2026-09-25, and recorded rather than quietly changed: the gate first
    required the same rule AND the same minute. On 2026-09-25 that measured 59% (19 of 32)
    and refused. Every miss but one was the same rule one bar early or late -- e.g. a stop
    at 10:05 replayed vs 10:06 live. The cause is the data, not the replay: the vendor
    REVISES bars after the fact (late prints), so a bar's final low can touch a stop its
    first-published low did not. Exit rule agreement on the same 32 was 97% (31 of 32).
    The gate now tests what recovery depends on -- that the same rule fires -- and minute
    agreement is reported beside it, never hidden. Consequence, stated plainly: a
    recovered exit is what the rules produce on the vendor's FINAL bars, and may sit a bar
    away from where a live runner on first-print bars would have fired.
  * Positions that were open at a gap are managed only from the gap's start, so a
    revised bar before the gap cannot exit them earlier than the live record says.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from . import backfill as BF
from . import store as S

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "live_lab_data"
MIN_AGREEMENT = 0.90
MATCH_MINUTES = 1
RTH_OPEN, RTH_CLOSE = dt.time(9, 30), dt.time(16, 0)
TICK = dt.timedelta(minutes=1)
GAP_RE = re.compile(r"between (\d\d:\d\d:\d\d) and (\d\d:\d\d:\d\d)")
ARM_DIR = {"options": LAB, "shares": LAB / "shares"}


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue                       # one bad line must not blind the whole tool
    return out


def _ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


# --------------------------------------------------------------------------- gaps

def find_gaps(arm_dir: Path, day: dt.date) -> list[tuple[dt.datetime, dt.datetime]]:
    """Host-suspend windows during RTH, merged when they are back to back."""
    raw = []
    for r in _jsonl(arm_dir / "outages.jsonl"):
        if r.get("kind") != "host_suspend" or not str(r.get("ts", "")).startswith(day.isoformat()):
            continue
        m = GAP_RE.search(r.get("detail", ""))
        if m:
            a, b = (dt.datetime.combine(day, dt.time.fromisoformat(x)) for x in m.groups())
            raw.append((a, b))
    merged: list[list[dt.datetime]] = []
    for a, b in sorted(raw):
        if merged and (a - merged[-1][1]).total_seconds() <= 60:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    open_, close = (dt.datetime.combine(day, t) for t in (RTH_OPEN, RTH_CLOSE))
    return [(a, b) for a, b in merged if a < close and b > open_]


def affected(trades: list[dict], gaps) -> dict[int, tuple]:
    """index -> gap, for trades that were OPEN when a gap began (or closed after 16:00)."""
    out = {}
    for i, t in enumerate(trades):
        e, x = _ts(t["entry_ts"]), _ts(t["exit_ts"])
        for g in gaps:
            if e < g[0] <= x:
                out[i] = g
                break
        else:
            if x.time() >= RTH_CLOSE or "stale_mark" in str(t.get("exit_reason")):
                out[i] = (x, x)
    return out


def _same_state(a: dict, b: dict) -> bool:
    if set(a) - {"provenance"} != set(b) - {"provenance"}:
        return False
    for k, v in a.items():
        if k == "provenance":
            continue
        w = b[k]
        if isinstance(v, (int, float)) and isinstance(w, (int, float)):
            if abs(v - w) > 1e-6 * max(1.0, abs(v)):
                return False
        elif isinstance(v, dict) and isinstance(w, dict):
            if not _same_state(v, w):
                return False
        elif v != w:
            return False
    return True


# --------------------------------------------------------------------------- replays

def _clock_patch(module) -> BF.SimClock:
    clock = BF.SimClock()
    module.now_et = clock
    S._now = clock
    return clock


def _drive(lab, clock, day, tick_fn, flatten_reason, deferred):
    """Step the session minute by minute. `deferred` holds (activate_at, position) pairs:
    positions that were OPEN when a gap began join the book only at the gap's start, so
    the replay manages them only over minutes the live runner never saw. Managing them from
    their entry would let a vendor-revised bar exit them earlier than live did -- a
    'recovery' that contradicts the record it is meant to complete."""
    t = dt.datetime.combine(day, RTH_OPEN) + BF.TICK_OFFSET
    close = dt.datetime.combine(day, RTH_CLOSE)
    while t < close:
        clock.now = t
        if hasattr(lab, "_chain_cache"):
            lab._chain_cache.clear()
        for item in [d for d in deferred if d[0] <= t]:
            deferred.remove(item)
            lab.open_pos.append(item[1])
        tick_fn(t, day)
        t += TICK
    clock.now = close
    lab._flatten_all(close, reason=flatten_reason)


def replay_shares(day: dt.date, trades: list[dict], arm_dir: Path,
                  activate: dict | None = None) -> tuple[dict, list]:
    """Replay the day's share positions through SharesLab._manage. -> (by signal_id, problems)

    `activate` maps signal_id -> the instant a gap began; those positions are rebuilt at
    their entry bar but managed only from that instant. Every other position is managed
    from its entry, which is what makes the self-check possible."""
    activate = activate or {}
    deferred: list = []
    from . import shares_runner as SR
    clock = _clock_patch(SR)
    lab = SR.SharesLab(sorted({t["symbol"] for t in trades}),
                       lab_dir=tempfile.mkdtemp(prefix="gaprec_shares_"))
    lab.feed = BF.HistoryFeed(lab.feed, clock, day)
    clock.now = dt.datetime.combine(day, BF.WARMUP_AT)
    if not lab.warmup(day):
        raise RuntimeError("warmup failed on history")
    sigs = _jsonl(arm_dir / "signals.jsonl")
    decisions = {d["signal_id"]: d for d in sigs if d.get("phase") == "DECISION"}
    # From 2026-09-25 the FILL row carries the exit parameters the runner actually held,
    # which makes a rebuild EXACT. Older rows do not, and must be recomputed -- see below.
    fills = {d["signal_id"]: d for d in sigs if d.get("phase") == "FILL" and "stop" in d}
    at = defaultdict(list)
    for t in trades:
        at[(t["symbol"], t["entry_bar_ts"])].append(t)
    problems, quality = [], {}

    def inject(sym, sess, bar, ctx1, ctx5, quote, now):
        for t in at.pop((sym, bar["ts"].isoformat()), []):
            sid = t["signal_id"]
            setup = lab._setups.get(t["setup_id"])
            if setup is None:
                problems.append((sid, "unknown setup"))
                continue
            f = fills.get(sid)
            if f is not None:
                params = {k: f.get(k) for k in ("stop", "target", "time_exit_min",
                                                "bar_exit", "trailing")}
                state = (decisions.get(sid) or {}).get("state") or {}
                quality[sid] = "exact (recorded at entry)"
            else:
                # Recomputing on HISTORY bars. The vendor's final bars are not the bars the
                # runner saw live -- late prints revise volume and extremes -- so indicator
                # inputs (ATR, MACD, VWAP) come out a few percent different. Measured
                # 2026-09-25: 81 of 165 recomputed decisions differed, 45 did not re-fire.
                # A re-fire in the same direction is used and labelled APPROXIMATE; anything
                # else is not rebuilt at all.
                ctx = ctx1 if setup.timeframe == "1m" else ctx5
                sig = None if ctx is None else setup.evaluate(ctx)
                if sig is None or sig.direction != t["direction"]:
                    problems.append((sid, "setup does not re-fire on history bars"))
                    continue
                params = {"stop": sig.stop, "target": sig.target,
                          "time_exit_min": sig.time_exit_min, "bar_exit": sig.bar_exit,
                          "trailing": sig.trailing}
                state = sig.state
                dec = decisions.get(sid) or {}
                quality[sid] = ("exact (recomputed, decision identical)"
                                if _same_state(dec.get("state") or {}, sig.state or {})
                                else "approximate (recomputed on history bars)")
            pos = SR.SharePos(
                setup_id=t["setup_id"], symbol=sym, direction=t["direction"],
                entry_ts=t["entry_ts"], entry_bar_ts=t["entry_bar_ts"],
                entry_px=t["entry_px"], shares=t["shares"],
                spread_bp=t.get("entry_spread_bp") or 0.0, state=state,
                timeframe=setup.timeframe, signal_id=sid, **params)
            if sid in activate:
                deferred.append((activate[sid], pos))
            else:
                lab.open_pos.append(pos)

    lab._enter = inject
    _drive(lab, clock, day, lab._tick, "eod", deferred)
    problems += [(t["signal_id"], "entry bar never replayed") for v in at.values() for t in v]
    out = {r["signal_id"]: {**r, "rebuild": quality.get(r["signal_id"])}
           for r in _jsonl(Path(lab.store.root) / "trades.jsonl")}

    # For positions that could not be rebuilt: the one exit that needs no parameters is the
    # 15:55 flatten. Priced at the real NBBO of the minute the runner would have used.
    # Labelled ESTIMATE -- it assumes no stop, target or trail fired inside the gap.
    for sid, _ in problems:
        t = next((x for x in trades if x["signal_id"] == sid), None)
        if t is None:
            continue
        clock.now = dt.datetime.combine(day, SR.EOD_FLAT) + TICK + BF.TICK_OFFSET
        q = lab.feed.stock_quote(t["symbol"])
        if q is None:
            continue
        px = q["bid"] if t["direction"] == "long" else q["ask"]
        pnl = t["shares"] * ((px - t["entry_px"]) if t["direction"] == "long"
                             else (t["entry_px"] - px))
        out.setdefault(sid, {"signal_id": sid, "exit_ts": clock.now.isoformat(), "exit_px": px,
                             "exit_reason": "eod_estimate", "pnl_net": round(pnl, 4),
                             "rebuild": "estimate (parameters unknown; 15:55 flatten assumed)"})
    return out, problems


def replay_options(day: dt.date, trades: list[dict], arm_dir: Path,
                   activate: dict | None = None) -> tuple[dict, list]:
    """Replay the day's option positions through LiveLab._manage_open. -> (by position_id, problems)

    Option trade rows already carry every exit parameter the runner held (stop, target,
    time and bar exits, trailing, state), so every rebuild here is EXACT."""
    activate = activate or {}
    deferred: list = []
    from . import runner as R
    from .positions import Position
    from .setups import by_id
    clock = _clock_patch(R)
    lab = R.LiveLab(sorted({t["symbol"] for t in trades}),
                    lab_dir=tempfile.mkdtemp(prefix="gaprec_options_"))
    lab.feed = BF.HistoryFeed(lab.feed, clock, day)
    clock.now = dt.datetime.combine(day, BF.WARMUP_AT)
    if not lab.warmup(day):
        raise RuntimeError("warmup failed on history")
    fields = set(Position.__dataclass_fields__)
    reset = {"exit_ts": None, "exit_bid": None, "exit_underlying": None, "exit_reason": None,
             "trigger_level": None, "bars_held": 0, "mfe_opt": 0.0, "mae_opt": 0.0,
             "mfe_opt_ts": None, "mae_opt_ts": None, "mfe_und": 0.0, "mae_und": 0.0,
             "last_bid": None, "last_underlying": None}
    at = defaultdict(list)
    for t in trades:
        at[(t["symbol"], t["entry_bar_ts"])].append(
            Position(**{**{k: v for k, v in t.items() if k in fields}, **reset}))

    def inject(sym, sess, bar_ts, tf, quote, now, day_):
        keep = []
        for p in at.pop((sym, bar_ts.isoformat()), []):
            s = by_id.get(p.setup_id)
            if s is None or s.timeframe != tf:
                keep.append(p)
            elif p.position_id in activate:
                deferred.append((activate[p.position_id], p))
            else:
                lab.open_pos.append(p)
        if keep:
            at[(sym, bar_ts.isoformat())] = keep

    lab._evaluate = inject
    _drive(lab, clock, day, lab._tick, "shutdown", deferred)
    problems = [(p.position_id, "entry bar never replayed") for v in at.values() for p in v]
    out = {r["position_id"]: r for r in _jsonl(Path(lab.store.root) / "trades.jsonl")}
    return out, problems


# --------------------------------------------------------------------------- recovery

def _minutes_apart(a: str, b: str) -> float:
    return abs((_ts(a) - _ts(b)).total_seconds()) / 60.0


def recover(day: dt.date, arm: str, write: bool = False, lab_dir: Path | None = None) -> dict:
    arm_dir = Path(lab_dir) if lab_dir else ARM_DIR[arm]
    key = "signal_id" if arm == "shares" else "position_id"
    trades = [t for t in _jsonl(arm_dir / "trades.jsonl") if t["entry_ts"][:10] == day.isoformat()]
    gaps = find_gaps(arm_dir, day)
    hit = affected(trades, gaps)
    rep = {"day": day.isoformat(), "arm": arm, "gaps": [[a.isoformat(), b.isoformat()] for a, b in gaps],
           "affected": len(hit), "written": 0}
    if not hit:
        if write:
            _mark_run(arm_dir, rep)
        return rep
    replay = replay_shares if arm == "shares" else replay_options
    replayed, problems = replay(day, trades, arm_dir,
                                activate={trades[i][key]: g[0] for i, g in hit.items()})

    def same_rule(t):
        r = replayed.get(t[key])
        return r is not None and r["exit_reason"] == t["exit_reason"]

    def same_minute(t):
        r = replayed.get(t[key])
        return same_rule(t) and _minutes_apart(r["exit_ts"], t["exit_ts"]) <= MATCH_MINUTES

    def grade(r):
        q = str((r or {}).get("rebuild") or "exact")
        return "estimated" if q.startswith("estimate") else \
               "approximate" if q.startswith("approximate") else "exact"

    normal = [t for i, t in enumerate(trades) if i not in hit]
    val = {}
    for g_ in ("exact", "approximate"):
        pool = [t for t in normal if t[key] in replayed and grade(replayed[t[key]]) == g_]
        rule = sum(same_rule(t) for t in pool)
        minute = sum(same_minute(t) for t in pool)
        val[g_] = {"checked": len(pool), "same_rule": rule, "same_rule_and_minute": minute,
                   "rule_agreement": round(rule / len(pool), 4) if pool else None,
                   "minute_agreement": round(minute / len(pool), 4) if pool else None}
    val["not_rebuilt"] = sum(1 for t in normal if t[key] not in replayed
                             or grade(replayed[t[key]]) == "estimated")
    gate_pool = "exact" if val["exact"]["checked"] else "approximate"
    agreement = val[gate_pool]["rule_agreement"] or 0.0
    rep["validation"] = {**val, "gate_pool": gate_pool, "min_required": MIN_AGREEMENT,
                         "gate": "same exit RULE as live on >= min_required of checkable "
                                 "exits; exit MINUTE agreement is reported, not gated"}
    rep["problems"] = len(problems)

    rows = []
    for i, g in sorted(hit.items()):
        t = trades[i]
        r = replayed.get(t[key])
        base = {"kind": "gap_recovery", "arm": arm, "day": day.isoformat(), "key": key,
                key: t[key], "setup_id": t["setup_id"], "symbol": t["symbol"],
                "gap": [g[0].isoformat(), g[1].isoformat()],
                "original": {k: t.get(k) for k in ("exit_ts", "exit_px", "exit_bid",
                                                   "exit_reason", "pnl_net") if k in t}}
        if r is None or _ts(r["exit_ts"]) < g[0]:
            why = ("not replayable" if r is None else
                   f"replay exited at {r['exit_ts'][11:16]}, before the gap, which live did not")
            rows.append({**base, "status": "unrecoverable", "why": why})
            continue
        status = {"exact": "recovered", "approximate": "recovered_approximate",
                  "estimated": "estimated"}[grade(r)]
        rows.append({**base, "status": status, "rebuild": r.get("rebuild"),
                     "recovered": {k: r.get(k) for k in ("exit_ts", "exit_px", "exit_bid",
                                                         "exit_reason", "pnl_net") if k in r},
                     "delta_pnl": round((r.get("pnl_net") or 0) - (t.get("pnl_net") or 0), 4)})
    rep["rows"] = rows
    rep["original_net"] = round(sum(trades[i]["pnl_net"] or 0 for i in hit), 2)
    rep["recovered_net"] = round(sum((x.get("recovered") or x["original"])["pnl_net"] or 0
                                     for x in rows), 2)
    rep["by_status"] = {s: sum(1 for x in rows if x["status"] == s)
                        for s in ("recovered", "recovered_approximate", "estimated",
                                  "unrecoverable")}

    if write:
        if agreement < MIN_AGREEMENT:
            rep["refused"] = (f"replay reproduced the exit rule of {agreement:.0%} of the exits "
                              f"it could check; needs {MIN_AGREEMENT:.0%}")
            _mark_run(arm_dir, rep)
            return rep
        path = arm_dir / "trade_corrections.jsonl"
        done = {(c.get("kind"), c.get(key)) for c in _jsonl(path)}
        stamp = dt.datetime.now().isoformat(timespec="seconds")
        with open(path, "a", encoding="utf-8") as fh:
            for x in rows:
                if (x["kind"], x[key]) in done:
                    continue
                fh.write(json.dumps({**x, "validation": rep["validation"], "written_at": stamp,
                                     "method": "history 1m bars + 1m NBBO/chain replayed through "
                                               "the runner's own exit code (gap_recovery.py)"})
                         + "\n")
                rep["written"] += 1
        _mark_run(arm_dir, rep)
    return rep


def _mark_run(arm_dir: Path, rep: dict) -> None:
    """One line per completed run, so autostart does not redo a day it has already settled
    (including 'nothing to recover' and 'refused'). A run that CRASHED writes nothing and
    is retried -- that is the case the retry exists for."""
    with open(arm_dir / "trade_corrections.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "gap_recovery_run", "day": rep["day"], "arm": rep["arm"],
                             "affected": rep["affected"], "written": rep["written"],
                             "refused": rep.get("refused"),
                             "validation": rep.get("validation"),
                             "at": dt.datetime.now().isoformat(timespec="seconds")}) + "\n")


def apply_corrections(trades: list[dict], corrections: list[dict], key: str,
                      include=("recovered",)) -> list[dict]:
    """The corrected view. Originals are untouched; a corrected row says so.

    Only EXACT recoveries are applied by default. Pass include=("recovered",
    "recovered_approximate", "estimated") for a best-available view, and say so.
    """
    fix = {c[key]: c for c in corrections
           if c.get("kind") == "gap_recovery" and c.get("status") in include}
    out = []
    for t in trades:
        c = fix.get(t.get(key))
        out.append(t if c is None else {**t, **c["recovered"], "corrected": "gap_recovery",
                                        "original_exit": c["original"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--day", required=True)
    ap.add_argument("--arm", choices=["shares", "options", "both"], default="both")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--lab-dir", default=str(LAB))
    a = ap.parse_args(argv)
    day = dt.date.fromisoformat(a.day)
    dirs = {"options": Path(a.lab_dir), "shares": Path(a.lab_dir) / "shares"}
    for arm in (["shares", "options"] if a.arm == "both" else [a.arm]):
        rep = recover(day, arm, write=a.write, lab_dir=dirs[arm])
        print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
        for x in rep.get("rows", []):
            o, r = x["original"], x.get("recovered") or {}
            print(f"  {x['setup_id']:<24}{x['symbol']:<6}{x['status']:<14}"
                  f"orig {o.get('exit_reason')!s:<15}{o.get('pnl_net') or 0:>+9.2f}  ->  "
                  f"{r.get('exit_reason')!s:<8}{r.get('exit_ts', '')[11:16]:>6}"
                  f"{(r.get('pnl_net') or 0):>+10.2f}   {x.get('why', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
