"""STRANDED POSITIONS -- the exits of positions a session never closed, from history.

    python -m trade_analysis.live_lab.stranded --before 2026-10-07            # report only
    python -m trade_analysis.live_lab.stranded --before 2026-10-07 --write    # append them

WHY THIS EXISTS
---------------
2026-10-05 and 2026-10-06: the phone hotspot dropped at 14:08 and 12:24 ET and did not come
back until after the close. Both runners froze (catchup.py), history stayed unreachable, the
15:55 flatten never ran, and the 16:10 backstop terminated them. Every position still open --
65 on Monday, 69 on Tuesday -- got NO exit row at all. gap_recovery.py could not help: it
re-prices rows that exist, and these have none. They survive only in the runner's checkpoint
(positions_open.json, moved to recovery_archive/ when the next session starts).

The hole is not random. It deletes exactly the positions still alive late in the day; on
those two days that was +$2,279, mostly winners held to the close.

WHAT IT DOES, AND THE LINES IT DOES NOT CROSS
---------------------------------------------
  * EXITS ONLY, for checkpoint positions that have no trade row. It never opens anything.
  * Each position is rebuilt from the checkpoint -- the very object the runner held, stop,
    target and trail state included -- and managed on history only from the minute after
    `managed_through`, the last minute the runner actually managed
    (store.save_open_positions), through the runner's own exit code. The same method as
    catchup.py, run later.
  * IT CHECKS ITSELF FIRST, with gap_recovery's gate (`validate`): that session's recorded
    trades are replayed in the same pass from their entries and must exit by the same rule
    as live on >= MIN_AGREEMENT of them. Below that no replayed exit is written.
  * APPENDED AND FLAGGED. Rows go to trades.jsonl exactly as the runner writes them, plus
    "exit_mode": "reconstructed_from_history", "reconstructed": true, and a
    "reconstruction" block naming the checkpoint, the window and the self-check. One
    filter excludes them; nothing already in the record is edited.
  * Today's checkpoint is the LIVE book until the session is over and is never touched
    before SETTLED_AT.
  * Idempotent: a position that already has a row is skipped, and a session whose
    self-check failed is recorded in stranded_recovery.jsonl and never replayed again.

CORRECTED 2026-10-21, and recorded rather than quietly changed
--------------------------------------------------------------
"A checkpoint without `managed_through` is reported and never resolved" left the 2026-09-08
and 09-10 shares books -- 19 and 32 positions -- open in the record for six weeks, while
autostart logged them as "left for a human" up to 26 times a day. Three changes:

  * saved_at is still never trusted. Instead the SHARES arm's own record bounds the last
    minute it managed (`record_anchor`). shares_runner._tick skips BOTH managing and
    entering on a stale or missing quote, so a signal row proves the book was managed on
    the tick that wrote it, through that row's bar. The latest row is used only when the
    trace is dense right up to it (TRACE_MIN of the TRACE_WINDOW minutes before it carry a
    row), so the silence after it is the runner stopping rather than a quiet minute; and a
    symbol whose quotes or bars failed inside that window is not anchored at all. 09-10 is
    why saved_at fails: rows every minute to 15:19, then none, and saved_at says 15:42 --
    22 minutes of stops nobody checked would have been skipped.
    The OPTIONS arm evaluates setups -- and writes SKIP and DECISION rows -- even on a tick
    where its quote was stale and _manage_open returned early (2026-10-21 wrote 95 such
    decisions while managing nothing). Its rows prove nothing, so an options checkpoint
    without managed_through is never anchored this way.
  * NOTHING IS LEFT OPEN. A position the replay cannot settle -- no provable anchor, a
    failed self-check, or no replayed exit -- is closed at the 15:55 flatten priced from the
    history NBBO, the one exit that needs no parameters (gap_recovery's estimate). Flagged
    "estimate": true, exit_reason "eod_estimate", exit_mode ESTIMATE_MODE: it assumes no
    stop, target or trail fired after the last managed minute. A position with no history
    price at all stays open and is retried at the next window.
  * A shares row carries the config hash its session ran under. The replay used to stamp
    the CURRENT config's hash on every exit it rebuilt, so a September position would have
    been filed under a config that did not exist when it was opened.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

from . import gap_recovery as G
from .clock import now_et

EXIT_MODE = "reconstructed_from_history"
ESTIMATE_MODE = "estimated_from_history"
RUNS = "stranded_recovery.jsonl"
SETTLED_AT = dt.time(16, 5)     # today's checkpoint stops being the live book after this
ARMS = {"options": ("", "position_id"), "shares": ("shares", "signal_id")}
METHOD = ("checkpoint position managed from the minute after managed_through on history 1m "
          "bars + 1m NBBO/chain, through the runner's own exit code (stranded.py)")
ESTIMATE_METHOD = ("the 15:55 flatten priced from the history 1m NBBO (options: the contract's "
                   "1m bid) at the minute the runner would have used; assumes no stop, target "
                   "or trail fired after the last managed minute (stranded.py)")
TRACE_WINDOW = 10     # minutes before a record anchor that are checked for rows
TRACE_MIN = 9         # how many of them must carry one for the silence after it to mean anything
QUOTE_MISS = {"stale_quote", "future_quote", "quote_failed", "bars_failed"}


def _arm_dir(lab_dir, arm) -> Path:
    sub = ARMS[arm][0]
    return Path(lab_dir) / sub if sub else Path(lab_dir)


def _checkpoints(arm_dir: Path):
    for p in [arm_dir / "positions_open.json",
              *sorted((arm_dir / "recovery_archive").glob("positions_open_*.json"))]:
        try:
            yield p, json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue


def _ident(p: dict) -> tuple:
    return (p.get("setup_id"), p.get("symbol"), str(p.get("entry_ts")))


def _outage_symbol(r: dict) -> str | None:
    # shares_runner's stale_quote carries the symbol only as the detail's first word
    return r.get("symbol") or (str(r.get("detail") or "").split() or [None])[0]


def record_anchor(arm_dir: Path, day: str, blob: dict) -> dict:
    """The last minute the SHARES arm's own record proves it managed `day`'s book, for a
    checkpoint written before managed_through existed. See the 2026-10-21 correction.

    -> {"managed_through": iso or None, "why": reason when None, the evidence, and
        "unproven_symbols": symbols whose quotes or bars failed inside the trace window}"""
    from . import store as S
    saved = str(blob.get("saved_at") or "")
    ev = {"source": "record_trace", "saved_at": saved or None, "managed_through": None}
    if not saved:
        return {**ev, "why": "checkpoint has no saved_at"}
    rows = [s for s in S.read_signals(arm_dir, dt.date.fromisoformat(day))
            if str(s.get("ts") or "")[:10] == day and str(s["ts"]) <= saved]
    # A row's bar was managed on the tick that wrote it: the book is managed through the
    # close of that bar, the same instant a live runner would record as managed_through.
    marks = [(dt.datetime.fromisoformat(s["bar_ts"]) + G.TICK).isoformat()
             for s in rows if s.get("bar_ts")]
    marks += [str(t["exit_ts"]) for t in G._jsonl(arm_dir / "trades.jsonl")
              if str(t.get("exit_ts") or "")[:10] == day and str(t["exit_ts"]) <= saved
              and not t.get("reconstructed")]
    if not rows or not marks:
        return {**ev, "why": "no record of the runner managing that session"}
    last_row = dt.datetime.fromisoformat(max(str(s["ts"]) for s in rows))
    end = last_row.replace(second=0, microsecond=0)
    window = {(end - dt.timedelta(minutes=k)).isoformat()[:16] for k in range(TRACE_WINDOW)}
    covered = len(window & {str(s["ts"])[:16] for s in rows})
    anchor = min(max(marks), saved)
    ev.update({"last_record": last_row.isoformat(), "records": len(rows),
               "trace_minutes": f"{covered}/{TRACE_WINDOW}"})
    if covered < TRACE_MIN:
        return {**ev, "why": f"the record is too sparse before its last row ({covered} of the "
                             f"{TRACE_WINDOW} minutes carry one; needs {TRACE_MIN}) to prove "
                             f"the runner stopped there"}
    # Only failures INSIDE the window the anchor vouches for matter. Ones after it (09-10's
    # 15:42 DNS errors) are the runner stopping, which the replay from the anchor covers.
    since, until = (end - dt.timedelta(minutes=TRACE_WINDOW)).isoformat(), last_row.isoformat()
    bad = sorted({_outage_symbol(r) for r in G._jsonl(arm_dir / "outages.jsonl")
                  if r.get("kind") in QUOTE_MISS and since <= str(r.get("ts") or "") <= until}
                 - {None})
    return {**ev, "managed_through": anchor, "unproven_symbols": bad}


def pending(lab_dir, before: dt.date, now: dt.datetime | None = None) -> list[dict]:
    """Sessions before `before` whose checkpoint holds positions with no trade row.

    One item per (arm, session), from its latest-saved copy. `status` is "pending" (a
    managed_through or a record anchor exists: replayed), "no_managed_through" (none can be
    proven: estimated only) or "refused" (a self-check failed before: estimated only)."""
    now = now or now_et()
    out = []
    for arm, (_, key) in ARMS.items():
        arm_dir = _arm_dir(lab_dir, arm)
        rows = G._jsonl(arm_dir / "trades.jsonl")
        have = {r.get(key) for r in rows if r.get(key)}
        have_ident = {_ident(r) for r in rows}
        refused = {r.get("session_date") for r in G._jsonl(arm_dir / RUNS) if r.get("refused")}
        best: dict[str, dict] = {}
        for path, blob in _checkpoints(arm_dir):
            day = blob.get("session_date")
            if not day or day >= before.isoformat():
                continue
            if day == now.date().isoformat() and now.time() < SETTLED_AT:
                continue                      # still the live book
            missing = [p for p in blob.get("positions") or []
                       if not (p.get(key) and p[key] in have) and _ident(p) not in have_ident]
            if not missing:
                continue
            prev = best.get(day)
            if prev is None or str(blob.get("saved_at")) > str(prev["blob"].get("saved_at")):
                best[day] = {"arm": arm, "session_date": day, "checkpoint": path,
                             "blob": blob, "missing": missing}
        for day, item in sorted(best.items()):
            if day in refused:
                item["status"] = "refused"
            elif item["blob"].get("managed_through"):
                item["status"] = "pending"
            elif arm == "shares":
                item["anchor"] = record_anchor(arm_dir, day, item["blob"])
                item["status"] = "pending" if item["anchor"]["managed_through"] \
                    else "no_managed_through"
            else:
                item["status"] = "no_managed_through"
            out.append(item)
    return out


# --------------------------------------------------------------------------- estimates

def _history(day: dt.date):
    """-> (history feed, its clock, closer). Patched by the tests."""
    from . import backfill as BF
    from .feed import ThetaLiveFeed
    real = ThetaLiveFeed()
    clock = BF.SimClock()
    return BF.HistoryFeed(real, clock, day), clock, real.close


def estimate(arm: str, day: dt.date, positions: list[dict], config_hash=None
             ) -> tuple[list[dict], list[dict]]:
    """The 15:55 flatten, priced from history. -> (rows, positions that could not be priced)"""
    from . import backfill as BF
    hist, clock, close = _history(day)
    rows, unpriced = [], []
    try:
        if arm == "shares":
            from .shares_runner import EOD_FLAT
            # shares closes on the 15:55 BAR, which is admitted at 15:56:02
            clock.now = dt.datetime.combine(day, EOD_FLAT) + G.TICK + BF.TICK_OFFSET
        else:
            from .runner import EOD_FLAT
            clock.now = dt.datetime.combine(day, EOD_FLAT) + BF.TICK_OFFSET
        at = clock.now
        for p in positions:
            try:
                row = (_estimate_share(p, hist, at, config_hash) if arm == "shares"
                       else _estimate_option(p, hist, at))
            except Exception:                               # noqa: BLE001
                row = None
            (rows if row else unpriced).append(row or p)
    finally:
        close()
    return rows, unpriced


def _estimate_share(p: dict, hist, at: dt.datetime, config_hash) -> dict | None:
    q = hist.stock_quote(p["symbol"])
    if q is None:
        return None
    long_ = p["direction"] == "long"
    px = q["bid"] if long_ else q["ask"]
    pnl = p["shares"] * ((px - p["entry_px"]) if long_ else (p["entry_px"] - px))
    notional = p["shares"] * p["entry_px"]
    entry = dt.datetime.fromisoformat(p["entry_ts"])
    return {"arm": "SHARES", "config_hash": config_hash, "setup_id": p["setup_id"],
            "symbol": p["symbol"], "direction": p["direction"], "entry_ts": p["entry_ts"],
            "exit_ts": at.isoformat(), "entry_bar_ts": p["entry_bar_ts"],
            "entry_px": p["entry_px"], "exit_px": px, "shares": p["shares"],
            "notional": round(notional, 2), "exit_reason": "eod_estimate",
            "pnl_net": round(pnl, 4), "return_pct": pnl / notional if notional else None,
            "entry_spread_bp": p.get("spread_bp"), "signal_id": p.get("signal_id"),
            "hold_minutes": round((at - entry).total_seconds() / 60.0, 2),
            "decision_lag_min": round((entry - dt.datetime.fromisoformat(p["entry_bar_ts"]))
                                      .total_seconds() / 60.0, 2)}


def _estimate_option(p: dict, hist, at: dt.datetime) -> dict | None:
    from .positions import Position
    q = hist.stock_quote(p["symbol"])
    chain = hist.chain_quotes(p["symbol"], dt.date.fromisoformat(p["expiration"]))
    c = next((c for c in chain if abs(c["strike"] - p["strike"]) < 1e-9
              and c["right"] == p["right"]), None)
    if q is None or c is None:
        return None
    pos = Position(**{k: v for k, v in p.items() if k in Position.__dataclass_fields__})
    return pos.close(bid=c["bid"], underlying=q["mid"], ts=at, reason="eod_estimate")


# --------------------------------------------------------------------------- resolving

def _session_hash(recorded: list[dict]) -> str | None:
    hashes = Counter(t.get("config_hash") for t in recorded if t.get("config_hash"))
    return next(iter(hashes)) if len(hashes) == 1 else None


def resolve(item: dict, lab_dir, write: bool = False) -> dict:
    """Settle one stranded session. -> report; with write=True, appends the exits.

    Replayed when a managed_through or record anchor exists and no earlier self-check
    failed; every position the replay did not settle is then estimated (see the module
    docstring)."""
    arm, key = item["arm"], ARMS[item["arm"]][1]
    arm_dir = _arm_dir(lab_dir, arm)
    day = dt.date.fromisoformat(item["session_date"])
    rep = {"arm": arm, "session_date": item["session_date"],
           "checkpoint": Path(item["checkpoint"]).name, "positions": len(item["missing"]),
           "written": 0, "estimated": 0, "net": 0.0}
    anchor = item.get("anchor")
    if anchor:
        rep["anchor"] = anchor
    mt = item["blob"].get("managed_through") or (anchor or {}).get("managed_through")
    recorded = [t for t in G._jsonl(arm_dir / "trades.jsonl")
                if t["entry_ts"][:10] == item["session_date"] and not t.get("reconstructed")]
    session_hash = _session_hash(recorded) if arm == "shares" else None
    unproven = set((anchor or {}).get("unproven_symbols") or [])

    rows: list[dict] = []
    if item.get("status") == "refused":
        rep["refused"] = "an earlier run's self-check failed on this session; not replayed again"
    elif not mt:
        rep["refused"] = ((anchor or {}).get("why")
                          or "checkpoint has no managed_through; its saved_at cannot be trusted")
    else:
        start = dt.datetime.fromisoformat(mt).replace(second=0, microsecond=0) + G.TICK
        rep["managed_from"] = start.isoformat()
        rows = _replay(item, arm_dir, day, start, recorded, unproven, rep)

    keys, idents = {r.get(key) for r in rows if r.get(key)}, {_ident(r) for r in rows}
    left = [p for p in item["missing"]
            if not ((p.get(key) and p[key] in keys) or _ident(p) in idents)]
    est, unpriced = estimate(arm, day, left, session_hash) if left else ([], [])
    rep["unpriced"] = len(unpriced)
    rep["net"] = round(sum(r.get("pnl_net") or 0 for r in rows), 2)
    rep["estimated_net"] = round(sum(r.get("pnl_net") or 0 for r in est), 2)
    rep["rows"], rep["estimates"] = rows, est
    if not write:
        return rep

    from . import store as S
    store = S.LabStore(arm_dir)
    stamp = now_et().isoformat(timespec="seconds")
    common = {"checkpoint": rep["checkpoint"], "managed_through": mt,
              "managed_through_source": ("checkpoint" if item["blob"].get("managed_through")
                                         else (anchor or {}).get("source")),
              "written_at": stamp}
    for r in rows:
        row = {k: v for k, v in r.items() if k not in ("seq", "rebuild")}
        recon = {**common, "managed_from": rep["managed_from"], "method": METHOD,
                 "validation": rep["validation"]}
        if session_hash and row.get("config_hash") != session_hash:
            recon["replay_config_hash"] = row.get("config_hash")
            row["config_hash"] = session_hash
        row.update({"exit_mode": EXIT_MODE, "reconstructed": True, "reconstruction": recon})
        store.write_trade(row)
        rep["written"] += 1
    for r in est:
        row = {k: v for k, v in r.items() if k != "seq"}
        row.update({"exit_mode": ESTIMATE_MODE, "reconstructed": True, "estimate": True,
                    "reconstruction": {**common, "method": ESTIMATE_METHOD,
                                       "why": rep.get("refused") or "the replay gave it no exit"}})
        store.write_trade(row)
        rep["estimated"] += 1
    with open(arm_dir / RUNS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "stranded_recovery_run",
                             **{k: v for k, v in rep.items() if k not in ("rows", "estimates")},
                             "at": now_et().isoformat(timespec="seconds")}) + "\n")
    return rep


def _replay(item, arm_dir, day, start, recorded, unproven, rep) -> list[dict]:
    """Replay the session with the stranded positions joining at `start`. -> the rows it
    may write (none when the self-check fails); fills rep's validation and refusal."""
    arm, key = item["arm"], ARMS[item["arm"]][1]
    from . import store as S
    if arm == "shares":
        from .shares_runner import SharePos as cls
    else:
        from .positions import Position as cls
    fields = set(cls.__dataclass_fields__)
    missing = [p for p in item["missing"] if p.get(key) and p.get("symbol") not in unproven]
    extra = [(start, cls(**{k: v for k, v in p.items() if k in fields})) for p in missing]
    replay = G.replay_shares if arm == "shares" else G.replay_options
    saved_now = S._now
    try:
        replayed, _problems = replay(day, recorded, arm_dir, extra=extra)
    finally:
        S._now = saved_now            # the replay patches the store's clock
    rep["validation"], agreement = G.validate(recorded, replayed, key)
    if agreement < G.MIN_AGREEMENT:
        rep["refused"] = (f"replay reproduced the exit rule of {agreement:.0%} of the exits "
                          f"it could check; needs {G.MIN_AGREEMENT:.0%}")
        return []
    return [replayed[p[key]] for p in missing if p[key] in replayed]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--before", required=True, help="resolve sessions before this date")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--lab-dir", default=str(G.LAB))
    a = ap.parse_args(argv)
    items = pending(a.lab_dir, dt.date.fromisoformat(a.before))
    if not items:
        print("[stranded] nothing stranded")
    rc = 0
    for item in items:
        tag = f"[stranded] {item['arm']} {item['session_date']}"
        try:
            rep = resolve(item, a.lab_dir, write=a.write)
        except Exception as exc:                             # noqa: BLE001
            print(f"{tag}: replay failed, will retry: {exc!r}")
            rc = 1
            continue
        how = ""
        if "validation" in rep:
            v = rep["validation"][rep["validation"]["gate_pool"]]
            how = (f"replayed from {rep['managed_from'][11:16]}, self-check "
                   f"{v['same_rule']}/{v['checked']} same rule, net {rep['net']:+.2f}; ")
        print(f"{tag}: {rep['positions']} position(s): {how}"
              f"{len(rep['estimates'])} estimated at the 15:55 flatten "
              f"(net {rep['estimated_net']:+.2f}), unpriced {rep['unpriced']}, "
              f"written {rep['written'] + rep['estimated']}"
              + (f" -- replay refused: {rep['refused']}" if rep.get("refused") else ""))
        for r in rep["rows"] + rep["estimates"]:
            print(f"    {r['setup_id']:<26}{r['symbol']:<6}{r.get('direction', ''):<6}"
                  f"{r.get('arm', '')!s:<7}{r['exit_reason']:<13}{r['exit_ts'][11:16]}"
                  f"{r.get('pnl_net') or 0:>+10.2f}")
        if rep["unpriced"]:
            rc = 1                     # retried at the next window
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
