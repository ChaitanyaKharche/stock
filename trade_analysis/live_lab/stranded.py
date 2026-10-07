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
    as live on >= MIN_AGREEMENT of them. Below that nothing is written.
  * APPENDED AND FLAGGED. Rows go to trades.jsonl exactly as the runner writes them, plus
    "exit_mode": "reconstructed_from_history", "reconstructed": true, and a
    "reconstruction" block naming the checkpoint, the window and the self-check. One
    filter excludes them; nothing already in the record is edited.
  * A checkpoint without `managed_through` (written before 2026-10-02) is reported and
    never resolved: its `saved_at` can be an hour past the last minute anyone managed.
  * Today's checkpoint is the LIVE book until the session is over and is never touched
    before SETTLED_AT.
  * Idempotent: a position that already has a row is skipped, and a session whose
    self-check failed is recorded in stranded_recovery.jsonl and not retried.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from . import gap_recovery as G
from .clock import now_et

EXIT_MODE = "reconstructed_from_history"
RUNS = "stranded_recovery.jsonl"
SETTLED_AT = dt.time(16, 5)     # today's checkpoint stops being the live book after this
ARMS = {"options": ("", "position_id"), "shares": ("shares", "signal_id")}
METHOD = ("checkpoint position managed from the minute after managed_through on history 1m "
          "bars + 1m NBBO/chain, through the runner's own exit code (stranded.py)")


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


def pending(lab_dir, before: dt.date, now: dt.datetime | None = None) -> list[dict]:
    """Sessions before `before` whose checkpoint holds positions with no trade row.

    One item per (arm, session), from its latest-saved copy. `status` is "pending" (can be
    resolved), "no_managed_through" (too old to trust) or "refused" (self-check failed
    before; not retried)."""
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
            item["status"] = ("refused" if day in refused else
                              "pending" if item["blob"].get("managed_through") else
                              "no_managed_through")
            out.append(item)
    return out


def resolve(item: dict, lab_dir, write: bool = False) -> dict:
    """Replay one stranded session. -> report; with write=True, appends the exits."""
    arm, key = item["arm"], ARMS[item["arm"]][1]
    arm_dir = _arm_dir(lab_dir, arm)
    day = dt.date.fromisoformat(item["session_date"])
    rep = {"arm": arm, "session_date": item["session_date"],
           "checkpoint": Path(item["checkpoint"]).name, "positions": len(item["missing"]),
           "written": 0}
    mt = item["blob"].get("managed_through")
    if not mt:
        rep["refused"] = "checkpoint has no managed_through; its saved_at cannot be trusted"
        return rep
    start = dt.datetime.fromisoformat(mt).replace(second=0, microsecond=0) + G.TICK
    rep["managed_from"] = start.isoformat()

    from . import store as S
    if arm == "shares":
        from .shares_runner import SharePos as cls
    else:
        from .positions import Position as cls
    fields = set(cls.__dataclass_fields__)
    missing = [p for p in item["missing"] if p.get(key)]
    extra = [(start, cls(**{k: v for k, v in p.items() if k in fields})) for p in missing]
    recorded = [t for t in G._jsonl(arm_dir / "trades.jsonl")
                if t["entry_ts"][:10] == item["session_date"] and not t.get("reconstructed")]
    replay = G.replay_shares if arm == "shares" else G.replay_options
    saved_now = S._now
    try:
        replayed, _problems = replay(day, recorded, arm_dir, extra=extra)
    finally:
        S._now = saved_now            # the replay patches the store's clock
    rep["validation"], agreement = G.validate(recorded, replayed, key)

    rows = [replayed[p[key]] for p in missing if p[key] in replayed]
    rep["unpriced"] = len(item["missing"]) - len(rows)
    rep["net"] = round(sum(r.get("pnl_net") or 0 for r in rows), 2)
    rep["rows"] = rows
    if agreement < G.MIN_AGREEMENT:
        rep["refused"] = (f"replay reproduced the exit rule of {agreement:.0%} of the exits "
                          f"it could check; needs {G.MIN_AGREEMENT:.0%}")
    if not write:
        return rep
    if "refused" not in rep:
        store = S.LabStore(arm_dir)
        stamp = now_et().isoformat(timespec="seconds")
        for r in rows:
            row = {k: v for k, v in r.items() if k not in ("seq", "rebuild")}
            row.update({"exit_mode": EXIT_MODE, "reconstructed": True, "reconstruction": {
                "checkpoint": rep["checkpoint"], "managed_through": mt,
                "managed_from": rep["managed_from"], "method": METHOD,
                "validation": rep["validation"], "written_at": stamp}})
            store.write_trade(row)
            rep["written"] += 1
    with open(arm_dir / RUNS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "stranded_recovery_run",
                             **{k: v for k, v in rep.items() if k != "rows"},
                             "at": now_et().isoformat(timespec="seconds")}) + "\n")
    return rep


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
        if item["status"] != "pending":
            print(f"{tag}: {len(item['missing'])} position(s) left for a human "
                  f"({item['status']}, {Path(item['checkpoint']).name})")
            continue
        try:
            rep = resolve(item, a.lab_dir, write=a.write)
        except Exception as exc:                             # noqa: BLE001
            print(f"{tag}: replay failed, will retry: {exc!r}")
            rc = 1
            continue
        v = rep["validation"][rep["validation"]["gate_pool"]]
        print(f"{tag}: {rep['positions']} position(s) from {rep['managed_from'][11:16]}, "
              f"net {rep['net']:+.2f}, unpriced {rep['unpriced']}, self-check "
              f"{v['same_rule']}/{v['checked']} same rule, written {rep['written']}"
              + (f" -- REFUSED: {rep['refused']}" if rep.get("refused") else ""))
        for r in rep["rows"]:
            print(f"    {r['setup_id']:<26}{r['symbol']:<6}{r.get('direction', ''):<6}"
                  f"{r.get('arm', '')!s:<7}{r['exit_reason']:<10}{r['exit_ts'][11:16]}"
                  f"{r.get('pnl_net') or 0:>+10.2f}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
