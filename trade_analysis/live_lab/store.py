"""Durable, append-only persistence for the live lab.

The central guarantee of this whole system lives here: `write_decision()` fsyncs the
signal to disk BEFORE any option price is requested. A decision is therefore durable
before any number that could have influenced it exists in the process.

Files (under LAB_DIR, one tree per environment):
    config/<hash>.json     frozen setup definitions, git sha, spec version
    events.jsonl           append-only, monotonic seq, every observation
    signals.jsonl          every signal INCLUDING rejected ones, with skip_reason
    positions_open.json    atomic-replace snapshot, for crash recovery
    trades.jsonl           completed trades, one line per (signal x strike arm)
    outages.jsonl          disconnects, stale quotes, degraded bars
    daily/<date>.json      end-of-session summary
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

def _now() -> dt.datetime:
    """Exchange time. Never the machine clock -- see clock.py."""
    from .clock import now_et
    return now_et()


DEFAULT_LAB_DIR = Path(__file__).resolve().parents[2] / "live_lab_data"


def _json_default(o):
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


class LabStore:
    def __init__(self, root: Path | str = DEFAULT_LAB_DIR):
        self.root = Path(root)
        (self.root / "config").mkdir(parents=True, exist_ok=True)
        (self.root / "daily").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = self._recover_seq()

    # ------------------------------------------------------------------ core append

    def _append(self, name: str, rec: dict, fsync: bool) -> None:
        path = self.root / name
        line = json.dumps(rec, default=_json_default, separators=(",", ":"))
        with self._lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                if fsync:
                    os.fsync(fh.fileno())

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _recover_seq(self) -> int:
        """Resume the monotonic counter after a restart."""
        hi = 0
        for name in ("events.jsonl", "signals.jsonl", "trades.jsonl"):
            p = self.root / name
            if not p.exists():
                continue
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        try:
                            s = json.loads(line).get("seq")
                        except json.JSONDecodeError:
                            continue          # a torn final line survives a crash
                        if isinstance(s, int) and s > hi:
                            hi = s
            except OSError:
                continue
        return hi

    # ------------------------------------------------------------------ config

    def freeze_config(self, config: dict) -> str:
        """Write the frozen definitions keyed by their own hash. Idempotent."""
        h = config["config_hash"]
        path = self.root / "config" / f"{h}.json"
        if not path.exists():
            path.write_text(json.dumps(config, indent=2, default=_json_default),
                            encoding="utf-8")
        return h

    # ------------------------------------------------------------------ records

    def event(self, kind: str, **fields) -> None:
        self._append("events.jsonl",
                     {"seq": self._next_seq(), "ts": _now().isoformat(),
                      "kind": kind, **fields}, fsync=False)

    def outage(self, kind: str, detail: str, **fields) -> None:
        self._append("outages.jsonl",
                     {"seq": self._next_seq(), "ts": _now().isoformat(),
                      "kind": kind, "detail": detail, **fields}, fsync=True)

    def write_decision(self, *, setup_id: str, config_hash: str, symbol: str,
                       direction: str, bar_ts: dt.datetime, state: dict) -> str:
        """THE ordering guarantee.

        Called at signal time, BEFORE any option chain request. Returns the signal uuid.
        fsync is mandatory here and is the reason this method exists separately from
        `event()`: if the process dies immediately after, the decision still exists on
        disk with no knowledge of any price.
        """
        sid = uuid.uuid4().hex
        self._append("signals.jsonl", {
            "seq": self._next_seq(),
            "signal_id": sid,
            "phase": "DECISION",
            "ts": _now().isoformat(),
            "bar_ts": bar_ts,
            "setup_id": setup_id,
            "config_hash": config_hash,
            "symbol": symbol,
            "direction": direction,
            "state": state,
        }, fsync=True)
        return sid

    def write_fill(self, signal_id: str, **fields) -> None:
        """Written only AFTER write_decision has returned."""
        self._append("signals.jsonl",
                     {"seq": self._next_seq(), "signal_id": signal_id, "phase": "FILL",
                      "ts": _now().isoformat(), **fields}, fsync=True)

    def write_skip(self, *, setup_id: str, config_hash: str, symbol: str,
                   bar_ts: dt.datetime, reason: str, **fields) -> None:
        """A signal that fired but was not opened. Recorded so the denominator is honest."""
        self._append("signals.jsonl", {
            "seq": self._next_seq(), "signal_id": uuid.uuid4().hex, "phase": "SKIP",
            "ts": _now().isoformat(), "bar_ts": bar_ts,
            "setup_id": setup_id, "config_hash": config_hash, "symbol": symbol,
            "skip_reason": reason, **fields,
        }, fsync=True)

    def write_trade(self, trade: dict) -> None:
        self._append("trades.jsonl",
                     {"seq": self._next_seq(), **trade}, fsync=True)

    # ------------------------------------------------------------------ crash recovery

    def save_open_positions(self, positions: list[dict],
                            session_date=None) -> None:
        """Atomic replace so a crash mid-write cannot corrupt the recovery file.

        `session_date` stamps which trading day these positions belong to. Recovery
        refuses to reopen positions from a different day -- see load_open_positions.
        """
        path = self.root / "positions_open.json"
        payload = json.dumps({"saved_at": _now().isoformat(),
                              "session_date": (session_date.isoformat()
                                               if session_date else None),
                              "positions": positions},
                             default=_json_default, indent=1)
        fd, tmp = tempfile.mkstemp(dir=str(self.root), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def load_open_positions(self, expect_date=None) -> tuple[list[dict], str]:
        """Returns (positions, note).

        A process killed with positions open -- laptop lid, power loss, Task Manager --
        never runs its shutdown flatten, so the file survives into the next session.
        Reopening those positions on a later day would mark them against the wrong day's
        quotes on a contract that has already expired, writing garbage into a frozen
        record. So positions are handed back ONLY when their stamped session date matches
        the day being run; otherwise they are archived for inspection and dropped.

        Files written before this stamp existed carry `session_date: null`. Those are
        treated as stale too -- refusing is the conservative direction, and the only cost
        is one abandoned recovery.
        """
        path = self.root / "positions_open.json"
        if not path.exists():
            return [], ""
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return [], "unreadable recovery file"
        pos = blob.get("positions", [])
        if not pos:
            return [], ""
        stamped = blob.get("session_date")
        want = expect_date.isoformat() if expect_date else None
        if want is not None and stamped != want:
            why = (f"recovery file is for session {stamped or 'UNSTAMPED'}, "
                   f"today is {want}")
            arch = self.root / "recovery_archive"
            arch.mkdir(parents=True, exist_ok=True)
            dest = arch / f"positions_open_{stamped or 'unstamped'}_{_now():%Y%m%d%H%M%S}.json"
            try:
                dest.write_text(json.dumps(blob, indent=1), encoding="utf-8")
            except OSError:
                pass
            self.event("recovery_rejected", reason=why, n_positions=len(pos),
                       archived_to=dest.name)
            return [], why
        return pos, ""

    # ------------------------------------------------------------------ reading

    def read(self, name: str) -> list[dict]:
        path = self.root / name
        if not path.exists():
            return []
        out = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue      # tolerate a torn final line from an unclean shutdown
        return out

    def write_daily(self, day: dt.date, summary: dict[str, Any]) -> None:
        (self.root / "daily" / f"{day.isoformat()}.json").write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
