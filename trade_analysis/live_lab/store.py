"""Durable, append-only persistence for the live lab.

The central guarantee of this whole system lives here: `write_decision()` fsyncs the
signal to disk BEFORE any option price is requested. A decision is therefore durable
before any number that could have influenced it exists in the process.

Files (under LAB_DIR, one tree per environment):
    config/<hash>.json     frozen setup definitions, git sha, spec version
    events.jsonl           append-only, monotonic seq, every observation
    signals.jsonl          every signal INCLUDING rejected ones, with skip_reason --
                           TODAY's rows only; earlier sessions are rolled, byte for byte,
                           into signals_archive/<date>.jsonl.gz (roll_signals)
    positions_open.json    atomic-replace snapshot, for crash recovery
    trades.jsonl           completed trades, one line per (signal x strike arm)
    outages.jsonl          disconnects, stale quotes, degraded bars
    daily/<date>.json      end-of-session summary
"""
from __future__ import annotations

import datetime as dt
import gzip
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
SIGNALS = "signals.jsonl"
SIGNALS_ARCHIVE = "signals_archive"


# ---------------------------------------------------------------------- signals archive
#
# signals.jsonl is the biggest file in the lab: the shares arm writes a SKIP row every
# time a capped setup fires again (shares_lab_preregistration.md item 6 -- "every capped
# signal is written as a SKIP so the denominator stays honest"), 3-4.5 MB a session. One
# ever-growing file would pass the 100 MB per-file limit of the git host the lab pushes
# to every night. So the rows are kept -- every one, unchanged -- but each finished
# session's rows move into their own gzipped file, and the active file holds only today.

def _row_day(line: str) -> str | None:
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        return None
    return str(r.get("ts") or r.get("bar_ts") or "")[:10] or None


def _parse_lines(lines) -> list[dict]:
    out = []
    for line in lines:
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # tolerate a torn final line from an unclean shutdown
    return out


def _archive_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:      # reads every gzip member
        return [l for l in fh.read().splitlines() if l.strip()]


def read_signals(root: Path | str, day=None) -> list[dict]:
    """Every signal row under `root`, archived sessions first, in session order.

    With `day`, only that session's rows: its archive, if it has been rolled, plus the
    active file's rows for that day. Every reader of signals goes through here.
    """
    root = Path(root)
    d = (day.isoformat() if hasattr(day, "isoformat") else str(day)) if day else None
    arch = root / SIGNALS_ARCHIVE
    if not arch.exists():
        files = []
    elif d:
        files = [arch / f"{d}.jsonl.gz"]
    else:
        files = sorted(arch.glob("*.jsonl.gz"))
    out = []
    for f in files:
        if f.exists():
            out += _parse_lines(_archive_lines(f))
    active = root / SIGNALS
    if active.exists():
        with open(active, "r", encoding="utf-8") as fh:
            rows = _parse_lines(fh)
        out += [r for r in rows
                if not d or str(r.get("ts") or r.get("bar_ts") or "")[:10] == d]
    return out


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
        """Resume the monotonic counter after a restart.

        A rolled signals file leaves the active one holding only today, so the newest
        archive is scanned too: without it a restart early in a session could reuse
        sequence numbers already spent the session before.
        """
        hi = 0
        paths = [self.root / n for n in ("events.jsonl", SIGNALS, "trades.jsonl")]
        arch = sorted((self.root / SIGNALS_ARCHIVE).glob("*.jsonl.gz"))
        for p in paths + arch[-1:]:
            if not p.exists():
                continue
            try:
                opener = gzip.open if p.suffix == ".gz" else open
                with opener(p, "rt", encoding="utf-8") as fh:
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

    # ------------------------------------------------------------------ signals archive

    def roll_signals(self, today) -> dict:
        """Move every signals.jsonl row from a session before `today` into
        signals_archive/<that date>.jsonl.gz, byte for byte; keep the rest active.

        Called by each runner at start, under its single-instance lock, before it writes
        anything. Crash-safe in either order: an archive is written and fsynced before the
        active file is replaced, and a row already in its archive is never added twice, so
        a crash between the two steps leaves at worst a duplicate that the next roll drops.
        """
        today = today.isoformat() if hasattr(today, "isoformat") else str(today)
        active = self.root / SIGNALS
        if not active.exists():
            return {}
        with self._lock:
            with open(active, "r", encoding="utf-8") as fh:
                lines = [l.rstrip("\n") for l in fh if l.strip()]
            old, keep, last = {}, [], None
            for line in lines:
                d = _row_day(line) or last          # a torn line goes with its session
                last = d
                if d is not None and d < today:
                    old.setdefault(d, []).append(line)
                else:
                    keep.append(line)
            if not old:
                return {}
            arch = self.root / SIGNALS_ARCHIVE
            arch.mkdir(exist_ok=True)
            moved = {}
            for d, rows in sorted(old.items()):
                path = arch / f"{d}.jsonl.gz"
                have = set(_archive_lines(path)) if path.exists() else set()
                new = [r for r in rows if r not in have]
                if new:
                    with open(path, "ab") as raw:
                        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                            gz.write(("\n".join(new) + "\n").encode("utf-8"))
                        raw.flush()
                        os.fsync(raw.fileno())
                moved[d] = len(rows)
            fd, tmp = tempfile.mkstemp(dir=str(self.root), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write("".join(l + "\n" for l in keep))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, active)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
        self.event("signals_rolled", sessions=len(moved), rows=sum(moved.values()),
                   kept=len(keep))
        return moved

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
                            session_date=None, managed_through=None) -> None:
        """Atomic replace so a crash mid-write cannot corrupt the recovery file.

        `session_date` stamps which trading day these positions belong to. Recovery
        refuses to reopen positions from a different day -- see load_open_positions.

        `managed_through` is the last minute the runner actually managed (its catch-up
        `last_good`). It is NOT `saved_at`: a frozen runner keeps saving its book every idle
        poll while managing nothing, so on 2026-10-02 `saved_at` reached 13:05 while the
        book had last been managed at ~12:03, and the 15:07 restart caught up from 13:05 --
        an hour of stops and targets never checked. A restart resumes from this field.
        """
        path = self.root / "positions_open.json"
        payload = json.dumps({"saved_at": _now().isoformat(),
                              "managed_through": (managed_through.isoformat()
                                                  if managed_through else None),
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

    def decision_counts(self, day) -> tuple[dict, dict]:
        """Per-day signal counts for `day`, rebuilt from the durable DECISION record.

        Returns ({(setup_id, symbol): n}, {(setup_id, symbol, direction): n}), which is
        exactly what max_per_day and max_per_direction are counted against.

        This exists because rebuilding those caps from recovered OPEN positions is wrong in
        two opposite ways, and both bugs were live:

          * every signal that already CLOSED is forgotten, so a mid-session restart resets
            the cap to zero and the setup fires again. On 2026-09-02 the shares arm
            restarted at 15:00 and re-entered ORB_5min and ORB_15min, each of which has
            max_per_day = 1 and had already traded that morning.
          * the options arm opens THREE positions per signal (ATM, ATM+-1), so counting one
            per recovered position triples the count and over-suppresses instead.

        signals.jsonl carries exactly one DECISION per signal in both arms, fsynced before
        any price is requested, so it is the authoritative count for both.
        """
        d = day.isoformat() if hasattr(day, "isoformat") else str(day)
        per, per_dir = {}, {}
        for s in read_signals(self.root, d):
            if s.get("phase") != "DECISION":
                continue
            stamp = str(s.get("bar_ts") or s.get("ts") or "")[:10]
            if stamp != d:
                continue
            sid, sym, direc = s.get("setup_id"), s.get("symbol"), s.get("direction")
            if not sid or not sym:
                continue
            per[(sid, sym)] = per.get((sid, sym), 0) + 1
            if direc:
                per_dir[(sid, sym, direc)] = per_dir.get((sid, sym, direc), 0) + 1
        return per, per_dir

    def decision_keys(self, day) -> set[tuple[str, str, str]]:
        """(setup_id, symbol, bar_ts) of every DECISION already durable for `day`.

        A decision is taken ONCE. A restart, a catch-up, a replayed batch or a bar the feed
        re-publishes must never produce a second DECISION for the same setup, symbol and
        bar -- the idempotency a broker gives you with a unique client order id, and FIX
        with PossDupFlag. The per-day caps already stop most repeats; this stops all of
        them, including setups whose cap is above one.
        """
        d = day.isoformat() if hasattr(day, "isoformat") else str(day)
        out = set()
        for s in read_signals(self.root, d):
            if s.get("phase") != "DECISION":
                continue
            b = str(s.get("bar_ts") or "")[:19]
            if b[:10] == d and s.get("setup_id") and s.get("symbol"):
                out.add((s["setup_id"], s["symbol"], b))
        return out

    # ------------------------------------------------------------------ reading

    def read(self, name: str) -> list[dict]:
        if name == SIGNALS:
            return read_signals(self.root)        # archived sessions + today
        path = self.root / name
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as fh:
            return _parse_lines(fh)

    def write_daily(self, day: dt.date, summary: dict[str, Any]) -> None:
        (self.root / "daily" / f"{day.isoformat()}.json").write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
