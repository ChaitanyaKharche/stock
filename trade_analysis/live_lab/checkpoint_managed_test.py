"""A restart catches up from the last minute MANAGED, not the last time the book was saved.

    python -m trade_analysis.live_lab.checkpoint_managed_test

On 2026-10-02 the hotspot dropped at 11:51. The runners froze, and while frozen they kept
re-saving positions_open.json every idle poll, so `saved_at` reached 13:05 although the
book had last been managed at ~12:03. When the process died and the 15:07 restart read that
checkpoint, it caught up from 13:05: every stop and target in 12:04-13:05 went unchecked.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

from .catchup import checkpoint_time
from .clock import now_et
from .store import LabStore


def _day_and_managed():
    day = now_et().date()
    return day, dt.datetime.combine(day, dt.time(12, 3))


def test_restart_resumes_from_last_managed_minute():
    day, managed = _day_and_managed()
    with tempfile.TemporaryDirectory() as tmp:
        store = LabStore(tmp)
        store.save_open_positions([], session_date=day, managed_through=managed)
        saved = json.loads((Path(tmp) / "positions_open.json").read_text(encoding="utf-8"))
        assert saved["managed_through"] == managed.isoformat()
        assert checkpoint_time(tmp, day) == managed


def test_old_checkpoint_without_the_field_still_reads_saved_at():
    day, _ = _day_and_managed()
    stamp = dt.datetime.combine(day, dt.time(13, 5))
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "positions_open.json").write_text(json.dumps(
            {"saved_at": stamp.isoformat(), "session_date": day.isoformat(),
             "positions": []}), encoding="utf-8")
        assert checkpoint_time(tmp, day) == stamp


def test_another_days_checkpoint_is_ignored():
    day, managed = _day_and_managed()
    with tempfile.TemporaryDirectory() as tmp:
        LabStore(tmp).save_open_positions([], session_date=day - dt.timedelta(days=1),
                                          managed_through=managed)
        assert checkpoint_time(tmp, day) is None


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
