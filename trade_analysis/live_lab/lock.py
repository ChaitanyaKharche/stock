"""Single-instance lock for the live lab.

Two runners writing the same append-only files would interleave sequence numbers and
double-count every signal, which corrupts the frozen record rather than merely degrading
it. This makes that impossible.

Uses an OS-level file lock held for the process lifetime, not a PID file. The difference
matters: a PID file has to be cleaned up, and a killed process (laptop lid, power loss,
Task Manager) never gets to clean up, leaving a stale lock that blocks every later run.
An OS lock is released by the kernel when the process dies, however it dies.

    with SingleInstance("runner") as ok:
        if not ok:
            return
        ...

The PID and start time are still written into the file, purely so a human reading it can
see who holds it.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class SingleInstance:
    """Cooperative exclusive lock. Truthy on acquire, falsy if someone else holds it."""

    def __init__(self, name: str, lab_dir: str | os.PathLike | None = None):
        from .store import DEFAULT_LAB_DIR
        root = Path(lab_dir) if lab_dir else DEFAULT_LAB_DIR
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{name}.lock"
        self.info = root / f"{name}.lock.info"
        self.fh = None
        self.acquired = False

    def acquire(self) -> bool:
        try:
            self.fh = open(self.path, "a+", encoding="utf-8")
        except OSError:
            return False
        try:
            # MUST seek to 0 first. msvcrt.locking() locks n bytes from the CURRENT file
            # position, and "a+" leaves two processes at different offsets once anything
            # has been written -- so without this they lock different bytes and both
            # "succeed". fcntl.flock is whole-file and does not care, but the seek is
            # harmless there.
            self.fh.seek(0)
            if os.name == "nt":
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            self.fh = None
            return False
        self.acquired = True
        # Holder metadata goes in a SEPARATE file. Writing into the locked file would mean
        # truncating a byte range we hold a lock on, which is exactly the kind of thing
        # that behaves differently on every platform.
        try:
            self.info.write_text(
                f"pid={os.getpid()} since={dt.datetime.now().isoformat()}\n",
                encoding="utf-8")
        except OSError:
            pass
        return True

    def holder(self) -> str:
        """Whatever the current holder wrote, for logging. Empty if unreadable."""
        try:
            return self.info.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def release(self) -> None:
        if self.fh is None:
            return
        try:
            if self.acquired:
                self.fh.seek(0)
                if os.name == "nt":
                    msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                self.fh.close()
            except OSError:
                pass
            self.fh = None
            self.acquired = False

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
