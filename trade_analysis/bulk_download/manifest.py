"""SQLite record of every request the archive has ever made.

Resumption cannot be driven by "does the output file exist", because the most
common outcome at scale is a legitimately EMPTY one: a holiday, a symbol not yet
listed, a strike that never traded. Those produce no file, so a file-existence
check re-requests them on every single run - tens of thousands of pointless
calls that never terminate into a finished archive.

So each (layer, symbol, key) gets a row with its terminal status. NO_DATA and
NOT_ENTITLED are as final as OK. Only ERROR is retried, because only ERROR is
transient. This is also the only place that can answer "what is actually in the
archive, and what is missing because we are not paying for it" - a question the
directory tree cannot answer, since both look like an absent file.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from .config import STATE_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS downloads (
    layer      TEXT NOT NULL,   -- e.g. stock.quote.1s
    symbol     TEXT NOT NULL,
    key        TEXT NOT NULL,   -- date, YYYY-MM month, or expiration:date
    status     TEXT NOT NULL,   -- OK | NO_DATA | NOT_ENTITLED | ERROR
    rows       INTEGER DEFAULT 0,
    bytes      INTEGER DEFAULT 0,
    path       TEXT,
    detail     TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (layer, symbol, key)
);
CREATE INDEX IF NOT EXISTS idx_layer_status ON downloads(layer, status);

CREATE TABLE IF NOT EXISTS entitlements (
    layer       TEXT PRIMARY KEY,
    available   INTEGER NOT NULL,
    first_date  TEXT,
    note        TEXT,
    checked_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class Manifest:
    def __init__(self, db_path: Path = STATE_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # One connection per thread: sqlite3 objects are not shareable across
        # threads, and the downloaders are thread-pooled.
        self._local = threading.local()
        # WAL so reader threads (progress reporting) never block the writers.
        with self._conn() as c:
            c.executescript(SCHEMA)
            c.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=60.0)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------ write
    def record(self, layer: str, symbol: str, key: str, status: str,
               rows: int = 0, nbytes: int = 0, path: str | None = None,
               detail: str = ""):
        with self._conn() as c:
            c.execute(
                "INSERT INTO downloads (layer,symbol,key,status,rows,bytes,path,detail,updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,datetime('now'))"
                " ON CONFLICT(layer,symbol,key) DO UPDATE SET"
                "  status=excluded.status, rows=excluded.rows, bytes=excluded.bytes,"
                "  path=excluded.path, detail=excluded.detail, updated_at=datetime('now')",
                (layer, symbol, key, status, rows, nbytes, path, detail[:500]),
            )

    def record_entitlement(self, layer: str, available: bool,
                           first_date: str | None, note: str = ""):
        with self._conn() as c:
            c.execute(
                "INSERT INTO entitlements (layer,available,first_date,note,checked_at)"
                " VALUES (?,?,?,?,datetime('now'))"
                " ON CONFLICT(layer) DO UPDATE SET available=excluded.available,"
                "  first_date=excluded.first_date, note=excluded.note,"
                "  checked_at=datetime('now')",
                (layer, int(available), first_date, note[:500]),
            )

    # ------------------------------------------------------------------- read
    def done_keys(self, layer: str, symbol: str | None = None) -> set[tuple[str, str]]:
        """(symbol, key) pairs that are settled and must not be re-requested.

        ERROR is excluded on purpose - it is the transient bucket and the whole
        point of tracking it separately is that the next run picks it back up.
        """
        q = ("SELECT symbol,key FROM downloads WHERE layer=? "
             "AND status IN ('OK','NO_DATA','NOT_ENTITLED')")
        args = [layer]
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        with self._conn() as c:
            return {(r["symbol"], r["key"]) for r in c.execute(q, args)}

    def summary(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT layer, status, COUNT(*) n, SUM(rows) rows, SUM(bytes) bytes"
                " FROM downloads GROUP BY layer, status ORDER BY layer, status")]

    def failures(self, layer: str | None = None, limit: int = 50) -> list[dict]:
        q = "SELECT * FROM downloads WHERE status='ERROR'"
        args: list = []
        if layer:
            q += " AND layer=?"
            args.append(layer)
        q += " ORDER BY updated_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args)]

    def entitlements(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM entitlements ORDER BY layer")]
