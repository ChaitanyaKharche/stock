"""Record the live market to disk at per-second resolution.

The history downloaders can only reach yesterday. This is the other half: it
captures today, at the resolution the monitoring requirement asks for, and
writes it in the same shape as the historical archive so a strategy can run over
"history then today" without a special case.

WHY SNAPSHOTS RATHER THAN A STREAM. The obvious approach is one request per
symbol per second - 104 requests/second against an account limited to 4
concurrent. That does not work. But /stock/snapshot/quote accepts `symbol=*` and
returns the ENTIRE market (~26,000 symbols, ~800 KB) in a single call, and the
snapshot endpoints carry `x-skip-concurrent-limit`, so they do not consume the
concurrency budget at all. So the whole universe costs ONE request per poll
regardless of how many symbols are being watched, and watching 104 costs exactly
what watching 2 would.

WHAT A SNAPSHOT IS NOT. It is a point-in-time read of the vendor's last-known
value, not a tick stream: if two prints land between polls, the first is not
recorded. So this is a faithful 1-second SAMPLE, which is what the 1s history
layer is too (that layer is explicitly sample-and-hold), and the two are
therefore consistent with each other. It is NOT a substitute for the tick
layers, and anything that needs every print must use those. Each row carries the
vendor's own `timestamp` alongside our `polled_at`, so staleness is measurable
after the fact rather than assumed away - a symbol that has not traded for a
minute will show an old vendor timestamp against a fresh poll time.

Output: one file per session per feed, appended as it goes, so a crash costs at
most the current buffer rather than the day.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from .config import BASE_URL, DATA_ROOT, ET
from . import universe

LIVE_DIR = DATA_ROOT / "live"
ET_TZ = ZoneInfo(ET)

# The feeds worth polling. `quote` is the NBBO (what you could trade against),
# `ohlc` is the running session bar (needed for an opening range and for the
# high/low a breakout is measured against), `trade` is the last print.
FEEDS = {
    "quote": "/stock/snapshot/quote",
    "ohlc": "/stock/snapshot/ohlc",
    "trade": "/stock/snapshot/trade",
}


class LiveRecorder:
    def __init__(self, symbols: list[str], feeds: list[str], interval: float = 1.0,
                 out_dir: Path = LIVE_DIR):
        self.symbols = set(s.upper() for s in symbols)
        self.feeds = feeds
        self.interval = interval
        self.out_dir = out_dir
        # A short timeout on purpose: a poll that takes longer than the interval
        # is useless, and retrying next tick is better than blocking the loop.
        self._client = httpx.Client(timeout=httpx.Timeout(8.0, connect=3.0),
                                    headers={"Accept-Encoding": "gzip"})
        self._writers: dict[str, tuple] = {}
        self._stop = False
        self.polls = 0
        self.rows = 0
        self.errors = 0

    # ------------------------------------------------------------------ output
    def _writer(self, feed: str, session: str):
        """Open (or reuse) the gzip CSV writer for this feed and session."""
        key = f"{feed}:{session}"
        if key in self._writers:
            return self._writers[key]
        # Close yesterday's handles if the session rolled over mid-run.
        for k, (fh, _, _) in list(self._writers.items()):
            if k.startswith(f"{feed}:"):
                fh.close()
                del self._writers[k]
        path = self.out_dir / feed / f"{feed}_{session}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        fh = gzip.open(path, "at", newline="", encoding="utf-8")
        w = csv.writer(fh)
        self._writers[key] = (fh, w, exists)
        return self._writers[key]

    def _poll(self, feed: str) -> int:
        path = FEEDS[feed]
        try:
            resp = self._client.get(f"{BASE_URL}{path}",
                                    params={"symbol": "*", "format": "csv"})
        except httpx.HTTPError as exc:
            self.errors += 1
            return 0
        if resp.status_code != 200:
            self.errors += 1
            return 0

        polled_at = datetime.now(timezone.utc).astimezone(ET_TZ)
        session = polled_at.date().isoformat()
        reader = csv.reader(io.StringIO(resp.text))
        try:
            header = next(reader)
        except StopIteration:
            return 0
        try:
            sym_idx = header.index("symbol")
        except ValueError:
            self.errors += 1
            return 0

        fh, w, had_header = self._writer(feed, session)
        if not had_header:
            w.writerow(["polled_at"] + header)
            self._writers[f"{feed}:{session}"] = (fh, w, True)

        stamp = polled_at.isoformat(timespec="milliseconds")
        n = 0
        for row in reader:
            if len(row) <= sym_idx:
                continue
            # CSV quotes the symbol; strip it before matching.
            if row[sym_idx].strip('"').upper() not in self.symbols:
                continue
            w.writerow([stamp] + row)
            n += 1
        fh.flush()
        return n

    # -------------------------------------------------------------------- loop
    def run(self, until: str | None = None, max_polls: int | None = None):
        def _sigint(*_):
            self._stop = True
            print("\nstopping after current poll...")
        signal.signal(signal.SIGINT, _sigint)

        stop_at = None
        if until:
            h, m = (until.split(":") + ["0"])[:2]
            now = datetime.now(ET_TZ)
            stop_at = now.replace(hour=int(h), minute=int(m), second=0,
                                  microsecond=0)
            if stop_at <= now:
                stop_at += timedelta(days=1)

        print(f"recording {len(self.symbols)} symbols, feeds={','.join(self.feeds)}, "
              f"every {self.interval}s -> {self.out_dir}")
        if stop_at:
            print(f"will stop at {stop_at:%Y-%m-%d %H:%M %Z}")
        print("Ctrl-C to stop.\n")

        t_start = time.time()
        while not self._stop:
            if stop_at and datetime.now(ET_TZ) >= stop_at:
                print("\nreached stop time")
                break
            if max_polls and self.polls >= max_polls:
                print("\nreached max polls")
                break
            tick = time.time()
            got = sum(self._poll(f) for f in self.feeds)
            self.polls += 1
            self.rows += got

            if self.polls % 10 == 0:
                el = time.time() - t_start
                sys.stdout.write(
                    f"\r  {self.polls:,} polls  {self.rows:,} rows  "
                    f"{self.errors} errors  {el/60:.1f} min  "
                    f"{self.rows/max(el,1e-9):.0f} rows/s   ")
                sys.stdout.flush()

            # Sleep the REMAINDER of the interval, not the whole interval, so
            # the cadence stays on a 1-second grid instead of drifting by the
            # request duration on every iteration.
            slack = self.interval - (time.time() - tick)
            if slack > 0:
                time.sleep(slack)

        self.close()
        el = time.time() - t_start
        print(f"\ndone: {self.polls:,} polls, {self.rows:,} rows, "
              f"{self.errors} errors in {el/60:.1f} min")

    def close(self):
        for fh, _, _ in self._writers.values():
            fh.close()
        self._writers.clear()
        self._client.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between polls (default 1.0)")
    ap.add_argument("--feeds", nargs="*", default=["quote", "ohlc"],
                    choices=list(FEEDS), help="default: quote ohlc")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="default: the whole QQQ+ETF universe")
    ap.add_argument("--until", default=None,
                    help="stop at this ET wall-clock time, e.g. 16:00")
    ap.add_argument("--max-polls", type=int, default=None)
    args = ap.parse_args()

    symbols = args.symbols or universe.load()
    rec = LiveRecorder(symbols, args.feeds, args.interval)
    rec.run(until=args.until, max_polls=args.max_polls)


if __name__ == "__main__":
    main()
