"""The download engine: turn a Layer into requests, run them, record outcomes.

One engine drives every layer because the only things that vary are the chunking
rule and the request kwargs, both of which live on the Layer. What is shared -
and is the part worth getting right once - is:

  RESUMPTION. Task lists are built, then filtered against the manifest, so a
  re-run costs nothing for work already settled. This is not an optimisation:
  the full archive is ~10^5 requests over many hours and WILL be interrupted.
  A pipeline that cannot resume cheaply is a pipeline that never finishes.

  A DISK GUARD. The tick layers are ~25 MB per symbol-session; a careless scope
  fills the volume and then every subsequent write fails in a way that looks
  like a data problem. The engine stops cleanly at a floor instead.

  ORDERING. Tasks run newest-date-first. If a run is cut short - and long ones
  are - the data you have is the data you most likely want, rather than a
  complete 2017 and nothing recent.

  BOUNDED CONCURRENCY. The gateway queues above the account's concurrent limit
  (4 here) and returns 429 past a queue depth of 16. Staying under the limit is
  faster than being throttled, so the pool size comes from config, not taste.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from . import layers as layers_mod
from . import universe
from .config import (DATA_ROOT, MAX_CONCURRENCY, RAW_DIR, STOCK_VENUE)
from .layers import Layer
from .manifest import Manifest
from .theta_v3 import ThetaV3, TerminalNotRunning
from .trading_days import month_chunks, trading_days

# Stop before the volume is actually full: a partially written 25 MB tick file
# is indistinguishable from a truncated download, and recovering means deleting
# files by hand. 15 GB is roughly one day of the widest tick scope.
DISK_FLOOR_BYTES = 15 * 1024 ** 3


@dataclass
class Task:
    layer: Layer
    symbol: str
    key: str            # manifest key: "2024-08" (monthly) or "2024-08-07"
    start: str
    end: str
    expiration: str | None = None

    @property
    def dest(self) -> Path:
        # Partitioned symbol/year so a directory listing stays usable at 10^5
        # files and a single year can be copied or deleted independently.
        year = self.key[:4]
        return (RAW_DIR / self.layer.dir_name / self.symbol / year
                / f"{self.symbol}_{self.key}.csv.gz")


def build_tasks(layer: Layer, symbols: list[str], start: str, end: str,
                available: dict[str, set[str]] | None = None,
                expand_to_symbol_floor: bool = False) -> list[Task]:
    """Expand a layer into concrete requests over [start, end].

    `available` is {symbol: {session dates}} from /stock/list/dates. It removes
    dates a symbol cannot have: CRWV has 451 sessions and FER 571, against 3,581
    for a mature name, so requesting the full range for them is ~3,000
    guaranteed-empty calls each.

    `expand_to_symbol_floor` additionally lets a symbol start EARLIER than
    `start`. This matters because the measured floors come from probing SPY, and
    SPY is unusually SHALLOW here - it is not on the UTP tape, so its stock
    history begins years after names that are, and many constituents reach back
    to ~2012. Using SPY's floor as a universal start therefore truncates them.

    It is opt-in, and off by default, because it must NOT override an explicit
    `--start`: "give me one day" has to mean one day. The orchestrator turns it
    on only when `start` was itself derived from the entitlement probe.
    """
    tasks: list[Task] = []
    start_d, end_d = _as_date(start), _as_date(end)

    for sym in symbols:
        avail = (available or {}).get(sym)
        lo_d = start_d
        if avail and expand_to_symbol_floor:
            lo_d = min(start_d, _as_date(min(avail)))

        if layer.chunk == "monthly":
            for lo, hi in month_chunks(lo_d, end_d):
                if avail is not None and not any(
                        d.isoformat() in avail for d in trading_days(lo, hi)):
                    continue     # month holds no session for this symbol
                key = f"{lo.year:04d}-{lo.month:02d}"
                tasks.append(Task(layer, sym, key, lo.isoformat(), hi.isoformat()))
        elif layer.chunk == "daily":
            for d in trading_days(lo_d, end_d):
                key = d.isoformat()
                if avail is not None and key not in avail:
                    continue
                tasks.append(Task(layer, sym, key, key, key))
        else:
            raise ValueError(f"unknown chunk rule {layer.chunk!r}")

    # Newest first - see module docstring.
    tasks.sort(key=lambda t: (t.key, t.symbol), reverse=True)
    return tasks


def _as_date(v) -> date:
    if isinstance(v, date):
        return v
    s = str(v)[:10]
    return date(int(s[:4]), int(s[5:7]), int(s[8:10]))


def _request(api: ThetaV3, task: Task):
    """Issue the one request this task represents."""
    layer = task.layer
    if layer.security == "stock":
        return api.fetch_csv(
            layer.path, task.dest,
            symbol=task.symbol,
            start_date=task.start.replace("-", ""),
            end_date=task.end.replace("-", ""),
            interval=layer.interval,
            venue=STOCK_VENUE,
        )

    # ---- options ----
    params = dict(symbol=task.symbol, interval=layer.interval)
    # expiration=* pulls every listed contract; max_dte then narrows it
    # server-side. This is the whole reason option coverage is affordable: the
    # alternative is one request per contract per day.
    params["expiration"] = "*" if layer.wildcard_expiration else task.expiration
    if layer.max_dte is not None:
        params["max_dte"] = layer.max_dte
    if layer.strike_range is not None:
        params["strike_range"] = layer.strike_range

    if layer.endpoint in ("eod", "open_interest") and layer.chunk == "monthly":
        # These take a date RANGE, not a single date.
        params["start_date"] = task.start.replace("-", "")
        params["end_date"] = task.end.replace("-", "")
    else:
        params["date"] = task.start.replace("-", "")
    return api.fetch_csv(layer.path, task.dest, **params)


def run_layer(layer: Layer, symbols: list[str], start: str, end: str,
              man: Manifest, dry_run: bool = False,
              concurrency: int | None = None,
              disk_floor: int = DISK_FLOOR_BYTES,
              available: dict[str, set[str]] | None = None,
              expand_to_symbol_floor: bool = False) -> dict:
    tasks = build_tasks(layer, symbols, start, end, available,
                        expand_to_symbol_floor)
    total_built = len(tasks)

    done = man.done_keys(layer.name)
    tasks = [t for t in tasks if (t.symbol, t.key) not in done]

    est = len(tasks) * layer.est_bytes_per_request
    print(f"\n=== {layer.name} ===")
    print(f"  {len(symbols)} symbols x {start}..{end}  "
          f"-> {total_built:,} requests, {len(tasks):,} outstanding "
          f"({total_built - len(tasks):,} already settled)")
    print(f"  estimated download: ~{est / 1024**3:.1f} GB   ({layer.notes})")
    if dry_run or not tasks:
        return {"layer": layer.name, "built": total_built,
                "outstanding": len(tasks), "est_bytes": est, "ran": 0}

    n_workers = concurrency or MAX_CONCURRENCY.get(layer.security, 2)
    counts = {"OK": 0, "NO_DATA": 0, "NOT_ENTITLED": 0, "ERROR": 0}
    bytes_got = 0
    t0 = time.time()
    lock = threading.Lock()
    stop = threading.Event()

    # One client per worker: httpx.Client is thread-safe, but a dedicated client
    # per thread keeps each worker's connection pool from contending on the
    # long-running streams these layers produce.
    clients = [ThetaV3() for _ in range(n_workers)]
    local = threading.local()

    def worker(task: Task):
        if stop.is_set():
            return None
        api = getattr(local, "api", None)
        if api is None:
            with lock:
                api = local.api = clients.pop()
        try:
            res = _request(api, task)
        except TerminalNotRunning:
            stop.set()
            raise
        man.record(layer.name, task.symbol, task.key, res.status,
                   res.rows, res.bytes_written,
                   str(res.path) if res.path else None, res.detail)
        return task, res

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(worker, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    out = fut.result()
                except TerminalNotRunning as exc:
                    print(f"\n  ABORT: {exc}")
                    break
                except Exception as exc:                       # noqa: BLE001
                    # A worker dying must not take the run with it; the task
                    # stays unsettled and the next run retries it.
                    counts["ERROR"] += 1
                    print(f"\n  worker error: {type(exc).__name__}: {exc}")
                    continue
                if out is None:
                    continue
                task, res = out
                counts[res.status] = counts.get(res.status, 0) + 1
                bytes_got += res.bytes_written

                if res.status == "NOT_ENTITLED":
                    # The whole layer is unavailable, not just this request.
                    # Continuing would burn thousands of calls to learn the same
                    # thing, so stop and say so loudly.
                    print(f"\n  NOT ENTITLED - aborting layer: {res.detail[:160]}")
                    man.record_entitlement(layer.name, False, None, res.detail)
                    stop.set()
                    break

                if i % 25 == 0 or i == len(tasks):
                    rate = i / max(time.time() - t0, 1e-9)
                    eta = (len(tasks) - i) / max(rate, 1e-9)
                    free = shutil.disk_usage(DATA_ROOT).free
                    sys.stdout.write(
                        f"\r  {i:,}/{len(tasks):,}  ok={counts['OK']:,} "
                        f"empty={counts['NO_DATA']:,} err={counts['ERROR']:,}  "
                        f"{bytes_got/1024**3:.2f}GB  {rate:.1f} req/s  "
                        f"eta {eta/3600:.1f}h  free {free/1024**3:.0f}GB   ")
                    sys.stdout.flush()
                    if free < disk_floor:
                        print(f"\n  DISK FLOOR reached ({free/1024**3:.1f}GB "
                              f"free) - stopping this layer cleanly.")
                        stop.set()
                        break
    finally:
        for c in clients:
            c.close()
        # Clients handed out to workers are closed by GC; explicit close on the
        # pool leftovers is what matters for the sockets held open right now.

    dt = time.time() - t0
    print(f"\n  done in {dt/60:.1f} min: ok={counts['OK']:,} "
          f"empty={counts['NO_DATA']:,} err={counts['ERROR']:,} "
          f"{bytes_got/1024**3:.2f} GB")
    return {"layer": layer.name, "built": total_built, "ran": sum(counts.values()),
            "counts": counts, "bytes": bytes_got, "seconds": dt}


def resolve_symbols(layer: Layer, override: list[str] | None) -> list[str]:
    if override:
        return [s.upper() for s in override]
    if layer.scope == "etf":
        return universe.ETFS
    syms = universe.load(tradable_only=True,
                         options_only=(layer.security == "option"))
    return syms


def main():
    ap = argparse.ArgumentParser(
        description="Download ThetaData layers into the archive.",
        epilog="e.g.  python -m trade_analysis.bulk_download.download "
               "--layers stock.ohlc.1m --start 2016-01-04")
    ap.add_argument("--layers", nargs="+", default=None,
                    help="layer names, or 'tier:N' for every layer in tier N")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="override the layer's default symbol scope")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD; default = layer floor")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--dry-run", action="store_true",
                    help="print request counts and size estimates, download nothing")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="list known layers and exit")
    ap.add_argument("--no-available-dates", action="store_true",
                    help="do not clip task lists to each symbol's known "
                         "sessions (requests every date in range instead)")
    args = ap.parse_args()

    if args.list:
        print(f"{'LAYER':<26} {'TIER':<5} {'CHUNK':<8} {'SCOPE':<6} NOTES")
        for l in sorted(layers_mod.LAYERS.values(), key=lambda x: (x.tier, x.name)):
            print(f"{l.name:<26} {l.tier:<5} {l.chunk:<8} {l.scope:<6} {l.notes[:60]}")
        return

    if not args.layers:
        ap.error("--layers is required (or use --list to see what exists)")

    names: list[str] = []
    for spec in args.layers:
        if spec.startswith("tier:"):
            t = int(spec.split(":")[1])
            names += [l.name for l in layers_mod.by_tier(t, t)]
        else:
            names.append(spec)

    from .probe_entitlements import load_entitlements
    try:
        ent = load_entitlements()["layers"]
    except FileNotFoundError:
        ent = {}

    # Per-symbol session lists, if download_reference has been run. Absent is
    # fine - it just means more empty requests, not wrong data.
    from .download_reference import load_available_dates
    available = None if args.no_available_dates else load_available_dates("trade")
    if available:
        print(f"clipping task lists to {len(available)} symbols' actual "
              f"session dates")
    elif not args.no_available_dates:
        print("note: no available_dates on disk - run download_reference first "
              "to avoid requesting sessions that cannot exist")

    man = Manifest()
    results = []
    for name in names:
        layer = layers_mod.get(name)
        symbols = resolve_symbols(layer, args.symbols)

        # Start from the MEASURED floor for this layer, not a guess. Falling
        # back to the layer's own probe name lets e.g. option.quote.1m.0dte
        # inherit the option.quote.1m floor.
        start = args.start
        start_from_probe = start is None
        if start is None:
            # Prefer the layer's own measured floor; fall back to the layer it
            # declares as its floor source (a 0DTE slice inherits from the
            # probed option.quote.1m, a 5m resample from stock.ohlc.1m).
            rec = ent.get(name) or ent.get(layer.floor_layer or "") or {}
            start = rec.get("first_date")
            if not start:
                print(f"skipping {name}: no measured floor and no --start given "
                      f"(run probe_entitlements, or pass --start)")
                continue
        try:
            results.append(run_layer(layer, symbols, start, args.end, man,
                                     dry_run=args.dry_run,
                                     concurrency=args.concurrency,
                                     available=available,
                                     expand_to_symbol_floor=start_from_probe))
        except TerminalNotRunning as exc:
            print(f"\nFATAL: {exc}")
            sys.exit(2)

    if args.dry_run:
        tot = sum(r["est_bytes"] for r in results if "est_bytes" in r)
        req = sum(r.get("outstanding", 0) for r in results)
        print(f"\nTOTAL outstanding: {req:,} requests, "
              f"~{tot/1024**3:.1f} GB estimated")
        print(f"free on volume: {shutil.disk_usage(DATA_ROOT).free/1024**3:.0f} GB")


if __name__ == "__main__":
    main()
