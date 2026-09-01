"""What is in the archive right now, and what is missing and why.

The directory tree cannot answer this. An absent file means one of four
completely different things - never downloaded, no data exists, not entitled, or
it errored - and only the manifest distinguishes them. Run this after (or during)
any download to see real coverage rather than inferring it from file counts.
"""
from __future__ import annotations

import argparse
import json

from .config import DATA_ROOT, PARQUET_DIR, RAW_DIR, REFERENCE_DIR
from .manifest import Manifest


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:,.1f}{unit}"
        n /= 1024
    return f"{n:,.1f}PB"


def show_entitlements():
    man = Manifest()
    rows = man.entitlements()
    if not rows:
        print("no entitlement probe recorded yet - run probe_entitlements")
        return
    print("\n=== ENTITLEMENTS (measured against the live gateway) ===")
    print(f"{'LAYER':<26} {'AVAIL':<7} {'FIRST DATE':<12} NOTE")
    print("-" * 96)
    for r in rows:
        note = (r["note"] or "")
        # The 403 body is the useful part when a layer is unavailable; the
        # "binary search, N probes" boilerplate is not.
        if r["available"]:
            note = ""
        print(f"{r['layer']:<26} {'yes' if r['available'] else 'NO':<7} "
              f"{r['first_date'] or '-':<12} {note[:52]}")


def show_downloads(verbose: bool = False):
    man = Manifest()
    rows = man.summary()
    if not rows:
        print("\nno downloads recorded yet")
        return
    print("\n=== DOWNLOAD PROGRESS ===")
    print(f"{'LAYER':<26} {'STATUS':<13} {'REQUESTS':>9} {'ROWS':>14} {'SIZE':>10}")
    print("-" * 78)
    tot_bytes = tot_rows = 0
    for r in rows:
        b = r["bytes"] or 0
        n = r["rows"] or 0
        if r["status"] == "OK":
            tot_bytes += b
            tot_rows += n
        print(f"{r['layer']:<26} {r['status']:<13} {r['n']:>9,} {n:>14,} "
              f"{human(b):>10}")
    print("-" * 78)
    print(f"{'TOTAL (OK only)':<40} {tot_rows:>14,} {human(tot_bytes):>10}")

    fails = man.failures(limit=15)
    if fails:
        print(f"\n=== TRANSIENT FAILURES (will be retried on next run) ===")
        for f in fails:
            print(f"  {f['layer']:<24} {f['symbol']:<8} {f['key']:<22} "
                  f"{(f['detail'] or '')[:60]}")


def show_disk():
    print("\n=== DISK ===")
    for label, d in (("reference", REFERENCE_DIR), ("raw (csv.gz)", RAW_DIR),
                     ("parquet", PARQUET_DIR)):
        if not d.exists():
            print(f"  {label:<16} -")
            continue
        n = size = 0
        for p in d.rglob("*"):
            if p.is_file():
                n += 1
                size += p.stat().st_size
        print(f"  {label:<16} {n:>8,} files  {human(size):>10}")
    try:
        import shutil
        du = shutil.disk_usage(DATA_ROOT)
        print(f"  {'free on volume':<16} {human(du.free):>21}")
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entitlements", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.json:
        man = Manifest()
        print(json.dumps({"entitlements": man.entitlements(),
                          "downloads": man.summary(),
                          "failures": man.failures(limit=100)}, indent=2))
        return

    print(f"archive root: {DATA_ROOT}")
    show_entitlements()
    if not args.entitlements:
        show_downloads()
        show_disk()


if __name__ == "__main__":
    main()
