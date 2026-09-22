"""Entry-time histogram for the discretionary journal.

Why this exists: two research documents quoted "median entry 11:39" as though it were the
hour he trades. A median is not a location. For a two-humped distribution it lands in the
valley between the humps -- an hour that may hold almost no trades. Nobody ever looked at
the shape, because `research/round_trips.csv` is gitignored and only exists on the lab
machine.

This script is descriptive. It computes no p-value and tests no hypothesis. Its one job is
to print the shape so the median stops being quoted as a cluster.

    python -m trade_analysis.backtesting.entry_histogram
    python -m trade_analysis.backtesting.entry_histogram --bucket 30 --symbol QQQ
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRADES = os.path.join(ROOT, "research", "round_trips.csv")

RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)


def load(path, symbol=None):
    """Read round_trips.csv -> list of (entry_datetime, symbol, right, net)."""
    if not os.path.exists(path):
        raise SystemExit(
            "round_trips.csv not found at %s\n"
            "It is gitignored (.gitignore:66 '*.csv'), so it only exists on the lab\n"
            "machine. Run this there." % path
        )
    out, bad = [], 0
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if symbol and r.get("symbol", "").upper() != symbol.upper():
                continue
            try:
                ent = dt.datetime.fromisoformat(r["entry_ts"]).replace(tzinfo=None)
            except (ValueError, KeyError):
                bad += 1
                continue
            try:
                net = float(r.get("net", "nan"))
            except ValueError:
                net = float("nan")
            out.append((ent, r.get("symbol", "?"), r.get("right", "?"), net))
    out.sort(key=lambda t: t[0])
    return out, bad


def buckets(rows, width_min):
    """Group entries into fixed-width buckets from 09:30. Returns ordered bucket list.

    Buckets are minutes-since-09:30 so nothing straddles the open. Entries outside RTH
    are kept in their own bucket rather than dropped -- dropping them is how you
    accidentally manufacture a clean single peak.
    """
    n_buckets = (
        (RTH_CLOSE.hour * 60 + RTH_CLOSE.minute) - (RTH_OPEN.hour * 60 + RTH_OPEN.minute)
    ) // width_min
    counts = [0] * n_buckets
    pnls = [[] for _ in range(n_buckets)]
    pre, post = [], []
    open_min = RTH_OPEN.hour * 60 + RTH_OPEN.minute
    for ent, _sym, _right, net in rows:
        m = ent.hour * 60 + ent.minute - open_min
        if m < 0:
            pre.append(net)
            continue
        idx = m // width_min
        if idx >= n_buckets:
            post.append(net)
            continue
        counts[idx] += 1
        pnls[idx].append(net)
    return counts, pnls, pre, post


def bucket_label(idx, width_min):
    open_min = RTH_OPEN.hour * 60 + RTH_OPEN.minute
    a = open_min + idx * width_min
    return "%02d:%02d" % (a // 60, a % 60)


def peaks(counts, min_share=0.40, valley_share=0.70):
    """Local maxima with a prominence rule.

    A bucket is a peak if it is >= both neighbours, holds at least `min_share` of the
    biggest bucket, and is separated from the previous peak by a valley dropping below
    `valley_share` of the smaller of the two. Without the valley rule every jagged pair
    of adjacent buckets reads as two modes.
    """
    if not any(counts):
        return []
    top = max(counts)
    cand = []
    for i, c in enumerate(counts):
        lo = counts[i - 1] if i > 0 else -1
        hi = counts[i + 1] if i + 1 < len(counts) else -1
        if c >= lo and c >= hi and c >= min_share * top:
            cand.append(i)
    kept = []
    for i in cand:
        if not kept:
            kept.append(i)
            continue
        j = kept[-1]
        valley = min(counts[j : i + 1]) if i > j else counts[i]
        if valley < valley_share * min(counts[i], counts[j]):
            kept.append(i)
        elif counts[i] > counts[j]:
            kept[-1] = i
    return kept


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", type=int, default=15, help="bucket width in minutes")
    ap.add_argument("--symbol", default=None, help="restrict to one symbol")
    ap.add_argument("--csv", default=TRADES)
    a = ap.parse_args(argv)

    rows, bad = load(a.csv, a.symbol)
    if not rows:
        raise SystemExit("no rows matched")

    counts, pnls, pre, post = buckets(rows, a.bucket)
    days = {e.date() for e, _, _, _ in rows}
    mins = [e.hour * 60 + e.minute for e, _, _, _ in rows]
    med = statistics.median(mins)
    med_t = "%02d:%02d" % (int(med) // 60, int(med) % 60)

    print("=" * 66)
    print("ENTRY-TIME HISTOGRAM  %s  bucket=%dm" % (a.symbol or "ALL", a.bucket))
    print("=" * 66)
    print("round trips       %6d" % len(rows))
    print("distinct days     %6d" % len(days))
    print("trades per day    %6.2f" % (len(rows) / len(days)))
    print("median entry      %6s" % med_t)
    if bad:
        print("unparseable ts    %6d" % bad)
    if pre:
        print("before 09:30      %6d" % len(pre))
    if post:
        print("after 16:00       %6d" % len(post))
    print()

    top = max(counts)
    scale = 44.0 / top if top else 0.0
    print("%-7s %5s  %-44s %s" % ("bucket", "n", "", "mean net $"))
    for i, c in enumerate(counts):
        lab = bucket_label(i, a.bucket)
        m = statistics.fmean([p for p in pnls[i] if p == p]) if pnls[i] else float("nan")
        mark = " <-- median" if int(med) // a.bucket == (
            (RTH_OPEN.hour * 60 + RTH_OPEN.minute) // a.bucket + i
        ) else ""
        bar = "#" * int(round(c * scale))
        print("%-7s %5d  %-44s %9.2f%s" % (lab, c, bar, m, mark))
    print()

    pk = peaks(counts)
    print("modes detected    %6d   at %s" % (
        len(pk), ", ".join(bucket_label(i, a.bucket) for i in pk) or "-"))

    # The headline question: does he actually trade at the median?
    open_min = RTH_OPEN.hour * 60 + RTH_OPEN.minute
    med_idx = (int(med) - open_min) // a.bucket
    if 0 <= med_idx < len(counts):
        share = 100.0 * counts[med_idx] / len(rows)
        print("median bucket     %6d trades (%.1f%% of all), vs %.1f%% if uniform"
              % (counts[med_idx], share, 100.0 / len(counts)))
        if len(pk) > 1 and med_idx not in pk:
            print()
            print("  >> The distribution is multi-modal and the median is NOT at a mode.")
            print("  >> Quoting '%s' as the hour he trades is therefore wrong." % med_t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
