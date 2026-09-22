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


def peaks(counts, min_share=0.40, sigmas=2.0):
    """Local maxima whose separating valley is deeper than counting noise.

    Returns (indices, margins) where margins[k] is how many standard deviations of
    Poisson counting noise separate peak k from the previous one. margins[0] is None.

    CORRECTED 2026-09-22. The first version required the valley to fall below a fixed
    `valley_share=0.70` of the smaller peak. That ignores the fact that bucket counts are
    themselves noisy: a bucket holding n trades varies by about sqrt(n) run to run, so at
    n=25 a dip to 19 is one standard deviation and means nothing. On the real journal that
    loose rule reported "2 modes" for what is actually one peak at 09:45 followed by a
    bumpy declining plateau -- a confident output that measured less than it claimed,
    which is this project's standard bug class.

    The rule now is: the valley must sit at least `sigmas` * sqrt(smaller peak) BELOW the
    smaller of the two peaks. Reporting the margin matters as much as the verdict; a split
    that clears by 0.1 sigma is not a finding.
    """
    if not any(counts):
        return [], []
    top = max(counts)
    cand = []
    for i, c in enumerate(counts):
        lo = counts[i - 1] if i > 0 else -1
        hi = counts[i + 1] if i + 1 < len(counts) else -1
        if c >= lo and c >= hi and c >= min_share * top:
            cand.append(i)
    kept, margins = [], []
    for i in cand:
        if not kept:
            kept.append(i)
            margins.append(None)
            continue
        j = kept[-1]
        valley = min(counts[j : i + 1]) if i > j else counts[i]
        lo_peak = min(counts[i], counts[j])
        noise = lo_peak ** 0.5 or 1.0
        margin = (lo_peak - valley) / noise
        if margin >= sigmas:
            kept.append(i)
            margins.append(margin)
        elif counts[i] > counts[j]:
            kept[-1] = i
    return kept, margins


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

    open_min0 = RTH_OPEN.hour * 60 + RTH_OPEN.minute
    med_bucket = (int(med) - open_min0) // a.bucket

    top = max(counts)
    scale = 26.0 / top if top else 0.0
    # `top1 $` is the single biggest trade in the bucket and `top1 %` its share of the
    # bucket's total. A bucket whose mean is positive only because of one fill is not a
    # time-of-day effect, it is one trade. Median net is shown for the same reason: mean
    # and median disagreeing in sign is the signature of a tail-driven bucket.
    print("%-7s %4s  %-26s %9s %9s %9s %6s"
          % ("bucket", "n", "", "mean $", "med $", "top1 $", "top1%"))
    for i, c in enumerate(counts):
        vals = [p for p in pnls[i] if p == p]
        lab = bucket_label(i, a.bucket)
        bar = "#" * int(round(c * scale))
        if not vals:
            print("%-7s %4d  %-26s %9s %9s %9s %6s"
                  % (lab, c, bar, "-", "-", "-", "-"))
            continue
        m = statistics.fmean(vals)
        md = statistics.median(vals)
        big = max(vals, key=abs)
        tot_abs = sum(vals)
        share = (100.0 * big / tot_abs) if tot_abs else float("nan")
        mark = " <-- median" if i == med_bucket else ""
        print("%-7s %4d  %-26s %+9.2f %+9.2f %+9.2f %5.0f%%%s"
              % (lab, c, bar, m, md, big, share, mark))
    print()

    pk, margins = peaks(counts)
    bits = []
    for k, i in enumerate(pk):
        s = bucket_label(i, a.bucket)
        if margins[k] is not None:
            s += " (+%.1f sigma)" % margins[k]
        bits.append(s)
    print("modes detected    %6d   at %s" % (len(pk), ", ".join(bits) or "-"))
    if len(pk) > 1 and min(m for m in margins if m is not None) < 3.0:
        print("  NOTE: the weakest split clears by only %.1f sigma of counting noise."
              % min(m for m in margins if m is not None))
        print("        Treat the shape as one peak plus a noisy tail, not clean humps.")

    # The headline question: does he actually trade at the median?
    if 0 <= med_bucket < len(counts):
        share = 100.0 * counts[med_bucket] / len(rows)
        print("median bucket     %6d trades (%.1f%% of all), vs %.1f%% if uniform"
              % (counts[med_bucket], share, 100.0 / len(counts)))
    if pk:
        mode_i = max(pk, key=lambda i: counts[i])
        mode_lab = bucket_label(mode_i, a.bucket)
        gap = (int(med) - (open_min0 + mode_i * a.bucket))
        print("busiest bucket    %6s (%d trades)" % (mode_lab, counts[mode_i]))
        if abs(gap) >= a.bucket:
            print()
            print("  >> He trades MOST at %s. The median is %s, %d min later."
                  % (mode_lab, med_t, gap))
            print("  >> The distribution is right-skewed: a long afternoon tail drags the")
            print("     median hours past the bucket he actually trades in. Quoting '%s'"
                  % med_t)
            print("     as 'the hour he trades' is therefore wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
