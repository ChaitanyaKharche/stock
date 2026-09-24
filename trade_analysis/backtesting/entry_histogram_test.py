"""Tests for the entry-time histogram.

The part that can lie here is `peaks()`. If it under-detects, a two-humped distribution
reads as one hump and the median keeps getting quoted as a cluster -- which is the exact
error this script was written to stop. So the central test builds the bimodal case with
the median in the valley and asserts the detector sees two modes and excludes the median.
"""

from __future__ import annotations

import csv
import datetime as dt
import statistics

import pytest

from trade_analysis.backtesting.entry_histogram import (
    RTH_OPEN,
    bucket_label,
    buckets,
    load,
    peaks,
)


def _rows(times, net=1.0):
    """Build (entry, symbol, right, net) rows from 'HH:MM' strings."""
    day = dt.date(2026, 9, 21)
    out = []
    for t in times:
        h, m = (int(x) for x in t.split(":"))
        out.append((dt.datetime.combine(day, dt.time(h, m)), "QQQ", "call", net))
    return out


# ------------------------------------------------------------------ peak detection

def test_single_peak_is_one_mode():
    counts = [1, 3, 9, 20, 9, 3, 1]
    assert peaks(counts)[0] == [3]


def test_two_humps_separated_by_a_deep_valley_are_two_modes():
    counts = [2, 18, 20, 4, 1, 2, 19, 17, 3]
    assert len(peaks(counts)[0]) == 2


def test_jagged_adjacent_buckets_are_not_two_modes():
    """Without the valley rule, 19/20 side by side would read as two peaks."""
    counts = [1, 19, 20, 19, 18, 4, 1]
    assert len(peaks(counts)[0]) == 1


def test_a_small_bump_is_not_a_mode():
    """A 3-count wiggle next to a 40-count peak is noise, not a second mode."""
    counts = [1, 40, 8, 1, 3, 2, 1]
    assert peaks(counts)[0] == [1]


def test_flat_all_zero_returns_no_modes():
    assert peaks([0, 0, 0, 0]) == ([], [])


# ------------------- THE CASE THE WRITE-UPS GOT WRONG: median lands in the valley

def test_bimodal_median_falls_in_a_trough_he_barely_trades():
    """Morning cluster + afternoon cluster. The median hour holds almost nothing.

    This is the shape that makes 'median entry 11:39' a false statement about where he
    trades. If this test ever fails, the histogram script has stopped being able to
    detect the error it exists to detect.

    The two clusters must be balanced (41 and 41) for the median to land between them.
    An earlier version of this test put 42 in the morning and 40 in the afternoon, which
    pushed the median inside the morning cluster and made the test fail for a reason that
    had nothing to do with the code. Median here is 12:15, holding zero trades.
    """
    times = ["09:45"] * 20 + ["10:00"] * 21 + ["14:30"] * 21 + ["14:45"] * 20
    rows = _rows(times)
    counts, _pnls, pre, post = buckets(rows, 15)

    mins = [e.hour * 60 + e.minute for e, _, _, _ in rows]
    med = statistics.median(mins)
    open_min = RTH_OPEN.hour * 60 + RTH_OPEN.minute
    med_idx = (int(med) - open_min) // 15

    found, _margins = peaks(counts)
    assert len(found) == 2, "two clusters must read as two modes"
    assert med_idx not in found, "the median must not coincide with a mode here"
    assert counts[med_idx] == 0, "he makes zero trades at his own median"
    assert not pre and not post


# -------------------------------------------- the valley rule must respect count noise

# His real journal, 15-minute buckets, 443 round trips over 158 days, run 2026-09-22.
HIS_JOURNAL = [25, 41, 29, 18, 20, 19, 29, 20, 22, 26, 18, 18, 12, 14, 13,
               14, 15, 12, 8, 12, 12, 11, 11, 7, 8, 8]


def test_a_dip_inside_two_sigma_of_counting_noise_is_not_a_split():
    """10 -> 5 -> 10 is where the old and new valley rules actually disagree.

    Old fixed-ratio rule: splits if valley < 0.70 * 10 = 7.0, and 5 < 7.0, so it split.
    Noise-aware rule: needs valley <= 10 - 2*sqrt(10) = 3.7, and 5 > 3.7, so it does not.
    A dip of 5 on a count of 10 is 1.6 standard deviations of Poisson noise -- the kind of
    wiggle you get for free from 443 trades landing in 26 buckets.

    This is the test that makes the rule change real. An earlier version of this test used
    25 -> 20 -> 25, which BOTH rules reject, so it passed against the unfixed code and
    proved nothing.
    """
    counts = [1, 10, 5, 10, 1]
    idx, _ = peaks(counts, sigmas=2.0)
    assert len(idx) == 1, "a 1.6-sigma dip must not read as two modes"


def test_his_real_journal_split_is_barely_over_the_line():
    """The measured shape: one peak at 09:45, then a bumpy declining plateau.

    HONEST NOTE: the noise-aware rule does NOT change the verdict on his journal. Both the
    old fixed-ratio rule and this one return two modes. The peak pair is (41, 29) with a
    valley of 18; the old rule needed valley < 20.3 and the new one needs <= 18.2, and 18
    squeaks under both. What changed is that the margin is now reported, so "2 modes" can
    no longer be read as strong evidence.

    2.04 sigma is pinned tightly here on purpose. It is the number that justifies calling
    the shape right-skewed rather than bimodal, and if the margin computation drifts this
    test should fail rather than quietly relabel a plateau as two humps.
    """
    idx, margins = peaks(HIS_JOURNAL)
    assert idx[0] == 1, "the busiest bucket is 09:45"
    real = [m for m in margins if m is not None]
    assert real, "expected at least one reported split margin"
    assert 1.95 < max(real) < 2.15, "measured margin is 2.04 sigma"
    assert max(real) < 3.0, "must never be sold as confidently bimodal"


def test_a_genuinely_bimodal_shape_clears_comfortably():
    """Contrast case: real humps clear by a wide margin, his journal does not."""
    _idx, margins = peaks([2, 18, 20, 4, 1, 2, 19, 17, 3])
    assert max(m for m in margins if m is not None) > 3.5


def test_margins_align_with_returned_indices():
    idx, margins = peaks(HIS_JOURNAL)
    assert len(idx) == len(margins)
    assert margins[0] is None, "the first mode has nothing to be separated from"


def test_his_median_is_hours_past_his_busiest_bucket():
    """The actual defect, restated in the measured numbers: mode 09:45, median 11:44."""
    idx, _ = peaks(HIS_JOURNAL)
    mode_i = max(idx, key=lambda i: HIS_JOURNAL[i])
    # cumulative count to locate the median bucket
    n = sum(HIS_JOURNAL)
    run, med_i = 0, None
    for i, c in enumerate(HIS_JOURNAL):
        run += c
        if run >= n / 2:
            med_i = i
            break
    assert mode_i == 1, "busiest bucket is 09:45"
    assert med_i == 8, "median falls in the 11:30 bucket"
    assert (med_i - mode_i) * 15 >= 90, "median lands at least 90 min after the mode"


# ----------------------------------------------------------------------- bucketing

def test_entries_outside_rth_are_counted_not_silently_dropped():
    """Dropping them is how you manufacture a clean peak out of a messy one."""
    rows = _rows(["09:00", "10:00", "16:30"])
    counts, _pnls, pre, post = buckets(rows, 15)
    assert len(pre) == 1
    assert len(post) == 1
    assert sum(counts) == 1


def test_no_bucket_straddles_the_open():
    counts, _pnls, _pre, _post = buckets(_rows(["09:30"]), 15)
    assert counts[0] == 1
    assert bucket_label(0, 15) == "09:30"


def test_bucket_labels_advance_by_width():
    assert bucket_label(1, 15) == "09:45"
    assert bucket_label(2, 30) == "10:30"
    assert bucket_label(4, 60) == "13:30"


def test_pnl_is_grouped_with_its_own_bucket():
    day = dt.date(2026, 9, 21)
    rows = [
        (dt.datetime.combine(day, dt.time(9, 35)), "QQQ", "call", 100.0),
        (dt.datetime.combine(day, dt.time(9, 40)), "QQQ", "call", 300.0),
        (dt.datetime.combine(day, dt.time(14, 5)), "QQQ", "put", -50.0),
    ]
    counts, pnls, _pre, _post = buckets(rows, 15)
    assert counts[0] == 2
    assert statistics.fmean(pnls[0]) == 200.0
    assert pnls[18] == [-50.0]


# --------------------------------------------------------------------------- loader

def test_load_rejects_a_missing_csv_with_the_reason(tmp_path):
    with pytest.raises(SystemExit) as e:
        load(str(tmp_path / "nope.csv"))
    assert "gitignored" in str(e.value)


def test_load_skips_unparseable_timestamps_and_counts_them(tmp_path):
    p = tmp_path / "round_trips.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["entry_ts", "symbol", "right", "net"])
        w.writeheader()
        w.writerow({"entry_ts": "2026-09-21T09:45:00", "symbol": "QQQ",
                    "right": "call", "net": "10"})
        w.writerow({"entry_ts": "not-a-time", "symbol": "QQQ",
                    "right": "call", "net": "10"})
    rows, bad = load(str(p))
    assert len(rows) == 1
    assert bad == 1


def test_load_filters_by_symbol(tmp_path):
    p = tmp_path / "round_trips.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["entry_ts", "symbol", "right", "net"])
        w.writeheader()
        w.writerow({"entry_ts": "2026-09-21T09:45:00", "symbol": "QQQ",
                    "right": "call", "net": "10"})
        w.writerow({"entry_ts": "2026-09-21T10:45:00", "symbol": "SPY",
                    "right": "put", "net": "-5"})
    rows, _ = load(str(p), symbol="qqq")
    assert len(rows) == 1
    assert rows[0][1] == "QQQ"


def test_rows_come_back_in_time_order(tmp_path):
    p = tmp_path / "round_trips.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["entry_ts", "symbol", "right", "net"])
        w.writeheader()
        for t in ["2026-09-21T14:05:00", "2026-09-21T09:45:00"]:
            w.writerow({"entry_ts": t, "symbol": "QQQ", "right": "call", "net": "1"})
    rows, _ = load(str(p))
    assert [r[0].time().hour for r in rows] == [9, 14]
