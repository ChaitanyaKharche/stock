"""Does the coverage record still count the sessions it cannot see?

    python -m trade_analysis.live_lab.ledger_test
    pytest trade_analysis/live_lab/ledger_test.py

This exists because on 2026-09-16 `ledger --lab-dir live_lab_data` printed
`usable 7/7 = 100.0%` while six consecutive completed sessions -- 2026-09-09 through
2026-09-16 -- had no record of any kind. Nothing crashed and nothing warned.

`coverage()` built its day set from ledger records alone, so a trading day with no record
was absent from the numerator AND the denominator. It was invisible rather than counted,
and no quantity of unrecorded sessions could move a percentage that only ever divided the
ledger by itself. The module docstring says a denominator you cannot reconstruct is not a
denominator; the function asserted one instead.

That is the same defect shape as everything else in this project -- a confident number
measuring nothing -- and it had no test, in a module whose own `record()` docstring boasts
that a false docstring claim was once caught by one. So: this.

The two halves are deliberately different kinds of assertion:

  * COUNTING (1-6) pins the denominator. `test_the_regression` is the specific 7/13 the
    old code got wrong, hardcoded, so a future refactor cannot quietly restore 100%.
  * NOT WRITING (7-10) pins the boundary that matters more. `coverage()` must never
    append, and `reconcile()` must never call a day MISSED when the daily file proves it
    ran -- the bug that once backfilled six good sessions as "the lab did not run", and
    the reason `--reconcile` may only run on the lab machine.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

from . import ledger

# A collected session, minus the date. `record()` writes more fields than this; coverage
# only reads `outcome` and `is_session`, and a fixture that mirrors the writer exactly
# would pass even if the writer's schema drifted.
_COLLECTED = {"outcome": "COLLECTED", "reason": "", "attempts": 1, "is_session": True}

# 2026-09-16 was a Wednesday. 09-12/13 were a weekend and 09-07 was Labor Day, so a
# calendar-driven denominator from 08-28 must land on 13 trading days, not 15 or 20.
_LAST_CLOSED = dt.date(2026, 9, 16)


def _lab(records=(), share_dailies=(), option_dailies=()) -> Path:
    """A throwaway lab directory with a ledger and whatever daily-file evidence."""
    root = Path(tempfile.mkdtemp(prefix="ledger_test_"))
    (root / "daily").mkdir(parents=True)
    (root / "shares" / "daily").mkdir(parents=True)
    with (root / ledger.LEDGER_NAME).open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    for stem in share_dailies:
        (root / "shares" / "daily" / f"{stem}.json").write_text("{}")
    for stem in option_dailies:
        (root / "daily" / f"{stem}.json").write_text("{}")
    return root


def _day(date_str, **over):
    return {"date": date_str, **_COLLECTED, **over}


# --------------------------------------------------------------------------- counting

def test_empty_ledger_invents_no_denominator():
    """No evidence at all must yield 0 days, not a calendar walked from nowhere."""
    cov = ledger.coverage(_lab())
    assert cov["trading_days"] == 0
    assert cov["coverage_pct"] == 0.0
    assert cov["days"] == {}


def test_the_regression_six_unrecorded_sessions_are_counted():
    """The exact number the old code got wrong. 7 recorded of 13 expected, not 7 of 7."""
    lab = _lab(
        records=[_day(d) for d in ("2026-08-28", "2026-08-31", "2026-09-01",
                                   "2026-09-02", "2026-09-03", "2026-09-04")]
        + [_day("2026-09-08", outcome="PARTIAL", reason="options blocked", attempts=3)],
        share_dailies=["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"],
    )
    cov = ledger.coverage(lab, today=_LAST_CLOSED)
    assert cov["trading_days"] == 13, cov["trading_days"]
    assert cov["usable"] == 7
    assert cov["coverage_pct"] == 53.8, cov["coverage_pct"]
    assert cov["counts"]["UNRECORDED"] == 6
    assert [d for d, r in cov["days"].items() if r["outcome"] == "UNRECORDED"] == [
        "2026-09-09", "2026-09-10", "2026-09-11",
        "2026-09-14", "2026-09-15", "2026-09-16",
    ]


def test_non_sessions_never_become_unrecorded():
    """A weekend or holiday is not a gap. Counting one would inflate the denominator."""
    cov = ledger.coverage(_lab(records=[_day("2026-09-04")]), today=_LAST_CLOSED)
    for closed in ("2026-09-05", "2026-09-06", "2026-09-07", "2026-09-12", "2026-09-13"):
        assert closed not in cov["days"], closed


def test_a_fully_recorded_run_still_reads_100pct():
    """The fix must not cry wolf: no gap when the ledger reaches the last closed day."""
    cov = ledger.coverage(_lab(records=[_day("2026-09-08")]), today=dt.date(2026, 9, 8))
    assert cov["trading_days"] == 1
    assert cov["coverage_pct"] == 100.0
    assert "UNRECORDED" not in cov["counts"]


def test_best_achieved_state_still_wins():
    """RANK gained a -1 floor for UNRECORDED; a real record must not fall through it.

    `coverage` reads RANK with a default, so a typo'd or unknown outcome lands on the
    default rather than raising. If that default sat at or above COLLECTED, a day that
    aborted at 09:20 and collected at 11:34 could read as ABORTED.
    """
    lab = _lab(records=[_day("2026-09-08", outcome="ABORTED", reason="no feed"),
                        _day("2026-09-08")])
    cov = ledger.coverage(lab, today=dt.date(2026, 9, 8))
    assert cov["days"]["2026-09-08"]["outcome"] == "COLLECTED"
    assert cov["usable"] == 1


def test_today_is_excluded_until_the_close():
    """A session in progress has legitimately not been recorded yet.

    Counting it would report a hole that does not exist, and at 09:31 every morning the
    tool would claim the lab had already missed the day.
    """
    lab = _lab(records=[_day("2026-09-15")])
    real = ledger.now_et

    class _Fixed(dt.datetime):
        pass

    def at(hhmm):
        h, m = hhmm
        return lambda: dt.datetime(2026, 9, 16, h, m)

    try:
        ledger.now_et = at((9, 31))
        assert ledger._last_closed_session() == dt.date(2026, 9, 15), "pre-close"
        ledger.now_et = at((16, 0))
        assert ledger._last_closed_session() == dt.date(2026, 9, 16), "at the close"
        ledger.now_et = at((22, 33))          # when the gap was actually found
        assert ledger._last_closed_session() == dt.date(2026, 9, 16), "post-close"
    finally:
        ledger.now_et = real
    assert lab.exists()


# ------------------------------------------------------------------------ not writing

def test_coverage_never_appends():
    """Reporting must be side-effect free, or reading the record would alter it."""
    lab = _lab(records=[_day("2026-09-08")])
    path = lab / ledger.LEDGER_NAME
    before = path.read_bytes()
    ledger.coverage(lab, today=_LAST_CLOSED)
    ledger.coverage(lab, today=_LAST_CLOSED)
    assert path.read_bytes() == before


def test_unrecorded_is_synthetic_and_never_persisted():
    """UNRECORDED is a reading of absence, not a state the lab ever reached.

    Writing it would turn a gap in the record into a claim about what happened.
    """
    lab = _lab(records=[_day("2026-09-08")])
    assert ledger.coverage(lab, today=_LAST_CLOSED)["counts"]["UNRECORDED"] == 6
    assert "UNRECORDED" not in (lab / ledger.LEDGER_NAME).read_text()


def test_reconcile_will_not_call_a_day_missed_when_a_daily_file_proves_it_ran():
    """The regression that turned a working forward test into a phantom one.

    `_evidence_days()` exists because the first `reconcile` backfilled six good sessions
    as "the lab did not run". It globs daily/ on the host it runs on -- which is also why
    --reconcile may only run on the lab machine, and why the 2026-09-16 audit refused to
    run it from a fresh clone.
    """
    lab = _lab(share_dailies=["2026-09-09"], option_dailies=["2026-09-10"])
    ledger.reconcile(lab, today=dt.date(2026, 9, 10))
    outcomes = {r["date"]: r["outcome"] for r in ledger._read(lab)}
    assert outcomes.get("2026-09-09") == "COLLECTED", outcomes
    assert outcomes.get("2026-09-10") == "COLLECTED", outcomes
    assert "MISSED" not in outcomes.values()


def test_reconcile_is_idempotent_and_records_a_genuine_miss():
    """A day with neither a record nor a daily file is MISSED -- once, not once per run."""
    lab = _lab(records=[_day("2026-09-08")], share_dailies=["2026-09-08"])
    first = ledger.reconcile(lab, today=dt.date(2026, 9, 10))
    second = ledger.reconcile(lab, today=dt.date(2026, 9, 10))
    missed = [r for r in ledger._read(lab) if r["outcome"] == "MISSED"]
    assert {w["date"] for w in first if w["outcome"] == "MISSED"} == {"2026-09-09"}
    assert second == [], second
    assert len(missed) == 1, missed


def test_reconcile_with_an_injected_today_ignores_the_wall_clock():
    """Two runs, same arguments, different hour -> the SAME ledger.

    This caught a real bug. `reconcile` decided whether its last day was finished with
    `now_et().time() >= 16:00`, even when the caller had passed `today` explicitly. So a
    backfill of one date range produced a different durable record depending on when it
    ran: before 16:00 the last day was in progress, after 16:00 it was MISSED. The test
    above passed at 22:33 ET and failed at 19:27 the next day, which is how it surfaced.

    Determinism is not a nicety here. The ledger's whole job is being the record, and a
    record that depends on the hour you asked is not one.
    """
    real = ledger.now_et

    def at(h, m):
        return lambda: dt.datetime(2026, 9, 10, h, m)

    outcomes = []
    try:
        for h, m in ((9, 31), (15, 59), (16, 0), (23, 59)):
            lab = _lab(records=[_day("2026-09-08")], share_dailies=["2026-09-08"])
            ledger.now_et = at(h, m)
            ledger.reconcile(lab, today=dt.date(2026, 9, 10))
            outcomes.append({r["date"]: r["outcome"] for r in ledger._read(lab)
                             if r["outcome"] == "MISSED"})
    finally:
        ledger.now_et = real
    assert all(o == outcomes[0] for o in outcomes), outcomes
    assert outcomes[0] == {"2026-09-09": "MISSED"}, outcomes[0]


def test_reconcile_without_a_today_still_uses_the_close_to_decide():
    """The production path must keep its wall-clock behaviour.

    autostart calls `reconcile()` with no `today`. Before the close the current day is in
    progress and must not be MISSED; after it, an unrecorded day must be. Fixing the
    injected case above must not have flattened this one.
    """
    real = ledger.now_et

    def at(h, m):
        return lambda: dt.datetime(2026, 9, 10, h, m)

    try:
        lab = _lab(records=[_day("2026-09-08")], share_dailies=["2026-09-08"])
        ledger.now_et = at(9, 31)                   # market open, 09-10 in progress
        ledger.reconcile(lab)
        missed = {r["date"] for r in ledger._read(lab) if r["outcome"] == "MISSED"}
        assert missed == {"2026-09-09"}, missed

        lab2 = _lab(records=[_day("2026-09-08")], share_dailies=["2026-09-08"])
        ledger.now_et = at(16, 5)                   # after the close
        ledger.reconcile(lab2)
        missed2 = {r["date"] for r in ledger._read(lab2) if r["outcome"] == "MISSED"}
        assert missed2 == {"2026-09-09", "2026-09-10"}, missed2
    finally:
        ledger.now_et = real


def test_repeated_failures_still_increment_attempts():
    """`record(dedupe=False)` must append every call.

    An earlier version deduped unconditionally while claiming the counter still rose. It
    did not -- a suppressed write suppressed the count with it -- so a day that failed all
    day recorded `attempts: 1`, and "aborted once" became indistinguishable from "aborted
    every 30 minutes until the close". That claim lived in the docstring; this pins it.
    """
    lab = _lab()
    for _ in range(3):
        ledger.record("2026-09-09", "ABORTED", "no feed", lab)
    recs = [r for r in ledger._read(lab) if r["date"] == "2026-09-09"]
    assert [r["attempts"] for r in recs] == [1, 2, 3], recs

    # dedupe=True is for idempotent backfill and must suppress the identical row.
    assert ledger.record("2026-09-09", "ABORTED", "no feed", lab, dedupe=True) is False


# ------------------------------------------------------------------------------- runner

CHECKS = [(name, fn) for name, fn in sorted(globals().items())
          if name.startswith("test_") and callable(fn)]


def main() -> int:
    ok = True
    print("=" * 78)
    print("SESSION LEDGER -- COVERAGE DENOMINATOR AND WRITE BOUNDARY")
    print("=" * 78)
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  [PASS] {name[5:].replace('_', ' ')}")
        except AssertionError as exc:
            ok = False
            print(f"  [FAIL] {name[5:].replace('_', ' ')}\n         {exc}")
        except Exception as exc:                              # noqa: BLE001
            ok = False
            print(f"  [FAIL] {name[5:].replace('_', ' ')}\n         raised {exc!r}")
    print(f"\n  {len(CHECKS)} checks")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print("\n  A gap you can explain is data. A gap nobody counts is not even a gap.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
