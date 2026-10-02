"""A stale chain the options arm does not trade must not abort the session.

    python -m trade_analysis.live_lab.foreign_options_gate_test

On 2026-10-02 the hotspot dropped at 11:51 and the scheduled task restarted autostart at
13:30, 14:00 and 14:30. Each restart ran the full preflight on all 16 symbols with the
market open. On Fridays the sector ETFs list a weekly expiring that day, their thin chains
carry quotes minutes old, and preflight printed `[FAIL] OPTIONS: XLY options DELAYED by
~21.0 min.` autostart correctly disabled the options arm for it -- then aborted the shares
arm too, because its "any line says DELAYED" check did not skip OPTIONS lines. Every
underlying and both traded chains (QQQ, SPY) were REAL-TIME. The last 2.5 hours were lost.

Lines are built with preflight's own constants and wording (preflight.py, the options
freshness check), so a change to that wording breaks these tests instead of the gate.
"""
from __future__ import annotations

import sys

from .autostart import _drop_foreign_options, _gate_verdict
from .preflight import FAIL, OK, OPTIONS_TAG

OPTION_SYMBOLS = ["QQQ", "SPY"]


def _opt_delayed(sym, minutes):
    return f"    {FAIL} {OPTIONS_TAG} {sym} options DELAYED by ~{minutes:.1f} min."


# The 2026-10-02 14:31 report, reduced to the lines that decided it.
REPORT_1002 = "\n".join([
    f"    {OK}   QQQ underlying REAL-TIME (age -0.1s)",
    f"    {OK}   XLY underlying REAL-TIME (age 0.2s)",
    f"    {OK}   QQQ options REAL-TIME (median quote age -52.3s)",
    f"    {OK}   SPY options REAL-TIME (median quote age -52.9s)",
    _opt_delayed("XLY", 21.0),
    _opt_delayed("XLV", 3.8),
    _opt_delayed("XLF", 2.3),
])


def test_foreign_chain_delay_no_longer_aborts():
    # The bug, pinned: without the filter this report stops every arm.
    assert _gate_verdict(REPORT_1002) == ("all", "feed is DELAYED; prices would be stale")
    kept, dropped = _drop_foreign_options(REPORT_1002, OPTION_SYMBOLS)
    assert len(dropped) == 3
    assert _gate_verdict(kept) is None


def test_traded_chain_delay_still_aborts():
    report = REPORT_1002 + "\n" + _opt_delayed("QQQ", 4.0)
    kept, dropped = _drop_foreign_options(report, OPTION_SYMBOLS)
    assert len(dropped) == 3
    assert _gate_verdict(kept) == ("all", "feed is DELAYED; prices would be stale")


def test_underlying_delay_is_never_dropped():
    line = f"    {FAIL} XLY underlying is DELAYED by ~3.0 min. Every entry"
    kept, dropped = _drop_foreign_options(line, OPTION_SYMBOLS)
    assert dropped == [] and kept == line
    assert _gate_verdict(kept) == ("all", "feed is DELAYED; prices would be stale")


def test_other_options_failure_forms_are_filtered_by_symbol():
    lines = [f"{FAIL} {OPTIONS_TAG} XLE: chain returned 0 usable rows",
             f"{FAIL} {OPTIONS_TAG} SPY: chain snapshot failed -- entitlement? (403)"]
    kept, dropped = _drop_foreign_options("\n".join(lines), OPTION_SYMBOLS)
    assert dropped == [lines[0]]
    assert kept == lines[1]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
