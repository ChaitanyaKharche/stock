"""Does autostart's abort gate still agree with what preflight actually prints?

    python -m trade_analysis.live_lab.preflight_gate_test

This exists because that coupling broke silently and would have cost every session from
2026-09-08 onward.

autostart decides whether to abort by reading preflight's stdout. It used to classify a
line as non-fatal with `"REACHABLE from" not in line` -- a match against the check's
English. On 2026-09-04 the exposure check was rewritten to query Windows Firewall instead
of self-connecting to the LAN address, which changed the wording to "N inbound ALLOW
rule(s) match ...". Nothing failed, nothing warned; the string simply stopped matching, and
a deliberately non-blocking security warning silently became `ABORT: structural preflight
failed`. It ran green on 2026-09-04 only because the change was committed after that
session had finished.

The lesson is not "use a better string". It is that a contract between two modules has to
be checked by something. These tests are that something.
"""
from __future__ import annotations

import sys

from .autostart import _all_symbols, _fatal_lines
from .preflight import EXPOSURE_TAG, FAIL, WARN, check_exposure

DATA_FAIL = f"{FAIL} QQQ underlying: quote is DELAYED by 900s"
EXPO_FAIL = f"{FAIL} {EXPOSURE_TAG} 2 inbound ALLOW rule(s) match the Theta Terminal"
EXPO_WARN = f"{WARN} {EXPOSURE_TAG} 2 inbound ALLOW rule(s) match the Theta Terminal"


class _Args:
    def __init__(self, symbols, share_symbols=None):
        self.symbols = symbols
        self.share_symbols = share_symbols


def main() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    print("=" * 78)
    print("PREFLIGHT <-> AUTOSTART GATE CONTRACT")
    print("=" * 78)

    # 1. the live exposure check must actually carry the token autostart looks for.
    #    This is the assertion that would have caught the 2026-09-04 regression.
    lines = check_exposure()
    verdicts = [l for l in lines if l.startswith(FAIL) or l.startswith(WARN)]
    tagged = [l for l in verdicts if EXPOSURE_TAG in l]
    check("live exposure verdicts carry EXPOSURE_TAG",
          bool(verdicts) and len(tagged) == len(verdicts),
          f"{len(tagged)}/{len(verdicts)} tagged")
    for l in verdicts:
        print(f"         {l.strip()[:88]}")

    # 2. classification
    check("a data FAIL aborts", _fatal_lines(DATA_FAIL) == [DATA_FAIL])
    check("an exposure FAIL does NOT abort", _fatal_lines(EXPO_FAIL) == [])
    check("an exposure WARN does NOT abort", _fatal_lines(EXPO_WARN) == [])
    mixed = "\n".join([EXPO_FAIL, DATA_FAIL, EXPO_WARN])
    check("mixed output aborts on the data FAIL only",
          _fatal_lines(mixed) == [DATA_FAIL])
    check("clean output aborts on nothing", _fatal_lines("  [OK]   all good") == [])

    # 3. preflight must be handed --ignore-exposure, so the gate is belt AND braces
    import inspect

    from . import autostart as A
    src = inspect.getsource(A._preflight)
    check("autostart passes --ignore-exposure to preflight",
          "--ignore-exposure" in src)

    # 4. both arms' universes reach preflight. Checking only the options universe would
    #    let the shares arm start on 13 names nothing had verified.
    a = _Args(["QQQ", "SPY"], ["SPY", "QQQ", "IWM", "NVDA"])
    got = _all_symbols(a)
    check("preflight sees the UNION of both universes",
          got == ["QQQ", "SPY", "IWM", "NVDA"], f"{got}")
    check("union is order-stable and de-duplicated",
          _all_symbols(_Args(["A", "B"], ["B", "A", "C"])) == ["A", "B", "C"])
    check("missing share_symbols degrades to the options universe",
          _all_symbols(_Args(["QQQ"], None)) == ["QQQ"])

    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
