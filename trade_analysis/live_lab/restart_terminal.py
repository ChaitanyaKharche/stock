"""Nudge Theta Terminal back up WITHOUT touching the running arms.

    python -m trade_analysis.live_lab.restart_terminal            # report only
    python -m trade_analysis.live_lab.restart_terminal --fix      # restart if it is down

THE GAP THIS FILLS
------------------
`autostart.supervise()` already restarts the terminal -- but only on the path where an ARM
has died and is being restarted:

    elif time.monotonic() >= pending[ch.name]:
        if not terminal_up():
            start_terminal()

An arm that survives does not take that path. And the arms DO survive a dead feed on
purpose: `runner` catches `FeedOutage` per tick, records it, and keeps looping. So the
failure mode is a terminal that wedges while both arms stay healthy -- every tick throws,
every throw is recorded, entries are suppressed, and nothing ever restarts the terminal.
That can quietly cost the rest of the session's collection while the log looks busy.

A network change is the most likely cause: the terminal keeps its HTTP listener bound on
127.0.0.1 (so `terminal_up()`-by-port would still pass) while its upstream connection is
gone. That is precisely the "listening but not answering" state `_evict_stale_terminal()`
was written for on 2026-09-08.

WHY THIS IS SAFE TO RUN MID-SESSION
-----------------------------------
It touches the terminal only. It does not signal, restart, or even look at the arms, and it
takes no lock they hold. The arms poll 127.0.0.1:25503 on every tick with retries, so once
the terminal answers again they reconnect on their next tick with no restart and no
recovery step -- open positions are never reloaded and so can never be double-counted.

It also cannot fork the frozen config hash: that hash covers symbols, fees, arms, family
size and the setup definitions (`runner.build_config`), none of which live here.

It reuses `autostart`'s own `terminal_up` / `start_terminal` / `_evict_stale_terminal`
rather than reimplementing them, so there is one definition of "is the terminal healthy"
and this file cannot drift from the supervisor's.
"""
from __future__ import annotations

import argparse
import sys

from .autostart import start_terminal, terminal_up


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fix", action="store_true",
                    help="restart the terminal if it is not answering. Without this the "
                         "command only reports, so it is safe to run blind.")
    a = ap.parse_args(argv)

    up = terminal_up()
    print(f"  Theta Terminal on 127.0.0.1:25503 : {'ANSWERING' if up else 'NOT ANSWERING'}")
    if up:
        print("  nothing to do. The arms poll localhost every tick, so if they are logging")
        print("  outages while this says ANSWERING, the problem is UPSTREAM of the")
        print("  terminal (blocked egress, captive portal, proxy) and restarting it will")
        print("  not help -- check whether the network allows the vendor's endpoints.")
        return 0

    if not a.fix:
        print("\n  Terminal is down and --fix was not given, so nothing was changed.")
        print("  Re-run with --fix to evict the wedged process and relaunch:")
        print("    python -m trade_analysis.live_lab.restart_terminal --fix")
        return 1

    print("\n  restarting (evicts a listening-but-unresponsive terminal first) ...")
    ok = start_terminal()
    print(f"  result: {'UP' if ok else 'STILL DOWN'}")
    if ok:
        print("  The arms will reconnect on their next tick. No arm restart is needed and")
        print("  no position recovery happens, so nothing can be double-counted.")
    else:
        print("  Launch failed or timed out. Check that java is on PATH and that")
        print("  ThetaTerminalv3.jar is where autostart.TERMINAL_DIR points.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
