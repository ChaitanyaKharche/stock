"""Verification for the 2026-09-10 host-suspend guards. Run directly:

    python -m trade_analysis.live_lab.autostart_guard_test

What happened, and what each check defends:

On 2026-09-10 the host idle-slept at 15:41:57 ET and woke at 17:35:47. The shares runner
was mid-session with 32 positions open. On wake it would have recovered by itself -- its
loop breaks once `now >= RTH_CLOSE`, then it flattens and calls `write_daily` -- but the
supervisor's 16:10 backstop fired ONE SECOND after the wake and terminated it first. The
cost was the EOD flatten, 32 positions, the daily summary, and the session-stop event,
all while the log read `rc=0`.

So there are two distinct failures and three guards:

  1. the host slept at all            -> `hold_system_awake`   (check 4)
  2. nothing in the lab could see it  -> `_sleep_watched`      (checks 3a, 3b)
  3. the backstop killed the recovery -> grace window          (checks 1, 2)

These use the `child_cls` injection hook that `supervise()` already exposes, so no real
process, feed or lab directory is touched.
"""
from __future__ import annotations

import datetime as dt
import sys
import tempfile
import time
import types

from . import autostart as A

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def _args(lab_dir: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(lab_dir=lab_dir, start_terminal=False)


def _fake_child_cls(exits_after):
    """A child that exits `exits_after` seconds after start, or never if None."""

    class FakeChild:
        made: list["FakeChild"] = []

        def __init__(self, name, cmd, root):
            self.name, self.cmd, self.root = name, cmd, root
            self.done = False
            self.rc = None
            self.restarts = 0
            self.terminated = False
            self._t0 = None
            FakeChild.made.append(self)

        def start(self):
            self._t0 = time.monotonic()

        def poll(self):
            if self._t0 is None:
                return None
            if exits_after is None:
                return None
            return 0 if (time.monotonic() - self._t0) >= exits_after else None

        def terminate(self):
            # Mirrors the real _Child: terminating an already-exited process is a no-op.
            if self.poll() is None:
                self.terminated = True

    FakeChild.made = []
    return FakeChild


def _run_supervise(exits_after, grace, now_time):
    """Drive supervise() with the clock pinned past HARD_STOP."""
    day = dt.date(2026, 9, 10)
    frozen = dt.datetime.combine(day, now_time)
    old_now, old_grace = A.now_et, A.HARD_STOP_GRACE_SEC
    A.now_et = lambda: frozen
    A.HARD_STOP_GRACE_SEC = grace
    cls = _fake_child_cls(exits_after)
    try:
        with tempfile.TemporaryDirectory() as td:
            t0 = time.monotonic()
            A.supervise(_args(td), day, specs=[("shares", ["x"])], child_cls=cls)
            return cls.made[0], time.monotonic() - t0
    finally:
        A.now_et, A.HARD_STOP_GRACE_SEC = old_now, old_grace


def main() -> int:
    print("autostart host-suspend guards\n")

    # -- 1. the regression itself -------------------------------------------------------
    # Past the backstop, an arm that is mid-recovery must be WAITED for, not killed.
    print("1. a waking runner is given time to flatten and write")
    child, elapsed = _run_supervise(exits_after=3, grace=30, now_time=dt.time(17, 35, 46))
    check("not terminated", not child.terminated,
          "this is the 2026-09-10 bug: it was killed 1s after wake")
    check("actually waited for it", elapsed >= 2.5, f"waited {elapsed:.1f}s")
    check("did not burn the whole grace window", elapsed < 25, f"{elapsed:.1f}s < 30s")

    # -- 2. the grace window is bounded -------------------------------------------------
    # A genuinely wedged arm must still die, or the supervisor never returns.
    print("\n2. a wedged runner is still terminated once grace expires")
    child, elapsed = _run_supervise(exits_after=None, grace=4, now_time=dt.time(16, 10, 1))
    check("terminated", child.terminated)
    check("only after the grace window", elapsed >= 3.5, f"{elapsed:.1f}s")

    # -- 3. a suspend becomes visible in the record -------------------------------------
    print("\n3. a wall-clock jump is detected and recorded")
    seen: list[tuple] = []
    old_rec, old_now = A._record_suspend, A.now_et
    try:
        A._record_suspend = lambda lab, gap, b, a: seen.append((lab, gap, b, a))

        def scripted(values):
            """Return each value in turn, then hold the last -- `log` reads the clock too."""
            box = {"i": 0}

            def _now():
                v = values[min(box["i"], len(values) - 1)]
                box["i"] += 1
                return v
            return _now

        base = dt.datetime(2026, 9, 10, 15, 41, 57)
        # 1h53m50s vanishes across a nominal 5s sleep -- the real 2026-09-10 gap.
        A.now_et = scripted([base, base + dt.timedelta(seconds=6835)])
        A._sleep_watched("unused", 0.01)
        check("suspend detected", len(seen) == 1)
        check("gap measured correctly", seen and abs(seen[0][1] - 6834.99) < 1.0,
              f"{seen[0][1]:.0f}s" if seen else "not called")

        # ...and an ordinary sleep must stay silent, or the record fills with noise.
        seen.clear()
        t = dt.datetime(2026, 9, 10, 12, 0, 0)
        A.now_et = scripted([t, t + dt.timedelta(seconds=5)])
        A._sleep_watched("unused", 5)
        check("a normal sleep is silent", not seen)
    finally:
        A._record_suspend, A.now_et = old_rec, old_now

    # -- 4. the guard installs on this host ---------------------------------------------
    print("\n4. the OS actually grants the wake lock on this machine")
    state = A.hold_system_awake()
    A.release_system_awake()
    check("guard held", state.startswith("HELD"), state)

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
