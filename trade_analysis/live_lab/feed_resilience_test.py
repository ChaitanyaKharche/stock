"""Does a network switch still cost a session?

    python -m trade_analysis.live_lab.feed_resilience_test
    pytest trade_analysis/live_lab/feed_resilience_test.py

This exists because the lab machine travels in a car and moves between WiFi and a phone
hotspot mid-session, and the record says what that cost:

  * 282 of 284 feed outages in `live_lab_data/outages.jsonl` are one error --
    `HTTP 503: Unable to resolve host mdds-01.thetadata.us`
  * 202 of them on 2026-09-01 alone, 76 more on 2026-09-04
  * one whole session lost on 2026-08-28, `events.jsonl` seq 319,
    `{"kind":"aborted","reason":"warmup_feed_unreachable"}`

The shape of the bug was that all three of these were indistinguishable to the code:

    the terminal is not running          -> must relaunch
    the terminal is up, upstream is not  -> must WAIT; fixes itself in tens of seconds
    the data genuinely does not exist    -> neither

`/stock/snapshot/quote` is served by the terminal's own process and answers 200 right
through an MDDS outage, so `terminal_up()` could not see the middle case at all -- and
`_wait_for_history` only ran off the back of `start_terminal()`, which never fires when
the terminal is already up. Which is the normal case.

Nothing here touches a network. A fake transport plays back the exact recorded error
strings, so the classification is tested against what the feed actually returned rather
than against what I imagine it returns.
"""
from __future__ import annotations

import datetime as dt
import sys

import httpx

from . import feed as F
from .feed import (FeedOutage, TerminalUnreachable, ThetaLiveFeed,
                   UpstreamUnreachable, _classify)

# Verbatim from live_lab_data/outages.jsonl. Do not tidy these strings; they are data.
MDDS_503 = "Unable to resolve host mdds-01.thetadata.us"
REFUSED = ("ConnectError('[WinError 10061] No connection could be made because the "
           "target machine actively refused it')")
OSERR_22 = "OSError(22, 'Invalid argument')"


class _Transport:
    """Stands in for httpx.Client. `script` is consumed one entry per GET.

    An entry is either an int status code (with a body) or an exception to raise, so a
    test can spell out "503 five times, then 200" -- which is what a hotspot switch is.
    """

    def __init__(self, script):
        self.script = list(script)
        self.gets = 0

    def get(self, url, params=None):
        self.gets += 1
        item = self.script.pop(0) if self.script else ("ok", "")
        if isinstance(item, Exception):
            raise item
        kind, body = item
        if kind == "ok":
            return httpx.Response(200, text="ms_of_day,open,high,low,close,volume\n"
                                            "34200000,1,1,1,1,100\n")
        return httpx.Response(int(kind), text=body)

    def close(self):
        pass


def _feed(script, on_outage=None, max_retries=2):
    f = ThetaLiveFeed.__new__(ThetaLiveFeed)
    f.base_url, f.max_retries, f._on_outage = "http://x/v3", max_retries, on_outage
    f._client = _Transport(script)
    f.calls = f.retries = f.cache_hits = f.cache_misses = 0
    f._bar_cache = {}
    f._outage_since, f._outage_calls, f._outage_detail = None, 0, ""
    f.outage_episodes = 0
    return f


# ------------------------------------------------------------------- classification

def test_the_recorded_mdds_error_is_upstream_not_terminal():
    """The 282-occurrence error. Getting this one wrong is the entire bug."""
    assert _classify(503, MDDS_503) is UpstreamUnreachable


def test_a_refused_connection_is_the_terminal_not_the_upstream():
    """Opposite response: waiting never fixes this, so it must not be waited on."""
    assert _classify(None, REFUSED) is TerminalUnreachable


def test_a_socket_error_from_the_interface_change_is_upstream():
    """OSError(22), recorded once on 2026-09-02 -- a send on a route that just vanished."""
    assert _classify(None, OSERR_22) is UpstreamUnreachable


def test_a_bare_5xx_is_upstream_because_the_terminal_answered():
    for code in (502, 503, 504):
        assert _classify(code, "") is UpstreamUnreachable, code


def test_an_unrecognised_failure_stays_the_base_class():
    """Never guess upward. A 500 with no signature is not known to be recoverable, and
    calling it recoverable would make the lab wait 180s on a real fault."""
    assert _classify(500, "internal error") is FeedOutage
    assert _classify(None, "ValueError('nope')") is FeedOutage


def test_the_classified_error_is_what_get_csv_actually_raises():
    """Classification is worthless if the raise site drops it."""
    f = _feed([("503", MDDS_503), ("503", MDDS_503)])
    try:
        f._get_csv("/stock/history/ohlc", symbol="QQQ")
    except UpstreamUnreachable:
        pass
    else:
        raise AssertionError("did not raise UpstreamUnreachable")

    f = _feed([httpx.ConnectError(REFUSED), httpx.ConnectError(REFUSED)])
    try:
        f._get_csv("/stock/history/ohlc", symbol="QQQ")
    except TerminalUnreachable:
        pass
    else:
        raise AssertionError("did not raise TerminalUnreachable")


def test_a_472_is_still_no_data_and_not_an_outage():
    """The documented empty response. Misreading it as an outage would make every
    holiday and every pre-open probe look like a network fault."""
    f = _feed([("472", "")])
    assert f._get_csv("/stock/history/ohlc", symbol="QQQ") == []
    assert f._outage_since is None


# ------------------------------------------------------------------ the actual switch

def test_wait_for_upstream_returns_once_the_switch_completes():
    """The hotspot switch, end to end: down for a while, then back."""
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        if calls["n"] < 4:
            raise UpstreamUnreachable(MDDS_503)

    notes = []
    f = _feed([])
    assert f.wait_for_upstream(budget_s=30.0, probe_every=0.0, probe=probe,
                              on_wait=notes.append) is True
    assert calls["n"] == 4
    assert any("waiting up to" in n for n in notes), notes
    assert any("ready after 4 probe" in n for n in notes), notes


def test_wait_for_upstream_gives_up_on_a_genuinely_dead_upstream():
    """Patience must be bounded, or a dead upstream hangs the session instead of
    aborting it with a reason."""
    notes = []
    f = _feed([])
    assert f.wait_for_upstream(
        budget_s=0.05, probe_every=0.0,
        probe=lambda: (_ for _ in ()).throw(UpstreamUnreachable(MDDS_503)),
        on_wait=notes.append) is False
    assert any("still down after" in n for n in notes), notes


def test_the_probe_must_exercise_the_upstream_not_the_terminal():
    """`_default_probe` must ask for history, which goes through MDDS.

    Probing /stock/snapshot/quote would report recovery that has not happened, because
    the terminal serves that one from its own process. This is the same mistake
    `terminal_up()` makes, and the reason it could not see the failure.
    """
    f = _feed([])
    asked = []
    f._fetch_minute_bars = lambda sym, day: asked.append((sym, day))
    f._default_probe()
    assert len(asked) == 1, asked
    sym, day = asked[0]
    assert sym == "QQQ"
    assert F._is_session(day), f"probed {day}, which is not a trading day"
    assert day < now_date(), "probe must use a PRIOR session, not today"


def now_date() -> dt.date:
    return F.now_et().date()


# ------------------------------------------------------------- outage log collapsing

def test_one_episode_not_one_record_per_failed_call():
    """202 identical lines is the same fact 202 times, and it buried the one that
    differed. The counter still rises; the rows do not."""
    seen = []
    f = _feed([("503", MDDS_503)] * 6, on_outage=lambda p, d: seen.append((p, d)),
              max_retries=2)
    for _ in range(3):
        try:
            f._get_csv("/stock/history/ohlc", symbol="QQQ")
        except UpstreamUnreachable:
            pass
    assert len(seen) == 1, f"{len(seen)} records for one episode: {seen}"
    assert f._outage_calls == 3, f._outage_calls


def test_a_different_error_during_an_episode_is_still_reported():
    """Collapsing must not hide a change of cause. The OSError(22) on 2026-09-02 was
    the single most interesting line in the file and would have been swallowed."""
    seen = []
    f = _feed([("503", MDDS_503), ("503", MDDS_503),
               OSError(22, "Invalid argument"), OSError(22, "Invalid argument")],
              on_outage=lambda p, d: seen.append(d), max_retries=2)
    for _ in range(2):
        try:
            f._get_csv("/stock/history/ohlc", symbol="QQQ")
        except FeedOutage:
            pass
    assert len(seen) == 2, seen
    assert "Invalid argument" in seen[1], seen


def test_recovery_closes_the_episode_and_says_how_long():
    """A gap you can explain is data. An outage with no end is not explained."""
    seen = []
    f = _feed([("503", MDDS_503), ("503", MDDS_503), ("ok", "")],
              on_outage=lambda p, d: seen.append((p, d)), max_retries=2)
    try:
        f._get_csv("/stock/history/ohlc", symbol="QQQ")
    except UpstreamUnreachable:
        pass
    assert f._outage_since is not None
    f._get_csv("/stock/history/ohlc", symbol="QQQ")          # succeeds
    assert f._outage_since is None
    assert f.outage_episodes == 1
    assert seen[-1][0] == "__recovered__" and "recovered after" in seen[-1][1], seen


def test_stats_exposes_whether_the_upstream_is_down_right_now():
    """The daily summary needs to be able to say the session ran through an outage."""
    f = _feed([("503", MDDS_503), ("503", MDDS_503)], max_retries=2)
    assert f.stats()["upstream_down_now"] is False
    try:
        f._get_csv("/stock/history/ohlc", symbol="QQQ")
    except UpstreamUnreachable:
        pass
    assert f.stats()["upstream_down_now"] is True


# --------------------------------------------------------- the session-killing paths

def test_a_tick_still_fails_fast_and_does_not_block():
    """Patience belongs in warmup, never in a tick.

    A tick that blocked for 180s would stall all 15 symbols and turn one symbol's
    hiccup into a session-wide stall. The per-call budget must stay small.
    """
    import time as _t
    f = _feed([("503", MDDS_503)] * 4, max_retries=4)
    t0 = _t.perf_counter()
    try:
        f._get_csv("/stock/history/ohlc", symbol="QQQ")
    except UpstreamUnreachable:
        pass
    assert _t.perf_counter() - t0 < 30.0, "tick-path retry budget grew into a stall"


def test_autostart_checks_the_upstream_even_when_the_terminal_is_already_up():
    """The hole that cost 2026-08-28.

    `_wait_for_history` was only reachable through `start_terminal()`, so the branch
    where the terminal is ALREADY answering -- the normal case, and the one a network
    switch produces -- never checked the upstream at all. Asserted on the source,
    because the alternative is a live terminal.
    """
    import inspect

    from . import autostart as A
    src = inspect.getsource(A.main)
    assert "_wait_for_history" in src, \
        "autostart.main never confirms the historical upstream"
    i, j = src.index("if not terminal_up()"), src.index("_wait_for_history")
    assert i < j, "the upstream check is not on the already-up branch"
    assert "UPSTREAM_WAIT_SEC" in src, "the wait budget is not the shared constant"


def test_both_arms_wait_on_the_same_budget():
    """A resilience rule that holds on one arm and not the other makes the two records
    differ for reasons unrelated to the strategies under test."""
    import inspect

    from . import runner, shares_runner
    for mod in (runner, shares_runner):
        src = inspect.getsource(mod.warmup if hasattr(mod, "warmup")
                                else mod.__dict__[
                                    "LiveLab" if mod is runner else "SharesLab"].warmup)
        assert "UpstreamUnreachable" in src, f"{mod.__name__} warmup does not wait"
        assert "wait_for_upstream" in src, f"{mod.__name__} warmup does not wait"
    assert runner.UPSTREAM_WAIT_SEC == shares_runner.UPSTREAM_WAIT_SEC == F.UPSTREAM_WAIT_SEC


# ------------------------------------------- the two guards have to compose, not just coexist

def test_the_upstream_wait_is_suspend_aware():
    """The hole that only exists once BOTH branches land.

    `_wait_for_history` used bare `time.sleep(3)`. That was fine at 45s reachable only
    from `start_terminal()`. It is not fine at UPSTREAM_WAIT_SEC on every already-up
    path: the host can suspend inside that window, and the suspend guard would never
    see it -- so the record would show a clean wait where the machine was asleep.

    Neither branch was wrong alone. Merging them created the gap, which is exactly the
    kind of thing a clean textual auto-merge will not tell you about.
    """
    import inspect

    from . import autostart as A
    src = inspect.getsource(A._wait_for_history)
    assert "_sleep_watched" in src, \
        "_wait_for_history sleeps without the suspend guard"
    assert "lab_dir" in inspect.signature(A._wait_for_history).parameters, \
        "_wait_for_history cannot reach a store to record a suspend"
    # And main must actually pass one, or the parameter is decoration.
    main_src = inspect.getsource(A.main)
    assert "lab_dir=args.lab_dir" in main_src, \
        "main calls _wait_for_history without a lab_dir"


def test_the_wake_lock_is_released_after_the_archive_not_before():
    """Order matters: the push can take tens of seconds.

    Releasing the lock earlier would let the host sleep mid-push, leaving the day's
    record on one disk -- the precise failure the archive exists to prevent.
    """
    import inspect

    from . import autostart as A
    src = inspect.getsource(A.main)
    assert "release_system_awake()" in src, "the wake lock is never released"
    assert src.index("archive_session") < src.index("release_system_awake()"), \
        "the wake lock is released before the archive push"


CHECKS = [(n, f) for n, f in sorted(globals().items())
          if n.startswith("test_") and callable(f)]


def main() -> int:
    ok = True
    print("=" * 78)
    print("FEED RESILIENCE -- WIFI <-> HOTSPOT TRANSITIONS")
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
    print(f"\n  {len(CHECKS)} checks, no network touched")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print("\n  An open socket is not a working service.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
