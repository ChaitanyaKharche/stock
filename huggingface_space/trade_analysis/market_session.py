"""Is the US equity market open right now, and which session is this signal from?

Reported 2026-09-07 at 22:22 ET: the app served Friday's tape with no indication it was
Friday's. That date was **Labor Day** -- a full closure -- so a visitor got a signal built
from bars two calendar days old, labelled as if it were live. Both halves of that are
wrong, and only one of them is about the clock: a weekday check alone would still have
called Labor Day a session.

The holiday table is the repo's maintained one (`trading_days.py`, back to 2004, including
the irregular closures), copied in because the Space must build without the parent repo and
kept honest by sync_shared.py.

`market_status()` returns a dict, never raises, and degrades to "unknown" rather than
guessing, because an unavailable timezone database must not be reported as "closed".
"""
from __future__ import annotations

import datetime as dt

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:                                   # pragma: no cover - tzdata missing
    _ET = None

try:
    from . import trading_days as _cal
except Exception:                                   # pragma: no cover
    _cal = None

RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)
EARLY_CLOSE = dt.time(13, 0)


def _is_session(d: dt.date) -> bool:
    if _cal is not None:
        try:
            return bool(_cal.is_trading_day(d))
        except Exception:
            pass
    return d.weekday() < 5          # last resort; will be wrong on holidays


def _early(d: dt.date) -> bool:
    if _cal is not None:
        try:
            return bool(_cal.is_early_close(d))
        except Exception:
            pass
    return False


def previous_session(d: dt.date) -> dt.date:
    """The most recent trading day strictly before `d`."""
    probe = d - dt.timedelta(days=1)
    for _ in range(15):             # covers the longest run of closures
        if _is_session(probe):
            return probe
        probe -= dt.timedelta(days=1)
    return probe


def market_status(now: dt.datetime | None = None) -> dict:
    """Open/closed, why, and which session the data therefore belongs to.

    `reference_session` is the day whose bars a signal computed right now actually
    reflects -- today once the opening bell has gone, otherwise the previous session. It
    is what the UI should date the result with, so a stale-looking number is explained
    rather than merely stale.
    """
    if _ET is None:
        return {"state": "unknown", "is_open": False,
                "reason": "No timezone database available; cannot determine ET.",
                "reference_session": None, "as_of_et": None, "is_stale": None}

    now_et = now.astimezone(_ET) if now is not None else dt.datetime.now(_ET)
    today = now_et.date()
    close_t = EARLY_CLOSE if _early(today) else RTH_CLOSE
    session_today = _is_session(today)

    if not session_today:
        if today.weekday() >= 5:
            state, reason = "weekend", f"Weekend ({today:%A})."
        else:
            state, reason = "holiday", f"Market holiday ({today:%A %d %B %Y})."
        ref = previous_session(today)
    elif now_et.time() < RTH_OPEN:
        state = "premarket"
        reason = f"Pre-market. Opens 09:30 ET ({_fmt_delta(now_et, RTH_OPEN)})."
        ref = previous_session(today)          # today has not printed a bar yet
    elif now_et.time() >= close_t:
        state = "closed"
        reason = ("Closed for the day"
                  + (" (early close 13:00 ET)." if close_t == EARLY_CLOSE
                     else " at 16:00 ET."))
        ref = today
    else:
        state, reason, ref = "open", "Open.", today

    nxt = today if (session_today and now_et.time() < RTH_OPEN) else None
    if nxt is None and state != "open":
        probe = today + dt.timedelta(days=1)
        for _ in range(15):
            if _is_session(probe):
                nxt = probe
                break
            probe += dt.timedelta(days=1)

    return {
        "state": state,                                  # open|closed|premarket|weekend|holiday
        "is_open": state == "open",
        "reason": reason,
        "reference_session": ref.isoformat() if ref else None,
        "as_of_et": now_et.strftime("%Y-%m-%d %H:%M ET"),
        "next_open": nxt.isoformat() if nxt else None,
        # True when the newest bar cannot be from today -- the case that prompted this.
        "is_stale": state != "open" and ref != today,
    }


def _fmt_delta(now_et: dt.datetime, target: dt.time) -> str:
    delta = dt.datetime.combine(now_et.date(), target, tzinfo=_ET) - now_et
    mins = max(0, int(delta.total_seconds() // 60))
    return f"in {mins // 60}h {mins % 60}m" if mins >= 60 else f"in {mins}m"


def banner(status: dict) -> str:
    """One line for the UI. Says what the data IS, not merely that the market is shut."""
    if status.get("state") == "unknown":
        return "Market status unavailable."
    ref = status.get("reference_session")
    if status.get("is_open"):
        return f"Market open — live bars for {ref}."
    tail = f" Showing the last completed session: {ref}." if ref else ""
    nxt = status.get("next_open")
    return status["reason"] + tail + (f" Next open {nxt} 09:30 ET." if nxt else "")
