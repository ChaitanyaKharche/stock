"""Event-file rules fixed by research/news_arms/preregistration.md (15.4, 14.4).

Overnight marker: a geopolitical row whose headline carries a parenthetical naming a
closed-market day (Saturday, Sunday, weekend, "holiday", "early close", or a named US market
holiday) counts as overnight: every trade that session is post. "(half-day session)" is not
a marker. Applied to the frozen file this marks exactly 14 rows (test).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

HOLIDAY_WORDS = (r"Saturday|Sunday|weekend|holiday|early close|Presidents|MLK|"
                 r"Martin Luther King|Labor Day|Memorial Day|Good Friday|Thanksgiving|"
                 r"Christmas|New Year|Juneteenth|Independence Day|July 4")
OVERNIGHT = re.compile(r"\(([^)]*\b(?:%s)\b[^)]*)\)" % HOLIDAY_WORDS, re.I)
RTH = ("09:30", "16:00")

GEO = Path(__file__).resolve().parents[2] / "research" / "news_impact" / "events_geopolitical.csv"


def is_overnight(row: dict) -> bool:
    return bool(OVERNIGHT.search(row.get("headline", "")))


def in_session_break(row: dict) -> bool:
    t = row.get("break_time_et", "unknown")
    return t != "unknown" and RTH[0] <= t < RTH[1]


def event_minute(row: dict):
    """The minute from which a trade counts as post (14.4 / 15.4): None means every trade
    of the session is post (pre-open release, overnight or weekend shock, or unknown time
    which Q5 excludes anyway)."""
    if not in_session_break(row) or is_overnight(row):
        return None
    return row["break_time_et"]


def load_geopolitical(path: Path = GEO) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
