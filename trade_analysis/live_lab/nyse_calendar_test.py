"""The closure calendar against known dates; a wrong holiday turns a dead feed into a 'skip'."""
import datetime as dt

from . import nyse_calendar as C


def test_known_closures():
    assert C.easter(2024) == dt.date(2024, 3, 31) and C.easter(2019) == dt.date(2019, 4, 21)
    h = C.holidays(2022)
    assert {dt.date(2022, 1, 17), dt.date(2022, 2, 21), dt.date(2022, 4, 15), dt.date(2022, 5, 30),
            dt.date(2022, 6, 20), dt.date(2022, 7, 4), dt.date(2022, 9, 5), dt.date(2022, 11, 24),
            dt.date(2022, 12, 26)} <= h
    assert dt.date(2022, 1, 1) not in h and dt.date(2021, 12, 31) not in C.holidays(2021)   # Sat New Year: none
    assert dt.date(2023, 1, 2) in C.holidays(2023)                                       # Sun New Year -> Mon
    assert dt.date(2021, 7, 5) in C.holidays(2021) and dt.date(2020, 7, 3) in C.holidays(2020)
    assert dt.date(2018, 12, 5) in C.holidays(2018) and dt.date(2025, 1, 9) in C.holidays(2025)
    assert dt.date(2021, 6, 18) not in C.holidays(2021)                                   # Juneteenth from 2022


def test_the_23_lost_2022_sessions_are_regular_days():
    lost = ["2022-05-13", "2022-07-15", "2022-07-18", "2022-08-05", "2022-10-07", "2022-11-02"]
    assert not any(C.is_closed(dt.date.fromisoformat(d)) for d in lost)
    assert C.is_closed(dt.date(2022, 11, 24)) and C.is_closed(dt.date(2022, 11, 26))


def test_feature_calendar_agrees_with_the_rule_calendar():
    """Every date in research/news_impact/day_features_QQQ.csv is an open day by rule, and
    every rule-open weekday 2018-2025 is in the file (half days included)."""
    import csv
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "research" / "news_impact" / "day_features_QQQ.csv"
    feat = {dt.date.fromisoformat(r["date"]) for r in csv.DictReader(open(p, encoding="utf-8"))}
    assert not [d for d in feat if C.is_closed(d)]
    d = dt.date(2018, 1, 1)
    missing = []
    while d <= dt.date(2025, 12, 31):
        if not C.is_closed(d) and d not in feat:
            missing.append(d)
        d += dt.timedelta(days=1)
    assert missing == []


def test_early_closes():
    for d in ("2018-07-03", "2018-11-23", "2018-12-24", "2019-07-03", "2019-11-29", "2019-12-24",
              "2020-11-27", "2020-12-24", "2021-11-26", "2022-11-25", "2023-07-03", "2023-11-24",
              "2024-07-03", "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24"):
        assert C.early_close(dt.date.fromisoformat(d)), d
    for d in ("2020-07-03", "2021-12-24", "2022-07-01", "2022-11-02", "2021-07-02"):
        assert not C.early_close(dt.date.fromisoformat(d)), d
