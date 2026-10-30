"""specs/001: the snapshot writes once, never overwrites, never raises, never leaks the key."""
from __future__ import annotations

import datetime as dt
import json

from . import news_snapshot as N

NOW = dt.datetime(2026, 11, 3, 8, 31, 5)       # a Tuesday


def _ok(params):
    assert "api_key" in params and params["query"] == N.QUERY
    return [{"title": "Fed holds", "link": "https://x/y", "publishDate": "2026-11-02T18:00:00",
             "source": "www.reuters.com", "summary": "s", "sentiment": None, "confidence": None}]


def test_window_starts_at_the_previous_weekday_close():
    start, end = N.window(NOW)
    assert start == dt.datetime(2026, 11, 2, 16, 0) and end == NOW
    mon = dt.datetime(2026, 11, 2, 8, 30)
    assert N.window(mon)[0] == dt.datetime(2026, 10, 30, 16, 0)      # Friday, skipping the weekend


def test_writes_the_session_file_with_articles_and_metadata(tmp_path):
    p = N.snapshot(tmp_path, NOW, fetch=_ok, key="SECRET_KEY_ZQX")
    assert p.name == "2026-11-03.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["n"] == 1 and d["articles"][0]["title"] == "Fed holds"
    assert d["window_from"] == "2026-11-02T16:00:00" and d["query"] == N.QUERY
    assert "SECRET_KEY_ZQX" not in p.read_text(encoding="utf-8")        # key never written


def test_a_second_run_the_same_day_never_overwrites(tmp_path):
    p1 = N.snapshot(tmp_path, NOW, fetch=_ok, key="k")
    p2 = N.snapshot(tmp_path, NOW.replace(minute=45), fetch=_ok, key="k")
    assert p1 != p2 and p1.exists() and p2.exists()
    assert p2.name == "2026-11-03_084505.json"


def test_missing_key_writes_an_error_file_and_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.delenv(N.KEY_ENV, raising=False)
    p = N.snapshot(tmp_path, NOW, fetch=_ok)
    assert p.name == "2026-11-03.error.json"
    assert N.KEY_ENV in json.loads(p.read_text(encoding="utf-8"))["error"]


def test_a_failing_fetch_writes_an_error_file_with_the_key_redacted(tmp_path):
    def boom(params):
        raise RuntimeError("401 for key SECRET123")
    p = N.snapshot(tmp_path, NOW, fetch=boom, key="SECRET123")
    text = p.read_text(encoding="utf-8")
    assert p.name.endswith(".error.json") and "RuntimeError" in text
    assert "SECRET123" not in text and "<redacted>" in text


def test_missing_client_is_a_soft_failure(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def no_finlight(name, *a, **k):
        if name.startswith("finlight_client"):
            raise ModuleNotFoundError("No module named 'finlight_client'")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_finlight)
    p = N.snapshot(tmp_path, NOW, key="k")              # default_fetch -> import fails
    assert p.name.endswith(".error.json")
    assert "finlight_client" in json.loads(p.read_text(encoding="utf-8"))["error"]
