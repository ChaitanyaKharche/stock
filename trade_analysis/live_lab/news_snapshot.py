"""Daily Finlight news snapshot for the lab. Collection only. specs/001-finlight-daily-snapshot.md

    python -m trade_analysis.live_lab.news_snapshot [--lab-dir live_lab_data]

Writes live_lab_data/news/<session>.json once per session, never overwrites, never raises,
always exits 0. The API key comes from the FINLIGHT_API_KEY environment variable and is
never written, logged or passed as an argument. Uses the vendor's own client
(`pip install finlight-client`) so no endpoint is guessed; if it is not installed, or the
key is missing, or the call fails, a <session>.error.json says why and the session goes on.

Frozen constants (R5): change only with a dated note in the spec.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from .clock import now_et

QUERY = ("(Fed OR \"Federal Reserve\" OR inflation OR CPI OR jobs OR tariff OR tariffs OR war "
         "OR Treasury OR shutdown) AND (stocks OR markets OR economy)")
PAGE_SIZE = 100
LANGUAGE = "en"
PREV_CLOSE = dt.time(16, 0)
KEY_ENV = "FINLIGHT_API_KEY"
FIELDS = ("title", "link", "publishDate", "source", "summary", "sentiment", "confidence")


def window(now: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
    """[previous weekday 16:00 ET, now]. Holidays are not special-cased: a Tuesday after a
    Monday holiday simply spans two days, which is what the trader would read anyway."""
    d = now.date() - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return dt.datetime.combine(d, PREV_CLOSE), now


def out_path(news_dir: Path, now: dt.datetime, error: bool = False) -> Path:
    base = now.date().isoformat()
    suffix = ".error.json" if error else ".json"
    p = news_dir / f"{base}{suffix}"
    if p.exists():
        p = news_dir / f"{base}_{now:%H%M%S}{suffix}"
    return p


def default_fetch(params: dict) -> list[dict]:
    """The vendor client, imported lazily so a missing install is a soft failure (R3)."""
    from finlight_client import ApiConfig, FinlightApi                 # type: ignore
    from finlight_client.models import GetArticlesParams                # type: ignore
    client = FinlightApi(ApiConfig(api_key=params.pop("api_key")))
    resp = client.articles.fetch_articles(GetArticlesParams(**params))
    arts = getattr(resp, "articles", None) or (resp.get("articles") if isinstance(resp, dict) else [])
    out = []
    for a in arts:
        row = {}
        for f in FIELDS:
            v = getattr(a, f, None) if not isinstance(a, dict) else a.get(f)
            if hasattr(v, "isoformat"):
                v = v.isoformat()
            row[f] = v
        out.append(row)
    return out


def snapshot(lab_dir, now: dt.datetime | None = None, fetch=default_fetch,
             key: str | None = None) -> Path:
    """Take one snapshot. Returns the path written (data or error). Never raises."""
    now = now or now_et()
    news_dir = Path(lab_dir) / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    start, end = window(now)
    meta = {"session": now.date().isoformat(), "fetched_at_et": now.isoformat(timespec="seconds"),
            "query": QUERY, "window_from": start.isoformat(), "window_to": end.isoformat(),
            "page_size": PAGE_SIZE}
    key = key if key is not None else os.environ.get(KEY_ENV)
    if not key:
        p = out_path(news_dir, now, error=True)
        p.write_text(json.dumps({**meta, "error": f"{KEY_ENV} not set"}, indent=1), encoding="utf-8")
        return p
    try:
        arts = fetch({"api_key": key, "query": QUERY, "from_": start.isoformat(),
                      "to": end.isoformat(), "pageSize": PAGE_SIZE, "language": LANGUAGE})
    except Exception as exc:                                             # noqa: BLE001
        p = out_path(news_dir, now, error=True)
        msg = f"{type(exc).__name__}: {exc}"
        if key in msg:
            msg = msg.replace(key, "<redacted>")
        p.write_text(json.dumps({**meta, "error": msg}, indent=1), encoding="utf-8")
        return p
    p = out_path(news_dir, now)
    p.write_text(json.dumps({**meta, "n": len(arts), "articles": arts}, indent=1,
                            ensure_ascii=False), encoding="utf-8")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lab-dir", default="live_lab_data")
    a = ap.parse_args(argv)
    p = snapshot(a.lab_dir)
    print(f"[news] wrote {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
