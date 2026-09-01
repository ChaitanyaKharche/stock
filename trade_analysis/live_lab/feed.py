"""ThetaData access for the live lab.

Deliberately standalone -- imports nothing from trade_analysis.data_sources, so a change
to the historical research client cannot silently alter live behaviour.

Endpoints verified against a running Theta Terminal on 2026-08-27:
    /v3/stock/snapshot/quote     -> live NBBO, sub-second stamps
    /v3/stock/snapshot/ohlc      -> live session aggregate
    /v3/stock/history/ohlc       -> 1m bars, RTH only, 09:30..16:00 (16:00 is a null stub)
    /v3/option/snapshot/quote    -> WHOLE CHAIN for an expiration, ~333 rows in ~310 ms
    /v3/option/list/expirations  -> all listed expirations
    /v3/option/snapshot/open_interest
Greeks/IV endpoints do NOT exist on this subscription tier (all 404). IV and delta are
computed in options.py and always labelled *_derived.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import time
from typing import Any

import httpx

BASE_URL = "http://127.0.0.1:25503/v3"
SESSION_OPEN = dt.time(9, 30)
SESSION_LAST_BAR = dt.time(15, 59)


class FeedOutage(Exception):
    """Raised when the terminal cannot be reached after the configured retries."""


class ThetaLiveFeed:
    """Thin, resilient reader. Never fabricates a value; raises or returns None."""

    def __init__(self, base_url: str = BASE_URL, timeout: float = 15.0,
                 max_retries: int = 4, on_outage=None):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)
        self._on_outage = on_outage
        self.calls = 0
        self.retries = 0

    # ------------------------------------------------------------------ transport

    def _get_csv(self, path: str, **params) -> list[dict[str, Any]]:
        url = f"{self.base_url}{path}"
        delay = 1.0
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self.calls += 1
                r = self._client.get(url, params=params)
                if r.status_code == 472:          # documented "no data" response
                    return []
                if r.status_code != 200:
                    raise httpx.HTTPError(f"HTTP {r.status_code}: {r.text[:120]}")
                text = r.text
                if not text.strip():
                    return []
                return list(csv.DictReader(io.StringIO(text)))
            except Exception as exc:              # noqa: BLE001 - transport is broad by nature
                last = exc
                self.retries += 1
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
        if self._on_outage:
            self._on_outage(path, repr(last))
        raise FeedOutage(f"{path} failed after {self.max_retries} attempts: {last!r}")

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ underlying

    def stock_quote(self, symbol: str) -> dict | None:
        """Live NBBO. Returns None rather than a guess if the row is unusable."""
        rows = self._get_csv("/stock/snapshot/quote", symbol=symbol)
        if not rows:
            return None
        r = rows[-1]
        try:
            bid, ask = float(r["bid"]), float(r["ask"])
        except (KeyError, ValueError):
            return None
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        return {
            "ts": _parse_ts(r["timestamp"]),
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2.0,
            "bid_size": _f(r.get("bid_size")),
            "ask_size": _f(r.get("ask_size")),
        }

    def minute_bars(self, symbol: str, day: dt.date) -> list[dict]:
        """RTH 1-minute bars for `day`, 09:30..15:59.

        Drops the 16:00 settlement stub (all-zero OHLCV) and any zero-volume null row.
        Bars are returned in ascending time order with naive-ET timestamps.
        """
        iso = day.isoformat()
        rows = self._get_csv("/stock/history/ohlc", symbol=symbol,
                             start_date=iso, end_date=iso, interval="1m")
        out: list[dict] = []
        for r in rows:
            try:
                ts = _parse_ts(r["timestamp"])
                o, h, l, c = (float(r["open"]), float(r["high"]),
                              float(r["low"]), float(r["close"]))
                v = float(r["volume"])
            except (KeyError, ValueError):
                continue
            if ts.time() > SESSION_LAST_BAR or ts.time() < SESSION_OPEN:
                continue
            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                continue                       # null stub
            out.append({"ts": ts, "open": o, "high": h, "low": l,
                        "close": c, "volume": v})
        out.sort(key=lambda b: b["ts"])
        return out

    # ------------------------------------------------------------------ options

    def expirations(self, symbol: str) -> list[dt.date]:
        rows = self._get_csv("/option/list/expirations", symbol=symbol)
        out = []
        for r in rows:
            try:
                out.append(dt.date.fromisoformat(r["expiration"].strip('"')))
            except (KeyError, ValueError):
                continue
        return sorted(out)

    def zero_dte(self, symbol: str, day: dt.date) -> dt.date | None:
        """The same-day expiration, or None if the symbol has no 0DTE listed for `day`."""
        return day if day in set(self.expirations(symbol)) else None

    def chain_quotes(self, symbol: str, expiration: dt.date) -> list[dict]:
        """Whole-chain NBBO snapshot. One call, ~333 rows, ~310 ms.

        Rows with a non-positive or crossed market are dropped, not repaired.
        """
        rows = self._get_csv("/option/snapshot/quote", symbol=symbol,
                             expiration=expiration.isoformat())
        out: list[dict] = []
        for r in rows:
            try:
                bid, ask = float(r["bid"]), float(r["ask"])
                strike = float(r["strike"])
                right = r["right"].strip('"').upper()
            except (KeyError, ValueError):
                continue
            if ask <= 0 or ask < bid or bid < 0:
                continue
            out.append({
                "ts": _parse_ts(r["timestamp"]),
                "strike": strike,
                "right": "call" if right.startswith("C") else "put",
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2.0,
                "spread": ask - bid,
                "bid_size": _f(r.get("bid_size")),
                "ask_size": _f(r.get("ask_size")),
            })
        return out

    def open_interest(self, symbol: str, expiration: dt.date) -> dict[tuple[float, str], float]:
        rows = self._get_csv("/option/snapshot/open_interest", symbol=symbol,
                             expiration=expiration.isoformat())
        out: dict[tuple[float, str], float] = {}
        for r in rows:
            try:
                k = float(r["strike"])
                right = "call" if r["right"].strip('"').upper().startswith("C") else "put"
                out[(k, right)] = float(r["open_interest"])
            except (KeyError, ValueError):
                continue
        return out

    def stats(self) -> dict:
        return {"calls": self.calls, "retries": self.retries}


# --------------------------------------------------------------------------- helpers


def _parse_ts(raw: str) -> dt.datetime:
    """ThetaData stamps are ET wall clock. Stored naive-ET throughout the lab."""
    s = raw.strip().strip('"')
    if s.endswith("Z"):
        s = s[:-1]
    return dt.datetime.fromisoformat(s).replace(tzinfo=None)


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
