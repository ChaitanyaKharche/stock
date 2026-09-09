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

import concurrent.futures as cf
import csv
import datetime as dt
import io
import time
from typing import Any

import httpx

BASE_URL = "http://127.0.0.1:25503/v3"
SESSION_OPEN = dt.time(9, 30)
SESSION_LAST_BAR = dt.time(15, 59)
BAR_SECONDS = 60          # must match session.BAR_SECONDS; see _serve_from_cache


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
        self._bar_cache: dict[tuple[str, dt.date], list[dict]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

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

    def minute_bars(self, symbol: str, day: dt.date,
                    now: dt.datetime | None = None) -> list[dict]:
        """RTH 1-minute bars for `day`, 09:30..15:59.

        Drops the 16:00 settlement stub (all-zero OHLCV) and any zero-volume null row.
        Bars are returned in ascending time order with naive-ET timestamps.

        `now` is OPT-IN caching. Omit it and every call goes to the wire, which is what
        replay, walkforward and preflight want. Pass the live loop's clock and the result
        is served from memory whenever the cache provably already holds every bar the
        session could admit -- see `_serve_from_cache`. A live loop polls every 5s but a
        1-minute bar changes once a minute, so this removes ~12x of pure waste and is the
        prerequisite for running more than a couple of symbols inside one poll interval.
        """
        if now is not None:
            hit = self._serve_from_cache(symbol, day, now)
            if hit is not None:
                self.cache_hits += 1
                return hit
            self.cache_misses += 1
        out = self._fetch_minute_bars(symbol, day)
        if now is not None:
            self._bar_cache[(symbol, day)] = out
        return out

    def _serve_from_cache(self, symbol: str, day: dt.date,
                          now: dt.datetime) -> list[dict] | None:
        """The cached bars, but ONLY when serving them is exactly equivalent to refetching.

        A bar stamped T covers [T, T+60s) and `SessionState.bar_is_complete` refuses to
        admit it before T + 60s + settle. So at wall clock `now` the newest bar any session
        could possibly admit is stamped `floor_minute(now - 60s)`. If the cache already
        reaches that stamp then it contains every admissible bar -- bars are contiguous and
        ascending -- and `accept_bars` sees an identical set. That is an equality, not an
        approximation, and it is why this cache cannot change a single fill.

        Deliberately ignoring `settle` (rather than adding it) keeps the guarantee one-sided:
        the cache demands the bar up to 1.5s EARLIER than the session needs it, so any error
        is in the direction of refetching too often. It also means the guarantee holds for
        any settle_ms >= 0, so tuning settle can never silently break this.

        Two behaviours fall out for free and are the reason it is written as an invariant
        rather than a timer:
          * a bar the vendor publishes LATE is polled for every tick until it arrives,
            instead of being missed until the next minute;
          * during a network outage the cache stops answering within 60s, so it can never
            hide a dead link from the caller for longer than one bar.
        """
        bars = self._bar_cache.get((symbol, day))
        if bars is None:
            return None                      # never fetched -- must go to the wire
        if day < now.date():
            return bars                      # a closed session cannot gain bars
        newest = (now - dt.timedelta(seconds=BAR_SECONDS)).replace(second=0, microsecond=0)
        last_rth = dt.datetime.combine(day, SESSION_LAST_BAR)
        if newest > last_rth:
            newest = last_rth                # nothing is published after 15:59
        if newest < dt.datetime.combine(day, SESSION_OPEN):
            return bars                      # pre-open: no bar is admissible yet
        if bars and bars[-1]["ts"] >= newest:
            return bars
        return None                          # a newer admissible bar may exist

    def _fetch_minute_bars(self, symbol: str, day: dt.date) -> list[dict]:
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

    # ------------------------------------- concurrent fan-out (many symbols, one tick)

    def _fan_out(self, symbols, fn, workers: int = 8):
        """Run `fn(symbol)` across symbols concurrently, isolating per-symbol failures.

        Returns (results, errors). A symbol that raises lands in `errors` and is simply
        absent from `results` -- one unreachable symbol must not blank out the other
        fourteen, which is what a bare loop would do by propagating the first exception.

        Why this exists: the bar cache fixes SUSTAINED feed load but not the PEAK. Every
        symbol waits on the same minute boundary, so on the one tick per minute where the
        new bar lands, all of them refetch at once. Sequentially that tick costs
        N x 477 ms -- at 15 symbols the last name in the list fills ~14 s after its bar
        closed. These are momentum entries, so that delay is systematically adverse rather
        than zero-mean, and 14 s of QQQ drift is a meaningful slice of a 3.34 bp edge.
        Fanning out collapses the boundary tick to roughly one call's latency.

        httpx.Client is thread-safe for concurrent requests; the counters are plain ints
        and may under-count under contention, which is acceptable for a statistic and is
        never used for control flow beyond "did anything reach the wire".

        workers=8 is MEASURED, not guessed. Against the local gateway, 15 symbols
        (bars + quotes, best of 3, warm):

            workers   1      2      4      6      8      12     15
            seconds   6.76   3.38   2.40   2.09   1.59   1.93   2.55

        It gets WORSE past 8 -- the Theta Terminal is one local process and starts
        contending with itself, so raising this to match the universe size would slow
        the tick down rather than speed it up.
        """
        symbols = list(symbols)
        results: dict[str, Any] = {}
        errors: dict[str, Exception] = {}
        if not symbols:
            return results, errors
        if len(symbols) == 1:
            try:
                results[symbols[0]] = fn(symbols[0])
            except Exception as exc:              # noqa: BLE001
                errors[symbols[0]] = exc
            return results, errors
        with cf.ThreadPoolExecutor(max_workers=min(workers, len(symbols))) as pool:
            futs = {pool.submit(fn, s): s for s in symbols}
            for fut in cf.as_completed(futs):
                s = futs[fut]
                try:
                    results[s] = fut.result()
                except Exception as exc:          # noqa: BLE001
                    errors[s] = exc
        return results, errors

    def minute_bars_many(self, symbols, day: dt.date, now: dt.datetime | None = None,
                         workers: int = 8):
        return self._fan_out(symbols, lambda s: self.minute_bars(s, day, now=now), workers)

    def stock_quote_many(self, symbols, workers: int = 8):
        return self._fan_out(symbols, self.stock_quote, workers)

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
        return {"calls": self.calls, "retries": self.retries,
                "bar_cache_hits": self.cache_hits,
                "bar_cache_misses": self.cache_misses}


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
