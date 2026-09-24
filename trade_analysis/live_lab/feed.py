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

from .clock import now_et

try:
    from ..bulk_download.trading_days import is_trading_day as _is_session
except Exception:                                            # pragma: no cover
    def _is_session(d):                                      # noqa: D103
        return d.weekday() < 5

BASE_URL = "http://127.0.0.1:25503/v3"
SESSION_OPEN = dt.time(9, 30)
SESSION_LAST_BAR = dt.time(15, 59)
BAR_SECONDS = 60          # must match session.BAR_SECONDS; see _serve_from_cache

# How long a caller with time to spare should wait for the terminal's UPSTREAM before
# giving up on the session. Lives here, not in either runner, because both arms must
# use the same number: a resilience rule that holds on the shares arm and not the
# options arm would make the two records differ for reasons unrelated to strategy.
#
# 180s, against the ~7s per-call retry budget in _get_csv. The two are not the same
# kind of wait -- a tick cannot block without stalling every other symbol, whereas
# warmup runs pre-open with minutes going spare. Sized for a network transition
# (DHCP lease, new resolvers, the JVM's negative DNS cache expiring, MDDS
# reconnecting), not for a terminal cold start.
UPSTREAM_WAIT_SEC = 180.0


class FeedOutage(Exception):
    """Raised when the terminal cannot be reached after the configured retries."""


class UpstreamUnreachable(FeedOutage):
    """The terminal is answering, but IT cannot reach ThetaData. Recoverable by waiting.

    The signature, and 282 of 284 recorded feed outages in `live_lab_data/outages.jsonl`:

        HTTP 503: Unable to resolve host mdds-01.thetadata.us

    `/stock/snapshot/quote` is served by the terminal's own process and keeps answering
    200 throughout, while `/stock/history/ohlc` goes through MDDS and fails. So this
    state is invisible to any liveness check that pings the socket or asks for a quote.

    What produces it, on this lab: the machine moves between WiFi and a phone hotspot
    (it travels in a car). The interface change invalidates the JVM's DNS resolution and
    kills the terminal's upstream sockets, and the terminal then reconnects on its own
    schedule -- tens of seconds, sometimes longer. Nothing is broken and nothing needs
    restarting; the only correct response is to WAIT.

    Distinguished from its sibling because the responses are opposite: this one wants
    patience, `TerminalUnreachable` wants a relaunch. Treating them alike is how a
    30-second hotspot switch cost whole sessions -- 2026-08-28 `warmup_feed_unreachable`,
    and 202 outage records on 2026-09-01.
    """


class TerminalUnreachable(FeedOutage):
    """Nothing is listening on 25503, or it accepted and then dropped us.

    `ConnectError('[WinError 10061] ... actively refused it')`. Waiting does not fix
    this; the terminal has to be started or evicted.
    """


# Substrings that mean "the local terminal is fine, its upstream is not". Matched against
# the response body and the exception text, because the condition arrives as an HTTP 503
# from a process that is working correctly -- not as a transport error.
_UPSTREAM_SIGNS = (
    "unable to resolve host",     # the observed one: JVM DNS after an interface change
    "mdds",                       # the historical-data upstream, named in the 503 body
    "fpss",                       # the same for option chains
    "no route to host",
    "temporarily unavailable",
    "connection reset",
)

# Socket-level errors that a network interface change produces directly. OSError(22) was
# recorded once on 2026-09-02; it is what a send on a socket whose route just vanished
# looks like on Windows.
_TRANSIENT_SIGNS = (
    "winerror 10051",             # network is unreachable
    "winerror 10065",             # no route to host
    "winerror 10054",             # connection reset by peer
    "invalid argument",           # OSError(22) mid-switch
    "timed out",
    "timeout",
)


def _classify(status: int | None, text: str) -> type[FeedOutage]:
    """Which kind of outage this is. Defaults to the base class, never guesses upward."""
    blob = text.lower()
    if any(s in blob for s in _UPSTREAM_SIGNS):
        return UpstreamUnreachable
    if status in (502, 503, 504):
        return UpstreamUnreachable      # the terminal answered; something behind it did not
    if "10061" in blob or "actively refused" in blob or "connectionrefused" in blob:
        return TerminalUnreachable
    if any(s in blob for s in _TRANSIENT_SIGNS):
        return UpstreamUnreachable
    return FeedOutage


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
        # Outage EPISODES, not failed calls. See _note_upstream_fail.
        self._outage_since: dt.datetime | None = None
        self._outage_calls = 0
        self._outage_detail = ""
        self.outage_episodes = 0

    # ------------------------------------------------------------------ transport

    def _get_csv(self, path: str, **params) -> list[dict[str, Any]]:
        url = f"{self.base_url}{path}"
        delay = 1.0
        last: Exception | None = None
        kind: type[FeedOutage] = FeedOutage
        for attempt in range(self.max_retries):
            try:
                self.calls += 1
                r = self._client.get(url, params=params)
                if r.status_code == 472:          # documented "no data" response
                    return []
                if r.status_code != 200:
                    kind = _classify(r.status_code, r.text)
                    raise httpx.HTTPError(f"HTTP {r.status_code}: {r.text[:120]}")
                text = r.text
                if not text.strip():
                    return []
                self._note_upstream_ok()
                return list(csv.DictReader(io.StringIO(text)))
            except Exception as exc:              # noqa: BLE001 - transport is broad by nature
                last = exc
                if kind is FeedOutage:
                    kind = _classify(None, repr(exc))
                self.retries += 1
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
        # The per-call retry budget is deliberately still short (~7s). Blocking a tick
        # for minutes would stall every other symbol, and the bar cache plus the
        # stale-bar guard already make a missed tick harmless. Patience belongs in
        # wait_for_upstream(), called from the places that have time to spend.
        self._note_upstream_fail(path, repr(last))
        raise kind(f"{path} failed after {self.max_retries} attempts: {last!r}")

    # ------------------------------------------------------- outage episode tracking

    def _note_upstream_ok(self) -> None:
        """A successful call closes any open outage episode."""
        if self._outage_since is None:
            return
        began, n = self._outage_since, self._outage_calls
        self._outage_since, self._outage_calls, self._outage_detail = None, 0, ""
        secs = (now_et() - began).total_seconds()
        self.outage_episodes += 1
        if self._on_outage:
            self._on_outage("__recovered__",
                            f"upstream recovered after {secs:.0f}s and {n} failed call(s)")

    def _note_upstream_fail(self, path: str, detail: str) -> None:
        """Report the START of an outage, then stay quiet until it changes or ends.

        Every failed call used to append its own record. On 2026-09-01 that produced 202
        identical `Unable to resolve host mdds-01.thetadata.us` lines, and on 2026-09-04
        another 76. That is not a log, it is the same fact 278 times -- and it buried the
        one `OSError(22, 'Invalid argument')` that was genuinely different. Same reasoning
        as `ledger.record(dedupe=True)`: the counter still rises, the rows do not.
        """
        self._outage_calls += 1
        changed = detail[:120] != self._outage_detail
        if self._outage_since is None or changed:
            self._outage_since = self._outage_since or now_et()
            self._outage_detail = detail[:120]
            if self._on_outage:
                self._on_outage(path, detail)

    def wait_for_upstream(self, budget_s: float = 180.0, probe_every: float = 5.0,
                          probe=None, on_wait=None) -> bool:
        """Block until the terminal's UPSTREAM serves data again, or the budget runs out.

        Only for callers with time to spend -- warmup, preflight, a pre-open supervisor.
        Never call this from inside a tick.

        The probe must exercise the upstream, not the terminal. `/stock/snapshot/quote`
        is served by the terminal's own process and answers 200 right through an MDDS
        outage, so probing it would report recovery that has not happened. The default
        asks for yesterday's bars, which go through MDDS -- the thing that actually
        breaks. This is the same lesson as `autostart._wait_for_history`: "the socket is
        open" was never the question.

        Returns True if the upstream answered within the budget.
        """
        probe = probe or self._default_probe
        deadline = time.time() + budget_s
        attempt = 0
        while True:
            attempt += 1
            try:
                probe()
                if attempt > 1 and on_wait:
                    on_wait(f"upstream ready after {attempt} probe(s)")
                return True
            except FeedOutage as exc:
                if attempt == 1 and on_wait:
                    on_wait(f"waiting up to {budget_s:.0f}s for the upstream: "
                            f"{str(exc)[:90]}")
            if time.time() >= deadline:
                if on_wait:
                    on_wait(f"upstream still down after {budget_s:.0f}s "
                            f"and {attempt} probe(s)")
                return False
            time.sleep(min(probe_every, max(0.0, deadline - time.time())))

    def _default_probe(self) -> None:
        """Ask for the most recent prior session's bars: goes through MDDS, and is cheap."""
        day = now_et().date() - dt.timedelta(days=1)
        for _ in range(30):
            if _is_session(day):
                break
            day -= dt.timedelta(days=1)
        # Bypass the bar cache -- a cached answer would prove nothing about the upstream.
        self._fetch_minute_bars("QQQ", day)

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
            # WHEN THIS ROW ARRIVED, not when the caller started its poll. Quote age must
            # be measured against receipt, and both callers used to measure it against a
            # `now` sampled before a batch of network calls -- so a slow batch made every
            # quote look FUTURE-dated. Measured 2026-09-09: preflight took 145s per run
            # with `now` fixed at the start, and printed 15 x "[OK] REAL-TIME (age -50s)"
            # while the runner, minutes later, threw 30 `future_quote` outages on the same
            # feed. Neither the clock nor the feed was wrong; the reference point was.
            "recv_ts": now_et(),
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
                "bar_cache_misses": self.cache_misses,
                "outage_episodes": self.outage_episodes,
                "upstream_down_now": self._outage_since is not None}


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
