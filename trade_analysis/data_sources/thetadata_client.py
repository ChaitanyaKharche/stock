"""
ThetaData v3 client - OPTIONS ENDPOINTS ONLY.

SUBSCRIPTION REALITY (re-measured 2026-08-15 after the Stock STANDARD upgrade):
    Stock: STANDARD   Options: VALUE   Index: FREE

  - /option/*  -> Value: 1-minute bid/ask quotes and OHLC, whole-chain fetches
                  via strike=*. History floor 2020-01-01 (see below).
  - /stock/*   -> Standard: 1-minute OHLC verified back to 2016.
  - /index/*   -> still FREE, so SPX/VIX/VIX1D intraday all 403 with
                  "requires a standard subscription". EOD only.

`get()` HARD-REJECTS any path outside /option/ and /stock/. Enforced in code
rather than left as a convention so a future edit cannot silently reintroduce
a call that 403s at runtime - or worse, that quietly implies an upgrade is
needed. Index is deliberately excluded: it is the one line still unpaid.

THE BINDING CONSTRAINT HAS FLIPPED. Stock used to be the floor at 2021-01-01
and options reached further back; after the upgrade it is the reverse, and
OPTIONS now decides how far any joint study can go. Do not assume the stock
floor is the study floor - it is not, and it was the other way round for the
entire 2021-2026 run.

TRANSPORT: v3 is a LOCAL gateway, not a cloud API. The Theta Terminal must be
running:
    java -jar ThetaTerminalv3.jar --api-key <key>       (Java 21+ required)
It listens on 127.0.0.1:25503. There is no per-minute rate limit like Massive's
5/min, but the Value tier allows only 2 concurrent requests, so this client is
deliberately sequential.

RESPONSE SHAPE: {"response": [{"contract": {...}, "data": [ {...}, ... ]}]}
One element per contract, so a chain request returns many. `_flatten` folds the
contract metadata down into the rows.

TIMEZONES: timestamps come back naive ("2024-08-07T09:30:00.000") and are
Eastern. They are localized explicitly - this machine is on Mountain time, so
leaving them naive shifts every entry window by two hours.
"""
import gzip
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import httpx

from ..paths import CACHE_DIR

BASE_URL = "http://127.0.0.1:25503/v3"
ET = "America/New_York"
CACHE_ROOT = CACHE_DIR / "thetadata"

# Value tier: option history starts here. Requests before this return empty.
HISTORY_START = "2020-01-01"

# Endpoint families we actually hold a paid subscription for. Index is
# deliberately absent: it is still on the FREE tier and its intraday endpoints
# require Index STANDARD, so calling them only produces PERMISSION_DENIED noise.
_ALLOWED_PREFIXES = ("/option/", "/stock/")

# Earliest data each paid line will serve, established by binary search against
# the live terminal rather than taken from marketing copy. Re-measured
# 2026-08-15 after the Stock STANDARD upgrade:
#   options 1-min quotes -> 2019-12-20 = 403 tier, 2020-01-03 = data.
#                           Floor is 2020-01-01. (The previous 2020-03-04 here
#                           was simply the earliest date anyone had tried, not
#                           a measured boundary - it understated reach by 2mo.)
#   stock   1-min OHLC   -> verified back to 2016; no longer binding.
# OPTIONS is therefore now the binding constraint on any study needing both.
OPTION_HISTORY_START = "2020-01-01"
STOCK_HISTORY_START = "2016-01-01"

# The floor that actually matters for an options study needing underlying bars.
# Use this rather than either constant above so the two can never drift apart.
JOINT_HISTORY_START = max(OPTION_HISTORY_START, STOCK_HISTORY_START)


class ThetaTerminalNotRunning(RuntimeError):
    pass


class ThetaDataClient:
    def __init__(self, base_url=BASE_URL, cache_root=CACHE_ROOT, timeout=180):
        self.base_url = base_url
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self.live_calls = 0
        self.cache_hits = 0
        self.transport_retries = 0

    # ---------------- transport ----------------
    def _cache_path(self, path, params):
        digest = hashlib.sha1(
            json.dumps({"p": path, "q": params}, sort_keys=True).encode()
        ).hexdigest()[:20]
        bucket = path.strip("/").replace("/", "_")
        d = self.cache_root / bucket
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{digest}.json.gz"

    def get(self, path, **params):
        """GET an /option/ endpoint, with disk cache.

        Raises on any non-option path: we do not hold a paid Stock or Index
        subscription and calling those only produces PERMISSION_DENIED noise.
        """
        if not path.startswith(_ALLOWED_PREFIXES):
            raise ValueError(
                f"Refusing to call {path!r}. This client is restricted to "
                f"{_ALLOWED_PREFIXES} - those are the lines we hold paid (Value) "
                f"subscriptions for. Index is still FREE and its intraday "
                f"endpoints require Index STANDARD, so they would 403."
            )
        params.setdefault("format", "json")
        cache_file = self._cache_path(path, params)
        if cache_file.exists():
            self.cache_hits += 1
            with gzip.open(cache_file, "rt", encoding="utf-8") as fh:
                return json.load(fh)

        # Retry transient transport failures. This is not defensive padding: a
        # chain request is a large payload and the Value tier allows only 2
        # concurrent requests, so timeouts happen under sustained load. An
        # exception is NOT cached, so a caller that treats it as "no data" drops
        # that session from the sample silently. That is exactly what happened on
        # the first full run - 40 of 256 SPY sessions vanished and the reported
        # n was 216, with no error surfaced anywhere.
        resp = None
        data = None
        last_exc = None
        for attempt in range(4):
            try:
                resp = self._client.get(self.base_url + path, params=params)
                # Parse INSIDE the retry. A 200 whose body is truncated mid-array
                # raises JSONDecodeError, which is not an httpx error and so used
                # to escape the loop and kill the run outright - observed on a
                # large premarket chunk at char 184935. It is the same failure
                # class as the transport errors above (big payload, 2-concurrent
                # Value tier) and deserves the same treatment, not a crash.
                # 5xx from the local gateway is transient - it proxies an
                # upstream that intermittently returns "io exception" under
                # sustained load on the 2-concurrent Value tier. Observed
                # killing a 10-year run after the first symbol had completed.
                # Retry rather than propagate, same as the transport errors.
                if resp.status_code in (500, 502, 503, 504):
                    raise httpx.ReadError(f"HTTP {resp.status_code}: {resp.text[:80]}")
                if resp.status_code == 200:
                    data = resp.json()
                break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError,
                    httpx.ReadError, json.JSONDecodeError) as exc:
                last_exc = exc
                resp = None
                self.transport_retries += 1
                time.sleep(1.5 * (attempt + 1))
        if resp is None:
            if isinstance(last_exc, httpx.ConnectError):
                raise ThetaTerminalNotRunning(
                    "Could not reach the Theta Terminal on 127.0.0.1:25503. Start it "
                    "with:  java -jar ThetaTerminalv3.jar --api-key <key>   (Java 21+)"
                ) from last_exc
            raise RuntimeError(
                f"{path} failed after 4 attempts: {type(last_exc).__name__}: {last_exc}"
            ) from last_exc

        self.live_calls += 1
        if resp.status_code == 200:
            pass  # already parsed inside the retry loop above
        elif resp.status_code in (472, 404):
            # documented "no data for this request" - a permanent fact, cache it
            data = {"response": [], "_note": "no data"}
        elif resp.status_code == 403:
            raise PermissionError(
                f"403 from {path}: {resp.text[:200]}\n"
                f"This client should only touch /option/ endpoints on the Value "
                f"tier - if you see this, the request exceeded that."
            )
        else:
            raise RuntimeError(f"HTTP {resp.status_code} from {path}: {resp.text[:200]}")

        with gzip.open(cache_file, "wt", encoding="utf-8") as fh:
            json.dump(data, fh)
        return data

    # ---------------- parsing ----------------
    @staticmethod
    def _flatten(payload):
        """{"response":[{contract, data:[...]}]} -> one tidy DataFrame."""
        blocks = payload.get("response", []) if isinstance(payload, dict) else payload
        frames = []
        for block in blocks or []:
            rows = block.get("data") or []
            if not rows:
                continue
            df = pd.DataFrame(rows)
            c = block.get("contract", {}) or {}
            df["strike"] = c.get("strike")
            df["right"] = (c.get("right") or "")[:1].upper()   # CALL/PUT -> C/P
            df["expiration"] = c.get("expiration")
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if "timestamp" in out.columns:
            out["timestamp"] = (pd.to_datetime(out["timestamp"])
                                  .dt.tz_localize(ET, nonexistent="shift_forward",
                                                  ambiguous=True))
            out = out.sort_values("timestamp")
        return out

    # ---------------- public API ----------------
    def list_expirations(self, symbol):
        data = self.get("/option/list/expirations", symbol=symbol)
        rows = data.get("response", []) if isinstance(data, dict) else data
        if not rows:
            return []
        return sorted({r["expiration"] for r in rows if "expiration" in r})

    def quotes(self, symbol, expiration, date, right=None, strike=None,
               interval="1m", start_time=None, end_time=None, strike_range=None):
        """1-minute (or other interval) NBBO quotes.

        Leave `strike` as None to pull the WHOLE chain in a single request -
        that is the big efficiency win over per-contract vendors. Narrow it with
        `strike_range` (strikes above/below spot) to keep the payload sane.

        Returns columns: timestamp (ET), bid, ask, bid_size, ask_size, strike,
        right, expiration.
        """
        params = {"symbol": symbol, "expiration": str(expiration).replace("-", ""),
                  "date": str(date).replace("-", ""), "interval": interval}
        if right:
            params["right"] = "call" if right.upper().startswith("C") else "put"
        if strike is not None:
            params["strike"] = f"{float(strike):.3f}"
        if strike_range is not None:
            params["strike_range"] = strike_range
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        return self._flatten(self.get("/option/history/quote", **params))

    def ohlc(self, symbol, expiration, date, right=None, strike=None,
             interval="1m", strike_range=None):
        """Traded OHLC for the same contracts - useful to cross-check whether
        traded prices tracked executable quotes."""
        params = {"symbol": symbol, "expiration": str(expiration).replace("-", ""),
                  "date": str(date).replace("-", ""), "interval": interval}
        if right:
            params["right"] = "call" if right.upper().startswith("C") else "put"
        if strike is not None:
            params["strike"] = f"{float(strike):.3f}"
        if strike_range is not None:
            params["strike_range"] = strike_range
        return self._flatten(self.get("/option/history/ohlc", **params))

    # ---------------- stock (Stock Value tier) ----------------
    def stock_minute_bars(self, symbol, start, end, interval="1m"):
        """Regular-session minute bars for an equity/ETF.

        Returned already limited to 09:30-16:00 (391 bars/session), unlike
        Massive's feed which serves 04:00-20:00 and needed both a filter and a
        chunk-size workaround to avoid silently truncating at a 50k page cap.
        """
        # venue=utp_cta is the consolidated tape (UTP+CTA). Required because the
        # Stock Value tier has no REAL-TIME entitlement, and any range touching
        # today fails with "Real time data unavailable with current stock
        # subscription". Verified it does NOT degrade history: for 2022-06-21 the
        # default and utp_cta responses are 391 bars with byte-identical OHLC.
        payload = self.get("/stock/history/ohlc", symbol=symbol,
                           start_date=str(start).replace("-", ""),
                           end_date=str(end).replace("-", ""), interval=interval,
                           venue="utp_cta")
        blocks = payload.get("response", []) if isinstance(payload, dict) else payload
        rows = []
        for block in blocks or []:
            rows.extend(block.get("data", []) if isinstance(block, dict) and "data" in block
                        else [block])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["timestamp"] = (pd.to_datetime(df["timestamp"])
                             .dt.tz_localize(ET, nonexistent="shift_forward",
                                             ambiguous=True))
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                "close": "Close", "volume": "Volume",
                                "count": "n_trades"})
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "vwap",
                            "n_trades"] if c in df.columns]
        return df.set_index("timestamp")[keep].sort_index()

    def stats(self):
        return {"live_calls": self.live_calls, "cache_hits": self.cache_hits,
                "retries": self.transport_retries}


if __name__ == "__main__":
    c = ThetaDataClient()
    print("guard check:")
    try:
        c.get("/stock/history/ohlc", symbol="SPY")
    except ValueError as e:
        print(f"  refused non-option path as designed: {str(e)[:90]}...")

    exps = c.list_expirations("SPY")
    print(f"\nSPY expirations listed: {len(exps)} ({exps[0]} -> {exps[-1]})")

    q = c.quotes("SPY", "2024-08-07", "2024-08-07", right="call", strike=532)
    q = q[(q.bid > 0) & (q.ask > 0)]
    print(f"\nSPY 532C 0DTE quotes: {len(q)} quoted minutes")
    print(q[["timestamp", "bid", "ask", "bid_size", "ask_size"]].head(3).to_string(index=False))
    print(f"\n{c.stats()}")
