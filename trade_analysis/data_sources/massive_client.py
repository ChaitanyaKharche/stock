"""
Rate-limited, disk-cached client for Massive.com (formerly Polygon.io).

Built for the FREE "Basic" Stocks + Options tiers, whose constraints shape
every design decision in here:

  - 5 API calls per minute. That is the binding constraint on everything. A
    full 0DTE study is thousands of calls, so every response is cached to disk
    and the fetch is resumable: re-running after an interruption costs nothing
    for work already done.
  - 2 years of history.
  - Minute aggregates, and reference data including expired contracts.
  - NO bid/ask. NBBO quotes are Advanced-tier ($199/mo) only. So option bars
    here are TRADED prices (o/h/l/c/vw), not quotes. Any fill assumption must
    come from elsewhere - the `bas` column in the Vilkov SPX panel is the
    intended calibration source - and `n` (transaction count per bar) is the
    liquidity filter that partly compensates.

TIMEZONE: the vendor returns epoch milliseconds. Rendering those with local
time is a real bug on a non-ET machine (this one is Mountain, so 09:30 ET
renders as 07:30 and a 9:30-9:45 entry window silently becomes 11:30-11:45).
Everything here converts to America/New_York explicitly, always.

Holds no strategy logic - fetch and cache only.
"""
import gzip
import hashlib
import json
import os
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from ..paths import CACHE_DIR

BASE_URL = "https://api.massive.com"
ET = "America/New_York"

# 5 calls/min = one per 12s. 12.6 buys headroom against clock skew; getting
# 429-throttled costs far more than the extra half-second.
MIN_SECONDS_BETWEEN_CALLS = 12.6
MAX_RETRIES = 4

CACHE_ROOT = CACHE_DIR / "massive"


def _load_api_key():
    """Read MASSIVE_API_KEY from the environment, falling back to the project's
    .env (which is gitignored). No third-party dotenv dependency needed."""
    key = os.environ.get("MASSIVE_API_KEY")
    if key:
        return key
    env_path = CACHE_DIR.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("MASSIVE_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "MASSIVE_API_KEY not set. Put it in the project's .env as "
        "MASSIVE_API_KEY=... (that file is gitignored) or export it."
    )


def occ_ticker(symbol, expiry, option_type, strike):
    """Build an OCC option ticker, e.g. O:SPY260114C00696000.

    Deterministic, which matters a lot on a 5-call/min budget: constructing the
    ticker directly avoids ~1 reference call per session just to discover it.
    `option_type` is 'C'/'P' or 'call'/'put'; strike is in dollars.
    """
    if isinstance(expiry, str):
        expiry = date.fromisoformat(expiry)
    ct = option_type[0].upper()
    if ct not in ("C", "P"):
        raise ValueError(f"option_type must be call/put, got {option_type!r}")
    # strike is encoded as strike * 1000, zero-padded to 8 digits
    strike_code = f"{int(round(float(strike) * 1000)):08d}"
    return f"O:{symbol}{expiry:%y%m%d}{ct}{strike_code}"


class MassiveClient:
    def __init__(self, api_key=None, cache_root=CACHE_ROOT, verbose=True):
        self.api_key = api_key or _load_api_key()
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._session = requests.Session()
        self._last_call = 0.0
        self.calls_made = 0      # live calls only; cache hits don't count
        self.cache_hits = 0

    # ---------------- caching ----------------
    def _cache_path(self, path, params):
        payload = json.dumps({"path": path, "params": params}, sort_keys=True)
        digest = hashlib.sha1(payload.encode()).hexdigest()[:20]
        bucket = path.strip("/").split("/")[1] if "/" in path.strip("/") else "misc"
        d = self.cache_root / bucket
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{digest}.json.gz"

    # ---------------- transport ----------------
    def _throttle(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
        self._last_call = time.monotonic()

    def get(self, path, **params):
        """GET with disk cache, throttling and retry.

        Empty results are cached too. A 0DTE strike that never traded is a
        permanent fact, and re-asking costs 12 seconds each time.
        """
        cache_file = self._cache_path(path, params)
        if cache_file.exists():
            self.cache_hits += 1
            with gzip.open(cache_file, "rt", encoding="utf-8") as fh:
                return json.load(fh)

        last_err = None
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                resp = self._session.get(
                    BASE_URL + path,
                    params={**params, "apiKey": self.api_key},
                    timeout=60,
                )
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(2 ** attempt)
                continue

            self.calls_made += 1
            if resp.status_code == 200:
                data = resp.json()
                with gzip.open(cache_file, "wt", encoding="utf-8") as fh:
                    json.dump(data, fh)
                return data
            if resp.status_code == 429:
                wait = MIN_SECONDS_BETWEEN_CALLS * (attempt + 2)
                if self.verbose:
                    print(f"    429 rate-limited, backing off {wait:.0f}s")
                time.sleep(wait)
                last_err = "429"
                continue
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"HTTP {resp.status_code} from Massive - key rejected or "
                    f"endpoint not on the free tier: {resp.text[:200]}"
                )
            # 404 and other 4xx: treat as a definitive empty answer and cache it
            if 400 <= resp.status_code < 500:
                data = {"status": "NOT_FOUND", "results": [],
                        "_http_status": resp.status_code}
                with gzip.open(cache_file, "wt", encoding="utf-8") as fh:
                    json.dump(data, fh)
                return data
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            time.sleep(2 ** attempt)

        raise RuntimeError(f"{path} failed after {MAX_RETRIES} attempts: {last_err}")

    # ---------------- bars ----------------
    @staticmethod
    def _bars_to_frame(results):
        """Vendor aggregate rows -> DataFrame indexed by ET timestamp.

        Columns kept: Open/High/Low/Close/Volume plus vwap and n_trades. vwap is
        usually a better fill proxy than Close on a thin 0DTE strike, and
        n_trades is the only liquidity signal available without quotes.
        """
        if not results:
            return pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume", "vwap", "n_trades"]
            )
        df = pd.DataFrame(results)
        df["timestamp"] = (pd.to_datetime(df["t"], unit="ms", utc=True)
                             .dt.tz_convert(ET))
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                                "c": "Close", "v": "Volume",
                                "vw": "vwap", "n": "n_trades"})
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume",
                            "vwap", "n_trades"] if c in df.columns]
        return df.set_index("timestamp")[keep].sort_index()

    def _aggregates(self, ticker, start, end, multiplier=1, timespan="minute"):
        path = (f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/"
                f"{start}/{end}")
        data = self.get(path, adjusted="true", sort="asc", limit=50000)
        frame = self._bars_to_frame(data.get("results") or [])
        # 50k is the hard page size; warn rather than silently truncate
        if len(frame) >= 50000 and self.verbose:
            print(f"    WARNING {ticker} {start}->{end} hit the 50,000-row page "
                  f"limit; narrow the range or results are truncated.")
        return frame

    def stock_minute_bars(self, symbol, start, end):
        """Minute bars for an equity/ETF. One call covers a whole date range,
        so this is cheap - ~90 calendar days fits inside the 50k page."""
        return self._aggregates(symbol, start, end)

    def option_minute_bars(self, ticker, day):
        """Minute bars for one option contract on one session.

        `ticker` is a full OCC ticker (see occ_ticker). Returns an empty frame
        for a strike that never traded, which is normal for far-OTM 0DTE.
        """
        return self._aggregates(ticker, day, day)

    def expired_contracts(self, underlying, expiration_date, limit=1000):
        """Reference data for contracts expiring on a given (past) date.

        Only needed to discover the real strike ladder; occ_ticker() avoids
        this call when the strike increment is already known.
        """
        data = self.get("/v3/reference/options/contracts",
                        underlying_ticker=underlying,
                        expiration_date=str(expiration_date),
                        expired="true", limit=limit)
        results = data.get("results") or []
        if not results:
            return pd.DataFrame(columns=["ticker", "contract_type", "strike_price"])
        return pd.DataFrame(results)[["ticker", "contract_type", "strike_price"]]

    def stats(self):
        return {"live_calls": self.calls_made, "cache_hits": self.cache_hits}


if __name__ == "__main__":
    c = MassiveClient()
    print("fetching 5 sessions of SPY minute bars...")
    bars = c.stock_minute_bars("SPY", "2026-01-12", "2026-01-16")
    print(f"{len(bars)} bars | {bars.index.min()} -> {bars.index.max()}")
    print("\nregular-hours slice of the first session (ET), first 4 rows:")
    day = bars[bars.index.date == bars.index.date.min()]
    print(day.between_time("09:30", "09:45").head(4).to_string())
    print(f"\n{c.stats()}")
