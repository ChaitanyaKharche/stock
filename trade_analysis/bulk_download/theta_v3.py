"""Low-level ThetaData v3 client built for bulk archival, not for analysis.

Why this exists alongside trade_analysis/data_sources/thetadata_client.py:
that client is tuned for interactive research - it asks for JSON, parses into a
DataFrame, and caches whole responses in memory on the way. Both choices are
wrong here. A single SPY tick-quote session is 185 MB of JSON; holding it in
memory to build a DataFrame is how you OOM a 100k-request run. So this client:

  * asks for format=csv          (~half the bytes of the equivalent JSON)
  * STREAMS straight to a .csv.gz on disk, never materialising the body
  * classifies the response instead of raising, so a caller looping over
    100k (symbol, date) pairs can record and continue

The response classification is the important part. These four outcomes are NOT
interchangeable and conflating them silently corrupts a dataset:

  OK          got rows; written to disk
  NO_DATA     472/404 - the API's documented "this request has no data". A
              PERMANENT fact (holiday, contract didn't exist, symbol not listed
              yet). Recorded so it is never requested again.
  NOT_ENTITLED 403 - the subscription does not cover this. Also permanent, but
              means something completely different: the data EXISTS and we
              can't see it. Must never be recorded as NO_DATA, or the archive
              will look complete when it has a tier-shaped hole in it.
  ERROR       transport/5xx after retries - TRANSIENT. Must be retried on the
              next run, so it is deliberately not treated as terminal.

The distinction between NO_DATA and ERROR is the one that has bitten this
project before: an exception swallowed as "no data" dropped 40 of 256 sessions
from a sample and the run reported n=216 with no warning anywhere.
"""
from __future__ import annotations

import gzip
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import BASE_URL, STOCK_VENUE

# Terminal-side "no rows for this request". 472 is ThetaData's own code; 404
# from a *history* path means the same thing (a 404 from a path that does not
# exist returns HTML, which is checked for separately).
NO_DATA_CODES = (472, 404)
RETRY_CODES = (429, 500, 502, 503, 504)


class TerminalNotRunning(RuntimeError):
    """The local gateway isn't up. Not retryable by waiting inside a request."""


@dataclass
class Result:
    status: str          # OK | NO_DATA | NOT_ENTITLED | ERROR
    rows: int = 0
    bytes_written: int = 0
    path: Path | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    @property
    def terminal(self) -> bool:
        """True if re-requesting this will never produce a different answer."""
        return self.status in ("OK", "NO_DATA", "NOT_ENTITLED")


class ThetaV3:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 600.0,
                 max_attempts: int = 5):
        self.base_url = base_url.rstrip("/")
        # A long read timeout is not laziness: a whole-chain tick request
        # legitimately streams for minutes. connect stays short so a dead
        # terminal is detected immediately rather than after 10 minutes.
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"Accept-Encoding": "gzip"},
        )
        self.max_attempts = max_attempts
        self.live_calls = 0
        self.retries = 0

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ core
    def fetch_csv(self, path: str, dest: Path, **params) -> Result:
        """GET `path` as CSV and stream it gzipped to `dest`.

        Writes to `dest.tmp` and renames only on success, so an interrupted run
        can never leave a half-written file that a later run mistakes for a
        completed download. That matters more than usual here: resumption is
        driven by "does the file exist", across runs that take hours.
        """
        params = {k: v for k, v in params.items() if v is not None}
        params["format"] = "csv"
        url = f"{self.base_url}{path}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")

        last = ""
        for attempt in range(self.max_attempts):
            try:
                with self._client.stream("GET", url, params=params) as resp:
                    if resp.status_code in NO_DATA_CODES:
                        body = resp.read()[:200].decode("utf-8", "replace")
                        # A 404 with an HTML body is a wrong PATH, which is a
                        # bug in the caller, not an empty result. Surfacing it
                        # as NO_DATA would silently skip a whole layer.
                        if "<html" in body.lower():
                            return Result("ERROR", detail=f"bad path {path}")
                        return Result("NO_DATA", detail=body.strip())
                    if resp.status_code == 403:
                        body = resp.read()[:300].decode("utf-8", "replace")
                        return Result("NOT_ENTITLED", detail=body.strip())
                    if resp.status_code in RETRY_CODES:
                        last = f"HTTP {resp.status_code}"
                        raise httpx.ReadError(last)
                    if resp.status_code != 200:
                        body = resp.read()[:300].decode("utf-8", "replace")
                        return Result("ERROR",
                                      detail=f"HTTP {resp.status_code}: {body.strip()}")

                    n_bytes = 0
                    with gzip.open(tmp, "wb", compresslevel=6) as fh:
                        for chunk in resp.iter_bytes(1 << 20):
                            fh.write(chunk)
                            n_bytes += len(chunk)

                self.live_calls += 1

                # An empty body, or a header line with nothing under it, is a
                # real "no data" - some endpoints answer 200 with just a header
                # rather than 472.
                rows = _count_data_rows(tmp)
                if rows == 0:
                    tmp.unlink(missing_ok=True)
                    return Result("NO_DATA", detail="200 with no data rows")

                tmp.replace(dest)
                return Result("OK", rows=rows,
                              bytes_written=dest.stat().st_size, path=dest)

            except httpx.ConnectError as exc:
                tmp.unlink(missing_ok=True)
                raise TerminalNotRunning(
                    "No Theta Terminal on 127.0.0.1:25503. Start it with:\n"
                    "  java -jar ThetaTerminalv3.jar --api-key <key>   (Java 21+)"
                ) from exc
            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError,
                    httpx.RemoteProtocolError, httpx.PoolTimeout) as exc:
                tmp.unlink(missing_ok=True)
                last = f"{type(exc).__name__}: {exc}"
                self.retries += 1
                time.sleep(min(2.0 * (attempt + 1), 15.0))

        return Result("ERROR", detail=f"failed after {self.max_attempts}: {last}")

    def probe(self, path: str, **params) -> tuple[int, str]:
        """Cheap status-only call, for entitlement and history-floor mapping.

        Reads at most a few KB - a floor search does ~15 of these per layer and
        must not pull a 185 MB body each time just to learn "yes, rows exist".
        """
        params = {k: v for k, v in params.items() if v is not None}
        params["format"] = "csv"
        try:
            with self._client.stream("GET", f"{self.base_url}{path}",
                                     params=params) as resp:
                head = b""
                if resp.status_code == 200:
                    for chunk in resp.iter_bytes(4096):
                        head = chunk
                        break
                else:
                    head = resp.read()[:300]
                resp.close()
            return resp.status_code, head.decode("utf-8", "replace")
        except httpx.ConnectError as exc:
            raise TerminalNotRunning("No Theta Terminal on 25503") from exc
        except httpx.HTTPError as exc:
            return -1, f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------- reference lists
    def list_json(self, path: str, **params) -> list[dict]:
        """Reference endpoints only (expirations/strikes/roots). Small enough
        to want JSON, since the CSV shape of these varies by endpoint."""
        params = {k: v for k, v in params.items() if v is not None}
        params["format"] = "json"
        resp = self._client.get(f"{self.base_url}{path}", params=params)
        if resp.status_code in NO_DATA_CODES:
            return []
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("response", []) if isinstance(payload, dict) else payload

    def list_expirations(self, symbol: str) -> list[str]:
        rows = self.list_json("/option/list/expirations", symbol=symbol)
        return sorted({r["expiration"] for r in rows if r.get("expiration")})

    def list_strikes(self, symbol: str, expiration: str) -> list[float]:
        rows = self.list_json("/option/list/strikes", symbol=symbol,
                              expiration=_d(expiration))
        return sorted({float(r["strike"]) for r in rows if r.get("strike") is not None})

    # ------------------------------------------------------------ stock calls
    def stock_history(self, endpoint: str, symbol: str, start: str, end: str,
                      dest: Path, interval: str | None = None) -> Result:
        """endpoint in {ohlc, quote, trade, eod}. `interval` omitted => tick."""
        return self.fetch_csv(
            f"/stock/history/{endpoint}", dest,
            symbol=symbol, start_date=_d(start), end_date=_d(end),
            interval=interval, venue=STOCK_VENUE,
        )

    # ----------------------------------------------------------- option calls
    def option_history(self, endpoint: str, symbol: str, expiration: str,
                       date: str, dest: Path, interval: str | None = None,
                       strike: float | None = None, right: str | None = None,
                       strike_range: int | None = None) -> Result:
        """endpoint in {quote, ohlc, open_interest, eod}.

        Leaving `strike` as None pulls the WHOLE chain for that expiration in
        one request - the single biggest efficiency win available here, and the
        reason option coverage is affordable at all. `strike_range` caps how far
        from spot to reach so the payload stays sane.
        """
        params = dict(symbol=symbol, expiration=_d(expiration), date=_d(date),
                      interval=interval, strike_range=strike_range)
        if strike is not None:
            # The gateway wants 3 decimal places. "532000" silently returns 472
            # rather than erroring, which reads exactly like "no such contract" -
            # this cost an hour of believing option data was unavailable.
            params["strike"] = f"{float(strike):.3f}"
        if right is not None:
            params["right"] = "call" if right.upper().startswith("C") else "put"
        return self.fetch_csv(f"/option/history/{endpoint}", dest, **params)


def _d(value) -> str:
    """Any date-ish thing -> YYYYMMDD."""
    return str(value).replace("-", "").replace("/", "")[:8]


def _count_data_rows(gz_path: Path) -> int:
    """Data rows in a gzipped CSV, excluding the header. Streams; these files
    reach hundreds of MB and must not be read into memory to be counted."""
    try:
        with gzip.open(gz_path, "rb") as fh:
            n = sum(1 for _ in fh)
    except (OSError, EOFError):
        return 0
    return max(0, n - 1)


def decompress_preview(gz_path: Path, n_lines: int = 5) -> list[str]:
    """First few lines, for eyeballing a layer without unpacking a huge file."""
    out = []
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n_lines:
                break
            out.append(line.rstrip("\n"))
    return out


def gunzip_to(gz_path: Path, dest: Path):
    with gzip.open(gz_path, "rb") as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst)
