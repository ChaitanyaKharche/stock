"""Where the archive lives and what the v3 gateway will actually serve.

Everything here was measured against the live terminal on 2026-08-20, not read
off marketing copy. The numbers that matter are the REQUEST SHAPE LIMITS below:
they are the reason this package exists as something other than a for-loop.
"""
from pathlib import Path
import os

# ---------------------------------------------------------------- destination
# The archive is deliberately NOT inside the repo. It reaches tens of GB and
# would poison every git operation. Override with THETA_DATA_ROOT if the disk
# ever changes.
DATA_ROOT = Path(os.environ.get(
    "THETA_DATA_ROOT",
    Path.home() / "Desktop" / "data",
))

REFERENCE_DIR = DATA_ROOT / "reference"
RAW_DIR = DATA_ROOT / "raw"          # gzipped CSV exactly as the API returned it
PARQUET_DIR = DATA_ROOT / "parquet"  # typed, tz-aware, query-ready
LOG_DIR = DATA_ROOT / "logs"
STATE_DB = DATA_ROOT / "state" / "manifest.sqlite"

# ---------------------------------------------------------------- transport
BASE_URL = os.environ.get("THETA_BASE_URL", "http://127.0.0.1:25503/v3")
ET = "America/New_York"

# The terminal is a LOCAL gateway; the API key is configured on the terminal
# process, not sent per request. It is recorded here only so the launcher can
# start the terminal, and is read from the environment so it stays out of git.
API_KEY_ENV = "THETADATA_API_KEY"

# ---------------------------------------------------------- subscription tier
# Measured 2026-08-20 by calling each endpoint and reading the 403 text:
#   Stock  -> STANDARD : trade ticks + quote ticks + all intervals. Confirmed.
#   Option -> VALUE    : quote at any interval INCLUDING tick; /option/history/
#                        trade is 403 "requires a standard subscription".
#   Index  -> FREE     : intraday 403s. Excluded entirely.
TIERS = {"stock": "STANDARD", "option": "VALUE", "index": "FREE"}

# --------------------------------------------------------- REQUEST SHAPE LIMITS
# Enforced by the gateway, discovered by hitting them:
#   "Bulk history requests are limited to intervals of at least 1 minute."
#   "Bulk history requests are limited to no more than 1 month."
# Meaning: a request spanning >1 day MUST use interval >= 1m, and may never
# span more than a month. Anything sub-minute is strictly one date per request.
# This is what sets the shape of every downloader here: minute data is chunked
# monthly (cheap), second/tick data is one request per symbol per session
# (expensive), and there is no way to batch the latter.
BULK_MAX_DAYS = 31
BULK_MIN_INTERVAL_SECONDS = 60
SUBMINUTE_INTERVALS = {"tick", "100ms", "500ms", "1s", "5s", "10s", "15s", "30s"}

# Concurrency. The Value tier documents 2 concurrent requests; Standard allows
# more but the binding constraint is that the terminal proxies an upstream that
# starts returning "io exception" 500s under sustained load. 4 is empirically
# stable for stock, 2 for option.
MAX_CONCURRENCY = {"stock": 4, "option": 2}

# --------------------------------------------------------------- history floors
# Placeholders, overwritten by probe_entitlements.py which binary-searches the
# real first available date per (security, endpoint, interval). Never hardcode a
# study's start date from these - read reference/entitlements.json instead.
FALLBACK_FLOORS = {
    "stock.ohlc.1m": "2016-01-01",
    "stock.quote.1s": "2020-01-01",
    "stock.trade.tick": "2020-01-01",
    "option.quote.1m": "2020-01-01",
}

# venue=utp_cta is the consolidated tape. Required on any range touching today
# without a real-time entitlement, and verified byte-identical to the default
# on history, so it is applied unconditionally to every stock call.
STOCK_VENUE = "utp_cta"

for _d in (REFERENCE_DIR, RAW_DIR, PARQUET_DIR, LOG_DIR, STATE_DB.parent):
    _d.mkdir(parents=True, exist_ok=True)
