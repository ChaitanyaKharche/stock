"""Layer definitions: every distinct dataset the archive can hold.

A "layer" is one (endpoint, interval, chunking rule) combination. Declaring them
as data rather than writing a script per dataset is what makes the orchestrator,
the manifest, the cost estimator and the parquet converter share one code path -
and it means adding a dataset is one dict entry, not a new file.

CHUNKING is the field that actually matters, because the gateway enforces two
rules that pull in opposite directions:

    "Bulk history requests are limited to no more than 1 month."
    "Bulk history requests are limited to intervals of at least 1 minute."

So a layer is either MONTHLY (interval >= 1m: one request covers a whole month -
cheap, ~130 requests per symbol for a decade) or DAILY (sub-minute: one request
per session, unavoidably - ~2400 requests per symbol for the same decade). That
20x difference in request count, not storage, is what makes the second/tick
layers the expensive ones and it is why they are scoped to fewer symbols.

TIERS here are download PRIORITY, not subscription tiers:
    0  reference    tiny, one-off, needed by everything else
    1  eod          whole history, whole universe, megabytes
    2  minute       whole history, whole universe, gigabytes
    3  second       the per-second requirement; scoped by symbol group
    4  tick         full resolution; ETFs only by default
    5  option       SPY/QQQ chains
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Layer:
    name: str
    security: str          # stock | option
    endpoint: str          # ohlc | quote | trade | trade_quote | eod | open_interest
    interval: str | None   # None = endpoint takes no interval
    chunk: str             # monthly | daily
    tier: int
    # Which symbol group this layer runs over by default. "all" = every symbol
    # in the universe, "etf" = SPY/QQQ only. The expensive layers default to
    # etf; --symbols overrides at the command line.
    scope: str = "all"
    # Which PROBED layer's measured history floor applies here. Derived layers
    # (a 5m resample, a 0DTE slice) are not probed individually - they inherit
    # from the layer whose endpoint+interval class they share. Without this a
    # derived layer has no start date and gets silently skipped.
    floor_layer: str | None = None
    # Options only: pull the whole chain with expiration=*, optionally capped by
    # days-to-expiry. max_dte=0 is the 0DTE chain, which is the point of the
    # exercise. Note /option/history/ohlc REJECTS expiration=*, so any layer
    # using it must set wildcard_expiration=False and be driven per-expiration.
    wildcard_expiration: bool = False
    max_dte: int | None = None
    strike_range: int | None = None
    # Approximate compressed bytes per request, measured on SPY. Used only by
    # the cost estimator, so a rough number is fine - but it must be measured
    # rather than guessed, or the estimate is worse than no estimate.
    est_bytes_per_request: int = 100_000
    notes: str = ""

    @property
    def path(self) -> str:
        return f"/{self.security}/history/{self.endpoint}"

    @property
    def dir_name(self) -> str:
        return self.name.replace(".", "_")


LAYERS: dict[str, Layer] = {}


def _reg(layer: Layer):
    LAYERS[layer.name] = layer
    return layer


# ---------------------------------------------------------------- tier 1: EOD
# Free-tier endpoints, one request per symbol-month, and they carry the daily
# settle plus the closing NBBO. Cheapest useful data in the whole archive.
_reg(Layer("stock.eod", "stock", "eod", None, "monthly", tier=1, scope="all",
           est_bytes_per_request=3_000,
           notes="daily OHLCV + closing bid/ask. Free tier, whole universe."))

# expiration=* here means one request returns EVERY contract's EOD for the
# month - the full option surface history. Free tier. Extremely high value per
# byte and the only affordable way to get whole-surface option history.
_reg(Layer("option.eod", "option", "eod", None, "monthly", tier=1, scope="etf",
           wildcard_expiration=True, est_bytes_per_request=4_000_000,
           notes="whole-chain daily option settle, all expirations. Free tier."))

_reg(Layer("option.open_interest", "option", "open_interest", None, "monthly",
           tier=1, scope="etf", wildcard_expiration=True,
           est_bytes_per_request=1_500_000,
           notes="OI per contract per day, all expirations."))

# ------------------------------------------------------------- tier 2: minute
_reg(Layer("stock.ohlc.1m", "stock", "ohlc", "1m", "monthly", tier=2, scope="all",
           est_bytes_per_request=400_000,
           notes="391 bars/session incl. vwap and trade count. The workhorse."))

_reg(Layer("stock.quote.1m", "stock", "quote", "1m", "monthly", tier=2, scope="all",
           est_bytes_per_request=380_000,
           notes="NBBO sampled at 1m - spread history, needed for fill realism."))

_reg(Layer("stock.ohlc.5m", "stock", "ohlc", "5m", "monthly", tier=2, scope="all",
           floor_layer="stock.ohlc.1m", est_bytes_per_request=90_000, notes="convenience resample."))

# ------------------------------------------------------------- tier 3: second
# One request per symbol per session, forced by the sub-minute rule. These are
# the layers that satisfy the per-second monitoring requirement.
_reg(Layer("stock.quote.1s", "stock", "quote", "1s", "daily", tier=3, scope="all",
           est_bytes_per_request=250_000,
           notes="23400 rows/session. Sample-and-hold: last quote at or before "
                 "each second, so gaps mean 'unchanged', not 'missing'."))

_reg(Layer("stock.ohlc.1s", "stock", "ohlc", "1s", "daily", tier=3, scope="all",
           est_bytes_per_request=220_000,
           notes="per-second traded bars; pairs with quote.1s for breakout work."))

_reg(Layer("stock.ohlc.100ms", "stock", "ohlc", "100ms", "daily", tier=4,
           scope="etf", floor_layer="stock.ohlc.1s",
           est_bytes_per_request=1_800_000,
           notes="sub-second bars, ETFs only - 10x the rows of 1s."))

# --------------------------------------------------------------- tier 4: tick
# trade_quote is preferred over trade: it attaches the prevailing NBBO to each
# print, which is exactly what deciding whether a breakout print was buyer- or
# seller-initiated requires. Getting the same thing by joining separate trade
# and quote pulls is both slower and easy to misalign.
_reg(Layer("stock.trade_quote.tick", "stock", "trade_quote", None, "daily",
           tier=4, scope="etf", floor_layer="stock.trade.tick",
           est_bytes_per_request=5500000,   # measured: 99 SPY sessions
           notes="every print with its prevailing NBBO. Largest layer by far."))

_reg(Layer("stock.trade.tick", "stock", "trade", None, "daily", tier=4,
           scope="etf", est_bytes_per_request=12_000_000,
           notes="prints only; subset of trade_quote, kept for cross-checking."))

_reg(Layer("stock.quote.tick", "stock", "quote", "tick", "daily", tier=4,
           scope="etf", est_bytes_per_request=30_000_000,
           notes="every NBBO change."))

# ------------------------------------------------------------- tier 5: options
# Option quote accepts expiration=* AND has an interval, so one request gets the
# whole chain at 1m for a date. max_dte caps how far out, which is the lever
# that keeps this affordable: max_dte=0 is 0DTE only.
_reg(Layer("option.quote.1m.0dte", "option", "quote", "1m", "daily", tier=5,
           scope="etf", floor_layer="option.quote.1m",
           wildcard_expiration=True, max_dte=0,
           est_bytes_per_request=6_000_000,
           notes="whole 0DTE chain at 1m. The core options dataset."))

_reg(Layer("option.quote.1m.week", "option", "quote", "1m", "daily", tier=5,
           scope="etf", floor_layer="option.quote.1m",
           wildcard_expiration=True, max_dte=7,
           est_bytes_per_request=20_000_000,
           notes="all expirations within a week - covers the weekly cycle."))

_reg(Layer("option.quote.1s.0dte", "option", "quote", "1s", "daily", tier=6,
           scope="etf", floor_layer="option.quote.1s",
           wildcard_expiration=True, max_dte=0, strike_range=15,
           est_bytes_per_request=60_000_000,
           notes="per-second 0DTE chain, +/-15 strikes. Value tier DOES serve "
                 "this - only /option/history/trade needs Standard."))

# Deliberately NOT registered, with reasons, so nobody re-adds them by mistake:
#   option.trade / option.trade_quote / option greeks / index.*  -> 403 on this
#   account (Options=VALUE, Index=FREE). See reference/entitlements.json.
#   option.ohlc.*  -> the endpoint rejects expiration=*, so a whole-chain pull
#   would need one request per contract per day: ~10^6 requests for SPY alone.
#   The quote layers already give bid/ask, which is what a fill needs.

UNAVAILABLE_BY_TIER = {
    "option.trade.tick": "Options=VALUE; /option/history/trade needs STANDARD",
    "option.trade_quote.tick": "needs STANDARD",
    "option.greeks.*": "needs STANDARD (1st order) / PRO (2nd+)",
    "index.*": "Index=FREE; intraday index needs STANDARD",
    "*.flat_file": "needs PROFESSIONAL",
}


def by_tier(max_tier: int, min_tier: int = 0) -> list[Layer]:
    return [l for l in LAYERS.values() if min_tier <= l.tier <= max_tier]


def get(name: str) -> Layer:
    if name not in LAYERS:
        raise KeyError(f"unknown layer {name!r}. Known: {', '.join(sorted(LAYERS))}")
    return LAYERS[name]
