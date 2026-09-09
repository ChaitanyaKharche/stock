import asyncio
import datetime as dt
import pandas as pd
import numpy as np
import httpx
import yfinance as yf
from datetime import datetime, timedelta
from . import cache, config
import asyncpraw
import json
import os
from pathlib import Path
import pytz

# Multi-timeframe spec. Keys MUST stay "15m"/"hourly"/"daily" -- momentum_trading_engine
# .generate_enhanced_signal() looks up exactly those three, and enhanced_api hands
# ohlcv_data["daily"] to the TFT. yfinance caps intraday history: 15m is 60 days, 1h is
# 730 days, so those periods are the provider's limit, not a choice.
# The UI offers 15m / 1h / 4h / 1d. 1m and 5m were REMOVED on 2026-09-07: yfinance caps
# 1m history at 7 days, far too few bars for the 20-period indicators to warm up, and with
# the fast settings gone the 5m set no longer earned its network call. Offering a horizon
# the data cannot support is the same class of error as a dropdown that does nothing.
#
# 4h is NOT fetched -- yfinance has no 4h interval. It is aggregated from the hourly bars
# by fold_to_4h() below, which groups WITHIN a session so a bucket never straddles the
# overnight break.
TIMEFRAME_SPEC = {
    "15m":    ("60d",  "15m"),
    "hourly": ("180d", "1h"),
    "daily":  ("2y",   "1d"),
}

_AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def fold_to_4h(hourly):
    """Aggregate hourly bars into 4-hour bars, one session at a time.

    A plain `resample("4h")` buckets by wall clock, so a bucket would span the close and
    the next morning's open -- an overnight gap silently folded into an intraday range.
    US regular hours are 6.5 hours, so each session yields two buckets (4 bars + the
    remainder), stamped at the first bar in the bucket.
    """
    if hourly is None or hourly.empty:
        return hourly
    out = []
    for _, day in hourly.groupby(hourly.index.date):
        for k in range(0, len(day), 4):
            chunk = day.iloc[k:k + 4]
            if chunk.empty:
                continue
            row = {c: (chunk[c].iloc[0] if f == "first" else
                       chunk[c].iloc[-1] if f == "last" else
                       chunk[c].max() if f == "max" else
                       chunk[c].min() if f == "min" else
                       chunk[c].sum()) for c, f in _AGG.items()}
            out.append((chunk.index[0], row))
    if not out:
        return hourly.iloc[0:0]
    import pandas as _pd
    return _pd.DataFrame([r for _, r in out], index=[i for i, _ in out])[list(_AGG)]


OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# This part you added is kept, as it's a good fallback
LOCAL_DATA_DIR = Path(os.getenv(
    "TRADE_LOCAL_DATA",
    "./local_data" # Simplified default for portability
))
LOCAL_DATA_DIR.mkdir(exist_ok=True)


def _process_finnhub_data(data: dict, symbol: str) -> pd.DataFrame:
    """Processes JSON from Finnhub into a clean DataFrame."""
    if not data or data.get('s') != 'ok' or 'c' not in data:
        print(f"No valid data received from Finnhub for {symbol}.")
        return pd.DataFrame()

    df = pd.DataFrame({
        'Open': data['o'],
        'High': data['h'],
        'Low': data['l'],
        'Close': data['c'],
        'Volume': data['v']
    })
    # Finnhub timestamps are UNIX timestamps
    df.index = pd.to_datetime(data['t'], unit='s', utc=True)
    df.dropna(inplace=True)
    return df


class UnifiedDataProvider:
    def __init__(self):
        # Create a single, reusable client for all API calls
        self.client = httpx.AsyncClient(timeout=20.0)
        self.reddit_instance = None
        self.local_data_cache = {}
        print("✅ UnifiedDataProvider initialized with Finnhub.")

    def _load_local_data(self, symbol: str) -> dict | None:
        """Loads symbol JSON from local files if present."""
        path = LOCAL_DATA_DIR / f"{symbol.upper()}_external_data.json"
        if path.exists():
            print(f"Loading local data from: {path}")
            with open(path, "r") as f:
                data = json.load(f)
            self.local_data_cache[symbol.upper()] = data
            return data
        return None

    
    @staticmethod
    def _yf_bars(symbol: str, period: str, interval: str) -> pd.DataFrame:
        """One timeframe of real OHLCV from yfinance. Blocking; call via executor."""
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval,
                                           auto_adjust=False)
        except Exception as e:                                    # noqa: BLE001
            print(f"  - yfinance {interval} failed for {symbol}: {e}")
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        # yfinance can hand back a MultiIndex if the ticker resolves oddly; flatten it
        # rather than let a tuple-keyed column silently produce an empty selection.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        missing = [c for c in OHLCV_COLS if c not in df.columns]
        if missing:
            print(f"  - yfinance {interval} for {symbol} missing {missing}")
            return pd.DataFrame()
        return df[OHLCV_COLS].dropna()

    async def _finnhub_quote_frame(self, symbol: str) -> pd.DataFrame:
        """LAST-RESORT single-row frame. Kept only as a fallback -- see the note in
        fetch_multi_timeframe_stock_data about why it must never be the primary."""
        try:
            res = await self.client.get("https://finnhub.io/api/v1/quote",
                                        params={"symbol": symbol,
                                                "token": config.FINNHUB_KEY})
            res.raise_for_status()
            d = res.json()
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            print(f"ERROR: Finnhub quote failed for {symbol}: {e}")
            return pd.DataFrame()
        if not d or d.get("c") in (None, 0):
            return pd.DataFrame()
        df = pd.DataFrame([{"Open": d["o"], "High": d["h"], "Low": d["l"],
                            "Close": d["c"], "Volume": 0}])
        df.index = pd.to_datetime([datetime.now(tz=pytz.UTC)])
        return df

    async def fetch_multi_timeframe_stock_data(self, symbol: str) -> dict:
        """Real 15m / hourly / daily OHLCV history.

        THIS FUNCTION WAS THE ENTIRE BUG. It previously called Finnhub's /quote endpoint
        -- "historical candle data is a premium feature" -- and returned a ONE-ROW frame
        with Volume hardcoded to 0. Every downstream consumer needs a series:

          * indicators.identify_current_setup() opens with `if df.empty or len(df) < 2`
            and returns {"adx": 0, "rsi": 50, "error": "Insufficient data"}. One row is
            always < 2, so that branch fired 100% of the time, for every symbol.
          * the momentum engine then reported confidence 0, and enhanced_api's
            weighted_confidence = 0*0.4 + 50*0.3 + 0*0.3 = exactly 15 -- which is why the
            app returned "HOLD, 15%" for every ticker anyone typed. It was a constant, not
            a prediction.
          * the TFT needs >= 96 daily rows and never once got them.

        yfinance was already in requirements.txt and already imported by agent.py,
        deploy.py and live_signals.py -- just not here, in the one place that decided
        every signal. It serves the intraday history Finnhub's free tier withholds, with
        no API key.

        Timeframes are fetched concurrently in threads because yfinance is blocking and
        this is an async endpoint; three sequential network calls would stall the loop.
        A timeframe that comes back empty is OMITTED rather than filled with a stub, so
        the "Insufficient data" path still means what it says.
        """
        print(f"Fetching multi-timeframe OHLCV for {symbol} (yfinance)...")
        loop = asyncio.get_event_loop()
        names = list(TIMEFRAME_SPEC)
        results = await asyncio.gather(*[
            loop.run_in_executor(None, self._yf_bars, symbol, *TIMEFRAME_SPEC[tf])
            for tf in names
        ])

        dfs = {}
        for tf, df in zip(names, results):
            if not df.empty:
                dfs[tf] = df
                print(f"  - {tf}: {len(df)} bars")

        # 4h is derived, not fetched -- yfinance has no 4h interval.
        if "hourly" in dfs:
            four = fold_to_4h(dfs["hourly"])
            if four is not None and not four.empty:
                dfs["4h"] = four
                print(f"  - 4h: {len(four)} bars (folded from hourly)")

        if not dfs:
            # Every timeframe failed: bad ticker, or yfinance unreachable. Fall back to
            # the live quote so the app degrades to its old behaviour instead of 500ing,
            # but say so loudly -- a single row cannot produce a real signal.
            print(f"  - NO history for {symbol}; falling back to a single live quote. "
                  f"Indicators will report 'Insufficient data'.")
            quote = await self._finnhub_quote_frame(symbol)
            if not quote.empty:
                dfs["daily"] = quote
        return dfs

    async def fetch_news(self, symbol: str, days: int = 3) -> tuple:
        """Fetches news from Finnhub, with local file fallback."""
        local_data = self._load_local_data(symbol)
        if local_data and 'news_data' in local_data:
            return local_data['news_data'], "local_file"

        cache_key = f"news_{symbol}_{days}"
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data, "cache"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        url = f"https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": symbol,
            "from": start_date.strftime('%Y-%m-%d'),
            "to": end_date.strftime('%Y-%m-%d'),
            "token": config.FINNHUB_KEY
        }
        
        try:
            res = await self.client.get(url, params=params)
            res.raise_for_status()
            data = res.json()
            cache.put(cache_key, data)
            return data, "api"
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            print(f"ERROR: Finnhub news request failed for {symbol}: {e}")
            return [], "error"

    def _get_reddit_instance(self):
        """Initializes the asyncpraw Reddit instance."""
        if self.reddit_instance is None:
            self.reddit_instance = asyncpraw.Reddit(
                client_id=config.REDDIT_CLIENT_ID,
                client_secret=config.REDDIT_CLIENT_SECRET,
                user_agent=config.REDDIT_USER_AGENT
            )
        return self.reddit_instance

    async def fetch_reddit_data(self, symbol: str, limit: int = 25) -> tuple:
        """Fetches Reddit data, with local file fallback."""
        local_data = self._load_local_data(symbol)
        if local_data and 'reddit_data' in local_data:
            return local_data['reddit_data'], "local_file"

        cache_key = f"reddit_{symbol}_{limit}"
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data, "cache"
        
        reddit = self._get_reddit_instance()
        submissions_data = []
        query = f'"{symbol}" OR "${symbol}"'
        subreddits = ["stocks", "wallstreetbets", "options"]
        
        for sub_name in subreddits:
            try:
                subreddit = await reddit.subreddit(sub_name)
                async for submission in subreddit.search(query, limit=limit, sort='new'):
                    submissions_data.append({
                        'title': submission.title, 'score': submission.score, 
                        'url': submission.url, 'created_utc': submission.created_utc, 
                        'subreddit': sub_name
                    })
            except Exception as e:
                print(f"ERROR: Reddit fetch for {symbol} in {sub_name} failed: {e}")
                
        cache.put(cache_key, submissions_data)
        return submissions_data, "api"
        
    def get_alternative_data(self, symbol: str) -> dict:
        """VIX, sector, and options positioning.

        put_call_ratio and iv_rank used to be the literals 0.85 and 45.5 with a comment
        saying they "require an options data provider". They were rendered in the app's
        JSON output as if measured, which is the kind of detail that costs credibility
        with anyone who reads it closely. yfinance exposes a free option chain, so both
        are now computed from real open interest and real implied vols.

        `iv_rank` is also GONE as a name, because it was wrong against its own consumer.
        IV rank has a precise meaning -- where current IV sits in its own 52-week range,
        0-100 -- but the only reader of this field was:

            implied_vol = alternative_data.get('iv_rank', 50) / 100.0

        which divides by 100 and uses the result as a volatility. That only type-checks
        if the value is an IV PERCENTAGE, not a rank; feeding a genuine rank into it would
        be a category error that happens to produce a plausible number. The field is now
        `implied_vol_pct`, which is what the consumer always wanted and what the chain
        actually provides. A true rank needs an IV history this data tier has no access
        to, so it is not offered at all rather than approximated under its own name.

        VIX likewise: Finnhub's free tier does not serve ^VIX, so that call silently
        failed and left the 20.0 default in place on every single request. yfinance
        serves it.

        Anything that genuinely cannot be sourced is reported as None and flagged in
        `data_quality`, never as a plausible-looking number.
        """
        out = {"vix_level": None, "sector": "Unknown",
               "put_call_ratio": None, "implied_vol_pct": None,
               "data_quality": {}}

        try:
            vix = yf.Ticker("^VIX").history(period="5d", interval="1d")
            if not vix.empty:
                out["vix_level"] = round(float(vix["Close"].iloc[-1]), 2)
        except Exception as e:                                    # noqa: BLE001
            out["data_quality"]["vix"] = f"unavailable: {e}"

        try:
            with httpx.Client(timeout=10.0) as sync_client:
                r = sync_client.get(
                    "https://finnhub.io/api/v1/stock/profile2",
                    params={"symbol": symbol, "token": config.FINNHUB_KEY})
                if r.status_code == 200:
                    out["sector"] = r.json().get("finnhubIndustry", "Unknown")
        except Exception as e:                                    # noqa: BLE001
            out["data_quality"]["sector"] = f"unavailable: {e}"

        try:
            tk = yf.Ticker(symbol)
            expirations = tk.options or []
            if expirations:
                chain = tk.option_chain(expirations[0])
                calls, puts = chain.calls, chain.puts
                call_oi = float(calls["openInterest"].fillna(0).sum())
                put_oi = float(puts["openInterest"].fillna(0).sum())
                if call_oi > 0:
                    out["put_call_ratio"] = round(put_oi / call_oi, 3)
                ivs = pd.concat([calls["impliedVolatility"],
                                 puts["impliedVolatility"]]).dropna()
                ivs = ivs[(ivs > 0) & (ivs < 5)]
                if len(ivs):
                    # Kept as a diagnostic only. The MEDIAN ACROSS THE WHOLE CHAIN is not
                    # a usable volatility: it averages deep out-of-the-money strikes whose
                    # yfinance implied vols are stale, zero-bid or simply wrong, so it
                    # swings wildly with whatever strikes the provider happens to return.
                    # Observed on NVDA within one session: 90.63 and then 12.5.
                    out["implied_vol_chain_median_pct"] = round(
                        float(ivs.median()) * 100, 2)

                # ATM implied vol, BACKED OUT OF THE TRADED STRADDLE PRICE.
                #
                # yfinance's own `impliedVolatility` field is unusable here: at the ATM
                # strike it returns the placeholder 0.00001, with bid and ask both 0.0 and
                # openInterest 0, on a strike that traded 38,104 contracts. Only lastPrice
                # and volume are real. Taking a chain median of that field produced
                # "implied vol" of 90.63 and then 12.5 for NVDA inside one session, and an
                # ATM read of 0.78% -- numbers that are not volatilities.
                #
                # So the call and put nearest spot are combined into a straddle and
                # inverted with Brenner-Subrahmanyam: straddle ~= 0.7979 * S * sigma *
                # sqrt(T). Exact at the money and closed-form, so it cannot fail to
                # converge on a thin quote.
                #
                # CAVEAT, carried into the response: lastPrice is a TRADE price and may be
                # stale, and the free chain exposes no bid/ask, so nothing here can measure
                # the spread. The 0DTE cost-model study needed real bid/ask and used the
                # ThetaData archive for exactly that reason.
                spot = None
                _h = tk.history(period="2d", interval="1d")
                if _h is not None and not _h.empty:
                    spot = float(_h["Close"].iloc[-1])
                if spot and spot > 0:
                    exp_date = dt.date.fromisoformat(expirations[0])
                    dte = (exp_date - dt.date.today()).days
                    # Trading time, floored at half a session so a 0DTE cannot divide by
                    # zero. Matches the 252-day annualisation used by the HAR forecast.
                    T = max(dte * (252.0 / 365.0), 0.5) / 252.0
                    legs = []
                    for side in (calls, puts):
                        sub = side.dropna(subset=["strike", "lastPrice"])
                        sub = sub[sub["lastPrice"] > 0]
                        if sub.empty:
                            continue
                        k = (sub["strike"] - spot).abs().idxmin()
                        legs.append(float(sub.loc[k, "lastPrice"]))
                    if len(legs) == 2:
                        iv = sum(legs) / (0.7978845608 * spot * np.sqrt(T))
                        if 0.01 < iv < 5.0:
                            out["implied_vol_pct"] = round(iv * 100, 2)
                            out["data_quality"]["iv_source"] = (
                                f"ATM straddle lastPrice, {dte}DTE, "
                                "Brenner-Subrahmanyam; no bid/ask on the free chain")
                out["data_quality"]["options_expiration_used"] = expirations[0]
        except Exception as e:                                    # noqa: BLE001
            out["data_quality"]["options"] = f"unavailable: {e}"

        return out

    async def close(self):
        """Closes all persistent connections."""
        if self.reddit_instance:
            await self.reddit_instance.close()
            print("Reddit instance closed.")
        await self.client.aclose()
        print("HTTPX client closed.")