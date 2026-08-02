# trade_analysis/data_sources/unified_data_provider.py
import yfinance as yf
import pandas as pd
import numpy as np
import httpx
from datetime import datetime, timedelta
from ..utils import cache
from .. import config
import asyncpraw
import json # --- NEW ---
import os   # --- NEW ---
from pathlib import Path

from ..paths import LOCAL_DATA_DIR as _DEFAULT_LOCAL_DATA_DIR

LOCAL_DATA_DIR = Path(os.getenv("TRADE_LOCAL_DATA", str(_DEFAULT_LOCAL_DATA_DIR)))

# _process_yf_data function remains the same (omitted for brevity)
def _process_yf_data(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [col.capitalize() for col in df.columns]
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            df[col] = np.nan
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    return df

class UnifiedDataProvider:
    def __init__(self):
        self.yf_ticker_cache = {}
        self.reddit_instance = None
        self.local_data_cache = {} # --- NEW ---
        print("✅ UnifiedDataProvider initialized.")

    # --- NEW HELPER FUNCTION ---
    def _load_local_data(self, symbol: str) -> dict | None:
        """load symbol JSON from LOCAL_DATA_DIR if present"""
        sym = symbol.upper()
        if sym in self.local_data_cache:
            return self.local_data_cache[sym]
    
        path = LOCAL_DATA_DIR / f"{sym}_external_data.json"
        if path.exists():
            print(f"Loading local data: {path}")
            with open(path, "r") as f:
                data = json.load(f)
            self.local_data_cache[sym] = data
            return data
        return None

    # fetch_multi_timeframe_stock_data remains the same (omitted for brevity)
    async def fetch_multi_timeframe_stock_data(self, symbol: str) -> dict:
        print(f"Fetching multi-timeframe OHLCV for {symbol}...")
        dfs = {}
        timeframes = {"daily": ("180d", "1d"), "hourly": ("730d", "60m"), "15m": ("60d", "15m")}
        for name, (period, interval) in timeframes.items():
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
            processed_df = _process_yf_data(df, symbol, name)
            if not processed_df.empty:
                print(f"  - Fetched and processed {len(processed_df)} points for {name} interval.")
                dfs[name] = processed_df
            else:
                dfs[name] = pd.DataFrame()
        return dfs

    # --- MODIFIED ---
    async def fetch_news(self, symbol: str, client: httpx.AsyncClient, days: int = 3) -> tuple:
        # 1. Check for local file first
        local_data = self._load_local_data(symbol)
        if local_data and 'news_data' in local_data:
            return local_data['news_data'], "local_file"

        # 2. Fallback to cache and API call
        cache_key = f"news_{symbol}_{days}"
        cached_data = cache.get(cache_key)
        if cached_data: return cached_data, "cache"
        end_date = datetime.now(); start_date = end_date - timedelta(days=days)
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&token={config.FINNHUB_KEY}"
        try:
            res = await client.get(url, timeout=10.0)
            res.raise_for_status(); data = res.json()
            cache.put(cache_key, data); return data, "api"
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            print(f"ERROR: Finnhub request failed for {symbol}: {e}"); return [], "error"

    def _get_reddit_instance(self):
        if self.reddit_instance is None:
            self.reddit_instance = asyncpraw.Reddit(client_id=config.REDDIT_CLIENT_ID, client_secret=config.REDDIT_CLIENT_SECRET, user_agent=config.REDDIT_USER_AGENT)
        return self.reddit_instance

    # --- MODIFIED ---
    async def fetch_reddit_data(self, symbol: str, limit: int = 25) -> tuple:
        # 1. Check for local file first
        local_data = self._load_local_data(symbol)
        if local_data and 'reddit_data' in local_data:
            return local_data['reddit_data'], "local_file"
        
        # 2. Fallback to cache and API call (which will fail on HPC, but logic is kept)
        cache_key = f"reddit_{symbol}_{limit}"; cached_data = cache.get(cache_key)
        if cached_data: return cached_data, "cache"
        reddit = self._get_reddit_instance(); submissions_data = []
        query = f'"{symbol}" OR "${symbol}"'; subreddits = ["stocks", "wallstreetbets", "options"]
        for sub_name in subreddits:
            try:
                subreddit = await reddit.subreddit(sub_name)
                async for submission in subreddit.search(query, limit=limit, sort='new'):
                    submissions_data.append({'title': submission.title, 'score': submission.score, 'url': submission.url, 'created_utc': submission.created_utc, 'subreddit': sub_name})
            except Exception as e:
                print(f"ERROR: Reddit fetch for {symbol} in {sub_name} failed: {e}")
        cache.put(cache_key, submissions_data); return submissions_data, "api"

    async def close(self):
        """Closes any persistent connections."""
        if self.reddit_instance:
            await self.reddit_instance.close()
            self.reddit_instance = None
            print("Reddit instance closed.")
            
    # get_alternative_data remains the same (omitted for brevity)
    def get_alternative_data(self, symbol: str) -> dict:
        try:
            vix_data = yf.download('^VIX', period='10d', progress=False)
            last_vix = vix_data['Close'].iloc[-1].item() if not vix_data.empty else 20.0
            sector = yf.Ticker(symbol).info.get('sector', 'Unknown')
            return {"vix_level": round(last_vix, 2), "sector": sector, "put_call_ratio": 0.85, "iv_rank": 45.5}
        except Exception as e:
            print(f"ERROR fetching alternative data for {symbol}: {e}")
            return {"vix_level": 20.0, "sector": "Error", "put_call_ratio": 1.0, "iv_rank": 50.0}