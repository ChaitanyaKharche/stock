import os
import datetime as dt
import yfinance as yf
import httpx
import asyncio
import asyncpraw
import pandas as pd
from . import cache
import numpy as np

# Load .env variables at the top of the module
from dotenv import load_dotenv
dotenv_path_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
if os.path.exists(dotenv_path_data): load_dotenv(dotenv_path=dotenv_path_data)
else: load_dotenv() # Try default locations

FINN = os.getenv("FINNHUB_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID") # Use consistent naming
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

print(f"DEBUG data.py: FINNHUB_API_KEY at data.py import: {'SET' if FINN else 'NOT SET'}")
print(f"DEBUG data.py: REDDIT_CLIENT_ID at data.py import: {'SET' if REDDIT_CLIENT_ID else 'NOT SET'}")

def _process_yf_data(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """Helper to standardize yfinance DataFrame processing."""
    if df.empty:
        print(f"Warning: No data returned from yfinance for {symbol} at {interval} interval.")
        return pd.DataFrame() # Return empty DataFrame with expected columns later if needed

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1) # Flatten MultiIndex if present

    # Standardize column names to Capitalized
    df.columns = [col.capitalize() for col in df.columns]

    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Create missing columns with NaN if they don't exist after capitalization
    for col in expected_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' missing for {symbol} ({interval}). Initializing with NaN.")
            df[col] = np.nan

    df = df[expected_cols].copy() # Select and order

    for col_price in ['Open', 'High', 'Low', 'Close']:
        df[col_price] = pd.to_numeric(df[col_price], errors='coerce').astype(float)
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)

    df.index = pd.to_datetime(df.index)
    # For intraday data, yfinance might return timezone-aware index. Convert to UTC for consistency.
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC')
    else: # If naive, assume UTC (or decide on a consistent timezone, e.g., America/New_York)
        # This might need adjustment based on how yfinance returns daily vs intraday timezone info
        try:
            df.index = df.index.tz_localize('UTC') # Be careful with this if data is truly naive
        except TypeError: # Already timezone-aware (should have been caught by df.index.tz is not None)
            pass
            
    # Remove rows with NaN in critical price columns after conversion
    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    print(f"Processed {len(df)} data points for {symbol} at {interval} interval.")
    return df

async def fetch_multi_timeframe_stock_data(symbol: str) -> dict[str, pd.DataFrame]:
    """
    Fetches daily, hourly, and 15-minute stock data for a given symbol.
    Returns a dictionary with keys 'daily', 'hourly', '15m'.
    """
    print(f"Fetching multi-timeframe data for {symbol}...")
    loop = asyncio.get_running_loop()
    
    timeframe_params = {
        "daily": {"period": "180d", "interval": "1d", "days_for_yf": None}, # Longer period for daily context
        "hourly": {"period": "60d", "interval": "60m", "days_for_yf": 60}, # yf max for 60m often 730d, but 60d is plenty for recent
        "15m": {"period": "15d", "interval": "15m", "days_for_yf": 15}   # yf max for 15m often 60d, 15d for recent focus
    }
    # For intraday, yfinance 'period' is often overridden by 'start'/'end' or limited.
    # It's safer to use shorter periods for intraday or rely on yfinance's max for that interval.
    # Using 'period' for intraday with yf.download sometimes fetches less than expected for certain intervals.
    # A more robust way for intraday is to calculate start/end dates.
    # However, for simplicity with yf.download's direct period arg:
    # Max periods for intervals: 1m=7d, 2m,5m,15m,30m=60d, 60m,90m=730d

    data_frames = {}

    for tf, params in timeframe_params.items():
        print(f" Fetching {tf} data for {symbol} (period: {params['period']}, interval: {params['interval']})...")
        try:
            # Use yf.Ticker().history for more control over start/end for intraday if 'period' is problematic
            # For now, sticking to yf.download for consistency with your previous fetch_stock
            current_period = params['period']
            if tf != "daily" and params["days_for_yf"]: # For intraday, yf period is tricky
                 # Let's use start/end for intraday to be more explicit about data quantity
                 end_date = dt.datetime.now(dt.timezone.utc)
                 start_date = end_date - dt.timedelta(days=params["days_for_yf"])
                 df = await loop.run_in_executor(None, lambda: yf.download(
                     tickers=symbol, start=start_date, end=end_date, interval=params['interval'],
                     progress=False, auto_adjust=True, group_by='ticker'
                 ))
            else: # For daily
                 df = await loop.run_in_executor(None, lambda: yf.download(
                     tickers=symbol, period=current_period, interval=params['interval'],
                     progress=False, auto_adjust=True, group_by='ticker'
                 ))

            data_frames[tf] = _process_yf_data(df, symbol, tf)
        except Exception as e:
            print(f"Error fetching {tf} data for {symbol}: {e}")
            data_frames[tf] = pd.DataFrame() # Store empty DataFrame on error

    return data_frames

# Keep your existing fetch_news and fetch_reddit, they are fine.
# Modify the old fetch_stock or remove it if fetch_multi_timeframe_stock_data replaces its use case.
# For now, I'll assume the main API will switch to using fetch_multi_timeframe_stock_data.

async def fetch_news(sym: str, client: httpx.AsyncClient, days: int = 3) -> tuple[list, bool]:
    # ... (Your existing fetch_news function - it's good) ...
    cache_key_name = f"news:{sym}:{days}"
    js_from_cache = cache.get(cache_key_name)
    if js_from_cache is not None:
        print(f"DEBUG data.py fetch_news: Returning CACHED news for {sym} ({days} days). Count: {len(js_from_cache)}")
        return js_from_cache, True
    if not FINN:
        print("ERROR data.py fetch_news: FINNHUB_API_KEY is not set.")
        return [], False
    today = dt.date.today(); start_date = today - dt.timedelta(days=days)
    url = f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={start_date}&to={today}&token={FINN}"
    print(f"DEBUG data.py fetch_news: Fetching URL: {url}")
    try:
        response = await client.get(url, timeout=20)
        response.raise_for_status()
        fetched_data = response.json()
        if not isinstance(fetched_data, list): return [], False
        print(f"DEBUG data.py fetch_news: Fetched {len(fetched_data)} news items for {sym} from API.")
        if fetched_data:
            for i, item in enumerate(fetched_data[:2]): # Print first 2
                 print(f"  Sample News {i+1} for {sym}: Headline='{item.get('headline', 'N/A')[:70]}...'")
        cache.put(cache_key_name, fetched_data)
        return fetched_data, False
    except Exception as e:
        print(f"ERROR data.py fetch_news for {sym}: {e}")
        return [], False

_reddit_instance = None
def get_reddit_instance():
    global _reddit_instance
    if _reddit_instance: return _reddit_instance
    print("DEBUG data.py get_reddit_instance: Initializing PRAW.")
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")
    if not all([client_id, client_secret, user_agent]):
        print("ERROR data.py get_reddit_instance: Reddit API credentials missing.")
        return None
    try:
        _reddit_instance = asyncpraw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
        print("DEBUG data.py get_reddit_instance: PRAW instance created.")
    except Exception as e:
        print(f"ERROR data.py get_reddit_instance: {e}")
    return _reddit_instance

async def fetch_reddit(sym: str, subs=("stocks", "wallstreetbets","options"), limit: int = 15) -> tuple[list, bool]: # Reduced limit
    cache_key_name = f"reddit:{sym}:{limit}"
    js_from_cache = cache.get(cache_key_name)
    if js_from_cache is not None:
        print(f"DEBUG data.py fetch_reddit: CACHED Reddit posts for {sym}. Count: {len(js_from_cache)}")
        return js_from_cache, True
    reddit = get_reddit_instance()
    if not reddit: return [], False
    posts = []
    query = f'"{sym}" OR ${sym}' # Broader search
    print(f"DEBUG data.py fetch_reddit: Searching Reddit for: {query} in {subs}")
    try:
        for s_name in subs:
            subreddit = await reddit.subreddit(s_name)
            async for submission in subreddit.search(query, sort="new", time_filter="day", limit=(limit // len(subs) + 1)):
                posts.append({
                    "datetime": dt.datetime.fromtimestamp(submission.created_utc, tz=dt.timezone.utc).isoformat(),
                    "title": submission.title, "subreddit": s_name, "score": int(submission.score),
                    "num_comments": int(submission.num_comments), "url": f"https://www.reddit.com{submission.permalink}"
                })
        print(f"DEBUG data.py fetch_reddit: Found {len(posts)} Reddit posts for {sym} from API.")
        cache.put(cache_key_name, posts)
    except Exception as e:
        print(f"ERROR data.py fetch_reddit for {sym} in {s_name}: {e}")
    return posts, False
