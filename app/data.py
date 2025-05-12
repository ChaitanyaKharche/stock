import os
import datetime as dt
import yfinance as yf
import httpx
import asyncio
import asyncpraw
import pandas as pd
from . import cache

from dotenv import load_dotenv 
dotenv_path_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
if os.path.exists(dotenv_path_data): load_dotenv(dotenv_path=dotenv_path_data)
else: load_dotenv()

FINN = os.getenv("FINNHUB_API_KEY")
# Ensure Reddit env vars are checked here if _r() is defined in this file
REDDIT_CLIENT_ID_DATA = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET_DATA = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT_DATA = os.getenv("REDDIT_USER_AGENT", f"script:GapPredictor:v0.1 (by /u/YourRedditUsername)")


print(f"DEBUG data.py: FINNHUB_API_KEY at data.py import: {'SET' if FINN else 'NOT SET'}")
print(f"DEBUG data.py: REDDIT_CLIENT_ID at data.py import: {'SET' if REDDIT_CLIENT_ID_DATA else 'NOT SET'}")


async def fetch_stock(sym: str, days: int = 45) -> pd.DataFrame:
    # ... (your existing fetch_stock function - assumed to be working)
    def _dl():
        try:
            data = yf.download(
                sym, period=f"{days}d", progress=False, auto_adjust=True, group_by=None
            )
            if isinstance(data, pd.DataFrame) and not data.empty:
                return data
            raise ValueError("Empty or invalid data returned from yfinance")
        except Exception as e:
            raise ValueError(f"Failed to download data for {sym}: {str(e)}")
    try:
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(None, _dl)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols: raise ValueError(f"Missing columns: {missing_cols}")
        df = df[expected_cols].copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        raise ValueError(f"Error processing stock data for {sym}: {e}")


async def fetch_news(sym: str, client: httpx.AsyncClient, days: int = 3) -> tuple[list, bool]:
    """Fetches news from Finnhub for a given symbol and period."""
    cache_key_name = f"news:{sym}:{days}" # Use a consistent variable name for the cache key
    js_from_cache = cache.get(cache_key_name)
    if js_from_cache is not None: # Check if not None, as an empty list from cache is valid
        print(f"DEBUG data.py fetch_news: Returning CACHED news for {sym} ({days} days). Count: {len(js_from_cache)}")
        return js_from_cache, True

    if not FINN:
        print("ERROR data.py fetch_news: FINNHUB_API_KEY is not set. Cannot fetch news.")
        return [], False

    today = dt.date.today()
    start_date = today - dt.timedelta(days=days)
    url = f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={start_date}&to={today}&token={FINN}"
    print(f"DEBUG data.py fetch_news: Fetching URL: {url}")

    try:
        response = await client.get(url, timeout=20) # Increased timeout slightly
        raw_response_text = response.text # Get raw text for debugging
        print(f"DEBUG data.py fetch_news: Raw response status for {sym}: {response.status_code}")
        # print(f"DEBUG data.py fetch_news: Raw response text for {sym} (first 500 chars): {raw_response_text[:500]}")

        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        fetched_data = response.json()

        if not isinstance(fetched_data, list):
            print(f"WARNING data.py fetch_news: Expected a list from Finnhub for {sym}, got {type(fetched_data)}.")
            print(f"WARNING data.py fetch_news: Content received: {str(fetched_data)[:500]}")
            # Handle common error structure from Finnhub if it's a dict with an 'error' key
            if isinstance(fetched_data, dict) and fetched_data.get('error'):
                print(f"ERROR data.py fetch_news: Finnhub API error for {sym}: {fetched_data.get('error')}")
            return [], False # Return empty list if not a list or if error detected

        print(f"DEBUG data.py fetch_news: Fetched {len(fetched_data)} news items for {sym} from API.")
        # Print a sample of headlines if data is fetched
        if fetched_data:
            for i, item in enumerate(fetched_data[:3]): # Print first 3 item headlines
                 print(f"  Sample News {i+1} for {sym}: ID={item.get('id')}, Headline='{item.get('headline', 'N/A')[:100]}...'")
        
        cache.put(cache_key_name, fetched_data)
        return fetched_data, False
    except httpx.HTTPStatusError as exc:
        print(f"ERROR data.py fetch_news: HTTP error for {sym}: {exc.response.status_code} - {exc.response.text}")
        return [], False
    except httpx.RequestError as exc:
        print(f"ERROR data.py fetch_news: Request error for {sym}: {exc}")
        return [], False
    except Exception as e:
        print(f"ERROR data.py fetch_news: General error for {sym}: {e}")
        import traceback
        traceback.print_exc()
        return [], False

_reddit_instance = None # Renamed for clarity
def get_reddit_instance(): # Changed to a getter function
    """Initializes and returns the PRAW Reddit instance."""
    global _reddit_instance
    if _reddit_instance:
        return _reddit_instance

    print("DEBUG data.py get_reddit_instance: Attempting to initialize new PRAW Reddit instance.")
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT") # Removed default here to rely on .env

    if not all([client_id, client_secret, user_agent]):
        print("ERROR data.py get_reddit_instance: Reddit API credentials (ID, Secret, or User Agent) missing from environment.")
        # Print which one is missing for more specific debug
        if not client_id: print("  - REDDIT_CLIENT_ID is missing or empty.")
        if not client_secret: print("  - REDDIT_CLIENT_SECRET is missing or empty.")
        if not user_agent: print("  - REDDIT_USER_AGENT is missing or empty.")
        return None

    try:
        _reddit_instance = asyncpraw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
            # check_for_async=False # Add if you encounter specific async context issues with praw
        )
        print("DEBUG data.py get_reddit_instance: PRAW Reddit instance created successfully.")
    except Exception as e:
        print(f"ERROR data.py get_reddit_instance: Failed to create PRAW Reddit instance: {e}")
        _reddit_instance = None
    return _reddit_instance

async def fetch_reddit(sym: str, subs=("stocks", "wallstreetbets"), limit: int = 25) -> tuple[list, bool]:
    cache_key_name = f"reddit:{sym}:{limit}"
    js_from_cache = cache.get(cache_key_name)
    if js_from_cache is not None:
        print(f"DEBUG data.py fetch_reddit: Returning CACHED Reddit posts for {sym}. Count: {len(js_from_cache)}")
        return js_from_cache, True

    reddit = get_reddit_instance() # Call the getter
    if not reddit:
        print(f"ERROR data.py fetch_reddit: Reddit instance not available for {sym}.")
        return [], False

    posts = []
    query = f'"{sym}"' # Search for the ticker itself
    print(f"DEBUG data.py fetch_reddit: Searching Reddit for query: {query} in subreddits: {subs}")

    for s_name in subs:
        try:
            subreddit = await reddit.subreddit(s_name)
            # Using search, sort by new, time_filter for relevance
            async for submission in subreddit.search(query, sort="new", time_filter="day", limit=limit//len(subs) if len(subs) > 0 else limit):
                # Check if symbol is in title or selftext (more robustly)
                title_lower = submission.title.lower() if hasattr(submission, 'title') and submission.title else ""
                selftext_lower = submission.selftext.lower() if hasattr(submission, 'selftext') and submission.selftext else ""
                sym_lower = sym.lower()
                
                if sym_lower in title_lower or sym_lower in selftext_lower:
                    posts.append({
                        "datetime": dt.datetime.fromtimestamp(submission.created_utc, tz=dt.timezone.utc).isoformat(),
                        "title": submission.title,
                        "subreddit": s_name,
                        "score": int(submission.score),
                        "num_comments": int(submission.num_comments),
                        "url": f"https://www.reddit.com{submission.permalink}" # Added permalink
                    })
        except Exception as e:
            print(f"ERROR data.py fetch_reddit: Error searching subreddit '{s_name}' for {sym}: {e}")
            # Consider logging traceback for deeper PRAW issues
            # import traceback
            # traceback.print_exc()
            continue
    
    print(f"DEBUG data.py fetch_reddit: Found {len(posts)} relevant Reddit posts for {sym} from API.")
    cache.put(cache_key_name, posts)
    return posts, False