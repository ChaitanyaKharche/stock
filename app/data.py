import os
import datetime as dt
import yfinance as yf
import httpx
import asyncio
import asyncpraw
import pandas as pd
from . import cache

FINN = os.getenv("FINNHUB_API_KEY")

async def fetch_stock(sym: str, days: int = 45) -> pd.DataFrame:
    def _dl():
        try:
            data = yf.download(
                sym,
                period=f"{days}d",
                progress=False,
                auto_adjust=True,
                group_by=None  # Using None works better
            )
            if isinstance(data, pd.DataFrame) and not data.empty:
                return data
            raise ValueError("Empty or invalid data returned from yfinance")
        except Exception as e:
            raise ValueError(f"Failed to download data for {sym}: {str(e)}")
    
    try:
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(None, _dl)
        
        # CRITICAL: Fix MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)
        
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']  # CAPITALIZED
        
        # Check for required columns
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Select and order columns consistently
        df = df[expected_cols].copy()
        
        # Convert types explicitly - Use capital column names
        for col in ['Open', 'High', 'Low', 'Close']:  # CAPITALIZED
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)
        
        df.index = pd.to_datetime(df.index)
        return df
    
    except Exception as e:
        raise ValueError(f"Error processing stock data for {sym}: {e}")

async def predict(symbol: str) -> dict:
    """Async prediction handler"""
    try:
        df = await fetch_stock(symbol)
        if df.empty:
            return {"error": "No data returned for symbol"}
            
        latest_price = float(df['Close'].iloc[-1])  # Note lowercase column name
        return {
            "prediction": "UP" if latest_price > 100 else "DOWN",
            "price": latest_price,
            "symbol": symbol,
            "timestamp": dt.datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}

async def fetch_news(sym, client, days=1):
    key=f"news:{sym}:{days}"; js=cache.get(key)
    if js: return js,True
    today=dt.date.today(); start=today-dt.timedelta(days=days)
    url=f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={start}&to={today}&token={FINN}"
    js=(await client.get(url,timeout=10)).json(); cache.put(key,js); return js,False

_reddit=None
def _r():
    global _reddit
    if _reddit: return _reddit
    _reddit=asyncpraw.Reddit(client_id=os.getenv("REDDIT_CLIENT_ID"),
                             client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                             user_agent=os.getenv("REDDIT_USER_AGENT","GapPredictor"))
    return _reddit

async def fetch_reddit(sym, subs=("stocks","wallstreetbets"), limit=25):
    key=f"reddit:{sym}:{limit}"
    try:
        js=cache.get(key)
        if js: 
            return js, True
        
        posts=[]
        q=f"\"{sym}\""
        for s in subs:
            try:
                subreddit=await _r().subreddit(s)
                async for p in subreddit.search(q, limit=limit, sort="new", time_filter="day"):
                    if sym.lower() in p.title.lower() or (hasattr(p, 'selftext') and sym.lower() in p.selftext.lower()):
                        posts.append({
                            "datetime": dt.datetime.fromtimestamp(p.created_utc).isoformat(),
                            "title": p.title,
                            "subreddit": s,
                            "score": int(p.score),  # Force integer conversion
                            "num_comments": int(p.num_comments)  # Force integer conversion
                        })
            except Exception as e:
                print(f"Reddit search error for {s}: {e}")
                continue
                
        cache.put(key, posts)
        return posts, False
    except Exception as e:
        print(f"fetch_reddit error: {e}")
        return [], False
