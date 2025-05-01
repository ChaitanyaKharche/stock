import os, datetime as dt, yfinance as yf, httpx, asyncio, asyncpraw, pandas as pd
from . import cache
FINN = os.getenv("FINNHUB_API_KEY")

async def fetch_stock(sym, days=45):
    loop = asyncio.get_running_loop()
    df = await loop.run_in_executor(None, lambda: yf.download(sym, period=f"{days}d", progress=False))
    df.index = pd.to_datetime(df.index).date; return df

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
    key=f"reddit:{sym}:{limit}"; js=cache.get(key)
    if js: return js,True
    posts=[]; q=f"\"{sym}\""
    for s in subs:
        subreddit=await _r().subreddit(s)
        async for p in subreddit.search(q, limit=limit, sort="new", time_filter="day"):
            if sym.lower() in p.title.lower() or sym.lower() in p.selftext.lower():
                posts.append({"datetime":dt.datetime.fromtimestamp(p.created_utc).isoformat(),
                              "title":p.title,"subreddit":s,"score":p.score,"num_comments":p.num_comments})
    cache.put(key,posts); return posts,False
