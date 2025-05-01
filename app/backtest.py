import sys, asyncio, numpy as np, yfinance as yf, pandas as pd, datetime as dt
from fastapi.testclient import TestClient
from . import api

def gaps(df): return np.sign(df["Open"].values[1:]-df["Close"].values[:-1])
async def main(sym):
    df=yf.download(sym,period="100d",progress=False); g=gaps(df)
    cli=TestClient(api.app); preds=[]
    for _ in g: preds.append({"UP":1,"DOWN":-1,"FLAT":0}[cli.post(f"/predict/?symbol={sym}").json()["predicted_direction"]])
    print(f"{sym}: {(g==np.array(preds)).mean():.2%} acc on {len(g)} gaps")
if __name__=="__main__": asyncio.run(main(sys.argv[1] if len(sys.argv)>1 else "AAPL"))
