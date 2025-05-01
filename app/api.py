from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import httpx, numpy as np

from . import data, indicators, sentiment, llm

app=FastAPI(title="Stock Gap Predictor",version="0.3.0")

class Resp(BaseModel):
    symbol:str; prediction_input:dict; predicted_direction:str|None; status:str; error_message:str|None

@app.post("/predict/",response_model=Resp)
async def predict(symbol:str=Query(...,examples={"sym":{"value":"AAPL"}})):
    try:
        async with httpx.AsyncClient() as c:
            stk=await data.fetch_stock(symbol); last=stk["Close"].iloc[-1]
            ind=indicators.enrich(stk).iloc[-1][["ADX","DMI_Plus","DMI_Minus","EMA_9_Close","EMA_9_Volume"]].fillna(0).to_dict()

            news_json,_=await data.fetch_news(symbol,c); news_df=sentiment.annotate(sentiment.pd.DataFrame(news_json),"headline")
            news_avg,news_cnt=sentiment.aggregate(news_df)

            red_json,_=await data.fetch_reddit(symbol); red_df=sentiment.annotate(sentiment.pd.DataFrame(red_json),"title")
            red_avg,red_cnt=sentiment.aggregate(red_df)

            dir=llm.predict(symbol,last,ind,news_avg,red_avg,news_cnt,red_cnt)
            return Resp(symbol=symbol,
                        prediction_input=dict(last_close=last,indicators=ind,
                                              news_sentiment_avg=news_avg,news_count=news_cnt,
                                              reddit_sentiment_avg=red_avg,reddit_count=red_cnt),
                        predicted_direction=dir,status="Success" if dir else "Error",error_message=None)
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
