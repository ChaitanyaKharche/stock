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
            stk=await data.fetch_stock(symbol)
            last=stk["Close"].iloc[-1]
            
            ind=indicators.enrich(stk).iloc[-1][["ADX","DMI_Plus","DMI_Minus","EMA_9_Close","EMA_9_Volume"]].fillna(0).to_dict()
            
            # Fix for news data
            news_json,_=await data.fetch_news(symbol,c)
            print(f"DEBUG - news_json type: {type(news_json)}")
            
            # Create DataFrame with proper index handling
            if isinstance(news_json, list):
                news_df = sentiment.pd.DataFrame(news_json)
            elif isinstance(news_json, dict):
                # For a single dictionary, create a DataFrame with one row
                news_df = sentiment.pd.DataFrame([news_json], index=[0])
            else:
                # Handle empty or unexpected data
                news_df = sentiment.pd.DataFrame()
            
            news_df = sentiment.annotate(news_df, "headline")
            news_avg, news_cnt = sentiment.aggregate(news_df)
            
            # Similar fix for Reddit data
            red_json,_ = await data.fetch_reddit(symbol)
            if isinstance(red_json, list):
                red_df = sentiment.pd.DataFrame(red_json)
            elif isinstance(red_json, dict):
                red_df = sentiment.pd.DataFrame([red_json], index=[0])
            else:
                red_df = sentiment.pd.DataFrame()
                
            red_df = sentiment.annotate(red_df, "title")
            red_avg, red_cnt = sentiment.aggregate(red_df)
            
            # Force type conversions
            last = float(last)
            news_avg = float(news_avg)
            red_avg = float(red_avg)
            
            dir = llm.predict(symbol, last, ind, news_avg, red_avg, news_cnt, red_cnt)
            return Resp(symbol=symbol,
                        prediction_input=dict(last_close=last, indicators=ind,
                                             news_sentiment_avg=news_avg, news_count=news_cnt,
                                             reddit_sentiment_avg=red_avg, reddit_count=red_cnt),
                        predicted_direction=dir, status="Success" if dir else "Error", error_message=None)
    except Exception as e:
        print(f"ERROR in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
