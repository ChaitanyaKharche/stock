from dotenv import load_dotenv
import os
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import httpx
import numpy as np 
import pandas as pd 

from . import data, indicators, sentiment, llm
from .llm import predict_gap_trading_signal 

if os.path.exists(dotenv_path):
    print(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Fallback for simpler structures if .env is in the CWD when uvicorn starts
    if load_dotenv():
         print("Loading .env file from current working directory or standard locations.")
    else:
        print("Warning: .env file not found or not loaded.")

# You can then check if they are loaded
print(f"DEBUG: FINNHUB_API_KEY loaded: {'FINNHUB_API_KEY' in os.environ and os.environ['FINNHUB_API_KEY'] is not None}")
print(f"DEBUG: REDDIT_CLIENT_ID loaded: {'REDDIT_CLIENT_ID' in os.environ and os.environ['REDDIT_CLIENT_ID'] is not None}")


app = FastAPI(title="Stock Gap Predictor", version="0.4.0") 

class PredictionInputDetail(BaseModel):
    last_close: float
    technical_indicators_summary: dict 
    news_sentiment_avg: float
    news_count: int
    reddit_sentiment_avg: float
    reddit_count: int
    economic_impact_score: float
    gap_analysis_setup: dict | None 

class PredictionResponse(BaseModel):
    symbol: str
    prediction_input_summary: PredictionInputDetail 
    signal: str
    confidence: float
    reasoning: list[str]
    technical_triggers: list[str]
    economic_score: float
    gap_setup: dict | None
    position_size: float | None
    status: str
    error_message: str | None

class Resp(BaseModel):
    symbol: str
    prediction_input: dict
    predicted_direction: str|None
    status: str
    error_message: str|None

class SignalResp(BaseModel):
    symbol: str
    signal: str
    direction: str
    confidence: int
    reasoning: list
    technical_triggers: list
    status: str
    error_message: str|None


@app.post("/predict/", response_model=PredictionResponse) # Updated response_model
async def predict(symbol: str = Query(..., description="Stock symbol to predict (e.g., AAPL)", examples={"AAPL": {"value": "AAPL"}})):
    try:
        async with httpx.AsyncClient() as client: # Renamed c to client for clarity
            # 1. Fetch Stock Data
            # Increased days for more robust RSI divergence and gap analysis
            stock_df = await data.fetch_stock(symbol, days=90) 
            if stock_df.empty:
                raise ValueError("Failed to fetch stock data or data is empty.")
            
            last_close_price = float(stock_df["Close"].iloc[-1])


            # 2. Enrich with Technical Indicators (including new momentum ones)
            enriched_df = indicators.enrich(stock_df.copy()) # Pass a copy
            if enriched_df.empty:
                raise ValueError("Failed to enrich stock data with indicators.")
            
            # Extract latest indicators for the prediction function
            # Ensure your technical_data dictionary aligns with what predict_gap_trading_signal expects
            latest_indicators = enriched_df.iloc[-1].fillna(0.0).to_dict()
            # Example: ensure keys like 'ADX', 'DMI_Plus', 'DMI_Minus', 'RSI_14', 'MACDh_12_26_9' are present
            # Make sure column names from indicators.py (e.g. 'DMP_14' from adx.df.ta) are mapped correctly if needed
            # For example, if pandas_ta names it ADX_14, but your function expects ADX:
            technical_data_for_prediction = {
                'ADX': latest_indicators.get('ADX', latest_indicators.get('ADX_14', 0.0)), # Handle default ADX name
                'DMI_Plus': latest_indicators.get('DMI_Plus', latest_indicators.get('DMP_14', 0.0)),
                'DMI_Minus': latest_indicators.get('DMI_Minus', latest_indicators.get('DMN_14', 0.0)),
                'RSI_14': latest_indicators.get('RSI_14', 50.0), # RSI_14 is default from .ta.rsi()
                'MACDh_12_26_9': latest_indicators.get('MACDh_12_26_9', 0.0),
                'EMA_9_Close': latest_indicators.get('EMA_9_Close', 0.0),
                'Volume': latest_indicators.get('Volume', 0.0) # Example, if needed directly
            }


            print(f"DEBUG /predict/ - Technical indicators for {symbol}:")
            print(f"  ADX: {technical_data_for_prediction.get('ADX')}")
            print(f"  DMI+: {technical_data_for_prediction.get('DMI_Plus')}")
            print(f"  DMI-: {technical_data_for_prediction.get('DMI_Minus')}")
            print(f"  RSI: {technical_data_for_prediction.get('RSI_14')}")    




            # 3. Fetch and Analyze News
            # Using client for httpx requests, ensure fetch_news uses it
            news_json, _ = await data.fetch_news(symbol, client, days=3) # Fetch news for recent days
            news_df = pd.DataFrame(news_json if isinstance(news_json, list) else ([news_json] if news_json else []))
            
            if not news_df.empty and 'headline' in news_df.columns:
                news_df = sentiment.annotate(news_df, "headline") # FinBERT sentiment
                news_sentiment_avg, news_count = sentiment.aggregate(news_df)
                economic_impact_data = sentiment.analyze_economic_impact(news_df, symbol) # New economic impact
            else:
                news_sentiment_avg, news_count = 0.0, 0
                economic_impact_data = {"impact_score": 0.0, "key_events": []}


            # 4. Fetch and Analyze Reddit Data
            reddit_json, _ = await data.fetch_reddit(symbol) # Default limit=25, subs
            reddit_df = pd.DataFrame(reddit_json if isinstance(reddit_json, list) else ([reddit_json] if reddit_json else []))

            if not reddit_df.empty and 'title' in reddit_df.columns:
                reddit_df = sentiment.annotate(reddit_df, "title") # FinBERT sentiment
                reddit_sentiment_avg, reddit_count = sentiment.aggregate(reddit_df)
            else:
                reddit_sentiment_avg, reddit_count = 0.0, 0
            
            # 5. Perform Gap Analysis (using the enriched_df)
            # This needs the full historical (enriched) dataframe
            gap_analysis_data = indicators.identify_gap_patterns(enriched_df)


            print(f"DEBUG /predict/ - Sending to predict_gap_trading_signal for {symbol}:")
            print(f"  technical_data: {technical_data_for_prediction}")
            print(f"  news_sentiment_avg: {news_sentiment_avg}")
            print(f"  reddit_sentiment_avg: {reddit_sentiment_avg}")
            print(f"  economic_impact: {economic_impact_data}")
            print(f"  gap_analysis: {gap_analysis_data}")





            # 6. Get Final Prediction Signal (using the new rule-based logic)
            prediction_result = predict_gap_trading_signal(
                technical_data=technical_data_for_prediction, # Pass the curated dict
                news_sentiment_avg=float(news_sentiment_avg),
                reddit_sentiment_avg=float(reddit_sentiment_avg),
                economic_impact=economic_impact_data,
                gap_analysis=gap_analysis_data
            )


            # 7. Prepare and Return Response
            input_summary = PredictionInputDetail(
                last_close=last_close_price,
                technical_indicators_summary={k: v for k, v in technical_data_for_prediction.items() if isinstance(v, (int, float))}, # Only serializable items
                news_sentiment_avg=float(news_sentiment_avg),
                news_count=int(news_count),
                reddit_sentiment_avg=float(reddit_sentiment_avg),
                reddit_count=int(reddit_count),
                economic_impact_score=economic_impact_data.get("impact_score", 0.0),
                gap_analysis_setup=gap_analysis_data.get("current_setup")
            )
            
            return PredictionResponse(
                symbol=symbol,
                prediction_input_summary=input_summary,
                signal=prediction_result.get("signal", "ERROR"),
                confidence=prediction_result.get("confidence", 0.0),
                reasoning=prediction_result.get("reasoning", ["Error in prediction logic."]),
                technical_triggers=prediction_result.get("technical_triggers", []),
                economic_score=prediction_result.get("economic_score", 0.0),
                gap_setup=prediction_result.get("gap_setup"),
                position_size=prediction_result.get("position_size"),
                status="Success" if prediction_result.get("signal") != "ERROR" else "Error",
                error_message=None
            )

    except ValueError as ve: # Catch specific data errors
        print(f"ValueError in predict endpoint for {symbol}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve)) # Bad request if data is bad
    except Exception as e:
        print(f"ERROR in predict endpoint for {symbol}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    


@app.post("/analyze/", response_model=SignalResp)
async def analyze(symbol: str = Query(..., description="Stock symbol to analyze (e.g., AAPL)", examples={"AAPL": {"value": "AAPL"}})):
    """Analyzes stock data and provides a trading signal using the primary rule-based engine."""
    try:
        async with httpx.AsyncClient() as client: # Use 'client' consistently
            # 1. Fetch Stock Data
            stk = await data.fetch_stock(symbol, days=90) # Using 90 days like /predict/
            if stk.empty:
                raise ValueError(f"No stock data fetched for {symbol} in /analyze/")
            
            # 2. Calculate technical indicators
            stk_ta = indicators.enrich(stk.copy()) # Use a copy
            if stk_ta.empty:
                raise ValueError(f"Indicator enrichment failed for {symbol} in /analyze/")
            
            # Get latest indicator values
            latest_idx = -1 # Last row
            technical_data = stk_ta.iloc[latest_idx].fillna(0.0).to_dict()
            
            # Prepare technical_data dict with expected keys (matching /predict/)
            # Ensure these keys match what indicators.enrich actually produces (e.g., ADX_9 vs ADX_14)
            technical_data_for_signal = {
                'ADX': technical_data.get('ADX_9', technical_data.get('ADX_14', technical_data.get('ADX',0.0))),
                'DMI_Plus': technical_data.get('DMP_9', technical_data.get('DMP_14', technical_data.get('DMI_Plus',0.0))),
                'DMI_Minus': technical_data.get('DMN_9', technical_data.get('DMN_14', technical_data.get('DMI_Minus',0.0))),
                'RSI_14': technical_data.get('RSI_14', 50.0),
                'MACDh_12_26_9': technical_data.get('MACDh_12_26_9', 0.0),
                'EMA_9_Close': technical_data.get('EMA_9_Close', 0.0),
                'Volume': technical_data.get('Volume', 0.0)
            }
            print(f"DEBUG /analyze/ Tech for {symbol}: ADX={technical_data_for_signal.get('ADX'):.2f}, RSI={technical_data_for_signal.get('RSI_14'):.2f}, MACDh={technical_data_for_signal.get('MACDh_12_26_9'):.2f}")

            # 3. Identify gap patterns using the enriched DataFrame
            # *** CORRECTED: Pass stk_ta (enriched_df) to identify_gap_patterns ***
            gap_analysis = indicators.identify_gap_patterns(stk_ta.copy()) # Use a copy

            # 4. Fetch and analyze news
            news_json, _ = await data.fetch_news(symbol, client, days=3) # Consistent days
            news_df = pd.DataFrame(news_json if isinstance(news_json, list) else ([news_json] if news_json else []))
            
            news_sentiment_avg = 0.0
            # news_count is not directly used by predict_gap_trading_signal, but good for input_summary
            economic_impact = {"impact_score": 0.0, "key_events": []} 
            if not news_df.empty and 'headline' in news_df.columns:
                news_df_annotated = sentiment.annotate(news_df.copy(), "headline")
                news_sentiment_avg, _ = sentiment.aggregate(news_df_annotated) # news_count not directly used by signal func
                economic_impact = sentiment.analyze_economic_impact(news_df, symbol) # Pass symbol
            
            # 5. Reddit data
            reddit_json, _ = await data.fetch_reddit(symbol)
            reddit_df = pd.DataFrame(reddit_json if isinstance(reddit_json, list) else ([reddit_json] if reddit_json else []))
            
            reddit_sentiment_avg = 0.0
            if not reddit_df.empty and 'title' in reddit_df.columns:
                reddit_df_annotated = sentiment.annotate(reddit_df.copy(), "title")
                reddit_sentiment_avg, _ = sentiment.aggregate(reddit_df_annotated) # reddit_count not directly used

            # Debug print before calling the signal generator
            print(f"DEBUG /analyze/ - Sending to predict_gap_trading_signal for {symbol}:")
            print(f"  technical_data: {technical_data_for_signal}")
            print(f"  news_sentiment_avg: {news_sentiment_avg}")
            print(f"  reddit_sentiment_avg: {reddit_sentiment_avg}")
            print(f"  economic_impact: {economic_impact}")
            print(f"  gap_analysis: {gap_analysis}")

            # 6. Generate trading signal using the primary rule-based engine
            # *** CORRECTED: Call llm.predict_gap_trading_signal ***
            # *** Pass individual sentiment averages, not the 'sentiment_data' dict ***
            signal_result = llm.predict_gap_trading_signal(
                technical_data=technical_data_for_signal,
                news_sentiment_avg=float(news_sentiment_avg),
                reddit_sentiment_avg=float(reddit_sentiment_avg),
                economic_impact=economic_impact,
                gap_analysis=gap_analysis
            )
            
            # The old LLM direction check is removed as predict_gap_trading_signal is now the primary source.
            # If you want a secondary LLM check, it would need to be a separate call to a different llm function.

            # 7. Return response using SignalResp model
            # ** CORRECTED: Unpack signal_result directly into SignalResp **
            return SignalResp(
                symbol=symbol,
                **signal_result # Unpack all fields from predict_gap_trading_signal
            )

    except Exception as e:
        print(f"ERROR in /analyze/ endpoint for {symbol}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=f"An unexpected error in /analyze/: {str(e)}")
