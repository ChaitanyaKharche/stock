from dotenv import load_dotenv
import os
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import httpx
import numpy as np 
import pandas as pd 
from typing import Dict, List

from . import data, indicators, sentiment, llm
from .llm import predict_multi_timeframe_signal 
from .enhanced_sentiment import MultiModalSentimentAnalyzer
from .backtest import StrategyBacktester, BacktestResult, run_strategy_comparison
from .real_data_sources import RealDataProvider
from .tft_model import GapPredictionTFT
from .alternative_data import AlternativeDataProcessor

# global variables 
tft_predictor = GapPredictionTFT(context_length=96, prediction_length=1)
alt_data_processor = AlternativeDataProcessor()
enhanced_sentiment_analyzer = MultiModalSentimentAnalyzer()
real_data_provider = RealDataProvider()

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


app = FastAPI(title="Stock Gap Predictor", version="0.6.2") 

# --- Pydantic Models ---
class TimeframeAnalysisDetail(BaseModel):
    direction: str | None
    streak: int | None
    adx: float | None
    rsi: float | None
    gap_risk: str | None
    timeframe: str | None # Ensure indicators.py adds this to current_setup

class PredictionInputDetail(BaseModel):
    last_close: float # Corrected field name
    technical_indicators_summary: dict # Corrected field name
    news_sentiment_avg: float
    news_count: int
    reddit_sentiment_avg: float
    reddit_count: int
    economic_impact_score: float
    gap_analysis_setup: TimeframeAnalysisDetail | None # Corrected field name, holds daily setup for summary

class PredictionResponse(BaseModel):
    symbol: str
    prediction_input_summary: PredictionInputDetail | None = None # Optional
    signal: str
    confidence: float
    reasoning: List[str] # Explicitly List[str]
    technical_triggers: List[str] # Explicitly List[str]
    economic_score: float
    economic_components: Dict[str, float] | None = None
    gap_setup: Dict[str, TimeframeAnalysisDetail | None] | None
    position_size: float | None
    status: str
    error_message: str | None

# Resp and SignalResp are for the /analyze endpoint, which you might update or remove.
# For now, keeping them but ensuring SignalResp aligns better if /analyze uses the new llm function.
# class Resp(BaseModel):
#     symbol: str
#     prediction_input: dict
#     predicted_direction: str | None
#     status: str
#     error_message: str | None

# class SignalResp(BaseModel): # For /analyze/ endpoint
#     symbol: str
#     signal: str
#     confidence: float
#     reasoning: list[str]
#     technical_triggers: list[str]
#     economic_score: float
#     gap_setup: dict[str, TimeframeAnalysisDetail | None] | None # Aligned with PredictionResponse
#     position_size: float | None
#     status: str
#     error_message: str | None


@app.post("/predict/", response_model=PredictionResponse)
async def predict_endpoint_multi_tf(symbol: str = Query(..., description="Stock symbol", example="SPY")):
    print(f"--- PREDICT ENDPOINT (api.py VERSION_API_004) called for {symbol} ---")
    try:
        async with httpx.AsyncClient() as client:
            multi_tf_data_raw = await data.fetch_multi_timeframe_stock_data(symbol)
            
            if multi_tf_data_raw.get("daily", pd.DataFrame()).empty:
                raise ValueError(f"Failed to fetch essential daily stock data for {symbol}.")

            enriched_data = {}
            latest_indicators_tf = {}
            gap_analyses_tf = {}

            timeframes_to_process = ["daily", "hourly", "15m"]
            for tf_str in timeframes_to_process:
                df_raw = multi_tf_data_raw.get(tf_str)
                if df_raw is not None and not df_raw.empty:
                    enriched_df = indicators.enrich(df_raw.copy(), interval_str=tf_str)
                    enriched_data[tf_str] = enriched_df
                    if not enriched_df.empty:
                        latest_indicators_tf[tf_str] = enriched_df.iloc[-1].fillna(0.0).to_dict()
                        gap_analyses_tf[tf_str] = indicators.identify_gap_patterns(enriched_df.copy(), timeframe_str=tf_str)
                    else:
                        latest_indicators_tf[tf_str] = {}
                        gap_analyses_tf[tf_str] = {"gaps": [], "current_setup": None}
                else:
                    latest_indicators_tf[tf_str] = {}
                    gap_analyses_tf[tf_str] = {"gaps": [], "current_setup": None}

            def _get_tech_dict(indicators_dict_raw):
                return {
                    'ADX': indicators_dict_raw.get('ADX_9', indicators_dict_raw.get('ADX', 0.0)),
                    'DMI_Plus': indicators_dict_raw.get('DMP_9', indicators_dict_raw.get('DMI_Plus', 0.0)),
                    'DMI_Minus': indicators_dict_raw.get('DMN_9', indicators_dict_raw.get('DMI_Minus', 0.0)),
                    'RSI_14': indicators_dict_raw.get('RSI_14', 50.0),
                    'MACDh_12_26_9': indicators_dict_raw.get('MACDh_12_26_9', 0.0),
                    'EMA_9_Close': indicators_dict_raw.get('EMA_9_Close', 0.0),
                    'Volume': indicators_dict_raw.get('Volume', 0.0),
                    'ATR_14': indicators_dict_raw.get('ATR_14', 0.0),
                    'Close': indicators_dict_raw.get('Close', 0.0)
                }

            tech_daily = _get_tech_dict(latest_indicators_tf.get("daily", {}))
            tech_hourly = _get_tech_dict(latest_indicators_tf.get("hourly", {}))
            tech_15m = _get_tech_dict(latest_indicators_tf.get("15m", {}))
            
            print(f"DEBUG /predict/ Daily Tech for {symbol}: ADX={tech_daily.get('ADX'):.2f}, RSI={tech_daily.get('RSI_14'):.2f}, MACDh={tech_daily.get('MACDh_12_26_9'):.2f}")

            # ===== FETCH NEWS AND REDDIT DATA FIRST =====
            news_json, _ = await data.fetch_news(symbol, client, days=3)
            news_df = pd.DataFrame(news_json if isinstance(news_json, list) else ([news_json] if news_json else []))

            reddit_json, _ = await data.fetch_reddit(symbol)
            reddit_df = pd.DataFrame(reddit_json if isinstance(reddit_json, list) else ([reddit_json] if reddit_json else []))

            # ===== NOW USE ENHANCED SENTIMENT ANALYSIS =====
            news_sentiment_avg, news_count = 0.0, 0
            reddit_sentiment_avg, reddit_count = 0.0, 0
            economic_impact_data = {"impact_score": 0.0, "key_events": []}
            sentiment_confidence = "MEDIUM" 

            try:
                print(f"Using enhanced sentiment analysis for {symbol}")
                comprehensive_sentiment = enhanced_sentiment_analyzer.analyze_comprehensive_sentiment(
                    news_data=news_df,
                    reddit_data=reddit_df,
                    symbol=symbol
                )
                
                news_sentiment_avg = comprehensive_sentiment["financial_sentiment"]
                reddit_sentiment_avg = comprehensive_sentiment["social_sentiment"]
                economic_impact_data = {
                    "impact_score": comprehensive_sentiment["economic_impact"],
                    "key_events": comprehensive_sentiment["key_themes"]
                }
                
                news_count = len(news_df) if not news_df.empty else 0
                reddit_count = len(reddit_df) if not reddit_df.empty else 0
                
            except Exception as e:
                print(f"Enhanced sentiment failed: {e}, falling back to basic sentiment")
                
                # Fallback to basic sentiment
                if not news_df.empty and 'headline' in news_df.columns:
                    news_df_annotated = sentiment.annotate(news_df.copy(), "headline")
                    news_sentiment_avg, news_count = sentiment.aggregate(news_df_annotated)
                    economic_impact_data = sentiment.analyze_economic_impact(news_df, symbol)
                
                if not reddit_df.empty and 'title' in reddit_df.columns:
                    reddit_df_annotated = sentiment.annotate(reddit_df.copy(), "title")
                    reddit_sentiment_avg, reddit_count = sentiment.aggregate(reddit_df_annotated)

                        # ===== ENHANCED TFT PREDICTION =====
            try:
                print(f"Running TFT gap prediction for {symbol}")
                
                # Prepare data for TFT
                daily_df = multi_tf_data_raw.get("daily", pd.DataFrame())
                
                if not daily_df.empty and len(daily_df) >= 100:
                    # Train TFT if not already trained (you might want to do this offline)
                    if not tft_predictor.is_trained:
                        print("Training TFT model...")
                        tft_predictor.train(daily_df, epochs=30)
                    
                    # Get TFT predictions
                    tft_prediction = tft_predictor.predict_gap_probability(daily_df)
                    
                    print(f"TFT Prediction: {tft_prediction['expected_direction']} with {tft_prediction['confidence']} confidence")
                else:
                    tft_prediction = {
                        "gap_probability": 50.0,
                        "expected_direction": "FLAT",
                        "confidence": "LOW"
                    }
                    
            except Exception as e:
                print(f"TFT prediction failed: {e}")
                tft_prediction = {"gap_probability": 50.0, "expected_direction": "FLAT", "confidence": "LOW"}

            # ===== ALTERNATIVE DATA FEATURES =====
            try:
                print(f"Generating alternative data features for {symbol}")
                alt_features = alt_data_processor.generate_alternative_features(symbol)
                print(f"Alternative data: VIX={alt_features['vix_level']}, Sector={alt_features['sector_rotation_signal']:.3f}")
            except Exception as e:
                print(f"Alternative data generation failed: {e}")
                alt_features = {}

            # Update your signal_result call to include TFT and alternative data:
            signal_result = llm.predict_multi_timeframe_signal(
                tech_daily=tech_daily, 
                setup_daily=gap_analyses_tf.get("daily", {}).get("current_setup"),
                tech_hourly=tech_hourly, 
                setup_hourly=gap_analyses_tf.get("hourly", {}).get("current_setup"),
                tech_15m=tech_15m, 
                setup_15m=gap_analyses_tf.get("15m", {}).get("current_setup"),
                news_sentiment_avg=float(news_sentiment_avg),
                reddit_sentiment_avg=float(reddit_sentiment_avg),
                economic_impact=economic_impact_data,
                tft_prediction=tft_prediction,  # Add TFT prediction
                alternative_data=alt_features   # Add alternative data
            )
            # ===== CONTINUE WITH LLM SIGNAL GENERATION =====
            signal_result = llm.predict_multi_timeframe_signal(
                tech_daily=tech_daily, setup_daily=gap_analyses_tf.get("daily", {}).get("current_setup"),
                tech_hourly=tech_hourly, setup_hourly=gap_analyses_tf.get("hourly", {}).get("current_setup"),
                tech_15m=tech_15m, setup_15m=gap_analyses_tf.get("15m", {}).get("current_setup"),
                news_sentiment_avg=float(news_sentiment_avg),
                reddit_sentiment_avg=float(reddit_sentiment_avg),
                economic_impact=economic_impact_data
            )
            
            daily_df_for_summary = multi_tf_data_raw.get("daily")
            last_close_daily_val = daily_df_for_summary["Close"].iloc[-1] if daily_df_for_summary is not None and not daily_df_for_summary.empty else 0.0
            
            input_summary_detail_data = {
                "last_close": last_close_daily_val,
                "technical_indicators_summary": tech_daily,
                "news_sentiment_avg": news_sentiment_avg,
                "news_count": news_count,
                "reddit_sentiment_avg": reddit_sentiment_avg,
                "reddit_count": reddit_count,
                "economic_impact_score": economic_impact_data.get("impact_score", 0.0),
                "gap_analysis_setup": gap_analyses_tf.get("daily", {}).get("current_setup")
            }
            
            print(f"DEBUG /predict/ Data for PredictionInputDetail: {input_summary_detail_data}")
            input_summary_detail = PredictionInputDetail(**input_summary_detail_data)

            final_response_data = {
                "symbol": symbol,
                "prediction_input_summary": input_summary_detail,
                "signal": signal_result.get("signal", "ERROR"),
                "confidence": signal_result.get("confidence", 0.0),
                "reasoning": signal_result.get("reasoning", []) + [f"Sentiment Confidence: {sentiment_confidence}"],
                "technical_triggers": signal_result.get("technical_triggers", []),
                "economic_score": signal_result.get("economic_score", 0.0),
                "economic_components": signal_result.get("economic_components"),
                "gap_setup": signal_result.get("gap_setup"),
                "position_size": signal_result.get("position_size"),
                "status": "Success" if signal_result.get("signal", "ERROR") != "ERROR" else "Error",
                "error_message": None
            }
            return PredictionResponse(**final_response_data)

    except Exception as e:
        print(f"ERROR in /predict/ endpoint (multi-TF) for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return PredictionResponse(
            symbol=symbol,
            prediction_input_summary=None,
            signal="ERROR",
            confidence=0.0,
            reasoning=[f"Failed: {str(e)}"],
            technical_triggers=[],
            economic_score=0.0,
            gap_setup=None,
            position_size=0.0,
            status="Error",
            error_message=str(e)
        )


@app.post("/backtest/")
async def backtest_strategy(
    symbol: str = Query(..., description="Stock symbol"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    initial_capital: float = Query(100000, description="Initial capital"),
    min_confidence: int = Query(50, description="Minimum confidence threshold"),
    max_position_size: float = Query(0.25, description="Maximum position size")
):
    """
    Run historical backtest of gap prediction strategy.
    """
    try:
        # Initialize backtester
        backtester = StrategyBacktester(initial_capital=initial_capital)
        
        # Strategy parameters
        strategy_params = {
            "min_confidence": min_confidence,
            "max_position_size": max_position_size,
            "stop_loss": 0.03,
            "take_profit": 0.06,
            "hold_days": 1
        }
        
        # Run backtest
        result = backtester.run_historical_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            strategy_params=strategy_params
        )
        
        return {
            "status": "success",
            "backtest_result": result.__dict__,
            "strategy_params": strategy_params
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "backtest_result": None
        }

@app.post("/enhanced-data/")
async def get_enhanced_market_data(
    symbol: str = Query(..., description="Stock symbol")
):
    """
    Get enhanced market data from multiple real sources.
    """
    try:
        enhanced_data = real_data_provider.get_enhanced_market_data(symbol)
        market_regime = real_data_provider.get_market_regime_analysis()
        
        return {
            "status": "success",
            "enhanced_data": enhanced_data,
            "market_regime": market_regime
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "enhanced_data": None
        }

@app.post("/strategy-comparison/")
async def compare_strategies(
    symbols: List[str] = Query(..., description="List of symbols to compare"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    """
    Compare strategy performance across multiple symbols.
    """
    try:
        comparison_results = run_strategy_comparison(symbols, start_date, end_date)
        
        return {
            "status": "success",
            "comparison_results": comparison_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "comparison_results": None
        }
    






# @app.post("/analyze/", response_model=SignalResp)
# async def analyze(symbol: str = Query(..., description="Stock symbol to analyze (e.g., AAPL)", examples={"AAPL": {"value": "AAPL"}})):
#     """Analyzes stock data and provides a trading signal using the primary rule-based engine."""
#     try:
#         async with httpx.AsyncClient() as client: # Use 'client' consistently
#             # 1. Fetch Stock Data
#             stk = await data.fetch_stock(symbol, days=90) # Using 90 days like /predict/
#             if stk.empty:
#                 raise ValueError(f"No stock data fetched for {symbol} in /analyze/")
            
#             # 2. Calculate technical indicators
#             stk_ta = indicators.enrich(stk.copy()) # Use a copy
#             if stk_ta.empty:
#                 raise ValueError(f"Indicator enrichment failed for {symbol} in /analyze/")
            
#             # Get latest indicator values
#             latest_idx = -1 # Last row
#             technical_data = stk_ta.iloc[latest_idx].fillna(0.0).to_dict()
            
#             # Prepare technical_data dict with expected keys (matching /predict/)
#             # Ensure these keys match what indicators.enrich actually produces (e.g., ADX_9 vs ADX_14)
#             technical_data_for_signal = {
#                 'ADX': technical_data.get('ADX_9', technical_data.get('ADX_14', technical_data.get('ADX',0.0))),
#                 'DMI_Plus': technical_data.get('DMP_9', technical_data.get('DMP_14', technical_data.get('DMI_Plus',0.0))),
#                 'DMI_Minus': technical_data.get('DMN_9', technical_data.get('DMN_14', technical_data.get('DMI_Minus',0.0))),
#                 'RSI_14': technical_data.get('RSI_14', 50.0),
#                 'MACDh_12_26_9': technical_data.get('MACDh_12_26_9', 0.0),
#                 'EMA_9_Close': technical_data.get('EMA_9_Close', 0.0),
#                 'Volume': technical_data.get('Volume', 0.0)
#             }
#             print(f"DEBUG /analyze/ Tech for {symbol}: ADX={technical_data_for_signal.get('ADX'):.2f}, RSI={technical_data_for_signal.get('RSI_14'):.2f}, MACDh={technical_data_for_signal.get('MACDh_12_26_9'):.2f}")

#             # 3. Identify gap patterns using the enriched DataFrame
#             # *** CORRECTED: Pass stk_ta (enriched_df) to identify_gap_patterns ***
#             gap_analysis = indicators.identify_gap_patterns(stk_ta.copy()) # Use a copy

#             # 4. Fetch and analyze news
#             news_json, _ = await data.fetch_news(symbol, client, days=3) # Consistent days
#             news_df = pd.DataFrame(news_json if isinstance(news_json, list) else ([news_json] if news_json else []))
            
#             news_sentiment_avg = 0.0
#             # news_count is not directly used by predict_gap_trading_signal, but good for input_summary
#             economic_impact = {"impact_score": 0.0, "key_events": []} 
#             if not news_df.empty and 'headline' in news_df.columns:
#                 news_df_annotated = sentiment.annotate(news_df.copy(), "headline")
#                 news_sentiment_avg, _ = sentiment.aggregate(news_df_annotated) # news_count not directly used by signal func
#                 economic_impact = sentiment.analyze_economic_impact(news_df, symbol) # Pass symbol
            
#             # 5. Reddit data
#             reddit_json, _ = await data.fetch_reddit(symbol)
#             reddit_df = pd.DataFrame(reddit_json if isinstance(reddit_json, list) else ([reddit_json] if reddit_json else []))
            
#             reddit_sentiment_avg = 0.0
#             if not reddit_df.empty and 'title' in reddit_df.columns:
#                 reddit_df_annotated = sentiment.annotate(reddit_df.copy(), "title")
#                 reddit_sentiment_avg, _ = sentiment.aggregate(reddit_df_annotated) # reddit_count not directly used

#             # Debug print before calling the signal generator
#             print(f"DEBUG /analyze/ - Sending to predict_gap_trading_signal for {symbol}:")
#             print(f"  technical_data: {technical_data_for_signal}")
#             print(f"  news_sentiment_avg: {news_sentiment_avg}")
#             print(f"  reddit_sentiment_avg: {reddit_sentiment_avg}")
#             print(f"  economic_impact: {economic_impact}")
#             print(f"  gap_analysis: {gap_analysis}")

#             # 6. Generate trading signal using the primary rule-based engine
#             # *** CORRECTED: Call llm.predict_gap_trading_signal ***
#             # *** Pass individual sentiment averages, not the 'sentiment_data' dict ***
#             signal_result = llm.predict_gap_trading_signal(
#                 technical_data=technical_data_for_signal,
#                 news_sentiment_avg=float(news_sentiment_avg),
#                 reddit_sentiment_avg=float(reddit_sentiment_avg),
#                 economic_impact=economic_impact,
#                 gap_analysis=gap_analysis
#             )
            
#             # The old LLM direction check is removed as predict_gap_trading_signal is now the primary source.
#             # If you want a secondary LLM check, it would need to be a separate call to a different llm function.

#             # 7. Return response using SignalResp model
#             # ** CORRECTED: Unpack signal_result directly into SignalResp **
#             return SignalResp(
#                 symbol=symbol,
#                 **signal_result # Unpack all fields from predict_gap_trading_signal
#             )

#     except Exception as e:
#         print(f"ERROR in /analyze/ endpoint for {symbol}: {e}")
#         import traceback
#         traceback.print_exc() # Print full traceback for debugging
#         raise HTTPException(status_code=500, detail=f"An unexpected error in /analyze/: {str(e)}")
