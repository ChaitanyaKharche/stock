# trade_analysis/enhanced_api.py

import os
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from pathlib import Path

# Import only modules that still exist
from .data import UnifiedDataProvider
from .market_session import market_status, banner as session_banner
from .momentum_trading_engine import DIRECTION_AGREEMENT, DIRECTION_DEADBAND
from .indicators import enrich_with_indicators, identify_current_setup
from .enhanced_sentiment import EnhancedFinancialSentimentAnalyzer, analyze_momentum_sentiment
from .momentum_trading_engine import IntegratedMomentumEngine
from .enhanced_llm import EnhancedLLMEngine, generate_enhanced_llm_signal
from .tft_model import GapPredictionTFT
from .agent import TradingAgent, analyze_agent_performance

# Global dictionary to store TFT models
api_tft_models = {} 
trading_agent = None

def sanitize_for_json(data: any) -> any:
    """Recursively converts numpy and pandas types to JSON-serializable types."""
    if isinstance(data, dict):
        return {key: sanitize_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, (np.integer, np.int64)):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif isinstance(data, (pd.Series, pd.Index, np.ndarray)):
        return data.tolist()
    return data

class EnhancedSignalResponse(BaseModel):
    """Enhanced response model with momentum and LLM analysis"""
    symbol: str
    signal: str
    confidence: float
    reasoning: str
    position_size: float
    status: str
    details: Dict[str, Any]
    
    # Enhanced fields
    momentum_analysis: Dict[str, Any] = {}
    llm_ensemble: Dict[str, Any] = {}
    options_strategy: Dict[str, Any] = {}
    timeframe_recommendation: str = "15m"
    expected_hold_time: str = "Unknown"
    # The verdict's own explanation and the gates it was measured against.
    verdict: Dict[str, Any] = {}
    # Last price, previous close and the day's range, so the signal is grounded.
    quote: Dict[str, Any] = {}
    # Which SESSION these bars are from, and whether the market was open when asked.
    # Without this the app served Friday's tape on Labor Day with nothing saying so.
    market: Dict[str, Any] = {}

# Enhanced FastAPI App
app = FastAPI(
    title="Enhanced Intraday Momentum Engine", 
    version="2.0.0",
    description="SOTA Financial AI with multi-LLM ensemble and momentum analysis"
)

# Initialize enhanced components
data_provider = UnifiedDataProvider()
sentiment_analyzer = EnhancedFinancialSentimentAnalyzer()
momentum_engine = IntegratedMomentumEngine()
llm_engine = EnhancedLLMEngine()
tft_predictor = GapPredictionTFT(context_length=96, prediction_length=1)

@app.on_event("startup")
async def startup_event():
    """Initialize all AI models on startup and launch the agent."""
    print("🚀 Starting Enhanced Trading Engine...")

    # --- This new logic checks the environment before loading models ---
    from .deploy import DeploymentConfig
    config = DeploymentConfig.auto_detect()

    # Load sentiment models regardless of environment
    print("📊 Loading sentiment models...")
    sentiment_analyzer.initialize_models()

    # Only load LLMs if we are NOT on a CPU
    if config.device != "cpu":
        print("🧠 Loading LLM ensemble...")
        llm_engine.initialize_llm_models()
    else:
        print("🚫 CPU environment detected. Skipping LLM loading.")

    # Load TFT models. The previous version built a bare GapPredictionTFT and left the
    # weight loading as a comment -- "(The rest of your TFT loading logic...)" -- so every
    # instance had is_trained=False, and the request handler's else-branch trains a model
    # for 20 epochs INSIDE the HTTP request. That never fired only because the TFT was
    # gated behind >=96 daily rows and the old data layer returned one row.
    print("🤖 Loading TFT models...")
    symbols = ['QQQ', 'SPY', 'MSFT', 'TSLA', 'NVDA', 'META']
    for symbol in symbols:
        tft_instance = GapPredictionTFT()
        try:
            tft_instance.load_pretrained(symbol=symbol)
        except Exception as exc:                                 # noqa: BLE001
            # A missing or unreadable checkpoint must not take down startup. The instance
            # stays is_trained=False and the request path degrades to _default_prediction.
            print(f"⚠️  TFT for {symbol} not loaded ({type(exc).__name__}: {exc}); "
                  f"predictions for it will use the neutral default.")
        api_tft_models[symbol] = tft_instance
    loaded = [s for s, m in api_tft_models.items() if m.is_trained]
    print(f"🤖 TFT ready for {len(loaded)}/{len(symbols)}: {loaded or 'none'}")

    # Initialize and run the agent as a background task
    global trading_agent
    trading_agent = TradingAgent(api_url="http://localhost:7860")
    print("🤖 Launching Trading Agent as a background task...")
    asyncio.create_task(trading_agent.run())

    print("✅ Enhanced Trading Engine startup complete!")

@app.get("/")
def read_root():
    """Enhanced root endpoint with system info"""
    import torch
    
    gpu_info = "CPU only"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_info = f"{gpu_name} ({gpu_memory:.1f} GB)"
    
    return {
        "status": "operational",
        "engine": "Enhanced Intraday Momentum Engine v2.0.0",
        "gpu_info": gpu_info,
        "features": [
            "Multi-LLM Ensemble Analysis",
            "Advanced Sentiment Analysis (10+ models)",
            "High-Frequency Momentum Engine", 
            "Options Strategy Generation",
            "TFT Gap Prediction",
            "Autonomous Trading Agent"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/enhanced/", response_model=EnhancedSignalResponse)
async def predict_enhanced_signal(
    symbol: str = Query(..., description="Stock symbol (e.g., QQQ, SPY)"),
    timeframe: str = Query("5m", description="Trading timeframe: 15m, 1h, 4h, 1d"),
    strategy_mode: str = Query("momentum", description="Strategy: momentum, gap, reversal")
):
    """
    Enhanced prediction endpoint with full AI stack
    """
    try:
        start_time = datetime.now()
        
        # Fetch market data. No `async with httpx.AsyncClient()` here: UnifiedDataProvider
        # owns one long-lived client and reuses it, so opening a second one per request
        # only cost a TCP/TLS handshake per call. It existed to be passed to fetch_news --
        # whose signature is (symbol, days=3), so the client was bound to `days` and any
        # symbol WITHOUT a local_data/*.json snapshot hit timedelta(days=<AsyncClient>)
        # -> TypeError -> HTTP 500. The nine cached symbols return before that line, which
        # is why it survived: the demo symbols were exactly the ones that never reached it.
        print(f"📈 Fetching data for {symbol}...")
        ohlcv_data = await data_provider.fetch_multi_timeframe_stock_data(symbol)
        news_data, _ = await data_provider.fetch_news(symbol)
        reddit_data, _ = await data_provider.fetch_reddit_data(symbol)
        alt_data = data_provider.get_alternative_data(symbol)

        # Process dataframes
        news_df = pd.DataFrame(news_data) if news_data else pd.DataFrame()
        reddit_df = pd.DataFrame(reddit_data) if reddit_data else pd.DataFrame()
        
        # Technical analysis for each timeframe
        tech_setups = {}
        for tf, df in ohlcv_data.items():
            if not df.empty:
                enriched_df = enrich_with_indicators(df.copy(), tf)
                tech_setups[tf] = identify_current_setup(enriched_df, tf)
        
        print("🔄 Running AI analysis...")
        
        # 1. Enhanced Sentiment Analysis
        sentiment_analysis = await asyncio.get_event_loop().run_in_executor(
            None, 
            analyze_momentum_sentiment,
            news_df, reddit_df, symbol, timeframe
        )
        
        # 2. Momentum Analysis
        # `timeframe` is passed through so the user's selection actually reweights the
        # bars (see TIMEFRAME_WEIGHTS in momentum_trading_engine). Until 2026-09-07 it
        # stopped at enhanced_api and only moved a threshold, making the selector inert.
        momentum_analysis = momentum_engine.generate_enhanced_signal(
            ohlcv_data, sentiment_analysis, alt_data, timeframe
        )
        
        # 3. TFT Prediction
        daily_df = ohlcv_data.get("daily")
        tft_prediction = None
        tft_model = api_tft_models.get(symbol.upper())
        
        # NEVER train here. The old else-branch called tft_model.train(daily_df, epochs=20)
        # inside the request, which on cpu-basic hardware means an HTTP handler running a
        # neural-network training loop while a user waits. It was unreachable only because
        # the >=96-row gate never passed; supplying real history made it reachable, so it
        # is removed rather than left as a latent timeout. An unloaded model degrades to
        # the neutral prediction, which is what the weighted signal already handles.
        if daily_df is not None and len(daily_df) >= 96 and tft_model:
            if tft_model.is_trained:
                tft_prediction = tft_model.predict_gap_probability(daily_df)
                print(f"🚀 Using pretrained TFT model for {symbol}")
            else:
                print(f"⚠️  TFT for {symbol} has no loaded weights; using neutral default "
                      f"(training in a request handler is never acceptable).")
                tft_prediction = tft_model._default_prediction()
        else:
            if tft_model:
                tft_prediction = tft_model._default_prediction()
            else:
                temp_tft = GapPredictionTFT()
                tft_prediction = temp_tft._default_prediction()
        
        # 4. LLM Ensemble Analysis
        llm_analysis = {}
        try:
            llm_analysis = llm_engine.generate_enhanced_trading_signal(
                ohlcv_data, sentiment_analysis, momentum_analysis, alt_data
            )
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            conditions = {
                "is_vix_high": alt_data.get('vix_level', 0) > 25,
                "is_15m_rsi_bullish": tech_setups.get("15m", {}).get('rsi', 50) > 65,
                "is_15m_rsi_bearish": tech_setups.get("15m", {}).get('rsi', 50) < 35,
                "is_15m_volume_spike": tech_setups.get("15m", {}).get('volume_spike', False),
                "is_hourly_trend_bullish": tech_setups.get("hourly", {}).get('direction') == 'up',
                "is_hourly_trend_bearish": tech_setups.get("hourly", {}).get('direction') == 'down'
            }
            llm_analysis = generate_enhanced_llm_signal(conditions)
        
        # 4b. Real overnight gap, for strategy_mode="gap".
        # That mode used to be gated on tft_prediction["gap_probability"] > 70. The TFT
        # returns ~67.1 for every symbol (measured spread 0.10 across six symbols), so the
        # branch was UNREACHABLE and "gap" was a silent no-op -- one of the reasons all 12
        # timeframe x strategy combinations produced an identical answer. The gap is a
        # directly observable quantity, so it is now measured from the daily bars instead
        # of being predicted by a model that has collapsed to its prior.
        gap_pct = None
        if daily_df is not None and len(daily_df) >= 2:
            prev_close = float(daily_df["Close"].iloc[-2])
            today_open = float(daily_df["Open"].iloc[-1])
            if prev_close > 0:
                gap_pct = (today_open - prev_close) / prev_close * 100.0

        # 5. Master Signal Generation - FIXED FUNCTION NAME
        master_signal = _generate_master_signal(
            momentum_analysis, llm_analysis, sentiment_analysis, tft_prediction,
            timeframe, strategy_mode, gap_pct, tech_setups
        )
        
        # 6. Options Strategy - FIXED FUNCTION NAME
        options_strategy = _generate_options_strategy(
            master_signal, momentum_analysis, alt_data, timeframe, strategy_mode
        )
        
        # Session context. Computed from the clock and the exchange calendar, then
        # CROSS-CHECKED against the newest bar actually fetched -- if the two disagree the
        # data is older than the calendar says it should be (a provider lag or an
        # unlisted closure), and saying so is better than asserting a session we did not
        # actually receive.
        market = market_status()
        newest = None
        for _df in ohlcv_data.values():
            if _df is not None and not _df.empty:
                ts = _df.index[-1]
                newest = ts if newest is None or ts > newest else newest
        if newest is not None:
            market["latest_bar"] = str(newest)
            market["latest_bar_session"] = str(getattr(newest, "date", lambda: newest)())
            ref = market.get("reference_session")
            if ref and market["latest_bar_session"] != ref:
                market["warning"] = (
                    f"Newest bar is from {market['latest_bar_session']}, but the calendar "
                    f"says the last session was {ref}. Data may lag the exchange.")
        market["banner"] = session_banner(market)

        # Price context. A signal with no price is ungrounded: the reader cannot tell
        # whether "bearish" means down 0.2% or down 6%, and every judgement about the
        # suggested strikes depends on where the underlying actually is.
        quote = {}
        _fine = ohlcv_data.get("15m")
        _daily = ohlcv_data.get("daily")
        if _fine is not None and not _fine.empty:
            quote["price"] = round(float(_fine["Close"].iloc[-1]), 2)
        elif _daily is not None and not _daily.empty:
            quote["price"] = round(float(_daily["Close"].iloc[-1]), 2)
        if _daily is not None and len(_daily) >= 2 and quote.get("price"):
            prev = float(_daily["Close"].iloc[-2])
            if prev:
                quote["prev_close"] = round(prev, 2)
                quote["change_pct"] = round((quote["price"] - prev) / prev * 100, 2)
        if _daily is not None and not _daily.empty:
            quote["day_high"] = round(float(_daily["High"].iloc[-1]), 2)
            quote["day_low"] = round(float(_daily["Low"].iloc[-1]), 2)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        sanitized_details = sanitize_for_json({
            "tech_setups": tech_setups,
            "sentiment": sentiment_analysis,
            "alternative_data": alt_data,
            "tft_prediction": tft_prediction,
            "processing_time_seconds": processing_time,
            "data_sources": {
                "news_articles": len(news_df),
                "social_posts": len(reddit_df),
                "timeframes_analyzed": list(ohlcv_data.keys())
            }
        })
        
        return EnhancedSignalResponse(
            symbol=symbol,
            signal=master_signal["signal"],
            confidence=master_signal["confidence"],
            reasoning=master_signal["reasoning"],
            position_size=master_signal["position_size"],
            status="Success",
            details=sanitized_details,
            momentum_analysis=sanitize_for_json(momentum_analysis),
            llm_ensemble=sanitize_for_json(llm_analysis),
            options_strategy=sanitize_for_json(options_strategy),
            timeframe_recommendation=master_signal.get("timeframe", timeframe),
            expected_hold_time=master_signal.get("hold_time", "Unknown"),
            verdict={k: master_signal.get(k) for k in (
                "blocking_reason", "min_confidence", "score_threshold", "weighted_score",
                "direction", "direction_agreement", "overnight_gap_pct",
                "momentum_strategy", "strategy_mode")},
            market=market,
            quote=quote
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {e}")

@app.get("/agent/start")
async def start_agent():
    """Start the autonomous agent"""
    if trading_agent:
        asyncio.create_task(trading_agent.run())
        return {"status": "Agent started"}
    return {"status": "Agent not initialized"}

@app.get("/agent/stats")
async def get_agent_stats():
    """Get agent's performance stats"""
    if trading_agent:
        return trading_agent.get_stats()
    return {"error": "Agent not initialized"}

@app.get("/agent/positions")
async def get_agent_positions():
    """Get current positions"""
    if trading_agent:
        return {"positions": trading_agent.positions}
    return {"positions": {}}

@app.get("/agent/analyze")
async def analyze_agent():
    """Analyze agent's performance"""
    try:
        analyze_agent_performance()
        return {"status": "Analysis complete - check console output"}
    except Exception as e:
        return {"error": str(e)}

def _generate_master_signal(momentum_analysis: Dict, llm_analysis: Dict, 
                          sentiment_analysis: Dict, tft_prediction: Dict,
                          timeframe: str = "15m", strategy_mode: str = "momentum",
                          gap_pct: float = None, tech_setups: Dict = None) -> Dict:
    """Generate master trading signal from all analyses - FIXED VERSION"""
    
    # Extract signals
    momentum_signal = momentum_analysis.get("signal", "HOLD")
    momentum_confidence = momentum_analysis.get("confidence", 50)
    
    # Use the actual momentum analysis results
    momentum_master = momentum_analysis.get("momentum_analysis", {}).get("master_signal", {})
    momentum_strategy = momentum_master.get("strategy", "WAIT")
    
    llm_signal = llm_analysis.get("signal", "HOLD")
    sentiment_composite = sentiment_analysis.get("composite_score", 0)
    tft_direction = tft_prediction.get("expected_direction", "FLAT") if tft_prediction else "FLAT"
    
    # TIMEFRAME-SPECIFIC THRESHOLDS
    #
    # min_confidence was originally 70/65/60/55 and was UNREACHABLE: weighted_confidence
    # is a weighted mean of components whose own ceilings sit far below 100, so the
    # quantity being tested never approached the numbers it was tested against. It fired
    # on 0 of 80 observations -- not strict, impossible -- and that alone is why the app
    # answered HOLD for every symbol on every timeframe.
    #
    # Re-measured 2026-09-07 for the 15m/1h/4h/1d option set (20 symbols x 4 timeframes):
    # min 17 / p50 32 / p75 34 / p90 40 / max 52. The slower horizons carry HIGHER
    # confidence -- daily momentum is steadier than 15-minute momentum -- so a flat gate
    # concentrates every firing in 1d. These are graded to match: 36/36/37/38 fires on
    # 8.8% of the sample, spread 2/1/2/2 across the four timeframes instead of 2/2/3/6.
    #
    # CALIBRATION, NOT INFLATION: score thresholds and component weights are untouched, and
    # a gate firing under 10% of the time is still demanding. Re-run
    # tools/calibrate_confidence_gate.py when the volatility regime shifts -- this sample
    # is one quiet-tape snapshot (VIX ~14.5).
    timeframe_configs = {
        "15m": {
            "threshold": 0.30,
            "min_confidence": 36,
            "hold_time": "10-30 minutes",
            "position_multiplier": 1.0
        },
        "1h": {
            "threshold": 0.35,
            "min_confidence": 36,
            "hold_time": "30-90 minutes",
            "position_multiplier": 1.2
        },
        "4h": {
            "threshold": 0.38,
            "min_confidence": 37,
            "hold_time": "half a session to a day",
            "position_multiplier": 1.3
        },
        "1d": {
            "threshold": 0.42,
            "min_confidence": 38,
            "hold_time": "1-5 days",
            "position_multiplier": 1.5
        }
    }

    config = timeframe_configs.get(timeframe, timeframe_configs["15m"])
    
    # STRATEGY MODE ADJUSTMENTS
    # "scalp" was REMOVED as an option on 2026-09-07. Its entire effect was
    # `config["threshold"] *= 0.8`, and measured across 36 symbol x timeframe rows it was
    # byte-identical to "momentum" in every one: the score is either far past the threshold
    # or nowhere near it, so an 0.8x multiplier never changed an outcome. It was also not a
    # strategy but a hold-time preference. An API caller passing scalp now falls through to
    # plain momentum rather than being rejected.
    gap_note = None
    reversal_note = None
    if strategy_mode == "gap":
        # Gated on a MEASURED overnight gap, not the TFT's gap_probability -- that value
        # sits at ~67.1 for every symbol and never cleared its own >70 test, so this
        # branch never once executed. 0.5% is the floor for "a gap rather than noise".
        # When no gap qualifies the mode now SAYS SO in the reasoning; the old failure was
        # that it did nothing silently, which is indistinguishable from working.
        if gap_pct is None:
            gap_note = "Gap mode: no daily history to measure a gap"
        elif abs(gap_pct) >= 0.5:
            config["min_confidence"] -= 10
            config["hold_time"] = "Gap fill / continuation"
            gap_note = f"Gap mode: {gap_pct:+.2f}% overnight gap qualifies"
        else:
            gap_note = f"Gap mode: {gap_pct:+.2f}% overnight, below the 0.5% floor"
    
    # Calculate weighted score
    if momentum_strategy in ["AGGRESSIVE_SCALP", "STANDARD_MOMENTUM"]:
        # conviction is a MAGNITUDE and is never negative, so this branch used to make
        # `weighted_score < -threshold` unreachable: it could emit CALLS or HOLD and never
        # PUTS, for any input. The engine now carries a signed direction, and the sign is
        # applied here. A strong move with no agreed direction scores 0 and falls to HOLD,
        # which is the honest reading of "something is happening but not which way".
        _dir = momentum_master.get("direction", "NEUTRAL")
        _sign = 1.0 if _dir == "BULLISH" else -1.0 if _dir == "BEARISH" else 0.0
        weighted_score = momentum_master.get("conviction", 0) * _sign
        weighted_confidence = momentum_confidence
    else:
        # The TFT term was REMOVED from this weighting on 2026-09-06, after measuring it.
        #
        # Feeding the NVDA checkpoint six different symbols' full price histories moved
        # gap_probability by 0.10 on a 0-100 scale, and no class probability deviated from
        # 1/3 by more than 0.0077. The model does not respond to its input; it has
        # collapsed to the prior. Walked over the last 120 sessions it emitted ONE constant
        # direction per symbol (NVDA: DOWN 119/119, SPY and QQQ: UP 119/119) and its hit
        # rate equalled the majority-class baseline to the decimal -- 54.6% vs 54.6% on both
        # SPY and QQQ, and 48.7% vs 51.3% on NVDA, i.e. below it.
        #
        # Since signal_scores["tft"] is +-0.7 by direction, a constant direction made this a
        # FIXED per-symbol bias of +-0.105 on weighted_score. It looked like an input and
        # was an offset -- the same defect as the two dead confidence terms above, except
        # here the dead thing is a model rather than a misread key.
        #
        # The checkpoint's own scaler proves why: scaler_static has all scale_ == 1.0, so
        # the static branch saw ZERO variance in training (the repo README's "trained on a
        # constant placeholder", confirmed from the artifact).
        #
        # tft_prediction is still COMPUTED and returned in the response -- it is honest
        # diagnostic output and the README documents its weakness -- it simply no longer
        # votes. The remaining three weights are renormalised to sum to 1.0 so the score
        # keeps its scale; the 0.3 threshold below is unchanged and still means the same
        # thing.
        weights = {
            "momentum": 0.4 / 0.85,      # 0.4706
            "llm": 0.25 / 0.85,          # 0.2941
            "sentiment": 0.2 / 0.85,     # 0.2353
        }
        
        signal_scores = {}
        signal_scores["momentum"] = 1.0 if momentum_signal == "CALLS" else -1.0 if momentum_signal == "PUTS" else 0.0
        signal_scores["llm"] = 1.0 if llm_signal == "CALLS" else -1.0 if llm_signal == "PUTS" else 0.0
        signal_scores["sentiment"] = np.clip(sentiment_composite, -1, 1)
        # no signal_scores["tft"]: see the note on `weights` above
        
        weighted_score = sum(signal_scores[k] * weights[k] for k in weights)

        # Two of the three terms below were STRUCTURALLY DEAD, which is why every symbol
        # rendered inside a 15-31 band even though the underlying analysis differs by 3x
        # (measured: TSLA daily momentum 0.638 vs SPY 0.183).
        #
        #   * llm_analysis.get("conviction", 50) -- on CPU the LLM ensemble is skipped and
        #     the rule-based fallback generate_enhanced_llm_signal() returns "confidence",
        #     not "conviction": 40 base, 65 on momentum, minus 10 on high VIX. None of that
        #     was ever read. The .get() fell through to the literal 50 every time, so this
        #     contributed a CONSTANT 15 to every score.
        #   * (sentiment == "HIGH") * 80 -- _calculate_confidence returns HIGH only when
        #     std_dev < 0.2 AND mean_abs > 0.3. Headline sentiment clusters near 0 and 1,
        #     so the std_dev gate effectively never passes: a CONSTANT 0.
        #
        # 60% of the weight was therefore fixed, and only momentum_confidence * 0.4 could
        # move. Both are now read honestly, and MEDIUM is graded rather than discarded.
        #
        # The WEIGHTS ARE DELIBERATELY UNCHANGED. Widening the visible spread by reweighting
        # would be making the demo look livelier, not making it correct.
        llm_conf = llm_analysis.get("conviction")
        if llm_conf is None:
            llm_conf = llm_analysis.get("confidence", 50)

        # A component with NO READING is now EXCLUDED and the remaining weights are
        # renormalised -- it is not scored as "0% confident".
        #
        # sentiment confidence is LOW unless std_dev < 0.2 AND mean_abs > 0.3, which
        # headline sentiment essentially never satisfies. Averaging that in as a zero
        # spent 30% of the confidence budget on a permanent 0, capping weighted_confidence
        # at 70 even for a flawless setup and pinning the observed output in a 23-27 band.
        # This is the same correction already applied to the TFT weights above: a term
        # that has nothing to say should abstain, not vote zero.
        components = [(momentum_confidence, 0.4), (llm_conf, 0.3)]
        sent_conf = {"HIGH": 80, "MEDIUM": 40}.get(
            sentiment_analysis.get("confidence", "LOW"))
        if sent_conf is not None:
            components.append((sent_conf, 0.3))

        _wsum = sum(w for _, w in components)
        weighted_confidence = sum(v * w for v, w in components) / _wsum
    
    # REVERSAL: the counter-hypothesis, and the reason this dropdown is worth having.
    #
    # "momentum" and the removed "scalp" only ever differed by a threshold multiplier, so
    # the strategy selector could not express a genuinely different reading of the same
    # tape. Reversal can: it FADES an extended move rather than following it, so on the
    # same bars it will frequently take the opposite side of momentum.
    #
    # It fires only when there is something to fade -- an RSI extreme that the move is
    # still running into. Absent that, the score is zeroed and the mode says so, rather
    # than silently degrading into momentum (which is exactly how "gap" hid for so long).
    #
    # RSI, volume_exhaustion and near_resistance/near_support are all already computed by
    # identify_current_setup and were being discarded, the same as directional_bias was.
    if strategy_mode == "reversal":
        # UI timeframes are 1m/5m/15m/1h; the analysed bar sets are 5m/15m/hourly/daily.
        tf_key = {"15m": "15m", "1h": "hourly", "4h": "4h", "1d": "daily"}.get(timeframe, "15m")
        setup = (tech_setups or {}).get(tf_key, {})
        rsi = setup.get("rsi", 50) or 50
        confirmed = bool(setup.get("volume_exhaustion")) or (
            setup.get("near_resistance") if weighted_score > 0
            else setup.get("near_support"))

        if rsi >= 70 and weighted_score > 0:
            weighted_score = -weighted_score          # fade the overbought push
            reversal_note = f"Reversal: fading RSI {rsi:.0f} overbought"
        elif rsi <= 30 and weighted_score < 0:
            weighted_score = -weighted_score          # fade the oversold flush
            reversal_note = f"Reversal: fading RSI {rsi:.0f} oversold"
        else:
            weighted_score = 0.0
            reversal_note = f"Reversal: RSI {rsi:.0f}, no extension to fade"

        if reversal_note.startswith("Reversal: fading"):
            config["hold_time"] = "Until mean reversion"
            if confirmed:
                # exhaustion or an S/R level is corroboration, not the signal itself
                config["min_confidence"] -= 8
                reversal_note += " (confirmed)"

    # Generate final signal
    if weighted_score > config["threshold"] and weighted_confidence > config["min_confidence"]:
        final_signal = "CALLS"
        position_size = min(0.5, (weighted_confidence / 100) * config["position_multiplier"])
    elif weighted_score < -config["threshold"] and weighted_confidence > config["min_confidence"]:
        final_signal = "PUTS"
        position_size = min(0.5, (weighted_confidence / 100) * config["position_multiplier"])
    else:
        final_signal = "HOLD"
        position_size = 0.0
        config["hold_time"] = "Wait for better setup"
    
    # WHY it held. A bare "HOLD, 17%" is unreadable: the user cannot tell whether nothing
    # is happening, the timeframes disagree, or a gate is a hair away. Every one of those
    # has a different meaning and only the engine knows which applied, so it says so
    # rather than leaving the UI to re-derive it from parts it does not have.
    blocking = None
    if final_signal == "HOLD":
        _dirn = momentum_master.get("direction", "NEUTRAL")
        _agree = momentum_master.get("direction_agreement")
        _dscore = momentum_master.get("direction_score")
        if momentum_strategy == "WAIT" or momentum_master.get("signal") == "NO_MOMENTUM":
            blocking = ("No timeframe showed enough movement to act on "
                        "(momentum below the WAIT threshold).")
        elif _dirn == "NEUTRAL":
            # Only name the condition that ACTUALLY blocked. Listing a test that passed
            # ("timeframes 83% agreed vs 75% needed") next to one that failed reads as if
            # both were problems, and the reader cannot tell which to watch.
            bits = []
            if _dscore is not None and abs(_dscore) <= DIRECTION_DEADBAND:
                bits.append(f"direction {_dscore:+.2f}, needs +/-{DIRECTION_DEADBAND:.2f}")
            if _agree is not None and _agree < DIRECTION_AGREEMENT:
                bits.append(f"only {_agree*100:.0f}% of timeframes agreed, needs "
                            f"{DIRECTION_AGREEMENT*100:.0f}%")
            blocking = ("Movement, but no agreed direction: " + "; ".join(bits) + "."
                        if bits else "Movement, but the timeframes did not agree "
                                     "on a direction.")
        elif weighted_confidence <= config["min_confidence"]:
            blocking = (f"Direction is {_dirn.lower()}, but confidence "
                        f"{weighted_confidence:.0f} is under the {config['min_confidence']} "
                        f"gate for {timeframe}.")
        elif abs(weighted_score) <= config["threshold"]:
            blocking = (f"Direction is {_dirn.lower()}, but the score "
                        f"{weighted_score:+.2f} is under the "
                        f"+/-{config['threshold']:.2f} threshold for {timeframe}.")

    # Build reasoning
    reasoning = []
    reasoning.append(f"{strategy_mode.upper()} {timeframe}: {final_signal}")
    reasoning.append(f"Confidence: {weighted_confidence:.0f}%")
    
    if momentum_strategy != "WAIT":
        reasoning.append(f"Momentum: {momentum_strategy}")
    if abs(sentiment_composite) > 0.3:
        reasoning.append(f"Sentiment: {'Bullish' if sentiment_composite > 0 else 'Bearish'}")
    if reversal_note:
        reasoning.append(reversal_note)
    if gap_note:
        reasoning.append(gap_note)
    elif gap_pct is not None and abs(gap_pct) >= 0.5:
        reasoning.append(f"Overnight gap {gap_pct:+.2f}%")
    
    return {
        "signal": final_signal,
        "confidence": int(weighted_confidence),
        "reasoning": ". ".join(reasoning),
        "position_size": position_size,
        "timeframe": timeframe,
        "hold_time": config["hold_time"],
        "weighted_score": weighted_score,
        "strategy_mode": strategy_mode,
        "momentum_strategy": momentum_strategy,
        "overnight_gap_pct": gap_pct,
        # The gates this verdict was measured against, returned so the UI can show the
        # margin rather than an unexplained number.
        "min_confidence": config["min_confidence"],
        "score_threshold": config["threshold"],
        "blocking_reason": blocking,
        "direction": momentum_master.get("direction"),
        "direction_agreement": momentum_master.get("direction_agreement"),
    }

def _generate_options_strategy(master_signal: Dict, momentum_analysis: Dict, 
                               alt_data: Dict, timeframe: str = "15m", 
                               strategy_mode: str = "momentum") -> Dict:
    """Generate options strategy with timeframe awareness"""
    
    signal = master_signal["signal"]
    confidence = master_signal["confidence"]
    vix_level = alt_data.get("vix_level", 20)
    
    if signal == "HOLD":
        return {
            "strategy": "WAIT",
            "reasoning": "No clear directional bias",
            "contracts": [],
            "risk_management": "Wait for better setup"
        }
    
    # TIMEFRAME-SPECIFIC STRATEGIES
    # Two of these branches were DEAD when the options set changed underneath them:
    # the 1m/5m + "scalp" branch survived after all three of those options were removed,
    # and "15m and confidence > 70" tested a value whose measured ceiling is 52 -- the
    # same unreachable-gate defect that was fixed in the signal path but missed here.
    # Branches now match the shipped timeframes (15m/1h/4h/1d) and reachable thresholds.
    if timeframe == "15m" and confidence >= 40:
        strategy = {
            "strategy": "INTRADAY_15M",
            "reasoning": f"{timeframe} scalp: {signal} with {confidence}% confidence",
            "contracts": [
                {
                    "type": "CALL" if signal == "CALLS" else "PUT",
                    "strike": "ATM",
                    "quantity": min(int(confidence / 8), 15),
                    "dte": 0,
                    "target_profit": 20,
                    "stop_loss": 10
                }
            ],
            "max_hold_time": f"{timeframe} bars (max 5 minutes)",
            "risk_management": "Ultra-tight stops, quick exits"
        }
        
    elif timeframe == "1h":
        strategy = {
            "strategy": "HOURLY_MOMENTUM",
            "reasoning": f"15-minute momentum {signal} play, {confidence}% confidence",
            "contracts": [
                {
                    "type": "CALL" if signal == "CALLS" else "PUT",  
                    "strike": "1% ITM",
                    "quantity": min(int(confidence / 12), 8),
                    "dte": 1,
                    "target_profit": 40,
                    "stop_loss": 20
                }
            ],
            "max_hold_time": "30 minutes",
            "risk_management": "Standard momentum stops"
        }
        
    elif timeframe in ("4h", "1d"):
        strategy = {
            "strategy": "SWING_SPREAD",
            "reasoning": f"Hourly swing {signal}, {confidence}% confidence",
            "contracts": [
                {
                    "type": "CALL_SPREAD" if signal == "CALLS" else "PUT_SPREAD",
                    "long_strike": "ATM",
                    "short_strike": "3% OTM",
                    "quantity": min(int(confidence / 15), 5),
                    "dte": 3,
                    "target_profit": 35,
                    "stop_loss": 25
                }
            ],
            "max_hold_time": "2-4 hours",
            "risk_management": "Defined risk spreads"
        }
        
    else:
        strategy = {
            "strategy": "CONSERVATIVE",
            "reasoning": f"Lower conviction {signal}, using conservative approach",
            "contracts": [
                {
                    "type": "CALL_SPREAD" if signal == "CALLS" else "PUT_SPREAD",
                    "long_strike": "ATM",
                    "short_strike": "5% OTM",
                    "quantity": 3,
                    "dte": 7,
                    "target_profit": 25,
                    "stop_loss": 20
                }
            ],
            "max_hold_time": "End of day",
            "risk_management": "Limited risk, defined reward"
        }
    
    # VIX adjustments
    if vix_level > 30:
        strategy["reasoning"] += f". High VIX ({vix_level}) - reduced size"
        for contract in strategy["contracts"]:
            contract["quantity"] = max(1, contract["quantity"] // 2)
    
    return strategy

@app.post("/backtest/enhanced/")
async def enhanced_backtest(
    symbol: str = Query(..., description="Stock symbol"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    strategy_mode: str = Query("momentum", description="Strategy: momentum, gap, reversal"),
    initial_capital: float = Query(100000, description="Initial capital")
):
    """Enhanced backtesting with momentum strategies"""
    try:
        return {
            "status": "success",
            "message": "Enhanced backtesting ready",
            "features": [
                "Multi-timeframe momentum analysis",
                "LLM ensemble signal validation", 
                "Options strategy backtesting",
                "Risk-adjusted performance metrics",
                "Slippage and commission modeling"
            ]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed system health check"""
    import torch
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",
        "components": {}
    }
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated(0) / 1e9
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        health_status["components"]["gpu"] = {
            "status": "available",
            "device": torch.cuda.get_device_name(0),
            "memory_used_gb": gpu_memory_used,
            "memory_total_gb": gpu_memory_total,
            "utilization": f"{gpu_memory_used/gpu_memory_total*100:.1f}%"
        }
    else:
        health_status["components"]["gpu"] = {"status": "not_available"}
    
    # Check model status
    health_status["components"]["sentiment_models"] = {
        "loaded": len(sentiment_analyzer.models),
        "status": "ready" if sentiment_analyzer.models else "not_loaded"
    }
    
    health_status["components"]["llm_models"] = {
        "loaded": len(llm_engine.models),
        "status": "ready" if llm_engine.models else "not_loaded"
    }
    
    health_status["components"]["tft_model"] = {
        "status": "trained" if tft_predictor.is_trained else "not_trained"
    }
    
    health_status["components"]["agent"] = {
        "status": "initialized" if trading_agent else "not_initialized"
    }
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)