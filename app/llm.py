# In app/llm.py
import os, textwrap, torch
from transformers import AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM
from peft import PeftModel

USE = os.getenv("LLM_PROVIDER", "local")

# Signal thresholds
ADX_THRESHOLD = 25.0  # Lower from 30 for more signals
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
CONFIDENCE_THRESHOLD_FOR_ACTION = 25.0 
STRONG_NEWS_ECO_THRESHOLD = 10.0
CONFIDENCE_MIN = 15.0  # Lower from 30 for more actionable signals

if USE == "local":
    BASE_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    ADAPTER_ID = "bryandts/Finance-Alpaca-Llama-3.2-3B-Instruct-bnb-4bit"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("Loading Llama model for local predictions...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="auto",
            quantization_config=bnb_cfg
        )
        model = PeftModel.from_pretrained(model, ADAPTER_ID)
        print("Llama model loaded successfully.")
    except Exception as e:
        print(f"Error loading Llama model: {e}")
        model = None
        tokenizer = None

    def _run(prompt: str) -> str:
        if model is None or tokenizer is None:
            print("Local LLM not available, returning default prediction")
            return "ERROR"
        try:
            ids = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**ids, max_new_tokens=8)
            return tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error running local LLM prediction: {e}")
            return "ERROR"

else:
    from openai import OpenAI
    
    print("Setting up OpenAI for predictions...")
    try:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("OpenAI client initialized successfully.")
    except Exception as e:
        print(f"Error setting up OpenAI client: {e}")
        _client = None
    
    def _run(prompt: str) -> str:
        if _client is None:
            print("OpenAI client not available, returning default prediction")
            return "ERROR"
        try:
            r = _client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role":"system","content":"One-word answer: UP, DOWN or FLAT."},
                    {"role":"user","content":prompt}],
                temperature=0.1, max_tokens=5)
            return r.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "ERROR"

def calculate_dynamic_threshold(daily_atr: float, last_close: float, base_threshold: float = 20.0, volatility_factor: float = 0.5, min_thresh: float = 15.0, max_thresh: float = 40.0) -> float:
    """Calculates a dynamic confidence threshold based on volatility (ATR % of Close)."""
    if last_close == 0: # Avoid division by zero
        return base_threshold
    atr_percentage = (daily_atr / last_close) * 100
    # Example: Scale threshold slightly with volatility percentage
    # Adjust the scaling (e.g., volatility_factor) based on backtesting
    dynamic_threshold = base_threshold + (atr_percentage * volatility_factor)
    # Clamp threshold within min/max bounds
    return max(min_thresh, min(max_thresh, dynamic_threshold))


def predict_gap_trading_signal(technical_data: dict, news_sentiment_avg: float, reddit_sentiment_avg: float, economic_impact: dict, gap_analysis: dict) -> dict:
    # Extract key indicators for daily
    adx = technical_data.get('ADX', 0)
    dmi_plus = technical_data.get('DMI_Plus', 0)
    dmi_minus = technical_data.get('DMI_Minus', 0)
    rsi = technical_data.get('RSI_14', 50)
    macdh = technical_data.get('MACDh_12_26_9', 0)
    
    # Extract multi-timeframe data from gap_analysis
    daily_setup = gap_analysis.get('current_setup', {}).get('daily', {})
    hourly_setup = gap_analysis.get('current_setup', {}).get('hourly', {})
    fifteen_min_setup = gap_analysis.get('current_setup', {}).get('15m', {})
    
    daily_rsi = daily_setup.get('rsi', 50)
    hourly_rsi = hourly_setup.get('rsi', 50)
    fifteen_min_rsi = fifteen_min_setup.get('rsi', 50)
    
    signal = "WAIT"  # Default is no trade
    confidence = 0
    reasoning = []
    technical_triggers = []
    
    # Daily trend analysis
    if adx > 25 or (abs(dmi_plus - dmi_minus) > 15):
        if dmi_plus > dmi_minus:
            confidence += 25
            reasoning.append(f"[Daily] Bullish DMI. ADX:{adx:.1f}")
            signal = "LONG"
            if rsi > 70:
                confidence -= 15
                reasoning.append("Caution: RSI overbought above 70")
                signal = "WAIT"  # Avoid chasing overbought conditions
        elif dmi_minus > dmi_plus:
            confidence += 25
            reasoning.append(f"[Daily] Bearish DMI. ADX:{adx:.1f}")
            signal = "SHORT"
            if rsi < 30:
                confidence -= 15
                reasoning.append("Caution: RSI oversold below 30")
                signal = "WAIT"  # Avoid shorting oversold conditions
    
    # Adjust signal based on shorter timeframes (hourly and 15m)
    if hourly_rsi > 70 or fifteen_min_rsi > 70:
        reasoning.append(f"[Hourly/15m] RSI Overbought: Hourly={hourly_rsi:.1f}, 15m={fifteen_min_rsi:.1f}")
        if signal == "LONG":
            confidence -= 10
            if confidence < 20:
                signal = "WAIT"
    elif hourly_rsi < 30 or fifteen_min_rsi < 30:
        reasoning.append(f"[Hourly/15m] RSI Oversold: Hourly={hourly_rsi:.1f}, 15m={fifteen_min_rsi:.1f}")
        if signal == "SHORT":
            confidence -= 10
            if confidence < 20:
                signal = "WAIT"
    
    # Add MACD confirmation
    if macdh > 2.0 and signal == "LONG":
        confidence += 5
        reasoning.append(f"Bullish MACD Histogram: {macdh:.2f}")
        technical_triggers.append(f"Daily MACDh: {macdh:.2f}")
    elif macdh < -2.0 and signal == "SHORT":
        confidence += 5
        reasoning.append(f"Bearish MACD Histogram: {macdh:.2f}")
        technical_triggers.append(f"Daily MACDh: {macdh:.2f}")
    
    # Economic impact and sentiment modifiers
    economic_score = economic_impact.get('impact_score', 0)
    if abs(economic_score) > 5:
        if economic_score > 5:
            if signal == "LONG":
                confidence += 20
                reasoning.append(f"Positive economic news impact (Score: {economic_score:.2f}).")
                reasoning.append("Economic news reinforces bullish signal.")
            elif signal == "SHORT":
                confidence -= 10
                reasoning.append(f"Positive economic news contradicts bearish signal")
                if confidence < 30:
                    signal = "WAIT"
                    reasoning.append("Mixed signals suggest caution")
        elif economic_score < -5:
            if signal == "SHORT":
                confidence += 20
                reasoning.append(f"Negative economic news impact (Score: {economic_score:.2f}).")
                reasoning.append("Economic news reinforces bearish signal.")
            elif signal == "LONG":
                confidence -= 10
                reasoning.append(f"Negative economic news contradicts bullish signal")
                if confidence < 30:
                    signal = "WAIT"
                    reasoning.append("Mixed signals suggest caution")
    
    # Sentiment from Reddit
    if reddit_sentiment_avg < -0.05:
        reasoning.append(f"Caution: Negative Reddit sentiment ({reddit_sentiment_avg:.2f})")
        confidence -= 5
    elif reddit_sentiment_avg > 0.05:
        reasoning.append(f"Supportive Reddit sentiment ({reddit_sentiment_avg:.2f})")
        confidence += 5
    
    # Gap pattern analysis
    if daily_setup and daily_setup.get('gap_risk') == "high":
        if daily_setup.get('direction') == "up" and signal == "LONG":
            confidence -= 15
            reasoning.append(f"Caution: {daily_setup.get('streak')}-day up streak increases gap down risk")
        elif daily_setup.get('direction') == "down" and signal == "SHORT":
            confidence -= 15
            reasoning.append(f"Caution: {daily_setup.get('streak')}-day down streak increases gap up risk")
    
    # Final confidence adjustments
    if confidence > 100:
        confidence = 100
    elif confidence < 0:
        confidence = 0
        signal = "WAIT"
    
    # Add multi-timeframe setups to reasoning
    reasoning.append(f"[Daily Setup] Dir: {daily_setup.get('direction', 'N/A')}, Streak: {daily_setup.get('streak', 0)}, RSI: {daily_rsi:.2f}, ADX: {daily_setup.get('adx', 0):.1f}, GapRisk: {daily_setup.get('gap_risk', 'N/A')}")
    reasoning.append(f"[Hourly Setup] Dir: {hourly_setup.get('direction', 'N/A')}, Streak: {hourly_setup.get('streak', 0)}, RSI: {hourly_rsi:.2f}, ADX: {hourly_setup.get('adx', 0):.1f}, GapRisk: {hourly_setup.get('gap_risk', 'N/A')}")
    reasoning.append(f"[15min Setup] Dir: {fifteen_min_setup.get('direction', 'N/A')}, Streak: {fifteen_min_setup.get('streak', 0)}, RSI: {fifteen_min_rsi:.2f}, ADX: {fifteen_min_setup.get('adx', 0):.1f}, GapRisk: {fifteen_min_setup.get('gap_risk', 'N/A')}")
    
    # Add technical triggers
    technical_triggers.extend([
        f"Daily ADX: {adx:.2f}",
        f"Daily +DI: {dmi_plus:.2f}",
        f"Daily -DI: {dmi_minus:.2f}",
        f"Daily RSI: {rsi:.2f}",
        f"15m ADX: {fifteen_min_setup.get('adx', 0):.2f}",
        f"15m RSI: {fifteen_min_rsi:.2f}"
    ])
    
    return {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
        "technical_triggers": technical_triggers,
        "economic_score": economic_score,
        "gap_setup": {
            "daily": daily_setup,
            "hourly": hourly_setup,
            "15m": fifteen_min_setup
        },
        "position_size": min(confidence / 100, 0.5) if confidence > 0 else 0.0
    }


def predict_multi_timeframe_signal(
        tech_daily: dict, setup_daily: dict,
        tech_hourly: dict, setup_hourly: dict,
        tech_15m: dict, setup_15m: dict,
        news_sentiment_avg: float,
        reddit_sentiment_avg: float,
        economic_impact: dict,
        tft_prediction: dict = None,      # ADD THIS
        alternative_data: dict = None     # ADD THIS
        ) -> dict:
    """
    Generates a trading signal by combining analysis from multiple timeframes,
    news, sentiment, economic impact, TFT predictions, and alternative data.
    """
    reasoning = []
    technical_triggers = []

    # Extract key indicators from each timeframe
    daily_adx = tech_daily.get('ADX', 0.0)
    daily_dmi_plus = tech_daily.get('DMI_Plus', 0.0)
    daily_dmi_minus = tech_daily.get('DMI_Minus', 0.0)
    daily_rsi = tech_daily.get('RSI_14', 50.0)

    hourly_adx = tech_hourly.get('ADX', 0.0)
    hourly_dmi_plus = tech_hourly.get('DMI_Plus', 0.0)
    hourly_dmi_minus = tech_hourly.get('DMI_Minus', 0.0)
    hourly_rsi = tech_hourly.get('RSI_14', 50.0)

    fifteen_min_adx = tech_15m.get('ADX', 0.0)
    fifteen_min_dmi_plus = tech_15m.get('DMI_Plus', 0.0)
    fifteen_min_dmi_minus = tech_15m.get('DMI_Minus', 0.0)
    fifteen_min_rsi = tech_15m.get('RSI_14', 50.0)

    # Determine signals for each timeframe
    daily_signal = "LONG" if daily_dmi_plus > daily_dmi_minus else "SHORT"
    hourly_signal = "LONG" if hourly_dmi_plus > hourly_dmi_minus else "SHORT"
    fifteen_min_signal = "LONG" if fifteen_min_dmi_plus > fifteen_min_dmi_minus else "SHORT"

    # Add reasoning for each timeframe
    reasoning.append(f"[Daily] {'Bullish' if daily_signal == 'LONG' else 'Bearish'} DMI. ADX:{daily_adx:.1f}")
    reasoning.append(f"[Hourly] {'Bullish' if hourly_signal == 'LONG' else 'Bearish'} DMI. ADX:{hourly_adx:.1f}")
    reasoning.append(f"[15min] {'Bullish' if fifteen_min_signal == 'LONG' else 'Bearish'} DMI. ADX:{fifteen_min_adx:.1f}")

    # Add technical triggers
    technical_triggers.extend([
        f"Daily ADX: {daily_adx:.2f}",
        f"Daily +DI: {daily_dmi_plus:.2f}",
        f"Daily -DI: {daily_dmi_minus:.2f}",
        f"Daily RSI: {daily_rsi:.2f}",
        f"15m ADX: {fifteen_min_adx:.2f}",
        f"15m RSI: {fifteen_min_rsi:.2f}"
    ])

    # Determine overall signal (majority vote)
    signals = [daily_signal, hourly_signal, fifteen_min_signal]
    long_votes = signals.count("LONG")
    short_votes = signals.count("SHORT")

    if long_votes > short_votes:
        base_signal = "LONG"
    elif short_votes > long_votes:
        base_signal = "SHORT"
    else:
        base_signal = "HOLD"  # Tie

    # Calculate base confidence
    confidence = 30.0  # Base confidence

    # ADX strength bonus
    avg_adx = (daily_adx + hourly_adx + fifteen_min_adx) / 3
    if avg_adx > 25:
        confidence += 10
    elif avg_adx > 35:
        confidence += 15

    # Signal alignment bonus
    if long_votes == 3 or short_votes == 3:
        confidence += 20  # All timeframes agree
    elif long_votes == 2 or short_votes == 2:
        confidence += 10  # Majority agreement

    # ===== NEW: ADD TFT PREDICTION LOGIC =====
    if tft_prediction:
        tft_gap_prob = tft_prediction.get('gap_probability', 50)
        tft_direction = tft_prediction.get('expected_direction', 'FLAT')
        tft_confidence = tft_prediction.get('confidence', 'LOW')
        
        # Convert TFT direction to signal format
        tft_signal = "LONG" if tft_direction == "UP" else ("SHORT" if tft_direction == "DOWN" else "HOLD")
        
        # Boost confidence if TFT agrees with technical signal
        if tft_signal == base_signal:
            if tft_confidence == 'HIGH':
                confidence += 15
                reasoning.append(f"TFT HIGH confidence reinforces {base_signal} signal (+15 confidence)")
            elif tft_confidence == 'MEDIUM':
                confidence += 10
                reasoning.append(f"TFT MEDIUM confidence supports {base_signal} signal (+10 confidence)")
        
        # Reduce confidence if TFT contradicts
        elif tft_signal != base_signal and tft_signal != 'HOLD':
            confidence -= 10
            reasoning.append(f"TFT predicts {tft_direction} vs technical {base_signal} (-10 confidence)")
        
        # High gap probability increases overall confidence
        if tft_gap_prob > 70:
            confidence += 5
            reasoning.append(f"TFT high gap probability {tft_gap_prob}% (+5 confidence)")
        
        reasoning.append(f"TFT: {tft_gap_prob}% gap probability, direction {tft_direction}, confidence {tft_confidence}")

    # ===== NEW: ADD ALTERNATIVE DATA LOGIC =====
    if alternative_data:
        # VIX regime adjustment
        vix_regime = alternative_data.get('vix_regime_numeric', 2)
        vix_level = alternative_data.get('vix_level', 20)
        
        if vix_regime >= 3:  # HIGH or EXTREME volatility
            confidence -= 5
            reasoning.append(f"High VIX regime {vix_regime} (level {vix_level}) reduces confidence (-5)")
        elif vix_regime == 1:  # LOW volatility
            confidence += 3
            reasoning.append(f"Low VIX regime supports signal (+3 confidence)")
        
        # Sector rotation signal
        sector_signal = alternative_data.get('sector_rotation_signal', 0)
        if abs(sector_signal) > 0.1:
            if (sector_signal > 0 and base_signal == "LONG") or (sector_signal < 0 and base_signal == "SHORT"):
                confidence += 8
                reasoning.append(f"Sector rotation {sector_signal:.3f} supports {base_signal} (+8 confidence)")
            else:
                confidence -= 5
                reasoning.append(f"Sector rotation {sector_signal:.3f} contradicts {base_signal} (-5 confidence)")
        
        # Put/Call ratio
        put_call = alternative_data.get('put_call_ratio', 1.0)
        if put_call > 1.3 and base_signal == "SHORT":
            confidence += 5
            reasoning.append(f"High put/call ratio {put_call} supports SHORT (+5 confidence)")
        elif put_call < 0.7 and base_signal == "LONG":
            confidence += 5
            reasoning.append(f"Low put/call ratio {put_call} supports LONG (+5 confidence)")
        
        # Options sentiment
        options_sentiment = alternative_data.get('options_sentiment_numeric', 0)
        if (options_sentiment > 0 and base_signal == "LONG") or (options_sentiment < 0 and base_signal == "SHORT"):
            confidence += 3
            reasoning.append(f"Options sentiment {options_sentiment} aligns with {base_signal} (+3 confidence)")
        
        # Earnings impact
        earnings_impact = alternative_data.get('earnings_surprise_numeric', 0)
        if abs(earnings_impact) > 0:
            if (earnings_impact > 0 and base_signal == "LONG") or (earnings_impact < 0 and base_signal == "SHORT"):
                confidence += 7
                reasoning.append(f"Earnings momentum {earnings_impact} supports {base_signal} (+7 confidence)")
        
        reasoning.append(f"Alt Data: VIX={vix_level}, Sector={sector_signal:.3f}, P/C={put_call}, Beta={alternative_data.get('beta_estimate', 1.0)}")

    # Add gap setup information
    if setup_daily:
        reasoning.append(f"[Daily Setup] Dir: {setup_daily.get('direction', 'unknown')}, Streak: {setup_daily.get('streak', 0)}, RSI: {daily_rsi:.2f}, ADX: {daily_adx:.2f}, GapRisk: {setup_daily.get('gap_risk', 'unknown')}")
    
    if setup_hourly:
        reasoning.append(f"[Hourly Setup] Dir: {setup_hourly.get('direction', 'unknown')}, Streak: {setup_hourly.get('streak', 0)}, RSI: {hourly_rsi:.1f}, ADX: {hourly_adx:.2f}, GapRisk: {setup_hourly.get('gap_risk', 'unknown')}")
    
    if setup_15m:
        reasoning.append(f"[15min Setup] Dir: {setup_15m.get('direction', 'unknown')}, Streak: {setup_15m.get('streak', 0)}, RSI: {fifteen_min_rsi:.2f}, ADX: {fifteen_min_adx:.2f}, GapRisk: {setup_15m.get('gap_risk', 'unknown')}")

    # Economic impact
    economic_score = economic_impact.get('impact_score', 0.0)
    if abs(economic_score) > 1.0:
        if economic_score > 0 and base_signal == "LONG":
            confidence += 10
            reasoning.append(f"Positive economic impact {economic_score:.2f} supports LONG (+10)")
        elif economic_score < 0 and base_signal == "SHORT":
            confidence += 10
            reasoning.append(f"Negative economic impact {economic_score:.2f} supports SHORT (+10)")

    # Sentiment impact
    total_sentiment = news_sentiment_avg + reddit_sentiment_avg
    if abs(total_sentiment) > 0.1:
        if total_sentiment > 0 and base_signal == "LONG":
            confidence += 5
        elif total_sentiment < 0 and base_signal == "SHORT":
            confidence += 5

    # Dynamic threshold calculation
    close_price = tech_daily.get('Close', 100.0)
    atr = tech_daily.get('ATR_14', close_price * 0.02)
    dynamic_threshold = (atr / close_price) * 100 * 4.5
    reasoning.append(f"[Threshold] Dynamic threshold calculated: {dynamic_threshold:.2f} (based on ATR={atr:.2f}, Close={close_price:.2f})")

    # Cap confidence
    confidence = min(max(confidence, 0), 100)

    # Position sizing
    if confidence > 70:
        position_size = 0.5
    elif confidence > 50:
        position_size = 0.25
    else:
        position_size = 0.1

    return {
        "signal": base_signal,
        "confidence": round(confidence, 1),
        "reasoning": reasoning,
        "technical_triggers": technical_triggers,
        "economic_score": economic_score,
        "economic_components": None,
        "gap_setup": {
            "daily": setup_daily,
            "hourly": setup_hourly,
            "15m": setup_15m
        },
        "position_size": position_size
    }
 
# You might keep the old _run and LLM setup if you plan to use the LLM for other tasks
# or as part of a more complex ensemble prediction in the future.
# For now, the main predict function for the API will be the rule-based one.