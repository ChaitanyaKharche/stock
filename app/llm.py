# In app/llm.py
import os, textwrap, torch
from transformers import AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM
from peft import PeftModel

USE = os.getenv("LLM_PROVIDER", "local")

# Signal thresholds
ADX_THRESHOLD = 25.0  # Lower from 30 for more signals
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
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


def predict_gap_trading_signal(technical_data: dict, 
                               news_sentiment_avg: float, # Added for clarity
                               reddit_sentiment_avg: float, # Added for clarity
                               economic_impact: dict, 
                               gap_analysis: dict) -> dict:
    """Generate comprehensive trading signal based on our successful trading methods."""
    

    adx = technical_data.get('ADX', 0.0)
    dmi_plus = technical_data.get('DMI_Plus', technical_data.get('DMP_14', 0.0)) 
    dmi_minus = technical_data.get('DMI_Minus', technical_data.get('DMN_14', 0.0))
    rsi = technical_data.get('RSI_14', 50.0) # pandas_ta default is RSI_14
    macd_hist = technical_data.get('MACDh_12_26_9', 0.0) # This should now have a value


    economic_score = economic_impact.get('impact_score', 0.0)


    current_setup = gap_analysis.get('current_setup', {}) if gap_analysis else {} # Ensure gap_analysis is not None


    signal = "WAIT" 
    confidence = 0.0 
    reasoning = []


    # Strong trend conditions (ADX>30 with clear DMI direction)
    if adx > 25 or (abs(dmi_plus - dmi_minus) > 15):
        if dmi_plus > dmi_minus:
            signal = "LONG" # Initial signal based on DMI
            confidence += 25.0
            reasoning.append(f"Strong bullish trend (ADX: {adx:.2f}, +DMI: {dmi_plus:.2f} > -DMI: {dmi_minus:.2f})")
            if rsi > 75 and dmi_plus < dmi_minus: # Check for overbought *after* establishing trend
                confidence += 20
                reasoning.append("Strong SHORT signal: Overbought RSI + Bearish DMI crossover")
                # signal = "WAIT" # Reconsider if overbought always means WAIT. Sometimes strong trends stay overbought.
                                # Let's keep the LONG signal but with reduced confidence.
        elif dmi_minus > dmi_plus:
            signal = "SHORT" # Initial signal
            confidence += 25.0
            reasoning.append(f"Strong bearish trend (ADX: {adx:.2f}, -DMI: {dmi_minus:.2f} > +DMI: {dmi_plus:.2f})")
            if rsi < 25 and dmi_minus < dmi_plus: # Check for oversold
                confidence += 20
                reasoning.append("Strong LONG signal: Oversold RSI + Bullish DMI crossover")
                # signal = "WAIT" # Similar to above, keep SHORT but with reduced confidence.
    else:
        reasoning.append(f"Weak/No trend (ADX: {adx:.2f} <= 25). Seeking other confluences.")
        
        # ADD THIS CRUCIAL BLOCK - Handle extreme RSI values without strong ADX
        if rsi > 70:
            signal = "DOWN"  # Potential reversal from overbought
            confidence += 15.0
            reasoning.append(f"Extreme overbought RSI ({rsi:.2f} > 70) suggests potential reversal.")
        elif rsi < 30:
            signal = "UP"  # Potential reversal from oversold
            confidence += 15.0
            reasoning.append(f"Extreme oversold RSI ({rsi:.2f} < 30) suggests potential reversal.")


    # Economic impact modifiers
    # Only apply if a primary signal (LONG/SHORT) has been established by technicals
    if signal != "WAIT":
        if economic_score > 5: # Strong positive economic news
            reasoning.append(f"Positive economic news impact (Score: {economic_score:.2f}).")
            if signal == "LONG":
                confidence += 20.0
                reasoning.append("Economic news reinforces bullish signal.")
            elif signal == "SHORT": # Contradiction
                confidence -= 10.0 # Reduce confidence, potentially flip signal or move to WAIT
                reasoning.append("Economic news contradicts bearish signal.")
                if confidence < 20: # If confidence drops too low due to contradiction
                    # signal = "WAIT" # Option: revert to WAIT
                    reasoning.append("Mixed signals (technical vs economic) warrant caution.")
        elif economic_score < -5: # Strong negative economic news
            reasoning.append(f"Negative economic news impact (Score: {economic_score:.2f}).")
            if signal == "SHORT":
                confidence += 20.0
                reasoning.append("Economic news reinforces bearish signal.")
            elif signal == "LONG": # Contradiction
                confidence -= 10.0
                reasoning.append("Economic news contradicts bullish signal.")
                if confidence < 20:
                    # signal = "WAIT" # Option: revert to WAIT
                    reasoning.append("Mixed signals (technical vs economic) warrant caution.")
    
    # News Sentiment (FinBERT) and Reddit Sentiment Modifiers
    # These were in your original prompt structure but not explicitly in the python `predict_gap_trading_signal`
    # Adding them here as an example of how they could be integrated:
    if signal != "WAIT":
        combined_sentiment = (news_sentiment_avg + reddit_sentiment_avg) / 2
        if combined_sentiment > 0.25: # Arbitrary threshold for positive sentiment
            reasoning.append(f"Overall positive sentiment (News: {news_sentiment_avg:.2f}, Reddit: {reddit_sentiment_avg:.2f}).")
            if signal == "LONG": confidence += 10.0
            # else: confidence -= 5.0 # Mild contradiction
        elif combined_sentiment < -0.25: # Arbitrary threshold for negative sentiment
            reasoning.append(f"Overall negative sentiment (News: {news_sentiment_avg:.2f}, Reddit: {reddit_sentiment_avg:.2f}).")
            if signal == "SHORT": confidence += 10.0
            # else: confidence -= 5.0 # Mild contradiction


    # Gap pattern analysis from current_setup
    if current_setup and current_setup.get('gap_risk') == "high":
        streak_info = f"{current_setup.get('streak',0)}-day {current_setup.get('direction','')} streak"
        reasoning.append(f"High gap risk identified due to {streak_info} and RSI {current_setup.get('rsi', 50):.2f}.")
        if current_setup.get('direction') == "up" and signal == "LONG":
            confidence -= 20.0 # More significant reduction for high risk
            reasoning.append(f"Caution: {streak_info} increases gap down risk, contradicting LONG.")
        elif current_setup.get('direction') == "down" and signal == "SHORT":
            confidence -= 20.0
            reasoning.append(f"Caution: {streak_info} increases gap up risk, contradicting SHORT.")


    # Final confidence adjustments
    confidence = max(0.0, min(100.0, confidence)) # Clamp confidence between 0 and 100


    # If confidence is too low after all adjustments, revert to WAIT
    if confidence < CONFIDENCE_MIN and signal != "WAIT": # Arbitrary threshold for "too low"
        reasoning.append(f"Final confidence ({confidence:.2f} < 30) too low. Reverting to WAIT.")
        signal = "WAIT"
        # Reset confidence for WAIT or set to a neutral value like 0 or 50?
        # For clarity, if WAIT, confidence might not be as relevant or set to 0.
        # The original code does not reset confidence to 0 if signal becomes WAIT here.

    # If no strong technical signal initially, and other factors are neutral, it remains WAIT.
    if signal == "WAIT" and not any(s in r for r in reasoning for s in ["Strong bullish trend", "Strong bearish trend"]):
        if not reasoning: # If no reasoning was added at all
             reasoning.append("No clear trading signal based on available data.")
        confidence = 0.0 # Confidence for WAIT can be 0 or a specific value reflecting neutrality


    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "reasoning": reasoning,
        "technical_triggers": [ # Store the state of primary indicators for quick review
            f"ADX: {adx:.2f}",
            f"DMI_Plus: {dmi_plus:.2f}",
            f"DMI_Minus: {dmi_minus:.2f}",
            f"RSI_14: {rsi:.2f}",
            # f"MACD_Hist: {macd_hist:.2f}" # If you add it back
        ],
        "economic_score": economic_score, # Include for context
        "gap_setup": current_setup, # Include for context
        "position_size": (  # Line ~110
            1.0 if confidence > 60 else 
            0.5 if confidence > 40 else 
            0.25 if confidence > 20 else 
            0.0
        )
    }

# You might keep the old _run and LLM setup if you plan to use the LLM for other tasks
# or as part of a more complex ensemble prediction in the future.
# For now, the main predict function for the API will be the rule-based one.