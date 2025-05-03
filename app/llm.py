# In app/llm.py
import os, textwrap, torch
from transformers import AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM
from peft import PeftModel

USE = os.getenv("LLM_PROVIDER", "local")

if USE == "local":
    BASE_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    ADAPTER_ID = "bryandts/Finance-Alpaca-Llama-3.2-3B-Instruct-bnb-4bit"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        quantization_config=bnb_cfg
    )
    model = PeftModel.from_pretrained(model, ADAPTER_ID)

    def _run(prompt: str) -> str:
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=8)
        return tokenizer.decode(out[0], skip_special_tokens=True)

else:
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _run(prompt: str) -> str:
        r = _client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":"One-word answer: UP, DOWN or FLAT."},
                {"role":"user","content":prompt}],
            temperature=0.1, max_tokens=5)
        return r.choices[0].message.content

# ADD THE PREDICT FUNCTION HERE (OUTSIDE THE IF/ELSE BLOCKS)
def predict(symbol: str, last_close: float, indicators: dict, 
            news_sentiment: float, reddit_sentiment: float,
            news_count: int, reddit_count: int) -> str | None:
    """Predict the opening gap direction for the next trading day."""
    
    print(f"LLM PREDICT - Starting with inputs:")
    print(f"  symbol: {symbol} ({type(symbol)})")
    print(f"  last_close: {last_close} ({type(last_close)})")
    print(f"  indicators: {type(indicators)}")
    print(f"  news_sentiment: {news_sentiment} ({type(news_sentiment)})")
    print(f"  reddit_sentiment: {reddit_sentiment} ({type(reddit_sentiment)})")
    print(f"  news_count: {news_count} ({type(news_count)})")
    print(f"  reddit_count: {reddit_count} ({type(reddit_count)})")
    
    # Check provider being used
    print(f"LLM PROVIDER: {USE}")
    
    
    try:
        symbol = str(symbol)
        last_close = float(last_close)
        news_sentiment = float(news_sentiment)
        reddit_sentiment = float(reddit_sentiment)
        news_count = int(news_count)
        reddit_count = int(reddit_count)
    except Exception as e:
        print(f"Type conversion error in predict: {e}")
        return None

    def format_indicator(k, v):
        if isinstance(v, (int, float)):
            if 'Volume' in k:
                return f"{v:,.0f}"
            else:
                return f"{v:.2f}"
        return "N/A"
    
    indicator_text = "\n".join([f"- {k}: {format_indicator(k, v)}" for k, v in indicators.items()])
    
    prompt = f"""
    Analyze the financial data for {symbol} to predict the opening gap direction:
    
    Last close: ${last_close:.2f}
    
    Technical Indicators:
    {indicator_text}
    
    Sentiment Analysis:
    - News Sentiment ({news_count} items): {news_sentiment:.2f}
    - Reddit Sentiment ({reddit_count} posts): {reddit_sentiment:.2f}
    
    Will the stock gap UP, DOWN, or FLAT at next market open?
    Answer with a single word (UP/DOWN/FLAT).
    """
    
    try:
        result = _run(prompt).strip().upper()
        valid_predictions = {"UP", "DOWN", "FLAT"}
        if result in valid_predictions:
            return result
        elif "UP" in result:
            return "UP"
        elif "DOWN" in result:
            return "DOWN"
        elif "FLAT" in result or "NEUTRAL" in result:
            return "FLAT"
        else:
            return None
    except Exception as e:
        print(f"Error in LLM prediction: {e}")
        return None