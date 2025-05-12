from transformers import pipeline
import pandas as pd
import re
import datetime as dt

print("Loading FinBERT…")

try:
    _pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=0) 
    print("FinBERT model loaded successfully.")
    if _pipe:
        print(f"FinBERT pipeline device: {_pipe.device}")
except Exception as e:
    print(f"Error loading FinBERT pipeline: {e}")
    _pipe = None

def annotate(df: pd.DataFrame, col="headline"):
    """Analyze sentiment in a DataFrame column using the FinBERT model."""

    print(f"Starting sentiment analysis on {len(df)} rows with column '{col}'")

    if _pipe is None:
        print("FinBERT pipeline not loaded. Cannot perform sentiment annotation.")
        df["sentiment_score"] = 0.0
        return df
        
    if df.empty:
        print("WARNING: Empty DataFrame passed to sentiment.annotate()")
        df["sentiment_score"] = 0.0  # Add column even if empty for consistency
        return df
        
    if col not in df.columns:
        print(f"Warning: Column '{col}' not found in DataFrame for sentiment annotation. Available columns: {df.columns}")
        df["sentiment_score"] = 0.0
        return df

    try:
        # Debug info for sentiment analysis
        print(f"Running sentiment analysis on {len(df)} items in column '{col}'")
        
        # Ensure texts are strings and handle potential NaN/None values gracefully
        texts = [str(t) for t in df[col].fillna('').tolist() if t and len(str(t).strip()) > 5]  # Filter short/empty strings
        if not texts:
            print("No valid texts found for sentiment analysis")
            df["sentiment_score"] = 0.0
            return df
        
        print(f"Analyzing sentiment for {len(texts)} items (sample: '{texts[0][:50]}...')")
        
        # Process batches to avoid memory issues
        batch_size = min(16, len(texts))  # Smaller batch size for better stability
        results = []
        
        # Process in smaller batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:min(i+batch_size, len(texts))]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            batch_results = _pipe(batch, truncation=True)
            results.extend(batch_results)
        
        # Map results to sentiment scores
        sentiment_values = []
        for idx, s in enumerate(results):
            if idx < len(texts):  # Safety check
                if s['label'].lower() == 'positive':
                    sentiment_values.append(s['score'])
                elif s['label'].lower() == 'negative':
                    sentiment_values.append(-s['score'])
                else:  # neutral or other
                    sentiment_values.append(0.0)
        
        # Initialize sentiment_score column with zeros
        df["sentiment_score"] = 0.0
        
        # Map sentiment values to original texts
        valid_text_indices = [i for i, t in enumerate(df[col].fillna('').tolist()) if t and len(str(t).strip()) > 5]
        for i, score_idx in enumerate(valid_text_indices):
            if i < len(sentiment_values):
                df.iloc[score_idx, df.columns.get_loc("sentiment_score")] = sentiment_values[i]
        
        print(f"Sentiment analysis complete: Average score = {df['sentiment_score'].mean():.4f}")
        return df
    
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        df["sentiment_score"] = 0.0  # Ensure column exists even on error
        return df

def aggregate(df: pd.DataFrame): 
    """Calculate the average sentiment score and count from a DataFrame."""
    if df.empty or "sentiment_score" not in df.columns:
        print("Warning: Empty DataFrame or missing 'sentiment_score' column in aggregate")
        return 0.0, 0  # Return float for score, int for count
    
    # Ensure sentiment_score is numeric before mean, handle empty series after fillna if all were NaN
    numeric_scores = pd.to_numeric(df["sentiment_score"], errors='coerce').fillna(0.0)
    if numeric_scores.empty:
        return 0.0, 0
    
    avg = float(numeric_scores.mean())
    count = len(df)
    
    print(f"Aggregated sentiment: score={avg:.4f}, count={count}")
    print(f"Sentiment aggregation result: {avg:.4f} from {count} items")
    return avg, count

def analyze_economic_impact(news_df: pd.DataFrame, symbol: str) -> dict:
    """Analyze economic news impact on market trend based on trading experience."""
    if news_df is None or news_df.empty:
        print("No news data available for economic impact analysis or {symbol}: No news data provided." )
        return {"impact_score": 0, "key_events": []}
    
    # Key economic terms that moved markets in successful trades
    economic_keywords = {
        "jobless claims": {"weight": 3.0, "higher_is_bad": True},
        "unemployment": {"weight": 2.5, "higher_is_bad": True},
        "tariff": {"weight": 2.8, "higher_is_bad": True},
        "fed rate": {"weight": 3.0, "higher_is_bad": None},
        "interest rate": {"weight": 3.0, "higher_is_bad": None},
        "gdp": {"weight": 2.5, "higher_is_bad": False},
        "recession": {"weight": 3.0, "higher_is_bad": True},
        "inflation": {"weight": 2.8, "higher_is_bad": True},
        "earnings": {"weight": 2.5, "higher_is_bad": False},
        "revenue": {"weight": 2.3, "higher_is_bad": False},
        "forecast": {"weight": 2.2, "higher_is_bad": None},
        "guidance": {"weight": 2.4, "higher_is_bad": None},
        "layoffs": {"weight": 2.7, "higher_is_bad": True},
        "consumer": {"weight": 2.0, "higher_is_bad": False},
        "growth": {"weight": 2.1, "higher_is_bad": False}
    }
    
    impact_score = 0
    key_events = []
    
    # Determine which column to use for headlines
    headline_col = None
    for col in ["headline", "title", "summary"]:
        if col in news_df.columns:
            headline_col = col
            break
    
    if headline_col is None:
        print(f"ERROR sentiment.py analyze_economic_impact for {symbol}: Column '{headline_col}' not in news_df.")
        return {"impact_score": 0, "key_events": []}
    
    print(f"Analyzing economic impact using '{headline_col}' column from {len(news_df)} news items")
    
    for _, row in news_df.iterrows():
        headline = str(row[headline_col]).lower()
        for keyword, attr in economic_keywords.items():
            if keyword in headline:
                # Extract numeric values from headline if present
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', headline)
                
                # Create event record
                event = {
                    "keyword": keyword,
                    "headline": row[headline_col],
                    "time": row["datetime"] if "datetime" in row else None,
                    "numbers": numbers if numbers else []
                }
                key_events.append(event)
                
                # Score based on content
                if "better than expected" in headline or "beats" in headline or "positive" in headline:
                    direction = 1 if not attr["higher_is_bad"] else -1
                    impact_score += attr["weight"] * direction
                    print(f"Positive economic impact: '{keyword}' in '{headline[:50]}...' (Score: {attr['weight'] * direction:.2f})")
                elif "worse than expected" in headline or "misses" in headline or "negative" in headline:
                    direction = -1 if not attr["higher_is_bad"] else 1
                    impact_score += attr["weight"] * direction
                    print(f"Negative economic impact: '{keyword}' in '{headline[:50]}...' (Score: {attr['weight'] * direction:.2f})")
                else:
                    # Neutral mention
                    neutral_score = attr["weight"] * 0.2
                    impact_score += neutral_score
                    print(f"Neutral economic mention: '{keyword}' in '{headline[:50]}...' (Score: {neutral_score:.2f})")
    
    print(f"Economic impact analysis complete: Score = {impact_score:.2f}, Events = {len(key_events)}")
    return {"impact_score": round(impact_score, 2), "key_events": key_events}