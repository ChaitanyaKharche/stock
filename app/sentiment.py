from transformers import pipeline; import pandas as pd
print("Loading FinBERT…"); _pipe=pipeline("sentiment-analysis",model="ProsusAI/finbert",device=-1)
def annotate(df: pd.DataFrame, col="headline"):
    # Handle empty DataFrame
    if df.empty:
        df["sentiment_score"] = 0.0
        return df
        
    # Handle missing column
    if col not in df.columns:
        print(f"Warning: Column '{col}' not found in DataFrame. Available columns: {df.columns}")
        df["sentiment_score"] = 0.0
        return df
    
    # Process valid texts
    try:
        texts = [str(t) for t in df[col].tolist() if pd.notna(t) and t]
        if not texts:
            df["sentiment_score"] = 0.0
            return df
            
        scores = _pipe(texts, batch_size=16, truncation=True)
        df["sentiment_score"] = [{"positive":s["score"],"negative":-s["score"]}.get(s["label"].lower(), 0.0) 
                                for s in scores]
        return df
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        df["sentiment_score"] = 0.0
        return df
def aggregate(df): 
    return float(pd.to_numeric(df["sentiment_score"],errors="coerce").fillna(0.0).mean()), len(df)
