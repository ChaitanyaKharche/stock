from transformers import pipeline; import pandas as pd
print("Loading FinBERT…"); _pipe=pipeline("sentiment-analysis",model="ProsusAI/finbert")
def annotate(df: pd.DataFrame, col="headline"):
    if df.empty: df["sentiment_score"]=0.0; return df
    scores=_pipe(df[col].tolist(), batch_size=16, truncation=True)
    df["sentiment_score"]=[{"positive":s["score"],"negative":-s["score"]}.get(s["label"].lower(),0.0) for s in scores]
    return df
def aggregate(df): 
    return float(pd.to_numeric(df["sentiment_score"],errors="coerce").fillna(0.0).mean()), len(df)
