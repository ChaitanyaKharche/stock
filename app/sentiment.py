import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from typing import Tuple, Dict, List

print("--- LOADING sentiment.py (VERSION_S_003) ---")

# Global sentiment pipeline
finbert_pipeline = None

def initialize_finbert():
    """Initialize FinBERT with proper error handling and fallback."""
    global finbert_pipeline
    
    if finbert_pipeline is not None:
        return True
        
    print("Sentiment.py: Loading FinBERT...")
    
    try:
        # Use ProsusAI/finbert which is more stable
        model_name = "ProsusAI/finbert"
        
        # Check if CUDA is available and use appropriate device
        if torch.cuda.is_available():
            device = 0  # Use GPU 0
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = -1  # Use CPU
            print("Using CPU for FinBERT")
        
        # Initialize with explicit device and error handling
        finbert_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=device,
            return_all_scores=True,
            truncation=True,
            max_length=512
        )
        
        print("Sentiment.py: FinBERT loaded successfully")
        return True
        
    except Exception as e:
        print(f"Sentiment.py: Error loading FinBERT pipeline: {e}")
        
        # Fallback to RoBERTa-based financial sentiment
        try:
            print("Sentiment.py: Trying fallback model...")
            fallback_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            
            finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=fallback_model,
                device=device,
                return_all_scores=True,
                truncation=True,
                max_length=512
            )
            
            print("Sentiment.py: Fallback sentiment model loaded")
            return True
            
        except Exception as e2:
            print(f"Sentiment.py: Fallback model also failed: {e2}")
            finbert_pipeline = None
            return False

def annotate(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Annotate DataFrame with sentiment scores using FinBERT or fallback.
    """
    if df.empty or text_column not in df.columns:
        print(f"Sentiment.py annotate: DataFrame empty or missing column '{text_column}'")
        return df
    
    # Initialize FinBERT if not already done
    if not initialize_finbert():
        print("Sentiment.py annotate: No sentiment pipeline available. Adding neutral scores.")
        df['sentiment_score'] = 0.0
        df['sentiment_label'] = 'neutral'
        return df
    
    df_annotated = df.copy()
    sentiment_scores = []
    sentiment_labels = []
    
    print(f"Sentiment.py: Analyzing {len(df)} texts...")
    
    for idx, text in enumerate(df[text_column]):
        try:
            if pd.isna(text) or text.strip() == "":
                sentiment_scores.append(0.0)
                sentiment_labels.append('neutral')
                continue
            
            # Truncate text if too long
            text_clean = str(text)[:500]
            
            # Get sentiment prediction
            result = finbert_pipeline(text_clean)
            
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    # return_all_scores=True format
                    scores = result[0]
                    
                    # Map labels to scores (FinBERT uses positive/negative/neutral)
                    score_map = {}
                    for item in scores:
                        label = item['label'].lower()
                        score = item['score']
                        
                        if 'positive' in label or 'pos' in label:
                            score_map['positive'] = score
                        elif 'negative' in label or 'neg' in label:
                            score_map['negative'] = score
                        else:
                            score_map['neutral'] = score
                    
                    # Calculate compound score
                    pos_score = score_map.get('positive', 0)
                    neg_score = score_map.get('negative', 0)
                    
                    # Convert to -1 to +1 scale
                    compound_score = pos_score - neg_score
                    
                    # Determine label
                    if compound_score > 0.1:
                        label = 'positive'
                    elif compound_score < -0.1:
                        label = 'negative'
                    else:
                        label = 'neutral'
                    
                    sentiment_scores.append(compound_score)
                    sentiment_labels.append(label)
                    
                else:
                    # Single prediction format
                    label = result[0]['label'].lower()
                    score = result[0]['score']
                    
                    # Convert to compound score
                    if 'positive' in label:
                        compound_score = score
                    elif 'negative' in label:
                        compound_score = -score
                    else:
                        compound_score = 0.0
                    
                    sentiment_scores.append(compound_score)
                    sentiment_labels.append(label)
            else:
                sentiment_scores.append(0.0)
                sentiment_labels.append('neutral')
                
        except Exception as e:
            print(f"Sentiment.py: Error processing text {idx}: {e}")
            sentiment_scores.append(0.0)
            sentiment_labels.append('neutral')
    
    df_annotated['sentiment_score'] = sentiment_scores
    df_annotated['sentiment_label'] = sentiment_labels
    
    print(f"Sentiment.py: Completed sentiment analysis. Avg score: {np.mean(sentiment_scores):.3f}")
    
    return df_annotated

def aggregate(df: pd.DataFrame) -> Tuple[float, int]:
    """
    Aggregate sentiment scores from annotated DataFrame.
    """
    if df.empty or 'sentiment_score' not in df.columns:
        return 0.0, 0
    
    valid_scores = df['sentiment_score'].dropna()
    
    if valid_scores.empty:
        return 0.0, 0
    
    avg_sentiment = float(valid_scores.mean())
    count = len(valid_scores)
    
    return avg_sentiment, count

def analyze_economic_impact(news_df: pd.DataFrame, symbol: str) -> Dict:
    """
    Enhanced economic impact analysis with keyword weighting.
    """
    if news_df.empty or 'headline' not in news_df.columns:
        return {"impact_score": 0.0, "key_events": []}
    
    # Economic keywords with weights
    economic_keywords = {
        'fed': 3.0, 'federal reserve': 3.0, 'interest rate': 2.5, 'inflation': 2.5,
        'gdp': 2.0, 'unemployment': 2.0, 'earnings': 2.0, 'revenue': 1.5,
        'guidance': 2.0, 'outlook': 1.5, 'forecast': 1.5, 'target': 1.0,
        'upgrade': 1.5, 'downgrade': -1.5, 'beat': 1.0, 'miss': -1.0,
        'recession': -2.5, 'crisis': -2.0, 'volatility': -1.0
    }
    
    impact_scores = []
    key_events = []
    
    for idx, row in news_df.iterrows():
        headline = str(row.get('headline', '')).lower()
        base_sentiment = row.get('sentiment_score', 0.0)
        
        # Calculate keyword impact
        keyword_impact = 0.0
        matched_keywords = []
        
        for keyword, weight in economic_keywords.items():
            if keyword in headline:
                keyword_impact += weight * (1 + abs(base_sentiment))
                matched_keywords.append(keyword)
        
        # Apply sentiment direction
        if base_sentiment < 0:
            keyword_impact *= -1
        
        impact_scores.append(keyword_impact)
        
        if abs(keyword_impact) > 1.0:  # Significant impact
            key_events.append({
                'headline': row.get('headline', '')[:100],
                'impact': round(keyword_impact, 2),
                'keywords': matched_keywords,
                'sentiment': round(base_sentiment, 3)
            })
    
    total_impact = sum(impact_scores)
    
    return {
        "impact_score": round(total_impact, 2),
        "key_events": sorted(key_events, key=lambda x: abs(x['impact']), reverse=True)[:5]
    }

# Initialize on import
initialize_finbert()
