# app/enhanced_sentiment.py - Advanced Multi-Modal Sentiment
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List, Tuple
import requests
from datetime import datetime, timedelta

class MultiModalSentimentAnalyzer:
    """
    Advanced sentiment analysis combining multiple models and data sources.
    """
    
    def __init__(self):
        self.financial_sentiment = None
        self.economic_classifier = None
        self.social_sentiment = None
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize all sentiment models with fallbacks."""
        try:
            # Primary financial sentiment model
            self.financial_sentiment = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            print("✅ FinBERT loaded successfully")
        except:
            print("⚠️ FinBERT failed, using RoBERTa fallback")
            self.financial_sentiment = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
        
        try:
            # Economic event classifier
            self.economic_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Economic classifier loaded")
        except:
            print("⚠️ Economic classifier failed")
            
        try:
            # Social media sentiment
            self.social_sentiment = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Social sentiment analyzer loaded")
        except:
            print("⚠️ Social sentiment failed")
    
    def analyze_comprehensive_sentiment(self, news_data: pd.DataFrame, 
                                      reddit_data: pd.DataFrame, 
                                      symbol: str) -> Dict:
        """
        Comprehensive sentiment analysis combining all sources.
        """
        results = {
            "financial_sentiment": 0.0,
            "social_sentiment": 0.0,
            "economic_impact": 0.0,
            "composite_score": 0.0,
            "confidence": "LOW",
            "key_themes": []
        }
        
        # Analyze financial news
        if not news_data.empty and self.financial_sentiment:
            financial_scores = self._analyze_financial_news(news_data)
            results["financial_sentiment"] = financial_scores["avg_sentiment"]
            results["key_themes"].extend(financial_scores["themes"])
        
        # Analyze social sentiment
        if not reddit_data.empty and self.social_sentiment:
            social_scores = self._analyze_social_media(reddit_data)
            results["social_sentiment"] = social_scores["avg_sentiment"]
        
        # Economic impact analysis
        if not news_data.empty:
            economic_scores = self._analyze_economic_impact(news_data, symbol)
            results["economic_impact"] = economic_scores["impact_score"]
            results["key_themes"].extend(economic_scores["events"])
        
        # Calculate composite score with weights
        weights = {"financial": 0.4, "social": 0.3, "economic": 0.3}
        results["composite_score"] = (
            results["financial_sentiment"] * weights["financial"] +
            results["social_sentiment"] * weights["social"] +
            results["economic_impact"] * weights["economic"]
        )
        
        # Determine confidence
        sentiment_consistency = self._calculate_consistency([
            results["financial_sentiment"],
            results["social_sentiment"],
            results["economic_impact"]
        ])
        
        if sentiment_consistency > 0.7 and abs(results["composite_score"]) > 0.3:
            results["confidence"] = "HIGH"
        elif sentiment_consistency > 0.5 and abs(results["composite_score"]) > 0.1:
            results["confidence"] = "MEDIUM"
        else:
            results["confidence"] = "LOW"
        
        return results
    
    def _analyze_financial_news(self, news_data: pd.DataFrame) -> Dict:
        """Analyze financial news with advanced NLP."""
        sentiments = []
        themes = []
        
        for _, row in news_data.iterrows():
            text = str(row.get('headline', ''))
            if len(text) < 10:
                continue
                
            try:
                # Get sentiment
                result = self.financial_sentiment(text[:512])
                
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        # Multiple scores format
                        pos_score = next((item['score'] for item in result[0] if 'positive' in item['label'].lower()), 0)
                        neg_score = next((item['score'] for item in result[0] if 'negative' in item['label'].lower()), 0)
                        sentiment_score = pos_score - neg_score
                    else:
                        # Single score format
                        label = result[0]['label'].lower()
                        score = result[0]['score']
                        sentiment_score = score if 'positive' in label else -score if 'negative' in label else 0
                    
                    sentiments.append(sentiment_score)
                    
                    # Extract themes
                    if abs(sentiment_score) > 0.5:
                        themes.append({
                            "headline": text[:100],
                            "sentiment": round(sentiment_score, 3),
                            "impact": "HIGH" if abs(sentiment_score) > 0.7 else "MEDIUM"
                        })
                        
            except Exception as e:
                print(f"Error analyzing news: {e}")
                continue
        
        return {
            "avg_sentiment": np.mean(sentiments) if sentiments else 0.0,
            "sentiment_std": np.std(sentiments) if sentiments else 0.0,
            "themes": sorted(themes, key=lambda x: abs(x["sentiment"]), reverse=True)[:5]
        }
    
    def _analyze_social_media(self, reddit_data: pd.DataFrame) -> Dict:
        """Analyze social media sentiment with trend detection."""
        sentiments = []
        
        for _, row in reddit_data.iterrows():
            text = str(row.get('title', ''))
            if len(text) < 10:
                continue
                
            try:
                result = self.social_sentiment(text[:512])
                label = result[0]['label'].lower()
                score = result[0]['score']
                
                # Convert to -1 to +1 scale
                if 'positive' in label:
                    sentiment_score = score
                elif 'negative' in label:
                    sentiment_score = -score
                else:
                    sentiment_score = 0.0
                
                sentiments.append(sentiment_score)
                
            except Exception as e:
                continue
        
        return {
            "avg_sentiment": np.mean(sentiments) if sentiments else 0.0,
            "trend": "BULLISH" if np.mean(sentiments) > 0.1 else "BEARISH" if np.mean(sentiments) < -0.1 else "NEUTRAL"
        }
    
    def _analyze_economic_impact(self, news_data: pd.DataFrame, symbol: str) -> Dict:
        """Advanced economic impact analysis."""
        economic_keywords = {
            'fed': 3.0, 'federal reserve': 3.0, 'interest rate': 2.5, 'inflation': 2.5,
            'gdp': 2.0, 'unemployment': 2.0, 'earnings': 2.0, 'revenue': 1.5,
            'guidance': 2.0, 'outlook': 1.5, 'forecast': 1.5, 'target': 1.0,
            'upgrade': 1.5, 'downgrade': -1.5, 'beat': 1.0, 'miss': -1.0,
            'recession': -2.5, 'crisis': -2.0, 'volatility': -1.0,
            'merger': 2.0, 'acquisition': 1.8, 'ipo': 1.5, 'dividend': 1.2
        }
        
        impact_scores = []
        key_events = []
        
        for _, row in news_data.iterrows():
            headline = str(row.get('headline', '')).lower()
            
            # Calculate keyword impact
            event_impact = 0.0
            matched_keywords = []
            
            for keyword, weight in economic_keywords.items():
                if keyword in headline:
                    event_impact += weight
                    matched_keywords.append(keyword)
            
            if event_impact != 0:
                impact_scores.append(event_impact)
                
                if abs(event_impact) > 1.5:
                    key_events.append({
                        "headline": row.get('headline', '')[:100],
                        "impact": round(event_impact, 2),
                        "keywords": matched_keywords
                    })
        
        return {
            "impact_score": np.mean(impact_scores) if impact_scores else 0.0,
            "events": sorted(key_events, key=lambda x: abs(x["impact"]), reverse=True)[:3]
        }
    
    def _calculate_consistency(self, scores: List[float]) -> float:
        """Calculate consistency between different sentiment sources."""
        valid_scores = [s for s in scores if s != 0.0]
        if len(valid_scores) < 2:
            return 0.0
        
        # Check if all scores have same sign (direction)
        positive_count = sum(1 for s in valid_scores if s > 0)
        negative_count = sum(1 for s in valid_scores if s < 0)
        
        # Consistency is higher when scores agree on direction
        total_scores = len(valid_scores)
        max_agreement = max(positive_count, negative_count)
        
        return max_agreement / total_scores if total_scores > 0 else 0.0
