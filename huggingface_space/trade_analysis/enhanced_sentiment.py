# trade_analysis/enhanced_sentiment.py

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM, BitsAndBytesConfig, pipeline
)
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class EnhancedFinancialSentimentAnalyzer:
    """
    SOTA Financial Sentiment Analysis using 2025 models
    Optimized for H100/H200 GPUs and momentum trading
    """
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Enhanced model configuration - WORKING MODELS ONLY
        self.model_configs = {
            # # Tier 1: SOTA Financial Models (2025)
            # 'finbert_prosus': {
            #     'model_id': 'ProsusAI/finbert',
            #     'weight': 0.25,
            #     'type': 'classification',
            #     'specialization': 'general_financial'
            #  },
            # 'finbert_tone': {
            #     'model_id': 'yiyanghkust/finbert-tone', 
            #     'weight': 0.25,
            #     'type': 'classification',
            #     'specialization': 'tone_analysis'
            # },
            # 'roberta_financial': {
            #     'model_id': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            #     'weight': 0.20,
            #     'type': 'classification',
            #     'specialization': 'social_sentiment'
            # },
            'distilroberta_financial': {
                'model_id': 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
                'weight': 0.20,
                'type': 'classification',
                'specialization': 'news_sentiment'
            },
            
            # # Tier 2: Specialized Models
            # 'fintwit_bert': {
            #     'model_id': 'StephanAkkerman/FinTwitBERT-sentiment',
            #     'weight': 0.10,
            #     'type': 'classification', 
            #     'specialization': 'social_trading'
            # }
        }
        
        # Renormalize weights
        total_weight = sum(config['weight'] for config in self.model_configs.values())
        for config in self.model_configs.values():
            config['weight'] /= total_weight
    
    def initialize_models(self):
        """Load all sentiment models"""
        print("Loading Enhanced Financial Sentiment Models...")
        
        for model_key, config in self.model_configs.items():
            try:
                print(f"Loading {model_key}...")
                
                if config['type'] == 'classification':
                    # Load classification models
                    self.tokenizers[model_key] = AutoTokenizer.from_pretrained(
                        config['model_id'], 
                        trust_remote_code=True
                    )
                    self.models[model_key] = AutoModelForSequenceClassification.from_pretrained(
                        config['model_id'],
                        trust_remote_code=True
                    ).to(self.device)
                    
                elif config['type'] == 'causal':
                    # Skip causal models for now since they're having issues
                    print(f"Skipping causal model {model_key} - focusing on classification models")
                    config['weight'] = 0
                    continue
                        
                print(f"✅ {model_key} loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load {model_key}: {e}")
                config['weight'] = 0
        
        # Create sentiment pipeline for fast inference
        self._create_pipelines()
        print(f"✅ Loaded {len(self.models)} sentiment models")
    
    def _create_pipelines(self):
        """Create HuggingFace pipelines for efficient inference"""
        for model_key, config in self.model_configs.items():
            if config['weight'] > 0 and model_key in self.models:
                if config['type'] == 'classification':
                    try:
                        self.pipelines[model_key] = pipeline(
                            "sentiment-analysis",
                            model=self.models[model_key],
                            tokenizer=self.tokenizers[model_key],
                            device=0 if torch.cuda.is_available() else -1,
                            return_all_scores=True
                        )
                    except Exception as e:
                        print(f"Failed to create pipeline for {model_key}: {e}")
    
    def analyze_comprehensive_sentiment(self, news_df: pd.DataFrame, social_df: pd.DataFrame, symbol: str) -> Dict:
        """
        Comprehensive sentiment analysis for momentum trading
        """
        if news_df.empty and social_df.empty:
            return self._default_sentiment()
        
        # Prepare text data
        texts = []
        metadata = []
        
        # Add news headlines
        if not news_df.empty:
            for _, row in news_df.iterrows():
                text = row.get('headline', '') or row.get('title', '')
                if text:
                    texts.append(str(text))
                    metadata.append({
                        'source': 'news',
                        'timestamp': row.get('datetime', datetime.now()),
                        'impact': self._calculate_news_impact(str(text))
                    })
        
        # Add social media content
        if not social_df.empty:
            for _, row in social_df.iterrows():
                text = row.get('title', '') or row.get('content', '')
                if text:
                    texts.append(str(text))
                    metadata.append({
                        'source': 'social',
                        'timestamp': row.get('created_utc', datetime.now()),
                        'score': row.get('score', 0)
                    })
        
        if not texts:
            return self._default_sentiment()
        
        # Run ensemble sentiment analysis
        sentiment_results = self._run_ensemble_sentiment(texts)
        
        # Calculate weighted sentiment scores
        financial_sentiment = self._calculate_financial_sentiment(sentiment_results, metadata)
        social_sentiment = self._calculate_social_sentiment(sentiment_results, metadata)
        
        # Economic impact analysis
        economic_impact = self._analyze_economic_impact(texts)
        
        # Create momentum-focused composite score
        composite_score = self._calculate_momentum_composite(
            financial_sentiment, social_sentiment, economic_impact
        )
        
        # Generate key themes for transparency
        key_themes = self._extract_key_themes(texts, sentiment_results)
        
        return {
            'financial_sentiment': financial_sentiment,
            'social_sentiment': social_sentiment, 
            'economic_impact': economic_impact,
            'composite_score': composite_score,
            'confidence': self._calculate_confidence(sentiment_results),
            'key_themes': key_themes,
            'model_count': len([k for k, v in self.model_configs.items() if v['weight'] > 0])
        }
    
    def _run_ensemble_sentiment(self, texts: List[str]) -> Dict:
        """Run all available models on the text data"""
        results = {}
        
        for model_key, config in self.model_configs.items():
            if config['weight'] == 0 or model_key not in self.models:
                continue
                
            try:
                if config['type'] == 'classification':
                    # Use pipeline for fast inference
                    if model_key in self.pipelines:
                        predictions = []
                        for text in texts:
                            result = self.pipelines[model_key](text[:512])
                            # Convert to standardized score
                            if isinstance(result, list) and len(result) > 0:
                                if isinstance(result[0], dict):
                                    score = self._standardize_classification_score(result)
                                else:
                                    score = self._standardize_classification_score(result[0])
                            else:
                                score = 0.0
                            predictions.append(score)
                    else:
                        predictions = self._run_classification_batch(texts, model_key)
                        
                elif config['type'] == 'causal':
                    # Skip causal for now
                    continue
                    
                results[model_key] = {
                    'predictions': predictions,
                    'weight': config['weight'],
                    'specialization': config['specialization']
                }
                
            except Exception as e:
                print(f"Error running {model_key}: {e}")
                continue
                
        return results
    
    def _run_classification_batch(self, texts: List[str], model_key: str) -> List[float]:
        """Run classification model in batches"""
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        
        predictions = []
        batch_size = 8  # Reduced for stability
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    
                    for prob in probs:
                        if prob.shape[0] == 3:  # [negative, neutral, positive]
                            score = prob[2].item() - prob[0].item()
                        else:  # [negative, positive]
                            score = prob[1].item() - prob[0].item()
                        predictions.append(score)
            except Exception as e:
                print(f"Batch processing error: {e}")
                # Add neutral scores for failed batch
                predictions.extend([0.0] * len(batch_texts))
        
        return predictions
    
    def _standardize_classification_score(self, result) -> float:
        """Convert pipeline output to standardized score"""
        if not result:
            return 0.0
        
        try:
            # Handle nested list structure
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    result = result[0]
            
            # Convert to dict if not already
            if isinstance(result, list):
                scores = {}
                for item in result:
                    if isinstance(item, dict) and 'label' in item:
                        scores[item['label'].upper()] = item['score']
            else:
                return 0.0
            
            positive_labels = ['POSITIVE', 'POS', 'BULLISH', 'LABEL_2']
            negative_labels = ['NEGATIVE', 'NEG', 'BEARISH', 'LABEL_0']
            
            positive_score = sum(scores.get(label, 0) for label in positive_labels)
            negative_score = sum(scores.get(label, 0) for label in negative_labels)
            
            return positive_score - negative_score
        except Exception as e:
            print(f"Score standardization error: {e}")
            return 0.0
    
    def _calculate_financial_sentiment(self, results: Dict, metadata: List[Dict]) -> float:
        """Calculate weighted financial sentiment score"""
        if not results:
            return 0.0
        
        weighted_scores = []
        total_weight = 0
        
        for model_key, model_results in results.items():
            predictions = model_results['predictions']
            weight = model_results['weight']
            specialization = model_results['specialization']
            
            # Apply specialization bonus
            if specialization in ['general_financial', 'earnings', 'news_sentiment']:
                weight *= 1.2
                
            # Weight by news impact
            for i, pred in enumerate(predictions[:len(metadata)]):
                meta = metadata[i] if i < len(metadata) else {'source': 'unknown', 'impact': 1.0}
                if meta['source'] == 'news':
                    impact_weight = meta.get('impact', 1.0)
                    weighted_scores.append(pred * weight * impact_weight)
                    total_weight += weight * impact_weight
                else:
                    weighted_scores.append(pred * weight)
                    total_weight += weight
        
        return sum(weighted_scores) / max(total_weight, 1)
    
    def _calculate_social_sentiment(self, results: Dict, metadata: List[Dict]) -> float:
        """Calculate social media sentiment score"""
        if not results:
            return 0.0
        
        social_scores = []
        
        for model_key, model_results in results.items():
            predictions = model_results['predictions']
            specialization = model_results['specialization']
            
            # Prioritize social-specific models
            weight = 1.5 if specialization == 'social_sentiment' else 1.0
            
            for i, pred in enumerate(predictions[:len(metadata)]):
                meta = metadata[i] if i < len(metadata) else {'source': 'unknown', 'score': 0}
                if meta['source'] == 'social':
                    # Weight by social score (upvotes, likes, etc.)
                    social_weight = min(max(meta.get('score', 0) / 10, 0.5), 2.0)
                    social_scores.append(pred * weight * social_weight)
        
        return np.mean(social_scores) if social_scores else 0.0
    
    def _analyze_economic_impact(self, texts: List[str]) -> float:
        """Analyze economic impact using keyword analysis"""
        impact_keywords = {
            'high_impact': ['fed', 'federal reserve', 'inflation', 'gdp', 'unemployment', 'interest rate'],
            'medium_impact': ['earnings', 'revenue', 'profit', 'guidance', 'outlook'],
            'market_structure': ['merger', 'acquisition', 'ipo', 'split', 'dividend']
        }
        
        total_impact = 0
        impact_count = 0
        
        for text in texts:
            text_lower = text.lower()
            
            # High impact events
            high_matches = sum(1 for keyword in impact_keywords['high_impact'] 
                             if keyword in text_lower)
            if high_matches > 0:
                total_impact += high_matches * 3
                impact_count += 1
            
            # Medium impact events  
            medium_matches = sum(1 for keyword in impact_keywords['medium_impact']
                               if keyword in text_lower)
            if medium_matches > 0:
                total_impact += medium_matches * 2
                impact_count += 1
                
            # Market structure events
            structure_matches = sum(1 for keyword in impact_keywords['market_structure']
                                  if keyword in text_lower)
            if structure_matches > 0:
                total_impact += structure_matches * 1.5
                impact_count += 1
        
        return total_impact / max(impact_count, 1)
    
    def _calculate_momentum_composite(self, financial_sent: float, social_sent: float, 
                                   economic_impact: float) -> float:
        """Calculate composite score optimized for momentum trading"""
        # Momentum trading weights - prioritize speed and strength
        financial_weight = 0.5  # Primary signal
        social_weight = 0.2     # Secondary confirmation
        economic_weight = 0.3   # Impact multiplier
        
        composite = (financial_sent * financial_weight + 
                    social_sent * social_weight + 
                    economic_impact * economic_weight * 0.1)  # Scale economic impact
        
        # Apply momentum amplification for strong signals
        if abs(composite) > 0.5:
            composite *= 1.2
            
        return np.clip(composite, -1.0, 1.0)
    
    def _calculate_confidence(self, results: Dict) -> str:
        """Calculate confidence level based on model agreement"""
        if not results:
            return "LOW"
        
        all_predictions = []
        for model_results in results.values():
            all_predictions.extend(model_results['predictions'])
            
        if not all_predictions:
            return "LOW"
            
        # Calculate standard deviation for agreement
        std_dev = np.std(all_predictions)
        mean_abs = np.mean(np.abs(all_predictions))
        
        if std_dev < 0.2 and mean_abs > 0.3:
            return "HIGH"
        elif std_dev < 0.4 and mean_abs > 0.2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _extract_key_themes(self, texts: List[str], results: Dict) -> List[Dict]:
        """Extract key themes with sentiment scores"""
        themes = []
        
        # Simple theme extraction based on high-impact content
        for i, text in enumerate(texts[:10]):  # Limit for performance
            # Calculate average sentiment for this text
            avg_sentiment = 0
            model_count = 0
            
            for model_results in results.values():
                if i < len(model_results['predictions']):
                    avg_sentiment += model_results['predictions'][i]
                    model_count += 1
            
            if model_count > 0:
                avg_sentiment /= model_count
                
                # Only include significant sentiments
                if abs(avg_sentiment) > 0.3:
                    themes.append({
                        'headline': text[:100],
                        'sentiment': round(avg_sentiment, 3),
                        'impact': 'HIGH' if abs(avg_sentiment) > 0.6 else 'MEDIUM'
                    })
        
        return sorted(themes, key=lambda x: abs(x['sentiment']), reverse=True)[:5]
    
    def _calculate_news_impact(self, text: str) -> float:
        """Calculate news impact multiplier"""
        text_lower = text.lower()
        
        # High impact keywords
        high_impact = ['breaking', 'urgent', 'alert', 'crash', 'surge', 'halted']
        medium_impact = ['announces', 'reports', 'updates', 'guidance']
        
        multiplier = 1.0
        
        if any(keyword in text_lower for keyword in high_impact):
            multiplier = 2.0
        elif any(keyword in text_lower for keyword in medium_impact):
            multiplier = 1.5
            
        return multiplier
    
    def _default_sentiment(self) -> Dict:
        """Return default sentiment values"""
        return {
            'financial_sentiment': 0.0,
            'social_sentiment': 0.0,
            'economic_impact': 0.0,
            'composite_score': 0.0,
            'confidence': 'LOW',
            'key_themes': [],
            'model_count': 0
        }

# Momentum-specific analysis functions
class MomentumSentimentSignals:
    """Generate momentum trading signals from sentiment"""
    
    @staticmethod
    def generate_momentum_signals(sentiment_data: Dict, timeframe: str = '5m') -> Dict:
        """Generate momentum signals for scalping/day trading"""
        
        composite_score = sentiment_data.get('composite_score', 0)
        confidence = sentiment_data.get('confidence', 'LOW')
        economic_impact = sentiment_data.get('economic_impact', 0)
        
        # Momentum thresholds based on timeframe
        thresholds = {
            '1m': {'strong': 0.3, 'weak': 0.15},
            '5m': {'strong': 0.4, 'weak': 0.2},
            '15m': {'strong': 0.5, 'weak': 0.25}
        }
        
        thresh = thresholds.get(timeframe, thresholds['5m'])
        
        # Generate signals
        if composite_score > thresh['strong'] and confidence in ['HIGH', 'MEDIUM']:
            signal = 'STRONG_BULLISH'
            conviction = 0.8 if confidence == 'HIGH' else 0.6
        elif composite_score > thresh['weak']:
            signal = 'WEAK_BULLISH'
            conviction = 0.5
        elif composite_score < -thresh['strong'] and confidence in ['HIGH', 'MEDIUM']:
            signal = 'STRONG_BEARISH'
            conviction = 0.8 if confidence == 'HIGH' else 0.6
        elif composite_score < -thresh['weak']:
            signal = 'WEAK_BEARISH'
            conviction = 0.5
        else:
            signal = 'NEUTRAL'
            conviction = 0.3
        
        # Economic impact multiplier
        if economic_impact > 3:
            conviction *= 1.2
        
        return {
            'signal': signal,
            'conviction': min(conviction, 1.0),
            'timeframe': timeframe,
            'composite_score': composite_score,
            'economic_multiplier': economic_impact
        }

# Initialize global analyzer instance
sentiment_analyzer = None

def get_sentiment_analyzer():
    """Get or create sentiment analyzer instance"""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = EnhancedFinancialSentimentAnalyzer()
        sentiment_analyzer.initialize_models()
    return sentiment_analyzer

def analyze_momentum_sentiment(news_df: pd.DataFrame, social_df: pd.DataFrame, 
                             symbol: str, timeframe: str = '5m') -> Dict:
    """Main function for momentum sentiment analysis"""
    analyzer = get_sentiment_analyzer()
    
    # Get comprehensive sentiment
    sentiment_data = analyzer.analyze_comprehensive_sentiment(news_df, social_df, symbol)
    
    # Generate momentum signals
    momentum_signals = MomentumSentimentSignals.generate_momentum_signals(
        sentiment_data, timeframe
    )
    
    # Combine results
    return {
        **sentiment_data,
        'momentum_signals': momentum_signals
    }

# For backwards compatibility with existing code
class MultiModalSentimentAnalyzer(EnhancedFinancialSentimentAnalyzer):
    """Backwards compatibility class"""
    pass