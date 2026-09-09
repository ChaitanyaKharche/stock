# trade_analysis/enhanced_llm.py

import os
import sys
os.environ['TRANSFORMERS_ATTENTION_IMPL'] = 'eager'
os.environ['FLASH_ATTENTION_FORCE_DISABLE'] = '1'

class FlashAttnBlocker:
    def find_spec(self, name, path, target=None):
        if 'flash_attn' in name:
            return None
        return None
    
    def find_module(self, name, path=None):
        if 'flash_attn' in name:
            return None
        return None

sys.meta_path.insert(0, FlashAttnBlocker())


import torch
import json
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, pipeline
)
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

class EnhancedLLMEngine:
    """
    Enhanced LLM Engine with updated SOTA models.
    Disabled flash attention due to CUDA compatibility issues.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.tokenizers = {}
        
        # NEW: Updated LLM Configuration with Nemo and Phi-4
        # These models are more powerful than the previous ensemble.
        self.llm_configs = {
            # Mistral Nemo Instruct 12B: SOTA reasoning model
            'mistral_nemo_instruct': {
                'model_id': 'mistralai/Mistral-Nemo-Instruct-2407', # CRITICAL: Use the Instruct version
                'weight': 0.6, # Higher weight for the more capable model
                'load_in_4bit': True,
                'min_vram_gb': 12, # 12B model in 4-bit needs ~8-10 GB VRAM
                'specialization': 'advanced_reasoning',
                'context_length': 128000
            },
            
            # Phi-4 Mini Instruct: Fast and highly capable small model
            'phi4_mini_instruct': {
                'model_id': 'microsoft/Phi-3-medium-4k-instruct',
                'weight': 0.4,
                'load_in_4bit': True, # Quantization for speed and consistency
                'min_vram_gb': 6, # 4.2B model in 4-bit needs ~4-5 GB VRAM
                'specialization': 'rapid_inference',
                'context_length': 128000
            }
        }
        
        # Renormalize weights
        total_weight = sum(config['weight'] for config in self.llm_configs.values())
        if total_weight > 0:
            for config in self.llm_configs.values():
                config['weight'] /= total_weight
    
    def initialize_llm_models(self):
        """Initialize all LLM models"""
        print("Loading Enhanced LLM Models...")
        
        for model_key, config in self.llm_configs.items():
            try:
                print(f"Loading {model_key}...")
                
                # Setup quantization
                quant_config = None
                if config.get('load_in_4bit'):
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                
                # Load tokenizer
                self.tokenizers[model_key] = AutoTokenizer.from_pretrained(
                    config['model_id'],
                    trust_remote_code=True
                )
                
                # Load model WITHOUT flash attention
                self.models[model_key] = AutoModelForCausalLM.from_pretrained(
                    config['model_id'],
                    quantization_config=quant_config,
                    device_map="auto", # device_map="auto" is best practice for quantized models
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                
                # Set pad token
                if self.tokenizers[model_key].pad_token is None:
                    self.tokenizers[model_key].pad_token = self.tokenizers[model_key].eos_token
                
                print(f"✅ {model_key} loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load {model_key}: {e}")
                config['weight'] = 0
        
        print(f"✅ Loaded {len(self.models)} LLM models")

    # The rest of your class methods (_build_comprehensive_prompt, 
    # generate_enhanced_trading_signal, etc.) remain unchanged.
    # ... (paste the rest of your original EnhancedLLMEngine methods here)
    def generate_enhanced_trading_signal(self, market_data: Dict, sentiment_data: Dict, 
                                         momentum_data: Dict, alternative_data: Dict) -> Dict:
        if not self.models:
            # Fallback to rule-based signal if no models loaded
            return generate_enhanced_llm_signal({
                "is_vix_high": alternative_data.get('vix_level', 20) > 25,
                "is_15m_rsi_bullish": False,
                "is_15m_rsi_bearish": False,
                "is_15m_volume_spike": False,
                "is_hourly_trend_bullish": False,
                "is_hourly_trend_bearish": False
            })
        
        prompt = self._build_comprehensive_prompt(
            market_data, sentiment_data, momentum_data, alternative_data
        )
        
        predictions = {}
        for model_key, config in self.llm_configs.items():
            if config['weight'] == 0 or model_key not in self.models:
                continue
            
            try:
                prediction = self._generate_with_model(model_key, prompt)
                predictions[model_key] = {
                    'prediction': prediction,
                    'weight': config['weight'],
                    'specialization': config['specialization']
                }
            except Exception as e:
                print(f"Error with {model_key}: {e}")
                continue
        
        final_signal = self._ensemble_llm_predictions(predictions)
        return final_signal

    def _build_comprehensive_prompt(self, market_data: Dict, sentiment_data: Dict,
                                  momentum_data: Dict, alternative_data: Dict) -> str:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"""Trading Analysis - {current_time}

MARKET:
- Price: {self._format_market_data(market_data)}
- Volume: {self._format_volume_data(market_data)}

SENTIMENT:
- Score: {sentiment_data.get('composite_score', 0):.3f}
- Confidence: {sentiment_data.get('confidence', 'UNKNOWN')}

MOMENTUM:
- Signal: {momentum_data.get('master_signal', {}).get('signal', 'UNKNOWN')}
- Conviction: {momentum_data.get('master_signal', {}).get('conviction', 0):.2f}

VIX: {alternative_data.get('vix_level', 20):.2f}

Based on this data, provide:
1. SIGNAL: BULLISH, BEARISH, or NEUTRAL
2. CONFIDENCE: 0-100
3. REASONING: One sentence

Response:"""
        return prompt

    def _format_market_data(self, market_data: Dict) -> str:
        data_parts = []
        for timeframe, df in market_data.items():
            if not df.empty and 'Close' in df.columns:
                current_price = df['Close'].iloc[-1]
                if len(df) > 1:
                    price_change = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100
                    data_parts.append(f"{timeframe}: ${current_price:.2f} ({price_change:+.2f}%)")
                else:
                    data_parts.append(f"{timeframe}: ${current_price:.2f}")
        return ", ".join(data_parts) if data_parts else "No data"

    def _format_volume_data(self, market_data: Dict) -> str:
        volume_parts = []
        for timeframe, df in market_data.items():
            if not df.empty and 'Volume' in df.columns:
                current_vol = df['Volume'].iloc[-1]
                if len(df) > 20:
                    avg_vol = df['Volume'].tail(20).mean()
                    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
                    volume_parts.append(f"{timeframe}: {vol_ratio:.1f}x")
        return ", ".join(volume_parts) if volume_parts else "Normal"

    def _generate_with_model(self, model_key: str, prompt: str) -> Dict:
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        config = self.llm_configs[model_key]
        max_length = min(config.get('context_length', 2048), 1024)

        if 'phi4' in model_key.lower():
            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                ).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                response = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                return self._parse_llm_response(response, model_key)
            except (IndexError, RuntimeError) as e:
                print(f"Phi4 tensor error: {e}. Falling back to rule-based.")
                return {
                    'trade_signal': 'NEUTRAL',
                    'conviction': 50,
                    'reasoning': f"Phi4 issue: {str(e)[:50]}",
                    'model_source': model_key
                }

        # non-phi4 path
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        return self._parse_llm_response(response, model_key)


    def _parse_llm_response(self, response: str, model_key: str) -> Dict:
        import re
        result = {
            'trade_signal': 'NEUTRAL',
            'conviction': 50,
            'timeframe': 'INTRADAY',
            'reasoning': response[:100] if response else "No analysis",
            'risk_factors': 'Standard market risks',
            'options_play': 'Wait for better setup',
            'model_source': model_key
        }
        
        response_upper = response.upper()
        
        if 'BULLISH' in response_upper or 'BUY' in response_upper or 'CALLS' in response_upper:
            result['trade_signal'] = 'BULLISH'
        elif 'BEARISH' in response_upper or 'SELL' in response_upper or 'PUTS' in response_upper:
            result['trade_signal'] = 'BEARISH'
        
        numbers = re.findall(r'\d+', response)
        if numbers:
            for num in numbers:
                num_int = int(num)
                if 0 <= num_int <= 100:
                    result['conviction'] = num_int
                    break
        
        return result

    def _ensemble_llm_predictions(self, predictions: Dict) -> Dict:
        if not predictions:
            return {
                'signal': 'HOLD', 'conviction': 30, 'reasoning': 'No LLM models available',
                'timeframe': 'WAIT', 'options_strategy': 'WAIT', 'model_consensus': 'NO_CONSENSUS'
            }
        
        signals, convictions, weights = [], [], []
        for model_key, pred_data in predictions.items():
            pred = pred_data['prediction']
            weight = pred_data['weight']
            signals.append(pred['trade_signal'])
            convictions.append(pred['conviction'])
            weights.append(weight)
        
        if sum(weights) > 0:
            weighted_conviction = sum(c * w for c, w in zip(convictions, weights)) / sum(weights)
        else:
            weighted_conviction = 50
        
        signal_votes = {}
        for signal, weight in zip(signals, weights):
            signal_votes[signal] = signal_votes.get(signal, 0) + weight
        
        if signal_votes:
            consensus_signal = max(signal_votes, key=signal_votes.get)
            consensus_strength = signal_votes[consensus_signal] / sum(weights) if sum(weights) > 0 else 0
        else:
            consensus_signal = 'NEUTRAL'
            consensus_strength = 0
            
        final_signal_map = {'BULLISH': 'CALLS', 'BEARISH': 'PUTS'}
        final_signal = final_signal_map.get(consensus_signal, 'HOLD')
        
        reasoning = f"LLM Analysis ({len(predictions)} models). Consensus: {consensus_signal}. Conviction: {weighted_conviction:.0f}%"
        
        return {
            'signal': final_signal, 'conviction': int(weighted_conviction), 'reasoning': reasoning,
            'timeframe': 'INTRADAY', 'options_strategy': 'STANDARD', 'model_consensus': consensus_signal,
            'consensus_strength': consensus_strength, 'participating_models': list(predictions.keys())
        }

# Paste your original generate_enhanced_llm_signal function below this class

# Integration function for existing system
def generate_enhanced_llm_signal(conditions: Dict) -> Dict:
    """
    Enhanced signal generation using LLM ensemble
    Compatible with existing system
    """
    
    # Simple rule-based fallback when LLM not available
    is_vix_high = conditions.get("is_vix_high", False)
    is_15m_rsi_bullish = conditions.get("is_15m_rsi_bullish", False)
    is_15m_rsi_bearish = conditions.get("is_15m_rsi_bearish", False)
    is_15m_volume_spike = conditions.get("is_15m_volume_spike", False)
    is_hourly_trend_bullish = conditions.get("is_hourly_trend_bullish", False)
    is_hourly_trend_bearish = conditions.get("is_hourly_trend_bearish", False)
    
    signal = "HOLD"
    confidence = 40
    reasoning = "Rule-based analysis"
    position_size = 0.0
    
    # Enhanced momentum logic
    if is_15m_volume_spike:
        if is_15m_rsi_bullish and is_hourly_trend_bullish:
            signal = "CALLS"
            confidence = 85
            reasoning = "Strong bullish momentum: Volume + RSI + trend"
            position_size = 0.75
        elif is_15m_rsi_bearish and is_hourly_trend_bearish:
            signal = "PUTS"
            confidence = 85
            reasoning = "Strong bearish momentum: Volume + RSI + trend"
            position_size = 0.75
        elif is_15m_rsi_bullish or is_hourly_trend_bullish:
            signal = "CALLS"
            confidence = 65
            reasoning = "Moderate bullish momentum"
            position_size = 0.5
        elif is_15m_rsi_bearish or is_hourly_trend_bearish:
            signal = "PUTS" 
            confidence = 65
            reasoning = "Moderate bearish momentum"
            position_size = 0.5
    
    # VIX regime adjustment
    if is_vix_high:
        confidence -= 10
        position_size *= 0.8
        reasoning += ". High VIX"
    
    return {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
        "position_size": position_size
    }