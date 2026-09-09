# trade_analysis/deploy.py
"""
Deployment configuration for different environments
Run same code on HPC, local, or cloud
"""

import os
import torch
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class DeploymentMode(Enum):
    HPC = "hpc"           # Full models, unlimited resources
    LOCAL = "local"       # Quantized models, CPU/small GPU
    CLOUD = "cloud"       # RunPod/Colab, medium resources
    SERVERLESS = "serverless"  # Lambda/Vercel, minimal

@dataclass
class DeploymentConfig:
    """Adaptive configuration based on environment"""
    mode: DeploymentMode
    device: str
    max_gpu_memory: Optional[int]  # GB
    quantization: bool
    batch_size: int
    cache_dir: str
    
    @classmethod
    def auto_detect(cls):
        """Automatically detect and configure environment"""
        
        # Check for HPC markers
        if os.path.exists("/scratch") or "SLURM_JOB_ID" in os.environ:
            return cls(
                mode=DeploymentMode.HPC,
                device="cuda",
                max_gpu_memory=80,  # H100 has 80GB
                quantization=False,
                batch_size=32,
                cache_dir="/scratch/models"
            )
        
        # Check for Colab
        elif 'COLAB_GPU' in os.environ:
            return cls(
                mode=DeploymentMode.CLOUD,
                device="cuda" if torch.cuda.is_available() else "cpu",
                max_gpu_memory=15,  # T4 has 15GB
                quantization=True,
                batch_size=8,
                cache_dir="/content/models"
            )
        
        # Check for RunPod
        elif 'RUNPOD_POD_ID' in os.environ:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // 1e9 if torch.cuda.is_available() else 0
            return cls(
                mode=DeploymentMode.CLOUD,
                device="cuda" if torch.cuda.is_available() else "cpu",
                max_gpu_memory=int(gpu_mem),
                quantization=gpu_mem < 24,  # Quantize if less than 24GB
                batch_size=16,
                cache_dir="/workspace/models"
            )
        
        # Local machine
        else:
            has_gpu = torch.cuda.is_available()
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // 1e9 if has_gpu else 0
            
            return cls(
                mode=DeploymentMode.LOCAL,
                device="cuda" if has_gpu else "cpu",
                max_gpu_memory=int(gpu_mem) if has_gpu else None,
                quantization=True,  # Always quantize locally
                batch_size=4,
                cache_dir="./models"
            )

class ScalableModels:
    """Load models based on available resources"""
    
    def __init__(self):
        self.config = DeploymentConfig.auto_detect()
        print(f"🔧 Deployment Mode: {self.config.mode.value}")
        print(f"🔧 Device: {self.config.device}")
        print(f"🔧 Quantization: {self.config.quantization}")
    
    def load_llm(self):
        """Load LLM based on available resources"""
        
        if self.config.mode == DeploymentMode.HPC:
            # Full precision, large models
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_id = "mistralai/Mistral-Nemo-Instruct-2407"  # 12B model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=self.config.cache_dir
            )
            
        elif self.config.mode == DeploymentMode.CLOUD:
            # Quantized medium models
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # 7B model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                cache_dir=self.config.cache_dir
            )
            
        elif self.config.mode == DeploymentMode.LOCAL:
            # Small, efficient models
            from transformers import AutoModelForCausalLM
            
            if self.config.device == "cuda" and self.config.max_gpu_memory >= 6:
                # Use Phi-3 for small GPUs
                model_id = "microsoft/phi-2"  # 2.7B model
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=self.config.cache_dir
                )
            else:
                # CPU-only: Use GGUF quantized models with llama.cpp
                print("💡 For CPU, use llama.cpp with GGUF models")
                return None
        
        else:  # SERVERLESS
            # Use API endpoints instead
            print("💡 Use HuggingFace Inference API for serverless")
            return None
        
        return model
    
    def load_sentiment_models(self):
        """Load sentiment models based on resources"""
        
        models = []
        
        if self.config.mode in [DeploymentMode.HPC, DeploymentMode.CLOUD]:
            # Load all 5 models
            model_ids = [
                'ProsusAI/finbert',
                'yiyanghkust/finbert-tone',
                'cardiffnlp/twitter-roberta-base-sentiment-latest'
            ]
        else:
            # Load only the best model
            model_ids = ['ProsusAI/finbert']
        
        from transformers import AutoModelForSequenceClassification
        
        for model_id in model_ids:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    cache_dir=self.config.cache_dir
                ).to(self.config.device)
                models.append(model)
            except:
                pass
        
        return models
    
    def load_tft_model(self, symbol: str):
        """Load TFT with appropriate settings"""
        from .tft_model import GapPredictionTFT
        
        model = GapPredictionTFT()
        
        # Adjust model size based on resources
        if self.config.mode == DeploymentMode.LOCAL:
            # Reduce model size for local
            model.model.hidden_size = 64  # Halve the hidden size
            model.model.lstm_layers = 1   # Reduce LSTM layers
        
        # Try to load pretrained
        model_path = f"{self.config.cache_dir}/tft_{symbol}.pth"
        if os.path.exists(model_path):
            model.load_pretrained(path=model_path)
        
        return model

# Lightweight agent for production
class ProductionAgent:
    """Minimal agent that works everywhere"""
    
    def __init__(self):
        self.models = ScalableModels()
        self.config = self.models.config
        
    async def run_on_schedule(self):
        """Run analysis on schedule based on resources"""
        
        if self.config.mode == DeploymentMode.HPC:
            # Run every 5 minutes during market hours
            interval = 300
        elif self.config.mode == DeploymentMode.CLOUD:
            # Run every 15 minutes
            interval = 900
        else:
            # Run every 30 minutes locally
            interval = 1800
        
        while True:
            await self.analyze_markets()
            await asyncio.sleep(interval)
    
    async def analyze_markets(self):
        """Lightweight market analysis"""
        
        symbols = ['QQQ', 'SPY', 'NVDA']
        
        for symbol in symbols:
            # Quick signal check using yfinance only
            signal = await self.quick_signal(symbol)
            
            if signal['confidence'] > 75:
                print(f"🎯 SIGNAL: {symbol} - {signal['action']} ({signal['confidence']}%)")
                
                # Save to file for manual review
                with open('signals.txt', 'a') as f:
                    f.write(f"{datetime.now()},{symbol},{signal['action']},{signal['confidence']}\n")
    
    async def quick_signal(self, symbol: str):
        """Ultra-light signal generation"""
        
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        
        # Get recent data
        df = ticker.history(period='1d', interval='5m')
        if df.empty:
            return {'action': 'HOLD', 'confidence': 0}
        
        # Simple momentum
        close = df['Close']
        returns = (close.iloc[-1] / close.iloc[-10] - 1) if len(close) >= 10 else 0
        
        # Volume check
        vol_ratio = df['Volume'].iloc[-1] / df['Volume'].mean()
        
        # Decision
        if returns > 0.005 and vol_ratio > 1.5:
            return {'action': 'CALLS', 'confidence': 70 + min(30, returns * 1000)}
        elif returns < -0.005 and vol_ratio > 1.5:
            return {'action': 'PUTS', 'confidence': 70 + min(30, abs(returns) * 1000)}
        else:
            return {'action': 'HOLD', 'confidence': 50}