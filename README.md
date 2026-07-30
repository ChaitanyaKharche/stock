# Updated README.md

```markdown
# Quantitative Finance Advanced Trading System

## Overview

This repository combines advanced AI models (Temporal Fusion Transformer, LLM ensembles), momentum-based strategies, enhanced sentiment analysis, and sophisticated trading infrastructure. The system is designed for professional-grade quantitative trading with integrated backtesting and real-time monitoring.

---

## Setup

### Windows (Local Development)

1. **Prerequisites**
   - Python 3.11+
   - Miniconda/Anaconda

2. **Environment Setup**
```
cd C:\path\to\your\repo
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. **Fix Windows Encoding (REQUIRED)**
```
chcp 65001
$env:PYTHONUTF8="1"
```

4. **Environment Variables**
```
$env:FINNHUB_API_KEY="your-finnhub-key"
$env:TWELVE_KEY="your-twelve-data-key"  
$env:REDDIT_CLIENT_ID="your-reddit-client-id"
$env:REDDIT_CLIENT_SECRET="your-reddit-secret"
$env:REDDIT_USER_AGENT="script:stock-opinion-analyzer:v1.0 (by /u/YourUsername)"
$env:HF_TOKEN="your-huggingface-token"
$env:FLASH_ATTN_SKIP_CUDA_BUILD="TRUE"  # Skip flash-attention on Windows
```

5. **Data Collection**
```
python -m trade_analysis.collect_data --symbol MSFT
python -m trade_analysis.collect_data --symbol TSLA
python -m trade_analysis.collect_data --symbol QQQ
python -m trade_analysis.collect_data --symbol SPY
python -m trade_analysis.collect_data --symbol NVDA
python -m trade_analysis.collect_data --symbol META
python -m trade_analysis.collect_data --symbol AMD
```

6. **Model Training** (Optional - can be done on HPC)
```
python -m trade_analysis.train_tft --symbol MSFT --save_path trained_models/tft_MSFT_validated.pth --epochs 75
```

---

### HPC Cluster (GPU Training & API)

1. **Request GPU Resources**
```
srun --partition=sharing --gres=gpu:h100:1 --time=1:00:00 --mem=64G --pty bash
```

2. **Activate Environment**
```
conda activate trade-venv
```

3. **Environment Variables**
```
export FINNHUB_API_KEY="your-finnhub-key"
export TWELVE_KEY="your-twelve-data-key"
export REDDIT_CLIENT_ID="your-reddit-client-id"
export REDDIT_CLIENT_SECRET="your-reddit-secret"
export REDDIT_USER_AGENT="script:stock-opinion-analyzer:v1.0 (by /u/YourUsername)"
export HF_TOKEN="your-huggingface-token"
```

4. **Upload Data** (if collected locally)
Upload your `local_data` folder to `/scratch/username/username/trade_analysis/local_data/`

5. **Train TFT Models**
```
# Train individual models
python -m trade_analysis.train_tft --symbol QQQ --save_path trained_models/tft_QQQ_validated.pth --epochs 75
python -m trade_analysis.train_tft --symbol SPY --save_path trained_models/tft_SPY_validated.pth --epochs 75
python -m trade_analysis.train_tft --symbol MSFT --save_path trained_models/tft_MSFT_validated.pth --epochs 75
python -m trade_analysis.train_tft --symbol TSLA --save_path trained_models/tft_TSLA_validated.pth --epochs 75
python -m trade_analysis.train_tft --symbol NVDA --save_path trained_models/tft_NVDA_validated.pth --epochs 75
python -m trade_analysis.train_tft --symbol META --save_path trained_models/tft_META_validated.pth --epochs 75
```

6. **Launch Trading API**
```
python -m uvicorn trade_analysis.enhanced_api:app --host 0.0.0.0 --port 8000
```

---

## File Structure

```
/scratch/username/username/
├── trade_analysis/
│   ├── local_data/                    # Collected market data (JSON)
│   │   ├── QQQ_external_data.json
│   │   ├── SPY_external_data.json
│   │   ├── MSFT_external_data.json
│   │   ├── TSLA_external_data.json
│   │   ├── NVDA_external_data.json
│   │   ├── META_external_data.json
│   │   └── AMD_external_data.json
│   ├── trained_models/               # Trained TFT models
│   │   ├── tft_QQQ_validated.pth     # Model weights
│   │   ├── tft_QQQ_validated.joblib  # Scalers
│   │   ├── tft_SPY_validated.pth
│   │   ├── tft_SPY_validated.joblib
│   │   └── ... (for each symbol)
│   └── enhanced_api.py               # Main API server
└── trading_dashboard.html            # Web dashboard
```

---

## API Endpoints

Once the API is running, access these endpoints:

- **Health Check**: `GET http://localhost:8000/`
- **Live Signals**: `POST http://localhost:8000/predict/enhanced/?symbol=QQQ&timeframe=5m&strategy_mode=momentum`
- **Strategy Comparison**: `POST http://localhost:8000/strategy_comparison/?symbol=QQQ`
- **Market Regimes**: `GET http://localhost:8000/market_regimes/?symbols=QQQ,SPY,IWM`
- **Detailed Health**: `GET http://localhost:8000/health/detailed`

---

## Trading Dashboard

1. **Save the provided HTML as `trading_dashboard.html`**
2. **Open in web browser while API is running**
3. **Features:**
   - Real-time trading signals
   - Strategy comparison (your momentum vs institutional)
   - Market regime detection
   - System health monitoring
   - Auto-refresh every 30 seconds

---

## Model Training Results

Expected training output:
```
🚀 Starting Enhanced Trading Engine...
🤖 Loading TFT models...
✅ Loaded pretrained TFT model for QQQ
✅ Loaded pretrained TFT model for SPY
✅ Loaded pretrained TFT model for MSFT
✅ Loaded pretrained TFT model for TSLA
✅ Loaded pretrained TFT model for NVDA
✅ Loaded pretrained TFT model for META
```

Training typically achieves:
- **Training Loss**: ~0.0003-0.0004
- **Validation Loss**: ~0.0003-0.0005
- **Early Stopping**: Around epoch 35-50
- **Processing Time**: ~5-10 seconds per prediction (vs 60+ without pretrained models)

---

## Troubleshooting

### Windows Issues
- **UnicodeEncodeError**: Run `chcp 65001` and `$env:PYTHONUTF8="1"`
- **Flash-attention fails**: Set `$env:FLASH_ATTN_SKIP_CUDA_BUILD="TRUE"`
- **Packages keep reinstalling**: Make sure your venv stays activated

### HPC Issues
- **Models not loading**: Check file paths match between training `--save_path` and API loading
- **GPU not available**: Verify `srun` allocated GPU with `nvidia-smi`
- **Environment variables reset**: Re-export them in each new session

### API Issues
- **422 Validation Error**: Normal FastAPI behavior, not an actual error
- **Models training on-the-fly**: Load pretrained models in startup event
- **Slow responses**: Ensure TFT models are properly loaded at startup

---

## Next Steps

### Immediate Enhancements
1. **Risk Management**: Add position sizing limits and VaR calculations
2. **Trade Execution**: Log simulated trades with slippage modeling
3. **Real-time PnL**: WebSocket streaming for live P&L tracking
4. **Options Greeks**: Black-Scholes calculations for options strategies

### Advanced Features
1. **Kill Switch**: Emergency position flattening
2. **Compliance Logging**: Audit trail for every decision
3. **Latency Monitoring**: Track execution speeds
4. **Cross-Asset Correlation**: Monitor SPY/VIX/DXY relationships

---

## Performance Benchmarks

With pretrained TFT models:
- **API Response Time**: 5-10 seconds
- **TFT Inference**: <1 second
- **Memory Usage**: ~8GB GPU, ~16GB RAM
- **Concurrent Users**: 10+ (with proper load balancing)

---

For support, check logs and ensure all environment variables are correctly set. The system is designed for institutional-grade reliability and performance.
```

This updated README provides:

1. **Clear platform-specific instructions** for both Windows and HPC
2. **Exact commands** for data collection and model training
3. **Proper environment variable setup** for both platforms
4. **File structure explanation** showing where everything gets saved
5. **Complete API endpoint documentation**
6. **Dashboard integration instructions**
7. **Troubleshooting section** addressing common issues
8. **Performance expectations** and benchmarks
9. **Future enhancement roadmap**

The README now accurately reflects your current system architecture and provides step-by-step instructions for both local development and HPC deployment.