# Create a basic README.md
cat > README.md << 'EOF'
# Stock Gap Predictor

A machine learning application that predicts stock price gap directions using technical analysis, sentiment analysis, and LLM inference.

## Features

- **Data Acquisition**: Fetches stock data (yfinance), news data (Finnhub), and Reddit sentiment
- **Technical Analysis**: Calculates ADX, DMI, and EMA indicators
- **Sentiment Analysis**: Processes news headlines and Reddit posts using FinBERT
- **LLM Prediction**: Uses GPT-4o API or local Llama model to predict gap direction

## Architecture

The application follows a modular design with separation of concerns:
- `api.py`: FastAPI endpoints and request handling
- `data.py`: Data acquisition and caching
- `indicators.py`: Technical indicator calculation
- `sentiment.py`: Sentiment analysis with FinBERT
- `llm.py`: Language model prediction
- `cache.py`: Local file cache for API responses
- `backtest.py`: Backtesting framework for prediction accuracy

## Setup

1. Create virtual environment: `python -m venv venv`
2. Activate environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Set up environment variables in .env file
5. Run the application: `uvicorn app.api:app --host 127.0.0.1 --port 8000`

## API Endpoints

- POST `/predict/?symbol=AAPL`: Predicts opening gap direction for a given stock symbol
EOF

git add README.md
git commit -m "Add comprehensive README"
git push

financial sentiment analysis model used: https://huggingface.co/bryandts/Finance-Alpaca-Llama-3.2-3B-Instruct-bnb-4bit