# trade_analysis/train_tft.py
import argparse
import asyncio
import pandas as pd
import pandas_ta
import torch
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Import pandas_ta for technical indicators
try:
    import pandas_ta as ta
except ImportError:
    print("pandas_ta not installed. Install with: pip install pandas_ta")
    exit(1)

from .data import UnifiedDataProvider
from .tft_model import GapPredictionTFT

def calculate_technical_indicators(df):
    """Calculate technical indicators needed for TFT model."""
    # RSI
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    
    # MACD Histogram
    macd_data = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if 'MACDh_12_26_9' in macd_data.columns:
        df['MACDh_12_26_9'] = macd_data['MACDh_12_26_9']
    else:
        # Fallback if column name is different
        macd_cols = [col for col in macd_data.columns if 'MACD' in col and ('h' in col.lower() or 'hist' in col.lower())]
        df['MACDh_12_26_9'] = macd_data.iloc[:, -1] if macd_cols else 0
    
    # ADX
    adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=9)
    if isinstance(adx_data, pd.DataFrame):
        df['ADX_9'] = adx_data.iloc[:, 0]  # Take first column (usually ADX)
    else:
        df['ADX_9'] = adx_data
    
    # ATR
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # EMA
    df['EMA_9'] = ta.ema(df['Close'], length=9)
    
    # Fill NaN values
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    
    return df
    
    
def calculate_advanced_indicators(df):
    """Add more sophisticated technical indicators"""
    # Existing indicators...
    df = calculate_technical_indicators(df)
    
    # Volume-Price Trend (manual calculation)
    df['VPT'] = (df['Close'].pct_change() * df['Volume']).cumsum()
    
    # Bollinger Bands
    bb = ta.bbands(df['Close'], length=20)
    if bb is not None and not bb.empty:
        df['BB_upper'] = bb.iloc[:, 0]  # First column
        df['BB_lower'] = bb.iloc[:, 2]  # Third column  
        df['BB_percent'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    else:
        # Fallback manual Bollinger Bands
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['BB_upper'] = sma + (std * 2)
        df['BB_lower'] = sma - (std * 2)
        df['BB_percent'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Market regime features
    df['volatility_regime'] = df['Close'].rolling(20).std() / df['Close'].rolling(60).std()
    df['trend_strength'] = abs(df['Close'].rolling(20).apply(lambda x: np.polyfit(range(20), x, 1)[0]))
    
    # Time-based features (use defaults since we don't have datetime index)
    df['hour'] = 12
    df['day_of_week'] = 2
    
    # Fill NaN values
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    
    return df
  

def quantile_loss(self, predictions, targets, quantiles=[0.1, 0.5, 0.9]):
    """Quantile regression loss"""
    losses = []
    for i, q in enumerate(quantiles):
        pred = predictions[f'quantile_{q}']
        error = targets - pred
        loss = torch.maximum(q * error, (q - 1) * error)
        losses.append(loss.mean())
    return sum(losses)

    
    

async def main(symbol: str, save_path: str, epochs: int):
    """Main function to train the TFT model and save its weights and scalers."""
    print(f"--- Starting TFT Model Training for {symbol} ---")
    
    # 1. Fetch Data
    data_provider = UnifiedDataProvider()
    print("Fetching extensive historical data...")
    
    # Get ticker data
    ticker = yf.Ticker(symbol)
    df_daily = ticker.history(period="5y", interval="1d")  # 5 years is more realistic
    
    if df_daily.empty:
        print("No data fetched. Check symbol or connection.")
        return
    
    # Reset index to make Date a column, then set it back
    df_daily.reset_index(inplace=True)
    df_daily.set_index('Date', inplace=True)
    
    # 2. Calculate Technical Indicators
    print("Calculating technical indicators...")
    df_daily = calculate_technical_indicators(df_daily)
    df_daily = calculate_advanced_indicators(df_daily)
    df_daily['market_regime'] = np.where(
    df_daily['Close'] > df_daily['Close'].rolling(50).mean(), 1, 0)
    
    
    print(f"Fetched {len(df_daily)} total data points for training.")
    print(f"Columns available: {list(df_daily.columns)}")
    
    # 3. Initialize and Train Model
    tft_predictor = GapPredictionTFT()
    print(f"Training model for {epochs} epochs...")
    tft_predictor.train(df_daily, epochs=epochs)

    # 4. Save the Trained Model
    print(f"Saving model and scalers to {save_path}...")
    tft_predictor.save_model(save_path)
    print("--- Training Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Temporal Fusion Transformer Model.")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol to train on (e.g., QQQ).")
    parser.add_argument("--save_path", type=str, default="trained_models/tft_model.pth", help="Path to save the trained model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    
    args = parser.parse_args()
    
    # Ensure the directory for saving the model exists
    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    asyncio.run(main(args.symbol, args.save_path, args.epochs))
