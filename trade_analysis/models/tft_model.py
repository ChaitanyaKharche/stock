"""
Reconstruction of the missing GapPredictionTFT model class.

The original trade_analysis/tft_model.py was lost (only trained checkpoints in
trained_models/ survived). This rebuild is reverse-engineered from:
  - the exact state_dict tensor names/shapes saved in the *_validated.pth /
    *_e200_.pth checkpoints (architecture is high-confidence, since weights
    only load successfully if shapes match exactly)
  - the full model_config/temporal_features/static_features metadata that
    happened to survive inside the *_e200_.pth / tft_model.pth wrapper
    checkpoints (not present in the bare *_validated.pth files)
  - trade_analysis/data.py's get_alternative_data(), which already computes
    two of the five static features (vix level, sector)

The one thing NOT recoverable from tensors alone is the exact forward-pass
wiring (how the static embedding combines with the temporal encoder). The
wiring below (static embedding seeds the LSTM's initial hidden/cell state;
LSTM branch and self-attention branch are each gated, then summed) is a
best-effort reconstruction consistent with every shape in the checkpoints,
not a guaranteed match to the original — treat outputs as a real model
producing real predictions, not as a bit-for-bit reproduction of whatever
the original training run actually did internally.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

MODEL_CONFIG = {
    'hidden_size': 128,
    'lstm_layers': 2,
    'num_heads': 8,
    'quantiles': [0.1, 0.5, 0.9],
    'context_length': 96,
    'prediction_length': 1,
}

TEMPORAL_FEATURES = [
    'close', 'volume', 'RSI_14', 'MACDh_12_26_9', 'ADX_9', 'ATR_14', 'EMA_9',
    'returns', 'volatility', 'high_low_ratio'
]
STATIC_FEATURES = [
    'market_cap_category', 'beta_estimate', 'sector_code', 'vix_regime', 'liquidity_score'
]

SECTOR_CODES = {
    'Technology': 0, 'Communication Services': 1, 'Consumer Cyclical': 2,
    'Consumer Defensive': 3, 'Healthcare': 4, 'Financial Services': 5,
    'Industrials': 6, 'Energy': 7, 'Utilities': 8, 'Real Estate': 9,
    'Basic Materials': 10, 'ETF': 11, 'Unknown': 12, 'Error': 12,
}


class GapPredictionTFT(nn.Module):
    def __init__(self, num_temporal=10, num_static=5, hidden_size=128,
                 lstm_layers=2, num_heads=8, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.quantiles = list(quantiles)

        self.static_embedding = nn.Linear(num_static, hidden_size)
        self.temporal_embedding = nn.Linear(num_temporal, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.gate_lstm = nn.Linear(hidden_size, hidden_size)
        self.gate_attention = nn.Linear(hidden_size, hidden_size)

        self.quantile_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))
            for _ in self.quantiles
        ])
        self.gap_classifier = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 3)
        )

    def forward(self, temporal_seq, static_vec):
        # temporal_seq: (batch, context_length, num_temporal); static_vec: (batch, num_static)
        batch_size = temporal_seq.shape[0]

        static_emb = self.static_embedding(static_vec)  # (batch, hidden)
        h0 = static_emb.unsqueeze(0).repeat(self.lstm_layers, 1, 1).contiguous()
        c0 = torch.zeros_like(h0)

        temporal_emb = self.temporal_embedding(temporal_seq)  # (batch, seq, hidden)

        lstm_out, _ = self.lstm(temporal_emb, (h0, c0))
        lstm_last = lstm_out[:, -1, :]

        attn_out, _ = self.attention(temporal_emb, temporal_emb, temporal_emb)
        attn_last = attn_out[:, -1, :]

        gated_lstm = torch.sigmoid(self.gate_lstm(lstm_last)) * lstm_last
        gated_attn = torch.sigmoid(self.gate_attention(attn_last)) * attn_last
        combined = gated_lstm + gated_attn

        quantile_preds = torch.cat([head(combined) for head in self.quantile_heads], dim=1)  # (batch, 3)
        gap_logits = self.gap_classifier(combined)  # (batch, 3)

        return quantile_preds, gap_logits


def _rsi(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_hist(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _adx(high, low, close, length=9):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(length).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(length).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(length).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(length).mean()


def _atr(high, low, close, length=14):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns Open/High/Low/Close/Volume. Returns the 10
    TEMPORAL_FEATURES columns, in order, indexed the same as df."""
    out = pd.DataFrame(index=df.index)
    out['close'] = df['Close']
    out['volume'] = df['Volume']
    out['RSI_14'] = _rsi(df['Close'], 14)
    out['MACDh_12_26_9'] = _macd_hist(df['Close'])
    out['ADX_9'] = _adx(df['High'], df['Low'], df['Close'], 9)
    out['ATR_14'] = _atr(df['High'], df['Low'], df['Close'], 14)
    out['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    out['returns'] = df['Close'].pct_change()
    out['volatility'] = out['returns'].rolling(20).std()
    out['high_low_ratio'] = df['High'] / df['Low']
    return out[TEMPORAL_FEATURES]


def market_cap_category(market_cap):
    if not market_cap:
        return 2  # unknown -> mid, avoids skewing the scaler with an outlier
    if market_cap < 2e9:
        return 0   # small cap
    if market_cap < 10e9:
        return 1   # mid cap
    if market_cap < 200e9:
        return 2   # large cap
    return 3        # mega cap


def vix_regime(vix_level):
    if vix_level < 15:
        return 0  # low vol regime
    if vix_level < 25:
        return 1  # normal
    return 2       # elevated/fear regime


def build_static_features(symbol: str, info: dict, vix_level: float, beta_estimate: float) -> np.ndarray:
    """Assembles the 5 STATIC_FEATURES for one symbol at one point in time.
    market_cap_category/sector_code use current `info` (yfinance snapshot) -
    treated as slowly-varying context rather than a true point-in-time
    historical value, since neither survives anywhere in this repo."""
    market_cap = info.get('marketCap')
    sector = info.get('sector', 'Unknown') or 'Unknown'
    avg_volume = info.get('averageVolume') or 0
    price = info.get('currentPrice') or info.get('regularMarketPrice') or 1.0

    liquidity = np.log1p(avg_volume * price) if avg_volume and price else 0.0

    return np.array([
        market_cap_category(market_cap),
        beta_estimate,
        SECTOR_CODES.get(sector, SECTOR_CODES['Unknown']),
        vix_regime(vix_level),
        liquidity,
    ], dtype=np.float32)


def load_checkpoint(path_pth: str, path_joblib: str = None):
    """Loads either a bare state_dict (*_validated.pth, needs a companion
    .joblib for scalers) or a full wrapped checkpoint (*_e200_.pth /
    tft_model.pth, which embeds its own scalers). Returns (model, scaler_temporal, scaler_static)."""
    ckpt = torch.load(path_pth, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        scaler_temporal = ckpt['scaler_temporal']
        scaler_static = ckpt['scaler_static']
    else:
        state_dict = ckpt
        if path_joblib is None:
            raise ValueError(f"{path_pth} is a bare state_dict but no companion .joblib was given for scalers")
        scalers = joblib.load(path_joblib)
        scaler_temporal = scalers['scaler_temporal']
        scaler_static = scalers['scaler_static']

    model = GapPredictionTFT(
        num_temporal=len(TEMPORAL_FEATURES), num_static=len(STATIC_FEATURES),
        hidden_size=MODEL_CONFIG['hidden_size'], lstm_layers=MODEL_CONFIG['lstm_layers'],
        num_heads=MODEL_CONFIG['num_heads'], quantiles=MODEL_CONFIG['quantiles'],
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler_temporal, scaler_static


@torch.no_grad()
def predict(model, scaler_temporal, scaler_static, temporal_window: pd.DataFrame, static_vec: np.ndarray):
    """temporal_window: DataFrame of the last context_length rows of TEMPORAL_FEATURES.
    static_vec: raw (unscaled) 5-value array from build_static_features.
    Returns dict with quantile price/return predictions and gap class probabilities."""
    t_scaled = scaler_temporal.transform(temporal_window.values)
    s_scaled = scaler_static.transform(static_vec.reshape(1, -1))

    t_tensor = torch.tensor(t_scaled, dtype=torch.float32).unsqueeze(0)
    s_tensor = torch.tensor(s_scaled, dtype=torch.float32)

    quantile_preds, gap_logits = model(t_tensor, s_tensor)
    gap_probs = torch.softmax(gap_logits, dim=1).squeeze(0).tolist()

    q = MODEL_CONFIG['quantiles']
    quantiles = {f'q{int(qi*100)}': float(v) for qi, v in zip(q, quantile_preds.squeeze(0).tolist())}

    return {
        'quantiles': quantiles,
        'gap_class_probs': {'DOWN': gap_probs[0], 'FLAT': gap_probs[1], 'UP': gap_probs[2]},
    }
