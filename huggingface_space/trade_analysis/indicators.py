# trade_analysis/indicators.py
"""
Technical indicators module - Enhanced for momentum trading
Works with your enhanced_api.py and other modules
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# --------------------------------------------------------------------------------------
# Real Wilder indicators, used when pandas_ta is unavailable.
#
# This matters more than it looks. pandas_ta==0.3.14b0 does `from numpy import NaN`,
# which numpy 2.0 removed -- and requirements.txt pins numpy loosely, so on a fresh build
# the import fails and everything below silently falls back. The old fallback assigned
# CONSTANTS: `df_enriched['ADX_9'] = 25.0`. ADX is then gated three times in this file
# with `adx > 25` and `adx > 20`; at exactly 25.0 the first is permanently False and the
# second permanently True, so trend-strength filtering stopped existing rather than
# failing loudly.
#
# These are vectorised ports of trade_analysis/live_lab/indicators.py (copied alongside
# as lab_indicators.py), which is dependency-free, already proven in the live lab, and
# used by indicators_test.py as the oracle these must match.
# --------------------------------------------------------------------------------------


def _wilder_rma(series: "pd.Series", period: int) -> "pd.Series":
    """Wilder's smoothing: seed with the SMA of the first `period`, then recurse.

    NOT the same as ewm(alpha=1/period).mean(), which seeds from the first value alone
    and drifts for hundreds of bars on intraday data.
    """
    s = series.copy().astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if len(s) < period:
        return out
    seed = s.iloc[:period].mean()
    out.iloc[period - 1] = seed
    prev = seed
    vals = s.to_numpy()
    o = out.to_numpy()
    for i in range(period, len(s)):
        prev = (prev * (period - 1) + vals[i]) / period
        o[i] = prev
    return pd.Series(o, index=s.index)


def true_range(df: "pd.DataFrame") -> "pd.Series":
    prev_close = df["Close"].shift(1)
    return pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def wilder_atr(df: "pd.DataFrame", period: int = 14) -> "pd.Series":
    return _wilder_rma(true_range(df), period)


def wilder_rsi(close: "pd.Series", period: int = 14) -> "pd.Series":
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def wilder_adx(df: "pd.DataFrame", period: int = 14):
    """Returns (adx, plus_di, minus_di) as Series. Mirrors lab_indicators.adx_dmi."""
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)

    atr_ = _wilder_rma(true_range(df), period)
    plus_di = 100.0 * _wilder_rma(plus_dm, period) / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * _wilder_rma(minus_dm, period) / atr_.replace(0.0, np.nan)

    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    adx = _wilder_rma(dx.dropna(), period).reindex(df.index)
    return adx, plus_di, minus_di

def enrich_with_indicators(df: pd.DataFrame, interval_str: str) -> pd.DataFrame:
    """
    Enrich dataframe with technical indicators
    Compatible with your data.py output format
    """
    if df.empty or 'Close' not in df.columns:
        return pd.DataFrame()
    
    print(f"Enriching data for interval: {interval_str}, {len(df)} rows.")
    df_enriched = df.copy()
    
    # Ensure we have all OHLCV columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df_enriched.columns:
            # Handle case variations (Close vs close)
            lower_col = col.lower()
            if lower_col in df_enriched.columns:
                df_enriched[col] = df_enriched[lower_col]
            else:
                df_enriched[col] = np.nan
    
    try:
        # Try using pandas_ta if available
        import pandas_ta as ta
        
        # 1. ADX (Length 9) - Momentum strength
        try:
            adx_results = ta.adx(df_enriched['High'], df_enriched['Low'], 
                                df_enriched['Close'], length=9)
            if adx_results is not None and not adx_results.empty:
                df_enriched['ADX_9'] = adx_results.iloc[:, 0]
            else:
                df_enriched['ADX_9'] = 25.0  # Default neutral
        except:
            df_enriched['ADX_9'] = 25.0
        
        # 2. RSI (Length 14) - Momentum oscillator
        try:
            rsi = ta.rsi(df_enriched['Close'], length=14)
            df_enriched['RSI_14'] = rsi if rsi is not None else 50.0
        except:
            df_enriched['RSI_14'] = calculate_rsi_manual(df_enriched['Close'], 14)
        
        # 3. MACD - Trend following
        try:
            macd = ta.macd(df_enriched['Close'], fast=12, slow=26, signal=9)
            if macd is not None and not macd.empty:
                # Find the histogram column
                for col in macd.columns:
                    if 'h' in col.lower() or 'hist' in col.lower():
                        df_enriched['MACDh_12_26_9'] = macd[col]
                        break
                else:
                    df_enriched['MACDh_12_26_9'] = 0
            else:
                df_enriched['MACDh_12_26_9'] = 0
        except:
            df_enriched['MACDh_12_26_9'] = 0
        
        # 4. EMA (Length 9) - Fast moving average
        try:
            ema = ta.ema(df_enriched['Close'], length=9)
            df_enriched['EMA_9'] = ema if ema is not None else df_enriched['Close'].ewm(span=9).mean()
        except:
            df_enriched['EMA_9'] = df_enriched['Close'].ewm(span=9, adjust=False).mean()
        
        # 5. ATR (Length 14) - Volatility
        try:
            atr = ta.atr(df_enriched['High'], df_enriched['Low'], 
                        df_enriched['Close'], length=14)
            df_enriched['ATR_14'] = atr if atr is not None else calculate_atr_manual(df_enriched, 14)
        except:
            df_enriched['ATR_14'] = calculate_atr_manual(df_enriched, 14)
        
        # 6. VWAP - Only for intraday
        if interval_str in ['15m', '5m', '1m', 'hourly']:
            try:
                vwap = ta.vwap(df_enriched['High'], df_enriched['Low'], 
                              df_enriched['Close'], df_enriched['Volume'])
                df_enriched['VWAP'] = vwap if vwap is not None else df_enriched['Close']
            except:
                df_enriched['VWAP'] = calculate_vwap_manual(df_enriched)
        
    except ImportError:
        # Expected path. pandas_ta is deliberately not a dependency (see
        # requirements.txt); this branch is the primary implementation, not a
        # degraded one, and indicators_test.py checks it against lab_indicators.py.
        print("using verified Wilder indicators (ADX/ATR/RSI)")
        # ADX_9 used to be the constant 25.0 here. It is gated below with `adx > 25`
        # and `adx > 20`, so a constant 25.0 made the first test permanently False and
        # the second permanently True -- trend filtering silently stopped existing.
        # wilder_adx/atr/rsi are checked against the live lab's proven implementation
        # by indicators_test.py.
        df_enriched['RSI_14'] = wilder_rsi(df_enriched['Close'], 14)
        adx9, plus_di, minus_di = wilder_adx(df_enriched, 9)
        df_enriched['ADX_9'] = adx9
        df_enriched['DMP_9'] = plus_di
        df_enriched['DMN_9'] = minus_di
        df_enriched['MACDh_12_26_9'] = calculate_macd_histogram_manual(df_enriched['Close'])
        df_enriched['EMA_9'] = df_enriched['Close'].ewm(span=9, adjust=False).mean()
        df_enriched['ATR_14'] = wilder_atr(df_enriched, 14)
        if interval_str in ['15m', '5m', '1m', 'hourly']:
            df_enriched['VWAP'] = calculate_vwap_manual(df_enriched)
    
    # Custom momentum indicators for your strategy
    
    # 7. Volume analysis
    df_enriched['volume_ma_20'] = df_enriched['Volume'].rolling(20).mean()
    df_enriched['volume_spike'] = df_enriched['Volume'] > (df_enriched['volume_ma_20'] * 2.0)
    df_enriched['volume_exhaustion'] = (
        df_enriched['Volume'].rolling(5).mean() < (df_enriched['volume_ma_20'] * 0.8)
    )
    
    # 8. Price momentum
    df_enriched['returns'] = df_enriched['Close'].pct_change()
    df_enriched['momentum_5'] = df_enriched['Close'] / df_enriched['Close'].shift(5) - 1
    df_enriched['momentum_10'] = df_enriched['Close'] / df_enriched['Close'].shift(10) - 1
    
    # 9. Volatility
    df_enriched['volatility'] = df_enriched['returns'].rolling(20).std() * np.sqrt(252)
    
    # 10. High-Low ratio (for gap detection)
    df_enriched['high_low_ratio'] = (
        (df_enriched['High'] - df_enriched['Low']) / df_enriched['Close']
    )
    
    # 11. Support/Resistance levels
    df_enriched['resistance'] = df_enriched['High'].rolling(20).max()
    df_enriched['support'] = df_enriched['Low'].rolling(20).min()
    
    # 12. Trend strength
    sma_20 = df_enriched['Close'].rolling(20).mean()
    sma_50 = df_enriched['Close'].rolling(50).mean()
    df_enriched['trend_strength'] = (sma_20 - sma_50) / sma_50 * 100
    
    # Clean up
    # .bfill()/.ffill(), not fillna(method=...): the method= argument was deprecated in
    # pandas 2.1 and REMOVED in 3.0. This runs on every request, so on pandas 3 the image
    # would build clean and then raise TypeError on every analyze -- a failure that only
    # appears at request time, which is the hardest kind to catch before a user does.
    df_enriched.bfill(inplace=True)
    df_enriched.ffill(inplace=True)
    df_enriched.fillna(0, inplace=True)
    
    return df_enriched

def identify_current_setup(df: pd.DataFrame, timeframe_str: str) -> dict:
    """
    Identify current market setup for trading decisions
    Returns dict compatible with enhanced_api.py expectations
    """
    if df.empty or len(df) < 2:
        return {
            "timeframe": timeframe_str,
            "direction": "neutral",
            "adx": 0,
            "rsi": 50,
            "gap_risk": "unknown",
            "volume_spike": False,
            "volume_exhaustion": False,
            "error": "Insufficient data"
        }
    
    # Get latest values
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    # Determine direction
    if latest.get('Close', 0) > prev.get('Close', 0):
        direction = "up"
    elif latest.get('Close', 0) < prev.get('Close', 0):
        direction = "down"
    else:
        direction = "neutral"
    
    # Gap risk assessment
    atr_val = latest.get('ATR_14', 0)
    close_val = latest.get('Close', 0)
    gap_risk = "low"
    
    if close_val > 0 and atr_val > 0:
        atr_percentage = atr_val / close_val
        if atr_percentage > 0.02:
            gap_risk = "high"
        elif atr_percentage > 0.01:
            gap_risk = "moderate"
    
    # Momentum assessment
    rsi = latest.get('RSI_14', 50)
    momentum_5 = latest.get('momentum_5', 0)
    momentum_10 = latest.get('momentum_10', 0)
    
    # Trend assessment
    trend = "neutral"
    if latest.get('EMA_9', 0) > latest.get('Close', 0):
        trend = "bearish"
    elif latest.get('EMA_9', 0) < latest.get('Close', 0):
        trend = "bullish"
    
    # Volume analysis
    volume_spike = bool(latest.get('volume_spike', False))
    volume_exhaustion = bool(latest.get('volume_exhaustion', False))
    
    # Support/Resistance proximity
    close = latest.get('Close', 0)
    resistance = latest.get('resistance', close * 1.02)
    support = latest.get('support', close * 0.98)
    
    near_resistance = (resistance - close) / close < 0.005  # Within 0.5%
    near_support = (close - support) / close < 0.005
    
    # Build setup dictionary
    setup = {
        "timeframe": timeframe_str,
        "direction": direction,
        "adx": round(latest.get('ADX_9', 0), 2),
        "rsi": round(rsi, 2),
        "gap_risk": gap_risk,
        "volume_spike": volume_spike,
        "volume_exhaustion": volume_exhaustion,
        "trend": trend,
        "momentum_5": round(momentum_5 * 100, 2),  # As percentage
        "momentum_10": round(momentum_10 * 100, 2),
        "volatility": round(latest.get('volatility', 0), 4),
        "near_resistance": near_resistance,
        "near_support": near_support,
        "macd_histogram": round(latest.get('MACDh_12_26_9', 0), 4)
    }
    
    # Add timeframe-specific signals
    if timeframe_str == "15m":
        setup["scalp_ready"] = (
            volume_spike and 
            abs(momentum_5) > 0.005 and 
            30 < rsi < 70
        )
    elif timeframe_str == "hourly":
        setup["swing_ready"] = (
            trend != "neutral" and
            not volume_exhaustion and
            20 < rsi < 80
        )
    elif timeframe_str == "daily":
        setup["position_ready"] = (
            latest.get('ADX_9', 0) > 25 and
            trend != "neutral"
        )
    
    return setup

# Manual calculation functions (fallbacks)

def calculate_rsi_manual(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Manual RSI calculation"""
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)

def calculate_atr_manual(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Manual ATR calculation"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr.fillna(0)

def calculate_macd_histogram_manual(close_prices: pd.Series) -> pd.Series:
    """Manual MACD histogram calculation"""
    ema_12 = close_prices.ewm(span=12, adjust=False).mean()
    ema_26 = close_prices.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return histogram.fillna(0)

def calculate_vwap_manual(df: pd.DataFrame) -> pd.Series:
    """Manual VWAP calculation"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cumulative_tpv = (typical_price * df['Volume']).cumsum()
    cumulative_volume = df['Volume'].cumsum()
    vwap = cumulative_tpv / cumulative_volume
    
    return vwap.fillna(df['Close'])

# Additional helper functions for agent.py and other modules

def get_momentum_signals(df: pd.DataFrame) -> dict:
    """
    Get momentum signals for the agent
    Used by agent.py for quick decisions
    """
    if df.empty or len(df) < 20:
        return {"signal": "NEUTRAL", "strength": 0}
    
    latest = df.iloc[-1]
    
    # Check multiple momentum conditions
    rsi = latest.get('RSI_14', 50)
    momentum_5 = latest.get('momentum_5', 0)
    volume_spike = latest.get('volume_spike', False)
    macd_hist = latest.get('MACDh_12_26_9', 0)
    
    # Bullish signals
    if (rsi > 55 and momentum_5 > 0.01 and volume_spike and macd_hist > 0):
        return {"signal": "BULLISH", "strength": 0.8}
    elif (rsi > 50 and momentum_5 > 0.005):
        return {"signal": "BULLISH", "strength": 0.6}
    
    # Bearish signals
    elif (rsi < 45 and momentum_5 < -0.01 and volume_spike and macd_hist < 0):
        return {"signal": "BEARISH", "strength": 0.8}
    elif (rsi < 50 and momentum_5 < -0.005):
        return {"signal": "BEARISH", "strength": 0.6}
    
    # Neutral
    else:
        return {"signal": "NEUTRAL", "strength": 0.3}

def calculate_entry_signals(df: pd.DataFrame, timeframe: str) -> dict:
    """
    Calculate specific entry signals for different timeframes
    Used by enhanced_api.py for options strategies
    """
    if df.empty:
        return {"entry": False, "confidence": 0}
    
    setup = identify_current_setup(df, timeframe)
    
    # Timeframe-specific entry logic
    if timeframe in ["1m", "5m"]:
        # Scalping entries
        entry = (
            setup.get('volume_spike', False) and
            abs(setup.get('momentum_5', 0)) > 0.5 and
            30 < setup.get('rsi', 50) < 70
        )
        confidence = 70 if entry else 30
        
    elif timeframe == "15m":
        # Momentum entries
        entry = (
            setup.get('direction') != 'neutral' and
            setup.get('adx', 0) > 25 and
            not setup.get('volume_exhaustion', False)
        )
        confidence = 75 if entry else 40
        
    else:  # Daily/Hourly
        # Swing entries
        entry = (
            setup.get('trend') != 'neutral' and
            20 < setup.get('rsi', 50) < 80 and
            setup.get('adx', 0) > 20
        )
        confidence = 80 if entry else 35
    
    return {
        "entry": entry,
        "confidence": confidence,
        "setup_details": setup
    }