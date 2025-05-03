import pandas as pd, pandas_ta as ta
def enrich(df: pd.DataFrame):
    # Make a copy and ensure index is DatetimeIndex
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Ensure all columns are numeric with explicit conversion
    for col in ['High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    try:
        # Explicitly call ADX with error handling
        adx = df.ta.adx(high=df['High'], low=df['Low'], close=df['Close'])
        if adx is not None:
            df['ADX'] = adx.iloc[:, 0]
            df['DMI_Plus'] = adx.iloc[:, 1]
            df['DMI_Minus'] = adx.iloc[:, 2]
        else:
            # Fallback if calculation fails
            df['ADX'] = 0.0
            df['DMI_Plus'] = 0.0
            df['DMI_Minus'] = 0.0
    except Exception as e:
        print(f"ADX calculation error: {e}")
        # Provide default values to prevent downstream errors
        df['ADX'] = 0.0
        df['DMI_Plus'] = 0.0
        df['DMI_Minus'] = 0.0
    
    # Calculate EMAs with explicit Series objects, not string names
    try:
        df.ta.ema(close=df['Close'], length=9, append=True, col_names=("EMA_9_Close",))
        df.ta.ema(close=df['Volume'], length=9, append=True, col_names=("EMA_9_Volume",))
    except Exception as e:
        print(f"EMA calculation error: {e}")
        # Provide default values
        df['EMA_9_Close'] = df['Close'].rolling(9).mean()
        df['EMA_9_Volume'] = df['Volume'].rolling(9).mean()
    
    return df

