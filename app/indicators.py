import pandas as pd
import pandas_ta as ta 
import numpy as np

# --- Paste the new calculate_momentum_indicators function here ---
def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum-specific indicators that proved effective in our trading."""
    if df is None or df.empty:
        return pd.DataFrame()
    df_mom = df.copy()
    try:
        # 1. MACD with histogram analysis (more detailed than standard MACD)
        # Ensure 'Close' column exists and is numeric
        if 'Close' in df_mom.columns and pd.api.types.is_numeric_dtype(df_mom['Close']):
            df_mom.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
        else:
            print("Warning: 'Close' column not found or not numeric for MACD calculation.")
            # Add placeholder columns if MACD can't be calculated to avoid downstream errors
            for col_name in [f'MACD_{12}_{26}_{9}', f'MACDh_{12}_{26}_{9}', f'MACDs_{12}_{26}_{9}']:
                 df_mom[col_name] = 0.0


        # 2. RSI with overbought/oversold levels and divergence detection
        # Ensure 'Close' column exists and is numeric
        if 'Close' in df_mom.columns and pd.api.types.is_numeric_dtype(df_mom['Close']):
            df_mom.ta.rsi(close='Close', length=14, append=True) # Default RSI column name is RSI_14
        else:
            print("Warning: 'Close' column not found or not numeric for RSI calculation.")
            df_mom['RSI_14'] = 50.0 # Default neutral RSI


        # 3. Calculate RSI divergence (sophisticated version)
        if 'Close' in df_mom.columns and 'RSI_14' in df_mom.columns:
            window = 5  # Look back 5 bars for local maxima
            # Ensure sufficient data for rolling window
            if len(df_mom) > window:
                df_mom['price_high'] = df_mom['Close'].rolling(window=window, center=True, min_periods=1).apply(
                    lambda x: 1 if x.iloc[len(x)//2] == max(x) else 0, raw=True) # Adjusted for center=True and potential edge cases

                df_mom['rsi_divergence'] = 0
                for i in range(window, len(df_mom)-window): # Ensure index is within bounds
                    if df_mom['price_high'].iloc[i] == 1:
                        # Find previous price high, handle potential issues if no previous high
                        prev_high_indices = df_mom['price_high'].iloc[:i][df_mom['price_high'].iloc[:i] == 1].index
                        if not prev_high_indices.empty:
                            prev_high_idx_loc = prev_high_indices[-1] # Take the most recent previous high
                            # Ensure indices are valid before comparing
                            if df_mom['Close'].iloc[i] > df_mom['Close'].loc[prev_high_idx_loc] and \
                               df_mom['RSI_14'].iloc[i] < df_mom['RSI_14'].loc[prev_high_idx_loc]:
                                df_mom.loc[df_mom.index[i], 'rsi_divergence'] = -1  # Bearish divergence
            else:
                df_mom['price_high'] = 0
                df_mom['rsi_divergence'] = 0
        else:
            df_mom['price_high'] = 0
            df_mom['rsi_divergence'] = 0


        # 4. Volume analysis relative to price action
        if 'Volume' in df_mom.columns and 'Close' in df_mom.columns and pd.api.types.is_numeric_dtype(df_mom['Volume']) and pd.api.types.is_numeric_dtype(df_mom['Close']):
            # Avoid division by zero if 'Close' can be zero
            df_mom['volume_price_ratio'] = df_mom['Volume'] / (df_mom['Close'].replace(0, pd.NA)) 
            df_mom['volume_trend'] = df_mom['volume_price_ratio'].pct_change(3)
        else:
            print("Warning: 'Volume' or 'Close' column not found or not numeric for volume analysis.")
            df_mom['volume_price_ratio'] = 0.0
            df_mom['volume_trend'] = 0.0
        
        # Fill any NaNs created by indicators, especially at the beginning of the series
        df_mom.fillna(0.0, inplace=True)


        return df_mom
    except Exception as e:
        print(f"Error calculating momentum indicators: {e}")
        # Return the original dataframe columns plus any successfully added ones, fill others with 0
        existing_cols = df.columns.tolist()
        new_cols = [col for col in df_mom.columns if col not in existing_cols]
        for col in new_cols: # if a new col was partially created but failed
            if col not in df_mom:
                 df_mom[col] = 0.0
        df_mom.fillna(0.0, inplace=True)
        return df_mom

# --- Paste the new identify_gap_patterns function here ---
def identify_gap_patterns(df: pd.DataFrame) -> dict:
    """Identify gap patterns that we used in our successful trades."""
    if df is None or df.empty or len(df) < 2: # Ensure ADX and RSI_14 exist from enrich/calculate_momentum_indicators
        return {"gaps": [], "current_setup": None}
    
    # Ensure required columns exist from previous processing steps
    required_cols = ['Open', 'Close', 'Volume', 'ADX', 'RSI_14']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame for gap pattern identification. Filling with 0 or default.")
            if col == 'RSI_14':
                df[col] = 50.0 # Default neutral RSI
            else:
                df[col] = 0.0


    gaps = []
    for i in range(1, len(df)):
        # Calculate gap percentage
        # Ensure previous close is not zero to avoid division by zero
        prev_close = df['Close'].iloc[i-1]
        if prev_close == 0:
            gap_pct = 0.0
        else:
            gap_pct = (df['Open'].iloc[i] - prev_close) / prev_close * 100


        if abs(gap_pct) >= 0.2: # Only consider significant gaps (>= 0.2%)
            gap_type = "up" if gap_pct > 0 else "down"


            # Analyze follow-through
            # Ensure current open is not zero for day_change calculation
            current_open = df['Open'].iloc[i]
            if current_open == 0:
                day_change = 0.0
            else:
                day_change = (df['Close'].iloc[i] - current_open) / current_open * 100
            
            follow_through = "filled" if (gap_type == "up" and day_change < 0) or \
                                        (gap_type == "down" and day_change > 0) else "extended"


            # Calculate volume_vs_avg, ensure there are enough prior days for mean calculation
            # And handle cases where mean volume might be zero
            vol_mean_period = df['Volume'].iloc[max(0, i-20):i].mean()
            volume_vs_avg_val = df['Volume'].iloc[i] / vol_mean_period if vol_mean_period else 0


            gaps.append({
                "date": df.index[i].strftime('%Y-%m-%d') if isinstance(df.index[i], pd.Timestamp) else str(df.index[i]), # Format date
                "gap_type": gap_type,
                "gap_pct": gap_pct,
                "follow_through": follow_through,
                "volume_vs_avg": volume_vs_avg_val
            })


    # Analyze current setup (most recent completed day)
    current_setup = None
    if len(df) >= 2:
        # Use -1 for the latest complete data point if data is up-to-date
        # Or -2 if the last row is incomplete (live data)
        # Assuming df passed contains finalized historical data for completed days
        latest_idx_pos = len(df) - 1 


        # Look for trend streak (consecutive up/down days)
        streak = 0
        direction_val = 0 # 0 for flat, 1 for up, -1 for down


        if len(df) >= 2: # Need at least two days to determine a streak start
            if df['Close'].iloc[latest_idx_pos] > df['Close'].iloc[latest_idx_pos-1]:
                direction_val = 1
                streak = 1
            elif df['Close'].iloc[latest_idx_pos] < df['Close'].iloc[latest_idx_pos-1]:
                direction_val = -1
                streak = 1
            else: # Prices are the same
                 direction_val = 0
                 streak = 0 # Or 1, depending on definition. Let's say 0 for no change.


        # Iterate backwards to count the streak
        if streak > 0 : # Only count if there's an initial direction
            for i in range(latest_idx_pos - 1, 0, -1): # Iterate from second to last back to start of df
                current_day_close = df['Close'].iloc[i]
                prev_day_close = df['Close'].iloc[i-1]
                if (direction_val == 1 and current_day_close > prev_day_close) or \
                   (direction_val == -1 and current_day_close < prev_day_close):
                    streak += 1
                else:
                    break # Streak broken


        adx_val = df['ADX'].iloc[latest_idx_pos] if 'ADX' in df.columns else 0.0
        rsi_val = df['RSI_14'].iloc[latest_idx_pos] if 'RSI_14' in df.columns else 50.0


        gap_risk_val = "high" if (
            (streak >=3 and rsi_val > 70) or 
            (streak >=4 and adx_val > 25)
        ) else "moderate"


        current_setup = {
            "direction": "up" if direction_val == 1 else ("down" if direction_val == -1 else "flat"),
            "streak": streak,
            "adx": adx_val,
            "rsi": rsi_val,
            "gap_risk": gap_risk_val
        }
    return {"gaps": gaps, "current_setup": current_setup}


# --- Modify the existing enrich function ---
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df_enriched = df.copy()

    if not isinstance(df_enriched.index, pd.DatetimeIndex):
        df_enriched.index = pd.to_datetime(df_enriched.index)

    required_cols = ['High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col in df_enriched.columns:
            df_enriched[col] = pd.to_numeric(df_enriched[col], errors='coerce')
        else:
            print(f"Warning: Essential column '{col}' missing. Initializing with NaN.")
            df_enriched[col] = np.nan # Initialize if missing, will be handled by fillna/dropna

    # Crucial: Handle NaNs from coercion or missing initial data BEFORE TA Libs
    df_enriched.fillna(method='ffill', inplace=True) # Forward fill first
    df_enriched.dropna(subset=['High', 'Low', 'Close'], inplace=True) # Drop rows if HLC still NaN

    if df_enriched.empty:
        print("DataFrame is empty after NaN handling. Cannot calculate indicators.")
        # Return an empty DataFrame but with expected columns for graceful failure downstream
        for col_name in ['ADX_9', 'DMP_9', 'DMN_9', 'ADX', 'DMI_Plus', 'DMI_Minus', 
                         'EMA_9_Close', 'EMA_9_Volume', 'RSI_14', 
                         'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 
                         'price_high', 'rsi_divergence']:
            if col_name not in df_enriched.columns:
                 df_enriched[col_name] = np.nan
        return df_enriched

    # 1. ADX/DMI Calculation (Length 9, EMA)
    try:
        adx_results = df_enriched.ta.adx(
            high=df_enriched['High'], low=df_enriched['Low'], close=df_enriched['Close'],
            length=9, mamode="ema", append=False # Explicitly use EMA
        )
        if adx_results is not None and not adx_results.empty:
            # pandas-ta names columns with length, e.g., ADX_9, DMP_9, DMN_9
            df_enriched['ADX_9'] = adx_results[f'ADX_9']
            df_enriched['DMP_9'] = adx_results[f'DMP_9']
            df_enriched['DMN_9'] = adx_results[f'DMN_9']
            # For compatibility with api.py potentially looking for generic names first
            df_enriched['ADX'] = adx_results[f'ADX_9']
            df_enriched['DMI_Plus'] = adx_results[f'DMP_9']
            df_enriched['DMI_Minus'] = adx_results[f'DMN_9']
        else:
            print("ADX calculation returned None or empty. Setting defaults.")
            for col in ['ADX_9', 'DMP_9', 'DMN_9', 'ADX', 'DMI_Plus', 'DMI_Minus']: df_enriched[col] = 0.0
    except Exception as e:
        print(f"ADX calculation error: {e}")
        for col in ['ADX_9', 'DMP_9', 'DMN_9', 'ADX', 'DMI_Plus', 'DMI_Minus']: df_enriched[col] = 0.0

    # 2. EMA Calculations
    try:
        df_enriched.ta.ema(close=df_enriched['Close'], length=9, append=True, col_names=("EMA_9_Close",))
        df_enriched.ta.ema(close=df_enriched['Volume'], length=9, append=True, col_names=("EMA_9_Volume",))
    except Exception as e:
        print(f"EMA calculation error: {e}")
        df_enriched['EMA_9_Close'] = 0.0
        df_enriched['EMA_9_Volume'] = 0.0

    # 3. RSI Calculation (Length 14)
    try:
        df_enriched.ta.rsi(close=df_enriched['Close'], length=14, append=True) # Adds 'RSI_14'
    except Exception as e:
        print(f"RSI calculation error: {e}")
        if 'RSI_14' not in df_enriched.columns: df_enriched['RSI_14'] = 50.0

    # 4. MACD Calculation (12, 26, 9)
    try:
        df_enriched.ta.macd(close=df_enriched['Close'], fast=12, slow=26, signal=9, append=True)
        # Appends MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        if 'MACDh_12_26_9' not in df_enriched.columns: # Check if primary output column is missing
            print("MACD calculation did not append columns correctly. Setting defaults.")
            for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']: df_enriched[col] = 0.0
    except Exception as e:
        print(f"MACD calculation error: {e}")
        for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']: df_enriched[col] = 0.0
        
    # 5. RSI Divergence Detection (Simplified for robustness, can be expanded)
    window = 5
    if 'Close' in df_enriched.columns and 'RSI_14' in df_enriched.columns and len(df_enriched) >= window + 2:
        try:
            df_enriched['price_high'] = df_enriched['Close'].rolling(window=window, center=True, min_periods=1).apply(
                lambda x: 1 if not x.empty and len(x) >= (window // 2 + 1) and x.iloc[len(x) // 2] == x.max() else 0,
                raw=False
            )
            df_enriched['rsi_divergence'] = 0
            for i in range(window, len(df_enriched) - window):
                if i < 0 or i >= len(df_enriched): continue
                if df_enriched['price_high'].iloc[i] == 1:
                    prev_highs_series = df_enriched['price_high'].iloc[:i]
                    if prev_highs_series[prev_highs_series == 1].any():
                        prev_high_idx_label = prev_highs_series[prev_highs_series == 1].index[-1]
                        if (df_enriched['Close'].iloc[i] > df_enriched['Close'].loc[prev_high_idx_label] and
                            df_enriched['RSI_14'].iloc[i] < df_enriched['RSI_14'].loc[prev_high_idx_label]):
                            df_enriched.loc[df_enriched.index[i], 'rsi_divergence'] = -1
        except Exception as e:
            print(f"RSI Divergence calculation error: {e}")
            if 'price_high' not in df_enriched.columns: df_enriched['price_high'] = 0
            if 'rsi_divergence' not in df_enriched.columns: df_enriched['rsi_divergence'] = 0
    else: # Not enough data or columns for divergence
        if 'price_high' not in df_enriched.columns: df_enriched['price_high'] = 0
        if 'rsi_divergence' not in df_enriched.columns: df_enriched['rsi_divergence'] = 0

    df_enriched.fillna(0.0, inplace=True) # Final fill for any indicator-introduced NaNs

    # Debug print
    cols_to_check = ['ADX_9', 'DMP_9', 'DMN_9', 'RSI_14', 'MACDh_12_26_9']
    existing_cols = [col for col in cols_to_check if col in df_enriched.columns]
    if not df_enriched.empty and existing_cols:
        print("--- Tail of enriched_df in indicators.enrich ---")
        print(df_enriched[existing_cols].tail())
        print("-------------------------------------------------")
    return df_enriched

def identify_gap_patterns(df: pd.DataFrame) -> dict:
    # (Your existing identify_gap_patterns function - ensure it uses the correct ADX/RSI column names
    # e.g., df['ADX_9'] if that's what enrich now consistently produces and api.py passes)
    if df is None or df.empty or len(df) < 2:
        return {"gaps": [], "current_setup": None}
    
    # Ensure ADX and RSI columns exist, using the names produced by enrich
    adx_col_name = 'ADX_9' # or 'ADX' if enrich also creates that consistently
    rsi_col_name = 'RSI_14'

    required_cols = ['Open', 'Close', 'Volume', adx_col_name, rsi_col_name]
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame for gap pattern identification. Filling with default.")
            if col == rsi_col_name: df[col] = 50.0
            else: df[col] = 0.0
    # ... rest of your identify_gap_patterns logic ...
    # Make sure to use adx_col_name and rsi_col_name when accessing these indicators in this function.
    # For example:
    # adx_val = df[adx_col_name].iloc[latest_idx_pos] if adx_col_name in df.columns else 0.0
    # rsi_val = df[rsi_col_name].iloc[latest_idx_pos] if rsi_col_name in df.columns else 50.0

    # Placeholder for the rest of identify_gap_patterns logic
    # This is just to show where to adjust column names.
    # Your full logic for calculating gaps and current_setup should be here.
    latest_idx_pos = len(df) - 1
    current_setup = {
        "direction": "flat", "streak": 0,
        "adx": df[adx_col_name].iloc[latest_idx_pos] if adx_col_name in df.columns else 0.0,
        "rsi": df[rsi_col_name].iloc[latest_idx_pos] if rsi_col_name in df.columns else 50.0,
        "gap_risk": "moderate"
    }
    if len(df) >=2: # Basic direction
        if df['Close'].iloc[latest_idx_pos] > df['Close'].iloc[latest_idx_pos -1]:
            current_setup['direction'] = "up"
        elif df['Close'].iloc[latest_idx_pos] < df['Close'].iloc[latest_idx_pos -1]:
            current_setup['direction'] = "down"
    # (Your full streak and gap_risk logic needs to be here)

    return {"gaps": [], "current_setup": current_setup} # Replace with your actual gap list