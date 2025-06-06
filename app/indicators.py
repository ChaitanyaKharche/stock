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
def enrich(df: pd.DataFrame, interval_str: str = "1d") -> pd.DataFrame:
    """
    Enriches the DataFrame with technical indicators.
    Uses ADX Length 9 as per user's chart preference. Adds ATR_14.
    """
    df_enriched = df.copy()
    if df_enriched.empty: return df_enriched

    if not isinstance(df_enriched.index, pd.DatetimeIndex):
        # Attempt conversion, handle errors if index isn't datetime-like
        try:
            df_enriched.index = pd.to_datetime(df_enriched.index)
        except Exception as e:
            print(f"Error converting index to DatetimeIndex for {interval_str}: {e}. Cannot enrich.")
            return pd.DataFrame() # Return empty if index is bad

    required_cols = ['High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col in df_enriched.columns:
            df_enriched[col] = pd.to_numeric(df_enriched[col], errors='coerce')
        else:
            print(f"Warning: Required column '{col}' missing for {interval_str}. Initializing with NaN.")
            df_enriched[col] = np.nan # Initialize missing required columns

    # Forward fill first to handle potential gaps before dropping rows missing HLC
    df_enriched.ffill(inplace=True)
    # Drop rows where essential price data is still missing AFTER ffill
    df_enriched.dropna(subset=['High', 'Low', 'Close'], inplace=True)

    if df_enriched.empty:
        print(f"DataFrame empty after initial processing for {interval_str}. Cannot enrich.")
        return df_enriched

    print(f"Enriching data for interval: {interval_str}, {len(df_enriched)} rows.")

    # Ensure HLC are float for calculations
    for col in ['High', 'Low', 'Close']:
        df_enriched[col] = df_enriched[col].astype(float)
    if 'Volume' in df_enriched.columns:
         df_enriched['Volume'] = df_enriched['Volume'].astype(float) # Ensure volume is float for EMA calc


    # --- Indicator Calculations ---
    # 1. ADX/DMI (Length 9, EMA)
    try:
        # Ensure HLC exist before calling ADX
        if all(c in df_enriched.columns for c in ['High', 'Low', 'Close']):
            adx_results = df_enriched.ta.adx(length=9, mamode="ema", append=False) # Keep append=False
            if adx_results is not None and not adx_results.empty:
                # Selectively join results to avoid column conflicts
                df_enriched = df_enriched.join(adx_results[[f'ADX_9', f'DMP_9', f'DMN_9']])
                # Generic names mapping
                df_enriched['ADX'] = df_enriched[f'ADX_9']
                df_enriched['DMI_Plus'] = df_enriched[f'DMP_9']
                df_enriched['DMI_Minus'] = df_enriched[f'DMN_9']
            else:
                raise ValueError("ADX calculation returned empty results.")
        else:
             raise ValueError("Missing High, Low, or Close columns for ADX.")
    except Exception as e:
        print(f"Error calculating ADX_9 ({interval_str}): {e}")
        # Initialize columns to 0.0 if calculation fails
        for col in ['ADX_9', 'DMP_9', 'DMN_9', 'ADX', 'DMI_Plus', 'DMI_Minus']:
            df_enriched[col] = 0.0

    # 2. EMAs
    try:
        # Ensure Close and Volume exist
        if 'Close' in df_enriched.columns:
            df_enriched.ta.ema(close='Close', length=9, append=True, col_names=('EMA_9_Close',))
        else: df_enriched['EMA_9_Close'] = 0.0
        if 'Volume' in df_enriched.columns:
             df_enriched.ta.ema(close='Volume', length=9, append=True, col_names=('EMA_9_Volume',))
        else: df_enriched['EMA_9_Volume'] = 0.0
    except Exception as e:
        print(f"Error calculating EMAs ({interval_str}): {e}")
        if 'EMA_9_Close' not in df_enriched.columns: df_enriched['EMA_9_Close'] = 0.0
        if 'EMA_9_Volume' not in df_enriched.columns: df_enriched['EMA_9_Volume'] = 0.0

    # 3. RSI (Length 14)
    try:
        if 'Close' in df_enriched.columns:
            df_enriched.ta.rsi(close='Close', length=14, append=True) # Adds 'RSI_14'
        else: df_enriched['RSI_14'] = 50.0 # Default neutral
    except Exception as e:
        print(f"Error calculating RSI_14 ({interval_str}): {e}")
        df_enriched['RSI_14'] = 50.0

    # 4. MACD (12, 26, 9)
    try:
        if 'Close' in df_enriched.columns:
             df_enriched.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
             # Appends MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        else:
            for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']: df_enriched[col] = 0.0
    except Exception as e:
        print(f"Error calculating MACD ({interval_str}): {e}")
        for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
            if col not in df_enriched.columns: df_enriched[col] = 0.0

    # *** NEW: 5. ATR (Average True Range, Length 14) ***
    print(f"DEBUG ({interval_str}): Attempting ATR calculation (rename & join). Columns: {df_enriched.columns.tolist()}")
    try:
        # Ensure HLC exist and are numeric float types
        required_atr_cols = ['High', 'Low', 'Close']
        if all(c in df_enriched.columns for c in required_atr_cols):
            # Double check types just before calculation
            for col in required_atr_cols:
                 if not pd.api.types.is_numeric_dtype(df_enriched[col]):
                     raise TypeError(f"Column '{col}' is not numeric for ATR, type is {df_enriched[col].dtype}")

            print(f"DEBUG ({interval_str}): HLC columns verified numeric. Calling ta.atr with append=False...")
            # Calculate ATR separately
            atr_series = df_enriched.ta.atr(length=14, append=False) # Returns pd.Series, potentially named 'ATRr_14' or similar

            if atr_series is not None and isinstance(atr_series, pd.Series):
                # *** FIX: Rename the series to the expected name ***
                original_atr_name = atr_series.name
                expected_atr_name = 'ATR_14'
                atr_series.name = expected_atr_name
                print(f"DEBUG ({interval_str}): ta.atr returned Series (Original Name: {original_atr_name}). Renamed to: {atr_series.name}. Length: {len(atr_series)}. Joining...")

                # Explicitly join the RENAMED Series to the DataFrame
                df_enriched = df_enriched.join(atr_series)

                # Verify join using the EXPECTED name
                if expected_atr_name in df_enriched.columns:
                     print(f"DEBUG ({interval_str}): '{expected_atr_name}' column successfully joined. NaN count: {df_enriched[expected_atr_name].isnull().sum()}")
                else:
                     # This error should ideally not happen now
                     print(f"ERROR ({interval_str}): Failed to join '{expected_atr_name}' Series to DataFrame after renaming!")
                     df_enriched[expected_atr_name] = 0.0 # Fallback if join fails unexpectedly
            else:
                 print(f"ERROR ({interval_str}): ta.atr(append=False) did not return a valid Series.")
                 df_enriched['ATR_14'] = 0.0 # Fallback if calculation fails

        else:
            print(f"WARNING ({interval_str}): Missing HLC columns, cannot calculate ATR.")
            df_enriched['ATR_14'] = 0.0 # Set default

    except Exception as e:
        # Keep detailed error reporting
        print("\n" + "*"*20 + f" ERROR DURING ATR CALCULATION/RENAME/JOIN ({interval_str}) " + "*"*20)
        print(f"ERROR ({interval_str}): Failed to process ATR_14. Error Type: {type(e).__name__}, Message: {e}")
        import traceback
        print(traceback.format_exc())
        print("*"*60 + "\n")
        # Ensure column exists even on error, assign default using the expected name
        if 'ATR_14' not in df_enriched.columns:
             df_enriched['ATR_14'] = 0.0

    # Check final state of ATR_14 before moving on
    if 'ATR_14' in df_enriched.columns:
        print(f"DEBUG ({interval_str}): ATR_14 column exists before final fillna. Sample value (last): {df_enriched['ATR_14'].iloc[-1] if not df_enriched.empty else 'N/A'}")
    else:
        print(f"ERROR ({interval_str}): ATR_14 column STILL missing before final fillna!")
        df_enriched['ATR_14'] = 0.0 # Ensure it exists for final fillna

    # 6. RSI Divergence (Placeholder - Add full logic if needed)
    df_enriched['rsi_divergence'] = 0 # Placeholder

    # Final fillna for any remaining NaNs introduced by indicators (at the start of series)
    df_enriched.fillna(0.0, inplace=True)
    return df_enriched


# --- Keep the second identify_gap_patterns function, it is used by api.py ---
# --- You might want to implement the 'streak_val' calculation properly ---
def identify_gap_patterns(df: pd.DataFrame, timeframe_str: str = "daily") -> dict:
    """
    Identifies significant gaps based on timeframe-specific thresholds
    and analyzes current setup from the last complete bar.
    Expects DataFrame with 'Open', 'Close', 'Volume', 'ADX_9', 'RSI_14'.
    The 'df' passed should be an ENRICHED DataFrame.
    """
    if df is None or df.empty or len(df) < 2:
        return {"gaps": [], "current_setup": None}

    gap_thresholds = {
        'daily': 0.5,  # %
        'hourly': 0.3, # %
        '15m': 0.2,    # %
        'default': 0.2 # Default if timeframe_str not matched
    }
    threshold = gap_thresholds.get(timeframe_str, gap_thresholds['default'])

    base_cols = ['Open', 'Close'] # Volume is needed later for avg calc
    for col in base_cols:
        if col not in df.columns:
            print(f"Warning: Gap patterns - missing base column '{col}' for timeframe {timeframe_str}. Returning empty.")
            return {"gaps": [], "current_setup": None}
        # Ensure numeric after enrichment
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=base_cols, inplace=True) # Drop rows if O/C became NaN

    if df.empty or len(df) < 2: # Check again after potential dropna
         return {"gaps": [], "current_setup": None}

    df_gaps = df.copy()
    df_gaps['prev_Close'] = df_gaps['Close'].shift(1)
    # Avoid division by zero if prev_Close is 0 or NaN
    df_gaps['gap_pct'] = np.where(
        df_gaps['prev_Close'].isnull() | (df_gaps['prev_Close'] == 0),
        0.0,
        (df_gaps['Open'] - df_gaps['prev_Close']) / df_gaps['prev_Close'] * 100
    )

    significant_gaps_df = df_gaps[df_gaps['gap_pct'].abs() >= threshold].copy()

    gaps_list = []
    if not significant_gaps_df.empty:
        significant_gaps_df['gap_type'] = np.where(significant_gaps_df['gap_pct'] > 0, "up", "down")

        # Avoid division by zero for day_change_pct
        significant_gaps_df['day_change_pct'] = np.where(
            significant_gaps_df['Open'].isnull() | (significant_gaps_df['Open'] == 0),
             0.0,
            (significant_gaps_df['Close'] - significant_gaps_df['Open']) / significant_gaps_df['Open'] * 100
        )

        conditions_filled = [
            (significant_gaps_df['gap_type'] == "up") & (significant_gaps_df['day_change_pct'] < 0),
            (significant_gaps_df['gap_type'] == "down") & (significant_gaps_df['day_change_pct'] > 0)
        ]
        significant_gaps_df['follow_through'] = np.select(conditions_filled, ["filled", "filled"], default="extended")

        # Volume vs average (20-period) - Ensure 'Volume' exists and is numeric
        if 'Volume' in df_gaps.columns:
            df_gaps['Volume'] = pd.to_numeric(df_gaps['Volume'], errors='coerce').fillna(0)
            # Calculate rolling mean on the *original* index before filtering for gaps
            df_gaps['avg_vol_20'] = df_gaps['Volume'].shift(1).rolling(window=20, min_periods=5).mean().fillna(0)
            # Join the calculated average volume onto the significant_gaps_df using the index
            significant_gaps_df = significant_gaps_df.join(df_gaps['avg_vol_20'])
            significant_gaps_df['volume_vs_avg'] = np.where(
                 significant_gaps_df['avg_vol_20'].isnull() | (significant_gaps_df['avg_vol_20'] <= 0), # Check for null or zero/negative
                 0.0,
                 significant_gaps_df['Volume'] / significant_gaps_df['avg_vol_20']
            )
        else:
            significant_gaps_df['volume_vs_avg'] = 0.0

        for idx, row in significant_gaps_df.iterrows():
            gaps_list.append({
                "date": idx.strftime('%Y-%m-%d %H:%M:%S') if isinstance(idx, pd.Timestamp) else str(idx),
                "gap_type": row['gap_type'],
                "gap_pct": round(row['gap_pct'], 2) if pd.notna(row['gap_pct']) else 0.0,
                "follow_through": row['follow_through'],
                "volume_vs_avg": round(row.get('volume_vs_avg', 0.0), 2) if pd.notna(row.get('volume_vs_avg')) else 0.0
            })

    # Current Setup Analysis
    current_setup = None
    if len(df) >= 2:
        # Ensure required indicator columns exist from enrich step
        adx_col = 'ADX_9' if 'ADX_9' in df.columns else 'ADX'
        rsi_col = 'RSI_14'
        required_setup_cols = [adx_col, rsi_col, 'Close']

        if not all(c in df.columns for c in required_setup_cols):
            print(f"Warning: Gap patterns - missing ADX/RSI/Close for current_setup on {timeframe_str}.")
        else:
            latest_bar = df.iloc[-1]
            prev_bar = df.iloc[-2]

            direction = "flat"
            # Check for NaN before comparison
            if pd.notna(latest_bar['Close']) and pd.notna(prev_bar['Close']):
                if latest_bar['Close'] > prev_bar['Close']: direction = "up"
                elif latest_bar['Close'] < prev_bar['Close']: direction = "down"

            # --- Proper Streak Calculation ---
            streak_val = 0
            if direction != "flat" and len(df) >= 2:
                streak_val = 1
                current_direction_sign = 1 if direction == "up" else -1
                # Iterate backwards from the second-to-last bar
                for i in range(len(df) - 2, 0, -1): # Stop at index 1 (needs bar 0 for comparison)
                    current_close = df['Close'].iloc[i]
                    prev_close = df['Close'].iloc[i-1]
                    if pd.isna(current_close) or pd.isna(prev_close): break # Stop if NaN encountered

                    if (current_direction_sign == 1 and current_close > prev_close) or \
                       (current_direction_sign == -1 and current_close < prev_close):
                        streak_val += 1
                    else:
                        break # Streak broken
            # --- End Streak Calculation ---

            adx_val = latest_bar.get(adx_col, 0.0)
            rsi_val = latest_bar.get(rsi_col, 50.0)

            # Ensure values are numeric before rounding/comparison
            adx_val = float(adx_val) if pd.notna(adx_val) else 0.0
            rsi_val = float(rsi_val) if pd.notna(rsi_val) else 50.0

            gap_risk = "low" # Default to low
            # Example logic: High risk on extended streaks into overbought/oversold
            if (direction == "up" and streak_val >= 3 and rsi_val > 70 and adx_val > 20) or \
               (direction == "down" and streak_val >= 3 and rsi_val < 30 and adx_val > 20):
                gap_risk = "high"
            elif (direction == "up" and streak_val >= 2 and rsi_val > 65) or \
                 (direction == "down" and streak_val >= 2 and rsi_val < 35):
                 gap_risk = "moderate"


            current_setup = {
                "timeframe": timeframe_str, # Add timeframe label
                "direction": direction,
                "streak": streak_val, # Use calculated streak
                "adx": round(adx_val, 2),
                "rsi": round(rsi_val, 2),
                "gap_risk": gap_risk
            }

    return {"gaps": gaps_list, "current_setup": current_setup}

def identify_gap_patterns(df: pd.DataFrame, timeframe_str: str = "daily") -> dict:
    """
    Identifies significant gaps based on timeframe-specific thresholds
    and analyzes current setup from the last complete bar.
    Expects DataFrame with 'Open', 'Close', 'Volume', 'ADX_9', 'RSI_14'.
    The 'df' passed should be an ENRICHED DataFrame.
    """
    if df is None or df.empty or len(df) < 2:
        return {"gaps": [], "current_setup": None}

    gap_thresholds = {
        'daily': 0.5,  # %
        'hourly': 0.3, # %
        '15m': 0.2,    # %
        'default': 0.2 # Default if timeframe_str not matched
    }
    threshold = gap_thresholds.get(timeframe_str, gap_thresholds['default'])
    
    # Ensure required columns for gap calculation exist
    base_cols = ['Open', 'Close', 'Volume']
    for col in base_cols:
        if col not in df.columns:
            print(f"Warning: Gap patterns - missing base column '{col}' for timeframe {timeframe_str}. Returning empty.")
            return {"gaps": [], "current_setup": None}

    # Calculate gap percentage (current open vs previous close)
    df_gaps = df.copy() # Work on a copy
    df_gaps['prev_Close'] = df_gaps['Close'].shift(1)
    df_gaps['gap_pct'] = (df_gaps['Open'] - df_gaps['prev_Close']) / df_gaps['prev_Close'] * 100
    
    significant_gaps_df = df_gaps[df_gaps['gap_pct'].abs() >= threshold].copy()
    
    gaps_list = []
    if not significant_gaps_df.empty:
        significant_gaps_df['gap_type'] = np.where(significant_gaps_df['gap_pct'] > 0, "up", "down")
        significant_gaps_df['day_change_pct'] = (significant_gaps_df['Close'] - significant_gaps_df['Open']) / significant_gaps_df['Open'] * 100
        
        conditions_filled = [
            (significant_gaps_df['gap_type'] == "up") & (significant_gaps_df['day_change_pct'] < 0),
            (significant_gaps_df['gap_type'] == "down") & (significant_gaps_df['day_change_pct'] > 0)
        ]
        significant_gaps_df['follow_through'] = np.select(conditions_filled, ["filled", "filled"], default="extended")
        
        # Volume vs average (20-period)
        # This calculation needs to be careful about index alignment if NaNs were dropped.
        # For simplicity here, ensure 'Volume' exists.
        if 'Volume' in df_gaps.columns: # df_gaps has full history for rolling mean
            df_gaps['avg_vol_20'] = df_gaps['Volume'].shift(1).rolling(window=20, min_periods=5).mean()
            significant_gaps_df = significant_gaps_df.join(df_gaps['avg_vol_20']) # Join avg_vol
            significant_gaps_df['volume_vs_avg'] = np.where(significant_gaps_df['avg_vol_20'] > 0,
                                                          significant_gaps_df['Volume'] / significant_gaps_df['avg_vol_20'], 0)
        else:
            significant_gaps_df['volume_vs_avg'] = 0

        for idx, row in significant_gaps_df.iterrows():
            gaps_list.append({
                "date": idx.strftime('%Y-%m-%d %H:%M:%S') if isinstance(idx, pd.Timestamp) else str(idx),
                "gap_type": row['gap_type'],
                "gap_pct": round(row['gap_pct'], 2),
                "follow_through": row['follow_through'],
                "volume_vs_avg": round(row.get('volume_vs_avg',0), 2)
            })

    # Current Setup Analysis (from the last complete bar of the input df)
    current_setup = None
    if len(df) >= 2: # df here is the original enriched df for this timeframe
        adx_col = 'ADX_9' if 'ADX_9' in df.columns else 'ADX' # Match column from enrich
        rsi_col = 'RSI_14'
        
        # Ensure these columns exist from enrich step
        if not all(c in df.columns for c in [adx_col, rsi_col, 'Close']):
            print(f"Warning: Gap patterns - missing ADX/RSI/Close for current_setup on {timeframe_str}.")
        else:
            latest_bar = df.iloc[-1]
            prev_bar = df.iloc[-2]
            
            direction = "flat"
            if latest_bar['Close'] > prev_bar['Close']: direction = "up"
            elif latest_bar['Close'] < prev_bar['Close']: direction = "down"
            
            # Streak calculation needs more history - for simplicity, this is just last direction
            # A proper streak would iterate backwards on 'df'
            streak_val = 1 if direction != "flat" else 0 
            
            adx_val = latest_bar.get(adx_col, 0.0)
            rsi_val = latest_bar.get(rsi_col, 50.0)

            # Simplified gap_risk for example; your full logic can be more complex
            gap_risk = "moderate"
            if (direction == "up" and streak_val >= 3 and rsi_val > 70) or \
               (direction == "down" and streak_val >= 3 and rsi_val < 30):
                gap_risk = "high"

            current_setup = {
                "timeframe": timeframe_str,
                "direction": direction,
                "streak": streak_val, # Placeholder for actual streak
                "adx": round(adx_val, 2),
                "rsi": round(rsi_val, 2),
                "gap_risk": gap_risk
            }
            
    return {"gaps": gaps_list, "current_setup": current_setup}