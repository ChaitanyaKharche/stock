"""
Out-of-sample backtest for the reconstructed GapPredictionTFT checkpoints.

Evaluates each *_validated.pth (SPY/QQQ/TSLA/NVDA/MSFT/META) plus the AMZN
*_e200_.pth on daily data strictly AFTER each checkpoint's file mtime, i.e.
data the model could not have seen during training.

IMPORTANT: the static (market_cap/beta/sector/vix/liquidity) branch was
trained on a constant placeholder (see tft_model.py docstring) - real VIX/
beta values are NOT used here, since that would be out-of-distribution
relative to training. Only the temporal (price/technical-indicator) branch
carries real signal.
"""
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from . import tft_model as tm
from ..paths import TRAINED_MODELS_DIR

CONTEXT_LEN = tm.MODEL_CONFIG['context_length']
STATIC_CONST = np.array([2.0, 1.0, 1.0, 2.0, 50.0], dtype=np.float32)  # matches every scaler's training mean

def _paths(symbol, joblib=True):
    pth = str(TRAINED_MODELS_DIR / f"tft_{symbol}_validated.pth")
    jl = str(TRAINED_MODELS_DIR / f"tft_{symbol}_validated.joblib") if joblib else None
    return pth, jl

SYMBOLS = {
    'SPY': _paths('SPY'),
    'QQQ': _paths('QQQ'),
    'TSLA': _paths('TSLA'),
    'NVDA': _paths('NVDA'),
    'MSFT': _paths('MSFT'),
    'META': _paths('META'),
    'AMZN': (str(TRAINED_MODELS_DIR / 'tft_AMZN_e200_.pth'), None),
}


def evaluate_symbol(symbol, pth_path, joblib_path):
    model, scaler_temporal, scaler_static = tm.load_checkpoint(pth_path, joblib_path)

    cutoff = datetime.fromtimestamp(Path(pth_path).stat().st_mtime)
    fetch_start = cutoff - timedelta(days=CONTEXT_LEN * 2 + 30)  # buffer for weekends/indicator warmup

    df = yf.download(symbol, start=fetch_start.strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < CONTEXT_LEN + 30:
        return {'symbol': symbol, 'error': f'insufficient data ({len(df)} rows)'}

    feats = tm.build_temporal_features(df).dropna()
    closes = df['Close'].reindex(feats.index)

    eval_start_idx = feats.index.searchsorted(pd.Timestamp(cutoff) + pd.Timedelta(days=1))
    eval_start_idx = max(eval_start_idx, CONTEXT_LEN)

    rows = []
    for t in range(eval_start_idx, len(feats) - 1):
        window = feats.iloc[t - CONTEXT_LEN:t]
        if len(window) != CONTEXT_LEN:
            continue

        result = tm.predict(model, scaler_temporal, scaler_static, window, STATIC_CONST)

        last_close = closes.iloc[t - 1]
        next_close = closes.iloc[t]
        actual_return = (next_close - last_close) / last_close

        rows.append({
            'date': feats.index[t],
            'q10': result['quantiles']['q10'],
            'q50': result['quantiles']['q50'],
            'q90': result['quantiles']['q90'],
            'gap_down': result['gap_class_probs']['DOWN'],
            'gap_flat': result['gap_class_probs']['FLAT'],
            'gap_up': result['gap_class_probs']['UP'],
            'last_close': last_close,
            'actual_return': actual_return,
        })

    if not rows:
        return {'symbol': symbol, 'error': 'no evaluable rows after cutoff'}

    res = pd.DataFrame(rows)

    # Direction hit-rate under both possible target-unit interpretations,
    # since we don't know if quantile heads predict return or price level.
    hit_as_return = (np.sign(res['q50']) == np.sign(res['actual_return'])).mean()
    pred_price_delta = res['q50'] - res['last_close']
    hit_as_price = (np.sign(pred_price_delta) == np.sign(res['actual_return'])).mean()

    # Quantile coverage: how often actual return falls within [q10, q90]
    # (only meaningful under the "return" interpretation; reported for both)
    coverage_as_return = ((res['actual_return'] >= res['q10']) & (res['actual_return'] <= res['q90'])).mean()

    # Gap classifier: 3-way accuracy against a data-driven tercile split of actual returns
    std = res['actual_return'].std()
    def actual_class(r):
        if r < -0.3 * std:
            return 'DOWN'
        if r > 0.3 * std:
            return 'UP'
        return 'FLAT'
    res['actual_class'] = res['actual_return'].apply(actual_class)
    res['pred_class'] = res[['gap_down', 'gap_flat', 'gap_up']].idxmax(axis=1).map(
        {'gap_down': 'DOWN', 'gap_flat': 'FLAT', 'gap_up': 'UP'}
    )
    gap_accuracy = (res['pred_class'] == res['actual_class']).mean()
    baseline_accuracy = res['actual_class'].value_counts(normalize=True).max()  # always-predict-majority-class baseline

    return {
        'symbol': symbol,
        'n_days': len(res),
        'eval_start': str(res['date'].min().date()),
        'eval_end': str(res['date'].max().date()),
        'hit_rate_as_return': round(hit_as_return, 3),
        'hit_rate_as_price_level': round(hit_as_price, 3),
        'quantile_coverage_10_90': round(coverage_as_return, 3),
        'gap_classifier_accuracy': round(gap_accuracy, 3),
        'gap_classifier_majority_baseline': round(baseline_accuracy, 3),
    }


if __name__ == '__main__':
    all_results = []
    for symbol, (pth, jl) in SYMBOLS.items():
        print(f"Evaluating {symbol}...")
        try:
            r = evaluate_symbol(symbol, pth, jl)
        except Exception as e:
            r = {'symbol': symbol, 'error': f'{type(e).__name__}: {e}'}
        all_results.append(r)
        print(' ', r)

    print("\n" + "=" * 100)
    print(pd.DataFrame(all_results).to_string(index=False))
