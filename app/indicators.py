import pandas as pd, pandas_ta as ta
def enrich(df: pd.DataFrame):
    df=df.copy(); df.ta.adx(append=True); df.ta.ema("Close",9,append=True,col_names=("EMA_9_Close",))
    df.ta.ema("Volume",9,append=True,col_names=("EMA_9_Volume",))
    df.rename(columns={"ADX_14":"ADX","DMP_14":"DMI_Plus","DMN_14":"DMI_Minus"},inplace=True)
    return df
