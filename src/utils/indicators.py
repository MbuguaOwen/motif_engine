
import numpy as np
import pandas as pd

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h,l,c = df["high"].values, df["low"].values, df["close"].values
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h-l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(pd.Series(tr).rolling(window, min_periods=window).mean().values, index=df.index, name="atr")

def bb_width_percentile(df: pd.DataFrame, window: int = 20, k: float = 2.0, pct_window: int = 240) -> pd.Series:
    s = df["close"].rolling(window).std()
    width = (2*k*s) / (df["close"].rolling(window).mean() + 1e-9)
    pct = width.rolling(pct_window, min_periods=pct_window).rank(pct=True)
    return pct.rename("bb_width_pct")

def donchian_position(df: pd.DataFrame, window: int = 20) -> pd.Series:
    hh = df["high"].rolling(window).max()
    ll = df["low"].rolling(window).min()
    pos = (df["close"] - ll) / (hh - ll + 1e-9)
    return pos.clip(0,1).rename("donchian_pos")

def body_true_range_ratio(df: pd.DataFrame) -> pd.Series:
    body = (df["close"] - df["open"]).abs()
    tr = (df["high"] - df["low"]).replace(0, np.nan)
    return (body / tr).fillna(0.0).rename("body_tr_ratio")

def zscore(x: pd.Series, window: int = 100) -> pd.Series:
    r = x.rolling(window)
    m = r.mean()
    s = r.std().replace(0, np.nan)
    return ((x - m) / (s + 1e-9)).fillna(0.0)

def feature_frame(bars: pd.DataFrame, atr_win: int = 50) -> pd.DataFrame:
    df = bars.copy()
    df["atr"] = atr(df, window=atr_win)
    df["ret"] = np.log(df["close"]).diff()
    df["ret_z"] = zscore(df["ret"], window=atr_win*2)
    df["atr_z"] = zscore(df["atr"], window=atr_win*2)
    df["bb_width_pct"] = bb_width_percentile(df, window=20, k=2.0, pct_window=atr_win*8)
    df["donchian_pos"] = donchian_position(df, window=20)
    df["body_tr_ratio"] = body_true_range_ratio(df)
    df = df.dropna().reset_index(drop=True)
    return df
