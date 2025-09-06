# -*- coding: utf-8 -*-
"""
Extended indicator set used by the motif_engine.

This module keeps backward compatibility with the original minimal set
(atr, ret, ret_z, atr_z, bb_width_pct, donchian_pos, body_tr_ratio)
and adds a richer feature family that the motif miner / gate can use.

All implementations are dependency-free (NumPy/Pandas only).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------- Basic building blocks ----------

def _true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    prev_c = np.r_[np.nan, c[:-1]]
    return np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Wilder-style ATR via simple moving average of True Range.
    """
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    tr = _true_range(h, l, c)
    out = pd.Series(tr, index=df.index).rolling(window, min_periods=window).mean()
    return out.rename("atr")


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def zscore(s: pd.Series, window: int) -> pd.Series:
    r = s.rolling(window, min_periods=window)
    m = r.mean()
    v = r.std().replace(0, np.nan)
    return ((s - m) / (v + 1e-9)).fillna(0.0)


def bb_width_percentile(df: pd.DataFrame, window: int = 20, k: float = 2.0, pct_window: int = 240) -> pd.Series:
    s = df["close"].rolling(window).std()
    width = (2*k*s) / (df["close"].rolling(window).mean() + 1e-9)
    pct = width.rolling(pct_window, min_periods=pct_window).rank(pct=True)
    return pct.rename("bb_width_pct")


def donchian(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    hi = df["high"].rolling(window, min_periods=window).max()
    lo = df["low"].rolling(window, min_periods=window).min()
    return hi.rename("don_hi"), lo.rename("don_lo")


def donchian_position(df: pd.DataFrame, window: int = 20) -> pd.Series:
    hi, lo = donchian(df, window)
    rng = (hi - lo).replace(0, np.nan)
    pos = (df["close"] - lo) / (rng + 1e-9)
    return pos.clip(0, 1).fillna(0.5).rename("donchian_pos")


def body_true_range_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Candle body / true range. Values near 1 => full-body drive; near 0 => doji/wicky.
    """
    o,c,h,l = df["open"], df["close"], df["high"], df["low"]
    body = (c - o).abs()
    tr = pd.Series(_true_range(h.values, l.values, c.values), index=df.index)
    return (body / (tr + 1e-9)).rename("body_tr_ratio")


# ---------- New feature family ----------

def rolling_linreg_slope(s: pd.Series, window: int) -> pd.Series:
    """
    Fast rolling OLS slope of s ~ t, t = 0..w-1.
    Uses closed-form slope = cov(t, s) / var(t). Evaluated per-window.
    """
    t = np.arange(window, dtype=float)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    def _slope(x: np.ndarray) -> float:
        xm = x.mean()
        cov = ((t - t_mean) * (x - xm)).sum()
        return cov / (t_var + 1e-12)

    return s.rolling(window, min_periods=window).apply(_slope, raw=True).rename(f"slope_w{window}")


def rolling_r2(s: pd.Series, window: int) -> pd.Series:
    """
    R^2 of linear fit s ~ t over each window.
    """
    t = np.arange(window, dtype=float)

    def _r2(x: np.ndarray) -> float:
        if np.all(~np.isfinite(x)):
            return np.nan
        xc = x - x.mean()
        tc = t - t.mean()
        denom = np.sqrt((xc**2).sum() * (tc**2).sum()) + 1e-12
        r = (xc * tc).sum() / denom
        return float(r*r)

    return s.rolling(window, min_periods=window).apply(_r2, raw=True).rename(f"r2_w{window}")


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average Directional Index (classic, Wilder). Returns ADX.
    """
    h,l,c = df["high"], df["low"], df["close"]
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move

    tr = pd.Series(_true_range(h.values, l.values, c.values), index=df.index)

    atr_ = tr.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / (atr_ + 1e-12))
    minus_di = 100 * (minus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / (atr_ + 1e-12))
    dx = ( (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12) ) * 100.0
    adx_ = dx.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    return adx_.rename(f"adx_{window}")


def kama(price: pd.Series, er_window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """
    Kaufman Adaptive Moving Average.
    """
    change = price.diff(er_window).abs()
    vol = price.diff().abs().rolling(er_window).sum()
    er = change / (vol + 1e-12)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = np.empty_like(price.values, dtype=float)
    out[:] = np.nan
    seed = price.rolling(er_window, min_periods=er_window).mean()
    started = False
    last = np.nan
    for i, (p, s, sco) in enumerate(zip(price.values, seed.values, sc.values)):
        if not started:
            if np.isfinite(s):
                last = s
                out[i] = s
                started = True
            else:
                out[i] = np.nan
                continue
        else:
            last = last + sco * (p - last)
            out[i] = last
    return pd.Series(out, index=price.index, name="kama")


def wick_ratios(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    o,c,h,l = df["open"], df["close"], df["high"], df["low"]
    upper = (h - np.maximum(o, c)).clip(lower=0)
    lower = (np.minimum(o, c) - l).clip(lower=0)
    tr = pd.Series(_true_range(h.values, l.values, c.values), index=df.index)
    uw = (upper / (tr + 1e-9)).rename("wick_upper_tr")
    lw = (lower / (tr + 1e-9)).rename("wick_lower_tr")
    return uw.fillna(0.0), lw.fillna(0.0)


def clv(df: pd.DataFrame) -> pd.Series:
    """
    Close Location Value in [-1,1].
    """
    high, low, close = df["high"], df["low"], df["close"]
    rng = (high - low).replace(0, np.nan)
    return ((2*close - high - low) / (rng + 1e-9)).fillna(0.0).rename("clv")


def prior_break_counts(close: pd.Series, don_hi: pd.Series, don_lo: pd.Series, lookback: int = 100) -> Tuple[pd.Series, pd.Series]:
    up_break = (close > don_hi.shift(1)).astype(int)
    dn_break = (close < don_lo.shift(1)).astype(int)
    return (up_break.rolling(lookback).sum().rename("prior_break_up_cnt"),
            dn_break.rolling(lookback).sum().rename("prior_break_dn_cnt"))


def bars_since_extrema(close: pd.Series, lookback: int = 100) -> Tuple[pd.Series, pd.Series]:
    """
    Bars since the rolling high/low within the lookback window.
    """
    def _bars_since_max(x: np.ndarray) -> float:
        return float(len(x) - 1 - int(np.argmax(x)))
    def _bars_since_min(x: np.ndarray) -> float:
        return float(len(x) - 1 - int(np.argmin(x)))
    return (
        close.rolling(lookback, min_periods=lookback).apply(_bars_since_max, raw=True).rename("bars_since_high"),
        close.rolling(lookback, min_periods=lookback).apply(_bars_since_min, raw=True).rename("bars_since_low"),
    )


def distance_to_donchian(df: pd.DataFrame, don_hi: pd.Series, don_lo: pd.Series) -> Tuple[pd.Series, pd.Series]:
    c = df["close"]
    atr_col = df.get("atr")
    if atr_col is None:
        dh = ((don_hi - c) / (c + 1e-9)).rename("dist_to_don_hi")
        dl = ((c - don_lo) / (c + 1e-9)).rename("dist_to_don_lo")
    else:
        dh = ((don_hi - c) / (atr_col + 1e-9)).rename("dist_to_don_hi_atr")
        dl = ((c - don_lo) / (atr_col + 1e-9)).rename("dist_to_don_lo_atr")
    return dh, dl


def tick_imbalance(df: pd.DataFrame) -> pd.Series:
    """
    Optional: if bars include aggregated tick counts as columns 'tick_buys' and 'tick_sells',
    compute (buys - sells) / (buys + sells). Otherwise returns NaN.
    """
    if "tick_buys" in df.columns and "tick_sells" in df.columns:
        tb = df["tick_buys"].astype(float)
        ts = df["tick_sells"].astype(float)
        denom = (tb + ts).replace(0, np.nan)
        return ((tb - ts) / (denom + 1e-12)).rename("tick_imbalance").fillna(0.0)
    return pd.Series(np.nan, index=df.index, name="tick_imbalance")


# ---------- Feature assembly ----------

def feature_frame(
    bars: pd.DataFrame,
    atr_win: int = 50,
    don_win: int = 20,
    hv_short: int = 20,
    hv_long: int = 100,
    tsmom_win: int = 60,
    trend_r2_win: int = 60,
    adx_win: int = 14,
    kama_er_win: int = 10,
    kama_fast: int = 2,
    kama_slow: int = 30,
    swing_lookback: int = 100,
) -> pd.DataFrame:
    """
    Build a rich feature frame. Defaults chosen to be stable on 1-min crypto.

    Backward compatible: the original seven columns are included.
    """
    df = bars.copy()

    # base signals
    df["atr"] = atr(df, window=atr_win)
    df["ret"] = np.log(df["close"]).diff()
    df["ret_z"] = zscore(df["ret"], window=max(atr_win, hv_long))
    df["atr_z"] = zscore(df["atr"], window=max(atr_win, hv_long))
    df["bb_width_pct"] = bb_width_percentile(df, window=20, k=2.0, pct_window=max(atr_win, hv_long)*4)
    df["donchian_pos"] = donchian_position(df, window=don_win)
    df["body_tr_ratio"] = body_true_range_ratio(df)

    # regime / trend
    logp = np.log(df["close"])
    df["tsmom_slope"] = rolling_linreg_slope(logp, window=tsmom_win)
    df["tsmom_slope_z"] = zscore(df["tsmom_slope"], window=max(atr_win, hv_long))
    df[f"adx_{adx_win}"] = adx(df, window=adx_win)
    df["kama"] = kama(df["close"], er_window=kama_er_win, fast=kama_fast, slow=kama_slow)
    df["kama_slope"] = df["kama"].diff()
    df[f"trend_r2_w{trend_r2_win}"] = rolling_r2(logp, window=trend_r2_win)

    # volatility / energy
    df["atr_pct_price"] = (df["atr"] / (df["close"] + 1e-9))
    df["rv_short"] = df["ret"].rolling(hv_short, min_periods=hv_short).std()
    df["rv_long"] = df["ret"].rolling(hv_long, min_periods=hv_long).std()
    df["hv_ratio"] = (df["rv_short"] / (df["rv_long"] + 1e-12))
    df["atr_med"] = df["atr"].rolling(hv_long, min_periods=hv_long).median()

    # structure (micro)
    uw, lw = wick_ratios(df)
    df["wick_upper_tr"], df["wick_lower_tr"] = uw, lw
    df["clv"] = clv(df)
    prev_close = df["close"].shift(1)
    gap = (df["open"] - prev_close) / (prev_close + 1e-12)
    df["gap_up"] = (gap > 0.002).astype(int)   # 0.2% default threshold
    df["gap_dn"] = (gap < -0.002).astype(int)
    df["impulse_score"] = (df["ret"].abs() / ( (df["atr_med"] / (df["close"] + 1e-12)) + 1e-12 ) )

    # positioning / breakout context
    don_hi, don_lo = donchian(df, window=don_win)
    d_hi, d_lo = distance_to_donchian(df, don_hi, don_lo)
    df[d_hi.name] = d_hi
    df[d_lo.name] = d_lo
    up_cnt, dn_cnt = prior_break_counts(df["close"], don_hi, don_lo, lookback=swing_lookback)
    df["prior_break_up_cnt"], df["prior_break_dn_cnt"] = up_cnt, dn_cnt
    bs_hi, bs_lo = bars_since_extrema(df["close"], lookback=swing_lookback)
    df["bars_since_high"], df["bars_since_low"] = bs_hi, bs_lo
    roll_max = df["close"].rolling(swing_lookback, min_periods=swing_lookback).max()
    roll_min = df["close"].rolling(swing_lookback, min_periods=swing_lookback).min()
    df["pullback_from_high_pct"] = ((roll_max - df["close"]) / (roll_max + 1e-12)).clip(lower=0)
    df["pullup_from_low_pct"] = ((df["close"] - roll_min) / (roll_min + 1e-12)).clip(lower=0)

    # (optional) flow
    if "volume" in df.columns:
        df["volume_z"] = zscore(df["volume"].astype(float), window=max(atr_win, hv_long))
        df["dollar_volume"] = (df["volume"].astype(float) * df["close"].astype(float))

    # tick_imbalance may be all-NaN if tick_buys/tick_sells are absent; keep the rows.
    df["tick_imbalance"] = tick_imbalance(df).fillna(0.0)

    # finalize: only demand OHLC+ATR; keep rows even if some *features* are NaN
    core = ["open", "high", "low", "close", "atr"]
    df = df.dropna(subset=core).reset_index(drop=True)
    return df


# Notes

# Defaults are conservative for 1-min crypto; tweak in feature_frame(...) args if you like
# (e.g., tsmom_win, don_win, swing_lookback).

# tick_imbalance fills NaN unless your bars include tick_buys/tick_sells.

# Everything remains dependency-free; run_motifs can keep calling
# feature_frame(bars, atr_win=50) without changes.

