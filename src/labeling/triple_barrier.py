import numpy as np
import pandas as pd


def _as_float_array(a):
    arr = np.asarray(a, dtype=np.float64)
    return arr


def triple_barrier_labels(
    bars: pd.DataFrame,
    atr_col: str,
    up_mult: float,
    dn_mult: float,
    timeout_bars: int,
    *,
    use_high_low: bool = True,
    include_equals: bool = True,
) -> pd.Series:
    """
    Label each bar by the first barrier hit within a forward timeout window.
    +1 if upper barrier is hit first, -1 if lower barrier is hit first, 0 otherwise.

    Key details:
    - Uses HIGH/LOW to detect barrier touches (default) — not just CLOSE.
    - Barriers are flat lines based on ATR at t: up = close[t] + up_mult * atr[t], dn = close[t] - dn_mult * atr[t]
    - NaN ATR or NaN price => label 0
    - 'include_equals'=True counts equality as a hit (>= for up, <= for dn)

    Complexity: O(n * horizon) worst-case (like your previous version), but correct.
    """
    req_cols = {"close", "high", "low", atr_col}
    missing = [c for c in req_cols if c not in bars.columns]
    if missing:
        raise ValueError(f"Bars missing columns: {missing}")

    c = _as_float_array(bars["close"].values)
    h = _as_float_array(bars["high"].values)
    l = _as_float_array(bars["low"].values)
    atr = _as_float_array(bars[atr_col].values)

    n = len(c)
    labels = np.zeros(n, dtype=np.int8)
    if n == 0:
        return pd.Series(labels, index=bars.index, name="tb_label")

    # basic nan mask
    finite_mask = np.isfinite(c) & np.isfinite(h) & np.isfinite(l) & np.isfinite(atr)
    horizon = int(max(0, min(timeout_bars, n - 1)))

    # Choose comparators
    ge = np.greater_equal if include_equals else np.greater
    le = np.less_equal     if include_equals else np.less

    for t in range(n):
        if not finite_mask[t]:
            labels[t] = 0
            continue
        up = c[t] + up_mult * atr[t]
        dn = c[t] - dn_mult * atr[t]
        end = t + horizon + 1
        if end <= t + 1:
            labels[t] = 0
            continue

        # Segment (t+1 ... end-1)
        if use_high_low:
            seg_hi = h[t+1:end]
            seg_lo = l[t+1:end]
            hit_up_idx = np.where(ge(seg_hi, up))[0]
            hit_dn_idx = np.where(le(seg_lo, dn))[0]
        else:
            seg_c = c[t+1:end]
            hit_up_idx = np.where(ge(seg_c, up))[0]
            hit_dn_idx = np.where(le(seg_c, dn))[0]

        if hit_up_idx.size and hit_dn_idx.size:
            labels[t] = 1 if hit_up_idx[0] < hit_dn_idx[0] else -1
        elif hit_up_idx.size:
            labels[t] = 1
        elif hit_dn_idx.size:
            labels[t] = -1
        else:
            labels[t] = 0

    return pd.Series(labels, index=bars.index, name="tb_label")
