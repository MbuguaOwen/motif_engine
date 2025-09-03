
import numpy as np
import pandas as pd

def triple_barrier_labels(bars: pd.DataFrame, atr_col: str, up_mult: float, dn_mult: float, timeout_bars: int):
    c = bars["close"].values
    atr = bars[atr_col].values
    n = len(bars)
    labels = np.zeros(n, dtype=int)
    horizon = min(timeout_bars, n-1)
    for t in range(n):
        up = c[t] + up_mult * atr[t]
        dn = c[t] - dn_mult * atr[t]
        end = min(n, t + horizon + 1)
        seg = c[t+1:end]
        if len(seg)==0:
            labels[t] = 0
            continue
        hit_up = np.where(seg >= up)[0]
        hit_dn = np.where(seg <= dn)[0]
        if len(hit_up)>0 and len(hit_dn)>0:
            labels[t] = 1 if hit_up[0] < hit_dn[0] else -1
        elif len(hit_up)>0:
            labels[t] = 1
        elif len(hit_dn)>0:
            labels[t] = -1
        else:
            labels[t] = 0
    return pd.Series(labels, index=bars.index, name="tb_label")
