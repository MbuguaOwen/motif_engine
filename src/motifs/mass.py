
import numpy as np

def _moving_mean_std(x, m):
    cumsum = np.cumsum(np.r_[0.0, x])
    cumsum2 = np.cumsum(np.r_[0.0, x**2])
    sumx = cumsum[m:] - cumsum[:-m]
    sumx2 = cumsum2[m:] - cumsum2[:-m]
    mean = sumx / m
    var = (sumx2 / m) - mean**2
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean, std

def sliding_dot_product(q, t):
    n = len(t); m = len(q)
    k = 1 << (n + m - 1).bit_length()
    fft_q = np.fft.rfft(q[::-1], k)
    fft_t = np.fft.rfft(t, k)
    inv = np.fft.irfft(fft_q * fft_t, k)
    return inv[m-1:n]

def mass(query, ts):
    q = np.asarray(query, dtype=float)
    t = np.asarray(ts, dtype=float)
    m = len(q)
    q = (q - q.mean()) / (q.std() + 1e-12)
    dot = sliding_dot_product(q, t)
    mean_t, std_t = _moving_mean_std(t, m)
    denom = (std_t * m)
    denom = np.where(denom < 1e-12, np.inf, denom)
    dist = 2 * (m - (dot - m * mean_t * q.mean()) / denom )
    dist = np.maximum(dist, 0.0)
    return np.sqrt(dist)

def zdist(a, b):
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return np.linalg.norm(a - b)


def zdist_multi(A, B):
    """
    Column-wise z-normalized L2 distance for multivariate windows.
    A, B: shape (L, F) numpy arrays.
    """
    import numpy as np
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        return float("inf")
    A = (A - A.mean(axis=0, keepdims=True)) / (A.std(axis=0, keepdims=True) + 1e-12)
    B = (B - B.mean(axis=0, keepdims=True)) / (B.std(axis=0, keepdims=True) + 1e-12)
    return float(np.linalg.norm(A - B))
