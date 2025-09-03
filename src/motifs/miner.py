
import numpy as np
from .mass import mass
from src.ui.progress import wrap_iter

def sample_candidates(series: np.ndarray, L: int, stride: int) -> np.ndarray:
    n = len(series)
    if n < L: 
        return np.array([], dtype=int)
    return np.arange(0, n - L + 1, stride, dtype=int)

def neighbor_density_pick(series: np.ndarray, L: int, candidates: np.ndarray, k: int = 10, top_k: int = 5,
                          yaml_cfg: dict = None, cli_disable: bool = False, desc: str = None):
    if len(candidates)==0:
        return np.array([], dtype=int), np.array([], dtype=int), None
    N = len(candidates)
    dmeans = np.zeros(N)
    # Enumerate candidates with optional progress bar
    it = wrap_iter(range(N), total=N, desc=(desc or "Mine"), yaml_cfg=yaml_cfg or {}, cli_disable=cli_disable)
    for i in it:
        s = candidates[i]
        q = series[s:s+L]
        dprof = mass(q, series)
        left = max(0, s - L//2)
        right = min(len(dprof), s + L//2)
        dprof[left:right] = np.inf  # trivial match exclusion
        dmeans[i] = np.partition(dprof, min(k, len(dprof)-1))[:k].mean()
    order = np.argsort(dmeans)
    motif_idxs = candidates[order[:min(top_k, len(order))]]
    discord_idxs = candidates[order[::-1][:min(top_k, len(order))]]
    return motif_idxs, discord_idxs, dmeans

def extract_shapelets(series: np.ndarray, L: int, starts: np.ndarray):
    return [series[s:s+L].copy() for s in starts]
