
import numpy as np
from .mass import zdist_multi  # new

class ShapeletMatcher:
    def __init__(self, shapelets):
        self.shapelets = shapelets  # list of {vec, eps}

    def match_score(self, window: np.ndarray):
        if not self.shapelets:
            return 0.0, False
        scores = []
        for sh in self.shapelets:
            vec = sh["vec"]; eps = sh["eps"]
            if len(window) < len(vec): 
                return 0.0, False
            w = (window - window.mean())/(window.std()+1e-12)
            v = (vec - vec.mean())/(vec.std()+1e-12)
            d = np.linalg.norm(w - v)
            hit = d <= eps
            score = max(0.0, 1.0 - d/(eps + 1e-9))
            scores.append((score, hit))
        best = max(scores, key=lambda x: x[0])
        return best[0], best[1]


class MultiShapeletMatcher:
    """
    Shapelets stored as dicts: {"mat": np.ndarray[L, F], "eps": float}
    match_score(window[L,F]) -> (score[0..1], hit:bool)
    """

    def __init__(self, shapelets):
        self.shapelets = shapelets or []

    def match_score(self, window):
        if not self.shapelets:
            return 0.0, False
        if window is None:
            return 0.0, False
        scores = []
        for sh in self.shapelets:
            mat = sh["mat"]; eps = float(sh["eps"])
            if mat.shape != window.shape:
                return 0.0, False
            d = zdist_multi(window, mat)
            hit = d <= eps
            score = max(0.0, 1.0 - d / (eps + 1e-9))
            scores.append((score, hit))
        return max(scores, key=lambda x: x[0]) if scores else (0.0, False)
