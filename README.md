
# Motif Engine (Matrix Profile + MASS) — Multi‑Horizon, Walk‑Forward

This project mines **motifs/shapelets** across **macro/meso/micro** horizons using a Matrix‑Profile–style
approach (MASS + neighbor density), calibrates distance thresholds **ε** for *economic edge* (expected R),
and evaluates with a **walk‑forward** backtest. It integrates a clean ATR/TSL risk manager.

### Data expected

Tick CSVs here (UTC, ms timestamp):

```
inputs/
  BTCUSDT/
    BTCUSDT-ticks-2025-01.csv
    BTCUSDT-ticks-2025-02.csv
    ...
  ETHUSDT/
  SOLUSDT/
```

Columns: `timestamp,price,qty,is_buyer_maker`

### Quick start

```
# 1) Put your tick CSVs under inputs/<SYMBOL>/
# 2) Run walk-forward mining + backtest (3m train → 1m test):
python run_motifs.py --config configs/motifs.yaml --mode walkforward
```

Outputs land under `outputs/` with logs, artifacts (shapelets per fold), and evaluation JSON.

### Notes

- No external net deps. Only `numpy`, `pandas`, `pyyaml`. MASS implemented via NumPy FFT.
- Matrix‑profile approximation: for each horizon we sample candidate windows, compute sliding MASS distance
  profiles, and select windows with **highest neighbor density** (small mean of k nearest distances).
  Those medoids become **shapelets**; **discords** are the opposite (largest mean nearest distances).
- High‑end default thresholds: up barrier 2.5×ATR, down barrier 1.0×ATR, strict score gating, robust TSL floor.
