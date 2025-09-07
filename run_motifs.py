import argparse, json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import yaml

# --- epsilon helpers ---

def _as_float(x, default=1.0):
    """
    Robustly convert x to a Python float.
    - None or non-finite -> default
    - array-like with >1 values -> median of finite values
    - numpy scalar or 0-d array -> scalar item
    """
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return float(default)
        if arr.size > 1:
            arr = arr[np.isfinite(arr)]
            return float(np.median(arr)) if arr.size else float(default)
        # 0-d array or scalar
        val = arr.reshape(()).item()
        return float(val) if np.isfinite(val) else float(default)
    except Exception:
        return float(default)


def _coerce_eps_tree(epsilon):
    """
    In-place: ensure every epsilon in the nested structure is a Python float.
    Structure:
      epsilon[h]["long"]["good"|"bad"]
      epsilon[h]["short"]["good"|"bad"]
      epsilon[h]["discords"]
    """
    for h in list(epsilon.keys()):
        for side in ("long", "short"):
            if side in epsilon[h]:
                for kind in ("good", "bad"):
                    if kind in epsilon[h][side]:
                        epsilon[h][side][kind] = _as_float(
                            epsilon[h][side][kind], 1.0
                        )
        # discords
        epsilon[h]["discords"] = _as_float(epsilon[h].get("discords", None), 1.0)


def _coerce_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _prep_ohlc_atr_for_labels(df, atr_window: int):
    """
    Make OHLC/ATR finite and consistent:
    - numeric coercion
    - fix high<low
    - compute ATR if missing/all-NaN
    - bfill/ffill
    - drop bad rows (but never to zero length)
    """
    df = df.copy()
    need = ["open","high","low","close","volume"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"[DATA] missing column '{c}' for labeling")
    _coerce_numeric(df, need + (["atr"] if "atr" in df.columns else []))

    inv = df["high"] < df["low"]
    if inv.any():
        hi = df.loc[inv, "high"].values
        df.loc[inv, "high"] = df.loc[inv, "low"].values
        df.loc[inv, "low"] = hi

    need_atr = ("atr" not in df.columns) or (~np.isfinite(df["atr"].to_numpy())).all()
    if need_atr:
        prev_close = df["close"].shift(1)
        tr = (df["high"] - df["low"]).abs()
        tr = np.maximum(tr, (df["high"] - prev_close).abs())
        tr = np.maximum(tr, (df["low"] - prev_close).abs())
        df["atr"] = tr.rolling(int(max(1, atr_window)), min_periods=1).mean()

    for c in ["open","high","low","close","volume","atr"]:
        df[c] = df[c].bfill().ffill()

    req = df[["high","low","close","atr"]]
    bad_mask = ~np.isfinite(req.to_numpy(float)).all(axis=1)
    dropped = int(bad_mask.sum())
    if dropped > 0:
        if dropped == len(df):
            print("[DATA] WARNING: all rows non-finite after cleaning; keeping rows (no drop).")
        else:
            df = df.loc[~bad_mask].reset_index(drop=True)
            print(f"[DATA] dropped {dropped} non-finite rows before labeling")
    if (df["high"] < df["low"]).any():
        print("[DATA] WARNING: high<low rows remain after cleaning")
    return df


def _mask_by_months(df, months):
    m = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    return m.isin(pd.Index(months)).to_numpy(bool)

from src.utils.io import (
    read_kline_1m_months,
    read_tick_months,
    ticks_to_bars_1m,
    resample_bars,
)
from src.utils.indicators import feature_frame
from src.labeling.triple_barrier import triple_barrier_labels
from src.motifs.miner import sample_candidates, neighbor_density_pick
from src.motifs.matcher import ShapeletMatcher, MultiShapeletMatcher  # NEW
from src.engine.gating import composite_score
from src.gating import passes_gate, GateDecision
from src.engine.risk import RiskManager
from src.ui.progress import wrap_iter, progress_redirect_logs, bar

# ---------- helpers ----------

def month_windows(months, train, test, step):
    out=[]; m=months
    for i in range(0, len(m)-train-test+1, step):
        out.append((m[i:i+train], m[i+train:i+train+test]))
    return out

def build_feats(bars_1m, tf, atr_window):
    if tf == "1min":
        b = bars_1m.copy()
    else:
        b = resample_bars(bars_1m, tf)
    f = feature_frame(b, atr_win=atr_window)
    f["timestamp"] = b["timestamp"].iloc[-len(f):].values
    return f

def save_parquet(df, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)

def load_parquet(path):
    return pd.read_parquet(path)

def fold_id(symbol, train_months, test_months):
    return f"{symbol}_{train_months[-1]}__{test_months[0]}"

# --- ADD configurable multivariate features (curate forward-safe set) ---
FEATURES_FOR_MOTIFS_DEFAULT = [  # NEW: can be overridden by YAML
    "ret_z","tsmom_slope_z","atr","atr_z","atr_pct_price","hv_ratio","bb_width_pct","impulse_score","atr_med",
    "body_tr_ratio","wick_upper_tr","wick_lower_tr","clv",
    "donchian_pos","dist_to_don_hi_atr","dist_to_don_lo_atr",
    "prior_break_up_cnt","prior_break_dn_cnt","bars_since_high","bars_since_low",
    "pullback_from_high_pct","pullup_from_low_pct",
    "trend_r2_w60","adx_14","kama_slope",
    "volume_z","tick_imbalance"
]

def _pick_features(df, cfg):  # NEW
    # Allow YAML override: motifs.features: [...]
    feats = (cfg.get("motifs", {}).get("features") or FEATURES_FOR_MOTIFS_DEFAULT)
    feats = [c for c in feats if c in df.columns]  # drop missing
    if len(feats) == 0:
        feats = ["ret_z"]  # last resort
    return feats

def _window_LF(df, end_row, L, cols):  # NEW
    s = end_row - L + 1
    if s < 0:
        return None
    W = df.iloc[s:end_row+1][cols].to_numpy(dtype=float)
    # explicit shape check (never rely on array truthiness)
    return W if W.shape == (L, len(cols)) else None

def _macro_sign_from_window(win_close):  # NEW
    return 1 if (win_close[-1] - win_close[0]) >= 0 else -1

# ---------- core fold run (unchanged simulation loop) ----------

def run_fold(cfg, symbol, train_months, test_months, cli_disable=False,
             export_csv=False, persist_artifacts=False, use_artifacts=False):
    atr_win = cfg["bars"]["atr_window"]

    # ----- Inputs: bars from inputs/ (new) or cached/build (legacy) -----
    src = cfg["bars"].get("source", "cache_or_build")  # "inputs_1m" | "cache_or_build"
    bars_file = Path(cfg["paths"]["outputs_dir"]) / "bars" / f"{symbol}.parquet"

    if src == "inputs_1m":
        bars_1m = read_kline_1m_months(
            cfg["paths"]["inputs_dir"],
            symbol,
            train_months + test_months,
            yaml_cfg=cfg,
            cli_disable=cli_disable,
        )
    elif bars_file.exists():
        bars_1m = load_parquet(bars_file)
    else:
        # legacy: build from ticks if present
        ticks = read_tick_months(
            cfg["paths"]["inputs_dir"],
            symbol,
            train_months + test_months,
            yaml_cfg=cfg,
            cli_disable=cli_disable,
        )
        bars_1m = ticks_to_bars_1m(ticks)

    # ----- Features: either cached or compute -----
    feats_dir = Path(cfg["paths"]["outputs_dir"]) / "features"
    feats = {}
    for h, tf in [("macro", cfg["bars"]["macro_tf"]),
                  ("meso",  cfg["bars"]["meso_tf"]),
                  ("micro", cfg["bars"]["micro_tf"])]:
        fpath = feats_dir / f"{symbol}_{h}.parquet"
        if fpath.exists():
            feats[h] = load_parquet(fpath)
        else:
            feats[h] = build_feats(bars_1m, tf, atr_win)
            save_parquet(feats[h], fpath)

    # ----- Clean micro BEFORE labeling and BEFORE masks -----
    micro = feats["micro"].copy()
    micro = _prep_ohlc_atr_for_labels(micro, atr_window=int(cfg["bars"]["atr_window"]))

    # Decide label config for this phase (mine uses labels_mining)
    is_mine_phase = (cfg.get("_phase") == "mine")
    lbl_cfg = cfg["labels_mining"] if (is_mine_phase and "labels_mining" in cfg) else cfg["labels"]

    # Compute labels (high/low aware)
    labels = triple_barrier_labels(
        micro[["open","high","low","close","volume","atr"]],
        atr_col="atr",
        up_mult=float(lbl_cfg["barrier_up_atr"]),
        dn_mult=float(lbl_cfg["barrier_dn_atr"]),
        timeout_bars=int(lbl_cfg["timeout_bars_micro"]),
        use_high_low=True,
        include_equals=True,
    )
    micro["tb_label"] = labels.values
    feats["micro"] = micro  # keep in feats map

    # Audits (all rows)
    vc_all = pd.Series(micro["tb_label"]).value_counts().reindex([-1,0,1], fill_value=0)
    print(f"[LABELS] phase={'mine' if is_mine_phase else cfg.get('_phase')} "
          f"used up={float(lbl_cfg['barrier_up_atr'])} dn={float(lbl_cfg['barrier_dn_atr'])} "
          f"timeout={int(lbl_cfg['timeout_bars_micro'])} use_high_low=True | "
          f"counts(all)={{-1:{int(vc_all.loc[-1])},0:{int(vc_all.loc[0])},1:{int(vc_all.loc[1])}}}")

    # NOW build month masks using the cleaned frames (sizes will match)
    masks = {
        h: {
            "train": _mask_by_months(feats[h], train_months),
            "test":  _mask_by_months(feats[h], test_months),
        }
        for h in feats
    }

    # If mining and TRAIN has no ±1, auto-relax labels for this fold and relabel
    if is_mine_phase:
        mtrain = masks["micro"]["train"]
        vc_tr = pd.Series(micro.loc[mtrain, "tb_label"]).value_counts().reindex([-1,0,1], fill_value=0)
        print(
            f"[LABELS] TRAIN label counts {{-1:{int(vc_tr.loc[-1])},0:{int(vc_tr.loc[0])},1:{int(vc_tr.loc[1])}}} "
            f"| rows_by_month={pd.to_datetime(micro.loc[mtrain,'timestamp'], utc=True, errors='coerce').dt.strftime('%Y-%m').value_counts().sort_index().to_dict()}"
        )
        if int(vc_tr.loc[-1]) == 0 and int(vc_tr.loc[1]) == 0:
            fb = cfg.get("labels_mining_relax", {"barrier_up_atr":24, "barrier_dn_atr":8, "timeout_bars_micro":2880})
            print(
                f"[LABELS] RELAX: no ±1 in TRAIN → retry mining labels with up={fb['barrier_up_atr']}, dn={fb['barrier_dn_atr']}, timeout={fb['timeout_bars_micro']}"
            )
            labels = triple_barrier_labels(
                micro[["open","high","low","close","volume","atr"]],
                atr_col="atr",
                up_mult=float(fb["barrier_up_atr"]),
                dn_mult=float(fb["barrier_dn_atr"]),
                timeout_bars=int(fb["timeout_bars_micro"]),
                use_high_low=True,
                include_equals=True,
            )
            micro["tb_label"] = labels.values
            feats["micro"] = micro
            vc_tr = pd.Series(micro.loc[mtrain, "tb_label"]).value_counts().reindex([-1,0,1], fill_value=0)
            print(f"[LABELS] TRAIN (relaxed) counts {{-1:{int(vc_tr.loc[-1])},0:{int(vc_tr.loc[0])},1:{int(vc_tr.loc[1])}}}")

    save_parquet(micro, Path(cfg["paths"]["outputs_dir"]) / "features" / f"{symbol}_micro.parquet")
    micro_full = micro

    # ----- Mining / ε -----
    shapelets = {}; epsilon = {}; matchers = {}
    if use_artifacts:
        import pickle
        art_dir = Path("artifacts") / fold_id(symbol, train_months, test_months)
        art_file = art_dir / "shapelet_artifacts.pkl"
        if not art_file.exists():
            raise FileNotFoundError(f"Artifacts not found: {art_file}")
        with open(art_file, "rb") as f:
            arts = pickle.load(f)
        if "classes" in arts["horizons"]["macro"]:  # NEW schema
            feats_list = arts.get("features", _pick_features(feats["micro"], cfg))
            multi_shapelets = {h: {"long": {"good": [], "bad": []},
                                   "short": {"good": [], "bad": []},
                                   "discords": []} for h in ["macro", "meso", "micro"]}
            epsilon = {h: {"long": {"good": None, "bad": None},
                           "short": {"good": None, "bad": None},
                           "discords": None} for h in ["macro", "meso", "micro"]}
            for h in ["macro", "meso", "micro"]:
                H = arts["horizons"][h]
                for side in ["long", "short"]:
                    for kind in ["good", "bad"]:
                        epsilon[h][side][kind] = float(H["classes"][side][kind]["epsilon"])
                        multi_shapelets[h][side][kind] = [np.asarray(W, dtype=float) for W in H["classes"][side][kind]["shapelets"]]
                epsilon[h]["discords"] = float(H.get("discords", {}).get("epsilon", 1.0))
                multi_shapelets[h]["discords"] = [np.asarray(W, dtype=float) for W in H.get("discords", {}).get("shapelets", [])]
            _coerce_eps_tree(epsilon)
            # matchers built later in simulate section
        else:
            for h in ["macro", "meso", "micro"]:
                shapelets[h] = [np.asarray(x, dtype=float) for x in arts["horizons"][h]["shapelets"]]
                epsilon[h] = float(arts["horizons"][h]["epsilon"])
                matchers[h] = ShapeletMatcher([{"vec": s, "eps": epsilon[h]}
                                               for s in shapelets[h][:cfg["motifs"]["horizons"][h]["keep"]]])
            feats_list = _pick_features(feats["micro"], cfg)
    else:
        # Fresh mining from TRAIN months (keep 1-D candidate enumeration for speed)
        # --- Basic data integrity checks ---
        try:
            md = feats["micro"][["high", "low", "atr"]]
            n_nan = int((~np.isfinite(md.to_numpy(float))).sum())
            n_flip = int((feats["micro"]["high"] < feats["micro"]["low"]).sum())
            if n_nan > 0 or n_flip > 0:
                print(f"[DATA] anomalies: non-finite(OHLC/ATR)={n_nan}, high<low rows={n_flip}")
        except Exception as _e:
            print(f"[DATA] integrity check error: {_e}")

        # --- Train-only label audit BEFORE mining begins ---
        try:
            micro_df = feats["micro"].reset_index(drop=True)
            ts = pd.to_datetime(micro_df["timestamp"], utc=True, errors="coerce")
            mtrain = np.asarray(masks["micro"]["train"]).astype(bool)
            vc_train = pd.Series(micro_df.loc[mtrain, "tb_label"]).value_counts().reindex([-1,0,1], fill_value=0)
            months_train = ts.loc[mtrain].dt.strftime("%Y-%m")
            months_map = months_train.value_counts().sort_index().to_dict()
            print(f"[LABELS] TRAIN label counts {{-1:{int(vc_train.loc[-1])},0:{int(vc_train.loc[0])},1:{int(vc_train.loc[1])}}} "
                  f"| rows_by_month={months_map}")
            if int(vc_train.loc[1]) == 0 and int(vc_train.loc[-1]) == 0:
                print("[WARN] TRAIN labels have no +1/-1 (all timeouts). Consider raising timeout_bars or using labels_mining (e.g., 30/10, 1440).")
        except Exception as _e:
            print(f"[LABELS] train-audit error: {_e}")
        motif_starts = {}     # keep motif start indices per horizon
        discord_starts = {}   # NEW: keep discord start indices per horizon

        for h in wrap_iter(["macro","meso","micro"], total=3, desc=f"Mine[{symbol}]", yaml_cfg=cfg, cli_disable=cli_disable):
            L = cfg["motifs"]["horizons"][h]["L"]
            stride = cfg["motifs"]["horizons"][h]["candidate_stride"]
            top_k = cfg["motifs"]["horizons"][h]["top_k"]
            series = feats[h]["ret_z"].values[masks[h]["train"]]
            cand = sample_candidates(series, L, stride)
            if len(cand) == 0:
                raise RuntimeError(f"No candidates for {h}")

            motif_idx, discord_idx, _ = neighbor_density_pick(
                series, L, cand, k=10, top_k=top_k,
                yaml_cfg=cfg, cli_disable=cli_disable, desc=f"Mine {h}@L={L}"
            )

            motif_starts[h]  = motif_idx
            discord_starts[h] = discord_idx     # NEW
            shapelets[h] = [series[s:s+L].copy() for s in motif_idx]

        # --- Build multivariate class-aware banks per horizon (unchanged scaffold) ---
        from collections import defaultdict
        multi_shapelets = {h: {"long": {"good": [], "bad": []},
                               "short":{"good": [], "bad": []},
                               "discords": []} for h in ["macro","meso","micro"]}
        epsilon = {h: {"long": {"good": None, "bad": None},
                       "short":{"good": None, "bad": None},
                       "discords": None} for h in ["macro","meso","micro"]}

        # Map micro timestamp -> tb_label (+1 up, -1 down, 0 timeout)
        micro_map = dict(zip(micro_full["timestamp"].astype(str), micro["tb_label"].astype(int)))

        # Feature list (persist later)
        feats_list = _pick_features(feats["micro"], cfg)

        for h in ["macro","meso","micro"]:
            L = cfg["motifs"]["horizons"][h]["L"]
            feats_h = feats[h].reset_index(drop=True)
            idx_train = np.where(masks[h]["train"])[0]
            if len(idx_train) < L:
                continue

            # ---- Motifs → multivariate windows with guards ----
            for s in motif_starts[h]:
                if s + L - 1 >= len(idx_train):   # NEW guard
                    continue
                end_row = idx_train[s + L - 1]
                ts = str(feats_h.loc[end_row, "timestamp"])
                tb = int(micro_map.get(ts, 0))
                # macro sign from this horizon's close window
                close_win = feats_h["close"].iloc[end_row-L+1:end_row+1].to_numpy(float)
                msign = _macro_sign_from_window(close_win)

                if tb == 0:
                    continue
                elif tb == 1 and msign > 0:
                    side, kind = "long", "good"
                elif tb == -1 and msign < 0:
                    side, kind = "short", "good"
                elif tb == -1 and msign > 0:
                    side, kind = "long", "bad"
                elif tb == 1 and msign < 0:
                    side, kind = "short", "bad"
                else:
                    continue

                W = _window_LF(feats_h, end_row, L, feats_list)
                if W is not None and np.isfinite(W).all():
                    multi_shapelets[h][side][kind].append(W)

            # ---- Discords → multivariate windows per horizon (no boolean array) ----
            ds = discord_starts.get(h, None)
            iter_ds = [] if ds is None else (ds.tolist() if hasattr(ds, "tolist") else list(ds))
            for s in iter_ds:
                if s + L - 1 >= len(idx_train):   # guard
                    continue
                end_row = idx_train[s + L - 1]
                Wd = _window_LF(feats_h, end_row, L, feats_list)
                if Wd is not None and np.isfinite(Wd).all():
                    multi_shapelets[h]["discords"].append(Wd)

        # --- Calibrate ε per (horizon, bank) ---
        from src.motifs.mass import zdist_multi

        EPS_MAX = int(cfg["motifs"].get("eps_cal_sample_max", 5000))
        PCT = float(cfg["motifs"].get("eps_cal_pctile", 15))

        def _cal_eps(h, side, kind):
            mats = multi_shapelets[h][side][kind]
            if not mats:
                return 1.0
            feats_h = feats[h].reset_index(drop=True)
            L = cfg["motifs"]["horizons"][h]["L"]
            idx_train = np.where(masks[h]["train"])[0]

            # gather candidate end rows matching this class
            end_rows = []
            for e in idx_train:
                if e < L-1:
                    continue
                ts = str(feats_h.loc[e, "timestamp"])
                tb = int(micro_map.get(ts, 0))
                close_win = feats_h["close"].iloc[e-L+1:e+1].to_numpy(float)
                msign = _macro_sign_from_window(close_win)
                cls = None
                if tb == 1 and msign > 0:
                    cls = ("long", "good")
                elif tb == -1 and msign < 0:
                    cls = ("short", "good")
                elif tb == -1 and msign > 0:
                    cls = ("long", "bad")
                elif tb == 1 and msign < 0:
                    cls = ("short", "bad")
                if cls == (side, kind):
                    end_rows.append(e)

            if len(end_rows) > EPS_MAX:
                step = max(1, len(end_rows) // EPS_MAX)
                end_rows = end_rows[::step]

            dpool = []
            for mat in mats:
                for e in end_rows:
                    W = _window_LF(feats_h, e, L, feats_list)
                    if W is None or not np.isfinite(W).all():
                        continue
                    dpool.append(zdist_multi(mat, W))
            return float(np.percentile(np.asarray(dpool), PCT)) if dpool else 1.0

        # logging wrapper
        for h in ["macro", "meso", "micro"]:
            for side in ["long", "short"]:
                for kind in ["good", "bad"]:
                    n_mats = len(multi_shapelets[h][side][kind])
                    t0 = time.time()
                    epsilon[h][side][kind] = _cal_eps(h, side, kind)
                    print(
                        f"[EPS] {h}/{side}.{kind}: mats={n_mats} -> eps={epsilon[h][side][kind]:.4f} in {time.time()-t0:.1f}s"
                    )
            epsilon[h]["discords"] = 1.0
        _coerce_eps_tree(epsilon)

    # --- UPDATE artifact persist to store features + banks (handle persist_artifacts + mine mode) ---
    if persist_artifacts and cfg.get("_phase") == "mine":
        import pickle
        art_dir = Path("artifacts") / f"{symbol}_{train_months[-1]}__{test_months[0]}"
        art_dir.mkdir(parents=True, exist_ok=True)
        artifact_dict = {"symbol": symbol,
                         "fold": {"train_end": str(train_months[-1]), "test_start": str(test_months[0])},
                         "features": feats_list,
                         "horizons": {}}
        for h in ["macro", "meso", "micro"]:
            _disc_eps = _as_float(epsilon[h].get("discords", None), 1.0)
            artifact_dict["horizons"][h] = {
                "L": int(cfg["motifs"]["horizons"][h]["L"]),
                "classes": {
                    "long":  {"good": {"epsilon": float(epsilon[h]["long"]["good"]),
                                         "shapelets": [np.asarray(W, dtype=float) for W in multi_shapelets[h]["long"]["good"]]},
                               "bad":  {"epsilon": float(epsilon[h]["long"]["bad"]),
                                         "shapelets": [np.asarray(W, dtype=float) for W in multi_shapelets[h]["long"]["bad"]]}},
                    "short": {"good": {"epsilon": float(epsilon[h]["short"]["good"]),
                                         "shapelets": [np.asarray(W, dtype=float) for W in multi_shapelets[h]["short"]["good"]]},
                               "bad":  {"epsilon": float(epsilon[h]["short"]["bad"]),
                                         "shapelets": [np.asarray(W, dtype=float) for W in multi_shapelets[h]["short"]["bad"]]}},
                },
                "discords": {
                    "epsilon": _disc_eps,
                    "shapelets": [np.asarray(W, dtype=float) for W in multi_shapelets[h]["discords"]]
                }
            }
        tmp = art_dir / "shapelet_artifacts.pkl.tmp"
        out = art_dir / "shapelet_artifacts.pkl"
        with open(tmp, "wb") as fh:
            pickle.dump(artifact_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(out)
        print(f"[INFO] persisted artifacts to {out}")
        return {"symbol": symbol, "train_months": train_months, "test_months": test_months,
                "trades": 0, "sum_R": 0.0, "avg_R": 0.0, "median_R": 0.0}

    # ----- Simulation (test) -----
    risk = RiskManager(sl_mult=cfg["risk"]["sl_mult"],
                       tp_mult=cfg["risk"]["tp_mult"],
                       be_at_R=cfg["risk"]["be_at_R"],
                       tsl=cfg["risk"]["tsl"])

    micro_full = feats["micro"]; meso_full = feats["meso"]; macro_full = feats["macro"]
    micro_test = micro_full[masks["micro"]["test"]].reset_index(drop=True)
    Ls = {h: cfg["motifs"]["horizons"][h]["L"] for h in ["macro","meso","micro"]}
    weights = {h: cfg["motifs"]["horizons"][h]["weight"] for h in ["macro","meso","micro"]}
    pick_side = cfg.get("gating", {}).get("pick_side", "argmax")
    cooldown_bars = cfg["gating"]["cooldown_bars"]
    cooldown = 0

    feats_list = _pick_features(feats["micro"], cfg)  # ensure defined
    class_matchers = {"long": {}, "short": {}, "discords": {}}
    if 'multi_shapelets' in locals():
        for h in ["macro", "meso", "micro"]:
            class_matchers["long"][h] = {
                "good": MultiShapeletMatcher([{"mat": m, "eps": epsilon[h]["long"]["good"]} for m in multi_shapelets[h]["long"]["good"]]),
                "bad":  MultiShapeletMatcher([{"mat": m, "eps": epsilon[h]["long"]["bad"]}  for m in multi_shapelets[h]["long"]["bad"]]),
            }
            class_matchers["short"][h] = {
                "good": MultiShapeletMatcher([{"mat": m, "eps": epsilon[h]["short"]["good"]} for m in multi_shapelets[h]["short"]["good"]]),
                "bad":  MultiShapeletMatcher([{"mat": m, "eps": epsilon[h]["short"]["bad"]}  for m in multi_shapelets[h]["short"]["bad"]]),
            }
            _disc_eps = _as_float(epsilon[h].get("discords", None), 1.0)
            class_matchers["discords"][h] = MultiShapeletMatcher(
                [{"mat": m, "eps": _disc_eps} for m in multi_shapelets[h]["discords"]]
            )

    def slice_LF_asof(df, ts, L, cols):
        d = df[df["timestamp"] <= ts]
        if len(d) < L: return None
        return d.iloc[-L:][cols].to_numpy(dtype=float)

    debug_cap = int(cfg.get("ui", {}).get("simulate_debug_first_n", 0))
    dbg_printed = 0
    rej_counts = {"trend_guard":0, "macro_req":0, "meso_req":0, "micro_req":0, "k_of_n":0,
                  "bad":0, "score":0, "margin":0, "discord":0, "ok":0}

    def _dbg(msg):
        nonlocal dbg_printed
        if dbg_printed < debug_cap:
            print("[SIMDBG]", msg)
            dbg_printed += 1

    results = []
    sim_start = max(Ls.values())
    sim_end = len(micro_test) - cfg["labels"]["timeout_bars_micro"] - 2
    sim_total = max(0, sim_end - sim_start)
    sim_iter = wrap_iter(range(sim_start, sim_end), total=sim_total, desc=f"Sim {symbol}",
                         yaml_cfg=cfg, cli_disable=cli_disable)
    for i in sim_iter:
        if cooldown > 0:
            cooldown -= 1
            continue
        ts = micro_test.loc[i, "timestamp"]

        macro_up = None
        try:
            macro_row = macro_full[macro_full["timestamp"] <= ts].iloc[-1]
            macro_up = bool(macro_row.get("tsmom_slope_z", 0.0) >= 0.0)
        except Exception:
            try:
                micro_row = micro_test.loc[i]
                macro_up = bool(micro_row.get("tsmom_slope_z", 0.0) >= 0.0)
            except Exception:
                macro_up = None  # no regime signal available

        wins = {
            "micro": slice_LF_asof(micro_full, ts, Ls["micro"], feats_list),
            "meso":  slice_LF_asof(meso_full,  ts, Ls["meso"],  feats_list),
            "macro": slice_LF_asof(macro_full, ts, Ls["macro"], feats_list),
        }
        if any(v is None for v in wins.values()):
            continue

        has_long = ('class_matchers' in locals()) and (len(class_matchers.get("long", {})) > 0)
        if has_long:
            scores_long_good = {h: float(class_matchers["long"][h]["good"].match_score(wins[h])[0]) for h in ["macro", "meso", "micro"]}
            scores_short_good = {h: float(class_matchers["short"][h]["good"].match_score(wins[h])[0]) for h in ["macro", "meso", "micro"]}
            scores_long_bad = {h: float(class_matchers["long"][h]["bad"].match_score(wins[h])[0])   for h in ["macro", "meso", "micro"]}
            scores_short_bad = {h: float(class_matchers["short"][h]["bad"].match_score(wins[h])[0])  for h in ["macro", "meso", "micro"]}

            Sg_long_comp = composite_score(scores_long_good["macro"], scores_long_good["meso"], scores_long_good["micro"],
                                           wM=weights["macro"], wm=weights["meso"], wmu=weights["micro"])
            Sg_short_comp = composite_score(scores_short_good["macro"], scores_short_good["meso"], scores_short_good["micro"],
                                            wM=weights["macro"], wm=weights["meso"], wmu=weights["micro"])

            if pick_side == "argmax":
                if Sg_long_comp >= Sg_short_comp:
                    side, Sg, Sb = "long", scores_long_good, scores_long_bad
                else:
                    side, Sg, Sb = "short", scores_short_good, scores_short_bad
            else:
                mac_close = macro_full[macro_full["timestamp"] <= ts]["close"].tail(Ls["macro"]).to_numpy(float)
                side = "long" if mac_close[-1] - mac_close[0] >= 0 else "short"
                Sg = scores_long_good if side == "long" else scores_short_good
                Sb = scores_long_bad if side == "long" else scores_short_bad

            discord_scores = {}
            if len(class_matchers.get("discords", {})) > 0:
                discord_scores = {h: float(class_matchers["discords"][h].match_score(wins[h])[0]) for h in ["macro", "meso", "micro"]}

            decision = passes_gate(side, Sg, Sb, discord_scores, cfg, macro_up=macro_up)
            rej_counts["ok" if decision.ok else decision.reason] += 1

            if not decision.ok:
                _dbg(f"t={ts} side={side} reason={decision.reason} details={decision.details}")
                continue

            _dbg(f"t={ts} side={side} reason=OK details={decision.details}")
        else:
            def slice_asof_df(df, L):
                d = df[df["timestamp"] <= ts]
                return None if len(d) < L else d.iloc[-L:]
            wins_df = {"micro": slice_asof_df(micro_full, Ls["micro"]),
                       "meso":  slice_asof_df(meso_full,  Ls["meso"]),
                       "macro": slice_asof_df(macro_full, Ls["macro"])}
            if any(v is None for v in wins_df.values()):
                continue
            scores = {}; hits = {}
            for h in ["macro", "meso", "micro"]:
                svec = wins_df[h]["ret_z"].values
                score, hit = matchers[h].match_score(svec)
                scores[h] = float(score); hits[h] = hit
            if not (hits["macro"] and hits["meso"] and hits["micro"]):
                continue
            mac_close_df = wins_df["macro"]["close"].values
            side = "long" if mac_close_df[-1] - mac_close_df[0] > 0 else "short"
            Sg = scores
            Sb = {"macro": 0.0, "meso": 0.0, "micro": 0.0}
            discord_scores = {}

            decision = passes_gate(side, Sg, Sb, discord_scores, cfg, macro_up=macro_up)
            rej_counts["ok" if decision.ok else decision.reason] += 1

            if not decision.ok:
                _dbg(f"t={ts} side={side} reason={decision.reason} details={decision.details}")
                continue

            _dbg(f"t={ts} side={side} reason=OK details={decision.details}")

        entry = float(micro_test.loc[i, "close"])
        atr_entry = float(micro_test.loc[i, "atr"])
        median_atr = float(micro_test["atr"].rolling(500, min_periods=100).median().iloc[i])
        future_close = micro_test["close"].iloc[i:i+cfg["labels"]["timeout_bars_micro"]+2].values
        future_atr = micro_test["atr"].iloc[i:i+cfg["labels"]["timeout_bars_micro"]+2].values
        R, hold = RiskManager(sl_mult=cfg["risk"]["sl_mult"],
                              tp_mult=cfg["risk"]["tp_mult"],
                              be_at_R=cfg["risk"]["be_at_R"],
                              tsl=cfg["risk"]["tsl"]).simulate_trade(
                                  side, entry, atr_entry, future_close, future_atr, median_atr
                              )
        results.append({"i": int(i), "timestamp": str(ts), "side": side,
                        "entry": entry, "R": float(R), "hold_bars": int(hold)})
        cooldown = cooldown_bars

    print("[SIMDBG] decision summary:", dict(sorted(rej_counts.items())))

    R_list = [x["R"] for x in results]
    summary = {"symbol": symbol, "train_months": train_months, "test_months": test_months,
               "trades": len(results), "sum_R": float(np.sum(R_list) if len(R_list) > 0 else 0.0),
               "avg_R": float(np.mean(R_list) if len(R_list) > 0 else 0.0),
               "median_R": float(np.median(R_list) if len(R_list) > 0 else 0.0)}

    outdir = Path(cfg["paths"]["outputs_dir"]) / fold_id(symbol, train_months, test_months)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "trades.json", "w") as f: json.dump(results, f, indent=2)
    with open(outdir / "summary.json", "w") as f: json.dump(summary, f, indent=2)
    if export_csv:
        pd.DataFrame(results).to_csv(outdir / "trades.csv", index=False)
    return summary

# ---------- main with 4 modes + existing walkforward ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["bars", "features", "mine", "simulate", "walkforward"],
                    default="walkforward")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable progress bars (overrides YAML ui.progress)")
    ap.add_argument("--export-csv", action="store_true",
                    help="Export trades.csv per fold and walkforward_summaries.csv at the end.")
    ap.add_argument("--persist-artifacts", action="store_true",
                    help="Persist mined shapelets and epsilon per horizon to artifacts/ for reuse.")
    ap.add_argument("--use-artifacts", action="store_true",
                    help="Load previously persisted shapelets/epsilon and skip mining.")
    ap.add_argument("--fold", type=int, default=None, help="Run only this walk-forward fold (0-based).")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    months = cfg["engine"]["months"]
    folds = month_windows(months, cfg["engine"]["train_months"],
                          cfg["engine"]["test_months"], cfg["engine"]["step_months"])
    if args.fold is not None:
        if args.fold < 0 or args.fold >= len(folds):
            raise SystemExit(f"--fold out of range (0..{len(folds)-1})")
        folds = [folds[args.fold]]
    outs = []

    bars_dir = Path(cfg["paths"]["outputs_dir"]) / "bars"
    feats_dir = Path(cfg["paths"]["outputs_dir"]) / "features"
    bars_dir.mkdir(parents=True, exist_ok=True)
    feats_dir.mkdir(parents=True, exist_ok=True)

    with progress_redirect_logs(cfg, cli_disable=args.no_progress):
        if args.mode == "bars":
            # Build bars for ALL configured months/symbols once
            sym_iter = wrap_iter(cfg["engine"]["symbols"], total=len(cfg["engine"]["symbols"]),
                                 desc="Bars: symbols", yaml_cfg=cfg, cli_disable=args.no_progress)
            for sym in sym_iter:
                ticks = read_tick_months(cfg["paths"]["inputs_dir"], sym, months,
                                         yaml_cfg=cfg, cli_disable=args.no_progress)
                bars_1m = ticks_to_bars_1m(ticks)
                save_parquet(bars_1m, bars_dir / f"{sym}.parquet")

        elif args.mode == "features":
            # Requires bars; builds & caches features per horizon
            sym_iter = wrap_iter(
                cfg["engine"]["symbols"],
                total=len(cfg["engine"]["symbols"]),
                desc="Features: symbols",
                yaml_cfg=cfg,
                cli_disable=args.no_progress,
            )
            for sym in sym_iter:
                # Ensure bars parquet exists even when source is inputs_1m
                if cfg["bars"].get("source") == "inputs_1m":
                    bars_1m = read_kline_1m_months(
                        cfg["paths"]["inputs_dir"],
                        sym,
                        months,
                        yaml_cfg=cfg,
                        cli_disable=args.no_progress,
                    )
                    save_parquet(bars_1m, bars_dir / f"{sym}.parquet")
                else:
                    bars_1m = load_parquet(bars_dir / f"{sym}.parquet")

                for h, tf in [
                    ("macro", cfg["bars"]["macro_tf"]),
                    ("meso", cfg["bars"]["meso_tf"]),
                    ("micro", cfg["bars"]["micro_tf"]),
                ]:
                    f = build_feats(bars_1m, tf, cfg["bars"]["atr_window"])
                    save_parquet(f, feats_dir / f"{sym}_{h}.parquet")

                # Triple-barrier labels on micro (unchanged logic)
                micro = load_parquet(feats_dir / f"{sym}_micro.parquet")
                micro["atr"] = micro["atr"].bfill().ffill()
                lbl_cfg = cfg["labels"]
                micro["tb_label"] = triple_barrier_labels(
                    micro[["open", "high", "low", "close", "volume", "atr"]],
                    atr_col="atr",
                    up_mult=lbl_cfg["barrier_up_atr"],
                    dn_mult=lbl_cfg["barrier_dn_atr"],
                    timeout_bars=lbl_cfg["timeout_bars_micro"],
                ).values
                # --- Label sanity (all rows) ---
                try:
                    _vc_all = pd.Series(micro["tb_label"]).value_counts().reindex([-1,0,1], fill_value=0)
                    print(f"[LABELS] phase={cfg.get('_phase')} used up={float(lbl_cfg['barrier_up_atr'])} "
                          f"dn={float(lbl_cfg['barrier_dn_atr'])} timeout={int(lbl_cfg['timeout_bars_micro'])} "
                          f"use_high_low=True | counts(all)={{-1:{int(_vc_all.loc[-1])},0:{int(_vc_all.loc[0])},1:{int(_vc_all.loc[1])}}}")
                except Exception as _e:
                    print(f"[LABELS] count error (all): {_e}")
                save_parquet(micro, feats_dir / f"{sym}_micro.parquet")

        elif args.mode in ("mine", "simulate", "walkforward"):
            # Run per fold; mine-only or simulate (with or without artifacts)
            fold_iter = wrap_iter(list(enumerate(folds)), total=len(folds),
                                  desc="Folds", yaml_cfg=cfg, cli_disable=args.no_progress)
            for i_fold, (train, test) in fold_iter:
                print(f"[INFO] Fold {i_fold}: train={train} test={test}")
                sym_iter = wrap_iter(cfg["engine"]["symbols"], total=len(cfg["engine"]["symbols"]),
                                     desc=f"Symbols (fold {i_fold})", yaml_cfg=cfg, cli_disable=args.no_progress)
                for sym in sym_iter:
                    try:
                        # flag to let run_fold early-return after mining
                        cfg["_phase"] = "mine" if args.mode == "mine" else "simulate"
                        s = run_fold(cfg, sym, train, test,
                                     cli_disable=args.no_progress,
                                     export_csv=args.export_csv,
                                     persist_artifacts=args.persist_artifacts,
                                     use_artifacts=args.use_artifacts and args.mode != "mine")
                        print(s)
                        outs.append(s)
                    except Exception as e:
                        print(f"[WARN] {sym} {train}->{test} failed: {e}")

            Path(cfg["paths"]["outputs_dir"]).mkdir(parents=True, exist_ok=True)
            with open(Path(cfg["paths"]["outputs_dir"]) / "walkforward_summaries.json", "w") as f:
                json.dump(outs, f, indent=2)
            if args.export_csv:
                p = Path(cfg["paths"]["outputs_dir"]) / "walkforward_summaries.json"
                if p.exists():
                    df = pd.DataFrame(json.load(open(p, "r")))
                    df.to_csv(p.with_suffix(".csv"), index=False)
            print("DONE")

if __name__ == "__main__":
    main()
