import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, yaml

from src.utils.io import read_tick_months, ticks_to_bars_1m, resample_bars
from src.utils.indicators import feature_frame
from src.labeling.triple_barrier import triple_barrier_labels
from src.motifs.miner import sample_candidates, neighbor_density_pick, extract_shapelets
from src.motifs.matcher import ShapeletMatcher
from src.engine.gating import composite_score
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

# ---------- core fold run (unchanged simulation loop) ----------

def run_fold(cfg, symbol, train_months, test_months, cli_disable=False,
             export_csv=False, persist_artifacts=False, use_artifacts=False):
    atr_win = cfg["bars"]["atr_window"]

    # ----- Inputs: either cached bars or build from ticks -----
    bars_file = Path(cfg["paths"]["outputs_dir"]) / "bars" / f"{symbol}.parquet"
    if bars_file.exists():
        bars_1m = load_parquet(bars_file)
    else:
        ticks = read_tick_months(cfg["paths"]["inputs_dir"], symbol, train_months + test_months,
                                 yaml_cfg=cfg, cli_disable=cli_disable)
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

    # ----- Month masks -----
    def mask_by_months(df, months):
        mstr = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m").to_numpy()
        return np.isin(mstr, months)

    masks = {h: {"train": mask_by_months(feats[h], train_months),
                 "test":  mask_by_months(feats[h], test_months)} for h in feats}

    # ----- Triple-barrier on micro (labels only for inspection/caching) -----
    micro = feats["micro"].copy()
    micro["atr"] = micro["atr"].bfill().ffill()
    labels = triple_barrier_labels(
        micro[["open","high","low","close","volume","atr"]].rename(columns={"atr":"atr"}),
        atr_col="atr",
        up_mult=cfg["labels"]["barrier_up_atr"],
        dn_mult=cfg["labels"]["barrier_dn_atr"],
        timeout_bars=cfg["labels"]["timeout_bars_micro"]
    )
    micro["tb_label"] = labels.values
    save_parquet(micro, Path(cfg["paths"]["outputs_dir"]) / "features" / f"{symbol}_micro.parquet")

    # ----- Mining / ε -----
    shapelets = {}; epsilon = {}; matchers = {}
    if use_artifacts:
        # Load pre-mined shapelets/eps
        import pickle
        art_dir = Path("artifacts") / fold_id(symbol, train_months, test_months)
        art_file = art_dir / "shapelet_artifacts.pkl"
        if not art_file.exists():
            raise FileNotFoundError(f"Artifacts not found: {art_file}")
        with open(art_file, "rb") as f:
            arts = pickle.load(f)
        for h in ["macro","meso","micro"]:
            shapelets[h] = [np.asarray(x, dtype=float) for x in arts["horizons"][h]["shapelets"]]
            epsilon[h] = float(arts["horizons"][h]["epsilon"])
            matchers[h] = ShapeletMatcher([{"vec":s, "eps":epsilon[h]}
                                           for s in shapelets[h][:cfg["motifs"]["horizons"][h]["keep"]]])
    else:
        # Fresh mining from TRAIN months
        for h in wrap_iter(["macro","meso","micro"], total=3, desc=f"Mine[{symbol}]", yaml_cfg=cfg, cli_disable=cli_disable):
            L = cfg["motifs"]["horizons"][h]["L"]
            stride = cfg["motifs"]["horizons"][h]["candidate_stride"]
            top_k = cfg["motifs"]["horizons"][h]["top_k"]
            series = feats[h]["ret_z"].values[masks[h]["train"]]
            cand = sample_candidates(series, L, stride)
            if len(cand) == 0:
                raise RuntimeError(f"No candidates for {h}")
            motif_idx, discord_idx, _ = neighbor_density_pick(series, L, cand,
                                                              k=10, top_k=top_k,
                                                              yaml_cfg=cfg, cli_disable=cli_disable,
                                                              desc=f"Mine {h}@L={L}")
            shapelets[h] = [series[s:s+L].copy() for s in motif_idx]

        # ε from pooled distances (15th pct)
        from src.motifs.mass import mass
        for h in ["macro","meso","micro"]:
            keep = cfg["motifs"]["horizons"][h]["keep"]
            train_series = feats[h]["ret_z"].values[masks[h]["train"]]
            dpool = []
            for sh in shapelets[h]:
                dprof = mass(sh, train_series)
                arr = np.asarray(dprof)[np.isfinite(dprof)]
                if len(arr): dpool.append(arr)
            pool = np.concatenate(dpool) if dpool else np.array([1.0])
            eps = float(np.percentile(pool, 15))
            epsilon[h] = eps
            matchers[h] = ShapeletMatcher([{"vec":sh, "eps":eps} for sh in shapelets[h][:keep]])

    # If this run is just “mine”, persist and exit early
    if persist_artifacts and cfg.get("_phase") == "mine":
        import pickle, pathlib
        art_dir = pathlib.Path("artifacts") / fold_id(symbol, train_months, test_months)
        art_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {"symbol": symbol,
                     "fold": {"train_end": str(train_months[-1]), "test_start": str(test_months[0])},
                     "horizons": {}}
        for h in ["macro","meso","micro"]:
            artifacts["horizons"][h] = {"L": int(cfg["motifs"]["horizons"][h]["L"]),
                                        "shapelets": [np.asarray(s, dtype=float) for s in shapelets[h]],
                                        "epsilon": float(epsilon[h])}
        with open(art_dir / "shapelet_artifacts.pkl", "wb") as f:
            pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] persisted artifacts to {art_dir / 'shapelet_artifacts.pkl'}")
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
    score_min = cfg["gating"]["score_min"]; cooldown_bars = cfg["gating"]["cooldown_bars"]; cooldown = 0

    results = []
    sim_start = max(Ls.values())
    sim_end = len(micro_test) - cfg["labels"]["timeout_bars_micro"] - 2
    sim_total = max(0, sim_end - sim_start)
    sim_iter = wrap_iter(range(sim_start, sim_end), total=sim_total, desc=f"Sim {symbol}",
                         yaml_cfg=cfg, cli_disable=cli_disable)
    for i in sim_iter:
        if cooldown > 0:
            cooldown -= 1; continue
        ts = micro_test.loc[i, "timestamp"]

        def slice_asof(df, L):
            d = df[df["timestamp"] <= ts]
            return None if len(d) < L else d.iloc[-L:]

        wins = {"micro": slice_asof(micro_full, Ls["micro"]),
                "meso":  slice_asof(meso_full,  Ls["meso"]),
                "macro": slice_asof(macro_full, Ls["macro"])}
        if any(v is None for v in wins.values()):
            continue

        scores = {}; hits = {}
        for h in ["macro","meso","micro"]:
            svec = wins[h]["ret_z"].values
            score, hit = matchers[h].match_score(svec)
            scores[h] = score; hits[h] = hit
        if not (hits["macro"] and hits["meso"] and hits["micro"]):
            continue

        S = composite_score(scores["macro"], scores["meso"], scores["micro"],
                            wM=weights["macro"], wm=weights["meso"], wmu=weights["micro"])
        if S < score_min:
            continue

        mac = wins["macro"]["close"].values
        side = "long" if mac[-1] - mac[0] > 0 else "short"

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

    R_list = [x["R"] for x in results]
    summary = {"symbol": symbol, "train_months": train_months, "test_months": test_months,
               "trades": len(results), "sum_R": float(np.sum(R_list) if R_list else 0.0),
               "avg_R": float(np.mean(R_list) if R_list else 0.0),
               "median_R": float(np.median(R_list) if R_list else 0.0)}

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
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    months = cfg["engine"]["months"]
    folds = month_windows(months, cfg["engine"]["train_months"],
                          cfg["engine"]["test_months"], cfg["engine"]["step_months"])
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
            sym_iter = wrap_iter(cfg["engine"]["symbols"], total=len(cfg["engine"]["symbols"]),
                                 desc="Features: symbols", yaml_cfg=cfg, cli_disable=args.no_progress)
            for sym in sym_iter:
                bars_1m = load_parquet(bars_dir / f"{sym}.parquet")
                for h, tf in [("macro", cfg["bars"]["macro_tf"]),
                              ("meso",  cfg["bars"]["meso_tf"]),
                              ("micro", cfg["bars"]["micro_tf"])]:
                    f = build_feats(bars_1m, tf, cfg["bars"]["atr_window"])
                    save_parquet(f, feats_dir / f"{sym}_{h}.parquet")
                # Also compute & store micro triple-barrier labels for inspection
                micro = load_parquet(feats_dir / f"{sym}_micro.parquet")
                micro["atr"] = micro["atr"].bfill().ffill()
                micro["tb_label"] = triple_barrier_labels(
                    micro[["open","high","low","close","volume","atr"]],
                    atr_col="atr",
                    up_mult=cfg["labels"]["barrier_up_atr"],
                    dn_mult=cfg["labels"]["barrier_dn_atr"],
                    timeout_bars=cfg["labels"]["timeout_bars_micro"]
                ).values
                save_parquet(micro, feats_dir / f"{sym}_micro.parquet")

        elif args.mode in ("mine", "simulate", "walkforward"):
            # Run per fold; mine-only or simulate (with or without artifacts)
            fold_iter = wrap_iter(list(enumerate(folds)), total=len(folds),
                                  desc="Folds", yaml_cfg=cfg, cli_disable=args.no_progress)
            for i_fold, (train, test) in fold_iter:
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
