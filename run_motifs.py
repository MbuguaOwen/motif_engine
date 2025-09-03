
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

def month_windows(months, train, test, step):
    out=[]; m=months
    for i in range(0, len(m)-train-test+1, step):
        out.append((m[i:i+train], m[i+train:i+train+test]))
    return out

def build_feats(bars_1m, tf, atr_window):
    if tf=="1min": b=bars_1m.copy()
    else: b=resample_bars(bars_1m, tf)
    f=feature_frame(b, atr_win=atr_window); f["timestamp"]=b["timestamp"].iloc[-len(f):].values
    return f

def run_fold(cfg, symbol, train_months, test_months):
    inputs_dir = cfg["paths"]["inputs_dir"]
    atr_win = cfg["bars"]["atr_window"]

    ticks = read_tick_months(inputs_dir, symbol, train_months + test_months)
    bars_1m = ticks_to_bars_1m(ticks)

    feats = {h: build_feats(bars_1m, cfg["bars"][f"{h}_tf"], atr_win) for h in ["macro","meso","micro"]}

    def mask_by_months(df, months):
        mstr = pd.to_datetime(df["timestamp"]).strftime("%Y-%m").values
        import numpy as np
        return np.isin(mstr, months)

    masks = {h: {"train": mask_by_months(feats[h], train_months), "test": mask_by_months(feats[h], test_months)} for h in feats}

    micro = feats["micro"].copy()
    micro["atr"] = micro["atr"].fillna(method="bfill").fillna(method="ffill")
    labels = triple_barrier_labels(micro[["open","high","low","close","volume","atr"]].rename(columns={"atr":"atr"}),
                                   atr_col="atr",
                                   up_mult=cfg["labels"]["barrier_up_atr"],
                                   dn_mult=cfg["labels"]["barrier_dn_atr"],
                                   timeout_bars=cfg["labels"]["timeout_bars_micro"])
    micro["label"]=labels.values

    shapelets={}; discords={}
    for h in ["macro","meso","micro"]:
        L = cfg["motifs"]["horizons"][h]["L"]
        stride = cfg["motifs"]["horizons"][h]["candidate_stride"]
        top_k = cfg["motifs"]["horizons"][h]["top_k"]
        series = feats[h]["ret_z"].values[masks[h]["train"]]
        cand = sample_candidates(series, L, stride)
        if len(cand)==0: raise RuntimeError(f"No candidates for {h}")
        motif_idx, discord_idx, _ = neighbor_density_pick(series, L, cand, k=10, top_k=top_k)
        def extract(series, L, idxs): return [series[s:s+L].copy() for s in idxs]
        shapelets[h] = extract(series, L, motif_idx)
        discords[h] = extract(series, L, discord_idx)

    matchers={}
    for h in ["macro","meso","micro"]:
        keep = cfg["motifs"]["horizons"][h]["keep"]
        # crude ε: median of pairwise MASS distances inside train
        train_series = feats[h]["ret_z"].values[masks[h]["train"]]
        dpool=[]
        for sh in shapelets[h]:
            # Use sliding MASS distances within train
            from src.motifs.mass import mass
            dprof = mass(sh, train_series)
            dpool.append(dprof[np.isfinite(dprof)])
        import numpy as np
        pool = np.concatenate(dpool) if dpool else np.array([1.0])
        eps = float(np.percentile(pool, 15))  # conservative (tight) by default
        matchers[h] = ShapeletMatcher([{"vec":sh, "eps":eps} for sh in shapelets[h][:keep]])

    risk = RiskManager(sl_mult=cfg["risk"]["sl_mult"], tp_mult=cfg["risk"]["tp_mult"], be_at_R=cfg["risk"]["be_at_R"], tsl=cfg["risk"]["tsl"])

    micro_full = feats["micro"]
    micro_test = micro_full[masks["micro"]["test"]].reset_index(drop=True)
    meso_full = feats["meso"]; macro_full = feats["macro"]
    Ls = {h: cfg["motifs"]["horizons"][h]["L"] for h in ["macro","meso","micro"]}
    weights = {h: cfg["motifs"]["horizons"][h]["weight"] for h in ["macro","meso","micro"]}
    score_min = cfg["gating"]["score_min"]
    cooldown_bars = cfg["gating"]["cooldown_bars"]
    cooldown=0

    results=[]
    for i in range(max(Ls.values()), len(micro_test)-cfg["labels"]["timeout_bars_micro"]-2):
        if cooldown>0:
            cooldown-=1; continue
        ts = micro_test.loc[i, "timestamp"]
        def slice_asof(df, L):
            d = df[df["timestamp"]<=ts]
            return None if len(d)<L else d.iloc[-L:]
        wins = {"micro": slice_asof(micro_full, Ls["micro"]),
                "meso": slice_asof(meso_full,  Ls["meso"]),
                "macro": slice_asof(macro_full, Ls["macro"])}
        if any(v is None for v in wins.values()):
            continue

        scores={}; hits={}
        for h in ["macro","meso","micro"]:
            svec = wins[h]["ret_z"].values
            score, hit = matchers[h].match_score(svec)
            scores[h]=score; hits[h]=hit
        if not (hits["macro"] and hits["meso"] and hits["micro"]): 
            continue

        S = composite_score(scores["macro"], scores["meso"], scores["micro"], wM=weights["macro"], wm=weights["meso"], wmu=weights["micro"])
        if S < score_min: continue

        mac = wins["macro"]["close"].values
        side = "long" if mac[-1]-mac[0] > 0 else "short"

        entry = float(micro_test.loc[i,"close"])
        atr_entry = float(micro_test.loc[i,"atr"])
        median_atr = float(micro_test["atr"].rolling(500, min_periods=100).median().iloc[i])
        future_close = micro_test["close"].iloc[i:i+cfg["labels"]["timeout_bars_micro"]+2].values
        future_atr = micro_test["atr"].iloc[i:i+cfg["labels"]["timeout_bars_micro"]+2].values
        R, hold = risk.simulate_trade(side, entry, atr_entry, future_close, future_atr, median_atr)
        results.append({"i":int(i), "timestamp":str(ts), "side":side, "entry":entry, "R":float(R), "hold_bars":int(hold)})
        cooldown = cooldown_bars

    R_list=[x["R"] for x in results]
    summary={"symbol":symbol,"train_months":train_months,"test_months":test_months,
             "trades":len(results),"sum_R":float(np.sum(R_list) if R_list else 0.0),
             "avg_R":float(np.mean(R_list) if R_list else 0.0),
             "median_R":float(np.median(R_list) if R_list else 0.0)}

    outdir = Path(cfg["paths"]["outputs_dir"]) / f"{symbol}_{train_months[-1]}__{test_months[0]}"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir/"trades.json","w") as f: import json; json.dump(results, f, indent=2)
    with open(outdir/"summary.json","w") as f: import json; json.dump(summary, f, indent=2)
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["walkforward"], default="walkforward")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config,"r"))

    months = cfg["engine"]["months"]
    folds = month_windows(months, cfg["engine"]["train_months"], cfg["engine"]["test_months"], cfg["engine"]["step_months"])
    outs=[]
    for sym in cfg["engine"]["symbols"]:
        for train, test in folds:
            try:
                s = run_fold(cfg, sym, train, test)
                print(s)
                outs.append(s)
            except Exception as e:
                print(f"[WARN] {sym} {train}->{test} failed: {e}")
    Path(cfg["paths"]["outputs_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg["paths"]["outputs_dir"])/"walkforward_summaries.json","w") as f:
        import json; json.dump(outs, f, indent=2)
    print("DONE")

if __name__ == "__main__":
    main()
