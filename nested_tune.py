# nested_tune.py
# Train-only (nested) tuner for gating thresholds & horizon weights.
# Finds a gate that yields positive/stable R on inner folds (train months only),
# then freezes it and evaluates the outer test month.

import argparse, copy, json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import run_motifs as RM

# Patch: guaranteed bars + quiet inner runs
from contextlib import redirect_stdout
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def mk_bar(total, desc):
    return tqdm(total=total, desc=desc, dynamic_ncols=True, mininterval=0.2)

# ---------- local artifact helpers ----------
from pathlib import Path
import copy  # already imported, keep it

def _artifact_path(symbol, train_months, test_months):
    fid = RM.fold_id(symbol, train_months, test_months)
    return Path("artifacts") / fid / "shapelet_artifacts.pkl"

def ensure_artifacts_once(base_cfg, symbol, train_months, test_months):
    if _artifact_path(symbol, train_months, test_months).exists():
        return
    cfg_m = copy.deepcopy(base_cfg)
    cfg_m["_phase"] = "mine"
    cfg_m.setdefault("ui", {})
    cfg_m["ui"]["quiet"] = True
    cfg_m["ui"]["progress"] = False
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        RM.run_fold(cfg_m, symbol, train_months, test_months,
                    cli_disable=True, export_csv=False,
                    persist_artifacts=True, use_artifacts=False)

# ---------- simple tqdm fallback ----------
try:
    from src.ui.progress import bar
except Exception:
    from tqdm import tqdm
    def bar(total, desc, yaml_cfg=None, cli_disable=False):
        if cli_disable:
            class _B: 
                def update(self,*a,**k): pass
                def close(self): pass
            return _B()
        return tqdm(total=total, desc=desc, dynamic_ncols=True)

# ---------- sampling space ----------
def _rnd(a,b): return float(a + (b-a)*random.random())
def _clip(x,a,b): return float(max(a, min(b, x)))
def _normalize_weights(w):
    s = sum(w.values()); 
    return {k: (v/s if s>0 else 1/len(w)) for k,v in w.items()}

def sample_gate():
    return {
        "pick_side": random.choice(["argmax","macro_sign"]),
        "score_min": _rnd(0.08, 0.60),
        "score_min_per_horizon": {
            "macro": _rnd(0.05, 0.65),
            "meso":  _rnd(0.05, 0.65),
            "micro": _rnd(0.05, 0.65),
        },
        "alpha": _rnd(0.25, 0.75),
        "bad_max": _rnd(0.55, 0.95),
        "margin_min": _rnd(0.00, 0.12),
        "discord_block_min": _rnd(0.95, 0.999),
        "require_at_least_k_of": random.choice([0,1,2,3]),
        "cooldown_bars": int(_rnd(0, 60)),
        "weights": _normalize_weights({
            "macro": _rnd(0.20, 0.70),
            "meso":  _rnd(0.10, 0.60),
            "micro": _rnd(0.05, 0.50),
        }),
    }

def mutate_near(c0):
    c = copy.deepcopy(c0)
    def j(v,lo,hi,p=0.20): return _clip(v * _rnd(1-p,1+p), lo, hi)
    c["score_min"] = j(c["score_min"], 0.02, 0.70)
    for h in ("macro","meso","micro"):
        c["score_min_per_horizon"][h] = j(c["score_min_per_horizon"][h], 0.0, 0.8)
        c["weights"][h] = j(c["weights"][h], 0.01, 0.9)
    c["weights"] = _normalize_weights(c["weights"])
    c["alpha"] = j(c["alpha"], 0.1, 0.9)
    c["bad_max"] = j(c["bad_max"], 0.4, 0.99)
    c["margin_min"] = j(c["margin_min"], 0.0, 0.2)
    c["discord_block_min"] = j(c["discord_block_min"], 0.90, 0.9995)
    c["cooldown_bars"] = int(max(0, min(240, round(j(c["cooldown_bars"], 0, 240)))))
    if random.random() < 0.2:
        c["pick_side"] = "argmax" if c["pick_side"]=="macro_sign" else "macro_sign"
    if random.random() < 0.2:
        c["require_at_least_k_of"] = random.choice([0,1,2,3])
    return c

def apply_gate(cfg, cand):
    out = copy.deepcopy(cfg)
    g = out.setdefault("gating", {})
    g["pick_side"] = cand["pick_side"]
    g["score_min"] = cand["score_min"]
    g["score_min_per_horizon"] = cand["score_min_per_horizon"]
    g["alpha"] = cand["alpha"]
    g["bad_max"] = cand["bad_max"]
    g["margin_min"] = cand["margin_min"]
    g["discord_block_min"] = cand["discord_block_min"]
    g["require_at_least_k_of"] = cand["require_at_least_k_of"]
    g["cooldown_bars"] = cand["cooldown_bars"]
    for h in ("macro","meso","micro"):
        out["motifs"]["horizons"][h]["weight"] = float(cand["weights"][h])
    return out

# ---------- evaluation ----------
def eval_fold(cfg, symbol, train_months, test_months, use_artifacts=True):
    cfg = copy.deepcopy(cfg)
    cfg["_phase"] = "simulate"
    cfg.setdefault("ui", {})
    cfg["ui"]["quiet"] = True
    cfg["ui"]["progress"] = False
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        s = RM.run_fold(cfg, symbol, train_months, test_months,
                        cli_disable=True, export_csv=False,
                        persist_artifacts=False, use_artifacts=use_artifacts)
    return int(s.get("trades",0)), float(s.get("sum_R",0.0))

def eval_candidate_on_splits(cfg_c, symbols, splits_subset, use_artifacts=True, max_workers=2):
    """Evaluate a candidate on a subset of (train, val) inner splits × symbols in parallel (Windows-safe)."""
    # Fallback to serial if only 1 worker
    if max_workers <= 1 or len(splits_subset) == 0:
        trades_tot, sumR_list = 0, []
        for (tr_m, va_m) in splits_subset:
            for sym in symbols:
                t, R = eval_fold(cfg_c, sym, tr_m, va_m, use_artifacts)
                trades_tot += t; sumR_list.append(R)
        return trades_tot, sumR_list

    tasks = []
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for (tr_m, va_m) in splits_subset:
            for sym in symbols:
                tasks.append(ex.submit(eval_fold, cfg_c, sym, tr_m, va_m, use_artifacts))
        for fut in as_completed(tasks):
            t, R = fut.result()
            results.append((t, R))
    trades_tot = sum(t for t, _ in results)
    sumR_list = [R for _, R in results]
    return trades_tot, sumR_list

def objective(sum_R_mean, trades_pm, target_pm, avg_R, min_avg_r, stability_pen):
    if avg_R < min_avg_r: 
        return -1e9
    penalty = abs(trades_pm - target_pm) if target_pm > 0 else 0.0
    return float(sum_R_mean - 0.25*penalty - stability_pen)

def discover_on_train(base_cfg, symbols, outer_train_months, runs, refine_top, neighbors,
                      target_pm, min_avg_r, seed, outdir):
    random.seed(seed); np.random.seed(seed)

    # Build inner splits (leave-one-month-out on outer train months)
    months = list(outer_train_months)
    inner_splits = []
    if len(months) == 1:
        inner_splits = [(months, months)]
    else:
        for i in range(len(months)):
            tr_m = [m for j, m in enumerate(months) if j != i]
            va_m = [months[i]]
            inner_splits.append((tr_m, va_m))

    # Prewarm artifacts once (quiet) so the search starts fast
    todo = [(sym, tr_m, va_m) for (tr_m, va_m) in inner_splits for sym in symbols]
    pre = mk_bar(len(todo), "Prewarm artifacts")
    for (sym, tr_m, va_m) in todo:
        ensure_artifacts_once(base_cfg, sym, tr_m, va_m)
        pre.update(1)
    pre.close()

    # Successive-halving rung sizes: ~1/4 -> 1/2 -> all
    S = len(inner_splits)
    rung_sizes = [max(1, S//4), max(1, S//2), S] if S > 0 else [0, 0, 0]
    keep_frac = 1/3
    max_workers = max(1, min(os.cpu_count() or 2, 4))  # small pool plays nicest on Windows

    outer = mk_bar(runs + refine_top*neighbors, "Nested tune (train only)")

    # -------- Stage A: many candidates on small budget --------
    stageA = []
    for _ in range(runs):
        cand = sample_gate()
        cfg_c = apply_gate(base_cfg, cand)
        trades_tot, sumR_list = eval_candidate_on_splits(
            cfg_c, symbols, inner_splits[:rung_sizes[0]], use_artifacts=True, max_workers=max_workers
        )
        folds = max(1, rung_sizes[0] * max(1, len(symbols)))
        sumR_mean = float(np.mean(sumR_list)) if sumR_list else 0.0
        avg_R = (sumR_mean / (trades_tot / folds)) if trades_tot > 0 else 0.0
        trades_pm = trades_tot / folds
        med = float(np.median(sumR_list)) if sumR_list else 0.0
        worst = float(np.min(sumR_list)) if sumR_list else 0.0
        stab_pen = 0.1*max(0.0, -med) + 0.1*max(0.0, -worst)
        obj = objective(sumR_mean, trades_pm, target_pm, avg_R, min_avg_r, stab_pen)
        stageA.append({"objective": obj, "cand": cand})
        outer.update(1)

    survivorsA = sorted(stageA, key=lambda r: r["objective"], reverse=True)[:max(1, int(len(stageA)*keep_frac))]

    # -------- Stage B: survivors on medium budget --------
    stageB = []
    for r in survivorsA:
        cand = r["cand"]
        cfg_c = apply_gate(base_cfg, cand)
        trades_tot, sumR_list = eval_candidate_on_splits(
            cfg_c, symbols, inner_splits[:rung_sizes[1]], use_artifacts=True, max_workers=max_workers
        )
        folds = max(1, rung_sizes[1] * max(1, len(symbols)))
        sumR_mean = float(np.mean(sumR_list)) if sumR_list else 0.0
        avg_R = (sumR_mean / (trades_tot / folds)) if trades_tot > 0 else 0.0
        trades_pm = trades_tot / folds
        med = float(np.median(sumR_list)) if sumR_list else 0.0
        worst = float(np.min(sumR_list)) if sumR_list else 0.0
        stab_pen = 0.1*max(0.0, -med) + 0.1*max(0.0, -worst)
        obj = objective(sumR_mean, trades_pm, target_pm, avg_R, min_avg_r, stab_pen)
        stageB.append({"objective": obj, "cand": cand})

    elites = sorted(stageB, key=lambda r: r["objective"], reverse=True)[:max(1, int(len(stageB)*keep_frac))]

    # -------- Stage C: refine around elites on full budget --------
    results = []
    for e in elites[:refine_top]:
        base_cand = e["cand"]
        for _ in range(max(1, neighbors)):
            cand = mutate_near(base_cand)
            cfg_c = apply_gate(base_cfg, cand)
            trades_tot, sumR_list = eval_candidate_on_splits(
                cfg_c, symbols, inner_splits[:rung_sizes[2]], use_artifacts=True, max_workers=max_workers
            )
            folds = max(1, rung_sizes[2] * max(1, len(symbols)))
            sumR_mean = float(np.mean(sumR_list)) if sumR_list else 0.0
            avg_R = (sumR_mean / (trades_tot / folds)) if trades_tot > 0 else 0.0
            trades_pm = trades_tot / folds
            med = float(np.median(sumR_list)) if sumR_list else 0.0
            worst = float(np.min(sumR_list)) if sumR_list else 0.0
            stab_pen = 0.1*max(0.0, -med) + 0.1*max(0.0, -worst)
            obj = objective(sumR_mean, trades_pm, target_pm, avg_R, min_avg_r, stab_pen)
            results.append({
                "objective": obj, "sum_R_mean": sumR_mean, "trades_pm": trades_pm, "avg_R": avg_R,
                "med_R": med, "worst_R": worst, "cand": cand
            })
            outer.update(1)
    outer.close()

    # Persist search table + overlay
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        c = r["cand"]
        rows.append({
            "objective": r["objective"], "sum_R_mean": r["sum_R_mean"],
            "trades_pm": r["trades_pm"], "avg_R": r["avg_R"], "med_R": r["med_R"], "worst_R": r["worst_R"],
            "pick_side": c["pick_side"], "score_min": c["score_min"],
            "score_min_per_horizon": yaml.safe_dump(c["score_min_per_horizon"], sort_keys=True),
            "alpha": c["alpha"], "bad_max": c["bad_max"], "margin_min": c["margin_min"],
            "discord_block_min": c["discord_block_min"], "k_of_n": c["require_at_least_k_of"],
            "cooldown_bars": c["cooldown_bars"], "w_macro": c["weights"]["macro"],
            "w_meso": c["weights"]["meso"], "w_micro": c["weights"]["micro"],
        })
    pd.DataFrame(rows).to_csv(outdir / "nested_search.csv", index=False)

    best = max(results, key=lambda r: r["objective"])
    best_gate = best["cand"]
    overlay = {
        "gating": {
            "pick_side": best_gate["pick_side"],
            "score_min": best_gate["score_min"],
            "score_min_per_horizon": best_gate["score_min_per_horizon"],
            "alpha": best_gate["alpha"], "bad_max": best_gate["bad_max"],
            "margin_min": best_gate["margin_min"], "discord_block_min": best_gate["discord_block_min"],
            "require_at_least_k_of": best_gate["require_at_least_k_of"], "cooldown_bars": best_gate["cooldown_bars"],
        },
        "motifs": {"horizons": {
            "macro": {"weight": best_gate["weights"]["macro"]},
            "meso":  {"weight": best_gate["weights"]["meso"]},
            "micro": {"weight": best_gate["weights"]["micro"]},
        }}
    }
    yml_path = outdir / "gate_best.yaml"
    with open(yml_path, "w") as f:
        yaml.safe_dump(overlay, f, sort_keys=False)
    return outdir / "nested_search.csv", yml_path, {
        "objective": best["objective"], "avg_R": best["avg_R"], "trades_pm": best["trades_pm"], "cand": best_gate
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--runs", type=int, default=80)
    ap.add_argument("--refine-top", type=int, default=10)
    ap.add_argument("--neighbors", type=int, default=3)
    ap.add_argument("--target-trades-per-month", type=float, default=0.0)
    ap.add_argument("--min-avg-r", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--use-artifacts", action="store_true")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(args.base_config, "r"))
    symbols = list(base_cfg["engine"]["symbols"])
    folds = RM.month_windows(
        base_cfg["engine"]["months"],
        base_cfg["engine"]["train_months"],
        base_cfg["engine"]["test_months"],
        base_cfg["engine"]["step_months"],
    )
    root_out = Path(base_cfg["paths"]["outputs_dir"]) / "nested_tuning"
    all_rows = []

    # Outer loop: normal walk-forward folds
    for i, (outer_train, outer_test) in enumerate(folds):
        fold_out = root_out / f"fold_{i:02d}"
        # Ensure outer artifacts exist (train→test) but tuning will only use train splits
        if args.use_artifacts:
            for sym in symbols:
                ensure_artifacts_once(base_cfg, sym, outer_train, outer_test)

        csv_path, yml_path, best = discover_on_train(
            base_cfg, symbols, outer_train,
            runs=args.runs, refine_top=args.refine_top, neighbors=args.neighbors,
            target_pm=args.target_trades_per_month, min_avg_r=args.min_avg_r,
            seed=args.seed, outdir=fold_out
        )

        # Evaluate the chosen gate on the OUTER test month (one shot)
        cfg_test = apply_gate(base_cfg, best["cand"])
        cfg_test["_phase"] = "simulate"
        sum_R_test = 0.0; trades_test = 0
        for sym in symbols:
            t, R = eval_fold(cfg_test, sym, outer_train, outer_test, use_artifacts=True)
            trades_test += t; sum_R_test += R
        all_rows.append({
            "fold": i, "train_months": ",".join(outer_train), "test_months": ",".join(outer_test),
            "trades_test": trades_test, "sum_R_test": sum_R_test,
            "best_objective": best["objective"], "best_avg_R": best["avg_R"], "best_trades_pm": best["trades_pm"],
            "overlay": str(yml_path), "search_csv": str(csv_path)
        })

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(root_out / "outer_results.csv", index=False)
    print("\n[NESTED] Done. Per-fold overlays in outputs/nested_tuning/fold_*/gate_best.yaml")
    print("[NESTED] Summary saved to outputs/nested_tuning/outer_results.csv")

if __name__ == "__main__":
    main()
