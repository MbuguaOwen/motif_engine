# tune_gate.py
# Random search + local refine for gating & horizon weights to maximize walk-forward sum_R.
# Reuses persisted multivariate shapelet artifacts (no re-mining).
import argparse, copy, json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import run_motifs as RM  # uses month_windows & run_fold
try:
    from src.ui.progress import wrap_iter, progress_redirect_logs, bar
except Exception:
    from contextlib import nullcontext
    from tqdm import tqdm
    def wrap_iter(it, total=None, desc="", yaml_cfg=None, cli_disable=False):
        if cli_disable:
            for x in it:
                yield x
        else:
            yield from tqdm(it, total=total, desc=desc, dynamic_ncols=True)
    def progress_redirect_logs(*args, **kwargs): return nullcontext()
    def bar(total, desc, yaml_cfg=None, cli_disable=False):
        if cli_disable:
            class _B:
                def update(self,*a,**k): pass
                def close(self): pass
            return _B()
        return tqdm(total=total, desc=desc, dynamic_ncols=True)

# -------------------------------
# Helpers
# -------------------------------

def _folds(cfg):
    return RM.month_windows(
        cfg["engine"]["months"],
        cfg["engine"]["train_months"],
        cfg["engine"]["test_months"],
        cfg["engine"]["step_months"],
    )

def _fold_id(symbol, train, test):
    return f"{symbol}_{train[-1]}__{test[0]}"

def _artifact_path(symbol, train, test):
    return Path("artifacts") / _fold_id(symbol, train, test) / "shapelet_artifacts.pkl"

def _ensure_artifacts(cfg, folds, symbols, cli_disable=True):
    """Mine once per fold if missing."""
    need = []
    for train, test in folds:
        for sym in symbols:
            if not _artifact_path(sym, train, test).exists():
                need.append((sym, train, test))
    if not need:
        return
    print(f"[tune] Missing {len(need)} artifact(s) → mining once...")
    for (sym, train, test) in wrap_iter(need, total=len(need), desc="Mine artifacts", yaml_cfg=cfg, cli_disable=cli_disable):
        cfg_m = copy.deepcopy(cfg)
        cfg_m["_phase"] = "mine"
        cfg_m.setdefault("ui", {})
        cfg_m["ui"]["quiet"] = True
        cfg_m["ui"]["progress"] = False
        RM.run_fold(cfg_m, sym, train, test,
                    cli_disable=True,
                    export_csv=False,
                    persist_artifacts=True,
                    use_artifacts=False)

def _rnd(a, b):  # float uniform
    return float(a + (b - a) * random.random())

def _clip(x, a, b):
    return float(max(a, min(b, x)))

def _normalize_weights(w):
    s = sum(w.values())
    if s <= 0: return {"macro": 0.5, "meso": 0.3, "micro": 0.2}
    return {k: float(v / s) for k, v in w.items()}

def _sample_candidate():
    # Gating knobs
    cand = {
        "pick_side": random.choice(["argmax", "macro_sign"]),
        "score_min": _rnd(0.10, 0.55),
        "score_min_per_horizon": {
            "macro": _rnd(0.05, 0.60),
            "meso":  _rnd(0.05, 0.60),
            "micro": _rnd(0.05, 0.60),
        },
        "alpha": _rnd(0.25, 0.75),          # BAD penalty in margin
        "bad_max": _rnd(0.55, 0.95),        # hard veto threshold
        "margin_min": _rnd(0.00, 0.12),
        "discord_block_min": _rnd(0.95, 0.999),
        "require_at_least_k_of": random.choice([0, 1, 2, 3]),
        "cooldown_bars": int(_rnd(0, 60)),
        # Horizon weights (affects composite score)
        "weights": _normalize_weights({
            "macro": _rnd(0.2, 0.7),
            "meso":  _rnd(0.1, 0.6),
            "micro": _rnd(0.05, 0.5),
        }),
    }
    return cand

def _mutate_near(parent):
    c = copy.deepcopy(parent)
    def jitter(v, lo, hi, pct=0.20):
        return _clip(v * _rnd(1.0 - pct, 1.0 + pct), lo, hi)
    c["score_min"] = jitter(c["score_min"], 0.05, 0.65)
    for h in ("macro","meso","micro"):
        c["score_min_per_horizon"][h] = jitter(c["score_min_per_horizon"][h], 0.0, 0.8)
        c["weights"][h] = jitter(c["weights"][h], 0.01, 0.9)
    c["weights"] = _normalize_weights(c["weights"])
    c["alpha"] = jitter(c["alpha"], 0.1, 0.9)
    c["bad_max"] = jitter(c["bad_max"], 0.4, 0.99)
    c["margin_min"] = jitter(c["margin_min"], 0.0, 0.2)
    c["discord_block_min"] = jitter(c["discord_block_min"], 0.90, 0.9995)
    c["cooldown_bars"] = int(max(0, min(240, round(jitter(c["cooldown_bars"], 0, 240)))))
    if random.random() < 0.2:
        c["pick_side"] = "argmax" if parent["pick_side"] == "macro_sign" else "macro_sign"
    if random.random() < 0.2:
        c["require_at_least_k_of"] = random.choice([0,1,2,3])
    return c

def _apply_candidate(base_cfg, cand):
    cfg = copy.deepcopy(base_cfg)
    # gating block
    g = cfg.setdefault("gating", {})
    g["pick_side"] = cand["pick_side"]
    g["score_min"] = cand["score_min"]
    g["score_min_per_horizon"] = cand["score_min_per_horizon"]
    g["alpha"] = cand["alpha"]
    g["bad_max"] = cand["bad_max"]
    g["margin_min"] = cand["margin_min"]
    g["discord_block_min"] = cand["discord_block_min"]
    g["require_at_least_k_of"] = cand["require_at_least_k_of"]
    g["cooldown_bars"] = cand["cooldown_bars"]
    # horizon weights
    for h in ("macro","meso","micro"):
        cfg["motifs"]["horizons"][h]["weight"] = float(cand["weights"][h])
    return cfg

def _evaluate_candidate(base_cfg, cand, folds, symbols, use_artifacts=True, inner_bar=None, yaml_cfg=None, no_progress=False):
    cfg = _apply_candidate(base_cfg, cand)
    cfg["_phase"] = "simulate"
    cfg.setdefault("ui", {})
    cfg["ui"]["quiet"] = True       # keep inner logs quiet
    cfg["ui"]["progress"] = False   # suppress inner tqdm bars

    totals = {"trades": 0, "sum_R": 0.0}
    for (train, test) in folds:
        for sym in symbols:
            s = RM.run_fold(cfg, sym, train, test,
                            cli_disable=True,          # no inner bars
                            export_csv=False,
                            persist_artifacts=False,
                            use_artifacts=use_artifacts)
            totals["trades"] += int(s.get("trades", 0))
            totals["sum_R"]  += float(s.get("sum_R", 0.0))
            if inner_bar is not None:
                inner_bar.update(1)
    avg_R = (totals["sum_R"] / totals["trades"]) if totals["trades"] > 0 else 0.0
    return totals["trades"], totals["sum_R"], avg_R

def _objective(sum_R, trades_per_month, target_trades_pm, lam=0.25):
    # Encourage matching trade-rate while maximizing sum_R.
    if target_trades_pm <= 0:
        return sum_R
    penalty = abs(trades_per_month - target_trades_pm)
    return float(sum_R - lam * penalty)

def _save_outputs(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"gate_search_{ts}.csv"
    df.to_csv(csv_path, index=False)
    # Best overlay for immediate use
    best = max(results, key=lambda r: r["objective"])
    overlay = {
        "gating": {
            "pick_side": best["pick_side"],
            "score_min": best["score_min"],
            "score_min_per_horizon": yaml.safe_load(best["score_min_per_horizon"]),
            "alpha": best["alpha"],
            "bad_max": best["bad_max"],
            "margin_min": best["margin_min"],
            "discord_block_min": best["discord_block_min"],
            "require_at_least_k_of": int(best["require_at_least_k_of"]),
            "cooldown_bars": int(best["cooldown_bars"]),
        },
        "motifs": {
            "horizons": {
                "macro": {"weight": best["w_macro"]},
                "meso":  {"weight": best["w_meso"]},
                "micro": {"weight": best["w_micro"]},
            }
        }
    }
    yml_path = out_dir / "gate_best.yaml"
    with open(yml_path, "w") as f:
        yaml.safe_dump(overlay, f, sort_keys=False)
    return csv_path, yml_path

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True, help="Base motifs.yaml")
    ap.add_argument("--runs", type=int, default=120, help="Random runs before refine")
    ap.add_argument("--refine-top", type=int, default=10, help="Locally refine around this many elites")
    ap.add_argument("--neighbors", type=int, default=3, help="Neighbors per elite")
    ap.add_argument("--target-trades-per-month", type=float, default=0.0, help="Soft target; 0 to disable")
    ap.add_argument("--min-avg-r", type=float, default=0.0, help="Discard candidates with avg_R below this")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--use-artifacts", action="store_true", help="Skip mining; require artifacts exist")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    base_cfg = yaml.safe_load(open(args["base-config"] if isinstance(args, dict) else args.base_config, "r"))
    folds = _folds(base_cfg)
    symbols = list(base_cfg["engine"]["symbols"])
    out_dir = Path(base_cfg["paths"]["outputs_dir"]) / "tuning"

    # Ensure artifacts once (if asked to reuse)
    if args.use_artifacts:
        _ensure_artifacts(base_cfg, folds, symbols, cli_disable=args.no_progress)

    results = []

    # -------- Random Search --------
    with progress_redirect_logs(base_cfg, cli_disable=args.no_progress):
        outer = bar(total=args.runs, desc="Gate search", yaml_cfg=base_cfg, cli_disable=args.no_progress)
        for r in range(args.runs):
            # per-candidate inner bar (folds × symbols)
            steps = len(folds) * max(1, len(symbols))
            inner = bar(total=steps, desc=f"cand {r+1}/{args.runs}", yaml_cfg=base_cfg, cli_disable=args.no_progress)

            cand = _sample_candidate()
            trades, sum_R, avg_R = _evaluate_candidate(
                base_cfg, cand, folds, symbols,
                use_artifacts=args.use_artifacts,
                inner_bar=inner, yaml_cfg=base_cfg, no_progress=args.no_progress
            )
            inner.close()

            months = len(folds) * max(1, len(symbols))
            trades_pm = trades / months if months > 0 else 0.0
            obj = -1e9 if avg_R < args.min_avg_r else _objective(sum_R, trades_pm, args.target_trades_per_month)
            results.append({
                "objective": obj, "sum_R": sum_R, "trades": trades, "avg_R": avg_R, "trades_pm": trades_pm,
                "pick_side": cand["pick_side"], "score_min": cand["score_min"],
                "score_min_per_horizon": yaml.safe_dump(cand["score_min_per_horizon"], sort_keys=True),
                "alpha": cand["alpha"], "bad_max": cand["bad_max"],
                "margin_min": cand["margin_min"], "discord_block_min": cand["discord_block_min"],
                "require_at_least_k_of": cand["require_at_least_k_of"], "cooldown_bars": cand["cooldown_bars"],
                "w_macro": cand["weights"]["macro"], "w_meso": cand["weights"]["meso"], "w_micro": cand["weights"]["micro"],
            })
            outer.update(1)
        outer.close()

    # -------- Local Refine --------
    elites = sorted(results, key=lambda r: r["objective"], reverse=True)[:max(0, args.refine_top)]
    with progress_redirect_logs(base_cfg, cli_disable=args.no_progress):
        total_refine = max(0, len(elites) * max(1, args.neighbors))
        b2 = bar(total=total_refine, desc="Refine elites", yaml_cfg=base_cfg, cli_disable=args.no_progress)
        for elite in elites:
            base_cand = {
                "pick_side": elite["pick_side"],
                "score_min": elite["score_min"],
                "score_min_per_horizon": yaml.safe_load(elite["score_min_per_horizon"]),
                "alpha": elite["alpha"], "bad_max": elite["bad_max"],
                "margin_min": elite["margin_min"], "discord_block_min": elite["discord_block_min"],
                "require_at_least_k_of": int(elite["require_at_least_k_of"]), "cooldown_bars": int(elite["cooldown_bars"]),
                "weights": {"macro": elite["w_macro"], "meso": elite["w_meso"], "micro": elite["w_micro"]},
            }
            for _ in range(max(1, args.neighbors)):
                cand = _mutate_near(base_cand)
                trades, sum_R, avg_R = _evaluate_candidate(base_cfg, cand, folds, symbols, use_artifacts=args.use_artifacts)
                months = len(folds) * max(1, len(symbols))
                trades_pm = trades / months if months > 0 else 0.0
                if avg_R < args.min_avg_r:
                    obj = -1e9
                else:
                    obj = _objective(sum_R, trades_pm, args.target_trades_per_month)
                results.append({
                    "objective": obj, "sum_R": sum_R, "trades": trades, "avg_R": avg_R, "trades_pm": trades_pm,
                    "pick_side": cand["pick_side"],
                    "score_min": cand["score_min"],
                    "score_min_per_horizon": yaml.safe_dump(cand["score_min_per_horizon"], sort_keys=True),
                    "alpha": cand["alpha"], "bad_max": cand["bad_max"],
                    "margin_min": cand["margin_min"], "discord_block_min": cand["discord_block_min"],
                    "require_at_least_k_of": cand["require_at_least_k_of"], "cooldown_bars": cand["cooldown_bars"],
                    "w_macro": cand["weights"]["macro"], "w_meso": cand["weights"]["meso"], "w_micro": cand["weights"]["micro"],
                })
                b2.update(1)
        b2.close()

    csv_path, yml_path = _save_outputs(results, out_dir)
    top5 = sorted(results, key=lambda r: r["objective"], reverse=True)[:5]
    print("\n[tune] Top-5:")
    for i, r in enumerate(top5, 1):
        print(f"{i:>2}. obj={r['objective']:.3f} | sum_R={r['sum_R']:.2f} | avg_R={r['avg_R']:.3f} | trades_pm={r['trades_pm']:.2f} "
              f"| side={r['pick_side']} | k={r['require_at_least_k_of']} | w=({r['w_macro']:.2f},{r['w_meso']:.2f},{r['w_micro']:.2f})")

    print(f"\n[tune] Saved:\n - {csv_path}\n - {yml_path}\nUse the overlay with:  python -m run_motifs --config configs/motifs.yaml --use-artifacts")

if __name__ == "__main__":
    main()
