# pk_count.py
import pandas as pd, pickle, glob, numpy as np

SYMBOL = "BTCUSDT"
# Set the months to the TRAIN window you want to inspect (e.g., fold 0 below)
TRAIN_MONTHS = ["2025-01","2025-02","2025-03"]

def show_label_dist(symbol=SYMBOL, months=TRAIN_MONTHS):
    df = pd.read_parquet(f"outputs/features/{symbol}_micro.parquet")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.assign(month=ts.dt.strftime("%Y-%m"))
    dft = df[df["month"].isin(months)]
    vc = dft["tb_label"].value_counts().reindex([-1,0,1], fill_value=0)
    print(f"\n[LABELS] {symbol} TRAIN months {months} -> "
          f"{{-1:{int(vc.loc[-1])}, 0:{int(vc.loc[0])}, 1:{int(vc.loc[1])}}}")
    print("[LABELS] rows_by_month:", dft["month"].value_counts().sort_index().to_dict())

def _print_artifact(path):
    A = pickle.load(open(path, "rb"))
    print(f"\n== {path} ==")
    feats = A.get("features", [])
    print("features:", len(feats))
    for h, H in A.get("horizons", {}).items():
        L = H.get("L", None)
        if "classes" in H:  # new schema
            def cnt(side, kind): return len(H["classes"][side][kind]["shapelets"])
            def eps(side, kind): return float(H["classes"][side][kind]["epsilon"])
            disc = H.get("discords", {})
            print(f" {h}: L={L} | long.good {cnt('long','good')} (eps={eps('long','good'):.4f})"
                  f" | long.bad {cnt('long','bad')} (eps={eps('long','bad'):.4f})"
                  f" | short.good {cnt('short','good')} (eps={eps('short','good'):.4f})"
                  f" | short.bad {cnt('short','bad')} (eps={eps('short','bad'):.4f})"
                  f" | discords {len(disc.get('shapelets', []))}")
        else:  # legacy fallback
            shp = H.get("shapelets", [])
            eps = float(H.get("epsilon", 1.0))
            print(f" {h}: L={L} | legacy shapelets: {len(shp)} (eps={eps:.4f})")

def show_shapelet_counts():
    arts = sorted(glob.glob("artifacts/*/shapelet_artifacts.pkl"))
    if not arts:
        print("No artifacts found under artifacts/*")
        return
    for f in arts:
        try:
            _print_artifact(f)
        except Exception as e:
            print(f"\n== {f} ==\n<error reading artifact: {e}>")

if __name__ == "__main__":
    show_label_dist()
    show_shapelet_counts()

