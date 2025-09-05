import pandas as pd, pickle, glob

SYMBOL = "BTCUSDT"
TRAIN_MONTHS = ["2025-01","2025-02","2025-03"]

def show_label_dist(symbol=SYMBOL, months=TRAIN_MONTHS):
    df = pd.read_parquet(f"outputs/features/{symbol}_micro.parquet")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.assign(month=ts.dt.strftime("%Y-%m"))
    print(f"\nLabel distribution for {symbol} on months {months}:")
    print(df[df["month"].isin(months)]["tb_label"].value_counts(dropna=False).sort_index().to_dict())
    print("\nPer-month counts (rows):")
    print(df["month"].value_counts().sort_index().to_dict())

def show_shapelet_counts():
    for f in sorted(glob.glob("artifacts/*/shapelet_artifacts.pkl")):
        A = pickle.load(open(f, "rb"))
        print(f"\n== {f} ==")
        print("features:", len(A.get("features", [])))
        for h, H in A["horizons"].items():
            L = H["L"]
            def cnt(side, kind): return len(H["classes"][side][kind]["shapelets"])
            def eps(side, kind): return float(H["classes"][side][kind]["epsilon"])
            print(f" {h}: L={L} | long.good {cnt('long','good')} (eps={eps('long','good'):.4f})"
                  f" | long.bad {cnt('long','bad')} (eps={eps('long','bad'):.4f})"
                  f" | short.good {cnt('short','good')} (eps={eps('short','good'):.4f})"
                  f" | short.bad {cnt('short','bad')} (eps={eps('short','bad'):.4f})"
                  f" | discords {len(H.get('discords',{}).get('shapelets',[]))}")

if __name__ == "__main__":
    show_label_dist()
    show_shapelet_counts()
