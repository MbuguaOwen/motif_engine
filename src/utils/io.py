
from pathlib import Path
import pandas as pd

def read_tick_months(inputs_dir: str, symbol: str, months: list) -> pd.DataFrame:
    dfs = []
    symdir = Path(inputs_dir) / symbol
    for m in months:
        fpath = symdir / f"{symbol}-ticks-{m}.csv"
        if fpath.exists():
            df = pd.read_csv(fpath)
            if "timestamp" not in df.columns or "price" not in df.columns:
                raise ValueError(f"{fpath} missing required columns")
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No tick CSVs found under {symdir} for months={months}")
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp")
    return df

def ticks_to_bars_1m(ticks: pd.DataFrame) -> pd.DataFrame:
    ticks = ticks.copy().set_index("timestamp")
    if "qty" not in ticks.columns:
        ticks["qty"] = 0.0
    o = ticks["price"].resample("1min").first().rename("open")
    h = ticks["price"].resample("1min").max().rename("high")
    l = ticks["price"].resample("1min").min().rename("low")
    c = ticks["price"].resample("1min").last().rename("close")
    v = ticks["qty"].resample("1min").sum().rename("volume")
    bars = pd.concat([o,h,l,c,v], axis=1).dropna()
    bars.index.name = "timestamp"
    return bars.reset_index()

def resample_bars(bars_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    b = bars_1m.copy().set_index("timestamp")
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    out = b.resample(tf).agg(agg).dropna()
    out.index.name = "timestamp"
    return out.reset_index()
