
from pathlib import Path
import pandas as pd
from src.ui.progress import wrap_iter, bar

def read_tick_months(inputs_dir: str, symbol: str, months: list, yaml_cfg: dict = None, cli_disable: bool = False) -> pd.DataFrame:
    """
    Read per-month tick CSV files for a symbol.
    Progress is shown per-month when enabled via config/flag.
    """
    dfs = []
    symdir = Path(inputs_dir) / symbol
    months_iter = wrap_iter(months, total=len(months), desc=f"Load {symbol}", yaml_cfg=yaml_cfg or {}, cli_disable=cli_disable)
    for m in months_iter:
        fpath = symdir / f"{symbol}-ticks-{m}.csv"
        if fpath.exists():
            # Robust CSV load: tolerate truncated/bad rows and weird quoting.
            # Expect columns: timestamp, price, qty, is_buyer_maker
            try:
                df = pd.read_csv(
                    fpath,
                    engine="python",
                    on_bad_lines="skip",                 # pandas >= 1.3
                    usecols=["timestamp", "price", "qty", "is_buyer_maker"],
                )
            except TypeError:
                # Fallback for older pandas which uses error_bad_lines
                df = pd.read_csv(
                    fpath,
                    engine="python",
                    error_bad_lines=False,              # deprecated in newer pandas
                    usecols=["timestamp", "price", "qty", "is_buyer_maker"],
                )

            # Hygiene: coerce dtypes and timestamps; drop malformed rows.
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["qty"]   = pd.to_numeric(df["qty"], errors="coerce")
            # is_buyer_maker is 0/1; keep it small but safe
            df["is_buyer_maker"] = pd.to_numeric(df["is_buyer_maker"], errors="coerce").fillna(0).astype("int8")
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp", "price"]).sort_values("timestamp").reset_index(drop=True)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No tick CSVs found under {symdir} for months={months}")
    df = pd.concat(dfs, ignore_index=True)
    # Already coerced to datetime and cleaned; ensure final global sort
    df = df.sort_values("timestamp").reset_index(drop=True)
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
