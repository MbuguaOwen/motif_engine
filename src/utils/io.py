
from pathlib import Path
import logging
import pandas as pd
from src.ui.progress import wrap_iter, bar
import zipfile

# --- NEW: read 1m klines from inputs/ ---


def _read_single_kline_csv(path: Path) -> pd.DataFrame:
    """Read a single Binance kline CSV (possibly inside a .zip).
    Use first 6 columns and return standardized bars DF with inferred time unit.
    Columns (Binance): open_time, open, high, low, close, volume, close_time, ...
    """
    names   = ["open_time","open","high","low","close","volume",
               "close_time","qav","trades","tb_base","tb_quote","ignore"]
    usecols = ["open_time","open","high","low","close","volume"]
    dtypes  = {"open_time": "int64", "open": "float64", "high": "float64",
               "low": "float64", "close": "float64", "volume": "float64"}

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            inner = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not inner:
                raise FileNotFoundError(f"No CSV inside {path}")
            with z.open(inner[0]) as f:
                df = pd.read_csv(f, header=None, names=names, usecols=usecols, dtype=dtypes)
    else:
        df = pd.read_csv(path, header=None, names=names, usecols=usecols, dtype=dtypes)

    # --- Infer time unit from magnitude ---
    # s  ~ 1e9      (e.g., 1735689600)
    # ms ~ 1e12     (e.g., 1735689600000)
    # us ~ 1e15     (e.g., 1735689600000000)  <-- your file
    # ns ~ 1e18     (rare in public Binance dumps)
    ts0 = int(df["open_time"].iloc[0])
    if   ts0 > 1_000_000_000_000_000_000: unit = "ns"
    elif ts0 >     1_000_000_000_000:     unit = "us"
    elif ts0 >         1_000_000_000:     unit = "ms"
    else:                                 unit = "s"

    logging.getLogger(__name__).info("Parsed %s as %s timestamps", path.name, unit)

    df["timestamp"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    df = df.drop(columns=["open_time"])
    df = df.dropna(subset=["timestamp","open","high","low","close"]).drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp","open","high","low","close","volume"]]


def read_kline_1m_months(
    inputs_dir: str, symbol: str, months: list, yaml_cfg: dict = None, cli_disable: bool = False
) -> pd.DataFrame:
    """
    Load 1m klines for given months from inputs/ with flexible paths:
      inputs/<SYMBOL>/<SYMBOL>-1m-YYYY-MM.csv[.zip]
      inputs/<SYMBOL>/1m/<SYMBOL>-1m-YYYY-MM.csv[.zip]
      inputs/<SYMBOL>-1m-YYYY-MM.csv[.zip]
    Concatenate, sort and return standardized bars.
    """
    base = Path(inputs_dir)
    dfs = []
    for m in months:
        candidates = [
            base / symbol / f"{symbol}-1m-{m}.csv",
            base / symbol / "1m" / f"{symbol}-1m-{m}.csv",
            base / f"{symbol}-1m-{m}.csv",
            base / symbol / f"{symbol}-1m-{m}.zip",
            base / symbol / "1m" / f"{symbol}-1m-{m}.zip",
            base / f"{symbol}-1m-{m}.zip",
        ]
        for p in candidates:
            if p.exists():
                dfs.append(_read_single_kline_csv(p))
                break
        else:
            # Skips silently if a month is missing; you can raise if preferred
            pass
    if len(dfs) == 0:
        raise FileNotFoundError(
            f"No 1m kline files found for {symbol} months={months} under {inputs_dir}"
        )
    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out

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
