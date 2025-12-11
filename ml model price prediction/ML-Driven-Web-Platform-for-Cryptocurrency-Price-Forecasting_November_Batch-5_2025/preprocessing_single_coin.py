"""
preprocessing_single_coin.py

Single-coin preprocessing script that produces FULL OHLCV per day and multivariate
sliding-window sequences (Open, High, Low, Close, Volume) suitable for training
separate models per coin.

Features:
- Primary data source: CoinGecko `/market_chart/range` (intraday points) -> compute daily OHLCV
- Fallback: Binance klines (already OHLCV)
- Cleaning: dedupe, ffill/bfill
- Feature engineering: log_price, returns, moving averages
- Scaling: MinMax per column, saved
- Sequences: multivariate windows of shape (samples, window, features)
- Train/test split: time-based (chronological)

Usage:
    python preprocessing_single_coin.py --coin bitcoin --start 2018-01-01 --end 2025-11-18
    python preprocessing_single_coin.py --run-tests

Outputs (per coin in `output/<coin>/`):
- <coin>_raw.csv  (timestamp, Open, High, Low, Close, Volume, market_cap)
- <coin>_clean.csv
- <coin>_features.csv
- <coin>_scaled.csv
- <coin>_scalers.joblib
- <coin>_X_train.npy, <coin>_y_train.npy, <coin>_X_test.npy, <coin>_y_test.npy
- <coin>_model.joblib

"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ----------------- config -----------------
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
MAX_PUBLIC_DAYS = 365
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

SYMBOL_TO_BINANCE = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "solana": "SOLUSDT",
    "binancecoin": "BNBUSDT",
    "ripple": "XRPUSDT",
}

# ----------------- helpers -----------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def requests_session(retries: int = 3, backoff_factor: float = 0.3):
    s = requests.Session()
    from requests.adapters import HTTPAdapter, Retry
    retry = Retry(total=retries, backoff_factor=backoff_factor, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _to_coingecko_id(s: str) -> str:
    return s.strip().lower()


def _unix_ts(dt: datetime) -> int:
    return int(dt.timestamp())


def _trim_start_if_needed(start: datetime, allow_trim: bool = True) -> Tuple[datetime, bool]:
    today_utc = datetime.utcnow()
    earliest_allowed = today_utc - timedelta(days=MAX_PUBLIC_DAYS)
    if start < earliest_allowed:
        if allow_trim:
            return earliest_allowed, True
        raise RuntimeError(f"Requested start {start.date()} is older than {MAX_PUBLIC_DAYS} days and strict mode is set.")
    return start, False


# ----------------- fetchers -----------------

def fetch_coingecko_range_ohlcv(coin_id: str, start: str, end: str, vs_currency: str = "usd", allow_trim: bool = True) -> Optional[pd.DataFrame]:
    """Fetch intraday points and compute daily OHLCV DataFrame with columns Open,High,Low,Close,Volume,market_cap."""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    start_dt_trimmed, was_trimmed = _trim_start_if_needed(start_dt, allow_trim=allow_trim)
    if was_trimmed:
        print(f"Trimming start from {start_dt.date()} to {start_dt_trimmed.date()} due to CoinGecko public limits")

    params = {"vs_currency": vs_currency, "from": _unix_ts(start_dt_trimmed), "to": _unix_ts(end_dt + pd.Timedelta(days=1))}
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart/range"

    session = requests_session()
    api_key = os.environ.get("COINGECKO_API_KEY")
    headers = {"x-cg-pro-api-key": api_key} if api_key else None

    try:
        r = session.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 401:
            print("CoinGecko returned 401 — unauthorized")
            return None
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print("CoinGecko fetch error:", e)
        return None

    prices = data.get("prices", [])
    vols = data.get("total_volumes", [])
    mcaps = data.get("market_caps", [])

    if not prices:
        print("CoinGecko returned no prices for this range.")
        return None

    # Build temporary DataFrame of intraday points
    dfp = pd.DataFrame(prices, columns=["ts_ms", "price"]) 
    dfp["ts"] = pd.to_datetime(dfp["ts_ms"], unit="ms", utc=True)
    df_temp = dfp.set_index("ts")["price"].to_frame()

    if vols:
        dfv = pd.DataFrame(vols, columns=["ts_ms", "volume"]) 
        dfv["ts"] = pd.to_datetime(dfv["ts_ms"], unit="ms", utc=True)
        df_temp = df_temp.join(dfv.set_index("ts")["volume"], how="left")
    else:
        df_temp["volume"] = np.nan

    if mcaps:
        dfm = pd.DataFrame(mcaps, columns=["ts_ms", "market_cap"]) 
        dfm["ts"] = pd.to_datetime(dfm["ts_ms"], unit="ms", utc=True)
        df_temp = df_temp.join(dfm.set_index("ts")["market_cap"], how="left")
    else:
        df_temp["market_cap"] = np.nan

    # Resample to daily OHLC and aggregate volume
    ohlc = df_temp["price"].resample("D").agg(["first", "max", "min", "last"])  # open,max,min,close
    ohlc.columns = ["Open", "High", "Low", "Close"]
    vol = df_temp["volume"].resample("D").sum().rename("Volume")
    mcap = df_temp["market_cap"].resample("D").last().rename("market_cap")

    out = ohlc.join(vol).join(mcap)
    out = out.dropna(subset=["Close"])  # remove empty days
    out.index = out.index.tz_convert(None)

    # ensure columns exist
    for c in FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    out = out[["Open", "High", "Low", "Close", "Volume", "market_cap"]]
    return out


def fetch_binance_klines_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch daily klines from Binance and return DataFrame with Open,High,Low,Close,Volume,market_cap"""
    base = "https://api.binance.com/api/v3/klines"
    interval = "1d"
    limit = 1000
    rows = []
    start_time = int(pd.Timestamp("2014-01-01", tz="UTC").timestamp() * 1000)
    session = requests_session()

    while True:
        params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": start_time}
        try:
            r = session.get(base, params=params, timeout=30)
            r.raise_for_status()
            chunk = r.json()
        except Exception as e:
            print("Binance fetch error:", e)
            break

        if not chunk:
            break

        for k in chunk:
            ts = int(k[0])
            open_p = float(k[1])
            high_p = float(k[2])
            low_p = float(k[3])
            close_p = float(k[4])
            vol = float(k[5])
            rows.append((ts, open_p, high_p, low_p, close_p, vol))

        last_ts = int(chunk[-1][0])
        start_time = last_ts + 1
        if len(chunk) < limit:
            break
        time.sleep(0.05)

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["ts_ms", "Open", "High", "Low", "Close", "Volume"]) 
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("ts")[ ["Open", "High", "Low", "Close", "Volume"] ]
    df.index = df.index.tz_convert(None)
    df["market_cap"] = np.nan
    # resample safety
    df_daily = df.resample("D").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum","market_cap":"last"})
    df_daily = df_daily.dropna(subset=["Close"]) 
    return df_daily[["Open","High","Low","Close","Volume","market_cap"]]


# ----------------- preprocessing -----------------

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Index must be datetime")
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Close' not in df.columns:
        raise ValueError('Close column required')
    df['log_price'] = np.log(df['Close'].replace(0, np.nan)).fillna(method='ffill').fillna(method='bfill')
    df['return_1d'] = df['Close'].pct_change().fillna(0)
    df['ma_7'] = df['Close'].rolling(7, min_periods=1).mean()
    df['ma_21'] = df['Close'].rolling(21, min_periods=1).mean()
    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df


def scale_features_columnwise(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scalers = {}
    for c in numeric_cols:
        s = MinMaxScaler()
        df[[c]] = s.fit_transform(df[[c]])
        scalers[c] = s
    return df, scalers


def create_multivariate_sequences(df: pd.DataFrame, window: int=60) -> Tuple[np.ndarray, np.ndarray]:
    """Create multivariate sequences using FEATURE_COLUMNS order.
    Returns X shape (samples, window, features) and y shape (samples,)
    where y is next-day Close (unscaled)."""
    if any(col not in df.columns for col in FEATURE_COLUMNS):
        raise ValueError(f"DataFrame must contain columns: {FEATURE_COLUMNS}")
    arr = df[FEATURE_COLUMNS].values
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i-window:i])
        y.append(arr[i, FEATURE_COLUMNS.index('Close')])
    if not X:
        return np.empty((0, window, len(FEATURE_COLUMNS))), np.empty((0,))
    return np.array(X), np.array(y)


def time_split(X: np.ndarray, y: np.ndarray, test_frac: float=0.2):
    n = X.shape[0]
    if n == 0:
        return X, y, X, y
    split = int(np.ceil(n * (1 - test_frac)))
    return X[:split], y[:split], X[split:], y[split:]


# ----------------- model utilities -----------------

def build_model(model_type: str='mlp'):
    if model_type == 'mlp':
        return MLPRegressor(hidden_layer_sizes=(128,64), max_iter=300)
    elif model_type == 'random_forest':
        return RandomForestRegressor(n_estimators=200)
    else:
        raise ValueError('unknown model')


# ----------------- pipeline -----------------

def preprocess_and_train(coin: str, start: str, end: str, window: int, out_dir: str, use_binance_fallback: bool=True, force_save_empty: bool=False, model_type: str='mlp', allow_trim: bool=True):
    coin_slug = _to_coingecko_id(coin)
    outp = Path(out_dir) / coin_slug
    ensure_dir(outp)

    print(f"Processing {coin_slug} -> {outp}")

    df = fetch_coingecko_range_ohlcv(coin_slug, start, end, allow_trim=allow_trim)
    if (df is None or len(df) == 0) and use_binance_fallback:
        sym = SYMBOL_TO_BINANCE.get(coin_slug)
        if sym:
            print("Falling back to Binance...")
            df = fetch_binance_klines_ohlcv(sym)

    if df is None or len(df) == 0:
        print("No data fetched for coin.")
        if force_save_empty:
            placeholder = pd.DataFrame(columns=['timestamp']+FEATURE_COLUMNS+['market_cap'])
            placeholder.to_csv(outp / f"{coin_slug}_raw.csv", index=False)
            print(f"Wrote placeholder to {outp / f'{coin_slug}_raw.csv'}")
        return

    # ensure proper column names and order
    df_out = df.reset_index().rename(columns={'index':'timestamp'})
    # rename columns if lowercase
    df_out.columns = [c if c in ['timestamp']+FEATURE_COLUMNS+['market_cap'] else c.capitalize() for c in df_out.columns]
    # reorder
    cols_needed = ['timestamp'] + FEATURE_COLUMNS + ['market_cap']
    df_out = df_out.loc[:, [c for c in cols_needed if c in df_out.columns]]

    raw_csv = outp / f"{coin_slug}_raw.csv"
    df_out.to_csv(raw_csv, index=False)
    print(f"Saved raw CSV -> {raw_csv}")

    # cleaning
    df_clean = basic_cleaning(df)
    clean_csv = outp / f"{coin_slug}_clean.csv"
    df_clean.reset_index().rename(columns={'index':'timestamp'}).to_csv(clean_csv, index=False)
    print(f"Saved clean CSV -> {clean_csv}")

    # features
    df_feat = add_features(df_clean)
    feat_csv = outp / f"{coin_slug}_features.csv"
    df_feat.reset_index().rename(columns={'index':'timestamp'}).to_csv(feat_csv, index=False)
    print(f"Saved features CSV -> {feat_csv}")

    # scaling (fit on numeric columns including OHLCV + engineered numeric features)
    df_scaled, scalers = scale_features_columnwise(df_feat)
    scaler_path = outp / f"{coin_slug}_scalers.joblib"
    joblib.dump(scalers, scaler_path)
    scaled_csv = outp / f"{coin_slug}_scaled.csv"
    df_scaled.reset_index().rename(columns={'index':'timestamp'}).to_csv(scaled_csv, index=False)
    print(f"Saved scaled CSV -> {scaled_csv} and scalers -> {scaler_path}")

    # create multivariate sequences using OHLCV features
    X, y = create_multivariate_sequences(df_scaled, window=window)
    X_train, y_train, X_test, y_test = time_split(X, y, test_frac=0.2)

    np.save(outp / f"{coin_slug}_X_train.npy", X_train)
    np.save(outp / f"{coin_slug}_y_train.npy", y_train)
    np.save(outp / f"{coin_slug}_X_test.npy", X_test)
    np.save(outp / f"{coin_slug}_y_test.npy", y_test)
    print(f"Saved sequences -> samples={X.shape[0]}, train={X_train.shape[0]}, test={X_test.shape[0]}")

    # train (using sklearn that expects 2D input) -> flatten windows
    if X_train.shape[0] == 0:
        print("Not enough training samples to train model.")
        return

    n_train = X_train.shape[0]
    n_feat = X_train.shape[1] * X_train.shape[2]
    X_train_flat = X_train.reshape(n_train, n_feat)
    X_test_flat = X_test.reshape(X_test.shape[0], n_feat) if X_test.shape[0] > 0 else X_train_flat

    model = build_model(model_type)
    print(f"Training {model_type} model on {X_train_flat.shape[0]} samples...")
    model.fit(X_train_flat, y_train)

    y_pred = model.predict(X_test_flat)
    mse = mean_squared_error(y_test, y_pred) if y_test.shape[0] > 0 else float('nan')

    model_path = outp / f"{coin_slug}_model.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model -> {model_path}. Test MSE={mse}")


# ----------------- tests -----------------

def run_tests():
    print("Running tests...")
    # small synthetic OHLCV dataset
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=100, freq='D')
    arr = np.linspace(100, 200, 100)
    df = pd.DataFrame({"Open": arr-1, "High": arr+2, "Low": arr-2, "Close": arr, "Volume": np.arange(100)}, index=idx)
    df["market_cap"] = np.nan
    c = basic_cleaning(df)
    f = add_features(c)
    s, scalers = scale_features_columnwise(f)
    X, y = create_multivariate_sequences(s, window=5)
    assert X.shape[0] == 95
    print("All tests passed.")


# ----------------- CLI -----------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--coins', nargs='+', default=['bitcoin','ethereum','solana','binancecoin','ripple'])
    p.add_argument('--start', default='2018-01-01')
    p.add_argument('--end', default=str(pd.Timestamp.utcnow().date()))
    p.add_argument('--window', type=int, default=60)
    p.add_argument('--out-dir', default='output')
    p.add_argument('--no-fallback', dest='no_fallback', action='store_true')
    p.add_argument('--force-save-empty', dest='force_save_empty', action='store_true')
    p.add_argument('--model', choices=['mlp','random_forest'], default='mlp')
    p.add_argument('--run-tests', action='store_true')
    args = p.parse_args()

    if args.run_tests:
        run_tests()
    else:
        for coin in args.coins:
            preprocess_and_train(coin, args.start, args.end, args.window, args.out_dir, use_binance_fallback=not args.no_fallback, force_save_empty=args.force_save_empty, model_type=args.model)
