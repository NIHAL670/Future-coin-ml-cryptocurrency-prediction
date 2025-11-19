"""
Milestone 1 — Data Collection & Preprocessing (FINAL FIX)
File: Milestone1_Preprocessing.py

This is the corrected, syntax-error-free version of the preprocessing script with multi-coin support,
CoinGecko retry + API-key header support, Binance fallback, and offline unit tests.

Usage examples:
    # single coin
    python Milestone1_Preprocessing.py --coin bitcoin

    # multiple coins (batch)
    python Milestone1_Preprocessing.py --coins bitcoin ethereum solana cardano ripple

Optional env var for CoinGecko Pro API key:
    export COINGECKO_API_KEY=your_key_here

Requirements:
    pip install pandas requests scikit-learn pyarrow

"""

import os
import argparse
import requests
import pandas as pd
import numpy as np
import pickle
import time
from typing import Optional, Tuple
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Helpers
# -----------------------------

def ensure_dirs():
    for d in ("raw", "processed", "artifacts", "notebooks"):
        os.makedirs(d, exist_ok=True)


def _safe_save_df(df: pd.DataFrame, csv_path: str, parquet_path: str):
    """Save CSV and try parquet; if parquet fails, warn but continue."""
    df.to_csv(csv_path)
    try:
        df.to_parquet(parquet_path)
    except Exception as e:
        print(f"Warning: could not save parquet to {parquet_path} ({e}), CSV saved to {csv_path}.")


# -----------------------------
# Remote fetchers
# -----------------------------

def fetch_coingecko_market_chart_with_retries(
    coin_id: str = "bitcoin",
    vs_currency: str = "usd",
    max_retries: int = 5,
    backoff_factor: float = 1.0,
) -> Optional[pd.DataFrame]:
    """Fetch 'max' historical market data from CoinGecko with retries.

    Supports optional API key via environment variable COINGECKO_API_KEY (header: x-cg-pro-api-key).
    Returns a daily-resampled DataFrame (price, market_cap, volume) or None on persistent failure.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": "max"}

    headers = {}
    api_key = os.environ.get("COINGECKO_API_KEY")
    if api_key:
        headers["x-cg-pro-api-key"] = api_key
        print("Using CoinGecko API key from COINGECKO_API_KEY environment variable.")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[CoinGecko] attempt {attempt}/{max_retries} ...")
            r = requests.get(url, params=params, headers=headers or None, timeout=30)
            if r.status_code == 401:
                print("[CoinGecko] 401 Unauthorized — check COINGECKO_API_KEY or your access.")
                return None
            r.raise_for_status()
            data = r.json()

            prices = data.get("prices", [])
            market_caps = data.get("market_caps", [])
            total_volumes = data.get("total_volumes", [])

            if not prices:
                print("[CoinGecko] No price data returned by API.")
                return None

            df = pd.DataFrame({
                "timestamp": [int(p[0]) for p in prices],
                "price": [float(p[1]) for p in prices],
                "market_cap": [float(m[1]) for m in market_caps] if market_caps else [np.nan] * len(prices),
                "volume": [float(v[1]) for v in total_volumes] if total_volumes else [np.nan] * len(prices),
            })

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")

            # Resample to daily end-of-day
            df_daily = df.resample("D").agg({
                "price": "last",
                "market_cap": "last",
                "volume": "sum",
            })
            print(f"[CoinGecko] success — fetched {len(df_daily)} daily rows.")
            return df_daily

        except requests.RequestException as e:
            print(f"[CoinGecko] RequestException: {e}")
        except ValueError as e:
            print(f"[CoinGecko] JSON decode or parsing error: {e}")

        # exponential backoff before retrying
        sleep_time = min(backoff_factor * (2 ** (attempt - 1)), 60)
        print(f"[CoinGecko] retrying in {sleep_time:.0f}s ...")
        time.sleep(sleep_time)

    print("[CoinGecko] all retries failed — returning None for fallback.")
    return None


def fetch_binance_klines_daily(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Fallback: get daily candles from Binance REST API.

    Returns a DataFrame indexed by UTC date with columns: price, volume, market_cap (NaN).
    """
    base = "https://api.binance.com/api/v3/klines"
    interval = "1d"
    limit = 1000  # max per request
    all_rows = []

    # Start from a reasonably old date
    start_time = int(pd.Timestamp("2014-01-01", tz="UTC").timestamp() * 1000)

    while True:
        params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": start_time}
        try:
            print(f"[Binance] fetching startTime={start_time} ...")
            r = requests.get(base, params=params, timeout=30)
            r.raise_for_status()
            chunk = r.json()
        except requests.RequestException as e:
            print(f"[Binance] request failed: {e}")
            break

        if not chunk:
            break

        for k in chunk:
            ts = int(k[0])
            close_price = float(k[4])
            vol = float(k[5])
            all_rows.append((ts, close_price, vol))

        last_ts = int(chunk[-1][0])
        start_time = last_ts + 1

        if len(chunk) < limit:
            break

        if len(all_rows) > 20000:
            print("[Binance] safety break - too many rows")
            break

        time.sleep(0.2)

    df = pd.DataFrame(all_rows, columns=["timestamp_ms", "price", "volume"]).drop_duplicates()
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp")[ ["price", "volume"] ]
    df["market_cap"] = np.nan
    df_daily = df.resample("D").agg({"price": "last", "market_cap": "last", "volume": "sum"})
    print(f"[Binance] fetched {len(df_daily)} daily rows.")
    return df_daily


# -----------------------------
# Preprocessing utilities
# -----------------------------

def save_raw(df: pd.DataFrame, coin_id: str):
    path_csv = os.path.join("raw", f"{coin_id}_raw.csv")
    path_parquet = os.path.join("raw", f"{coin_id}_raw.parquet")
    _safe_save_df(df, path_csv, path_parquet)
    print(f"Saved raw data to {path_csv}")


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning pipeline:
    - Drop entirely NA rows
    - Remove duplicate timestamps (keep first)
    - Forward-fill then back-fill
    - Optional light smoothing for price
    """
    if df is None or len(df) == 0:
        return df

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be a DatetimeIndex for basic_cleaning")

    df = df.dropna(how="all")

    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]

    df = df.fillna(method="ffill").fillna(method="bfill")

    if "price" in df.columns and len(df) >= 7:
        df["price_smooth"] = df["price"].rolling(3, center=True, min_periods=1).mean()
        df["price"] = df["price_smooth"].fillna(df["price"])  # replace
        df.drop(columns=["price_smooth"], inplace=True)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple engineered features:
    - log_price, return_1d, ma_7, ma_21, dow, month
    """
    if df is None or len(df) == 0:
        return df

    df = df.copy()
    if "price" not in df.columns:
        raise ValueError("price column required in DataFrame for add_features")

    df["log_price"] = np.log(df["price"].replace(0, np.nan)).fillna(method="ffill").fillna(method="bfill")
    df["return_1d"] = df["price"].pct_change().fillna(0)

    df["ma_7"] = df["price"].rolling(7, min_periods=1).mean()
    df["ma_21"] = df["price"].rolling(21, min_periods=1).mean()

    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month

    df = df.fillna(method="ffill").fillna(method="bfill")
    return df


def scale_and_save(df: pd.DataFrame, coin_id: str, scaler_path: str = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found to scale")

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[numeric_cols])
    df_scaled = pd.DataFrame(scaled_values, index=df.index, columns=numeric_cols)

    for col in df.columns:
        if col not in numeric_cols:
            df_scaled[col] = df[col]

    processed_csv = os.path.join("processed", f"{coin_id}_scaled.csv")
    processed_parquet = os.path.join("processed", f"{coin_id}_scaled.parquet")
    _safe_save_df(df_scaled, processed_csv, processed_parquet)
    print(f"Saved scaled data to {processed_csv}")

    if scaler_path is None:
        scaler_path = os.path.join("artifacts", f"{coin_id}_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    return df_scaled, scaler


def time_split_save(df: pd.DataFrame, coin_id: str, train_frac: float = 0.7, val_frac: float = 0.15):
    n = len(df)
    if n == 0:
        raise ValueError("Empty DataFrame passed to time_split_save")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    paths = {}
    for name, piece in (("train", train), ("val", val), ("test", test)):
        csv_path = os.path.join("processed", f"{coin_id}_{name}.csv")
        parquet_path = os.path.join("processed", f"{coin_id}_{name}.parquet")
        _safe_save_df(piece, csv_path, parquet_path)
        paths[name] = (csv_path, parquet_path)
        print(f"Saved {name} -> {csv_path}")

    return paths


# -----------------------------
# Minimal offline unit tests (sanity checks)
# -----------------------------

def run_unit_tests():
    print("Running offline unit tests...")

    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({"price": [100, None, 105, None, 110, 115, None, 120, 125, None], "volume": [1, 2, None, 4, 5, 6, 7, None, 9, 10]}, index=idx)
    cleaned = basic_cleaning(df)
    assert cleaned.isna().sum().sum() == 0, "basic_cleaning should fill all NAs"

    feat = add_features(cleaned)
    for c in ("log_price", "return_1d", "ma_7", "ma_21", "dow", "month"):
        assert c in feat.columns, f"add_features should add column {c}"

    df_scaled, scaler = scale_and_save(feat, coin_id="testcoin")
    assert df_scaled.select_dtypes(include=[np.number]).max().max() <= 1.0 + 1e-6
    assert df_scaled.select_dtypes(include=[np.number]).min().min() >= -1e-6

    print("All offline unit tests passed.")


# -----------------------------
# Main flow (single coin function + multi-coin wrapper)
# -----------------------------

def main_single_coin(coin_id: str, use_fallback_if_needed: bool, run_tests_before: bool):
    if run_tests_before:
        try:
            run_unit_tests()
        except AssertionError as e:
            print(f"Unit tests failed: {e}")
            return

    df_raw = fetch_coingecko_market_chart_with_retries(coin_id=coin_id)

    if df_raw is None and use_fallback_if_needed:
        print("CoinGecko failed or unauthorized — attempting Binance fallback.")
        symbol_map = {
            "bitcoin": "BTCUSDT",
            "ethereum": "ETHUSDT",
            "cardano": "ADAUSDT",
            "ripple": "XRPUSDT",
            "solana": "SOLUSDT",
        }
        symbol = symbol_map.get(coin_id.lower(), "BTCUSDT")
        df_raw = fetch_binance_klines_daily(symbol=symbol)

    if df_raw is None or len(df_raw) == 0:
        print(f"Failed to fetch data for {coin_id}.")
        return

    save_raw(df_raw, coin_id)

    df_clean = basic_cleaning(df_raw)
    _safe_save_df(df_clean, os.path.join("processed", f"{coin_id}_clean.csv"), os.path.join("processed", f"{coin_id}_clean.parquet"))

    df_feat = add_features(df_clean)
    _safe_save_df(df_feat, os.path.join("processed", f"{coin_id}_features.csv"), os.path.join("processed", f"{coin_id}_features.parquet"))

    df_scaled, scaler = scale_and_save(df_feat, coin_id)
    time_split_save(df_scaled, coin_id)

    print(f"Finished processing {coin_id}\n")


def main(coin_id: str = "bitcoin", use_fallback_if_needed: bool = True, run_tests_before: bool = True, multi_coins: Optional[list] = None):
    ensure_dirs()

    if multi_coins:
        for c in multi_coins:
            print(f"\n==============================")
            print(f" Processing coin: {c}")
            print(f"==============================\n")
            main_single_coin(c, use_fallback_if_needed, run_tests_before=False)
        return

    # single coin
    main_single_coin(coin_id, use_fallback_if_needed, run_tests_before)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coin", default="bitcoin", help="coin id slug for single run")
    p.add_argument("--coins", nargs="*", help="list of multiple coins, e.g. --coins bitcoin ethereum solana")
    p.add_argument("--no-tests", dest="no_tests", action="store_true", help="Skip offline unit tests")
    p.add_argument("--no-fallback", dest="no_fallback", action="store_true", help="Do not attempt Binance fallback")
    args = p.parse_args()
    main(coin_id=args.coin, multi_coins=args.coins, use_fallback_if_needed=not args.no_fallback, run_tests_before=not args.no_tests)
