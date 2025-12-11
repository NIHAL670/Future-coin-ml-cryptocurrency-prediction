
"""
train_all.py - Milestone 2

Train one ML model per coin using preprocessed outputs from preprocessing_single_coin.py.

Reads per-coin:
  output/<coin>/<coin>_features.csv

Produces per-coin:
  models/<coin>/<coin>_model.<joblib|h5>
  models/<coin>/<coin>_scaler.joblib
  models/<coin>/<coin>_metrics.json
  models/<coin>/<coin>_predictions.csv

Also writes models/summary.csv.

Supports models: random_forest, mlp, xgboost (if installed), linear_regression, lstm (if TF installed).
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Tuple, List, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# -------------------------
# Feature engineering utils
# -------------------------
def add_more_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features: extra MAs, rolling std, vol MA, returns, calendar."""
    df = df.copy()
    if "Close" not in df.columns:
        raise ValueError("Close column required in features CSV")

    # moving averages
    df["ma_7"] = df["Close"].rolling(7, min_periods=1).mean()
    df["ma_14"] = df["Close"].rolling(14, min_periods=1).mean()
    df["ma_21"] = df["Close"].rolling(21, min_periods=1).mean()
    df["ma_50"] = df["Close"].rolling(50, min_periods=1).mean()

    # rolling std / volatility
    df["std_7"] = df["Close"].rolling(7, min_periods=1).std().fillna(0)
    df["std_21"] = df["Close"].rolling(21, min_periods=1).std().fillna(0)

    # returns
    df["return_1d"] = df["Close"].pct_change().fillna(0)
    df["return_7d"] = df["Close"].pct_change(7).fillna(0)

    # volume features
    if "Volume" in df.columns:
        df["vol_ma_7"] = df["Volume"].rolling(7, min_periods=1).mean()
        df["vol_ma_21"] = df["Volume"].rolling(21, min_periods=1).mean()
        df["vol_to_ma7"] = df["Volume"] / (df["vol_ma_7"].replace(0, np.nan))
    else:
        df["Volume"] = 0
        df["vol_ma_7"] = 0
        df["vol_ma_21"] = 0
        df["vol_to_ma7"] = 0

    # calendar features
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month

    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)
    return df

# -------------------------
# Sequence creation
# -------------------------
def create_sequences_from_df(df: pd.DataFrame, feature_cols: List[str], window: int=60) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Create multivariate sequences (samples, window, features) and target y = next-day Close (unscaled).
    Returns X, y, and list of timestamps aligned to y (the date of the predicted Close).
    """
    arr = df[feature_cols].values
    closes = df["Close"].values
    X, y = [], []
    timestamps = []
    for i in range(window, len(arr)):
        X.append(arr[i-window:i])
        y.append(closes[i])     # next-day Close == current row's Close after window
        timestamps.append(df.index[i])
    if not X:
        return np.empty((0, window, len(feature_cols))), np.empty((0,)), []
    return np.array(X), np.array(y), timestamps

# -------------------------
# Scaling helpers
# -------------------------
def fit_scaler_and_transform(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    df_scaled = pd.DataFrame(scaled, index=df.index, columns=numeric_cols)
    # keep non-numeric columns if any
    for c in df.columns:
        if c not in numeric_cols:
            df_scaled[c] = df[c]
    return df_scaled, scaler

def transform_with_scaler(df: pd.DataFrame, numeric_cols: List[str], scaler) -> pd.DataFrame:
    scaled = scaler.transform(df[numeric_cols])
    df_scaled = pd.DataFrame(scaled, index=df.index, columns=numeric_cols)
    for c in df.columns:
        if c not in numeric_cols:
            df_scaled[c] = df[c]
    return df_scaled

# -------------------------
# Model training helpers
# -------------------------
def flatten_for_sklearn(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        return X.reshape(X.shape[0], -1)
    return X

def train_random_forest(X, y, n_iter=20, n_jobs=1, random_state=42):
    param_dist = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4]
    }
    model = RandomForestRegressor(random_state=random_state)
    tscv = TimeSeriesSplit(n_splits=3)
    rs = RandomizedSearchCV(model, param_dist, n_iter=min(n_iter, 12), cv=tscv, n_jobs=n_jobs, scoring='neg_mean_squared_error', random_state=random_state)
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_

def train_mlp(X, y, n_iter=10, n_jobs=1, random_state=42):
    param_dist = {
        "hidden_layer_sizes": [(64,32), (128,64), (64,)],
        "alpha": [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 1e-4]
    }
    model = MLPRegressor(max_iter=500, random_state=random_state)
    tscv = TimeSeriesSplit(n_splits=3)
    rs = RandomizedSearchCV(model, param_dist, n_iter=min(n_iter, 10), cv=tscv, n_jobs=n_jobs, scoring='neg_mean_squared_error', random_state=random_state)
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_

def train_linear(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model, {}

def train_xgboost(X, y, n_iter=12, n_jobs=1, random_state=42):
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("xgboost not installed")
    param_dist = {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1]
    }
    model = xgb.XGBRegressor(random_state=random_state, verbosity=0, n_jobs=n_jobs)
    tscv = TimeSeriesSplit(n_splits=3)
    rs = RandomizedSearchCV(model, param_dist, n_iter=min(n_iter, 12), cv=tscv, n_jobs=n_jobs, scoring='neg_mean_squared_error', random_state=random_state)
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_

def build_lstm(input_shape, units=(64,32), dropout=0.0):
    model = Sequential()
    model.add(LSTM(units[0], return_sequences=len(units)>1, input_shape=input_shape))
    if len(units) > 1:
        model.add(LSTM(units[1]))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(X, y, epochs=50, batch_size=32, units=(64,32), patience=5):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not installed for LSTM")
    # simple train/val split
    val_split = int(0.8 * X.shape[0])
    X_tr, y_tr = X[:val_split], y[:val_split]
    X_val, y_val = X[val_split:], y[val_split:]
    model = build_lstm((X.shape[1], X.shape[2]), units=units)
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
    return model, {"units": units, "epochs": epochs, "batch_size": batch_size}

# -------------------------
# Evaluation & saving
# -------------------------
def evaluate_and_save_predictions(model, model_type: str, X_test, y_test, timestamps, out_dir: Path, coin: str):
    if X_test.shape[0] == 0:
        metrics = {"mse": None, "mae": None, "rmse": None}
        return metrics
    if model_type == "lstm":
        y_pred = model.predict(X_test).ravel()
    else:
        if X_test.ndim == 3:
            X_test_flat = flatten_for_sklearn(X_test)
        else:
            X_test_flat = X_test
        y_pred = model.predict(X_test_flat)
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(math.sqrt(mse))
    metrics = {"mse": mse, "mae": mae, "rmse": rmse}
    # Save predictions CSV
    dfp = pd.DataFrame({"timestamp": timestamps, "y_true": y_test, "y_pred": y_pred})
    dfp.to_csv(out_dir / f"{coin}_predictions_{model_type}.csv", index=False)
    # Save metrics JSON
    with open(out_dir / f"{coin}_metrics_{model_type}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

# -------------------------
# Main training loop
# -------------------------
def process_coin(coin: str,
                 out_features_dir: Path,
                 models_dir: Path,
                 window: int,
                 model_type: str,
                 n_iter: int,
                 n_jobs: int,
                 lstm_epochs: int,
                 lstm_batch: int):
    coin_slug = coin.lower()
    print(f"\n=== Processing {coin_slug} ===")
    feat_path = out_features_dir / coin_slug / f"{coin_slug}_features.csv"
    if not feat_path.exists():
        print(f"Missing features file for {coin_slug}: {feat_path} - skipping")
        return None

    df = pd.read_csv(feat_path, parse_dates=["ts"], index_col="ts")
    # Add more features
    df = add_more_features(df)

    # Choose feature columns to use for sequences (OHLCV + engineered numeric features)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # prefer a subset order
    preferred = ["Open","High","Low","Close","Volume","log_price","return_1d","ma_7","ma_14","ma_21","ma_50","std_7","std_21","vol_ma_7","vol_ma_21","vol_to_ma7"]
    feature_cols = [c for c in preferred if c in df.columns]
    # append any other numeric cols not included
    for c in numeric_cols:
        if c not in feature_cols:
            feature_cols.append(c)

    # Fit scaler on numeric columns (store per coin)
    scaler_cols = feature_cols.copy()
    df_num = df[scaler_cols]
    scaler = MinMaxScaler()
    df_scaled_vals = scaler.fit_transform(df_num)
    df_scaled = pd.DataFrame(df_scaled_vals, index=df.index, columns=scaler_cols)

    # Save scaler
    (models_dir / coin_slug).mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / coin_slug / f"{coin_slug}_scaler.joblib")

    # Create sequences and targets
    X, y, timestamps = create_sequences_from_df(df_scaled, feature_cols, window=window)
    if X.shape[0] == 0:
        print(f"Not enough data to create sequences for {coin_slug} (need > window rows).")
        return None

    # Split into train/test chronological
    split = int(np.ceil(X.shape[0] * 0.8))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    timestamps_test = timestamps[split:]

    print(f"{coin_slug}: samples={X.shape[0]}, train={X_train.shape[0]}, test={X_test.shape[0]}, features={len(feature_cols)}")

    # Train model
    if model_type in ("random_forest", "mlp", "linear_regression", "xgboost"):
        X_train_flat = flatten_for_sklearn(X_train)
        X_test_flat = flatten_for_sklearn(X_test) if X_test.shape[0] > 0 else X_train_flat

        if model_type == "random_forest":
            model, meta = train_random_forest(X_train_flat, y_train, n_iter=n_iter, n_jobs=n_jobs)
        elif model_type == "mlp":
            model, meta = train_mlp(X_train_flat, y_train, n_iter=min(n_iter, 10), n_jobs=n_jobs)
        elif model_type == "linear_regression":
            model, meta = train_linear(X_train_flat, y_train)
        elif model_type == "xgboost":
            model, meta = train_xgboost(X_train_flat, y_train, n_iter=min(n_iter, 12), n_jobs=n_jobs)
        else:
            raise ValueError("unknown sklearn model_type")
        # Save model
        joblib.dump(model, models_dir / coin_slug / f"{coin_slug}_model_{model_type}.joblib")
        # Evaluate
        metrics = evaluate_and_save_predictions(model, model_type, X_test_flat, y_test, timestamps_test, models_dir / coin_slug, coin_slug)
    elif model_type == "lstm":
        if not TF_AVAILABLE:
            print("TensorFlow not installed — skipping LSTM for", coin_slug)
            return None
        model, meta = train_lstm(X_train, y_train, epochs=lstm_epochs, batch_size=lstm_batch)
        model_path = models_dir / coin_slug / f"{coin_slug}_model_lstm.h5"
        model.save(model_path)
        metrics = evaluate_and_save_predictions(model, "lstm", X_test, y_test, timestamps_test, models_dir / coin_slug, coin_slug)
    else:
        raise ValueError("Unsupported model_type")

    # Save meta + metrics summary
    with open(models_dir / coin_slug / f"{coin_slug}_meta_{model_type}.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "meta": meta, "model_type": model_type}, f, indent=2)

    return {"coin": coin_slug, "model": model_type, **metrics}

# -------------------------
# CLI main
# -------------------------
def main(coins: List[str], out_dir: str, models_dir: str, window: int, model_type: str, n_iter: int, n_jobs: int, lstm_epochs: int, lstm_batch: int):
    out_features_dir = Path(out_dir)
    models_dir_p = Path(models_dir)
    models_dir_p.mkdir(parents=True, exist_ok=True)
    summary = []
    for coin in coins:
        try:
            res = process_coin(coin, out_features_dir, models_dir_p, window, model_type, n_iter, n_jobs, lstm_epochs, lstm_batch)
            if res:
                summary.append(res)
        except Exception as e:
            print(f"Error processing {coin}: {e}")
    if summary:
        df_summary = pd.DataFrame(summary)
        df_summary.to_csv(models_dir_p / "summary.csv", index=False)
        print(f"Saved summary -> {models_dir_p / 'summary.csv'}")
    else:
        print("No models trained / nothing to summarize.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=["bitcoin","ethereum","solana","binancecoin","ripple"])
    parser.add_argument("--out-dir", default="ML-Driven-Web-Platform-for-Cryptocurrency-Price-Forecasting_November_Batch-5_2025/output", help="where preprocessing outputs are (output/<coin>/... )")
    parser.add_argument("--models-dir", default="ML-Driven-Web-Platform-for-Cryptocurrency-Price-Forecasting_November_Batch-5_2025/models", help="where to save trained models")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--model", choices=["random_forest","mlp","xgboost","linear_regression","lstm"], default="random_forest")
    parser.add_argument("--n-iter", type=int, default=30, help="RandomizedSearch iterations")
    parser.add_argument("--n-jobs", type=int, default=1, help="parallel jobs for search")
    parser.add_argument("--lstm-epochs", type=int, default=50)
    parser.add_argument("--lstm-batch", type=int, default=32)
    args = parser.parse_args()

    main(coins=args.coins, out_dir=args.out_dir, models_dir=args.models_dir, window=args.window, model_type=args.model, n_iter=args.n_iter, n_jobs=args.n_jobs, lstm_epochs=args.lstm_epochs, lstm_batch=args.lstm_batch)
