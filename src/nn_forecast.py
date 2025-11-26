"""
Neural (GRU) multi-horizon forecasting:
last 168h -> next 24h, per country.

Outputs:
    outputs/<CC>_forecasts_nn.csv
Format:
    timestamp, y_true, yhat, lo, hi, horizon, train_end, split

We use the same train/dev/test split as SARIMA:
    - first 80%: train
    - next 10%: dev
    - final 10%: test
"""

import os
import sys

# Make sure project root (one level above src) is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.utils import load_config, ensure_dirs, save_csv
from src.metrics import mase, smape, coverage, mse, rmse


def build_sequences(series: np.ndarray, input_len: int, output_len: int):
    """
    Turn a 1D time series into overlapping
    (X, y, pred_start_index) samples.

    X shape: (num_samples, input_len)
    y shape: (num_samples, output_len)
    """
    X_list = []
    y_list = []
    idx_list = []  # index in original series where forecast starts

    n = len(series)
    max_start = n - (input_len + output_len)
    for start in range(max_start):
        x = series[start : start + input_len]
        y = series[start + input_len : start + input_len + output_len]
        X_list.append(x)
        y_list.append(y)
        idx_list.append(start + input_len)  # first forecast step index

    X = np.array(X_list)
    y = np.array(y_list)
    idx_array = np.array(idx_list)
    return X, y, idx_array


def split_train_dev_test(idx_array, n, train_frac=0.8, dev_plus=0.1, output_len=24):
    """
    Given pred_start indices (idx_array) and total length n,
    classify samples into train/dev/test based on where their
    forecast window lies (end index).

    Returns mask arrays: train_mask, dev_mask, test_mask
    """
    train_end = int(n * train_frac)
    dev_end = int(n * (train_frac + dev_plus))

    train_mask = []
    dev_mask = []
    test_mask = []

    for pred_start in idx_array:
        pred_end = pred_start + output_len - 1

        if pred_end < train_end:
            train_mask.append(True)
            dev_mask.append(False)
            test_mask.append(False)
        elif pred_end < dev_end:
            train_mask.append(False)
            dev_mask.append(True)
            test_mask.append(False)
        else:
            train_mask.append(False)
            dev_mask.append(False)
            test_mask.append(True)

    return np.array(train_mask), np.array(dev_mask), np.array(test_mask), train_end


def build_gru_model(input_len: int, output_len: int):
    """
    Simple GRU model:
        input: (168, 1)
        output: 24-step vector
    """
    model = Sequential()
    model.add(Input(shape=(input_len, 1)))
    model.add(GRU(64, return_sequences=False))
    model.add(Dense(output_len))
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def forecast_nn_for_country(cc: str, cfg):
    print(f"\n=== NN forecast for {cc} ===")

    input_len = cfg["nn"]["input_hours"]
    output_len = cfg["nn"]["output_hours"]
    epochs = cfg["nn"]["epochs"]
    batch_size = cfg["nn"]["batch_size"]

    seasonality = cfg["seasonality"]
    train_frac = cfg["forecast"]["train_frac"]

    # Load tidy data
    df = pd.read_csv(f"data/{cc}_tidy.csv", parse_dates=["timestamp"])
    series = df.set_index("timestamp")["load"].asfreq("H").interpolate()
    timestamps = series.index
    n = len(series)

    print(f"[{cc}] series length: {n}")

    # Normalize (simple scaling)
    values = series.values.astype("float32")
    mean = values.mean()
    std = values.std() + 1e-9
    values_norm = (values - mean) / std

    # Build sequences
    X, Y, idx_array = build_sequences(values_norm, input_len, output_len)
    print(f"[{cc}] total samples: {X.shape[0]}")

    # Split into train/dev/test based on prediction window
    train_mask, dev_mask, test_mask, train_end_idx = split_train_dev_test(
        idx_array, n, train_frac=train_frac, dev_plus=0.1, output_len=output_len
    )

    X_train = X[train_mask]
    Y_train = Y[train_mask]
    X_dev = X[dev_mask]
    Y_dev = Y[dev_mask]
    X_test = X[test_mask]
    Y_test = Y[test_mask]

    idx_train = idx_array[train_mask]
    idx_dev = idx_array[dev_mask]
    idx_test = idx_array[test_mask]

    print(
        f"[{cc}] train samples: {len(X_train)}, "
        f"dev: {len(X_dev)}, test: {len(X_test)}"
    )

    # Reshape for GRU: (samples, time, features)
    X_train = X_train[..., np.newaxis]
    X_dev = X_dev[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Build and train model
    model = build_gru_model(input_len, output_len)
    es = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_dev, Y_dev) if len(X_dev) > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1,
    )

    print(f"[{cc}] Training finished.")

    # Predict on dev and test samples
    Y_dev_pred = model.predict(X_dev) if len(X_dev) > 0 else np.empty_like(Y_dev)
    Y_test_pred = model.predict(X_test) if len(X_test) > 0 else np.empty_like(Y_test)

    # Convert predictions back to original scale
    def denorm(pred_norm):
        return pred_norm * std + mean

    Y_dev_true = denorm(Y_dev)
    Y_dev_pred = denorm(Y_dev_pred)
    Y_test_true = denorm(Y_test)
    Y_test_pred = denorm(Y_test_pred)

    # Build forecast rows for dev + test splits
    rows = []

    # We'll also compute residual std to approximate 80% PI
    all_resid = []

    for split_name, idx_split, Y_true_split, Y_pred_split in [
        ("dev", idx_dev, Y_dev_true, Y_dev_pred),
        ("test", idx_test, Y_test_true, Y_test_pred),
    ]:
        for sample_i, pred_start_idx in enumerate(idx_split):
            for h in range(output_len):
                ts_idx = pred_start_idx + h
                if ts_idx >= n:
                    continue
                timestamp = timestamps[ts_idx]
                y_true = float(Y_true_split[sample_i, h])
                yhat = float(Y_pred_split[sample_i, h])
                all_resid.append(y_true - yhat)
                rows.append(
                    {
                        "timestamp": timestamp,
                        "y_true": y_true,
                        "yhat": yhat,
                        "lo": np.nan,  # fill later
                        "hi": np.nan,  # fill later
                        "horizon": h + 1,
                        "train_end": timestamps[train_end_idx - 1],
                        "split": split_name,
                    }
                )

    df_fore = pd.DataFrame(rows).sort_values(["timestamp", "horizon"])

    # Approximate 80% PI using residual std (z ~ 1.28)
    if len(all_resid) > 0:
        resid_std = np.std(all_resid)
        z = 1.28  # approx for 80% two-sided
        df_fore["lo"] = df_fore["yhat"] - z * resid_std
        df_fore["hi"] = df_fore["yhat"] + z * resid_std

    # Save CSV
    out_path = f"outputs/{cc}_forecasts_nn.csv"
    save_csv(df_fore, out_path)

    # Compute metrics (like SARIMA) for dev & test, horizon=1
    metrics_all = {}
    for split_name in ["dev", "test"]:
        sub = df_fore[(df_fore["split"] == split_name) & (df_fore["horizon"] == 1)]
        if sub.empty:
            continue
        y_true = sub["y_true"].values
        yhat = sub["yhat"].values
        lo = sub["lo"].values
        hi = sub["hi"].values

        metrics_all[split_name] = {
            "MASE": mase(y_true, yhat, seasonality),
            "sMAPE": smape(y_true, yhat),
            "MSE": mse(y_true, yhat),
            "RMSE": rmse(y_true, yhat),
            "MAPE": float(
                np.mean(np.abs((y_true - yhat) / (y_true + 1e-9))) * 100
            ),
            "Coverage80": coverage(y_true, lo, hi),
        }

    print(f"[{cc}] NN metrics:")
    for split, m in metrics_all.items():
        print(f"  {split}: {m}")

    return df_fore, metrics_all


def main(config_path: str):
    cfg = load_config(config_path)
    ensure_dirs()

    all_metrics = {}
    for cc in cfg["countries"]:
        try:
            df_fore, metrics = forecast_nn_for_country(cc, cfg)
            all_metrics[cc] = metrics
        except Exception as e:
            print(f"[ERROR] NN forecasting for {cc}: {e}")

    # save metrics JSON
    import json

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/nn_forecast_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("[nn_forecast_metrics] saved to outputs/nn_forecast_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
