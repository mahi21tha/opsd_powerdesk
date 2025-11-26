"""
Expanding-origin backtest with SARIMAX.

Outputs:
- outputs/<CC>_forecasts.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import yaml

from src.utils import load_config, ensure_dirs, save_csv
from src.metrics import mase, smape, coverage, mse, rmse
from tqdm import tqdm


def expanding_backtest(cc: str, cfg, orders):
    print(f"\n=== Backtest for {cc} ===")
    df = pd.read_csv(f"data/{cc}_tidy.csv", parse_dates=["timestamp"])
    seasonality = cfg["seasonality"]
    horizon = cfg["forecast"]["horizon"]
    stride = cfg["forecast"]["stride_hours"]
    warmup = cfg["forecast"]["warmup_days"] * 24
    train_frac = cfg["forecast"]["train_frac"]

    series = df.set_index("timestamp")["load"].asfreq("H").interpolate()
    n = len(series)
    train_end_idx = int(n * train_frac)
    dev_end_idx = int(n * (train_frac + 0.1))

    p, d, q = orders["order"]
    P, D, Q, s = orders["seasonal_order"]

    rows = []

    for start_idx in tqdm(range(warmup, n - horizon, stride)):
        train = series.iloc[:start_idx]
        forecast_mid_idx = start_idx + horizon // 2
        split = "dev" if forecast_mid_idx < dev_end_idx else "test"

        try:
            model = SARIMAX(
                train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            pred = res.get_forecast(steps=horizon)
            mean = pred.predicted_mean
            ci = pred.conf_int(alpha=0.2)  # 80% PI

            for h in range(horizon):
                idx = series.index[start_idx + h]
                y_true = float(series.iloc[start_idx + h])
                rows.append(
                    {
                        "timestamp": idx,
                        "y_true": y_true,
                        "yhat": float(mean.iloc[h]),
                        "lo": float(ci.iloc[h, 0]),
                        "hi": float(ci.iloc[h, 1]),
                        "horizon": h + 1,
                        "train_end": train.index[-1],
                        "split": split,
                    }
                )
        except Exception as e:
            print(f"[WARN] SARIMAX fail at idx={start_idx}: {e}")

    df_fore = pd.DataFrame(rows)
    return df_fore


def metrics_summary(df_fore, seasonality):
    out = {}
    for split in ["dev", "test"]:
        s = df_fore[(df_fore["split"] == split) & (df_fore["horizon"] == 1)]
        if s.empty:
            continue
        y_true = s["y_true"].values
        yhat = s["yhat"].values
        lo = s["lo"].values
        hi = s["hi"].values

        out[split] = {
            "MASE": mase(y_true, yhat, seasonality),
            "sMAPE": smape(y_true, yhat),
            "MSE": mse(y_true, yhat),
            "RMSE": rmse(y_true, yhat),
            "MAPE": float(np.mean(np.abs((y_true - yhat) / (y_true + 1e-9))) * 100),
            "Coverage80": coverage(y_true, lo, hi),
        }
    return out


def main(config_path: str):
    cfg = load_config(config_path)
    ensure_dirs()

    with open("outputs/orders.yaml", "r") as f:
        orders_all = yaml.safe_load(f)

    all_metrics = {}

    for cc in cfg["countries"]:
        if cc not in orders_all:
            print(f"[WARN] No SARIMA order for {cc}, skipping.")
            continue

        df_fore = expanding_backtest(cc, cfg, orders_all[cc])
        save_csv(df_fore, f"outputs/{cc}_forecasts.csv")
        metrics = metrics_summary(df_fore, cfg["seasonality"])
        all_metrics[cc] = metrics
        print(f"\n[Metrics] {cc}")
        print(metrics)

    # Optionally save metrics JSON
    import json

    with open("outputs/forecast_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\n[forecast_metrics] saved to outputs/forecast_metrics.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)
