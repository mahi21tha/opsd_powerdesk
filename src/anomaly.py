"""
Compute rolling z-score anomalies + optional CUSUM.

Inputs:
- outputs/<CC>_forecasts.csv

Outputs:
- outputs/<CC>_anomalies.csv
"""

import argparse
import pandas as pd
import numpy as np

from src.utils import load_config, ensure_dirs, save_csv


def compute_z_and_cusum(df_fore: pd.DataFrame, cfg):
    df = df_fore.copy()
    # use only horizon=1 for anomaly detection
    df = df[df["horizon"] == 1].sort_values("timestamp")

    w = cfg["anomaly"]["z_window_hours"]
    m = cfg["anomaly"]["z_min_periods"]
    z_th = cfg["anomaly"]["z_threshold"]
    k = cfg["anomaly"]["cusum_k"]
    h = cfg["anomaly"]["cusum_h"]

    df["resid"] = df["y_true"] - df["yhat"]
    df["mu"] = df["resid"].rolling(window=w, min_periods=m).mean()
    df["sigma"] = df["resid"].rolling(window=w, min_periods=m).std()
    df["z_resid"] = (df["resid"] - df["mu"]) / (df["sigma"] + 1e-9)
    df["flag_z"] = (df["z_resid"].abs() >= z_th).astype(int)

    # CUSUM on z
    s_pos = 0.0
    s_neg = 0.0
    flags_c = []
    for z in df["z_resid"].fillna(0):
        s_pos = max(0.0, s_pos + z - k)
        s_neg = min(0.0, s_neg + z + k)
        flags_c.append(int((s_pos > h) or (abs(s_neg) > h)))
    df["flag_cusum"] = flags_c

    return df


def main(config_path: str):
    cfg = load_config(config_path)
    ensure_dirs()

    for cc in cfg["countries"]:
        try:
            df_fore = pd.read_csv(f"outputs/{cc}_forecasts.csv", parse_dates=["timestamp"])
        except FileNotFoundError:
            print(f"[WARN] outputs/{cc}_forecasts.csv not found, skipping.")
            continue

        df_anom = compute_z_and_cusum(df_fore, cfg)
        save_csv(df_anom, f"outputs/{cc}_anomalies.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)
