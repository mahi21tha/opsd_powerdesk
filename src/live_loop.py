"""
Live ingestion + online adaptation (rolling SARIMA) for one country.

Inputs:
- data/<LIVE_CC>_tidy.csv
- outputs/orders.yaml

Outputs:
- outputs/<LIVE_CC>_online_updates.csv
"""

import argparse
from time import time

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import yaml

from src.utils import load_config, ensure_dirs, save_csv


def live_simulate_sarima(cc: str, cfg):
    df = pd.read_csv(f"data/{cc}_tidy.csv", parse_dates=["timestamp"])
    series = df.set_index("timestamp")["load"].asfreq("H").interpolate()

    start_hist = cfg["live"]["start_history_days"] * 24
    simulate_hours = cfg["live"]["simulate_hours"]
    sarima_days = cfg["adaptation"]["sarima_days"]

    history = series.iloc[:start_hist].copy()
    stream = series.iloc[start_hist : start_hist + simulate_hours].copy()

    with open("outputs/orders.yaml", "r") as f:
        orders_all = yaml.safe_load(f)
    orders = orders_all[cc]

    p, d, q = orders["order"]
    P, D, Q, s = orders["seasonal_order"]

    model = SARIMAX(
        history,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    logs = []
    z_abs_ewma = []

    for ts, y_obs in stream.items():
        t0 = time()

        # one-step forecast
        pred = res.get_forecast(steps=1)
        yhat = float(pred.predicted_mean.iloc[0])
        resid = y_obs - yhat

        # approximate z-score using rolling std of history
        sigma = float(history[-336:].std() + 1e-9) if len(history) > 336 else float(history.std() + 1e-9)
        z_abs = abs(resid) / sigma
        if not z_abs_ewma:
            z_ewma = z_abs
        else:
            alpha = 0.1
            z_ewma = alpha * z_abs + (1 - alpha) * z_abs_ewma[-1]
        z_abs_ewma.append(z_ewma)

        history.loc[ts] = y_obs  # append new point

        # drift check: EWMA(|z|) > 95th percentile of last 30 days
        if len(z_abs_ewma) > 24 * 30:
            tail = np.array(z_abs_ewma[-24 * 30 :])
            thresh = np.percentile(tail, 95)
            drift = z_ewma > thresh
        else:
            drift = False

        reason = "scheduled"
        # refit daily at 00:00 or on drift
        if ts.hour == 0 or drift:
            start_idx = max(0, len(history) - sarima_days * 24)
            train = history.iloc[start_idx:]
            model = SARIMAX(
                train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            reason = "drift" if drift else "scheduled"

        duration_s = time() - t0
        logs.append(
            {
                "timestamp": ts,
                "strategy": cfg["adaptation"]["strategy"],
                "reason": reason,
                "duration_s": duration_s,
            }
        )

    df_logs = pd.DataFrame(logs)
    return df_logs


def main(config_path: str):
    cfg = load_config(config_path)
    ensure_dirs()

    cc = cfg["live_country"]
    df_logs = live_simulate_sarima(cc, cfg)
    save_csv(df_logs, f"outputs/{cc}_online_updates.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)
