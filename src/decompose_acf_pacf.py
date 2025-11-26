"""
STL decomposition + ACF/PACF + SARIMA order selection via pmdarima.auto_arima

Outputs:
- outputs/<CC>_stl.png
- outputs/<CC>_acf_pacf.png
- outputs/orders.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
import yaml

from src.utils import load_config, ensure_dirs


def stl_and_order_for_country(cc: str, cfg):
    seasonality = cfg["seasonality"]

    # 1) Load tidy country data
    df = pd.read_csv(f"data/{cc}_tidy.csv", parse_dates=["timestamp"])
    if df.empty:
        raise ValueError(f"data/{cc}_tidy.csv is empty")

    # 2) Build time series with hourly frequency
    ts = df.set_index("timestamp")["load"].asfreq("h").interpolate()

    if ts.isna().all() or len(ts) < seasonality * 10:
        raise ValueError(f"Not enough data points for {cc} to run STL/ARIMA")

    # 3) Use last 2 years (≈730 days) for STL & ACF/PACF
    end = ts.index.max()
    start = end - pd.Timedelta(days=730)   # 🔹 2 years
    ts_2y = ts.loc[start:end]

    print(f"\n=== STL for {cc} ===")
    print(f"[INFO] Using 2-year subset from {start.date()} to {end.date()} ({len(ts_2y)} points)")

    # 4) STL on 2-year window
    stl = STL(ts_2y, period=seasonality, robust=True).fit()
    fig = stl.plot()
    fig.suptitle(f"{cc} – STL (last 2 years)", fontsize=14)
    Path("outputs").mkdir(exist_ok=True)
    fig.savefig(f"outputs/{cc}_stl.png", bbox_inches="tight")
    plt.close(fig)

    # 5) ACF/PACF on same 2-year window
    print(f"[ACF/PACF] {cc}")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(ts_2y.dropna(), lags=48, ax=axes[0])
    plot_pacf(ts_2y.dropna(), lags=48, ax=axes[1])
    axes[0].set_title(f"{cc} – ACF (last 2 years)")
    axes[1].set_title(f"{cc} – PACF (last 2 years)")
    fig.savefig(f"outputs/{cc}_acf_pacf.png", bbox_inches="tight")
    plt.close(fig)

    # 6) For auto_arima, still use the same 2-year window,
    #    but downsample to reduce memory (keep every 3rd hour)
    ts_auto = ts_2y[::3]   # 🔹 3-hourly subsample

    print(f"[auto_arima] {cc} on 2-year subsample (len={len(ts_auto)}), reduced search...")

    try:
        stepwise = auto_arima(
            ts_auto,
            seasonal=True,
            m=seasonality,
            start_p=0,
            start_q=0,
            max_p=1,   # small grid for RAM
            max_q=1,
            start_P=0,
            start_Q=0,
            max_P=1,
            max_Q=1,
            d=None,
            D=None,
            max_d=1,
            max_D=1,
            information_criterion="bic",
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
        )

        order = {
            "order": list(stepwise.order),              # tuple -> list
            "seasonal_order": list(stepwise.seasonal_order),
        }
        print(f"[auto_arima] {cc} best order: {order}")
        return order

    except MemoryError:
        print(f"[auto_arima] {cc} MemoryError – fallback to SARIMA(1,0,1)x(1,0,1,{seasonality})")
        return {
            "order": [1, 0, 1],
            "seasonal_order": [1, 0, 1, seasonality],
        }

    except Exception as e:
        print(f"[auto_arima] {cc} failed with {e} – fallback to SARIMA(1,0,1)x(1,0,1,{seasonality})")
        return {
            "order": [1, 0, 1],
            "seasonal_order": [1, 0, 1, seasonality],
        }



def main(config_path: str):
    cfg = load_config(config_path)
    ensure_dirs()

    orders = {}
    for cc in cfg["countries"]:
        try:
            orders[cc] = stl_and_order_for_country(cc, cfg)
        except Exception as e:
            print(f"[ERROR] {cc} -> {e}")

    # Save orders as plain YAML (lists, no python/tuple tags)
    with open("outputs/orders.yaml", "w") as f:
        yaml.safe_dump(orders, f)
    print("\n[orders] written to outputs/orders.yaml")
    print(orders)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
