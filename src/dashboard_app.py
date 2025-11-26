"""
Streamlit dashboard for OPSD PowerDesk.

Run from project root:

    streamlit run src/dashboard_app.py -- --config config.yaml
"""

import argparse
from pathlib import Path
import os
import sys

import pandas as pd
import streamlit as st
import yaml

# Ensure project root on sys.path (one level above src)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_config   # if you want to use it elsewhere
from src.metrics import mase, coverage as cov_fun

# Helpers to load configuration and data

def load_cfg(path: str):
    """Load YAML config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_forecasts(cc: str) -> pd.DataFrame | None:
    """Load SARIMA forecasts for a given country."""
    fpath = Path(f"outputs/{cc}_forecasts.csv")
    if not fpath.exists():
        return None
    return pd.read_csv(fpath, parse_dates=["timestamp"])


def load_nn_forecasts(cc: str) -> pd.DataFrame | None:
    """Load Neural GRU forecasts for a given country."""
    fpath = Path(f"outputs/{cc}_forecasts_nn.csv")
    if not fpath.exists():
        return None
    return pd.read_csv(fpath, parse_dates=["timestamp"])


def load_anomalies(cc: str) -> pd.DataFrame | None:
    """Load anomaly detection outputs for a given country."""
    fpath = Path(f"outputs/{cc}_anomalies.csv")
    if not fpath.exists():
        return None
    return pd.read_csv(fpath, parse_dates=["timestamp"])


# Main Streamlit app

def main_app(cfg_path: str):
    cfg = load_cfg(cfg_path)

    st.set_page_config(layout="wide", page_title="OPSD PowerDesk")

    # Sidebar controls
    st.sidebar.title("OPSD PowerDesk Live Monitoring Dashboard")

    # Country selector
    cc = st.sidebar.selectbox(
        "Country",
        cfg["countries"],
        index=cfg["countries"].index(cfg["live_country"]),
    )

    # History window selector (for main time-series chart)
    history_option = st.sidebar.selectbox(
        "History window",
        ["Last 7 days", "Last 14 days", "Last 1 years"],
        index=0,
    )

    # Convert dropdown selection to pandas offset string
    if "7 days" in history_option:
        history_window = "7D"
    elif "14 days" in history_option:
        history_window = "14D"
    else:  # Last 1 years
        history_window = "365D"

    # Model selector (SARIMA vs GRU)
    model_option = st.sidebar.selectbox(
        "Model",
        ["SARIMA", "Neural GRU"],
        index=0,
    )

    # Load forecasts based on selection
    if model_option == "SARIMA":
        df_fore = load_forecasts(cc)
    else:
        df_fore = load_nn_forecasts(cc)

    df_anom = load_anomalies(cc)

    if df_fore is None or df_fore.empty:
        st.warning(f"No forecast found for {model_option} ({cc}). "
                   f"Run the corresponding pipeline first.")
        return

    st.title(f"Country: {cc} — {model_option} Forecasts")

    # 1) Main historical view: horizon=1 over chosen history window

    # filter horizon=1 (one-hour-ahead forecast)
    s1 = df_fore[df_fore["horizon"] == 1].set_index("timestamp").sort_index()
    s1_last = s1.last(history_window)

    if s1_last.empty:
        st.warning("No data available in the selected history window.")
        return

    st.subheader(f"{history_option}: true vs forecast (horizon=1) + 80% PI")
    st.line_chart(s1_last[["y_true", "yhat"]])
    st.area_chart(s1_last[["lo", "hi"]])

     # Day-ahead 24-hour forecast graph (horizons 1–24)
 
    st.subheader("Day-ahead 24-hour forecast profile (horizons 1–24)")

    # Use all horizons 1..24 and take the most recent forecast for each horizon
    df_24_all = df_fore[df_fore["horizon"].between(1, 24)].copy()

    if not df_24_all.empty:
        # For each horizon, keep the last row by timestamp (= most recent forecast)
        df_24_last = (
            df_24_all.sort_values(["horizon", "timestamp"])
            .groupby("horizon", as_index=False)
            .tail(1)
            .set_index("horizon")
            .sort_index()
        )

        # Plot forecast as function of "hours ahead"
        st.line_chart(df_24_last[["yhat"]])

        # If prediction intervals available, also show them
        if "lo" in df_24_last.columns and "hi" in df_24_last.columns:
            st.area_chart(df_24_last[["lo", "hi"]])
    else:
        st.info("24-hour ahead forecasts (horizons 1–24) are not available for this model/output file.")

    
    # 3) KPI section: rolling-7d metrics + anomaly summary
  
    st.subheader("KPIs (rough)")

    # Anomaly hours today (from anomaly file, if present)
    if df_anom is not None and not df_anom.empty:
        an = df_anom.set_index("timestamp").sort_index()
        an_today = an[an.index.date == an.index.max().date()]
        anomaly_hours_today = int(an_today["flag_z"].sum())
    else:
        anomaly_hours_today = 0

    c1, c2, c3, c4 = st.columns(4)

    # rolling-last-7-days metrics based on horizon=1 forecasts
    last_7d = s1_last.last("7D")
    if len(last_7d) > 0:
        mase_7d = mase(last_7d["y_true"], last_7d["yhat"], cfg["seasonality"])
        cov_7d = cov_fun(last_7d["y_true"], last_7d["lo"], last_7d["hi"])
    else:
        mase_7d = None
        cov_7d = None

    c1.metric("Model", model_option)
    c2.metric("rolling-7d MASE", f"{mase_7d:.3f}" if mase_7d is not None else "-")
    c3.metric("80% PI Coverage (7d)", f"{cov_7d:.3f}" if cov_7d is not None else "-")
    c4.metric("Anomaly hours today", anomaly_hours_today)


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main_app(args.config)
