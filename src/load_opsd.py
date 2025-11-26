"""
Load OPSD CSV and prepare per-country tidy frames.

This version:
- Reads the full OPSD time series CSV
- Keeps ONLY the last N days (lookback_days, default 730) for all countries
- Creates data/<CC>_tidy.csv, which all other scripts use

So the whole project (forecast, anomalies, live, etc.) runs on a consistent 2-year window.
"""

import argparse
from pathlib import Path
import pandas as pd

from src.utils import load_config, ensure_dirs, save_csv


def load_opsd_raw(path: str, cfg):
    print(f"[load_opsd_raw] Loading OPSD CSV from: {path}")

    # figure out how many days to keep (2 years by default)
    lookback_days = cfg.get("data", {}).get("lookback_days", 730)

    # only load columns we actually need: timestamp + load for each country
    usecols = ["utc_timestamp"]
    for cc in cfg["countries"]:
        usecols.append(f"{cc}_load_actual_entsoe_transparency")

    df = pd.read_csv(path, usecols=usecols, parse_dates=["utc_timestamp"])
    df = df.sort_values("utc_timestamp")

    # 🔹 Compute dynamic 2-year window based on max timestamp
    end = df["utc_timestamp"].max()
    start = end - pd.Timedelta(days=lookback_days)
    df = df[df["utc_timestamp"].between(start, end)]

    print(f"[load_opsd_raw] Using data from {start.date()} to {end.date()} ({len(df)} rows)")
    return df


def build_country_frames(df: pd.DataFrame, cfg):
    country_frames = {}
    for cc in cfg["countries"]:
        load_col = f"{cc}_load_actual_entsoe_transparency"
        if load_col not in df.columns:
            print(f"[WARN] no load column for {cc}, skipping.")
            continue

        sub = df[["utc_timestamp", load_col]].rename(
            columns={"utc_timestamp": "timestamp", load_col: "load"}
        )
        sub = sub.dropna(subset=["load"])
        sub = sub.sort_values("timestamp").reset_index(drop=True)

        print(f"[build_country_frames] {cc}: {len(sub)} rows after 2-year filter")
        country_frames[cc] = sub

        out_path = Path("data") / f"{cc}_tidy.csv"
        save_csv(sub, out_path)

    return country_frames


def main(config_path: str):
    cfg = load_config(config_path)
    ensure_dirs()

    opsd_csv = cfg["data"]["opsd_csv"]
    df_raw = load_opsd_raw(opsd_csv, cfg)
    build_country_frames(df_raw, cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)
