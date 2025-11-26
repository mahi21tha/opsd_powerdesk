"""
ML-based anomaly classifier (LightGBM) using silver labels.

Inputs:
- outputs/<CC>_anomalies.csv

Outputs:
- outputs/anomaly_ml_eval.json
"""

import argparse
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.utils import load_config, ensure_dirs


def create_silver_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_z"] = df["z_resid"].abs()
    outside_pi = (df["y_true"] < df["lo"]) | (df["y_true"] > df["hi"])

    df["pos"] = ((df["abs_z"] >= 3.5) | (outside_pi & (df["abs_z"] >= 2.5))).astype(int)
    df["neg"] = ((df["abs_z"] < 1.0) & (~outside_pi)).astype(int)

    df = df[(df["pos"] == 1) | (df["neg"] == 1)].copy()
    df["label"] = df["pos"]
    return df


def main(config_path: str):
    cfg = load_config(config_path)
    ensure_dirs()

    results = {}

    for cc in cfg["countries"]:
        try:
            df = pd.read_csv(f"outputs/{cc}_anomalies.csv", parse_dates=["timestamp"])
        except FileNotFoundError:
            print(f"[WARN] outputs/{cc}_anomalies.csv not found, skipping ML for {cc}.")
            continue

        df_lab = create_silver_labels(df)
        if df_lab.empty:
            print(f"[WARN] no silver labels for {cc}, skipping.")
            continue

        df_lab = df_lab.sort_values("timestamp")

        # simple lag features
        df_lab["lag1"] = df_lab["y_true"].shift(1)
        df_lab["lag24"] = df_lab["y_true"].shift(24)
        df_lab["rmean24"] = df_lab["y_true"].rolling(24).mean()
        df_lab["hour"] = df_lab["timestamp"].dt.hour
        df_lab["dow"] = df_lab["timestamp"].dt.dayofweek
        df_lab = df_lab.dropna()

        features = ["lag1", "lag24", "rmean24", "hour", "dow"]
        X = df_lab[features].values
        y = df_lab["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = lgb.LGBMClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train_s, y_train)

        probs = clf.predict_proba(X_test_s)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall, precision)

        best_f1 = 0.0
        for p_val, r_val, thr in zip(precision, recall, thresholds):
            if p_val >= 0.80:
                f1_val = 2 * p_val * r_val / (p_val + r_val + 1e-9)
                best_f1 = max(best_f1, f1_val)

        results[cc] = {"PR_AUC": float(pr_auc), "F1_at_P_0.80": float(best_f1)}
        print(f"[ML] {cc}: {results[cc]}")

    with open("outputs/anomaly_ml_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[anomaly_ml_eval] saved to outputs/anomaly_ml_eval.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)