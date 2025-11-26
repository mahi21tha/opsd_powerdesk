import numpy as np
from sklearn.metrics import mean_squared_error


def mase(y_true, y_pred, seasonality=24):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true) <= seasonality:
        return np.mean(np.abs(y_true - y_pred))
    denom = np.mean(np.abs(y_true[seasonality:] - y_true[:-seasonality]))
    if denom == 0:
        return np.mean(np.abs(y_true - y_pred))
    return np.mean(np.abs(y_true - y_pred)) / denom


def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (denom + 1e-9))


def coverage(y_true, lo, hi):
    y_true = np.array(y_true)
    lo = np.array(lo)
    hi = np.array(hi)
    inside = (y_true >= lo) & (y_true <= hi)
    return float(np.mean(inside.astype(float)))


def mse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
