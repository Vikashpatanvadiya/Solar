import math

import numpy as np


def compute_metrics(actual, predicted) -> dict:
    """MAE / RMSE / R2 / MAPE over paired monthly values (MU). R2 needs >= 3 points."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    n = len(a)
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "mape": None}
    err = a - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err ** 2)))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = float(1 - np.sum(err ** 2) / ss_tot) if n >= 3 and ss_tot > 0 else None
    nonzero = a != 0
    mape = float(np.mean(np.abs(err[nonzero] / a[nonzero])) * 100) if nonzero.any() else None
    return {"n": int(n), "mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def difference_and_pct(actual, predicted):
    """difference = actual - predicted; percentage_error = difference / actual * 100."""
    if actual is None or predicted is None:
        return None, None
    diff = actual - predicted
    pct = diff / actual * 100 if actual != 0 else None
    return diff, pct
