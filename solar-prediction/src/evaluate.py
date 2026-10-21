"""Evaluation metrics — MSE, RMSE, MAE, R2, MAPE (the paper's five)."""
import math

import numpy as np


def metrics(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    err = a - p
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    nonzero = a != 0
    return {
        "MSE": mse,
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(np.mean(np.abs(err))),
        "R2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "MAPE": float(np.mean(np.abs(err[nonzero] / a[nonzero])) * 100) if nonzero.any() else float("nan"),
    }
