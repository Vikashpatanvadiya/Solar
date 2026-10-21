"""
Shared data loader for all models.

Loads the processed dataset, selects a feature set (WITH or WITHOUT module
temperature — the paper's key comparison), splits chronologically (2024-25 train,
2026 test), and MinMax-scales features and target to [0,1] using statistics from the
TRAINING rows only (so the test year never leaks into scaling).

Every model trains on scaled data and predicts scaled values; call
`inverse_y` to turn predictions back into real MU for evaluation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

PROCESSED = config.PROCESSED_DIR / "allindia_daily.csv"
TARGET = config.TARGET_COL

# The paper's central experiment: the same models WITH vs WITHOUT module temperature.
SEASONAL = ["doy_sin", "doy_cos"]
FEATURES_WITH_TM = ["ghi", "ambient_temp", "module_temp", "wind_speed", "relative_humidity"] + SEASONAL
FEATURES_WITHOUT_TM = ["ghi", "ambient_temp", "wind_speed", "relative_humidity"] + SEASONAL


class Dataset:
    """Models learn weather -> capacity factor (generation per MW), which is stationary
    across years. Predictions are turned back into real MU by multiplying the predicted
    capacity factor by that day's installed capacity."""
    CF = "capacity_factor"

    def __init__(self, features):
        df = pd.read_csv(PROCESSED, parse_dates=["date"])
        self.features = features
        tr, te = df[df.split == "train"], df[df.split == "test"]
        self.dates_test = te["date"].reset_index(drop=True)
        self.cap_test = te["capacity_MW"].values     # MW, to rescale predictions to MU

        self.x_scaler = MinMaxScaler().fit(tr[features].values)
        self.y_scaler = MinMaxScaler().fit(tr[[self.CF]].values)

        self.Xtr = self.x_scaler.transform(tr[features].values)
        self.Xte = self.x_scaler.transform(te[features].values)
        self.ytr = self.y_scaler.transform(tr[[self.CF]].values).ravel()   # scaled capacity factor
        self.yte = self.y_scaler.transform(te[[self.CF]].values).ravel()

        self.ytr_real = tr[TARGET].values            # real MU, for reference
        self.yte_real = te[TARGET].values            # real MU, the test truth

    def to_generation(self, cf_scaled):
        """Scaled capacity-factor prediction -> real daily solar generation (MU)."""
        cf = self.y_scaler.inverse_transform(np.asarray(cf_scaled).reshape(-1, 1)).ravel()
        return cf * self.cap_test

    # kept for compatibility; predictions are capacity factors, convert with to_generation
    inverse_y = to_generation


def load(with_module_temp=True):
    return Dataset(FEATURES_WITH_TM if with_module_temp else FEATURES_WITHOUT_TM)
