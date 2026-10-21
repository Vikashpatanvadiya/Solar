import calendar

import joblib
import numpy as np
import pandas as pd

from .. import config
from . import features
from .training import BASELINE, MODEL_PATH


class ModelNotTrainedError(Exception):
    pass


class ForecastError(Exception):
    pass


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError("No trained model found. Train the model first.")
    return joblib.load(MODEL_PATH)


def generate(monthly: pd.DataFrame, bundle: dict, year: int) -> list[dict]:
    """Recursive 12-month forecast. Returns rows {year, month, region, predicted_generation (MU)}
    for the five regions plus ALL_INDIA (their sum)."""
    model, name = bundle["model"], bundle["model_name"]
    levels = {
        region: g.set_index("period")["avg_daily_mu"].to_dict()
        for region, g in monthly[monthly["complete"]].groupby("region")
    }
    out, all_india = [], {}
    for month in range(1, 13):
        period = pd.Timestamp(year=year, month=month, day=1)
        days = calendar.monthrange(year, month)[1]
        for region in config.MODEL_REGIONS:
            hist = levels[region]
            same = hist.get(period - pd.DateOffset(months=12))
            prevs = [hist.get(period - pd.DateOffset(months=k)) for k in (1, 2, 3)]
            if same is None or any(p is None for p in prevs):
                raise ForecastError(f"Missing history needed to forecast {region} {year}-{month:02d}.")
            if name == BASELINE:
                ratio = 1.0
            else:
                row = features.feature_row(region, year, month, prevs[0], same, float(np.mean(prevs)))
                ratio = max(float(model.predict(pd.DataFrame([row])[features.FEATURE_COLUMNS])[0]), 0.0)
            level = ratio * same
            hist[period] = level  # feeds the following months
            total = level * days
            out.append({"year": year, "month": month, "region": region, "predicted_generation": total})
            all_india[month] = all_india.get(month, 0.0) + total
    out += [
        {"year": year, "month": m, "region": config.ALL_INDIA, "predicted_generation": v}
        for m, v in all_india.items()
    ]
    return out
