"""Feature engineering.

Everything is expressed as average daily generation (MU/day) so months of different
length are comparable; monthly totals are recovered by multiplying by days-in-month.
The model target is the year-over-year ratio  level(t) / level(t-12):  it is scale-free
(NER and NR can share one model) and lets tree models predict growth without needing to
extrapolate beyond the generation levels seen in training.
"""
import math

import pandas as pd

from .. import config

REGION_COLUMNS = [f"region_{r}" for r in config.MODEL_REGIONS]
NUMERIC_FEATURES = [
    "month",
    "month_sin",
    "month_cos",
    "year",
    "prev_month_gen",
    "same_month_prev_year_gen",
    "rolling_avg_3",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + REGION_COLUMNS
TARGET = "yoy_ratio"


def feature_row(region, year, month, prev_month_gen, same_prev_year_gen, rolling_avg_3) -> dict:
    row = {
        "month": month,
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "year": year,
        "prev_month_gen": prev_month_gen,
        "same_month_prev_year_gen": same_prev_year_gen,
        "rolling_avg_3": rolling_avg_3,
    }
    for r in config.MODEL_REGIONS:
        row[f"region_{r}"] = 1.0 if r == region else 0.0
    return row


def build_training_frame(monthly: pd.DataFrame) -> pd.DataFrame:
    """One row per region-month that has every feature (previous month, same month last
    year, 3-month rolling mean) available from complete historical months."""
    rows = []
    for region, g in monthly[monthly["complete"]].groupby("region"):
        s = g.set_index("period")["avg_daily_mu"]
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))  # gaps become NaN
        prev = s.shift(1)
        same = s.shift(12)
        roll = pd.concat([s.shift(1), s.shift(2), s.shift(3)], axis=1).mean(axis=1, skipna=False)
        for period in s.index:
            level, p, sm, r3 = s[period], prev[period], same[period], roll[period]
            if any(pd.isna(v) for v in (level, p, sm, r3)):
                continue
            row = feature_row(region, period.year, period.month, p, sm, r3)
            row.update(
                {
                    "region": region,
                    "period": period,
                    "days_in_month": period.days_in_month,
                    "level": level,
                    "same_prev_level": sm,
                    TARGET: level / sm,
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["period", "region"]).reset_index(drop=True)
    return frame


def feature_config() -> dict:
    return {
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "target_definition": "average daily generation(t) / average daily generation(t-12 months)",
        "basis": "average daily generation, MU/day; monthly MU = level x days in month",
        "regions": config.MODEL_REGIONS,
        "rolling_window_months": 3,
        "all_india": "sum of the five regional forecasts",
    }
