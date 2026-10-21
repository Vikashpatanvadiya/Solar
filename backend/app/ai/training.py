import json

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from .. import config
from ..db import utc_now
from . import data_aggregation, evaluation, features

BASELINE = "seasonal_baseline"
MODEL_PATH = config.ARTIFACT_DIR / "model.joblib"
CONFIG_PATH = config.ARTIFACT_DIR / "feature_config.json"


class InsufficientDataError(Exception):
    pass


def _candidate_models() -> dict:
    return {
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=3, random_state=42
        ),
    }


def _predicted_monthly(frame: pd.DataFrame, ratio) -> pd.Series:
    return pd.Series(ratio, index=frame.index) * frame["same_prev_level"] * frame["days_in_month"]


def _score(frame: pd.DataFrame, ratio) -> dict:
    """Pooled metrics over region-months (MU), plus MAPE on the All-India monthly total."""
    actual = frame["level"] * frame["days_in_month"]
    predicted = _predicted_monthly(frame, ratio)
    metrics = evaluation.compute_metrics(actual, predicted)
    by_period = pd.DataFrame({"period": frame["period"], "a": actual, "p": predicted}).groupby("period").sum()
    metrics["all_india_mape"] = evaluation.compute_metrics(by_period["a"], by_period["p"])["mape"]
    return metrics


def train(monthly: pd.DataFrame | None = None) -> dict:
    if monthly is None:
        monthly = data_aggregation.monthly_by_region()
    frame = features.build_training_frame(monthly)

    periods = sorted(frame["period"].unique()) if not frame.empty else []
    if len(periods) <= config.VALIDATION_MONTHS:
        raise InsufficientDataError(
            f"Only {len(periods)} months have all required features; need more than {config.VALIDATION_MONTHS}."
        )
    val_start = periods[-config.VALIDATION_MONTHS]
    train_df = frame[frame["period"] < val_start]  # chronological split, no shuffling
    val_df = frame[frame["period"] >= val_start]
    if len(train_df) < config.MIN_TRAIN_ROWS or len(val_df) < config.MIN_VALIDATION_ROWS:
        raise InsufficientDataError(
            f"Not enough data for a chronological validation: {len(train_df)} training rows "
            f"(need {config.MIN_TRAIN_ROWS}) and {len(val_df)} validation rows (need {config.MIN_VALIDATION_ROWS})."
        )

    x_train, y_train = train_df[features.FEATURE_COLUMNS], train_df[features.TARGET]
    x_val = val_df[features.FEATURE_COLUMNS]

    candidates = {BASELINE: _score(val_df, 1.0)}
    for name, model in _candidate_models().items():
        model.fit(x_train, y_train)
        candidates[name] = _score(val_df, model.predict(x_val))

    best_name = min(candidates, key=lambda n: candidates[n]["rmse"])  # dict order puts the baseline first on ties

    if best_name == BASELINE:
        final_model = None
    else:
        final_model = _candidate_models()[best_name]
        final_model.fit(frame[features.FEATURE_COLUMNS], frame[features.TARGET])  # refit on every usable row

    history = data_aggregation.history_summary(monthly)
    fconfig = features.feature_config()
    fconfig.update(
        {
            "trained_at": utc_now(),
            "history": history,
            "train_periods": [pd.Timestamp(train_df["period"].min()).strftime("%Y-%m"), pd.Timestamp(train_df["period"].max()).strftime("%Y-%m")],
            "validation_periods": [pd.Timestamp(val_df["period"].min()).strftime("%Y-%m"), pd.Timestamp(val_df["period"].max()).strftime("%Y-%m")],
        }
    )

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_name": best_name, "model": final_model, "feature_config": fconfig}, MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps(fconfig, indent=2))

    warnings = _warnings(history, frame, train_df, val_df, candidates, best_name)
    return {
        "model_name": best_name,
        "validation_metrics": candidates[best_name],
        "candidates": candidates,
        "n_training_rows": int(len(train_df)),
        "n_validation_rows": int(len(val_df)),
        "n_usable_rows": int(len(frame)),
        "feature_config": fconfig,
        "warnings": warnings,
        "artifact_path": str(MODEL_PATH),
    }


def _warnings(history, frame, train_df, val_df, candidates, best_name) -> list[str]:
    w = [
        f"History covers only {history['n_months']} complete months ({history['first_month']} to {history['last_month']}), "
        "less than two full annual cycles. Each 2026 month is anchored on a single prior-year observation.",
        f"Only {len(frame)} region-months have all features. Validation used {len(val_df)} region-months "
        f"({len(train_df)} for training), so the accuracy figures are indicative, not precise.",
        "R² is pooled across regions of very different size, so it is inflated by between-region differences; "
        "the All-India MAPE is the more intuitive accuracy figure.",
        "Forecasts are generated month by month; each month feeds the next, so errors compound and later 2026 months are less reliable.",
    ]
    if frame["year"].nunique() == 1:
        w.append("The 'year' feature has a single value in the training data and carries no learned signal.")
    if best_name == BASELINE:
        w.append("Neither machine-learning model beat the seasonal baseline on validation, so the baseline is being used.")
    return w
