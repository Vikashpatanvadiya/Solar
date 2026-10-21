"""
Forward forecasting for a year that has not happened yet (2027).

This is deliberately a different kind of output from the 2026 result, and the code says so
in every return value. For 2026 we had the REAL weather that occurred, so the model could be
scored day by day (95.0% accurate). 2027 has no weather yet, so two things are assumed:

  1. Weather  -- a climatology ("typical year") profile: for every day of the year, the
     median of that same day across 2024, 2025 and 2026, smoothed with a +/-3 day window.
     A typical year has no heatwaves, no failed monsoon, no unusual cloud. Real 2027 will
     deviate from it, and the daily accuracy measured on 2026 does NOT carry over.
  2. Capacity -- projected past the published table (which ends Dec 2025). Two paths are
     produced, "compound" (recent 12-month compound growth) and "linear" (recent 12-month
     straight-line slope), so the fleet assumption is shown as a range instead of a
     single confident number.

What the model still contributes is the SHAPE: how generation responds to irradiance,
temperature, wind and humidity, and how that plays out across the seasons.

Run:  .venv/bin/python src/forecast_future.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
import config
import dataset
import preprocess
import regions as regions_mod
from models import ann, ann_pso, decision_tree, gpr_model, linear, svm_model

warnings.filterwarnings("ignore")

MODELS = [linear, decision_tree, svm_model, gpr_model, ann, ann_pso]
FUTURE_JSON = config.RESULTS_DIR / "future_forecast.json"
CLIMATOLOGY_WINDOW = 3          # +/- days blended into each day-of-year median
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------- climatology
def climatology(weather: pd.DataFrame, window: int = CLIMATOLOGY_WINDOW) -> pd.DataFrame:
    """Median weather for each day-of-year across every year we have.

    The +/-window smoothing matters: a single calendar day has only 3 observations, which
    is far too few for a stable median. Blending neighbouring days gives ~21.
    """
    w = weather.copy()
    w["doy"] = w.index.dayofyear
    cols = [c for c in w.columns if c != "doy"]
    out = {}
    for doy in range(1, 367):
        # circular window over the year, so 1 Jan borrows from late December
        offsets = [(doy - 1 + d) % 366 + 1 for d in range(-window, window + 1)]
        sel = w[w["doy"].isin(offsets)]
        out[doy] = sel[cols].median()
    return pd.DataFrame(out).T.rename_axis("doy")


def national_climatology() -> pd.DataFrame:
    return climatology(preprocess.add_module_temperature(preprocess.national_weather()))


def year_frame(year: int, climo: pd.DataFrame, cap_mode: str, region: str = None) -> pd.DataFrame:
    """A full synthetic year of model inputs built from the climatology profile."""
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    df = climo.reindex([d.dayofyear for d in dates]).reset_index(drop=True)
    df.insert(0, "date", dates)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["capacity_MW"] = preprocess.capacity_series(df["date"], region, mode=cap_mode).values
    return df


# ---------------------------------------------------------------- fitting on everything
def _fit(model_module, hist: pd.DataFrame, feats):
    """Train on EVERY historical row we have (2024 -> latest 2026), not just 2024-25.

    For a genuine forward forecast there is no reason to hold 2026 back -- there is
    nothing left to score against. The 2024-25 / 2026 split is kept for the accuracy
    numbers reported elsewhere; here we want the best-informed model.
    """
    xs = MinMaxScaler().fit(hist[feats].values)
    ys = MinMaxScaler().fit(hist[["capacity_factor"]].values)
    est = model_module.make()
    est.fit(xs.transform(hist[feats].values), ys.transform(hist[["capacity_factor"]].values).ravel())
    return est, xs, ys


def _predict(est, xs, ys, future: pd.DataFrame, feats) -> np.ndarray:
    cf = ys.inverse_transform(est.predict(xs.transform(future[feats].values)).reshape(-1, 1)).ravel()
    return cf * future["capacity_MW"].values


def _monthly(dates: pd.Series, values: np.ndarray) -> list:
    s = pd.Series(values, index=pd.DatetimeIndex(dates))
    return [round(float(v), 1) for v in s.groupby(s.index.month).sum().reindex(range(1, 13)).values]


# ---------------------------------------------------------------- national forecast
def forecast_national(year: int) -> dict:
    hist = pd.read_csv(config.PROCESSED_DIR / "allindia_daily.csv", parse_dates=["date"])
    feats = dataset.FEATURES_WITH_TM
    climo = national_climatology()

    per_mode = {}
    for cap_mode in ("compound", "linear"):
        future = year_frame(year, climo, cap_mode)
        models = {}
        for mod in MODELS:
            est, xs, ys = _fit(mod, hist, feats)
            pred = _predict(est, xs, ys, future, feats)
            models[mod.NAME] = {
                "monthly_MU": _monthly(future["date"], pred),
                "annual_MU": round(float(pred.sum()), 1),
            }
        per_mode[cap_mode] = {
            "models": models,
            "capacity_MW": [int(round(v)) for v in
                            future.groupby(future["date"].dt.month)["capacity_MW"].mean().values],
            "year_end_capacity_MW": int(round(future["capacity_MW"].iloc[-1])),
        }

    # consensus envelope across the six models, on the central (compound) capacity path
    central = per_mode["compound"]["models"]
    stack = np.array([m["monthly_MU"] for m in central.values()])
    consensus = {
        "min_MU": [round(float(v), 1) for v in stack.min(axis=0)],
        "max_MU": [round(float(v), 1) for v in stack.max(axis=0)],
        "mean_MU": [round(float(v), 1) for v in stack.mean(axis=0)],
    }
    return {
        "year": year, "months": MONTHS,
        "capacity_paths": {k: {"year_end_capacity_MW": v["year_end_capacity_MW"],
                               "capacity_MW": v["capacity_MW"],
                               "models": v["models"]} for k, v in per_mode.items()},
        "consensus": consensus,
        "history_rows": int(len(hist)),
        "history_range": [hist["date"].min().strftime("%Y-%m-%d"), hist["date"].max().strftime("%Y-%m-%d")],
    }


# ---------------------------------------------------------------- per-region forecast
def forecast_regions(year: int, model_module=gpr_model) -> dict:
    hubs = regions_mod.hub_frames()
    gen = regions_mod.region_generation()
    feats = dataset.FEATURES_WITH_TM
    out = {}
    for code, label in config.REGIONS.items():
        hist = regions_mod.build(code, hubs, gen)
        if len(hist) < 60:
            continue
        climo = climatology(regions_mod.region_weather(hubs, code))
        future = year_frame(year, climo, "compound", region=code)
        est, xs, ys = _fit(model_module, hist, feats)
        pred = _predict(est, xs, ys, future, feats)
        out[code] = {"label": label, "monthly_MU": _monthly(future["date"], pred),
                     "annual_MU": round(float(pred.sum()), 1)}
    return out


# ---------------------------------------------------------------- entry point
def compute(year: int = 2027) -> dict:
    import external_benchmarks
    nat = forecast_national(year)
    nat["regions"] = forecast_regions(year)
    nat["benchmarks"] = external_benchmarks.benchmarks(
        year, nat["capacity_paths"]["compound"]["capacity_MW"], national_climatology())
    nat["caveat"] = (
        "2027 is a SCENARIO, not a measured forecast. The weather is a typical-year "
        "(climatology) profile built from 2024-2026 medians, and installed capacity is "
        "projected beyond the published table. The 95.0% day-level accuracy measured on "
        "2026 does not transfer to 2027, because in 2026 the model was given the weather "
        "that actually happened."
    )
    return nat


def write_json(res: dict):
    import json
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FUTURE_JSON.write_text(json.dumps(res))
    return FUTURE_JSON


def main():
    res = compute(2027)
    write_json(res)
    print("=" * 74)
    print(f"{res['year']} SCENARIO — climatology weather + projected capacity")
    print("=" * 74)
    print(f"  trained on {res['history_rows']} days, {res['history_range'][0]} -> {res['history_range'][1]}\n")
    central = res["capacity_paths"]["compound"]["models"]
    rows = [{"model": k, "annual_MU": v["annual_MU"]} for k, v in central.items()]
    df = pd.DataFrame(rows).sort_values("annual_MU")
    df["annual_MU"] = df["annual_MU"].map(lambda v: f"{v:,.0f}")
    print("Our six models (compound capacity path):")
    print(df.to_string(index=False))
    lo = res["capacity_paths"]["linear"]["models"][config.PRODUCTION_MODEL]["annual_MU"]
    hi = central[config.PRODUCTION_MODEL]["annual_MU"]
    print(f"\n{config.PRODUCTION_MODEL} annual total: {lo:,.0f} MU (low-growth fleet) "
          f"to {hi:,.0f} MU (recent-growth fleet)")
    print("\nIndependent benchmarks:")
    for b in res["benchmarks"]["sources"]:
        print(f"  {b['name']:34s} {b['annual_MU']:>12,.0f} MU   {b['note']}")
    print(f"\nSaved: {FUTURE_JSON}")
    print("\n" + res["caveat"])


if __name__ == "__main__":
    main()
