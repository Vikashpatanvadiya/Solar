"""
Per-region daily solar forecasting (NR / WR / SR / ER / NER).

The headline model predicts All-India generation. The map view needs a *real* number
per region, so this module trains the same pipeline once per Grid-India region instead
of faking it by splitting the national figure.

For each region:
  weather  = weight-average of the config.SOLAR_HUBS that sit in that region
  target   = that region's daily solar generation (MU) from the Grid-India PSP reports
  scaling  = region generation / that REGION's own installed renewable capacity
             ("regional capacity factor"), from the same Grid-India monthly capacity
             table. Regions are not growing their fleets at the same rate, so each one
             is normalised by its own growth rather than the national average.
  split    = 2024-25 train, 2026 test (same chronological rule as the national model)

Run:  .venv/bin/python src/regions.py
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
import evaluate
import preprocess

warnings.filterwarnings("ignore")

WEATHER_DIR = config.RAW_DIR / "weather"
_W = preprocess._ALL_WEATHER
WEATHER_COLS = preprocess.WEATHER_COLS


def hub_frames() -> dict:
    """Each hub's raw daily weather, keyed by hub name (used by the map's day view)."""
    out = {}
    for name, lat, lon, weight, region in config.SOLAR_HUBS:
        df = pd.read_csv(WEATHER_DIR / f"{name.replace(' ', '_')}.csv",
                         parse_dates=["date"], index_col="date")[WEATHER_COLS].astype(float)
        df.columns = [_W[c] for c in WEATHER_COLS]
        df = preprocess.add_module_temperature(df)
        out[name] = {"lat": lat, "lon": lon, "weight": weight, "region": region, "weather": df}
    return out


def region_weather(hubs: dict, region: str) -> pd.DataFrame:
    """Weight-average of the hubs inside one region."""
    members = [(n, h) for n, h in hubs.items() if h["region"] == region]
    tot = sum(h["weight"] for _, h in members)
    frames = [h["weather"] * (h["weight"] / tot) for _, h in members]
    return sum(frames)


def region_generation() -> pd.DataFrame:
    """Daily solar generation per region, 2024-2026, from the PSP extractions."""
    parts = []
    for path in (config.SOLAR_2024_25_CSV, config.SOLAR_2026_CSV):
        d = pd.read_csv(path, parse_dates=["date"])
        parts.append(d[d["region"].isin(config.REGIONS)][["date", "region", config.TARGET_COL]])
    g = pd.concat(parts).dropna(subset=[config.TARGET_COL])
    return g.drop_duplicates(subset=["date", "region"], keep="last")


def build(region: str, hubs: dict, gen: pd.DataFrame) -> pd.DataFrame:
    w = region_weather(hubs, region)
    g = gen[gen["region"] == region].set_index("date")[[config.TARGET_COL]].sort_index()
    df = w.join(g, how="inner").reset_index()
    df["day_of_year"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["capacity_MW"] = preprocess.capacity_series(df["date"], region).values
    df["capacity_factor"] = df[config.TARGET_COL] / df["capacity_MW"]
    df = df.dropna(subset=dataset.FEATURES_WITH_TM + [config.TARGET_COL])
    df = df[df[config.TARGET_COL] > 0].sort_values("date").reset_index(drop=True)
    df["split"] = np.where(df["date"].dt.year.isin(config.TRAIN_YEARS), "train", "test")
    return df


def fit_region(df: pd.DataFrame, model_module):
    """Train on 2024-25, predict 2026. Returns (test_frame_with_prediction, metrics)."""
    feats = dataset.FEATURES_WITH_TM
    tr, te = df[df.split == "train"], df[df.split == "test"]
    if len(tr) < 60 or len(te) < 10:
        return None, None
    xs = MinMaxScaler().fit(tr[feats].values)
    ys = MinMaxScaler().fit(tr[["capacity_factor"]].values)
    est = model_module.make()
    est.fit(xs.transform(tr[feats].values), ys.transform(tr[["capacity_factor"]].values).ravel())
    cf = ys.inverse_transform(est.predict(xs.transform(te[feats].values)).reshape(-1, 1)).ravel()
    out = te.copy()
    out["predicted_MU"] = cf * te["capacity_MW"].values
    return out, evaluate.metrics(out[config.TARGET_COL].values, out["predicted_MU"].values)


def run(model_module) -> dict:
    """Train every region with the given model module. Returns a JSON-ready dict."""
    hubs = hub_frames()
    gen = region_generation()
    regions, series = {}, {}
    for code, label in config.REGIONS.items():
        df = build(code, hubs, gen)
        te, m = fit_region(df, model_module)
        if te is None:
            regions[code] = {"code": code, "label": label, "usable": False,
                             "reason": "not enough overlapping weather + generation days"}
            continue
        regions[code] = {
            "code": code, "label": label, "usable": True,
            "hubs": [n for n, h in hubs.items() if h["region"] == code],
            "train_days": int((df.split == "train").sum()), "test_days": int(len(te)),
            "accuracy": round(100 - m["MAPE"], 1), "MAPE": round(m["MAPE"], 2),
            "RMSE": round(m["RMSE"], 1), "MAE": round(m["MAE"], 1), "R2": round(m["R2"], 3),
            "total_actual_MU": round(float(te[config.TARGET_COL].sum()), 1),
            "total_predicted_MU": round(float(te["predicted_MU"].sum()), 1),
        }
        series[code] = {d.strftime("%Y-%m-%d"): [round(float(a), 1), round(float(p), 1)]
                        for d, a, p in zip(te["date"], te[config.TARGET_COL], te["predicted_MU"])}

    hub_meta = [{"name": n, "lat": h["lat"], "lon": h["lon"],
                 "weight": h["weight"], "region": h["region"]} for n, h in hubs.items()]
    weather = {}
    for n, h in hubs.items():
        w = h["weather"]
        weather[n] = {d.strftime("%Y-%m-%d"): [
            None if pd.isna(v) else round(float(v), 2)
            for v in (r["ghi"], r["ambient_temp"], r["module_temp"],
                      r["wind_speed"], r["relative_humidity"])]
            for d, r in w.iterrows()}
    return {
        "model": model_module.NAME,
        "regions": regions,
        "series": series,                  # region -> date -> [actual_MU, predicted_MU]
        "hubs": hub_meta,
        "hub_weather": weather,            # hub -> date -> [ghi, Ta, Tm, wind, RH]
        "weather_fields": ["ghi", "ambient_temp", "module_temp", "wind_speed", "relative_humidity"],
    }


REGIONS_JSON = config.RESULTS_DIR / "regions_data.json"


def compute(model_name: str = None) -> dict:
    """Run every region with the production model (by name, default GPR)."""
    from models import ann, ann_pso, decision_tree, gpr_model, linear, svm_model
    by_name = {m.NAME: m for m in (linear, decision_tree, svm_model, gpr_model, ann, ann_pso)}
    return run(by_name.get(model_name, gpr_model))


def write_json(res: dict):
    import json
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REGIONS_JSON.write_text(json.dumps(res))
    return REGIONS_JSON


def main():
    res = compute()
    write_json(res)
    print("=" * 66)
    print(f"PER-REGION FORECAST — {res['model']}, train 2024-25, test 2026")
    print("=" * 66)
    rows = [r for r in res["regions"].values() if r["usable"]]
    out = pd.DataFrame(rows)[["code", "label", "test_days", "accuracy", "MAPE", "RMSE", "R2",
                              "total_actual_MU", "total_predicted_MU"]]
    print(out.to_string(index=False))
    for r in res["regions"].values():
        if not r["usable"]:
            print(f"  {r['code']}: skipped - {r['reason']}")
    print(f"Saved: {REGIONS_JSON}")


if __name__ == "__main__":
    main()
