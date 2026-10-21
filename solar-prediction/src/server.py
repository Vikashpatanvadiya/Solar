"""
Dashboard server for the solar forecasting models.

  GET  /                 -> the dashboard (web/index.html)
  GET  /api/results      -> latest results JSON (trains once if none cached)
  GET  /api/regions      -> per-region forecasts + hub weather, for the map view
  GET  /api/forecast/{year}   -> forward scenario for a year that has not happened yet
  POST /api/forecast/{year}   -> recompute that scenario
  GET  /api/comparison/{year} -> our models vs the independent benchmarks, with divergence
  POST /api/train        -> retrain all 6 models (+ regions) and return fresh results

Run:  .venv/bin/python -m uvicorn src.server:app --port 8010
  (or:  .venv/bin/python src/server.py )
"""
import json
import sys
import threading
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import forecast_future
import regions as regions_mod
import run

WEB_DIR = config.ROOT / "web"
PROCESSED = config.PROCESSED_DIR / "allindia_daily.csv"
_lock = threading.Lock()
app = FastAPI(title="Solar Forecast Dashboard")


@app.get("/api/results")
def get_results():
    if run.JSON_OUT.exists():
        return json.loads(run.JSON_OUT.read_text())
    return train()


@app.get("/api/data")
def get_data():
    """The real fetched data for 2024-2026: daily solar generation (Grid-India) +
    weather (NASA POWER), used to train and test the models."""
    if not PROCESSED.exists():
        raise HTTPException(status_code=404, detail="Processed data missing. Run src/preprocess.py first.")
    df = pd.read_csv(PROCESSED, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    cols = ["date", "year", "solar_generation_MU", "ghi", "ambient_temp", "module_temp",
            "wind_speed", "relative_humidity", "capacity_MW", "split"]
    rows = []
    for _, r in df[cols].iterrows():
        rows.append({
            "date": r["date"].strftime("%Y-%m-%d"), "year": int(r["year"]),
            "solar_generation_MU": round(float(r["solar_generation_MU"]), 1),
            "ghi": round(float(r["ghi"]), 2), "ambient_temp": round(float(r["ambient_temp"]), 1),
            "module_temp": round(float(r["module_temp"]), 1), "wind_speed": round(float(r["wind_speed"]), 2),
            "relative_humidity": round(float(r["relative_humidity"]), 1),
            "capacity_MW": int(round(float(r["capacity_MW"]))), "split": r["split"],
        })
    years = sorted(df["year"].unique().tolist())
    summary = {}
    for y in years:
        d = df[df.year == y]
        summary[str(y)] = {
            "days": int(len(d)),
            "date_range": [d["date"].min().strftime("%Y-%m-%d"), d["date"].max().strftime("%Y-%m-%d")],
            "total_solar_MU": round(float(d["solar_generation_MU"].sum()), 0),
            "avg_daily_MU": round(float(d["solar_generation_MU"].mean()), 1),
            "role": "train" if y in config.TRAIN_YEARS else "test",
        }
    return {"rows": rows, "years": years, "summary": summary}


@app.get("/api/regions")
def get_regions():
    """Per-region actual vs predicted + each solar hub's daily weather (map view)."""
    if regions_mod.REGIONS_JSON.exists():
        return json.loads(regions_mod.REGIONS_JSON.read_text())
    res = regions_mod.compute()
    regions_mod.write_json(res)
    return res


def _check_year(year: int):
    """Only years we can actually build a scenario for. 2024-2026 are history and are
    served by /api/results instead."""
    if year not in config.FORECAST_YEARS:
        raise HTTPException(
            status_code=400,
            detail=(f"Forward forecasting is available for {config.FORECAST_YEARS}. "
                    f"{year} is either historical (use /api/results) or too far ahead to "
                    f"project capacity for honestly."))


def _future(year: int, recompute: bool = False) -> dict:
    _check_year(year)
    if not recompute and forecast_future.FUTURE_JSON.exists():
        cached = json.loads(forecast_future.FUTURE_JSON.read_text())
        if cached.get("year") == year:
            return cached
    if not _lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A training run is already in progress.")
    try:
        res = forecast_future.compute(year)
        forecast_future.write_json(res)
        return res
    finally:
        _lock.release()


@app.get("/api/forecast/{year}")
def get_forecast(year: int):
    return _future(year)


@app.post("/api/forecast/{year}")
def post_forecast(year: int):
    return _future(year, recompute=True)


@app.get("/api/comparison/{year}")
def get_comparison(year: int):
    """Our forecast next to every independent benchmark, month by month, with the
    divergence between them. Divergence is signed: positive means our model is higher."""
    import external_benchmarks
    f = _future(year)
    path = f["capacity_paths"]["compound"]
    ours = path["models"][config.PRODUCTION_MODEL]["monthly_MU"]
    annual_ours = path["models"][config.PRODUCTION_MODEL]["annual_MU"]

    comparisons = []
    for b in f["benchmarks"]["sources"]:
        monthly = [
            {"month": m, "ours_MU": o, "benchmark_MU": c,
             "difference_MU": round(o - c, 1),
             "divergence_pct": round((o - c) / c * 100, 2) if c else None}
            for m, o, c in zip(f["months"], ours, b["monthly_MU"])
        ]
        comparisons.append({
            "key": b["key"], "name": b["name"], "note": b["note"],
            "annual_benchmark_MU": b["annual_MU"],
            "annual_difference_MU": round(annual_ours - b["annual_MU"], 1),
            "annual_divergence_pct": round((annual_ours - b["annual_MU"]) / b["annual_MU"] * 100, 2),
            "monthly": monthly,
        })
    return {
        "year": year, "our_model": config.PRODUCTION_MODEL,
        "our_monthly_MU": ours, "our_annual_MU": annual_ours,
        "consensus": f["consensus"],
        "capacity_range_MU": {
            "low_growth": f["capacity_paths"]["linear"]["models"][config.PRODUCTION_MODEL]["annual_MU"],
            "recent_growth": annual_ours,
        },
        "comparisons": comparisons,
        "benchmark_validation": external_benchmarks.validate_on_2026(),
        "assumptions": f["benchmarks"]["assumptions"],
        "notes": f["benchmarks"]["notes"],
        "caveat": f["caveat"],
    }


@app.post("/api/train")
def train():
    if not _lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Training already in progress.")
    try:
        results = run.compute_results()
        run.write_json(results)
        regions_mod.write_json(regions_mod.compute(results["best_model"]))
        # the forward scenario is built from these same models, so it is now stale.
        # Rebuilt directly (not via _future) because this thread already holds the lock.
        for year in config.FORECAST_YEARS:
            forecast_future.write_json(forecast_future.compute(year))
        return results
    finally:
        _lock.release()


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8010, reload=False)
