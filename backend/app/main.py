import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, db
from .ai import data_aggregation, forecasting, storage, training
from .auth import require_admin

Region = Literal["NR", "WR", "SR", "ER", "NER", "ALL_INDIA"]
_train_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.migrate()
    yield


app = FastAPI(title="Solar Generation AI Forecast", lifespan=lifespan)


def _check_year(year: int) -> None:
    if year != config.FORECAST_YEAR:
        raise HTTPException(status_code=404, detail=f"Only the {config.FORECAST_YEAR} forecast is supported.")


class ActualRecord(BaseModel):
    month: int = Field(ge=1, le=12)
    region: Region
    actual_generation: float = Field(ge=0, allow_inf_nan=False, description="Monthly solar generation in MU")


class ActualPayload(BaseModel):
    records: list[ActualRecord] = Field(min_length=1, max_length=config.MAX_ADMIN_RECORDS)


@app.post("/api/ai/train")
def train_model():
    if not _train_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Training is already running.")
    try:
        result = training.train()
    except training.InsufficientDataError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        _train_lock.release()
    storage.save_model_run(result)
    return {
        "model_name": result["model_name"],
        "validation_metrics": result["validation_metrics"],
        "candidates": result["candidates"],
        "n_training_rows": result["n_training_rows"],
        "n_validation_rows": result["n_validation_rows"],
        "warnings": result["warnings"],
    }


@app.post("/api/ai/forecast/{year}")
def generate_forecast(year: int):
    _check_year(year)
    if storage.get_active_run() is None:
        raise HTTPException(status_code=409, detail="No trained model found. Train the model first.")
    try:
        bundle = forecasting.load_bundle()
        rows = forecasting.generate(data_aggregation.monthly_by_region(), bundle, year)
    except forecasting.ModelNotTrainedError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except forecasting.ForecastError as e:
        raise HTTPException(status_code=422, detail=str(e))
    n = storage.upsert_predictions(rows, bundle["model_name"])
    return {"year": year, "model_name": bundle["model_name"], "rows_written": n}


@app.get("/api/ai/forecast/{year}")
def read_forecast(year: int, region: Region | None = None):
    _check_year(year)
    rows = storage.get_forecast(year, region)
    predicted = [r for r in rows if r["predicted_generation"] is not None]
    return {
        "year": year,
        "generated": bool(predicted),
        "model_name": predicted[0]["model_name"] if predicted else None,
        "rows": rows,
    }


@app.get("/api/ai/comparison/{year}")
def read_comparison(year: int, region: Region = "ALL_INDIA"):
    _check_year(year)
    return storage.comparison(year, region)


@app.get("/api/ai/metrics")
def read_metrics():
    run = storage.get_active_run()
    if not run:
        return {"trained": False}
    return {"trained": True, **run}


@app.post("/api/admin/actual-generation/{year}", dependencies=[Depends(require_admin)])
def add_actual_generation(year: int, payload: ActualPayload):
    _check_year(year)
    seen = set()
    for r in payload.records:
        key = (r.month, r.region)
        if key in seen:
            raise HTTPException(status_code=422, detail=f"Duplicate record for month {r.month}, region {r.region}.")
        seen.add(key)
    result = storage.set_actuals(year, [r.model_dump() for r in payload.records])
    return {"year": year, **result}


@app.get("/api/results")
@app.get("/api/ai/models")
def get_prediction_models():
    """All 6 AI models benchmark results and daily predictions for 2026."""
    import json
    json_path = config.ROOT / "solar-prediction" / "results" / "dashboard_data.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Prediction models data not found. Run solar-prediction/src/run.py first.")
    return json.loads(json_path.read_text())


@app.get("/api/data")
@app.get("/api/ai/weather")
def get_weather_and_solar_data():
    """Daily All-India solar generation joined with NASA POWER weather data (2024-2026)."""
    import csv
    csv_path = config.ROOT / "solar-prediction" / "data" / "processed" / "allindia_daily.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Run solar-prediction/src/preprocess.py first.")
    rows = []
    years_set = set()
    summary = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            y = int(r["date"].split("-")[0])
            years_set.add(y)
            solar = round(float(r["solar_generation_MU"]), 1) if r.get("solar_generation_MU") else 0.0
            row_dict = {
                "date": r["date"],
                "year": y,
                "solar_generation_MU": solar,
                "ghi": round(float(r["ghi"]), 2),
                "ambient_temp": round(float(r["ambient_temp"]), 1),
                "module_temp": round(float(r["module_temp"]), 1),
                "wind_speed": round(float(r["wind_speed"]), 2),
                "relative_humidity": round(float(r["relative_humidity"]), 1),
                "capacity_MW": int(round(float(r["capacity_MW"]))),
                "split": r.get("split", "train"),
            }
            rows.append(row_dict)
            if str(y) not in summary:
                summary[str(y)] = {"days": 0, "total_solar_MU": 0.0, "min_date": r["date"], "max_date": r["date"], "role": "test" if y == 2026 else "train"}
            s = summary[str(y)]
            s["days"] += 1
            s["total_solar_MU"] += solar
            s["max_date"] = r["date"]

    for y_str, s in summary.items():
        s["total_solar_MU"] = round(s["total_solar_MU"], 0)
        s["avg_daily_MU"] = round(s["total_solar_MU"] / s["days"], 1) if s["days"] > 0 else 0
        s["date_range"] = [s["min_date"], s["max_date"]]

    return {"rows": rows, "years": sorted(list(years_set)), "summary": summary}


LAB_DIR = config.ROOT / "solar-prediction" / "web"
if LAB_DIR.exists():
    app.mount("/lab", StaticFiles(directory=LAB_DIR, html=True), name="lab")

if config.SITE_DIR.exists():
    app.mount("/", StaticFiles(directory=config.SITE_DIR, html=True), name="site")
