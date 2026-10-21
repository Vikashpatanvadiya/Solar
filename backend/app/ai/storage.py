import json

from .. import config
from ..db import connect, utc_now
from . import evaluation

_FORECAST_COLS = (
    "year, month, region, predicted_generation, actual_generation, difference, "
    "percentage_error, model_name, created_at, updated_at"
)


def upsert_predictions(rows: list[dict], model_name: str) -> int:
    """Insert or refresh predicted values. Any actuals already stored are kept, and the
    difference / percentage error are recomputed against the new prediction."""
    now = utc_now()
    with connect() as conn:
        for r in rows:
            existing = conn.execute(
                "SELECT actual_generation FROM solar_forecast WHERE year=? AND month=? AND region=?",
                (r["year"], r["month"], r["region"]),
            ).fetchone()
            actual = existing["actual_generation"] if existing else None
            diff, pct = evaluation.difference_and_pct(actual, r["predicted_generation"])
            if existing:
                conn.execute(
                    "UPDATE solar_forecast SET predicted_generation=?, difference=?, percentage_error=?, "
                    "model_name=?, updated_at=? WHERE year=? AND month=? AND region=?",
                    (r["predicted_generation"], diff, pct, model_name, now, r["year"], r["month"], r["region"]),
                )
            else:
                conn.execute(
                    "INSERT INTO solar_forecast (year, month, region, predicted_generation, model_name, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (r["year"], r["month"], r["region"], r["predicted_generation"], model_name, now, now),
                )
    return len(rows)


def set_actuals(year: int, records: list[dict]) -> dict:
    """Store administrator-supplied actuals and recompute difference / percentage error.
    When a month has all five regional actuals but no explicit ALL_INDIA value in this
    request, ALL_INDIA is derived as their sum."""
    now = utc_now()
    supplied = {(r["month"], r["region"]): r["actual_generation"] for r in records}
    with connect() as conn:
        for (month, region), value in supplied.items():
            _write_actual(conn, year, month, region, value, now)
        derived = []
        for month in sorted({m for m, _ in supplied}):
            if (month, config.ALL_INDIA) in supplied:
                continue
            vals = [
                conn.execute(
                    "SELECT actual_generation FROM solar_forecast WHERE year=? AND month=? AND region=?",
                    (year, month, r),
                ).fetchone()
                for r in config.MODEL_REGIONS
            ]
            if all(v is not None and v["actual_generation"] is not None for v in vals):
                _write_actual(conn, year, month, config.ALL_INDIA, sum(v["actual_generation"] for v in vals), now)
                derived.append(month)
    return {"stored": len(supplied), "all_india_derived_for_months": derived}


def _write_actual(conn, year, month, region, value, now) -> None:
    row = conn.execute(
        "SELECT predicted_generation FROM solar_forecast WHERE year=? AND month=? AND region=?",
        (year, month, region),
    ).fetchone()
    predicted = row["predicted_generation"] if row else None
    diff, pct = evaluation.difference_and_pct(value, predicted)
    if row:
        conn.execute(
            "UPDATE solar_forecast SET actual_generation=?, difference=?, percentage_error=?, updated_at=? "
            "WHERE year=? AND month=? AND region=?",
            (value, diff, pct, now, year, month, region),
        )
    else:
        conn.execute(
            "INSERT INTO solar_forecast (year, month, region, actual_generation, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (year, month, region, value, now, now),
        )


def get_forecast(year: int, region: str | None = None) -> list[dict]:
    sql = f"SELECT {_FORECAST_COLS} FROM solar_forecast WHERE year=?"
    args: list = [year]
    if region:
        sql += " AND region=?"
        args.append(region)
    sql += " ORDER BY region, month"
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, args)]


def comparison(year: int, region: str) -> dict:
    rows = get_forecast(year, region)
    paired = [r for r in rows if r["predicted_generation"] is not None and r["actual_generation"] is not None]
    live = evaluation.compute_metrics(
        [r["actual_generation"] for r in paired], [r["predicted_generation"] for r in paired]
    )
    predicted_total = sum(r["predicted_generation"] for r in rows if r["predicted_generation"] is not None)
    return {
        "year": year,
        "region": region,
        "rows": rows,
        "summary": {
            "months_with_actual": len(paired),
            "predicted_total_all_months": predicted_total if any(r["predicted_generation"] is not None for r in rows) else None,
            "predicted_total_matched_months": sum(r["predicted_generation"] for r in paired) if paired else None,
            "actual_total": sum(r["actual_generation"] for r in paired) if paired else None,
            "difference_total": sum(r["difference"] for r in paired) if paired else None,
            "live_metrics": live if paired else None,
        },
    }


def save_model_run(result: dict) -> int:
    with connect() as conn:
        conn.execute("UPDATE model_runs SET is_active=0")
        cur = conn.execute(
            "INSERT INTO model_runs (model_name, trained_at, n_training_rows, n_validation_rows, "
            "validation_metrics_json, candidates_json, feature_config_json, warnings_json, artifact_path, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (
                result["model_name"],
                result["feature_config"]["trained_at"],
                result["n_training_rows"],
                result["n_validation_rows"],
                json.dumps(result["validation_metrics"]),
                json.dumps(result["candidates"]),
                json.dumps(result["feature_config"]),
                json.dumps(result["warnings"]),
                result["artifact_path"],
            ),
        )
        return cur.lastrowid


def get_active_run() -> dict | None:
    with connect() as conn:
        r = conn.execute("SELECT * FROM model_runs WHERE is_active=1 ORDER BY id DESC LIMIT 1").fetchone()
    if not r:
        return None
    return {
        "id": r["id"],
        "model_name": r["model_name"],
        "trained_at": r["trained_at"],
        "n_training_rows": r["n_training_rows"],
        "n_validation_rows": r["n_validation_rows"],
        "validation_metrics": json.loads(r["validation_metrics_json"]),
        "candidates": json.loads(r["candidates_json"]),
        "feature_config": json.loads(r["feature_config_json"]),
        "warnings": json.loads(r["warnings_json"]),
    }
