import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_CSV = ROOT / "data" / "daily_regional.csv"
SITE_DIR = ROOT / "site"
MIGRATIONS_DIR = ROOT / "backend" / "migrations"
ARTIFACT_DIR = ROOT / "backend" / "artifacts"
DB_PATH = Path(os.environ.get("SOLAR_DB_PATH", ROOT / "backend" / "storage" / "solar_forecast.db"))

MODEL_REGIONS = ["NR", "WR", "SR", "ER", "NER"]
ALL_INDIA = "ALL_INDIA"
API_REGIONS = MODEL_REGIONS + [ALL_INDIA]

FORECAST_YEAR = 2026
VALIDATION_MONTHS = 4
MIN_TRAIN_ROWS = 20
MIN_VALIDATION_ROWS = 5
MAX_ADMIN_RECORDS = 72
