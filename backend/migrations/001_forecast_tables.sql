CREATE TABLE IF NOT EXISTS solar_forecast (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    year                 INTEGER NOT NULL,
    month                INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    region               TEXT    NOT NULL,
    predicted_generation REAL,            -- monthly solar generation, MU
    actual_generation    REAL,            -- NULL until an administrator adds it
    difference           REAL,            -- actual - predicted (MU)
    percentage_error     REAL,            -- (actual - predicted) / actual * 100
    model_name           TEXT,
    created_at           TEXT    NOT NULL,
    updated_at           TEXT    NOT NULL,
    UNIQUE (year, month, region)
);

CREATE INDEX IF NOT EXISTS idx_solar_forecast_year_region ON solar_forecast (year, region);

CREATE TABLE IF NOT EXISTS model_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name              TEXT    NOT NULL,
    trained_at              TEXT    NOT NULL,
    n_training_rows         INTEGER NOT NULL,
    n_validation_rows       INTEGER NOT NULL,
    validation_metrics_json TEXT    NOT NULL,   -- metrics of the selected model
    candidates_json         TEXT    NOT NULL,   -- metrics of every candidate model
    feature_config_json     TEXT    NOT NULL,
    warnings_json           TEXT    NOT NULL,
    artifact_path           TEXT    NOT NULL,
    is_active               INTEGER NOT NULL DEFAULT 1
);
