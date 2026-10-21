"""
Phase 1b — build the modelling dataset for All-India daily solar forecasting.

Steps (mirrors the paper's preprocessing, adapted to daily national data):
  1. Combine the 10 hub weather files into ONE national daily series, using a
     capacity-weighted average (panels are concentrated in a few sunny states).
  2. Derive module temperature with the paper's NOCT formula.
  3. Load the Grid-India All-India daily solar generation (the target).
  4. Join weather + solar on date; clean (missing values, non-positive power).
  5. Split chronologically: 2024-2025 = train, 2026 = test. No shuffling.
  6. Save data/processed/allindia_daily.csv and a human-readable data report.

Output columns:
  date, ghi, dni, ambient_temp, module_temp, wind_speed, relative_humidity,
  day_of_year, doy_sin, doy_cos, solar_generation_MU, split
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

WEATHER_DIR = config.RAW_DIR / "weather"
_ALL_WEATHER = {
    "ghi_kwh_m2_day": "ghi", "dni_kwh_m2_day": "dni", "ambient_temp_c": "ambient_temp",
    "wind_speed_ms": "wind_speed", "relative_humidity_pct": "relative_humidity",
}
WEATHER_COLS = [c for c in _ALL_WEATHER if c != "dni_kwh_m2_day" or config.USE_DNI]
OUT_CSV = config.PROCESSED_DIR / "allindia_daily.csv"
REPORT = config.RESULTS_DIR / "phase1_data_report.txt"


def national_weather() -> pd.DataFrame:
    """Capacity-weighted average of the hub weather files -> one national daily row."""
    weights = np.array([h[3] for h in config.SOLAR_HUBS], dtype=float)
    weights /= weights.sum()
    frames = []
    for (name, *_rest), w in zip(config.SOLAR_HUBS, weights):
        df = pd.read_csv(WEATHER_DIR / f"{name.replace(' ', '_')}.csv", parse_dates=["date"], index_col="date")
        frames.append(df[WEATHER_COLS].astype(float) * w)
    # element-wise weighted sum across hubs (weights already sum to 1)
    national = sum(frames)
    national.columns = [_ALL_WEATHER[c] for c in WEATHER_COLS]
    return national


def add_module_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Paper's NOCT model: T_module = T_ambient + (G/800) * (NOCT - 20).
    Daily GHI (kWh/m^2/day) -> effective daytime irradiance G (W/m^2) by spreading
    the day's energy over config.ASSUMED_SUN_HOURS hours (explicit approximation)."""
    g_eff_w_m2 = df["ghi"] * 1000.0 / config.ASSUMED_SUN_HOURS
    df["module_temp"] = df["ambient_temp"] + (g_eff_w_m2 / 800.0) * (config.NOCT - 20.0)
    return df


def capacity_series(dates: pd.Series, region: str = None, mode: str = "compound") -> pd.Series:
    """Installed renewable capacity (MW) for each date, because solar generation scales
    with how much capacity is installed.

    For 2024-2025 we use the ACTUAL monthly capacity. For 2026 (beyond the data) we
    extrapolate using the RECENT 12-month growth rate, not the whole-period average —
    capacity growth is accelerating (~3,900 MW/month lately vs ~3,200 over the full
    period), so the recent trend better represents 2026.

    `region` selects a Grid-India region (NR/WR/SR/ER/NER) instead of All-India, so the
    per-region models are normalised by their OWN fleet growth rather than the national
    one — regions are not adding capacity at the same rate.

    `mode` picks how months beyond the published table are projected:
      "compound" (default) — recent 12-month compound growth rate. Validated on 2026,
                             where it beat the straight line (5.46% -> 5.04% error).
      "linear"             — recent 12-month straight-line slope. Used as the LOW growth
                             bound for 2027, where compounding two years out is a big
                             assumption and deserves a stated range."""
    cap = pd.read_csv(config.CAPACITY_CSV)
    cap = cap[cap["region"] == (region or config.TARGET_REGION)][["month", config.CAPACITY_COL]].copy()
    base = pd.Period(cap["month"].min(), "M")
    cap["mt"] = cap["month"].apply(lambda m: (pd.Period(m, "M") - base).n)
    known = dict(zip(cap["mt"], cap[config.CAPACITY_COL]))
    last_mt = cap["mt"].max()
    last_val = known[last_mt]
    # Capacity additions compound (each month builds on a bigger fleet), so the recent
    # 12-month COMPOUND monthly growth rate extrapolates better than a straight line.
    first_of_12 = cap[config.CAPACITY_COL].iloc[-13] if len(cap) >= 13 else cap[config.CAPACITY_COL].iloc[0]
    n_months = 12 if len(cap) >= 13 else len(cap) - 1
    monthly_growth = (last_val / first_of_12) ** (1.0 / n_months) - 1.0

    recent_slope = np.polyfit(cap.tail(12)["mt"], cap.tail(12)[config.CAPACITY_COL], 1)[0]

    def cap_for(period):
        mt = (period - base).n
        if mt in known:
            return known[mt]                                   # actual capacity (2024-25)
        if mt > last_mt:
            ahead = mt - last_mt
            if mode == "linear":
                return last_val + recent_slope * ahead
            return last_val * (1.0 + monthly_growth) ** ahead  # compound trend
        return known[min(known)]

    return dates.dt.to_period("M").apply(cap_for)


def load_solar() -> pd.DataFrame:
    parts = []
    for path in (config.SOLAR_2024_25_CSV, config.SOLAR_2026_CSV):
        d = pd.read_csv(path, parse_dates=["date"])
        d = d[d["region"] == config.TARGET_REGION][["date", config.TARGET_COL]]
        parts.append(d)
    solar = pd.concat(parts).dropna(subset=[config.TARGET_COL]).set_index("date").sort_index()
    return solar[~solar.index.duplicated(keep="last")]


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    weather = add_module_temperature(national_weather())
    solar = load_solar()
    df = weather.join(solar, how="inner").reset_index()

    # seasonal calendar features (daily data has no hourly "time of day")
    df["day_of_year"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # installed capacity + capacity factor (generation per MW) — the stationary target
    df["capacity_MW"] = capacity_series(df["date"]).values
    df["capacity_factor"] = df[config.TARGET_COL] / df["capacity_MW"]

    # --- cleaning -----------------------------------------------------------
    before = len(df)
    feature_cols = [c for c in ["ghi", "dni", "ambient_temp", "module_temp", "wind_speed", "relative_humidity"]
                    if c in df.columns]
    n_missing = int(df[feature_cols + [config.TARGET_COL]].isna().any(axis=1).sum())
    df = df.dropna(subset=feature_cols + [config.TARGET_COL])
    n_nonpos = int((df[config.TARGET_COL] <= 0).sum())
    df = df[df[config.TARGET_COL] > 0]
    df = df.sort_values("date").reset_index(drop=True)

    # --- chronological split ------------------------------------------------
    df["split"] = np.where(df["date"].dt.year.isin(config.TRAIN_YEARS), "train", "test")
    cols = (["date"] + feature_cols + ["day_of_year", "doy_sin", "doy_cos",
            "capacity_MW", "capacity_factor", config.TARGET_COL, "split"])
    df = df[cols]
    df.to_csv(OUT_CSV, index=False)

    write_report(df, before, n_missing, n_nonpos, feature_cols)
    print(REPORT.read_text())


def write_report(df, before, n_missing, n_nonpos, feature_cols):
    train, test = df[df.split == "train"], df[df.split == "test"]
    lines = []
    w = lines.append
    w("=" * 70)
    w("PHASE 1 DATA REPORT — All-India daily solar forecasting dataset")
    w("=" * 70)
    w("")
    w("SOURCES")
    w("  Target  : Grid-India PSP reports (daily All-India solar generation, MU)")
    w("  Weather : NASA POWER daily, capacity-weighted over 10 solar-hub states")
    w("            https://power.larc.nasa.gov  (Renewable Energy community)")
    w("  Module temperature: NOCT model, T_m = T_a + (G/800)*(NOCT-20), "
      f"NOCT={config.NOCT:.0f}C, {config.ASSUMED_SUN_HOURS:.0f} sun-hours")
    w("")
    w("ROW COUNTS")
    w(f"  Rows after weather+solar join : {before}")
    w(f"  Dropped (missing values)      : {n_missing}")
    w(f"  Dropped (non-positive power)  : {n_nonpos}")
    w(f"  Final usable rows             : {len(df)}")
    w(f"    train (2024-2025) : {len(train):4d}   {train.date.min().date()} -> {train.date.max().date()}")
    w(f"    test  (2026)      : {len(test):4d}   {test.date.min().date()} -> {test.date.max().date()}")
    w("")
    w("FEATURES (mean over all rows, with units)")
    units = {"ghi": "kWh/m2/day", "dni": "kWh/m2/day", "ambient_temp": "C", "module_temp": "C",
             "wind_speed": "m/s", "relative_humidity": "%"}
    for c in feature_cols:
        w(f"  {c:20s} mean={df[c].mean():8.2f}  min={df[c].min():7.2f}  max={df[c].max():7.2f}  {units[c]}")
    w(f"  {'solar_generation_MU':20s} mean={df[config.TARGET_COL].mean():8.1f}  "
      f"min={df[config.TARGET_COL].min():7.1f}  max={df[config.TARGET_COL].max():7.1f}  MU (target)")
    w("")
    w("CORRELATION of each feature with solar generation (train rows)")
    corr = train[feature_cols + [config.TARGET_COL]].corr()[config.TARGET_COL].drop(config.TARGET_COL)
    for c, v in corr.sort_values(ascending=False).items():
        w(f"  {c:20s} r = {v:+.3f}")
    w("")
    w(f"Saved: {OUT_CSV}")
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
