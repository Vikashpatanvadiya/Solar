"""
Phase 1a — fetch daily weather for India's solar hubs from NASA POWER.

NASA POWER (Prediction Of Worldwide Energy Resources), NASA Langley Research Center.
Free, no API key. https://power.larc.nasa.gov  (Renewable Energy community)

One request per hub (see config.SOLAR_HUBS) covering the whole study period. Each
hub's response is cached as a tidy CSV under data/raw/weather/, so re-running is free
and offline-friendly. NASA's missing-value sentinel (-999) is converted to NaN.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

OUT_DIR = config.RAW_DIR / "weather"
START = "20240101"   # a little before the solar data (Apr 2024) for lag/rolling features
END = "20261021"
RENAME = {
    "ALLSKY_SFC_SW_DWN": "ghi_kwh_m2_day",
    "ALLSKY_SFC_SW_DNI": "dni_kwh_m2_day",
    "T2M": "ambient_temp_c",
    "WS10M": "wind_speed_ms",
    "RH2M": "relative_humidity_pct",
}


def fetch_hub(name, lat, lon) -> pd.DataFrame:
    url = (
        f"{config.NASA_POWER_URL}?parameters={','.join(config.NASA_PARAMS)}"
        f"&community={config.NASA_COMMUNITY}&longitude={lon}&latitude={lat}"
        f"&start={START}&end={END}&format=JSON"
    )
    raw = subprocess.run(["curl", "-fsSL", "--max-time", "90", url], capture_output=True, check=True).stdout
    params = json.loads(raw)["properties"]["parameter"]
    df = pd.DataFrame(params).rename(columns=RENAME)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    return df.sort_index().replace(-999.0, pd.NA)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, lat, lon, *_ in config.SOLAR_HUBS:
        dest = OUT_DIR / f"{name.replace(' ', '_')}.csv"
        if dest.exists():
            have = pd.read_csv(dest, parse_dates=["date"])["date"].max()
            if have >= pd.Timestamp(END):
                print(f"  cached   {name}")
                continue
            print(f"  stale    {name} (cached to {have.date()}, want {END}) - refetching")
        df = fetch_hub(name, lat, lon)
        df.to_csv(dest)
        print(f"  fetched  {name:16s} {df.index.min().date()} -> {df.index.max().date()}  ({len(df)} days)")
    print(f"Done. {len(config.SOLAR_HUBS)} hubs in {OUT_DIR}")


if __name__ == "__main__":
    main()
