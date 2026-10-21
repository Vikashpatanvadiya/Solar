"""
Project configuration for the All-India solar power forecasting model.

Everything that is a *choice* (data sources, locations, weights, the physics
constant for module temperature) lives here so it is transparent and easy to cite
in the report. Nothing here is hidden inside the model code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # solar-prediction/
REPO = ROOT.parent                                      # Solar/
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# --- Grid-India solar generation (the thing we predict) -------------------------
# These were extracted earlier in the project from Grid-India PSP reports.
SOLAR_2024_25_CSV = REPO / "data" / "daily_regional.csv"   # Apr 2024 - Dec 2025
SOLAR_2026_CSV = REPO / "data" / "daily_regional_2026.csv"  # 2026 (partial)
# Installed renewable (MNRE) capacity, All-India monthly (Apr 2024 - Dec 2025). Used to
# normalise generation into a capacity factor, so the model is not fooled by the fleet
# growing ~40% between 2024 and 2026. 2026 capacity is linearly extrapolated (documented).
CAPACITY_CSV = REPO / "data" / "monthly_installed_capacity.csv"
CAPACITY_COL = "res_mnre_MW"
TARGET_REGION = "ALL_INDIA"
TARGET_COL = "solar_generation_MU"                         # daily solar energy, million units

# --- Weather source: NASA POWER -------------------------------------------------
# NASA POWER (Prediction Of Worldwide Energy Resources), NASA Langley Research Center.
# Free, no API key. https://power.larc.nasa.gov  (Renewable Energy community)
# Parameter meanings (daily):
#   ALLSKY_SFC_SW_DWN  Global Horizontal Irradiance (GHI)      kWh/m^2/day
#   ALLSKY_SFC_SW_DNI  Direct Normal Irradiance (DNI)          kWh/m^2/day
#   T2M                Ambient air temperature at 2 m          deg C
#   WS10M              Wind speed at 10 m                      m/s
#   RH2M               Relative humidity at 2 m                %
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMS = ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "T2M", "WS10M", "RH2M"]
NASA_COMMUNITY = "RE"

# DNI is fetched and cached, but NOT used as a model feature: NASA POWER publishes
# DNI with a longer latency, so it is missing for recent months (Aug-Sep 2026).
# Requiring it would drop those days from the 2026 test set. GHI carries almost the
# same irradiance signal, so we exclude DNI to keep the fullest 2026 comparison.
USE_DNI = False

# Representative solar-hub locations for India, with weights ~ each state's share of
# national utility-scale solar capacity (approximate, from MNRE / CEA state-wise solar
# capacity, 2024-25). "All-India" daily weather = the capacity-weighted average of these
# points, so the national weather reflects where the panels actually are (concentrated
# in the sunny north-west and south, not spread evenly). Weights are normalised in code;
# edit this list to refine.
# The 5th field is the Grid-India region the hub belongs to, so the same weather points
# serve both the national model and the per-region models / map view.
SOLAR_HUBS = [
    # name,                 lat,    lon,  weight(%)  region   representative location
    ("Rajasthan",          27.0,   72.5,   22,      "NR"),   # Bhadla / Jodhpur belt
    ("Gujarat",            23.2,   71.5,   14,      "WR"),   # Charanka / Patan
    ("Karnataka",          14.3,   76.8,   12,      "SR"),   # Pavagada
    ("Andhra Pradesh",     15.7,   78.1,    9,      "SR"),   # Kurnool
    ("Tamil Nadu",          9.4,   78.4,    8,      "SR"),   # Kamuthi
    ("Telangana",          17.4,   78.2,    7,      "SR"),   # Hyderabad belt
    ("Maharashtra",        19.7,   74.6,    6,      "WR"),   # Shirdi / Ahmednagar
    ("Madhya Pradesh",     24.5,   81.3,    6,      "WR"),   # Rewa
    ("Uttar Pradesh",      26.5,   80.3,    4,      "NR"),   # Kanpur / Jhansi belt
    ("Rajasthan West",     28.0,   71.9,    6,      "NR"),   # Bikaner (extra NW weight)
    # Eastern and North-Eastern India carry under 1% of national solar generation, but
    # they need at least one weather point each for the regional models and the map.
    ("Bihar Jharkhand",    24.8,   85.4,    2,      "ER"),   # Gaya / Hazaribagh belt
    ("Assam",              26.2,   92.0,    0.5,    "NER"),  # Guwahati belt
]

# Grid-India regions, for the per-region models and the map view.
REGIONS = {
    "NR":  "Northern",
    "WR":  "Western",
    "SR":  "Southern",
    "ER":  "Eastern",
    "NER": "North-Eastern",
}

# --- Module temperature (NOCT) --------------------------------------------------
# Paper's formula:  T_module = T_ambient + (G / 800) * (NOCT - 20)
# G is plane-of-array irradiance in W/m^2; NOCT is the Nominal Operating Cell
# Temperature of the module (deg C). 45 C is the standard datasheet value for the
# monocrystalline silicon modules the paper studies.
NOCT = 45.0

# Daily GHI (kWh/m^2/day) is converted to an effective daytime irradiance in W/m^2
# for the NOCT formula by spreading the day's energy over an assumed number of
# sunlight hours. This is an explicit approximation (the paper used hourly data);
# documented so it can be defended / changed.
ASSUMED_SUN_HOURS = 8.0

# --- Train / test split (strictly chronological, no shuffling) ------------------
TRAIN_YEARS = [2024, 2025]
TEST_YEAR = 2026

# --- forward forecasting (a year that has not happened yet) ---------------------
# The model chosen for single-number headlines. GPR wins on the 2026 test set; change
# this if a retrain crowns a different model.
PRODUCTION_MODEL = "GPR"
FORECAST_YEARS = [2027]                 # years allowed by the /api/forecast endpoint

# --- PVWatts-style independent benchmark ----------------------------------------
# An intentionally SEPARATE, physics-only estimate with no learned parameters, so it can
# disagree with our models. Values are NREL PVWatts v8 defaults unless noted.
PVWATTS = {
    "system_losses": 0.1408,      # PVWatts default total system losses (soiling, wiring, ...)
    "inverter_eff": 0.96,         # PVWatts default inverter efficiency
    "temp_coeff_pmax": -0.0047,   # per deg C, standard crystalline-silicon module
    "transposition": 1.08,        # horizontal GHI -> tilted plane-of-array, approximate for
                                  # a latitude-tilt fixed array in India. An assumption.
    "dc_ac_ratio": 1.20,          # PVWatts default; used for clipping-free daily energy
}
# Share of the MNRE renewable fleet that is actually solar PV. The capacity table reports
# TOTAL renewables (solar + wind + biomass + small hydro), so a physics estimate that
# assumed all of it was solar would be far too high. Set from MNRE's national split.
SOLAR_SHARE_OF_RES = 0.56

# Optional: paste an official CEA / MNRE published annual solar generation target here
# (in MU) to show it alongside our forecast. Left as None because no figure is bundled
# with this repo -- the dashboard simply omits the row rather than inventing a number.
CEA_ANNUAL_TARGET_MU = None
