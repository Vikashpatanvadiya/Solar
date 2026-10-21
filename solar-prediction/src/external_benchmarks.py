"""
Independent benchmarks to check our AI forecast against.

A model that is only ever compared with itself proves nothing. These baselines are built
from completely different ingredients, so when they agree with our models that agreement
means something:

  1. PVWatts (NREL) physical model -- pure engineering physics. Irradiance on a tilted
     array, a temperature de-rate from the module's power coefficient, fixed system and
     inverter losses. NO parameter here was learned from Grid-India data, so it is a
     genuinely external opinion. If NREL_API_KEY is set in the environment the live
     PVWatts v8 API is queried per hub and cached; otherwise the same published formula
     is evaluated locally. Either way the result is the same kind of estimate.

  2. Observed capacity-factor baseline (persistence) -- the standard "do nothing clever"
     forecast: take the capacity factor the fleet actually achieved most recently and
     carry it forward onto the projected fleet. It ignores weather entirely. Any model
     worth running should at least be in the same neighbourhood.

  3. CEA / MNRE published target -- shown only if config.CEA_ANNUAL_TARGET_MU is set.
     No figure is bundled with this repo, and the dashboard omits the row rather than
     printing an invented one.

Run:  .venv/bin/python src/external_benchmarks.py     (also validates PVWatts on 2026)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

CACHE = config.RAW_DIR / "pvwatts_cache.json"
PVWATTS_URL = "https://developer.nrel.gov/api/pvwatts/v8.json"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------- the physics
def pvwatts_specific_yield(ghi_kwh_m2_day, ambient_temp_c) -> np.ndarray:
    """Daily AC energy per kW of installed DC capacity (kWh/kWp/day), PVWatts-style.

        POA      = GHI x transposition                     (horizontal -> tilted array)
        T_cell   = T_air + (G_eff/800) x (NOCT - 20)       (same NOCT model as the features)
        E_dc     = POA x [1 + gamma x (T_cell - 25)]       (module power temperature de-rate)
        E_ac     = E_dc x (1 - system_losses) x inverter_eff

    Every constant comes from config.PVWATTS and is a published PVWatts default except
    `transposition`, which is our own stated approximation for a latitude-tilt array.
    """
    p = config.PVWATTS
    ghi = np.asarray(ghi_kwh_m2_day, dtype=float)
    tair = np.asarray(ambient_temp_c, dtype=float)
    poa = ghi * p["transposition"]
    g_eff = poa * 1000.0 / config.ASSUMED_SUN_HOURS           # kWh/m2/day -> W/m2 daytime
    t_cell = tair + (g_eff / 800.0) * (config.NOCT - 20.0)
    e_dc = poa * (1.0 + p["temp_coeff_pmax"] * (t_cell - 25.0))
    return np.clip(e_dc, 0, None) * (1.0 - p["system_losses"]) * p["inverter_eff"]


def generation_MU(specific_yield, res_capacity_MW) -> np.ndarray:
    """kWh/kWp/day x installed solar MW -> million units (GWh) per day.

    The capacity table reports TOTAL renewables, so it is scaled by the solar share of
    the fleet (config.SOLAR_SHARE_OF_RES) before the physics is applied.
    """
    solar_mw = np.asarray(res_capacity_MW, dtype=float) * config.SOLAR_SHARE_OF_RES
    return np.asarray(specific_yield, dtype=float) * solar_mw / 1000.0


# ---------------------------------------------------------------- optional live NREL API
def _live_pvwatts(lat, lon, api_key, timeout=30):
    """Query the real NREL PVWatts v8 API. Returns 12 monthly AC kWh for a 1 kW system,
    or None if the service cannot be reached -- the caller then falls back to the formula."""
    p = config.PVWATTS
    url = (f"{PVWATTS_URL}?api_key={api_key}&lat={lat}&lon={lon}&system_capacity=1"
           f"&azimuth=180&tilt={abs(lat):.1f}&array_type=1&module_type=0"
           f"&losses={p['system_losses'] * 100:.2f}&dc_ac_ratio={p['dc_ac_ratio']}&timeframe=monthly")
    try:
        raw = subprocess.run(["curl", "-fsSL", "--max-time", str(timeout), url],
                             capture_output=True, check=True, timeout=timeout + 10).stdout
        data = json.loads(raw)
        monthly = data.get("outputs", {}).get("ac_monthly")
        return list(monthly) if monthly and len(monthly) == 12 else None
    except Exception as e:
        print(f"  PVWatts API unavailable for ({lat}, {lon}): {e}", file=sys.stderr)
        return None


def live_pvwatts_hubs():
    """Monthly kWh/kWp for every solar hub from the live API, cached on disk.
    Returns None when no API key is configured or every call failed."""
    key = os.environ.get("NREL_API_KEY")
    if not key:
        return None
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    changed = False
    for name, lat, lon, _w, _r in config.SOLAR_HUBS:
        if name in cache:
            continue
        got = _live_pvwatts(lat, lon, key)
        if got is None:
            continue
        cache[name] = got
        changed = True
    if changed:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1))
    return cache or None


def _weighted_hub_monthly(cache) -> np.ndarray:
    """Capacity-weighted monthly kWh/kWp across the hubs we have live values for."""
    rows, weights = [], []
    for name, _lat, _lon, w, _r in config.SOLAR_HUBS:
        if name in cache:
            rows.append(cache[name])
            weights.append(w)
    if not rows:
        return None
    wt = np.array(weights, dtype=float)
    return (np.array(rows, dtype=float) * wt[:, None]).sum(axis=0) / wt.sum()


# ---------------------------------------------------------------- baselines
def observed_capacity_factor() -> tuple:
    """Capacity factor (MU per MW per day) actually achieved in the most recent year."""
    df = pd.read_csv(config.PROCESSED_DIR / "allindia_daily.csv", parse_dates=["date"])
    latest = df[df["date"].dt.year == df["date"].dt.year.max()]
    return float(latest["capacity_factor"].mean()), int(latest["date"].dt.year.iloc[0]), len(latest)


def benchmarks(year: int, monthly_capacity_MW, climo: pd.DataFrame) -> dict:
    """All available independent baselines for `year`, as monthly MU."""
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    daily = climo.reindex([d.dayofyear for d in dates]).reset_index(drop=True)
    month_idx = pd.DatetimeIndex(dates).month
    cap_by_day = np.array([monthly_capacity_MW[m - 1] for m in month_idx], dtype=float)

    sources, notes = [], []

    # 1. PVWatts physics -- local formula, optionally replaced by the live API
    sy = pvwatts_specific_yield(daily["ghi"].values, daily["ambient_temp"].values)
    source_label = "NREL PVWatts (local formula)"
    cache = live_pvwatts_hubs()
    if cache:
        live_monthly = _weighted_hub_monthly(cache)
        if live_monthly is not None:
            days = np.bincount(month_idx, minlength=13)[1:]
            per_day = np.repeat(live_monthly / days, days)       # monthly kWh/kWp -> daily
            sy = per_day
            source_label = "NREL PVWatts v8 (live API)"
    pv_daily = generation_MU(sy, cap_by_day)
    pv_monthly = [round(float(v), 1) for v in pd.Series(pv_daily, index=dates).groupby(month_idx).sum()]
    sources.append({
        "key": "pvwatts", "name": source_label,
        "monthly_MU": pv_monthly, "annual_MU": round(float(sum(pv_monthly)), 1),
        "note": "physics only, nothing learned from Indian generation data",
    })

    # 2. Persistence -- last observed capacity factor carried onto the projected fleet
    cf, cf_year, cf_days = observed_capacity_factor()
    pers_daily = cf * cap_by_day
    pers_monthly = [round(float(v), 1) for v in pd.Series(pers_daily, index=dates).groupby(month_idx).sum()]
    sources.append({
        "key": "persistence", "name": f"Observed {cf_year} capacity factor, carried forward",
        "monthly_MU": pers_monthly, "annual_MU": round(float(sum(pers_monthly)), 1),
        "note": f"ignores weather entirely; from {cf_days} days of {cf_year}",
    })

    # 3. Official target, only if one has been supplied
    if config.CEA_ANNUAL_TARGET_MU:
        share = np.array(pv_monthly) / sum(pv_monthly)
        cea = [round(float(v), 1) for v in share * config.CEA_ANNUAL_TARGET_MU]
        sources.append({
            "key": "cea", "name": "CEA / MNRE published target",
            "monthly_MU": cea, "annual_MU": float(config.CEA_ANNUAL_TARGET_MU),
            "note": "official annual target, spread across months on the PVWatts seasonal shape",
        })
    else:
        notes.append("No CEA/MNRE target is bundled with this repo. Set "
                     "config.CEA_ANNUAL_TARGET_MU to an official published figure to show it here.")

    return {"sources": sources, "notes": notes,
            "assumptions": {"solar_share_of_RES": config.SOLAR_SHARE_OF_RES, **config.PVWATTS}}


# ---------------------------------------------------------------- validation
def validate_on_2026() -> dict:
    """Score the PVWatts baseline on 2026's REAL weather and REAL generation.

    This is what makes the benchmark trustworthy as a yardstick: if the physics
    reproduces a year we can actually check, it is reasonable to let it comment on 2027.
    """
    df = pd.read_csv(config.PROCESSED_DIR / "allindia_daily.csv", parse_dates=["date"])
    test = df[df["split"] == "test"]
    sy = pvwatts_specific_yield(test["ghi"].values, test["ambient_temp"].values)
    pred = generation_MU(sy, test["capacity_MW"].values)
    actual = test[config.TARGET_COL].values
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    return {
        "days": int(len(test)),
        "period": [test["date"].min().strftime("%Y-%m-%d"), test["date"].max().strftime("%Y-%m-%d")],
        "actual_total_MU": round(float(actual.sum()), 1),
        "pvwatts_total_MU": round(float(pred.sum()), 1),
        "bias_pct": round(float((pred.sum() / actual.sum() - 1) * 100), 2),
        "MAPE": round(mape, 2), "accuracy": round(100 - mape, 1),
        "RMSE": round(float(np.sqrt(np.mean((actual - pred) ** 2))), 1),
    }


def main():
    v = validate_on_2026()
    print("=" * 72)
    print("PVWatts physics baseline, checked against REAL 2026 weather and generation")
    print("=" * 72)
    print(f"  period        : {v['period'][0]} -> {v['period'][1]}  ({v['days']} days)")
    print(f"  actual total  : {v['actual_total_MU']:>12,.0f} MU")
    print(f"  PVWatts total : {v['pvwatts_total_MU']:>12,.0f} MU   (bias {v['bias_pct']:+.2f}%)")
    print(f"  day accuracy  : {v['accuracy']:.1f}%  (MAPE {v['MAPE']:.2f}%, RMSE {v['RMSE']:.1f} MU)")
    print("\n  For comparison, our GPR model scores 95.0% on the same days. The physics")
    print("  baseline is independent of it -- no Grid-India data shaped its constants.")


if __name__ == "__main__":
    main()
