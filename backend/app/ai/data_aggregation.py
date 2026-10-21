import pandas as pd

from .. import config


def load_daily(csv_path=config.DATA_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path, parse_dates=["date"])


def monthly_by_region(daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate daily solar generation (MU) to one row per region-month.

    Columns: region, period (month start), total_mu, days_present, days_in_month,
    complete (every day of the month has a value), avg_daily_mu.
    Only the five physical regions are modelled; All-India is derived later as their sum.
    """
    if daily is None:
        daily = load_daily()
    d = daily[daily["region"].isin(config.MODEL_REGIONS) & daily["solar_generation_MU"].notna()].copy()
    d["period"] = d["date"].dt.to_period("M").dt.to_timestamp()
    g = (
        d.groupby(["region", "period"])
        .agg(total_mu=("solar_generation_MU", "sum"), days_present=("date", "nunique"))
        .reset_index()
    )
    g["days_in_month"] = g["period"].dt.days_in_month
    g["complete"] = g["days_present"] == g["days_in_month"]
    g["avg_daily_mu"] = g["total_mu"] / g["days_present"]
    return g.sort_values(["region", "period"]).reset_index(drop=True)


def history_summary(monthly: pd.DataFrame) -> dict:
    complete = monthly[monthly["complete"]]
    periods = sorted(complete["period"].unique())
    return {
        "first_month": pd.Timestamp(periods[0]).strftime("%Y-%m") if periods else None,
        "last_month": pd.Timestamp(periods[-1]).strftime("%Y-%m") if periods else None,
        "n_months": len(periods),
        "incomplete_region_months": int((~monthly["complete"]).sum()),
    }
