#!/usr/bin/env python3
"""Administrator script: turn complete months of 2026 daily PSP data into forecast 'actuals'.

Reads data/daily_regional_2026.csv (see scripts/extract_daily_psp.py). A month becomes an
actual only if every day of that month has a solar value for all five regions; partial
months are skipped and reported, never extrapolated. All-India is the sum of the regions,
which matches how the forecast's All-India figure is built.

Dry run by default. Add --apply to write to the forecast database (needs local access to it,
like scripts/add_actual_generation.py).
"""
import argparse
import calendar
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app import config, db  # noqa: E402
from backend.app.ai import storage  # noqa: E402

CSV = config.ROOT / "data" / "daily_regional_2026.csv"


def complete_month_records(year: int, scale_partial: bool = False) -> tuple[list[dict], list[str]]:
    df = pd.read_csv(CSV, parse_dates=["date"])
    df = df[(df["date"].dt.year == year) & df["region"].isin(config.MODEL_REGIONS) & df["solar_generation_MU"].notna()]
    records, skipped = [], []
    for month in range(1, 13):
        n_days = calendar.monthrange(year, month)[1]
        m = df[df["date"].dt.month == month]
        per_region = m.groupby("region")["date"].nunique()
        if all(per_region.get(r, 0) == n_days for r in config.MODEL_REGIONS):
            totals = m.groupby("region")["solar_generation_MU"].sum()
            records += [
                {"month": month, "region": r, "actual_generation": round(float(totals[r]), 2)} for r in config.MODEL_REGIONS
            ]
        elif len(m) and scale_partial:
            totals = m.groupby("region")["solar_generation_MU"].sum()
            records += [
                {
                    "month": month,
                    "region": r,
                    "actual_generation": round(float((totals[r] / per_region[r]) * n_days), 2)
                } for r in config.MODEL_REGIONS if per_region.get(r, 0) > 0
            ]
            skipped.append(f"{year}-{month:02d}: {int(per_region.min()) if len(per_region) else 0}-{int(per_region.max()) if len(per_region) else 0} of {n_days} days present, normalized/scaled to full month")
        elif len(m):
            skipped.append(f"{year}-{month:02d}: {int(per_region.min()) if len(per_region) else 0}-{int(per_region.max()) if len(per_region) else 0} of {n_days} days, skipped (use --scale-partial to include)")
        else:
            skipped.append(f"{year}-{month:02d}: no daily reports, skipped")
    return records, skipped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, default=config.FORECAST_YEAR)
    p.add_argument("--apply", action="store_true", help="write to the database (default is a dry run)")
    # Scaling a partial month up to a full month invents generation Grid-India never
    # published, and then scores the model against those invented numbers. It is OFF by
    # default and must be asked for explicitly.
    p.add_argument("--scale-partial", action="store_true",
                   help="estimate a full month from the days present (INVENTS DATA - not for accuracy reporting)")
    args = p.parse_args()

    scale = args.scale_partial
    if scale:
        print("WARNING: --scale-partial estimates missing days. The resulting 'actuals' are\n"
              "         not published measurements and must not be quoted as accuracy.\n")
    records, skipped = complete_month_records(args.year, scale_partial=scale)
    months = sorted({r["month"] for r in records})
    print(f"Complete months found: {months or 'none'}")
    for line in skipped:
        print("  " + line)
    for r in records:
        print(f"  {args.year}-{r['month']:02d} {r['region']:>4}: {r['actual_generation']:>10,.2f} MU")
    if not records:
        return
    if not args.apply:
        print("Dry run only. Re-run with --apply to store these as actuals.")
        return
    db.migrate()
    result = storage.set_actuals(args.year, records)
    print(f"Stored {result['stored']} record(s). All-India derived for months: {result['all_india_derived_for_months']}")


if __name__ == "__main__":
    main()
