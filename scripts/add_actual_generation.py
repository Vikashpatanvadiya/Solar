#!/usr/bin/env python3
"""Administrator script: add actual 2026 monthly solar generation to the forecast database.

Runs directly against the local database (no HTTP, no key), so it is only usable by
someone with access to the server.

Usage:
  python scripts/add_actual_generation.py --month 1 --region NR --value 3120.5
  python scripts/add_actual_generation.py --csv actuals_2026.csv
CSV columns: month,region,actual_generation   (values are monthly MU)
"""
import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app import config, db  # noqa: E402
from backend.app.ai import storage  # noqa: E402


def parse_record(month, region, value) -> dict:
    month, value = int(month), float(value)
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1-12, got {month}")
    if region not in config.API_REGIONS:
        raise ValueError(f"region must be one of {config.API_REGIONS}, got {region!r}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"actual generation must be a finite, non-negative number, got {value}")
    return {"month": month, "region": region, "actual_generation": value}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, default=config.FORECAST_YEAR)
    p.add_argument("--month", type=int)
    p.add_argument("--region")
    p.add_argument("--value", type=float)
    p.add_argument("--csv")
    args = p.parse_args()

    if args.year != config.FORECAST_YEAR:
        p.error(f"only {config.FORECAST_YEAR} is supported")
    if args.csv:
        with open(args.csv, newline="") as f:
            records = [parse_record(r["month"], r["region"].strip(), r["actual_generation"]) for r in csv.DictReader(f)]
    elif None not in (args.month, args.region, args.value):
        records = [parse_record(args.month, args.region, args.value)]
    else:
        p.error("provide --csv, or all of --month, --region and --value")

    db.migrate()
    result = storage.set_actuals(args.year, records)
    print(f"Stored {result['stored']} record(s). All-India derived for months: {result['all_india_derived_for_months']}")


if __name__ == "__main__":
    main()
