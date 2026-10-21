#!/usr/bin/env python3
"""Extract daily solar/wind/hydro/energy/peak figures from the Grid-India daily PSP Excel files.

Input : PSP Reports/daily/files/*.xls   (see scripts/download_daily_psp.py)
Output: data/daily_regional_2026.csv    same columns as data/daily_regional.csv, 2026 dates only
        data/daily_psp_extraction_summary.json   coverage, gaps, duplicates, cross-check result
The same files are copied to site/data/ (the folder the dashboard is served from).

Files dated before 2026 are used only to cross-check this extractor against the existing
monthly-report data in data/daily_regional.csv; they are never written to the output.
The existing 2024-2025 CSV is not modified.
"""
import calendar
import csv
import json
import re
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import xlrd

ROOT = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT / "PSP Reports" / "daily" / "files"
DATA_DIR = ROOT / "data"
SITE_DATA_DIR = ROOT / "site" / "data"
OUT_CSV = DATA_DIR / "daily_regional_2026.csv"
SUMMARY = DATA_DIR / "daily_psp_extraction_summary.json"
EXISTING_CSV = DATA_DIR / "daily_regional.csv"

REGIONS = ["NR", "WR", "SR", "ER", "NER", "ALL_INDIA"]
ROWS = {  # output field -> label regex (lower-case, matched from the start) in the MOP_E sheet
    "solar_generation_MU": "solar gen",
    "wind_generation_MU": "wind gen",
    "hydro_generation_MU": "hydro gen",
    "energy_met_MU": "energy met",
    "evening_peak_demand_MW": r"demand met.*evening peak",  # wording and basis (19:00 vs 20:00) vary by file
}
FIELDS = ["date"] + ["region"] + list(ROWS)
NAME_RE = re.compile(r"^(\d{4}-\d\d-\d\d)__")


def to_num(cell):
    if isinstance(cell, (int, float)):
        return float(cell)
    s = str(cell).strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None  # '-' (no such generation in that region) or any other placeholder


def parse_workbook(path: Path) -> dict:
    sheet = xlrd.open_workbook(path).sheet_by_name("MOP_E")
    reporting = None
    found = {}
    for r in range(sheet.nrows):
        row = sheet.row_values(r)
        label = str(row[0]).strip().lower()
        if reporting is None and "date of reporting" in " ".join(str(c).lower() for c in row):
            tokens = [str(c).strip() for c in row if str(c).strip()]
            reporting = tokens[-1]
        for field, prefix in ROWS.items():
            if field not in found and re.match(prefix, label):
                cells = [c for c in row[1:] if str(c).strip() != ""]  # older files have an extra blank column
                if len(cells) >= 6:
                    found[field] = [to_num(c) for c in cells[:6]]
    return {"reporting": reporting, "values": found}


def main() -> int:
    files = sorted(FILES_DIR.glob("*.xls"))
    if not files:
        print(f"No files in {FILES_DIR}", file=sys.stderr)
        return 1

    by_date, warnings, duplicates = {}, [], {}
    for path in files:
        d = date.fromisoformat(NAME_RE.match(path.name).group(1))
        try:
            parsed = parse_workbook(path)
        except Exception as e:
            warnings.append(f"{path.name}: could not read workbook ({e})")
            continue
        missing = [f for f in ROWS if f not in parsed["values"]]
        if missing:
            warnings.append(f"{path.name}: missing rows {missing}")
            if "solar_generation_MU" in missing:
                continue
        try:
            rep = parsed["reporting"] and __import__("datetime").datetime.strptime(parsed["reporting"], "%d-%b-%Y").date()
        except ValueError:
            rep = None
        if rep != d + timedelta(days=1):
            warnings.append(f"{path.name}: 'Date of Reporting' {parsed['reporting']} is not the day after the file date {d}")
        if d in by_date:
            duplicates.setdefault(d.isoformat(), []).append(path.name)
            if by_date[d]["values"] != parsed["values"]:
                warnings.append(f"{d}: two files with DIFFERENT values; kept the later-sorted file {path.name}")
        by_date[d] = {"values": parsed["values"], "file": path.name}

    # Cross-check: dates that exist in the monthly-report CSV
    existing = {}
    with open(EXISTING_CSV) as f:
        for r in csv.DictReader(f):
            existing[(r["date"], r["region"])] = r
    checked, mismatches = 0, []
    for d, item in sorted(by_date.items()):
        if d.year >= 2026:
            continue
        for field in ROWS:
            for region, val in zip(REGIONS, item["values"].get(field, [None] * 6)):
                ref = existing.get((d.isoformat(), region), {}).get(field)
                if ref in (None, "") or val is None:
                    continue
                ref_f = float(ref)
                # monthly reports round to whole MU, except small solar values kept to 1 decimal
                tol = 0.06 if (ref_f < 10 and ref_f != int(ref_f)) else 0.55
                checked += 1
                if abs(float(ref) - val) > tol:
                    mismatches.append({"date": d.isoformat(), "region": region, "field": field, "daily": val, "monthly": float(ref)})

    rows = []
    for d, item in sorted(by_date.items()):
        if d.year < 2026:
            continue
        for i, region in enumerate(REGIONS):
            row = {"date": d.isoformat(), "region": region}
            for field in ROWS:
                v = item["values"].get(field, [None] * 6)[i]
                row[field] = "" if v is None else v
            rows.append(row)
    rows.sort(key=lambda r: (r["date"], REGIONS.index(r["region"])))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    days_2026 = sorted(d for d in by_date if d.year >= 2026)
    months = {}
    for m in range(1, days_2026[-1].month + 1):
        n = calendar.monthrange(2026, m)[1]
        have = sum(1 for d in days_2026 if d.month == m)
        months[f"2026-{m:02d}"] = {"days_with_report": have, "days_in_month": n, "complete": have == n}
    summary = {
        "files_read": len(files),
        "days_extracted_2026": len(days_2026),
        "range": [days_2026[0].isoformat(), days_2026[-1].isoformat()],
        "months": months,
        "duplicate_dates": duplicates,
        "cross_check": {"days": sum(1 for d in by_date if d.year < 2026), "values_compared": checked, "mismatches": mismatches},
        "warnings": warnings,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for p in (OUT_CSV, SUMMARY):
        shutil.copy(p, SITE_DATA_DIR / p.name)

    print(json.dumps({k: v for k, v in summary.items() if k != "months"}, indent=2))
    print("months:", {k: f"{v['days_with_report']}/{v['days_in_month']}" for k, v in months.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
