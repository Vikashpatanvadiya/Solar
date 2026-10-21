#!/usr/bin/env python3
"""
Extract structured data from Grid-India Monthly PSP (Power Supply Position) report PDFs.

Produces three CSVs under data/:
  - daily_regional.csv            date, region, solar/wind/hydro generation (MU), energy met (MU), evening peak demand (MW)
  - monthly_installed_capacity.csv  month, region, source-wise installed capacity (MW)
  - monthly_state_supply_position.csv  month, region, state, energy/peak requirement vs met

Plus data/extraction_summary.json with coverage stats, gaps and per-file warnings.

Each report is scanned by section header text rather than a fixed page index, since
page numbers shift slightly between reports (different month lengths, extra photos, etc).
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "PSP Reports"
DATA_DIR = ROOT / "data"

REGIONS = ["NR", "WR", "SR", "ER", "NER"]
DATE_RE = re.compile(r"^(\d{1,2}-[A-Za-z]{3,9}-\d{2,4})\s+(.*)$")
CID_RE = re.compile(r"\(cid:\d+\)")


def clean_cid(text):
    """Strip PDF glyph-id placeholders (unmapped Devanagari conjuncts) like '(cid:464)'."""
    return CID_RE.sub("", text)


def page_text(page):
    return clean_cid(page.extract_text() or "")
MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_report_date(token: str):
    """Parse dates like '01-Apr-24', '1-Dec-2025', '01-August-25'."""
    m = re.match(r"^(\d{1,2})-([A-Za-z]+)-(\d{2,4})$", token)
    if not m:
        return None
    day, mon_str, year = m.groups()
    mon_key = mon_str[:3].lower()
    if mon_key not in MONTH_ABBR:
        return None
    month = MONTH_ABBR[mon_key]
    year = int(year)
    if year < 100:
        year += 2000
    day = int(day)
    if day == 0 or year < 2000:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_daily_table(text, n_value_cols=6):
    """Parse a day-wise table: 'DD-Mon-YY  v1 v2 v3 v4 v5 v6' per row."""
    rows = []
    for line in text.split("\n"):
        m = DATE_RE.match(line.strip())
        if not m:
            continue
        date_token, rest = m.groups()
        d = parse_report_date(date_token)
        if d is None:
            continue
        parts = rest.split()
        nums = []
        for p in parts:
            if re.fullmatch(r"-{2,}", p):
                nums.append(None)  # region not applicable / no data (e.g. wind in ER/NER)
                continue
            p = p.replace(",", "")
            try:
                nums.append(float(p))
            except ValueError:
                nums = None
                break
        if nums is None or len(nums) < n_value_cols:
            continue
        rows.append((d, nums[:n_value_cols]))
    return rows


def find_pages(pdf, must_include, must_not_include=(), max_scan=45):
    hits = []
    for i, page in enumerate(pdf.pages[:max_scan]):
        text = page_text(page).upper()
        if all(k in text for k in must_include) and not any(k in text for k in must_not_include):
            hits.append(i)
    return hits


def extract_daily_section(pdf, header_keywords, warnings, label):
    pages = find_pages(pdf, header_keywords + ["MONTH:-"])
    if not pages:
        warnings.append(f"{label}: section not found")
        return []
    text = page_text(pdf.pages[pages[0]])
    rows = parse_daily_table(text)
    if not rows:
        warnings.append(f"{label}: section found but no rows parsed")
    return rows


def extract_installed_capacity(pdf, warnings):
    pages = find_pages(pdf, ["REGIONWISE INSTALLED CAPACITY", "AS ON"])
    if not pages:
        warnings.append("installed_capacity: section not found")
        return []
    text = page_text(pdf.pages[pages[0]])
    rows = []
    region_re = re.compile(
        r"^(\d+)\s+(NR|WR|SR|ER|NER|ISLANDS)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)$"
    )
    all_india_re = re.compile(
        r"^ALL INDIA\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)$"
    )
    for line in text.split("\n"):
        line = line.strip()
        m = region_re.match(line)
        if m:
            _, region, coal, lignite, gas, diesel, thermal_total, nuclear, hydro, res_mnre, grand_total = m.groups()
            rows.append({
                "region": region,
                "thermal_coal_MW": float(coal.replace(",", "")),
                "thermal_lignite_MW": float(lignite.replace(",", "")),
                "thermal_gas_MW": float(gas.replace(",", "")),
                "thermal_diesel_MW": float(diesel.replace(",", "")),
                "thermal_total_MW": float(thermal_total.replace(",", "")),
                "nuclear_MW": float(nuclear.replace(",", "")),
                "hydro_MW": float(hydro.replace(",", "")),
                "res_mnre_MW": float(res_mnre.replace(",", "")),
                "grand_total_MW": float(grand_total.replace(",", "")),
            })
            continue
        m = all_india_re.match(line)
        if m:
            coal, lignite, gas, diesel, thermal_total, nuclear, hydro, res_mnre, grand_total = m.groups()
            rows.append({
                "region": "ALL_INDIA",
                "thermal_coal_MW": float(coal.replace(",", "")),
                "thermal_lignite_MW": float(lignite.replace(",", "")),
                "thermal_gas_MW": float(gas.replace(",", "")),
                "thermal_diesel_MW": float(diesel.replace(",", "")),
                "thermal_total_MW": float(thermal_total.replace(",", "")),
                "nuclear_MW": float(nuclear.replace(",", "")),
                "hydro_MW": float(hydro.replace(",", "")),
                "res_mnre_MW": float(res_mnre.replace(",", "")),
                "grand_total_MW": float(grand_total.replace(",", "")),
            })
    if not rows:
        warnings.append("installed_capacity: section found but no rows parsed")
    return rows


TRAILING_11_NUMS = re.compile(
    r"(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)\s+"
    r"(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)$"
)
ALL_INDIA_ROW_7 = re.compile(
    r"ALL INDIA\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)$"
)
NAME_CANDIDATE_RE = re.compile(r"[A-Za-z][A-Za-z&()./'_\- ]*")


REGION_PREFIX_RE = re.compile(r"^(?:NR|WR|SR|ER|NER)\s+(?=\S)")


def extract_name(s):
    """Pull the (English) entity name out of a line that may be prefixed/mixed with
    Devanagari text and stray region-code letters left over from a merged table cell."""
    candidates = [c.strip() for c in NAME_CANDIDATE_RE.findall(s)]
    candidates = [c for c in candidates if c and c.upper() not in REGIONS and len(c) > 1]
    if not candidates:
        return None
    name = max(candidates, key=len)
    # A region-code cell sometimes lands glued onto the next state's name (e.g. "WR Goa").
    name = REGION_PREFIX_RE.sub("", name)
    return name


def extract_state_supply_position(pdf, warnings):
    pages = find_pages(pdf, ["ACTUAL POWER SUPPLY POSITION", "MU/DAY"])
    if not pages:
        warnings.append("state_supply_position: section not found")
        return []
    rows = []
    pending = []  # state rows accumulated since the last TOTAL, region not yet known
    region_iter = iter(REGIONS)
    deferred = None  # a parsed numbers-row still waiting for its name (from an adjacent line)
    # table can span more than one page
    for pi in range(pages[0], min(pages[0] + 3, len(pdf.pages))):
        text = page_text(pdf.pages[pi])
        if pi != pages[0] and "TOTAL" not in text.upper():
            break
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if ALL_INDIA_ROW_7.search(line):
                deferred = None
                continue  # national total, not a state row
            m = TRAILING_11_NUMS.search(line)
            if m:
                nums = [float(x.replace(",", "")) for x in m.groups()]
                prefix = line[:m.start()]
                (_req_day, _met_day, _dev_day_pct, req_mu, met_mu, _dev_mu, _dev_mu_pct,
                 peak_mw, peak_met_mw, _dev_mw, dev_mw_pct) = nums
                if re.search(r"\bTOTAL\b", prefix, re.IGNORECASE):
                    region = next(region_iter, None)
                    for p in pending:
                        p["region"] = region
                        rows.append(p)
                    pending = []
                    deferred = None
                    continue
                row = {
                    "state": extract_name(prefix),
                    "energy_requirement_MU": req_mu,
                    "energy_met_MU": met_mu,
                    "peak_demand_MW": peak_mw,
                    "peak_met_MW": peak_met_mw,
                    "peak_deficit_pct": dev_mw_pct,
                }
                if row["state"]:
                    pending.append(row)
                    deferred = None
                else:
                    deferred = row  # name must be on an adjacent line (wrapped table cell)
                continue
            # No numbers on this line: maybe it completes a deferred row's name,
            # otherwise it's a region marker / header / stray Devanagari fragment - ignore.
            if deferred is not None:
                name = extract_name(line)
                if name:
                    deferred["state"] = name
                    pending.append(deferred)
                    deferred = None
    if not rows:
        warnings.append("state_supply_position: section found but no rows parsed")
    return rows


def process_pdf(path: Path):
    warnings = []
    with pdfplumber.open(path) as pdf:
        solar = extract_daily_section(pdf, ["SOLAR GENERATION AT NATIONAL"], warnings, "solar")
        wind = extract_daily_section(pdf, ["WIND GENERATION AT NATIONAL"], warnings, "wind")
        hydro = extract_daily_section(pdf, ["HYDRO GENERATION AT NATIONAL"], warnings, "hydro")
        energy_met = extract_daily_section(pdf, ["ENERGY MET AT NATIONAL"], warnings, "energy_met")
        evening_peak = extract_daily_section(pdf, ["EVENING PEAK HOUR", "DEMAND MET AT NATIONAL"], warnings, "evening_peak")
        capacity = extract_installed_capacity(pdf, warnings)
        supply_position = extract_state_supply_position(pdf, warnings)

    by_date = {}
    for label, rows in [("solar", solar), ("wind", wind), ("hydro", hydro), ("energy_met", energy_met), ("peak", evening_peak)]:
        for d, nums in rows:
            for region, val in zip(REGIONS + ["ALL_INDIA"], nums):
                key = (d, region)
                by_date.setdefault(key, {})[label] = val

    daily_rows = []
    for (d, region), vals in by_date.items():
        daily_rows.append({
            "date": d.isoformat(),
            "region": region,
            "solar_generation_MU": vals.get("solar"),
            "wind_generation_MU": vals.get("wind"),
            "hydro_generation_MU": vals.get("hydro"),
            "energy_met_MU": vals.get("energy_met"),
            "evening_peak_demand_MW": vals.get("peak"),
        })

    month = None
    if daily_rows:
        any_date = min(r["date"] for r in daily_rows)
        month = any_date[:7]
    capacity_rows = [dict(month=month, **row) for row in capacity]
    supply_rows = [dict(month=month, **row) for row in supply_position]

    return {
        "file": path.name,
        "daily_rows": daily_rows,
        "capacity_rows": capacity_rows,
        "supply_rows": supply_rows,
        "warnings": warnings,
    }


def main():
    pdf_paths = sorted(REPORTS_DIR.glob("*/*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found under {REPORTS_DIR}", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)

    all_daily = []
    all_capacity = []
    all_supply = []
    file_reports = []

    for path in pdf_paths:
        print(f"Processing {path.relative_to(ROOT)} ...")
        try:
            result = process_pdf(path)
        except Exception as e:
            file_reports.append({"file": path.name, "error": str(e)})
            print(f"  ERROR: {e}")
            continue
        all_daily.extend(result["daily_rows"])
        all_capacity.extend(result["capacity_rows"])
        all_supply.extend(result["supply_rows"])
        file_reports.append({
            "file": result["file"],
            "daily_rows": len(result["daily_rows"]),
            "capacity_rows": len(result["capacity_rows"]),
            "supply_rows": len(result["supply_rows"]),
            "warnings": result["warnings"],
        })
        for w in result["warnings"]:
            print(f"  WARNING: {w}")

    # Deduplicate (date, region) in case of overlapping/reissued reports (e.g. "_R2")
    daily_dedup = {}
    for row in all_daily:
        key = (row["date"], row["region"])
        daily_dedup[key] = row  # later file in sorted order wins
    all_daily = list(daily_dedup.values())
    all_daily.sort(key=lambda r: (r["date"], r["region"]))

    all_capacity.sort(key=lambda r: (r["month"] or "", r["region"]))
    all_supply.sort(key=lambda r: (r["month"] or "", r["region"] or "", r["state"]))

    write_csv(DATA_DIR / "daily_regional.csv",
              ["date", "region", "solar_generation_MU", "wind_generation_MU", "hydro_generation_MU",
               "energy_met_MU", "evening_peak_demand_MW"], all_daily)
    write_csv(DATA_DIR / "monthly_installed_capacity.csv",
              ["month", "region", "thermal_coal_MW", "thermal_lignite_MW", "thermal_gas_MW",
               "thermal_diesel_MW", "thermal_total_MW", "nuclear_MW", "hydro_MW", "res_mnre_MW",
               "grand_total_MW"], all_capacity)
    write_csv(DATA_DIR / "monthly_state_supply_position.csv",
              ["month", "region", "state", "energy_requirement_MU", "energy_met_MU",
               "peak_demand_MW", "peak_met_MW", "peak_deficit_pct"], all_supply)

    summary = build_summary(all_daily, file_reports)
    with open(DATA_DIR / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary["headline"], indent=2))


def write_csv(path, fieldnames, rows):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


def build_summary(all_daily, file_reports):
    all_india = sorted(set(r["date"] for r in all_daily if r["region"] == "ALL_INDIA"))
    gaps = []
    if all_india:
        from datetime import timedelta
        d0 = date.fromisoformat(all_india[0])
        d1 = date.fromisoformat(all_india[-1])
        present = set(all_india)
        cur = d0
        while cur <= d1:
            if cur.isoformat() not in present:
                gaps.append(cur.isoformat())
            cur += timedelta(days=1)

    headline = {
        "files_processed": len(file_reports),
        "date_range": [all_india[0], all_india[-1]] if all_india else None,
        "total_all_india_day_rows": len(all_india),
        "missing_day_count": len(gaps),
        "total_daily_rows_all_regions": len(all_daily),
    }
    return {
        "headline": headline,
        "missing_dates": gaps,
        "per_file": file_reports,
    }


if __name__ == "__main__":
    main()
