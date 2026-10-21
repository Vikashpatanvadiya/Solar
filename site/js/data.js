// ---------------------------------------------------------------------------
// Data layer for the Solar Generation Dashboard.
//
// This is the ONLY module that knows about CSV files and column names.
// Everything downstream (main.js, charts.js) works with the plain objects
// this module returns. When Phase 2 adds weather data + a prediction model,
// it should land here as sibling loader/join functions (e.g. loadWeather(),
// joinWeatherToGeneration()) rather than touching main.js or charts.js.
// ---------------------------------------------------------------------------

export const REGIONS = ["NR", "WR", "SR", "ER", "NER"];

export const REGION_LABELS = {
  NR: "Northern",
  WR: "Western",
  SR: "Southern",
  ER: "Eastern",
  NER: "North-Eastern",
  ALL_INDIA: "All India",
};

const MONTH_NAMES = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

function toNum(v) {
  if (v === undefined || v === null || v === "") return null;
  const n = parseFloat(v);
  return Number.isNaN(n) ? null : n;
}

function parseCSV(text) {
  const lines = text.split("\n").filter((l) => l.length > 0);
  const headers = lines[0].split(",");
  const rows = new Array(lines.length - 1);
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(",");
    const row = {};
    for (let c = 0; c < headers.length; c++) {
      row[headers[c]] = cells[c];
    }
    rows[i - 1] = row;
  }
  return rows;
}

async function fetchCSV(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to load ${path}: ${res.status} ${res.statusText}`);
  }
  return parseCSV(await res.text());
}

let _dailyCache = null;

function typeRows(raw) {
  return raw.map((r) => {
    const year = parseInt(r.date.slice(0, 4), 10);
    const month = parseInt(r.date.slice(5, 7), 10);
    return {
      date: r.date,
      year,
      month,
      region: r.region,
      solar_generation_MU: toNum(r.solar_generation_MU),
      wind_generation_MU: toNum(r.wind_generation_MU),
      hydro_generation_MU: toNum(r.hydro_generation_MU),
      energy_met_MU: toNum(r.energy_met_MU),
      evening_peak_demand_MW: toNum(r.evening_peak_demand_MW),
    };
  });
}

/**
 * Loads daily rows: data/daily_regional.csv (2024-2025, from the monthly PSP reports) plus
 * data/daily_regional_2026.csv (2026, from the daily PSP reports) when it exists.
 * Rows: { date: "YYYY-MM-DD", year, month, region,
 *   solar_generation_MU, wind_generation_MU, hydro_generation_MU,
 *   energy_met_MU, evening_peak_demand_MW }
 */
export async function loadDailyRegional() {
  if (_dailyCache) return _dailyCache;
  const rows = typeRows(await fetchCSV("/data/daily_regional.csv"));
  try {
    rows.push(...typeRows(await fetchCSV("/data/daily_regional_2026.csv")));
  } catch {
    // the 2026 daily file is optional
  }
  _dailyCache = rows;
  return _dailyCache;
}

const DAY_MS = 86400000;
const dayNumber = (iso) => Math.round(Date.parse(iso + "T00:00:00Z") / DAY_MS);
const isoFromDay = (n) => new Date(n * DAY_MS).toISOString().slice(0, 10);

/**
 * Daily solar series for one region's rows, as [{ date, value }] where value is null on a
 * day with no report. fillGaps=true lists every calendar day between the first and last
 * date (gaps show as breaks at their true width); false only marks each gap with one null.
 */
export function dailySeries(rows, { fillGaps = true } = {}) {
  const present = rows
    .filter((r) => r.solar_generation_MU !== null)
    .sort((a, b) => (a.date < b.date ? -1 : 1));
  if (present.length === 0) return [];
  const out = [];
  let prev = null;
  for (const r of present) {
    const d = dayNumber(r.date);
    if (prev !== null && d - prev > 1) {
      if (fillGaps) for (let g = prev + 1; g < d; g++) out.push({ date: isoFromDay(g), value: null });
      else out.push({ date: isoFromDay(prev + 1), value: null });
    }
    out.push({ date: r.date, value: r.solar_generation_MU });
    prev = d;
  }
  return out;
}

/** Distinct years present in the dataset, ascending. */
export function availableYears(rows) {
  return Array.from(new Set(rows.map((r) => r.year))).sort();
}

/** Full date range covered, as [minDate, maxDate] ISO strings. */
export function dateRange(rows) {
  const dates = rows.map((r) => r.date).sort();
  return [dates[0], dates[dates.length - 1]];
}

/**
 * Filters daily rows by region and (optionally) year/month.
 * region: one of REGIONS or "ALL_INDIA"
 */
export function filterRows(rows, { region = "ALL_INDIA", year = null, month = null } = {}) {
  return rows.filter((r) => {
    if (r.region !== region) return false;
    if (year !== null && r.year !== year) return false;
    if (month !== null && r.month !== month) return false;
    return true;
  });
}

/** Sum a numeric field, ignoring nulls. */
function sum(rows, field) {
  let s = 0;
  for (const r of rows) if (r[field] !== null) s += r[field];
  return s;
}

/**
 * Headline KPIs for a filtered slice of daily rows (single region).
 * Returns null if there's no data in the slice.
 */
export function computeKPIs(rows) {
  const withSolar = rows.filter((r) => r.solar_generation_MU !== null);
  if (withSolar.length === 0) return null;
  const total = sum(withSolar, "solar_generation_MU");
  const avgPerDay = total / withSolar.length;
  let peak = withSolar[0];
  for (const r of withSolar) {
    if (r.solar_generation_MU > peak.solar_generation_MU) peak = r;
  }
  const [minDate, maxDate] = dateRange(withSolar);
  return {
    totalMU: total,
    avgPerDayMU: avgPerDay,
    peak: { date: peak.date, valueMU: peak.solar_generation_MU },
    days: withSolar.length,
    range: [minDate, maxDate],
    calendarDays: dayNumber(maxDate) - dayNumber(minDate) + 1,
  };
}

/**
 * Aggregates a single region's daily rows into monthly totals.
 * Returns an array of { key: "YYYY-MM", year, month, solarMU, days, daysInMonth }.
 * A month is complete when days === daysInMonth; partial months are not comparable totals.
 */
export function aggregateMonthly(rows) {
  const buckets = new Map();
  for (const r of rows) {
    if (r.solar_generation_MU === null) continue;
    const key = `${r.year}-${String(r.month).padStart(2, "0")}`;
    if (!buckets.has(key)) {
      buckets.set(key, { key, year: r.year, month: r.month, solarMU: 0, days: 0, daysInMonth: new Date(Date.UTC(r.year, r.month, 0)).getUTCDate() });
    }
    const b = buckets.get(key);
    b.solarMU += r.solar_generation_MU;
    b.days += 1;
  }
  return Array.from(buckets.values())
    .map((b) => ({ ...b, complete: b.days === b.daysInMonth }))
    .sort((a, b) => (a.key < b.key ? -1 : 1));
}

/**
 * Builds a year-over-year monthly comparison for a region.
 * Returns { months: ["Jan", ...], years: [2024, 2025], series: { 2024: [12 values|null], ... } }
 */
export function yoyMonthly(allRowsForRegion, years) {
  const series = {};
  for (const year of years) {
    const arr = new Array(12).fill(null);
    for (let m = 1; m <= 12; m++) {
      const rows = allRowsForRegion.filter((r) => r.year === year && r.month === m && r.solar_generation_MU !== null);
      if (rows.length > 0) arr[m - 1] = sum(rows, "solar_generation_MU");
    }
    series[year] = arr;
  }
  return { months: MONTH_NAMES, years, series };
}

/**
 * Total solar generation per region (excluding ALL_INDIA) for a filtered
 * year/month scope, sorted descending. Used for the region ranking list.
 */
export function regionTotals(allRows, { year = null, month = null } = {}) {
  const totals = REGIONS.map((region) => {
    const rows = filterRows(allRows, { region, year, month });
    const withSolar = rows.filter((r) => r.solar_generation_MU !== null);
    return { region, totalMU: sum(withSolar, "solar_generation_MU") };
  });
  totals.sort((a, b) => b.totalMU - a.totalMU);
  return totals;
}

export { MONTH_NAMES };
