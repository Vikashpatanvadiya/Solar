import {
  REGIONS, REGION_LABELS, MONTH_NAMES,
  loadDailyRegional, availableYears, dateRange,
  filterRows, computeKPIs, aggregateMonthly, yoyMonthly, regionTotals, dailySeries,
} from "./data.js";
import { renderLineChart, renderGroupedBarChart, fmtCompact } from "./charts.js";
import { initForecast, renderForecast, redrawForecastChart } from "./forecast.js";

const REGION_COLOR = {
  NR: "var(--region-nr)",
  WR: "var(--region-wr)",
  SR: "var(--region-sr)",
  ER: "var(--region-er)",
  NER: "var(--region-ner)",
};

const state = { region: "ALL_INDIA", year: null, month: null, granularity: "daily" };
const YOY_YEARS = [2024, 2025];

function fmtDate(iso) {
  const d = new Date(iso + "T00:00:00Z");
  return d.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" });
}
function fmtDateShort(iso) {
  const d = new Date(iso + "T00:00:00Z");
  return d.toLocaleDateString("en-IN", { day: "2-digit", month: "short", timeZone: "UTC" });
}
function fmtMU(n) {
  if (n === null || n === undefined) return "–";
  return n.toLocaleString("en-IN", { maximumFractionDigits: 0 }) + " MU";
}

function el(id) { return document.getElementById(id); }

function populateFilters(allRows) {
  const regionSelect = el("region-select");
  regionSelect.innerHTML = "";
  const allOpt = new Option(REGION_LABELS.ALL_INDIA, "ALL_INDIA");
  regionSelect.add(allOpt);
  for (const r of REGIONS) regionSelect.add(new Option(`${REGION_LABELS[r]} (${r})`, r));

  const years = availableYears(allRows);
  const yearSelect = el("year-select");
  yearSelect.innerHTML = "";
  yearSelect.add(new Option("All years", ""));
  for (const y of years) yearSelect.add(new Option(String(y), String(y)));

  const monthSelect = el("month-select");
  monthSelect.innerHTML = "";
  monthSelect.add(new Option("All months", ""));
  MONTH_NAMES.forEach((name, i) => monthSelect.add(new Option(name, String(i + 1))));

  const [minDate, maxDate] = dateRange(allRows.filter((r) => r.region === "ALL_INDIA"));
  el("subtitle").textContent = `Grid-India Power Supply Position reports · ${fmtDate(minDate)} – ${fmtDate(maxDate)}`;
}

function currentScopeRows(allRows) {
  return filterRows(allRows, { region: state.region, year: state.year, month: state.month });
}

function renderKPIs(allRows) {
  const rows = currentScopeRows(allRows);
  const kpis = computeKPIs(rows);
  if (!kpis) {
    el("kpi-total").textContent = "No data";
    el("kpi-avg").textContent = "–";
    el("kpi-peak-value").textContent = "–";
    el("kpi-days").textContent = "0";
    return;
  }
  el("kpi-total").textContent = fmtMU(kpis.totalMU);
  el("kpi-total-sub").textContent = `across ${kpis.days.toLocaleString("en-IN")} day${kpis.days === 1 ? "" : "s"}`;
  el("kpi-avg").textContent = kpis.avgPerDayMU.toFixed(1);
  el("kpi-peak-value").textContent = fmtMU(kpis.peak.valueMU);
  el("kpi-peak-date").textContent = fmtDate(kpis.peak.date);
  el("kpi-days").textContent = kpis.days.toLocaleString("en-IN");
  const missing = state.month === null ? kpis.calendarDays - kpis.days : 0;
  el("kpi-range").textContent =
    `${fmtDateShort(kpis.range[0])} – ${fmtDateShort(kpis.range[1])}` + (missing > 0 ? ` · ${missing} days without a report` : "");
}

function renderMainChart(allRows) {
  const rows = currentScopeRows(allRows);
  const container = el("main-chart");
  const tableBody = document.querySelector("#main-table tbody");
  const note = el("chart-note");
  tableBody.innerHTML = "";
  note.textContent = "";

  if (rows.filter((r) => r.solar_generation_MU !== null).length === 0) {
    container.innerHTML = '<p class="kpi-sub">No data for this selection.</p>';
    return;
  }

  const addTableRow = (period, value) => {
    const tr = document.createElement("tr");
    const td1 = document.createElement("td");
    td1.textContent = period;
    const td2 = document.createElement("td");
    td2.textContent = value.toFixed(1);
    tr.append(td1, td2);
    tableBody.appendChild(tr);
  };

  // Gaps (days/months with no published report) are shown as breaks in the line.
  const fillGaps = state.month === null;
  if (state.granularity === "daily") {
    const series = dailySeries(rows, { fillGaps });
    const multiYear = series[0].date.slice(0, 4) !== series[series.length - 1].date.slice(0, 4);
    const labels = series.map((d) => (multiYear ? `${fmtDateShort(d.date)} '${d.date.slice(2, 4)}` : fmtDateShort(d.date)));
    const points = series.map((d, i) => ({ x: i, y: d.value }));
    renderLineChart(container, {
      series: [{ name: "Solar generation", color: "var(--series-solar)", points }],
      xLabels: labels,
    });
    for (const d of series) if (d.value !== null) addTableRow(fmtDate(d.date), d.value);
    if (series.some((d) => d.value === null)) {
      note.textContent = "Breaks in the line are days with no published daily report.";
    }
  } else {
    const all = aggregateMonthly(rows);
    const monthly = all.filter((m) => m.complete);
    const hidden = all.filter((m) => !m.complete).map((m) => `${MONTH_NAMES[m.month - 1]} '${String(m.year).slice(2)}`);
    // one entry per calendar month between the first and last complete month, so missing months show as breaks
    const byKey = new Map(monthly.map((m) => [m.key, m]));
    const seq = [];
    if (monthly.length) {
      let y = monthly[0].year, mo = monthly[0].month;
      const last = monthly[monthly.length - 1];
      while (y < last.year || (y === last.year && mo <= last.month)) {
        const key = `${y}-${String(mo).padStart(2, "0")}`;
        if (byKey.has(key) || fillGaps) seq.push({ y, mo, m: byKey.get(key) || null });
        mo += 1;
        if (mo > 12) { mo = 1; y += 1; }
      }
    }
    const points = seq.map((e, i) => ({ x: i, y: e.m ? e.m.solarMU : null }));
    const labels = seq.map((e) => `${MONTH_NAMES[e.mo - 1]} '${String(e.y).slice(2)}`);
    renderLineChart(container, {
      series: [{ name: "Solar generation", color: "var(--series-solar)", points }],
      xLabels: labels,
    });
    for (const m of monthly) addTableRow(`${MONTH_NAMES[m.month - 1]} ${m.year}`, m.solarMU);
    const parts = [];
    if (hidden.length) parts.push(`Months with incomplete daily reports are hidden: ${hidden.join(", ")}.`);
    if (seq.some((e) => e.m === null)) parts.push("Breaks in the line are months with no complete data.");
    note.textContent = parts.join(" ");
  }
}

function renderYoY(allRows) {
  const regionRows = filterRows(allRows, { region: state.region });
  const years = YOY_YEARS;
  const { months, series } = yoyMonthly(regionRows, years);

  const yearColors = { 0: "var(--series-2024)", 1: "var(--series-2025)" };
  const chartSeries = years.map((y, i) => ({
    name: String(y),
    color: yearColors[i % 2],
    values: series[y],
  }));

  renderGroupedBarChart(el("yoy-chart"), { categories: months, series: chartSeries });

  const legend = el("yoy-legend");
  legend.innerHTML = "";
  chartSeries.forEach((s) => {
    const item = document.createElement("div");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.setProperty("--swatch-color", s.color);
    item.appendChild(swatch);
    const text = document.createElement("span");
    text.textContent = s.name;
    item.appendChild(text);
    legend.appendChild(item);
  });
}

function renderRegionRanking(allRows) {
  const totals = regionTotals(allRows, { year: state.year, month: state.month });
  const grandTotal = totals.reduce((s, t) => s + t.totalMU, 0);
  const maxTotal = Math.max(1, ...totals.map((t) => t.totalMU));

  const scopeLabel = [
    state.year || "all years",
    state.month ? MONTH_NAMES[parseInt(state.month, 10) - 1] : null,
  ].filter(Boolean).join(", ");
  el("rank-sub").textContent = `Share of total solar generation by region · ${scopeLabel}`;

  const list = el("rank-list");
  list.innerHTML = "";
  for (const t of totals) {
    const row = document.createElement("div");
    row.className = "rank-row";

    const name = document.createElement("div");
    name.className = "rank-name";
    name.textContent = `${REGION_LABELS[t.region]} (${t.region})`;

    const track = document.createElement("div");
    track.className = "rank-track";
    const fill = document.createElement("div");
    fill.className = "rank-fill";
    fill.style.width = `${(t.totalMU / maxTotal) * 100}%`;
    fill.style.background = REGION_COLOR[t.region];
    track.appendChild(fill);

    const valueWrap = document.createElement("div");
    const value = document.createElement("div");
    value.className = "rank-value";
    value.textContent = fmtMU(t.totalMU);
    const pct = document.createElement("div");
    pct.className = "rank-pct";
    pct.textContent = grandTotal > 0 ? `${((t.totalMU / grandTotal) * 100).toFixed(1)}% of total` : "";
    valueWrap.append(value, pct);

    row.append(name, track, valueWrap);
    list.appendChild(row);
  }
}

function renderAll(allRows) {
  renderKPIs(allRows);
  renderMainChart(allRows);
  renderYoY(allRows);
  renderRegionRanking(allRows);
}

function exportCSV(allRows) {
  const rows = currentScopeRows(allRows).slice().sort((a, b) => (a.date < b.date ? -1 : 1));
  const header = "date,region,solar_generation_MU,wind_generation_MU,hydro_generation_MU,energy_met_MU,evening_peak_demand_MW";
  const lines = rows.map((r) => [
    r.date, r.region, r.solar_generation_MU, r.wind_generation_MU, r.hydro_generation_MU, r.energy_met_MU, r.evening_peak_demand_MW,
  ].map((v) => (v === null || v === undefined ? "" : v)).join(","));
  const csv = [header, ...lines].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `solar_generation_${state.region}_${state.year || "all"}${state.month ? "-" + state.month : ""}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function main() {
  const allRows = await loadDailyRegional();
  populateFilters(allRows);
  renderAll(allRows);
  initForecast(state.region);

  el("region-select").addEventListener("change", (e) => { state.region = e.target.value; renderAll(allRows); renderForecast(state.region); });
  el("year-select").addEventListener("change", (e) => { state.year = e.target.value ? parseInt(e.target.value, 10) : null; renderAll(allRows); });
  el("month-select").addEventListener("change", (e) => { state.month = e.target.value ? parseInt(e.target.value, 10) : null; renderAll(allRows); });
  el("granularity-select").addEventListener("change", (e) => { state.granularity = e.target.value; renderMainChart(allRows); });
  el("toggle-table-btn").addEventListener("click", () => {
    const wrap = el("main-table-wrap");
    wrap.classList.toggle("hidden");
    el("toggle-table-btn").textContent = wrap.classList.contains("hidden") ? "View as table" : "Hide table";
  });
  el("export-btn").addEventListener("click", () => exportCSV(allRows));
  window.addEventListener("resize", () => { renderMainChart(allRows); renderYoY(allRows); redrawForecastChart(); });
}

main().catch((err) => {
  console.error(err);
  document.querySelector(".page").insertAdjacentHTML(
    "afterbegin",
    `<div class="card" style="border-color:#e34948;color:#e34948;margin-bottom:16px;">Failed to load data: ${String(err.message || err)}. Make sure you're serving this site over HTTP (not file://) from the project root.</div>`
  );
});
