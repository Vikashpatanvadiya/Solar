// ---------------------------------------------------------------------------
// 2026 AI Forecast section. Talks to the backend at /api/ai/*, reuses the
// dashboard's existing chart renderer, tiles, buttons and table styles.
// The public dashboard only reads predictions/actuals; entering actuals is an
// administrator-only backend operation and is intentionally not exposed here.
// ---------------------------------------------------------------------------
import { MONTH_NAMES, REGION_LABELS } from "./data.js";
import { renderGroupedBarChart } from "./charts.js";

const YEAR = 2026;
const $ = (id) => document.getElementById(id);

class ApiUnavailable extends Error {}

async function api(path, options) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new ApiUnavailable();
  }
  // A static file server answers unknown /api paths with HTML, not JSON.
  if (!(res.headers.get("content-type") || "").includes("application/json")) throw new ApiUnavailable();
  const body = await res.json();
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`);
  return body;
}

let currentRegion = "ALL_INDIA";
let view = null; // last loaded { metrics, comparison }
let busy = false;

const fmtMU = (n) => (n === null || n === undefined ? "–" : Math.round(n).toLocaleString("en-IN") + " MU");
const fmtSigned = (n) => (n === null || n === undefined ? "–" : (n > 0 ? "+" : n < 0 ? "−" : "") + Math.abs(Math.round(n)).toLocaleString("en-IN"));
const fmtPct = (n, signed = false) =>
  n === null || n === undefined ? "–" : (signed && n > 0 ? "+" : n < 0 ? "−" : "") + Math.abs(n).toFixed(1) + "%";
const modelLabel = (name) => (name || "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

function setStatus(text, isError = false) {
  const s = $("fc-status");
  s.textContent = text;
  s.classList.toggle("status-error", isError);
}

function setButtons() {
  const trained = !!(view && view.metrics.trained);
  const generated = !!(view && view.comparison.summary.predicted_total_all_months !== null);
  $("fc-train-btn").disabled = busy || !view;
  $("fc-generate-btn").disabled = busy || !trained;
  $("fc-compare-btn").disabled = busy || !generated;
}

function tile(label, value, sub, accent = false) {
  const t = document.createElement("div");
  t.className = "kpi-tile";
  const l = document.createElement("p");
  l.className = "kpi-label";
  l.textContent = label;
  const v = document.createElement("div");
  v.className = "kpi-value" + (accent ? " kpi-accent" : "");
  v.textContent = value;
  const s = document.createElement("p");
  s.className = "kpi-sub";
  s.textContent = sub;
  t.append(l, v, s);
  return t;
}

function renderWarnings(metrics, summary) {
  const box = $("fc-warnings");
  box.innerHTML = "";
  const items = [...(metrics.warnings || [])];
  if (summary.months_with_actual === 0) items.push("No actual 2026 generation has been added yet, so predictions cannot be compared with actuals.");
  if (items.length === 0) {
    box.classList.add("hidden");
    return;
  }
  const title = document.createElement("p");
  title.className = "notice-title";
  title.textContent = "Read before relying on these numbers";
  const ul = document.createElement("ul");
  for (const w of items) {
    const li = document.createElement("li");
    li.textContent = w;
    ul.appendChild(li);
  }
  box.append(title, ul);
  box.classList.remove("hidden");
}

function renderMetricTiles(metrics, summary) {
  const row = $("fc-metrics");
  row.innerHTML = "";
  const live = summary.live_metrics;
  let m, source;
  if (live) {
    m = live;
    source = `2026 actuals · ${live.n} month${live.n === 1 ? "" : "s"}`;
  } else {
    m = metrics.validation_metrics;
    const p = metrics.feature_config && metrics.feature_config.validation_periods;
    source = `Validation${p ? ` ${p[0]} to ${p[1]}` : ""} · all regions pooled`;
  }
  if (!m) return;
  row.append(
    tile("MAE", fmtMU(m.mae), source),
    tile("RMSE", fmtMU(m.rmse), source),
    tile("R²", m.r2 === null ? "n/a" : m.r2.toFixed(3), m.r2 === null ? "needs at least 3 months of actuals" : source),
    tile("Percentage error (MAPE)", fmtPct(m.mape), source),
  );
}

function renderCandidates(candidates, selectedModel) {
  let card = $("fc-candidates-card");
  if (!card) {
    card = document.createElement("div");
    card.id = "fc-candidates-card";
    card.style.margin = "20px 0 16px";
    card.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:8px;">
        <h3 style="font-size:15px; font-weight:600; margin:0;">AI Candidate Models & Validation Benchmarks</h3>
        <a href="/lab" target="_blank" style="font-size:13px; color:var(--primary); text-decoration:none; font-weight:500;">View 6 Daily ML Models (GPR, SVM, ANN, ANN-PSO...) &rarr;</a>
      </div>
      <div class="data-table-wrap">
        <table class="data-table" id="fc-candidates-table">
          <thead>
            <tr>
              <th>Model Architecture</th>
              <th>Status</th>
              <th>MAE</th>
              <th>RMSE</th>
              <th>R² Score</th>
              <th>Pooled MAPE</th>
              <th>All-India MAPE</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    `;
    const wrap = $("fc-table-wrap");
    wrap.parentNode.insertBefore(card, wrap);
  }
  const tbody = card.querySelector("tbody");
  tbody.innerHTML = "";
  if (!candidates) return;
  for (const [name, stats] of Object.entries(candidates)) {
    const isSelected = name === selectedModel;
    const tr = document.createElement("tr");
    if (isSelected) {
      tr.style.backgroundColor = "rgba(16, 185, 129, 0.12)";
      tr.style.fontWeight = "600";
    }
    tr.innerHTML = `
      <td>${modelLabel(name)}</td>
      <td>${isSelected ? "🏆 Active Production Model" : "Evaluated Candidate"}</td>
      <td>${fmtMU(stats.mae)}</td>
      <td>${fmtMU(stats.rmse)}</td>
      <td>${stats.r2 != null ? stats.r2.toFixed(3) : "–"}</td>
      <td>${fmtPct(stats.mape)}</td>
      <td><span style="color:${stats.all_india_mape < 10 ? '#10b981' : 'inherit'};font-weight:600;">${fmtPct(stats.all_india_mape)}</span></td>
    `;
    tbody.appendChild(tr);
  }
}

function renderTable(rows) {
  const body = document.querySelector("#fc-table tbody");
  body.innerHTML = "";
  const byMonth = new Map(rows.map((r) => [r.month, r]));
  for (let m = 1; m <= 12; m++) {
    const r = byMonth.get(m) || {};
    const tr = document.createElement("tr");
    const cells = [
      `${MONTH_NAMES[m - 1]} ${YEAR}`,
      r.predicted_generation == null ? "–" : r.predicted_generation.toFixed(1),
      r.actual_generation == null ? "–" : r.actual_generation.toFixed(1),
      r.difference == null ? "–" : fmtSigned(r.difference),
      fmtPct(r.percentage_error, true),
    ];
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = c;
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

function renderLegend(series) {
  const legend = $("fc-legend");
  legend.innerHTML = "";
  for (const s of series) {
    const item = document.createElement("div");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.setProperty("--swatch-color", s.color);
    const text = document.createElement("span");
    text.textContent = s.name;
    item.append(swatch, text);
    legend.appendChild(item);
  }
}

function drawChart() {
  if (!view) return;
  const rows = view.comparison.rows;
  const byMonth = new Map(rows.map((r) => [r.month, r]));
  const pick = (key) => MONTH_NAMES.map((_, i) => (byMonth.get(i + 1) || {})[key] ?? null);
  const series = [{ name: "Predicted", color: "var(--series-forecast)", values: pick("predicted_generation") }];
  if (rows.some((r) => r.actual_generation != null)) {
    series.push({ name: "Actual", color: "var(--series-solar)", values: pick("actual_generation") });
  }
  renderLegend(series);
  renderGroupedBarChart($("fc-chart"), { categories: MONTH_NAMES, series });
}

function render() {
  const { metrics, comparison } = view;
  const summary = comparison.summary;
  const generated = summary.predicted_total_all_months !== null;
  $("fc-region-label").textContent = `Region: ${REGION_LABELS[currentRegion]}. Month and year filters do not apply to this section.`;

  if (!metrics.trained) {
    setStatus("No model has been trained yet. Select Train AI Model to begin.");
  } else if (!generated) {
    setStatus(`Model trained (${modelLabel(metrics.model_name)}). Select Generate 2026 Forecast to create the predictions.`);
  } else {
    setStatus(`Forecast generated with ${modelLabel(metrics.model_name)}, trained ${metrics.trained_at.slice(0, 10)}.`);
  }
  $("fc-body").classList.toggle("hidden", !generated);
  $("fc-warnings").classList.toggle("hidden", !metrics.trained);
  if (metrics.trained) renderWarnings(metrics, summary);
  setButtons();
  if (!generated) return;

  const kpis = $("fc-kpis");
  kpis.innerHTML = "";
  kpis.append(
    tile("Total predicted 2026", fmtMU(summary.predicted_total_all_months), "all 12 months"),
    tile("Total actual 2026", summary.actual_total === null ? "Not yet added" : fmtMU(summary.actual_total), `${summary.months_with_actual} of 12 months available`, summary.actual_total !== null),
    tile("Difference", summary.difference_total === null ? "–" : fmtSigned(summary.difference_total) + " MU", "actual − predicted, months with actuals"),
    tile("Selected model", modelLabel(metrics.model_name), "best on chronological validation"),
  );
  renderMetricTiles(metrics, summary);
  renderCandidates(metrics.candidates, metrics.model_name);
  renderTable(comparison.rows);
  drawChart();
}

async function load() {
  try {
    const [metrics, comparison] = await Promise.all([
      api("/api/ai/metrics"),
      api(`/api/ai/comparison/${YEAR}?region=${encodeURIComponent(currentRegion)}`),
    ]);
    view = { metrics, comparison };
    render();
  } catch (err) {
    view = null;
    $("fc-body").classList.add("hidden");
    $("fc-warnings").classList.add("hidden");
    setButtons();
    setStatus(
      err instanceof ApiUnavailable
        ? "The AI service is not reachable. Start the backend with: .venv/bin/python -m uvicorn backend.app.main:app --port 8000"
        : err.message,
      true,
    );
  }
}

async function run(label, path) {
  busy = true;
  setButtons();
  setStatus(label);
  try {
    await api(path, { method: "POST" });
  } catch (err) {
    busy = false;
    setStatus(err instanceof ApiUnavailable ? "The AI service is not reachable." : err.message, true);
    setButtons();
    return;
  }
  busy = false;
  await load();
}

export function renderForecast(region) {
  currentRegion = region;
  return load();
}

export function redrawForecastChart() {
  drawChart();
}

export function initForecast(initialRegion) {
  currentRegion = initialRegion;
  $("fc-train-btn").addEventListener("click", () => run("Training the model on 2024–2025 data…", "/api/ai/train"));
  $("fc-generate-btn").addEventListener("click", () => run("Generating the 2026 forecast…", `/api/ai/forecast/${YEAR}`));
  $("fc-compare-btn").addEventListener("click", async () => {
    await load();
    $("fc-table-wrap").scrollIntoView({ behavior: "smooth", block: "center" });
  });
  return load();
}
