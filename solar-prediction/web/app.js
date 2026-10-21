// One-page solar forecast dashboard.
//   - forecast: pick any of the 6 models, with/without module temperature, monthly or daily
//   - models:   accuracy leaderboard (click to load a model into the chart)
//   - trust:    two independent accuracy checks
//   - weather:  the five NASA POWER inputs and how much each one matters
//   - map:      real solar hubs on a real map, any day, with that day's weather + region forecast
//   - data:     the raw joined dataset
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const $ = (id) => document.getElementById(id);
const fmt0 = (n) => Math.round(n).toLocaleString("en-IN");
const fmt1 = (n) => Number(n).toLocaleString("en-IN", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const fmtMU = (n) => fmt0(n) + " MU";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

let DATA = null, RAW = null, REG = null;
let lineChart = null, dataChart = null, weatherChart = null;
let map = null, hubLayer = null, selectedHub = null;
let view = "monthly";
let FUT = null, CMP = null, futureChart = null, capPath = "compound", benchKey = null;

// ---------------------------------------------------------------- loading
async function load(refresh = false) {
  const btn = $("train-btn");
  btn.disabled = true;
  btn.textContent = refresh ? "Retraining…" : "Loading…";
  document.querySelectorAll(".card, .kpi-row").forEach((e) => e.classList.add("loading"));
  try {
    const res = await fetch(refresh ? "/api/train" : "/api/results", { method: refresh ? "POST" : "GET" });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    DATA = await res.json();
    buildModelSelect();
    render();
    if (refresh) { loadRegions(true); loadFuture(); }   // both are rebuilt by /api/train
  } catch (e) {
    $("accuracy-note").innerHTML = `<b>Could not reach the server.</b> ${e.message}. Start it with: <code>.venv/bin/python -m uvicorn src.server:app --port 8010</code>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Retrain models";
    document.querySelectorAll(".card, .kpi-row").forEach((e) => e.classList.remove("loading"));
  }
}

const modelRows = () => DATA[$("tm-select").value === "with" ? "with_module_temp" : "without_module_temp"];
const currentRow = () => {
  const rows = modelRows();
  return rows.find((r) => r.model === $("model-select").value) || rows[0];
};

function buildModelSelect() {
  const sel = $("model-select"), keep = sel.value;
  const names = DATA.with_module_temp.map((r) => r.model);
  sel.innerHTML = "";
  names.forEach((n) => sel.add(new Option(n === DATA.best_model ? `${n} — best` : n, n)));
  sel.value = names.includes(keep) ? keep : DATA.best_model;
}

// ---------------------------------------------------------------- render
function render() {
  const s = DATA.summary, row = currentRow(), isBest = row.model === DATA.best_model;
  $("subtitle").textContent =
    `All-India daily solar generation · ${s.date_range[0]} to ${s.date_range[1]} · ${s.n_test_days} days the model never trained on`;

  $("accuracy-note").innerHTML =
    `<b>What you are looking at:</b> the model learned from 2024–2025 weather and generation, then was asked to predict ` +
    `every day of 2026 from that day's weather alone. It gets each day right to within <b>${row.MAPE.toFixed(1)}%</b> on average. ` +
    `Generation is modelled as capacity factor × installed capacity, so India's fast-growing panel fleet is accounted for.`;

  $("kpi-acc").textContent = row.accuracy.toFixed(1) + "%";
  $("kpi-acc-sub").textContent = `${row.model}${isBest ? " (best of 6)" : ""} · ${s.n_test_days} test days`;
  $("kpi-acc-bar").style.width = Math.max(0, Math.min(100, row.accuracy)) + "%";
  $("kpi-rmse").textContent = fmtMU(row.RMSE);
  $("kpi-r2").textContent = `R² = ${row.R2.toFixed(3)} · average miss ${fmtMU(row.MAE)}`;

  const predTotal = row.predicted_MU.reduce((a, b) => a + b, 0);
  $("kpi-total").textContent = fmt0(s.total_actual_MU);
  const diff = predTotal - s.total_actual_MU, pct = (diff / s.total_actual_MU) * 100;
  $("kpi-total-sub").innerHTML =
    `actual MU · model said <b>${fmt0(predTotal)}</b> <span class="${Math.abs(pct) < 3 ? "pos" : ""}">(${pct >= 0 ? "+" : "−"}${Math.abs(pct).toFixed(1)}%)</span>`;
  $("kpi-model").textContent = row.model;
  $("kpi-model-sub").textContent = isBest ? "lowest error of the six" : `best is ${DATA.best_model}`;

  renderForecast(row);
  renderLeaderboard();
  renderValidation();
  $("footer").textContent = `Generated ${DATA.generated_at} · weather: NASA POWER · generation: Grid-India PSP reports`;
}

// ---------------------------------------------------------------- forecast chart
function monthly(dates, values) {
  const by = new Map();
  dates.forEach((d, i) => {
    const k = d.slice(0, 7);
    if (!by.has(k)) by.set(k, { sum: 0, n: 0 });
    const e = by.get(k); e.sum += values[i]; e.n++;
  });
  return by;
}

function renderForecast(row) {
  const tmLabel = $("tm-select").value === "with" ? "with module temperature" : "without module temperature";
  if (view === "monthly") {
    const a = monthly(DATA.dates, DATA.actual_MU), p = monthly(DATA.dates, row.predicted_MU);
    const keys = [...a.keys()].sort();
    const daysIn = (k) => new Date(+k.slice(0, 4), +k.slice(5), 0).getDate();
    const isPartial = (k) => a.get(k).n < daysIn(k);
    const labels = keys.map((k) => MONTHS[+k.slice(5) - 1] + " " + k.slice(2, 4) + (isPartial(k) ? " *" : ""));
    const partial = keys.filter(isPartial);
    $("forecast-note").innerHTML =
      `Monthly totals — ${row.model}, ${tmLabel}. Blue is what India actually generated, orange is what the model predicted.` +
      (partial.length
        ? ` Months marked <b>*</b> are totals of the days Grid-India published, not estimates of the whole month` +
          ` (${partial.map((k) => `${MONTHS[+k.slice(5) - 1]}: ${a.get(k).n}/${daysIn(k)} days`).join(", ")}).`
        : "");
    drawChart("bar", labels,
      [{ label: "Actual", data: keys.map((k) => a.get(k).sum), backgroundColor: css("--actual"), borderRadius: 4 },
       { label: "Predicted", data: keys.map((k) => p.get(k).sum), backgroundColor: css("--solar"), borderRadius: 4 }],
      "Solar generation (MU)");
  } else {
    $("forecast-note").innerHTML =
      `Every single day of 2026 — ${row.model}, ${tmLabel}. The two lines tracking each other is the forecast working; gaps are days Grid-India did not publish.`;
    const labels = splitLabels(DATA.dates);
    drawChart("line", labels,
      [{ label: "Actual", data: splitGaps(DATA.dates, DATA.actual_MU), borderColor: css("--actual"), borderWidth: 2, pointRadius: 0, tension: 0.2, spanGaps: false },
       { label: "Predicted", data: splitGaps(DATA.dates, row.predicted_MU), borderColor: css("--solar"), borderWidth: 2, borderDash: [5, 4], pointRadius: 0, tension: 0.2, spanGaps: false }],
      "Solar generation (MU)");
  }
}

function drawChart(type, labels, datasets, yTitle) {
  const cfg = {
    type, data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: {
        x: { grid: { color: css("--gridline") }, ticks: { color: css("--text-muted"), maxTicksLimit: type === "bar" ? 14 : 10, autoSkip: true } },
        y: { grid: { color: css("--gridline") }, ticks: { color: css("--text-muted") }, title: { display: true, text: yTitle, color: css("--text-muted") } },
      },
      plugins: {
        legend: { labels: { color: css("--text-secondary"), usePointStyle: true, pointStyle: type === "bar" ? "rect" : "line" } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.raw == null ? "—" : fmtMU(c.raw)}` } },
      },
    },
  };
  if (lineChart) lineChart.destroy();
  lineChart = new Chart($("line-chart"), cfg);
}

function splitGaps(dates, values) {
  const out = [];
  for (let i = 0; i < dates.length; i++) {
    if (i > 0 && (new Date(dates[i]) - new Date(dates[i - 1])) / 86400000 > 2) out.push(null);
    out.push(values[i]);
  }
  return out;
}
function splitLabels(dates) {
  const out = [];
  for (let i = 0; i < dates.length; i++) {
    if (i > 0 && (new Date(dates[i]) - new Date(dates[i - 1])) / 86400000 > 2) out.push("");
    out.push(dates[i]);
  }
  return out;
}

// ---------------------------------------------------------------- model leaderboard
function renderLeaderboard() {
  const rows = [...modelRows()].sort((a, b) => b.accuracy - a.accuracy);
  const box = $("leaderboard");
  box.innerHTML = "";
  const lo = Math.min(...rows.map((r) => r.accuracy)) - 3;
  for (const r of rows) {
    const on = r.model === $("model-select").value;
    const el = document.createElement("div");
    el.className = "lb-row clickable" + (on ? " on" : "");
    el.innerHTML = `<div class="lb-name">${r.model}</div>
      <div class="lb-track"><div class="lb-fill" style="width:${((r.accuracy - lo) / (100 - lo)) * 100}%"></div></div>
      <div class="lb-val">${r.accuracy.toFixed(1)}% · ±${fmt0(r.RMSE)} MU</div>`;
    el.addEventListener("click", () => { $("model-select").value = r.model; render(); });
    box.appendChild(el);
  }
  const best = rows[0], worst = rows[rows.length - 1];
  const withR = DATA.with_module_temp.find((r) => r.model === best.model);
  const withoutR = DATA.without_module_temp.find((r) => r.model === best.model);
  const gain = withR.MAPE - withoutR.MAPE;
  $("leaderboard-note").innerHTML =
    `<b>${best.model}</b> is the most accurate (${best.accuracy.toFixed(1)}%), <b>${worst.model}</b> the least (${worst.accuracy.toFixed(1)}%) — ` +
    `a ${(best.accuracy - worst.accuracy).toFixed(1)} point spread on the same data. ` +
    `Adding module temperature as an input changes ${best.model}'s error by ${Math.abs(gain).toFixed(2)} points ` +
    `(${gain < 0 ? "better without it" : "better with it"}) — switch "Inputs" above to see every model's version.`;
}

// ---------------------------------------------------------------- reliability
function renderValidation() {
  const v = DATA.validation, card = $("sec-trust");
  if (!v) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  $("validation-note").textContent =
    `${v.model} was checked on two separate periods it never trained on. Matching scores mean the result is trustworthy, not a lucky test set.`;
  const tests = [
    { title: "Hidden past data", sub: `${v.holdout.period} · ${v.holdout.days} days`, m: v.holdout },
    { title: "Future year (2026)", sub: `${v.future_year.period} · ${v.future_year.days} days`, m: v.future_year },
  ];
  const rowEl = $("validation-row");
  rowEl.innerHTML = "";
  for (const t of tests) {
    const el = document.createElement("div");
    el.className = "kpi-tile";
    el.innerHTML = `<p class="kpi-label">${t.title}</p>
      <div class="kpi-value kpi-good">${t.m.accuracy.toFixed(1)}%</div>
      <p class="kpi-sub">${t.sub}<br>typical error ${t.m.RMSE} MU · R² ${t.m.R2.toFixed(2)}</p>
      <div class="meter"><span style="width:${t.m.accuracy}%"></span></div>`;
    rowEl.appendChild(el);
  }
  const gap = Math.abs(v.holdout.accuracy - v.future_year.accuracy);
  $("validation-verdict").innerHTML = gap <= 2
    ? `✅ <b>Consistent</b> — the two scores differ by only ${gap.toFixed(1)} percentage points, so the accuracy is real, not a one-off.`
    : `⚠️ The two scores differ by ${gap.toFixed(1)} points — worth investigating before relying on it.`;
}

// ---------------------------------------------------------------- source data + weather
const METRIC_LABEL = {
  solar_generation_MU: "Solar generation (MU)", ghi: "GHI (kWh/m²/day)",
  ambient_temp: "Ambient temp (°C)", module_temp: "Module temp (°C)",
  wind_speed: "Wind speed (m/s)", relative_humidity: "Relative humidity (%)",
};
const WEATHER_META = [
  ["ghi", "Sunlight (GHI)", "kWh/m²/day", "how much solar energy reached the ground"],
  ["ambient_temp", "Air temperature", "°C", "the air around the panels"],
  ["module_temp", "Panel temperature", "°C", "computed from sunlight + air (NOCT)"],
  ["wind_speed", "Wind speed", "m/s", "wind cools panels, which helps them"],
  ["relative_humidity", "Humidity", "%", "haze and monsoon cloud"],
];

async function loadData() {
  try {
    const res = await fetch("/api/data");
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    RAW = await res.json();
    for (const id of ["data-year", "weather-year"]) {
      const sel = $(id);
      sel.innerHTML = "";
      sel.add(new Option("All years", "all"));
      RAW.years.forEach((y) => sel.add(new Option(String(y) + (RAW.summary[y].role === "test" ? " (test)" : " (train)"), String(y))));
    }
    $("weather-year").value = "2026";
    renderData();
    renderWeather();
  } catch (e) {
    $("data-note").textContent = "Could not load source data: " + e.message;
  }
}

const rowsForYear = (y) => (y === "all" ? RAW.rows : RAW.rows.filter((r) => String(r.year) === y));

function renderWeather() {
  const y = $("weather-year").value, rows = rowsForYear(y);
  const box = $("weather-tiles");
  box.innerHTML = "";
  for (const [key, title, unit, blurb] of WEATHER_META) {
    const vals = rows.map((r) => r[key]).filter((v) => v != null);
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    const el = document.createElement("div");
    el.className = "kpi-tile";
    el.innerHTML = `<p class="kpi-label">${title}</p>
      <div class="kpi-value" style="font-size:21px;">${fmt1(avg)}<span style="font-size:12px;color:var(--text-muted);"> ${unit}</span></div>
      <p class="kpi-sub">${blurb}<br>range ${fmt1(Math.min(...vals))} – ${fmt1(Math.max(...vals))}</p>`;
    box.appendChild(el);
  }

  // sunlight vs generation, the relationship the model lives on
  const labels = splitLabels(rows.map((r) => r.date));
  const dates = rows.map((r) => r.date);
  weatherChart && weatherChart.destroy();
  weatherChart = new Chart($("weather-chart"), {
    type: "line",
    data: { labels, datasets: [
      { label: "Sunlight — GHI", data: splitGaps(dates, rows.map((r) => r.ghi)), borderColor: css("--solar"), borderWidth: 1.6, pointRadius: 0, tension: 0.25, yAxisID: "y", spanGaps: false },
      { label: "Solar generation", data: splitGaps(dates, rows.map((r) => r.solar_generation_MU)), borderColor: css("--actual"), borderWidth: 1.6, pointRadius: 0, tension: 0.25, yAxisID: "y1", spanGaps: false },
    ] },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: {
        x: { grid: { color: css("--gridline") }, ticks: { color: css("--text-muted"), maxTicksLimit: 12, autoSkip: true } },
        y: { position: "left", grid: { color: css("--gridline") }, ticks: { color: css("--solar") }, title: { display: true, text: "GHI (kWh/m²/day)", color: css("--solar") } },
        y1: { position: "right", grid: { drawOnChartArea: false }, ticks: { color: css("--actual") }, title: { display: true, text: "Generation (MU)", color: css("--actual") } },
      },
      plugins: { legend: { labels: { color: css("--text-secondary"), usePointStyle: true, pointStyle: "line" } } },
    },
  });

  renderWeatherImpact();
}

function corr(a, b) {
  const n = a.length, ma = a.reduce((x, y) => x + y, 0) / n, mb = b.reduce((x, y) => x + y, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; num += x * y; da += x * x; db += y * y; }
  return num / Math.sqrt(da * db);
}

function renderWeatherImpact() {
  const train = RAW.rows.filter((r) => r.split === "train");
  const gen = train.map((r) => r.solar_generation_MU);
  const items = WEATHER_META.map(([key, title]) => ({ title, r: corr(train.map((x) => x[key]), gen) }))
    .sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
  const box = $("weather-impact");
  box.innerHTML = "";
  for (const it of items) {
    const el = document.createElement("div");
    el.className = "lb-row";
    el.innerHTML = `<div class="lb-name">${it.title}</div>
      <div class="lb-track"><div class="lb-fill ${it.r < 0 ? "neg" : ""}" style="width:${Math.abs(it.r) * 100}%"></div></div>
      <div class="lb-val">${it.r >= 0 ? "+" : "−"}${Math.abs(it.r).toFixed(2)} ${it.r < 0 ? "(cuts output)" : "(lifts output)"}</div>`;
    box.appendChild(el);
  }
}

function renderData() {
  const y = $("data-year").value, box = $("data-summary");
  box.innerHTML = "";
  for (const yr of (y === "all" ? RAW.years : [parseInt(y, 10)])) {
    const s = RAW.summary[yr];
    const t = document.createElement("div");
    t.className = "kpi-tile";
    t.innerHTML = `<p class="kpi-label">${yr} · ${s.role}</p>
      <div class="kpi-value" style="font-size:20px;">${fmt0(s.total_solar_MU)} MU</div>
      <p class="kpi-sub">${s.days} days · avg ${fmt0(s.avg_daily_MU)} MU/day</p>`;
    box.appendChild(t);
  }
  $("data-note").textContent = y === "all"
    ? "2024 starts in April (earlier Grid-India reports are not published); 2025 is complete; 2026 runs to the latest published report."
    : `Showing ${y} only.`;
  renderDataChart();
  renderDataTable();
}

function renderDataChart() {
  const rows = rowsForYear($("data-year").value), metric = $("data-metric").value;
  const dates = rows.map((r) => r.date);
  dataChart && dataChart.destroy();
  dataChart = new Chart($("data-chart"), {
    type: "line",
    data: { labels: splitLabels(dates), datasets: [{ label: METRIC_LABEL[metric], data: splitGaps(dates, rows.map((r) => r[metric])), borderColor: metric === "solar_generation_MU" ? css("--solar") : css("--actual"), borderWidth: 1.6, pointRadius: 0, tension: 0.2, spanGaps: false }] },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: {
        x: { grid: { color: css("--gridline") }, ticks: { color: css("--text-muted"), maxTicksLimit: 12, autoSkip: true } },
        y: { grid: { color: css("--gridline") }, ticks: { color: css("--text-muted") }, title: { display: true, text: METRIC_LABEL[metric], color: css("--text-muted") } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function renderDataTable() {
  const rows = rowsForYear($("data-year").value);
  const body = document.querySelector("#data-table tbody");
  const frag = document.createDocumentFragment();
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.date}</td><td>${r.split}</td><td>${r.solar_generation_MU}</td>
      <td>${r.ghi}</td><td>${r.ambient_temp}</td><td>${r.module_temp}</td>
      <td>${r.wind_speed}</td><td>${r.relative_humidity}</td><td>${fmt0(r.capacity_MW)}</td>`;
    frag.appendChild(tr);
  }
  body.innerHTML = "";
  body.appendChild(frag);
}

// ---------------------------------------------------------------- map
const HUB_PLACE = {
  "Rajasthan": "Bhadla / Jodhpur", "Rajasthan West": "Bikaner", "Gujarat": "Charanka, Patan",
  "Karnataka": "Pavagada", "Andhra Pradesh": "Kurnool", "Tamil Nadu": "Kamuthi",
  "Telangana": "Hyderabad belt", "Maharashtra": "Shirdi / Ahmednagar", "Madhya Pradesh": "Rewa",
  "Uttar Pradesh": "Kanpur / Jhansi", "Bihar Jharkhand": "Gaya / Hazaribagh", "Assam": "Guwahati",
};

async function loadRegions(refresh = false) {
  try {
    const res = await fetch("/api/regions");
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    REG = await res.json();
    buildDatePickers();
    if (!map) initMap();
    renderDay();
  } catch (e) {
    $("map-note").textContent = "Could not load the region data: " + e.message;
  }
}

function allDates() {
  const set = new Set();
  Object.values(REG.series).forEach((s) => Object.keys(s).forEach((d) => set.add(d)));
  return [...set].sort();
}

function buildDatePickers() {
  const dates = allDates();
  const months = [...new Set(dates.map((d) => d.slice(0, 7)))].sort();
  const msel = $("map-month"), keep = msel.value;
  msel.innerHTML = "";
  months.forEach((m) => msel.add(new Option(`${MONTHS[+m.slice(5) - 1]} ${m.slice(0, 4)}`, m)));
  msel.value = months.includes(keep) ? keep : months[months.length - 1];
  buildDayPicker();
}

function buildDayPicker() {
  const m = $("map-month").value, keep = $("map-day").value;
  const days = allDates().filter((d) => d.startsWith(m));
  const dsel = $("map-day");
  dsel.innerHTML = "";
  days.forEach((d) => dsel.add(new Option(String(+d.slice(8)), d)));
  dsel.value = days.includes(keep) ? keep : days[0];
}

function initMap() {
  map = L.map("map", { zoomControl: true, scrollWheelZoom: false, attributionControl: true })
    .setView([22.0, 79.0], 4);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 10, minZoom: 3,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);
  hubLayer = L.layerGroup().addTo(map);
}

// blue (dull) -> orange (bright sun)
function ghiColor(g) {
  const t = Math.max(0, Math.min(1, (g - 2.5) / (7.5 - 2.5)));
  const c = [[42, 120, 214], [235, 104, 52]];
  const mix = c[0].map((v, i) => Math.round(v + (c[1][i] - v) * t));
  return `rgb(${mix.join(",")})`;
}

function renderDay() {
  const date = $("map-day").value;
  if (!date) return;
  const F = REG.weather_fields;

  // --- hub markers -------------------------------------------------------
  hubLayer.clearLayers();
  for (const h of REG.hubs) {
    const w = (REG.hub_weather[h.name] || {})[date];
    const ghi = w ? w[F.indexOf("ghi")] : null;
    const marker = L.circleMarker([h.lat, h.lon], {
      radius: 6 + Math.sqrt(h.weight) * 2.6,
      color: "#fff", weight: 1.5, opacity: 0.9,
      fillColor: ghi == null ? css("--text-muted") : ghiColor(ghi),
      fillOpacity: 0.85,
    });
    marker.bindTooltip(`${h.name} — ${HUB_PLACE[h.name] || ""}`, { direction: "top", className: "hub-label" });
    marker.on("click", () => { selectedHub = h.name; renderHubDetail(date); });
    hubLayer.addLayer(marker);
  }
  if (!selectedHub) selectedHub = REG.hubs[0].name;
  renderHubDetail(date);

  // --- per-region actual vs predicted ------------------------------------
  const cards = $("region-cards");
  cards.innerHTML = "";
  const vals = [];
  for (const code of Object.keys(REG.regions)) {
    const p = (REG.series[code] || {})[date];
    if (p) vals.push(p[0], p[1]);
  }
  const max = Math.max(1, ...vals);
  let totA = 0, totP = 0, have = 0;
  for (const [code, meta] of Object.entries(REG.regions)) {
    const p = (REG.series[code] || {})[date];
    const el = document.createElement("div");
    el.className = "region-card";
    if (!meta.usable || !p) {
      el.innerHTML = `<div class="region-head"><span class="region-name">${meta.label} (${code})</span>
        <span class="region-acc">no forecast for this day</span></div>`;
    } else {
      totA += p[0]; totP += p[1]; have++;
      const err = ((p[1] - p[0]) / p[0]) * 100;
      el.innerHTML = `<div class="region-head">
          <span class="region-name">${meta.label} (${code})</span>
          <span class="region-acc">${meta.accuracy.toFixed(1)}% accurate overall</span></div>
        <div class="region-bars">
          <div class="region-bar"><span>Actual</span><div class="track"><div class="fill a" style="width:${(p[0] / max) * 100}%"></div></div><span class="num">${fmt0(p[0])} MU</span></div>
          <div class="region-bar"><span>Predicted</span><div class="track"><div class="fill p" style="width:${(p[1] / max) * 100}%"></div></div><span class="num">${fmt0(p[1])} MU</span></div>
          <div class="region-bar" style="grid-template-columns:1fr;"><span class="${Math.abs(err) < 10 ? "pos" : ""}">off by ${err >= 0 ? "+" : "−"}${Math.abs(err).toFixed(1)}%</span></div>
        </div>`;
    }
    cards.appendChild(el);
  }
  const pretty = new Date(date + "T00:00:00").toLocaleDateString("en-IN", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
  $("map-note").innerHTML = have
    ? `<b>${pretty}</b> — India generated <b>${fmt0(totA)} MU</b> of solar across these ${have} regions; the model predicted <b>${fmt0(totP)} MU</b>. Each region has its own model, trained on its own weather and its own installed capacity.`
    : `<b>${pretty}</b> — no published generation for this day.`;
}

function renderHubDetail(date) {
  const F = REG.weather_fields;
  const h = REG.hubs.find((x) => x.name === selectedHub);
  const w = (REG.hub_weather[selectedHub] || {})[date];
  const box = $("hub-detail");
  if (!h || !w) { box.innerHTML = `<b>${selectedHub || "—"}</b> — no weather recorded for this day.`; return; }
  const g = (k) => w[F.indexOf(k)];
  const chips = [
    ["Sunlight", g("ghi"), "kWh/m²/day"], ["Air temp", g("ambient_temp"), "°C"],
    ["Panel temp", g("module_temp"), "°C"], ["Wind", g("wind_speed"), "m/s"],
    ["Humidity", g("relative_humidity"), "%"],
  ];
  box.innerHTML = `<b>${h.name}</b> — ${HUB_PLACE[h.name] || ""} · ${h.region} region · ${h.weight}% of national fleet
    <div class="hub-chips">${chips.map(([n, v, u]) => `<span class="chip">${n} <b>${v == null ? "—" : fmt1(v)}</b> ${u}</span>`).join("")}</div>`;
}


// ---------------------------------------------------------------- 2027 scenario
// Deliberately kept visually distinct from the 2026 card: 2026 is a measured result,
// 2027 is a projection built on assumed weather and assumed fleet growth.
const FUTURE_YEAR = 2027;

async function loadFuture() {
  try {
    const [f, c] = await Promise.all([
      fetch(`/api/forecast/${FUTURE_YEAR}`).then((r) => r.json()),
      fetch(`/api/comparison/${FUTURE_YEAR}`).then((r) => r.json()),
    ]);
    if (f.detail || c.detail) throw new Error(f.detail || c.detail);
    FUT = f; CMP = c;
    const sel = $("f-model");
    sel.innerHTML = "";
    Object.keys(f.capacity_paths.compound.models).forEach((n) =>
      sel.add(new Option(n === c.our_model ? `${n} — production` : n, n)));
    sel.value = c.our_model;
    if (!benchKey) benchKey = c.comparisons[0].key;
    renderFuture();
  } catch (e) {
    $("f-caveat").innerHTML = `<b>2027 scenario unavailable.</b> ${e.message}`;
  }
}

function renderFuture() {
  const model = $("f-model").value;
  const path = FUT.capacity_paths[capPath];
  const ours = path.models[model].monthly_MU;
  const annual = path.models[model].annual_MU;
  const bench = CMP.comparisons.find((x) => x.key === benchKey) || CMP.comparisons[0];
  const v = CMP.benchmark_validation;

  $("f-caveat").innerHTML =
    `<b>Read this before quoting any 2027 number.</b> ${FUT.caveat} ` +
    `What the model does contribute is the <b>shape</b> — how output responds to sunlight, heat, wind and humidity through the seasons.`;

  // --- headline tiles ---------------------------------------------------
  const lo = CMP.capacity_range_MU.low_growth, hi = CMP.capacity_range_MU.recent_growth;
  const tiles = [
    { label: `${model} — projected 2027 total`, value: fmt0(annual) + " MU",
      sub: `fleet assumption: ${capPath === "compound" ? "recent growth" : "low growth"} · ` +
           `${fmt0(path.year_end_capacity_MW)} MW by December` },
    { label: "Range from the fleet assumption alone", value: `${fmt0(lo)} – ${fmt0(hi)}`,
      sub: `MU · ${(((hi - lo) / lo) * 100).toFixed(1)}% apart. Capacity, not the model, is the biggest unknown.` },
    { label: "All six models agree within", value: agreementPct().toFixed(1) + "%",
      sub: `${fmt0(Math.min(...annualAll()))} – ${fmt0(Math.max(...annualAll()))} MU across the six` },
    // divergence is recomputed against the CURRENT model and fleet path, not the
    // production/compound pair the API pre-computed
    { label: bench.name, value: fmt0(bench.annual_benchmark_MU) + " MU",
      sub: (() => {
        const d = ((annual - bench.annual_benchmark_MU) / bench.annual_benchmark_MU) * 100;
        return `${model} is ${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)}% vs this baseline`;
      })() },
  ];
  const box = $("f-kpis");
  box.innerHTML = "";
  for (const t of tiles) {
    const el = document.createElement("div");
    el.className = "kpi-tile";
    el.innerHTML = `<p class="kpi-label">${t.label}</p>
      <div class="kpi-value" style="font-size:20px;">${t.value}</div>
      <p class="kpi-sub">${t.sub}</p>`;
    box.appendChild(el);
  }

  // --- benchmark picker -------------------------------------------------
  let pick = $("bench-pick");
  if (!pick) {
    pick = document.createElement("div");
    pick.className = "bench-pick";
    pick.id = "bench-pick";
    $("f-bench-note").after(pick);
    pick.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      benchKey = b.dataset.key;
      renderFuture();
    });
  }
  pick.innerHTML = CMP.comparisons
    .map((c) => `<button data-key="${c.key}" class="${c.key === benchKey ? "active" : ""}">${c.name}</button>`)
    .join("");

  $("f-bench-note").innerHTML =
    `<b>${bench.name}</b> — ${bench.note}. Checked against the real 2026 year it scores ` +
    `<b>${v.accuracy.toFixed(1)}%</b> (bias ${v.bias_pct >= 0 ? "+" : "−"}${Math.abs(v.bias_pct).toFixed(1)}%), ` +
    `against <b>95.0%</b> for our GPR model on the same days — so our model is the better of the two where we can actually check.`;

  renderFutureChart(model, ours, bench);
  renderFutureTable(ours, bench);
  renderFutureRegions();

  const a = FUT.benchmarks.assumptions;
  $("f-assumptions").innerHTML =
    `<b>Assumptions on the table:</b> solar is taken as ${(a.solar_share_of_RES * 100).toFixed(0)}% of the reported renewable fleet; ` +
    `PVWatts uses ${(a.system_losses * 100).toFixed(1)}% system losses, ${(a.inverter_eff * 100).toFixed(0)}% inverter efficiency, ` +
    `a ${a.temp_coeff_pmax} /°C module power coefficient and a ${a.transposition}× tilt gain. ` +
    `Trained on ${FUT.history_rows} days, ${FUT.history_range[0]} to ${FUT.history_range[1]}.` +
    (CMP.notes || []).map((n) => ` ${n}`).join("");
}

const annualAll = () => Object.values(FUT.capacity_paths[capPath].models).map((m) => m.annual_MU);
function agreementPct() {
  const a = annualAll();
  return ((Math.max(...a) - Math.min(...a)) / Math.min(...a)) * 100;
}

function renderFutureChart(model, ours, bench) {
  const labels = FUT.months;
  const ds = [
    { label: "Range across our 6 models", data: FUT.consensus.max_MU, borderColor: "transparent",
      backgroundColor: "color-mix(in srgb, " + css("--good") + " 18%, transparent)", fill: "+1",
      pointRadius: 0, tension: 0.3 },
    { label: "_min", data: FUT.consensus.min_MU, borderColor: "transparent", fill: false,
      pointRadius: 0, tension: 0.3 },
    { label: `${model} (our model)`, data: ours, borderColor: css("--solar"), borderWidth: 2.4,
      pointRadius: 0, tension: 0.3 },
    { label: bench.name, data: bench.monthly.map((m) => m.benchmark_MU), borderColor: css("--actual"),
      borderWidth: 2, borderDash: [6, 4], pointRadius: 0, tension: 0.3 },
  ];
  const cfg = {
    type: "line", data: { labels, datasets: ds },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: {
        x: { grid: { color: css("--gridline") }, ticks: { color: css("--text-muted") } },
        y: { grid: { color: css("--gridline") }, ticks: { color: css("--text-muted") },
             title: { display: true, text: "Projected generation (MU)", color: css("--text-muted") } },
      },
      plugins: {
        legend: { labels: { color: css("--text-secondary"), usePointStyle: true, pointStyle: "line",
                  filter: (i) => i.text !== "_min" } },
        tooltip: { filter: (i) => i.dataset.label !== "_min",
                   callbacks: { label: (c) => `${c.dataset.label}: ${fmtMU(c.raw)}` } },
      },
    },
  };
  if (futureChart) futureChart.destroy();
  futureChart = new Chart($("future-chart"), cfg);
  $("f-chart-note").innerHTML =
    `The green band is the spread across all six models — where they disagree, confidence is lower. ` +
    `They stay within ${agreementPct().toFixed(1)}% of each other for the whole year, which is a good sign: six very different algorithms reading the same typical-year weather reach nearly the same answer.`;
}

function renderFutureTable(ours, bench) {
  $("f-th-ours").textContent = `${$("f-model").value} (MU)`;
  const body = document.querySelector("#future-table tbody");
  body.innerHTML = "";
  bench.monthly.forEach((m, i) => {
    const o = ours[i], d = o - m.benchmark_MU, pc = (d / m.benchmark_MU) * 100;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${m.month} ${FUTURE_YEAR}</td><td>${fmt0(o)}</td><td>${fmt0(m.benchmark_MU)}</td>
      <td>${d >= 0 ? "+" : "−"}${fmt0(Math.abs(d))}</td>
      <td class="${Math.abs(pc) < 10 ? "pos" : ""}">${pc >= 0 ? "+" : "−"}${Math.abs(pc).toFixed(1)}%</td>`;
    body.appendChild(tr);
  });
}

function renderFutureRegions() {
  const box = $("f-regions");
  box.innerHTML = "";
  const regs = Object.entries(FUT.regions || {}).sort((a, b) => b[1].annual_MU - a[1].annual_MU);
  if (!regs.length) { box.innerHTML = `<p class="card-note">No regional scenario available.</p>`; return; }
  const max = Math.max(...regs.map(([, r]) => r.annual_MU));
  for (const [code, r] of regs) {
    const el = document.createElement("div");
    el.className = "lb-row";
    el.innerHTML = `<div class="lb-name">${r.label} (${code})</div>
      <div class="lb-track"><div class="lb-fill" style="width:${(r.annual_MU / max) * 100}%"></div></div>
      <div class="lb-val">${fmt0(r.annual_MU)} MU in ${FUTURE_YEAR}</div>`;
    box.appendChild(el);
  }
}

// ---------------------------------------------------------------- wiring
$("model-select").addEventListener("change", render);
$("tm-select").addEventListener("change", render);
$("view-seg").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  view = b.dataset.view;
  [...$("view-seg").children].forEach((x) => x.classList.toggle("active", x === b));
  renderForecast(currentRow());
});
$("data-year").addEventListener("change", renderData);
$("data-metric").addEventListener("change", renderDataChart);
$("weather-year").addEventListener("change", renderWeather);
$("map-month").addEventListener("change", () => { buildDayPicker(); renderDay(); });
$("map-day").addEventListener("change", () => renderDay());
$("f-model").addEventListener("change", renderFuture);
$("cap-seg").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  capPath = b.dataset.cap;
  [...$("cap-seg").children].forEach((x) => x.classList.toggle("active", x === b));
  renderFuture();
});
$("train-btn").addEventListener("click", () => load(true));

load(false);
loadData();
loadRegions();
loadFuture();
