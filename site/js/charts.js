// ---------------------------------------------------------------------------
// Minimal SVG chart primitives: a line/area chart (with crosshair tooltip)
// and a grouped column chart (with per-bar tooltip). No external dependency.
// Mark specs (bar thickness, line weight, gridlines, spacers) follow the
// dataviz style guide: thin 2px lines, <=24px bars with 4px rounded caps,
// hairline recessive gridlines, a 2px surface ring on end-markers.
// ---------------------------------------------------------------------------

const SVGNS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function niceMax(value) {
  if (value <= 0) return 10;
  const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
  const residual = value / magnitude;
  let niceResidual;
  if (residual <= 1) niceResidual = 1;
  else if (residual <= 2) niceResidual = 2;
  else if (residual <= 5) niceResidual = 5;
  else niceResidual = 10;
  return niceResidual * magnitude;
}

function ticksFor(max, count = 4) {
  const step = max / count;
  const ticks = [];
  for (let i = 0; i <= count; i++) ticks.push(Math.round(step * i));
  return ticks;
}

function fmtCompact(n) {
  if (n === null || n === undefined) return "–";
  if (Math.abs(n) >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
  return Math.round(n).toLocaleString("en-IN");
}

function makeTooltip(root) {
  const tip = document.createElement("div");
  tip.className = "chart-tooltip";
  tip.hidden = true;
  root.appendChild(tip);
  return tip;
}

/**
 * Renders a line chart. `series` = [{ name, color, points: [{x, y, label}] }]
 * `x` values must be shared indices across series (same length, aligned).
 */
export function renderLineChart(container, { series, xLabels, height = 260, yFormat = fmtCompact, showArea = true }) {
  container.innerHTML = "";
  const root = document.createElement("div");
  root.className = "chart-root";
  container.appendChild(root);

  const width = Math.max(container.clientWidth || 640, 320);
  const margin = { top: 16, right: 20, bottom: 28, left: 44 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const allValues = series.flatMap((s) => s.points.map((p) => p.y).filter((v) => v !== null));
  const maxVal = niceMax(Math.max(1, ...allValues));
  const ticks = ticksFor(maxVal, 4);
  const n = xLabels.length;

  const xAt = (i) => margin.left + (n <= 1 ? 0 : (i / (n - 1)) * innerW);
  const yAt = (v) => margin.top + innerH - (v / maxVal) * innerH;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height, class: "chart-svg" });

  // gridlines + y-axis labels
  for (const t of ticks) {
    const y = yAt(t);
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "chart-gridline" }));
    const label = svgEl("text", { x: margin.left - 8, y: y + 4, class: "chart-axis-label", "text-anchor": "end" });
    label.textContent = yFormat(t);
    svg.appendChild(label);
  }
  // baseline
  svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: yAt(0), y2: yAt(0), class: "chart-baseline" }));

  // x-axis labels (thin out to avoid crowding)
  const maxLabels = Math.floor(innerW / 56);
  const stride = Math.max(1, Math.ceil(n / Math.max(1, maxLabels)));
  for (let i = 0; i < n; i += stride) {
    const label = svgEl("text", { x: xAt(i), y: height - 6, class: "chart-axis-label", "text-anchor": "middle" });
    label.textContent = xLabels[i];
    svg.appendChild(label);
  }

  // series: area (single-series only) + line + end marker. Missing values (y === null)
  // break the line, so gaps in the data are visible rather than bridged.
  series.forEach((s) => {
    const segments = [];
    let current = [];
    for (const p of s.points) {
      if (p.y === null) {
        if (current.length) segments.push(current);
        current = [];
      } else current.push(p);
    }
    if (current.length) segments.push(current);
    if (segments.length === 0) return;

    for (const seg of segments) {
      if (seg.length === 1) {
        svg.appendChild(svgEl("circle", { cx: xAt(seg[0].x), cy: yAt(seg[0].y), r: 2.5, fill: s.color }));
        continue;
      }
      const pathD = seg.map((p, i) => `${i === 0 ? "M" : "L"} ${xAt(p.x)} ${yAt(p.y)}`).join(" ");
      if (showArea && series.length === 1) {
        const areaD = `${pathD} L ${xAt(seg[seg.length - 1].x)} ${yAt(0)} L ${xAt(seg[0].x)} ${yAt(0)} Z`;
        svg.appendChild(svgEl("path", { d: areaD, fill: s.color, opacity: 0.1, stroke: "none" }));
      }
      svg.appendChild(svgEl("path", { d: pathD, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    }

    const lastSeg = segments[segments.length - 1];
    const last = lastSeg[lastSeg.length - 1];
    svg.appendChild(svgEl("circle", { cx: xAt(last.x), cy: yAt(last.y), r: 5, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2 }));
  });

  // crosshair overlay
  const crosshair = svgEl("line", { x1: 0, x2: 0, y1: margin.top, y2: margin.top + innerH, class: "chart-crosshair" });
  crosshair.style.opacity = 0;
  svg.appendChild(crosshair);

  const overlay = svgEl("rect", {
    x: margin.left, y: margin.top, width: innerW, height: innerH, fill: "transparent",
  });
  svg.appendChild(overlay);
  root.appendChild(svg);

  const tip = makeTooltip(root);
  overlay.addEventListener("pointermove", (evt) => {
    const rect = svg.getBoundingClientRect();
    const scale = width / rect.width;
    const px = (evt.clientX - rect.left) * scale;
    const idx = Math.round(((px - margin.left) / innerW) * (n - 1));
    const clamped = Math.max(0, Math.min(n - 1, idx));
    const cx = xAt(clamped);
    crosshair.setAttribute("x1", cx);
    crosshair.setAttribute("x2", cx);
    crosshair.style.opacity = 1;

    let rows = "";
    series.forEach((s) => {
      const p = s.points.find((pt) => pt.x === clamped);
      const val = p && p.y !== null ? yFormat(p.y) : "–";
      rows += `<div class="chart-tooltip-row">
        <span class="chart-tooltip-key" style="--key-color:${s.color}"></span>
        <span class="chart-tooltip-value"></span>
        <span class="chart-tooltip-label"></span>
      </div>`;
    });
    tip.innerHTML = `<div class="chart-tooltip-title"></div>${rows}`;
    tip.querySelector(".chart-tooltip-title").textContent = xLabels[clamped];
    const rows2 = tip.querySelectorAll(".chart-tooltip-row");
    series.forEach((s, i) => {
      const p = s.points.find((pt) => pt.x === clamped);
      const val = p && p.y !== null ? yFormat(p.y) : "–";
      rows2[i].querySelector(".chart-tooltip-value").textContent = val;
      rows2[i].querySelector(".chart-tooltip-label").textContent = s.name;
    });

    tip.hidden = false;
    const leftPct = (cx / width) * 100;
    tip.style.left = `${leftPct}%`;
    tip.style.top = `${margin.top}px`;
    tip.classList.toggle("chart-tooltip-flip", leftPct > 65);
  });
  overlay.addEventListener("pointerleave", () => {
    tip.hidden = true;
    crosshair.style.opacity = 0;
  });

  return { svg, width, height };
}

/**
 * Grouped column chart. `categories` = ["Jan", ...]
 * `series` = [{ name, color, values: [v, v, ...] }] (aligned to categories)
 */
export function renderGroupedBarChart(container, { categories, series, height = 260, yFormat = fmtCompact }) {
  container.innerHTML = "";
  const root = document.createElement("div");
  root.className = "chart-root";
  container.appendChild(root);

  const width = Math.max(container.clientWidth || 640, 320);
  const margin = { top: 16, right: 20, bottom: 28, left: 44 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const allValues = series.flatMap((s) => s.values.filter((v) => v !== null));
  const maxVal = niceMax(Math.max(1, ...allValues));
  const ticks = ticksFor(maxVal, 4);
  const n = categories.length;
  const groupW = innerW / n;
  const barGap = 2;
  const barW = Math.min(24, (groupW - 12 - barGap * (series.length - 1)) / series.length);
  const groupContentW = barW * series.length + barGap * (series.length - 1);

  const yAt = (v) => margin.top + innerH - (v / maxVal) * innerH;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height, class: "chart-svg" });

  for (const t of ticks) {
    const y = yAt(t);
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "chart-gridline" }));
    const label = svgEl("text", { x: margin.left - 8, y: y + 4, class: "chart-axis-label", "text-anchor": "end" });
    label.textContent = yFormat(t);
    svg.appendChild(label);
  }
  svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: yAt(0), y2: yAt(0), class: "chart-baseline" }));

  const tip = makeTooltip(root);

  for (let i = 0; i < n; i++) {
    const groupX = margin.left + i * groupW + (groupW - groupContentW) / 2;
    const label = svgEl("text", { x: margin.left + i * groupW + groupW / 2, y: height - 6, class: "chart-axis-label", "text-anchor": "middle" });
    label.textContent = categories[i];
    svg.appendChild(label);

    series.forEach((s, si) => {
      const v = s.values[i];
      if (v === null || v === undefined) return;
      const barX = groupX + si * (barW + barGap);
      const barY = yAt(v);
      const barH = Math.max(0, yAt(0) - barY);
      const r = Math.min(4, barW / 2, barH);
      const d = barH <= r
        ? `M ${barX} ${yAt(0)} L ${barX} ${barY} L ${barX + barW} ${barY} L ${barX + barW} ${yAt(0)} Z`
        : `M ${barX} ${yAt(0)}
           L ${barX} ${barY + r}
           Q ${barX} ${barY} ${barX + r} ${barY}
           L ${barX + barW - r} ${barY}
           Q ${barX + barW} ${barY} ${barX + barW} ${barY + r}
           L ${barX + barW} ${yAt(0)} Z`;
      const rect = svgEl("path", { d, fill: s.color, class: "chart-bar" });
      svg.appendChild(rect);

      const hit = svgEl("rect", {
        x: barX - barGap / 2, y: margin.top, width: barW + barGap, height: innerH, fill: "transparent",
      });
      hit.addEventListener("pointerenter", () => rect.classList.add("chart-bar-hover"));
      hit.addEventListener("pointerleave", () => {
        rect.classList.remove("chart-bar-hover");
        tip.hidden = true;
      });
      hit.addEventListener("pointermove", (evt) => {
        const rect2 = svg.getBoundingClientRect();
        const scale = width / rect2.width;
        const px = (evt.clientX - rect2.left) * scale;
        tip.innerHTML = `<div class="chart-tooltip-title"></div>
          <div class="chart-tooltip-row">
            <span class="chart-tooltip-key" style="--key-color:${s.color}"></span>
            <span class="chart-tooltip-value"></span>
            <span class="chart-tooltip-label"></span>
          </div>`;
        tip.querySelector(".chart-tooltip-title").textContent = `${categories[i]} ${s.name}`;
        tip.querySelector(".chart-tooltip-value").textContent = yFormat(v);
        tip.querySelector(".chart-tooltip-label").textContent = "solar generation (MU)";
        tip.hidden = false;
        const leftPct = (px / width) * 100;
        tip.style.left = `${leftPct}%`;
        tip.style.top = `${margin.top}px`;
        tip.classList.toggle("chart-tooltip-flip", leftPct > 65);
      });
      svg.appendChild(hit);
    });
  }

  root.appendChild(svg);
  return { svg, width, height };
}

export { fmtCompact };
