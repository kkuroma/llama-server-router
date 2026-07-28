// ECharts rendering. Every render reads the live palette, so a theme change
// only needs dispose + re-render (app.js handles that).
import { pal, gpuColor } from './colors.js';
import { fmtNum } from './format.js';

function areaGradient(color, topAlpha, botAlpha) {
  return new echarts.graphic.LinearGradient(0, 0, 0, 1, [
    { offset: 0, color: color + topAlpha },
    { offset: 1, color: color + botAlpha },
  ]);
}

export function makeCharts(els) {
  const opts = { renderer: 'canvas' };
  return {
    gpu: echarts.init(els.gpu, null, opts),
    history: echarts.init(els.history, null, opts),
  };
}

// Discrete gaussian kernel over the evenly-sampled history points; the
// truncated edges renormalize by the visible kernel mass.
function gaussianSmooth(points, sigma = 2) {
  if (points.length < 3) return points;
  const radius = Math.ceil(sigma * 3);
  const kernel = [];
  for (let k = -radius; k <= radius; k++) {
    kernel.push(Math.exp(-(k * k) / (2 * sigma * sigma)));
  }
  const out = [];
  for (let i = 0; i < points.length; i++) {
    let acc = 0, mass = 0;
    for (let k = -radius; k <= radius; k++) {
      const j = i + k;
      if (j < 0 || j >= points.length) continue;
      const w = kernel[k + radius];
      acc += points[j][1] * w;
      mass += w;
    }
    out.push([points[i][0], acc / mass]);
  }
  return out;
}

// Split a series where sampling stopped (gap > 3x the typical interval) so
// downtime renders as a visible hole: each contiguous run is smoothed on its
// own (the kernel can't bridge a gap) and a null marker breaks the line.
function smoothWithGaps(points, sigma = 2) {
  if (points.length < 2) return points;
  const deltas = [];
  for (let i = 1; i < points.length; i++) deltas.push(points[i][0] - points[i - 1][0]);
  const sorted = [...deltas].sort((a, b) => a - b);
  const threshold = sorted[Math.floor(sorted.length / 2)] * 3;
  const out = [];
  let seg = [points[0]];
  for (let i = 1; i < points.length; i++) {
    if (deltas[i - 1] > threshold) {
      out.push(...gaussianSmooth(seg, sigma));
      out.push([points[i - 1][0] + deltas[i - 1] / 2, null]);
      seg = [];
    }
    seg.push(points[i]);
  }
  out.push(...gaussianSmooth(seg, sigma));
  return out;
}

// All GPUs on four grids — utilization, VRAM, temperature, power — one line
// per GPU in its fixed color, sharing a single zoom slider (default window:
// last 30 min). temp/power histories may be absent from an older backend;
// those grids just render empty.
export function renderGpu(chart, gpus) {
  const p = pal();

  // Carry the zoom window across the 5s re-renders; first render with data
  // defaults to the last 30 minutes.
  const now = Date.now();
  let tMin = Infinity;
  for (const g of gpus) {
    if (g.util_history.length) tMin = Math.min(tMin, g.util_history[0][0] * 1000);
  }
  const span = tMin === Infinity ? 0 : now - tMin;
  let zStart = 0, zEnd = 100;
  const prev = chart.getOption();
  if (chart.__zoomInit && prev && prev.dataZoom && prev.dataZoom.length) {
    zStart = prev.dataZoom[0].start ?? 0;
    zEnd = prev.dataZoom[0].end ?? 100;
  } else if (span > 0) {
    zStart = Math.max(0, (1 - 1800000 / span) * 100);
    chart.__zoomInit = true;
  }

  const maxVram = Math.max(1, ...gpus.map(g => g.total_vram_mb || 0));
  const totals = [...new Set(gpus.map(g => g.total_vram_mb).filter(Boolean))];
  const limits = [...new Set(gpus.map(g => g.power_limit_w).filter(Boolean))];
  const maxPower = limits.length ? Math.ceil(Math.max(...limits) * 1.05) : undefined;
  const n = gpus.length;
  const GRID_LEFT = ['0%', '25.75%', '51.5%', '77.25%'];

  const titleStyle = { color: p.overlay1, fontSize: 10, fontWeight: 600 };
  const axisBase = {
    type: 'time',
    splitNumber: 4, // the grids are ~1/4 card wide; more ticks run together
    axisLine: { lineStyle: { color: p.surface1 } },
    axisTick: { show: false },
    axisLabel: { color: p.overlay0, fontSize: 9, formatter: '{HH}:{mm}', hideOverlap: true },
    splitLine: { show: false },
  };
  const yBase = {
    type: 'value',
    splitLine: { lineStyle: { color: p.surface0, type: 'dashed' } },
    axisLine: { show: false }, axisTick: { show: false },
  };
  const lineBase = (g) => ({
    name: `GPU ${g.index}`, type: 'line', showSymbol: false, smooth: 0.25,
    lineStyle: { color: gpuColor(g.index), width: 1.5 },
    itemStyle: { color: gpuColor(g.index) },
  });

  const capLines = (values) => ({
    markLine: {
      silent: true, symbol: 'none',
      data: values.map(v => ({ yAxis: v, lineStyle: { color: p.red, type: 'dashed', width: 1, opacity: 0.5 } })),
      label: { show: false },
    },
  });
  const metricSeries = (axis, key, caps) => gpus.map((g, i) => ({
    ...lineBase(g), xAxisIndex: axis, yAxisIndex: axis,
    data: smoothWithGaps((g[key] || []).map(([t, v]) => [t * 1000, v])),
    ...(i === 0 && caps && caps.length ? capLines(caps) : {}),
  }));

  chart.setOption({
    animation: false,
    textStyle: { color: p.subtext0, fontFamily: 'ui-monospace, monospace', fontSize: 10 },
    title: [
      { text: 'UTILIZATION', left: GRID_LEFT[0], top: 0, textStyle: titleStyle },
      { text: 'VRAM', left: GRID_LEFT[1], top: 0, textStyle: titleStyle },
      { text: 'TEMP', left: GRID_LEFT[2], top: 0, textStyle: titleStyle },
      { text: 'POWER', left: GRID_LEFT[3], top: 0, textStyle: titleStyle },
    ],
    tooltip: {
      trigger: 'axis',
      backgroundColor: p.surface0, borderColor: p.surface1, borderWidth: 1,
      textStyle: { color: p.text, fontSize: 11 },
      formatter: params => {
        // The gap markers are [t, null] points; skip them or the tooltip
        // throws mid-mousemove and wedges hovering entirely.
        const live = (params || []).filter(s => s.value && s.value[1] != null);
        if (!live.length) return '';
        const head = new Date(live[0].value[0]).toLocaleTimeString();
        let body = '';
        for (const s of live) {
          const kind = Math.floor(s.seriesIndex / n);
          const v = kind === 0 ? `${s.value[1].toFixed(0)}%`
            : kind === 1 ? `${(s.value[1] / 1024).toFixed(1)} GB`
            : kind === 2 ? `${s.value[1].toFixed(0)}°C`
            : `${s.value[1].toFixed(0)}W`;
          const name = (gpus[s.seriesIndex % n] || {}).name || '';
          body += `<div style="display:flex;justify-content:space-between;gap:16px">` +
                  `<span>${s.marker}${s.seriesName} <span style="opacity:.55">${name}</span></span><b>${v}</b></div>`;
        }
        return `<div style="opacity:.6;margin-bottom:3px">${head}</div>${body}`;
      },
    },
    grid: GRID_LEFT.map(left => ({ top: 22, bottom: 44, left, width: '22.75%', containLabel: true })),
    xAxis: GRID_LEFT.map((_, i) => ({ ...axisBase, gridIndex: i })),
    yAxis: [
      { ...yBase, gridIndex: 0, min: 0, max: 100, axisLabel: { color: p.overlay0, fontSize: 9, formatter: '{value}%' } },
      { ...yBase, gridIndex: 1, min: 0, max: Math.ceil(maxVram), axisLabel: { color: p.overlay0, fontSize: 9, formatter: v => `${(v / 1024).toFixed(0)}G` } },
      { ...yBase, gridIndex: 2, min: 0, max: 100, axisLabel: { color: p.overlay0, fontSize: 9, formatter: '{value}°' } },
      { ...yBase, gridIndex: 3, min: 0, max: maxPower, axisLabel: { color: p.overlay0, fontSize: 9, formatter: '{value}W' } },
    ],
    dataZoom: [
      { type: 'inside', xAxisIndex: [0, 1, 2, 3], filterMode: 'none', start: zStart, end: zEnd },
      {
        type: 'slider', xAxisIndex: [0, 1, 2, 3], filterMode: 'none', height: 16, bottom: 2,
        left: 4, right: 4, start: zStart, end: zEnd,
        borderColor: 'transparent', backgroundColor: p.surface0 + '66',
        fillerColor: p.blue + '22',
        handleStyle: { color: p.blue, borderColor: p.blue },
        moveHandleStyle: { color: p.surface1 },
        dataBackground: { lineStyle: { color: p.surface1, width: 1 }, areaStyle: { color: p.surface0 } },
        selectedDataBackground: { lineStyle: { color: p.blue, opacity: 0.6 }, areaStyle: { color: p.blue, opacity: 0.12 } },
        textStyle: { color: p.overlay0, fontSize: 9 },
        labelFormatter: (v) => new Date(v).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ],
    series: [
      ...metricSeries(0, 'util_history'),
      ...metricSeries(1, 'vram_history', totals),
      ...metricSeries(2, 'temp_history'),
      ...metricSeries(3, 'power_history', limits),
    ],
  }, true);
}

// Hierarchical tick labels so axis resolution follows the zoom; min/max pin
// the historyBounds extent (first data point .. now).
function historyTimeAxis(p, bounds) {
  return {
    type: 'time',
    min: bounds[0],
    max: bounds[1],
    axisLine: { lineStyle: { color: p.surface1 } },
    axisTick: { show: false },
    axisLabel: {
      color: p.overlay0, fontSize: 9, hideOverlap: true,
      formatter: {
        year: '{yyyy}', month: '{MMM}', day: '{MMM} {d}',
        hour: '{HH}:{mm}', minute: '{HH}:{mm}', second: '{HH}:{mm}:{ss}',
        millisecond: '{HH}:{mm}:{ss}', none: '{yyyy}-{MM}-{dd}',
      },
    },
    splitLine: { show: false },
  };
}

// Scroll/drag zoom + a styled bottom slider. filterMode 'none' zooms the axis
// without dropping points, so narrowing in gains resolution instead of
// thinning the series.
function historyZoom(p) {
  return [
    { type: 'inside', filterMode: 'none' },
    {
      type: 'slider', filterMode: 'none', height: 18, bottom: 26,
      borderColor: 'transparent', backgroundColor: p.surface0 + '66',
      fillerColor: p.blue + '22',
      handleStyle: { color: p.blue, borderColor: p.blue },
      moveHandleStyle: { color: p.surface1 },
      dataBackground: { lineStyle: { color: p.surface1, width: 1 }, areaStyle: { color: p.surface0 } },
      selectedDataBackground: { lineStyle: { color: p.blue, opacity: 0.6 }, areaStyle: { color: p.blue, opacity: 0.12 } },
      textStyle: { color: p.overlay0, fontSize: 9 },
      labelFormatter: (v) => {
        const d = new Date(v);
        return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + '\n' +
               d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      },
    },
  ];
}

// The x extent the bars live in: starts at the first real data point (clamped
// to the selected window) and always ends at now, so a recent quiet period
// stays visible but the axis never spans dataless time before the first row.
function historyBounds(rows, rangeSec) {
  const now = Date.now();
  let first = Infinity;
  for (const x of rows) {
    const t = x.request_time * 1000;
    if (t < first) first = t;
  }
  const winStart = rangeSec ? now - rangeSec * 1000 : null;
  if (first === Infinity) return [winStart ?? now - 3600000, now];
  return [winStart ? Math.max(winStart, first) : first, now];
}

// ECharts sizes bars from the smallest gap between adjacent points, so bursts
// pin them at 1px no matter the zoom; size them from the visible count
// instead, capped at the 25th-percentile gap between visible neighbors so
// zoomed-in bars don't paint over each other (coincident timestamps still
// overlay exactly, which reads as one bar). sortedTs must be ascending.
function historyBarWidth(chart, sortedTs, bounds, startPct, endPct) {
  const [a, b] = bounds;
  const w0 = a + (b - a) * startPct / 100;
  const w1 = a + (b - a) * endPct / 100;
  let lo = 0, hi = sortedTs.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (sortedTs[m] < w0) lo = m + 1; else hi = m; }
  let hiEnd = sortedTs.length;
  let lo2 = lo;
  while (lo2 < hiEnd) { const m = (lo2 + hiEnd) >> 1; if (sortedTs[m] <= w1) lo2 = m + 1; else hiEnd = m; }
  const n = lo2 - lo;
  const px = Math.max(chart.getWidth() - 80, 100);
  const countWidth = (px / Math.max(n, 1)) * 0.6;
  const gaps = [];
  for (let i = lo + 1; i < lo2; i++) {
    const d = sortedTs[i] - sortedTs[i - 1];
    if (d > 0) gaps.push(d);
  }
  let gapWidth = Infinity;
  if (gaps.length) {
    gaps.sort((x, y) => x - y);
    const winSpan = (w1 - w0) || 1;
    gapWidth = (gaps[Math.floor(gaps.length * 0.25)] / winSpan) * px * 0.8;
  }
  return Math.max(1, Math.min(20, Math.floor(Math.min(countWidth, gapWidth))));
}

// A datazoom event carries percents (slider) or axis values (inside zoom);
// normalize to percents of the axis bounds.
function zoomPercents(z, bounds) {
  if (z.start != null) return [z.start, z.end];
  if (z.startValue != null) {
    const [a, b] = bounds;
    const span = (b - a) || 1;
    return [((z.startValue - a) / span) * 100, ((z.endValue - a) / span) * 100];
  }
  return [0, 100];
}

function historyTooltipFormatter(p) {
  return (params) => {
    if (!params || !params.length) return '';
    const head = new Date(params[0].value[0]).toLocaleString([], {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
    let body = '';
    for (const s of params) {
      body += `<div style="display:flex;justify-content:space-between;gap:16px">` +
              `<span>${s.marker}${s.seriesName}</span><b>${fmtNum(s.value[1])}</b></div>`;
    }
    return `<div style="opacity:.6;margin-bottom:3px">${head}</div>${body}`;
  };
}

export function renderHistory(chart, rows, mode, rangeSec) {
  const p = pal();
  // Carry the zoom window across re-renders (polling redraws with notMerge,
  // which would otherwise snap the zoom back to 100% every 5s).
  let zStart = 0, zEnd = 100;
  const prev = chart.getOption();
  if (prev && prev.dataZoom && prev.dataZoom.length) {
    zStart = prev.dataZoom[0].start ?? 0;
    zEnd = prev.dataZoom[0].end ?? 100;
  }
  const zoomOpts = historyZoom(p).map(z => ({ ...z, start: zStart, end: zEnd }));
  const bounds = historyBounds(rows, rangeSec);
  const common = {
    animation: false,
    textStyle: { color: p.subtext0, fontFamily: 'ui-monospace, monospace', fontSize: 10 },
    grid: { top: 30, right: 16, bottom: 62, left: 8, containLabel: true },
    legend: {
      data: ['Prompt', 'Predicted'], top: 2, right: 4,
      textStyle: { color: p.overlay0, fontSize: 10 },
      icon: 'roundRect', itemWidth: 10, itemHeight: 6,
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: p.surface0, borderColor: p.surface1, borderWidth: 1,
      textStyle: { color: p.text, fontSize: 11 },
      axisPointer: { type: 'line', lineStyle: { color: p.overlay0, opacity: 0.35, type: 'dashed' } },
      formatter: historyTooltipFormatter(p),
    },
    xAxis: historyTimeAxis(p, bounds),
    yAxis: {
      type: 'value',
      axisLabel: { color: p.overlay0, fontSize: 9, formatter: fmtNum },
      splitLine: { lineStyle: { color: p.surface0, type: 'dashed' } },
      axisLine: { show: false }, axisTick: { show: false },
    },
    dataZoom: zoomOpts,
  };

  chart.off('datazoom');
  clearTimeout(chart.__bwTimer);
  chart.__bwTimer = null;
  if (mode === 'cumulative') {
    const sorted = [...rows].sort((a, b) => a.request_time - b.request_time);
    let cumPrompt = 0, cumPredicted = 0;
    const promptData = [], predictedData = [];
    for (const item of sorted) {
      cumPrompt += item.prompt_n;
      cumPredicted += item.predicted_n;
      const t = item.request_time * 1000;
      promptData.push([t, cumPrompt]);
      predictedData.push([t, cumPredicted]);
    }
    chart.setOption({
      ...common,
      series: [
        {
          name: 'Prompt', type: 'line', stack: 'cumulative', showSymbol: false, smooth: 0.2,
          lineStyle: { color: p.blue, width: 1.5 },
          areaStyle: { color: areaGradient(p.blue, '35', '08') },
          data: promptData,
        },
        {
          name: 'Predicted', type: 'line', stack: 'cumulative', showSymbol: false, smooth: 0.2,
          lineStyle: { color: p.green, width: 1.5 },
          areaStyle: { color: areaGradient(p.green, '25', '05') },
          data: predictedData,
        },
      ],
    }, true);
  } else {
    const sortedTs = rows.map(x => x.request_time * 1000).sort((a, b) => a - b);
    let lastWidth = historyBarWidth(chart, sortedTs, bounds, zStart, zEnd);
    const width = lastWidth;
    chart.setOption({
      ...common,
      series: [
        {
          name: 'Prompt', type: 'bar', stack: 'tokens', barWidth: width,
          itemStyle: { color: p.blue },
          data: rows.map(x => [x.request_time * 1000, x.prompt_n]),
        },
        {
          name: 'Predicted', type: 'bar', stack: 'tokens', barWidth: width,
          itemStyle: { color: p.green, borderRadius: [2, 2, 0, 0] },
          data: rows.map(x => [x.request_time * 1000, x.predicted_n]),
        },
      ],
    }, true);
    // Trailing throttle: datazoom fires per wheel tick / drag frame, and
    // recomputing + re-applying the width that often janks the zoom. Only
    // touch the option when the width actually changes.
    let pendingZoom = null;
    chart.on('datazoom', (e) => {
      pendingZoom = (e.batch && e.batch[0]) || e;
      if (chart.__bwTimer) return;
      chart.__bwTimer = setTimeout(() => {
        chart.__bwTimer = null;
        const [s, en] = zoomPercents(pendingZoom, bounds);
        const w = historyBarWidth(chart, sortedTs, bounds, s, en);
        if (w !== lastWidth) {
          lastWidth = w;
          chart.setOption({ series: [{ barWidth: w }, { barWidth: w }] });
        }
      }, 150);
    });
  }
}
