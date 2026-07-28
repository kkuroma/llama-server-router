// Dashboard app: state, polling, and the wiring between the router API, the
// model list, and the charts.
import * as api from './api.js';
import * as charts from './charts.js';
import { pal, modelColor, gpuColor, statusColors } from './colors.js';
import { parseModel } from './models.js';
import { fmtNum, formatTime } from './format.js';

const T = window.LRTheme;
const { createApp, ref, reactive, computed, onMounted, onUnmounted, nextTick } = Vue;

createApp({
  setup() {
    // "Aa" popover state, mirroring LRTheme.state for active-highlighting.
    const ui = reactive({ ...T.state });
    const themeNames = T.themeNames;
    const modeOptions = [
      { value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' }, { value: 'system', label: 'Auto' },
    ];
    const sizeOptions = [
      { value: 'small', label: 'S' }, { value: 'medium', label: 'M' },
      { value: 'large', label: 'L' }, { value: 'xlarge', label: 'XL' },
    ];
    const swatchStyle = (name) => {
      const t = T.THEMES[name];
      return { background: `linear-gradient(135deg, ${t.light.primary} 50%, ${t.dark.primary} 50%)` };
    };
    const selectTheme = (n) => { T.setTheme(n); ui.theme = T.state.theme; };
    const selectVariant = (m) => { T.setVariant(m); ui.variantMode = T.state.variantMode; };
    const selectFontSize = (s) => { T.setFontSize(s); ui.fontSize = T.state.fontSize; };

    const routerStatus = ref('unknown');
    const routerPorts = ref([]);
    const numGpus = ref(0);
    const routerLoading = ref(null); // 'start' | 'stop' | 'restart' | null
    const models = ref([]);
    const gpus = ref([]);
    const history = ref([]);
    const historyModel = ref('');
    const historySort = ref({ key: 'request_time', asc: false });
    const historyChartMode = ref('static');
    const loadingModels = reactive({});

    // How far back to load; null means the whole DB. The chart's zoom slider
    // refines within the loaded window.
    const RANGES = [
      { key: '1h', label: '1H', sec: 3600 },
      { key: '6h', label: '6H', sec: 21600 },
      { key: '24h', label: '24H', sec: 86400 },
      { key: '7d', label: '7D', sec: 604800 },
      { key: '30d', label: '30D', sec: 2592000 },
      { key: 'all', label: 'All', sec: null },
    ];
    const historyRange = ref('24h');

    const gpuChart = ref(null);
    const historyChart = ref(null);

    let chartSet = null;
    let pollTimer = null;

    // Bumped on every theme change so palette-derived computeds re-run.
    const paletteTick = ref(0);
    const pal_blue = computed(() => (paletteTick.value, pal().blue));
    const pal_green = computed(() => (paletteTick.value, pal().green));

    const statusDotStyle = computed(() => {
      paletteTick.value;
      return { background: statusColors()[routerStatus.value] || pal().surface2 };
    });
    const statusPulse = computed(() => ['serving', 'swapping', 'starting', 'error'].includes(routerStatus.value));

    // Averages generation speed only over requests that produced tokens, so
    // idle/empty rows don't skew it.
    const historyStats = computed(() => {
      const h = history.value;
      let prompt = 0, predicted = 0, speedSum = 0, speedN = 0;
      for (const x of h) {
        prompt += x.prompt_n;
        predicted += x.predicted_n;
        const dur = x.response_time - x.request_time;
        if (dur > 0 && x.predicted_n > 0) { speedSum += x.predicted_n / dur; speedN++; }
      }
      return { reqs: h.length, prompt, predicted, tokPerSec: speedN ? speedSum / speedN : 0 };
    });

    function gpuName(idx) {
      const g = gpus.value.find(x => x.index === idx);
      return g ? g.name : '';
    }

    // One dot per physical GPU (falling back to the model's own span when the
    // total count is unknown), flagging the GPUs the model is pinned to.
    function gpuDots(m) {
      const assigned = new Set(m.gpus || []);
      const total = numGpus.value || gpuLegend.value.length ||
        ((m.gpus && m.gpus.length) ? Math.max(...m.gpus) + 1 : 0);
      const out = [];
      for (let g = 0; g < total; g++) {
        out.push({ idx: g, assigned: assigned.has(g), color: gpuColor(g) });
      }
      return out;
    }

    // GPU digits: GPU color when the model is pinned to that GPU, grey when not.
    function gpuNumStyle(d) {
      return { color: d.assigned ? d.color : pal().overlay0 };
    }

    // Left stripe: model color when loaded, grey when not.
    function stripColor(m) {
      return m.loaded ? modelColor(m.id) : pal().overlay0;
    }

    function tagStyle(t) {
      const p = pal();
      if (t.kind === 'param') return { color: p.blue, borderColor: p.blue, background: p.blue + '1f' };
      if (t.kind === 'active') return { color: p.mauve, borderColor: p.mauve, background: p.mauve + '1f' };
      return { color: p.subtext0, borderColor: 'transparent', background: p.surface1 };
    }

    // Every physical GPU when the count is known, else the union of GPUs any
    // model is pinned to.
    const gpuLegend = computed(() => {
      paletteTick.value;
      if (numGpus.value > 0) return Array.from({ length: numGpus.value }, (_, i) => i);
      const s = new Set();
      for (const m of models.value) for (const g of (m.gpus || [])) s.add(g);
      return [...s].sort((a, b) => a - b);
    });

    const sortedHistory = computed(() => {
      const arr = [...history.value];
      const { key, asc } = historySort.value;
      arr.sort((a, b) => {
        let va, vb;
        if (key === 'duration') {
          va = a.response_time - a.request_time;
          vb = b.response_time - b.request_time;
        } else {
          va = a[key];
          vb = b[key];
        }
        if (typeof va === 'string') return asc ? va.localeCompare(vb) : vb.localeCompare(va);
        return asc ? va - vb : vb - va;
      });
      return arr.slice(0, 100); // keep the DOM table light
    });

    function sortHistory(key) {
      if (historySort.value.key === key) {
        historySort.value.asc = !historySort.value.asc;
      } else {
        historySort.value = { key, asc: true };
      }
    }

    // --- data fetching ---

    async function fetchStatus() {
      try {
        const data = await api.getStatus();
        routerStatus.value = data.status;
        routerPorts.value = data.ports || [];
        if (data.num_gpus) numGpus.value = data.num_gpus;
      } catch {}
    }

    async function fetchModels() {
      try {
        const data = await api.getModels();
        const list = data.data || data.models || [];
        models.value = list.map(m => {
          const parsed = parseModel(m.id);
          return {
            id: m.id,
            displayName: parsed.name,
            // param/active counts ride line 1 with the name; plain
            // descriptors drop to line 2 with the GPU dots.
            paramTags: parsed.tags.filter(t => t.kind !== 'plain'),
            plainTags: parsed.tags.filter(t => t.kind === 'plain'),
            loaded: m.status?.value === 'loaded',
            gpus: m.gpus || [],
            loading: !!loadingModels[m.id],
          };
        });
        // Pre-assign colors so first-seen order stays stable.
        for (const m of models.value) modelColor(m.id);
      } catch {}
    }

    async function fetchGpu() {
      try {
        const data = await api.getGpu();
        gpus.value = data.gpus || [];
        updateGpuCharts();
      } catch {}
    }

    async function fetchHistory() {
      try {
        // Load only the selected window; cap rows so a huge DB can't stall.
        const params = new URLSearchParams({ limit: '10000' });
        if (historyModel.value) params.set('model', historyModel.value);
        const r = RANGES.find(x => x.key === historyRange.value);
        if (r && r.sec) params.set('since', String(Math.floor(Date.now() / 1000) - r.sec));
        history.value = await api.getHistory(params);
        updateHistoryChart();
      } catch {}
    }

    // --- actions ---

    function setHistoryRange(key) {
      historyRange.value = key;
      fetchHistory();
    }

    function setChartMode(mode) {
      historyChartMode.value = mode;
      updateHistoryChart();
    }

    async function routerAction(action) {
      routerLoading.value = action;
      try { await api.routerCommand(action); } catch {}
      await fetchStatus();
      await fetchModels();
      routerLoading.value = null;
    }

    async function toggleModel(m) {
      loadingModels[m.id] = true;
      m.loading = true;
      try { await api.modelAction(m.loaded ? 'unload' : 'load', m.id); } catch {}
      delete loadingModels[m.id];
      await fetchModels();
    }

    async function resetHistory() {
      try { await api.clearHistory(); } catch {}
      await fetchHistory();
    }

    // --- charts ---

    function updateGpuCharts() {
      if (!chartSet) return;
      charts.renderGpu(chartSet.gpu, gpus.value);
    }

    function updateHistoryChart() {
      if (!chartSet) return;
      const r = RANGES.find(x => x.key === historyRange.value);
      charts.renderHistory(chartSet.history, history.value, historyChartMode.value, r ? r.sec : null);
    }

    function initCharts() {
      chartSet = charts.makeCharts({ gpu: gpuChart.value, history: historyChart.value });
      updateGpuCharts();
      updateHistoryChart();
    }

    function disposeCharts() {
      if (chartSet) Object.values(chartSet).forEach(c => c.dispose());
      chartSet = null;
    }

    function rebuildCharts() {
      disposeCharts();
      nextTick(async () => {
        initCharts();
        await Promise.all([fetchGpu(), fetchHistory()]);
      });
    }

    // --- lifecycle ---

    async function pollAll() {
      await Promise.all([fetchStatus(), fetchModels(), fetchGpu(), fetchHistory()]);
    }

    onMounted(async () => {
      await nextTick();
      initCharts();
      await pollAll();
      pollTimer = setInterval(pollAll, 5000);
      window.addEventListener('resize', () => {
        if (chartSet) Object.values(chartSet).forEach(c => c.resize());
      });
      // Recolor charts + palette-bound styles when the theme changes.
      T.onChange(() => { paletteTick.value++; rebuildCharts(); });
    });

    onUnmounted(() => {
      clearInterval(pollTimer);
      disposeCharts();
    });

    return {
      ui, themeNames, modeOptions, sizeOptions, swatchStyle, selectTheme, selectVariant, selectFontSize,
      routerStatus, routerPorts, routerLoading, statusDotStyle, statusPulse,
      models, gpus, gpuName,
      gpuLegend, gpuDots, gpuNumStyle, stripColor, tagStyle, modelColor, gpuColor, toggleModel,
      history, historyModel, historySort, historyChartMode, sortedHistory, historyStats,
      ranges: RANGES, historyRange, setHistoryRange, setChartMode,
      fetchHistory, resetHistory, sortHistory, routerAction,
      fmtNum, formatTime, pal_blue, pal_green,
      gpuChart, historyChart,
    };
  },
}).mount('#app');
