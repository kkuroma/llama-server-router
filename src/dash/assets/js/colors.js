// Palette lookups for model/GPU/status color-coding. All keys exist in every
// theme/variant (see theme.js), and pal() reads the live palette so callers
// pick up theme changes on their next render.
const T = window.LRTheme;

export const pal = () => T.colors();

// Models cycle these keys in first-seen order, so a model keeps one color
// across the list, the charts, and the history table.
const MODEL_COLOR_KEYS = [
  'blue', 'green', 'mauve', 'peach', 'teal', 'pink', 'sapphire',
  'yellow', 'lavender', 'flamingo', 'sky', 'maroon', 'rosewater', 'red',
];
const modelColorMap = {};
let modelColorIdx = 0;

export function modelColor(modelId) {
  if (!modelColorMap[modelId]) {
    modelColorMap[modelId] = MODEL_COLOR_KEYS[modelColorIdx % MODEL_COLOR_KEYS.length];
    modelColorIdx++;
  }
  return pal()[modelColorMap[modelId]];
}

// Fixed key per physical GPU index, so a GPU is drawn in the same color
// everywhere it appears.
const GPU_COLOR_KEYS = [
  'sky', 'peach', 'green', 'mauve', 'yellow', 'teal', 'pink', 'flamingo', 'blue', 'lavender',
];

export function gpuColor(idx) {
  return pal()[GPU_COLOR_KEYS[idx % GPU_COLOR_KEYS.length]];
}

export function statusColors() {
  const p = pal();
  return {
    idle: p.blue, serving: p.green, swapping: p.mauve, starting: p.yellow,
    stopping: p.peach, inactive: p.surface2, error: p.red,
  };
}
