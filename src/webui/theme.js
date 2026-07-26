/**
 * Shared design-language runtime for the router web UIs (dashboard + translate).
 *
 * Ports the shunka-shuutou (春夏秋冬) seasonal palette system from kuroma's
 * personal site: four themes (Haruhana/Natsumikan/Akiba/Fuyuyuki), each with a
 * light + dark variant, applied as CSS custom properties on <html>. The daisyUI
 * fallback variables in theme.css alias these, so every daisyUI component (cards,
 * buttons, selects, badges) recolors from one palette source.
 *
 * `window.LRTheme` is framework-agnostic: it owns state + persistence and hands
 * the Vue apps the palette (for echarts + model color-coding) and setters for
 * the "Aa" appearance popover. State lives in localStorage under `llama-router-ui`.
 */

const THEMES = {
  'Haruhana': { // spring
    light: {
      base: '#EFF0F8', mantle: '#FFFFFF', crust: '#e1e2ed', text: '#252535',
      subtext1: '#393947', subtext0: '#565662', surface2: '#C8C8D8', surface1: '#d4d4e2',
      surface0: '#dddeea', overlay2: '#565662', overlay1: '#848491', overlay0: '#a0a0af',
      primary: '#C04868', secondary: '#6248B8', tertiary: '#2A8A60', red: '#B00020',
      maroon: '#8F3A47', peach: '#99400d', yellow: '#8A6A00', green: '#2A8A60',
      teal: '#206878', sky: '#1E7AA8', sapphire: '#286494', blue: '#2458B8',
      lavender: '#5C64C4', mauve: '#6248B8', pink: '#89246c', flamingo: '#B25A48', rosewater: '#A05464'
    },
    dark: {
      base: '#0C0F1E', mantle: '#171D36', crust: '#090B15', text: '#D4D0EA',
      subtext1: '#babbd3', subtext0: '#8B95A8', surface2: '#424D68', surface1: '#2B3758',
      surface0: '#232E4C', overlay2: '#8B95A8', overlay1: '#6a758b', overlay0: '#58637b',
      primary: '#F07898', secondary: '#B4A2E8', tertiary: '#6FC898', red: '#FF5370',
      maroon: '#D46A79', peach: '#fb9e6c', yellow: '#F8D06A', green: '#6FC898',
      teal: '#6BCAD8', sky: '#8AD4F0', sapphire: '#76baec', blue: '#7AA2F7',
      lavender: '#AEB5F2', mauve: '#B4A2E8', pink: '#F5A8D0', flamingo: '#F0B39E', rosewater: '#E8C4C0'
    }
  },
  'Natsumikan': { // summer
    light: {
      base: '#EEF0F5', mantle: '#FFFFFF', crust: '#e1e4eb', text: '#272B32',
      subtext1: '#3a3e47', subtext0: '#565B66', surface2: '#C8CDD8', surface1: '#d3d8e1',
      surface0: '#dde0e8', overlay2: '#565B66', overlay1: '#848994', overlay0: '#a0a5b0',
      primary: '#C15A20', secondary: '#8F5A00', tertiary: '#006070', red: '#B00020',
      maroon: '#8F3A47', peach: '#9c360d', yellow: '#8F5A00', green: '#4A7C59',
      teal: '#006070', sky: '#2878A0', sapphire: '#146c88', blue: '#2A5CB8',
      lavender: '#5A62C0', mauve: '#7C4DAA', pink: '#962665', flamingo: '#B05A40', rosewater: '#A25668'
    },
    dark: {
      base: '#0D1017', mantle: '#171D26', crust: '#0A0C12', text: '#D1D1C7',
      subtext1: '#babcb9', subtext0: '#8E959E', surface2: '#464B5D', surface1: '#253040',
      surface0: '#242C3A', overlay2: '#8E959E', overlay1: '#6e7481', overlay0: '#5c6170',
      primary: '#F5803E', secondary: '#FFB454', tertiary: '#39BAE6', red: '#FF5370',
      maroon: '#D9707E', peach: '#ff8d5f', yellow: '#FFB454', green: '#AAD94C',
      teal: '#4DD0B0', sky: '#6BD2F0', sapphire: '#4FA6E8', blue: '#73B8FF',
      lavender: '#A6ACF0', mauve: '#C792EA', pink: '#F58EC0', flamingo: '#F2A788', rosewater: '#E8C0B8'
    }
  },
  'Akiba': { // autumn
    light: {
      base: '#F0EAE4', mantle: '#FFFFFF', crust: '#e8ded7', text: '#2A1E18',
      subtext1: '#40322c', subtext0: '#60504A', surface2: '#D8C8C0', surface1: '#dfd2cb',
      surface0: '#e5dbd4', overlay2: '#60504A', overlay1: '#908079', overlay0: '#ae9e97',
      primary: '#902020', secondary: '#806000', tertiary: '#507840', red: '#A82828',
      maroon: '#7A3030', peach: '#86460d', yellow: '#806000', green: '#507840',
      teal: '#207090', sky: '#3088B8', sapphire: '#247498', blue: '#2A58A8',
      lavender: '#5E64B8', mauve: '#6048A0', pink: '#783460', flamingo: '#A05A3C', rosewater: '#9A5A64'
    },
    dark: {
      base: '#141210', mantle: '#221D1A', crust: '#0F0D0B', text: '#DDD0C0',
      subtext1: '#c6b8aa', subtext0: '#9C8C80', surface2: '#484038', surface1: '#403530',
      surface0: '#362D28', overlay2: '#9C8C80', overlay1: '#766a60', overlay0: '#61574e',
      primary: '#C84040', secondary: '#DCA561', tertiary: '#98BB6C', red: '#C84040',
      maroon: '#E86060', peach: '#d47d54', yellow: '#DCA561', green: '#98BB6C',
      teal: '#82C8AE', sky: '#90CCE0', sapphire: '#6AA5D8', blue: '#82AAF0',
      lavender: '#A8AEE0', mauve: '#957FB8', pink: '#D8879C', flamingo: '#E0A182', rosewater: '#D8B8A8'
    }
  },
  'Fuyuyuki': { // winter
    light: {
      base: '#EDF2F8', mantle: '#FFFFFF', crust: '#dde5ed', text: '#202530',
      subtext1: '#333946', subtext0: '#505868', surface2: '#C0CCD8', surface1: '#ced7e2',
      surface0: '#d9e1ea', overlay2: '#505868', overlay1: '#7d8695', overlay0: '#99a3b1',
      primary: '#2878C8', secondary: '#186878', tertiary: '#4060A8', red: '#B00020',
      maroon: '#8F3A47', peach: '#933e0d', yellow: '#806800', green: '#287848',
      teal: '#186878', sky: '#2E86B0', sapphire: '#2070a0', blue: '#2878C8',
      lavender: '#5A68C8', mauve: '#6A4CB0', pink: '#783064', flamingo: '#A85C48', rosewater: '#9A5870'
    },
    dark: {
      base: '#0B0E18', mantle: '#141C30', crust: '#080A10', text: '#CDD8E8',
      subtext1: '#b3c0d1', subtext0: '#8492A6', surface2: '#384860', surface1: '#273654',
      surface0: '#202D46', overlay2: '#8492A6', overlay1: '#627186', overlay0: '#4f5e75',
      primary: '#78B4E8', secondary: '#7CD4DC', tertiary: '#A0B8E8', red: '#FF5370',
      maroon: '#D97080', peach: '#e59970', yellow: '#D4C870', green: '#7CD4A0',
      teal: '#7CD4DC', sky: '#94DCF0', sapphire: '#7ac4e2', blue: '#78B4E8',
      lavender: '#A0B8E8', mauve: '#B49CE8', pink: '#E8A0C8', flamingo: '#EFB49A', rosewater: '#E8C8C8'
    }
  }
};

const FONT_SIZES = { small: '14px', medium: '16px', large: '18px', xlarge: '20px' };
const STORAGE_KEY = 'llama-router-ui';
const DEFAULT_THEME = 'Natsumikan';

/**
 * Reads persisted UI state, tolerating corruption and migrating the legacy
 * `llama-router-theme` (mocha/latte) key from the old Catppuccin dashboard.
 *
 * @returns {{theme: string, variantMode: string, fontSize: string}} saved state
 */
function loadState() {
  let s = {};
  try { s = JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch (e) { /* corrupted */ }
  if (!s.variantMode) {
    // Migrate the old two-theme toggle: mocha -> dark, latte -> light.
    const legacy = localStorage.getItem('llama-router-theme');
    if (legacy === 'latte') s.variantMode = 'light';
    else if (legacy === 'mocha') s.variantMode = 'dark';
  }
  return {
    theme: (s.theme && THEMES[s.theme]) ? s.theme : DEFAULT_THEME,
    variantMode: ['light', 'dark', 'system'].includes(s.variantMode) ? s.variantMode : 'dark',
    fontSize: FONT_SIZES[s.fontSize] ? s.fontSize : 'medium',
  };
}

const LRTheme = {
  THEMES,
  themeNames: Object.keys(THEMES),
  fontSizes: FONT_SIZES,
  state: loadState(),
  _listeners: [],

  /**
   * Resolves the effective light/dark variant, honoring the OS preference
   * when the mode is 'system'.
   *
   * @returns {'light'|'dark'} the concrete variant to render
   */
  variant() {
    if (this.state.variantMode === 'system') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return this.state.variantMode;
  },

  /**
   * Returns the active palette (named colors for charts + chrome vars).
   *
   * @returns {Object<string,string>} color name -> hex for the current theme/variant
   */
  colors() {
    return THEMES[this.state.theme][this.variant()];
  },

  /**
   * Writes the active palette to <html> CSS variables and sets the base font
   * size, then persists state. theme.css maps daisyUI fallbacks onto these, so
   * this single call reskins the whole page.
   *
   * @returns {void}
   */
  apply() {
    const root = document.documentElement;
    const palette = this.colors();
    for (const [name, value] of Object.entries(palette)) {
      root.style.setProperty(`--${name}`, value);
    }
    root.style.setProperty('--font-size-base', FONT_SIZES[this.state.fontSize] || '16px');
    root.style.setProperty('color-scheme', this.variant());
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(this.state)); } catch (e) { /* full/blocked */ }
  },

  /**
   * Switches the seasonal palette, applies it, persists, and notifies listeners.
   *
   * Unknown theme names are ignored so a stale saved value can't break the page.
   *
   * @param {string} name  - the theme to switch to (a key of THEMES)
   * @returns {void}
   */
  setTheme(name) { if (THEMES[name]) { this.state.theme = name; this.apply(); this._emit(); } },

  /**
   * Switches the color variant mode, applies it, persists, and notifies.
   *
   * 'system' resolves live against the OS preference via variant().
   *
   * @param {'light'|'dark'|'system'} mode  - the variant mode to apply
   * @returns {void}
   */
  setVariant(mode) { this.state.variantMode = mode; this.apply(); this._emit(); },

  /**
   * Switches the base font size, applies it, persists, and notifies listeners.
   *
   * Unknown size keys are ignored so a stale saved value can't break the page.
   *
   * @param {string} size  - the font-size key (small/medium/large/xlarge)
   * @returns {void}
   */
  setFontSize(size) { if (FONT_SIZES[size]) { this.state.fontSize = size; this.apply(); this._emit(); } },

  /**
   * Registers a callback fired after any theme/variant/size change.
   *
   * Used by the dashboard to recolor its echarts instances on theme changes.
   *
   * @param {function} cb  - invoked with no args after each change
   * @returns {void}
   */
  onChange(cb) { this._listeners.push(cb); },

  /**
   * Invokes every registered change listener, isolating exceptions so one
   * failing listener cannot block the others.
   *
   * @returns {void}
   */
  _emit() { this._listeners.forEach(cb => { try { cb(); } catch (e) { /* isolate */ } }); },

  /**
   * Applies saved state, wires the OS-preference listener so 'system' mode
   * tracks the theme live, and makes the "Aa" popover dismiss on outside-click
   * or Escape. Safe to call once at startup.
   *
   * @returns {void}
   */
  init() {
    this.apply();
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      if (this.state.variantMode === 'system') { this.apply(); this._emit(); }
    });
    // The appearance popover is a native <details>, which stays open until its
    // summary is clicked again; close it on any click outside it or on Escape.
    document.addEventListener('click', (e) => {
      document.querySelectorAll('details.settings-menu[open]').forEach(d => {
        if (!d.contains(e.target)) d.removeAttribute('open');
      });
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        document.querySelectorAll('details.settings-menu[open]').forEach(d => d.removeAttribute('open'));
      }
    });
  },
};

window.LRTheme = LRTheme;
