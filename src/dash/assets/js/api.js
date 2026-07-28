// Router API calls. Callers catch failures and keep the last good state.
const getJson = async (url) => (await fetch(url)).json();

export const getStatus = () => getJson('/router');
export const getModels = () => getJson('/router/models');
export const getGpu = () => getJson('/router/gpu');
export const getTimeline = () => getJson('/router/status_timeline');
export const getHistory = (params) => getJson(`/router/history?${params}`);

export const routerCommand = (action) => fetch(`/router/${action}`);
export const clearHistory = () => fetch('/router/reset_history');

export const modelAction = (action, model) => fetch(`/models/${action}`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ model }),
});
