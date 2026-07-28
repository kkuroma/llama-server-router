// Model id parsing for the list rows.

// A bare param count ("20B", "6.5b") or an MoE active-param count ("a3b").
const PARAM_RE = /^\d+(?:\.\d+)?[bB]$/;
const ACTIVE_RE = /^[aA]\d+(?:\.\d+)?[bB]$/;

// Splits a model id into a display name and typed tags: the id is split on
// '-', and the first param/active segment starts the tag list. Each tag is
// 'param', 'active', or 'plain' so the row can color counts and neutralize
// descriptors. No param segment -> the whole id is the name.
export function parseModel(id) {
  const parts = String(id).split('-');
  let idx = -1;
  for (let i = 0; i < parts.length; i++) {
    if (PARAM_RE.test(parts[i]) || ACTIVE_RE.test(parts[i])) { idx = i; break; }
  }
  if (idx === -1) return { name: id, tags: [] };
  const name = parts.slice(0, idx).join('-') || id;
  const tags = parts.slice(idx).map(t => ({
    text: t,
    kind: PARAM_RE.test(t) ? 'param' : ACTIVE_RE.test(t) ? 'active' : 'plain',
  }));
  return { name, tags };
}
