// barber — query-aware context trimming for LLM requests.
//
// Your context could use a trim.
//
//     import { trim } from "barber-llm";
//
//     const result = trim(messages, { keep: 0.6 });
//     send(result.messages);          // same conversation, fewer tokens
//
// Chunks survive verbatim or vanish. Nothing is rewritten, nothing is
// summarized, and no model is called at trim time. The selection algorithm is
// the validated, benchmarked one from the Python package (barber-llm on PyPI);
// this port replays its decisions identically on the golden fixture suite.

import {
  DEFAULT_CONFIG,
  charLen,
  makeSelectionTransform,
  textOf,
} from "./core.js";
import { lexical } from "./embedders.js";

export const VERSION = "0.3.2";
export { DEFAULT_CONFIG, TIER_KEEP_RATIO, makeSelectionTransform, splitChunks, textOf } from "./core.js";
export { lexical } from "./embedders.js";

// Bounded LRU cache to hand to trim/makeTransform via `cache`.
//
// A plain Map works and never evicts: one entry per distinct context block,
// held for the life of the process. That is fine for a script and a leak in a
// server, so use this instead. An entry holds one whole trimmed block, so
// size `maxsize` against your block size, not just your conversation count.
//
// Eviction re-opens a decision: if an evicted block reappears under a
// different question it is selected again, and the result can differ from
// the first time. Keep `maxsize` above your live-conversation count and that
// never happens. (Unlike the Python Cache there is no lock: this is
// single-threaded JS, and the whole pipeline is synchronous.)
export class Cache {
  constructor({ maxsize = 4096 } = {}) {
    this.maxsize = maxsize;
    this._m = new Map();
  }

  // Membership is always immediately followed by a read in this codebase, so
  // a hit counts as a use and refreshes recency — same as the Python Cache.
  has(key) {
    if (!this._m.has(key)) return false;
    const v = this._m.get(key);
    this._m.delete(key);
    this._m.set(key, v);
    return true;
  }

  get(key) {
    if (!this._m.has(key)) return undefined;
    const v = this._m.get(key);
    this._m.delete(key);
    this._m.set(key, v);
    return v;
  }

  set(key, value) {
    this._m.delete(key);
    this._m.set(key, value);
    while (this._m.size > this.maxsize) {
      this._m.delete(this._m.keys().next().value);
    }
    return this;
  }

  get size() {
    return this._m.size;
  }

  keys() {
    return [...this._m.keys()];
  }

  values() {
    return [...this._m.values()];
  }

  [Symbol.iterator]() {
    return this._m[Symbol.iterator]();
  }
}

// Token counting: the Python package uses tiktoken when installed and falls
// back to len // 4. This port always uses the fallback (code points // 4) to
// stay dependency-free; tokensSaved is an estimate either way.
const ntok = (s) => Math.floor(charLen(s) / 4);

const countTokens = (messages) =>
  messages.reduce((acc, m) => acc + ntok(textOf(m.content, true)), 0);

const reEscape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const markerRe = (template) =>
  new RegExp(reEscape(template).replaceAll("\\{n\\}", "(\\d+)"), "g");

// Count dropped chunks from the markers in the output, not from selection
// stats: a cache hit leaves the stats at zero while the content is still
// trimmed. Unchanged messages are the same object reference, so `===` skips
// them exactly like Python's `is`.
function countDropped(before, after, marker) {
  const rx = markerRe(marker);
  let total = 0;
  for (let i = 0; i < before.length; i++) {
    const b = before[i];
    const a = after[i];
    if (b === a) continue;
    let was = 0;
    for (const m of textOf(b.content, true).matchAll(rx)) was += Number(m[1]);
    let now = 0;
    for (const m of textOf(a.content, true).matchAll(rx)) now += Number(m[1]);
    total += now - was;
  }
  return total;
}

// Returns the ["barber", fn] pair for pipeline integration, where
// fn(messages) -> [messages, changed].
//
// Pass a shared `cache` to get freeze-on-first-sight memoization across
// turns: the first turn to see a block decides it, every later turn replays
// that decision byte-identically, keeping the provider prompt cache warm.
// Use `new Cache()` in a long-running process; a plain Map never evicts.
// `keep` pins the fraction of chunks retained (0.6 is the benchmark
// default), overriding minKeepRatio/maxKeepRatio on any `cfg` you pass.
export function makeTransform({ embedder = null, keep = 0.6, cfg = null, cache = null } = {}) {
  const effective = { ...DEFAULT_CONFIG, ...cfg, minKeepRatio: keep, maxKeepRatio: keep };
  const [, fn] = makeSelectionTransform({
    embedFn: embedder,
    cfg: effective,
    decisionCache: cache,
  });
  return ["barber", fn];
}

// Trim query-irrelevant chunks out of an OpenAI-style message list.
//
// messages: array of {role, content} objects.
// keep:     fraction of chunks retained per block (benchmark default 0.6).
// embedder: null -> deterministic lexical fallback (zero deps), or any
//           embed(texts) -> vectors function (term-weight Maps or dense
//           arrays). Must be synchronous.
//
// A message is only trimmed when ALL of these hold, so `changed` comes back
// false if none qualifies:
//   - role is "user", "tool", or "function" (system and assistant are never
//     touched)
//   - it is NOT the latest user message (that one is the question)
//   - content is a plain string, not an array of content parts
//   - at least 800 characters and at least 4 chunks
//
// So context and question packed into ONE user message is a no-op: put the
// context in its own earlier message.
//
// `tokensSaved` is signed. Negative means the markers cost more than the
// dropped chunks saved, which is worth acting on rather than hiding.
//
// Guards (lead/tail keep, deontic/PII pinning, rare-query-entity pinning,
// relevance floor) are always on. Deterministic: same input, same output.
export function trim(messages, { keep = 0.6, embedder = null, cfg = null, cache = null } = {}) {
  const effective = { ...DEFAULT_CONFIG, ...cfg, minKeepRatio: keep, maxKeepRatio: keep };
  const [, fn] = makeSelectionTransform({
    embedFn: embedder,
    cfg: effective,
    decisionCache: cache,
  });
  const [out, changed] = fn([...messages]);
  if (!changed) {
    // Nothing was substituted, so every message in `out` is the input object:
    // the counts are 0 by construction, no need to tokenize the prompt twice.
    return { messages: out, tokensSaved: 0, chunksDropped: 0, changed: false };
  }
  return {
    messages: out,
    tokensSaved: countTokens(messages) - countTokens(out),
    chunksDropped: countDropped(messages, out, effective.dropMarker),
    changed: true,
  };
}
