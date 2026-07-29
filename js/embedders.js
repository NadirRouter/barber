// Embedders — the scorers barber uses to rank chunks against the query.
//
// Each factory returns embed(texts: string[]) -> vectors. Vectors are either
// term-weight Maps (the lexical fallback) or dense arrays / typed arrays
// (real encoders); the core scorer picks the matching cosine automatically.
//
// Port of barber/embedders.py `lexical`. Decision-identical to the Python
// implementation; verified by the golden fixtures in test/. Term vectors are
// Maps, not plain objects: chunk tokens like "constructor" would otherwise
// collide with Object.prototype.

// Unicode-aware, and must stay decision-identical to Python's `_TOK`
// (`[CJK]|[^\W_]+`). JS `\w` is ASCII-only even under /u, so the word class is
// spelled out as \p{L}\p{N}. `[a-z0-9]+` — what this was — tokenized "café" as
// "caf" and any non-Latin script as nothing, so the fallback silently did
// nothing outside English. The leading alternative gives unspaced CJK one token
// per character instead of one token per sentence; character unigrams are the
// honest ceiling of a lexical scorer there.
const TOK = /[぀-ヿ㐀-䶿一-鿿豈-﫿]|[\p{L}\p{N}]+/gu;

export function tokenize(s) {
  return s.toLowerCase().match(TOK) ?? [];
}

// Deterministic BM25-lite scorer used as a fallback when no vector encoder is
// injected. Returns bag-of-words idf-weighted term vectors (as Maps). Zero
// dependencies, but lexical only: it matches words, not meaning. For real
// semantic relevance, inject an embedder backed by a retrieval-trained model.
export function lexical() {
  return function embed(texts) {
    // idf over the batch (the request's own chunks + query)
    const toksList = texts.map(tokenize);
    const df = new Map();
    for (const toks of toksList) {
      for (const w of new Set(toks)) df.set(w, (df.get(w) ?? 0) + 1);
    }
    const n = texts.length;
    return toksList.map((toks) => {
      const tf = new Map();
      for (const w of toks) tf.set(w, (tf.get(w) ?? 0) + 1);
      const vec = new Map();
      for (const [w, c] of tf) {
        vec.set(w, (c / toks.length) * Math.log((n + 1) / df.get(w)));
      }
      return vec;
    });
  };
}
